// Progressive path-traced Kerr black hole.
//
// Concatenated after kerr_math.wgsl. One jittered sample per pixel per dispatch,
// blended into the running average held in a double-buffered rgba16float texture.
//
// The accumulation is double-buffered rather than read-write because baseline
// WebGPU only permits read_write storage-texture access on r32float/r32uint/r32sint;
// rgba16float read-write sits behind the rw-storage-texture-tier-2 extension.

struct Uniforms {
  // xyz = eye position, w unused
  camPos: vec4f,
  // xyz = right basis vector, w = tan(fov/2)
  camRight: vec4f,
  // xyz = up basis vector, w = aspect ratio
  camUp: vec4f,
  // xyz = forward basis vector, w unused
  camFwd: vec4f,
  // spin a, disk outer radius, r_isco, r_plus
  params: vec4f,
  // frame index (0-based), resolution x, resolution y, exposure
  frame: vec4f,
  // disk enabled, max integration steps, unused, unused
  options: vec4f,
}

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var prevAccum: texture_2d<f32>;
@group(0) @binding(2) var nextAccum: texture_storage_2d<rgba16float, write>;

// R2 / Roberts low-discrepancy sequence: the 2D generalisation of the golden ratio.
// g is the plastic number, the real root of x^3 = x + 1.
const R2_A1: f32 = 0.7548776662466927;
const R2_A2: f32 = 0.5698402909980532;

const DISK_COOL: vec3f = vec3f(0.20, 0.40, 1.0);
const DISK_HOT: vec3f = vec3f(1.0, 0.97, 0.93);
/// Normalises the peak of the emissivity profile to roughly 1 before beaming.
const DISK_BRIGHTNESS: f32 = 13.0;

fn hash2(p: vec2u) -> vec2f {
  var v = p * vec2u(1664525u, 1013904223u);
  v.x += v.y * 1664525u;
  v.y += v.x * 1013904223u;
  v ^= v >> vec2u(16u);
  v.x += v.y * 1664525u;
  v.y += v.x * 1013904223u;
  v ^= v >> vec2u(16u);
  return vec2f(v) * (1.0 / 4294967296.0);
}

fn hash1(p: vec3f) -> f32 {
  let q = fract(p * 0.1031);
  let s = q + dot(q, q.yzx + 33.33);
  return fract((s.x + s.y) * s.z);
}

/// Sub-pixel offset for this sample.
///
/// The R2 sequence gives a well-stratified progression as the frame index rises;
/// a per-pixel hash then applies a Cranley-Patterson rotation so neighbouring
/// pixels do not march through an identical pattern in lockstep.
fn sampleOffset(pixel: vec2u, frameIndex: u32) -> vec2f {
  let lds = fract(vec2f(R2_A1, R2_A2) * f32(frameIndex + 1u));
  return fract(lds + hash2(pixel));
}

// ---------------------------------------------------------------------------
// Background
// ---------------------------------------------------------------------------

/// Cube-face parameterisation: (u, v, face). Avoids the pole clustering that a
/// naive spherical parameterisation would give the starfield.
fn cubeFace(d: vec3f) -> vec3f {
  let ad = abs(d);
  if (ad.x >= ad.y && ad.x >= ad.z) {
    return vec3f(d.yz / ad.x, select(1.0, 0.0, d.x > 0.0));
  }
  if (ad.y >= ad.z) {
    return vec3f(d.xz / ad.y, select(3.0, 2.0, d.y > 0.0));
  }
  return vec3f(d.xy / ad.z, select(5.0, 4.0, d.z > 0.0));
}

/// One layer of stars on a jittered grid. The 3x3 neighbourhood is walked so
/// stars near a cell boundary are not clipped in half.
fn starLayer(fuv: vec3f, density: f32, sparsity: f32, seed: f32) -> f32 {
  let g = fuv.xy * density;
  let base = floor(g);
  var total = 0.0;

  for (var dy = -1; dy <= 1; dy++) {
    for (var dx = -1; dx <= 1; dx++) {
      let cell = base + vec2f(f32(dx), f32(dy));
      let key = vec3f(cell, fuv.z * 17.0 + seed);

      let presence = hash1(key);
      if (presence > sparsity) {
        continue;
      }

      let jitter = vec2f(hash1(key + 1.7), hash1(key + 4.3));
      let magnitude = hash1(key + 9.1);
      let d = length(g - (cell + jitter));

      // Tight core plus a faint halo, so stars survive tone mapping without aliasing.
      let core = exp(-d * d * 900.0) * (0.35 + magnitude * magnitude * 3.0);
      let halo = exp(-d * d * 60.0) * 0.045 * magnitude;
      total += core + halo;
    }
  }
  return total;
}

/// Deterministic — required for the accumulation to converge. The per-sample
/// jitter is what antialiases the stars, so this is where progressive
/// refinement is most visible.
fn background(d: vec3f) -> vec3f {
  let fuv = cubeFace(d);

  // Faint cool gradient along the spin axis, plus a soft band near the equator.
  let axis = clamp(d.z * 0.5 + 0.5, 0.0, 1.0);
  let band = exp(-d.z * d.z * 5.0);
  var col = mix(vec3f(0.010, 0.012, 0.020), vec3f(0.016, 0.022, 0.038), axis);
  col += vec3f(0.014, 0.018, 0.030) * band * 0.5;

  // Two star layers at different scales.
  let bright = starLayer(fuv, 46.0, 0.055, 0.0);
  let faint = starLayer(fuv, 115.0, 0.09, 31.0);

  // Slight colour variation so the field is not uniformly white.
  let tintSeed = hash1(floor(fuv * 60.0));
  let tint = mix(vec3f(0.72, 0.82, 1.0), vec3f(1.0, 0.94, 0.84), tintSeed);

  col += tint * (bright * 1.15 + faint * 0.35);
  return col;
}

// ---------------------------------------------------------------------------
// Accretion disk (stylized — not radiative transfer)
// ---------------------------------------------------------------------------

/// Shade an equatorial-plane crossing.
///
/// Temperature falls as r^-0.75 and a crude Doppler/beaming factor is derived
/// from the local prograde Keplerian angular velocity omega = 1/(r^1.5 + a).
/// Colour is biased from cool blue toward hot white with intensity.
fn diskColor(xc: vec3f, pc: vec3f, rc: f32, a: f32) -> vec3f {
  let rIsco = U.params.z;
  let rOuter = U.params.y;

  let temperature = pow(max(rc / rIsco, 1e-3), -0.75);

  // Local orbital velocity: prograde Keplerian, tangential in the equatorial plane.
  let omega = 1.0 / (pow(rc, 1.5) + a);
  let cylindrical = max(length(xc.xy), 1e-4);
  let tangent = vec3f(-xc.y, xc.x, 0.0) / cylindrical;
  let speed = clamp(omega * cylindrical, 0.0, 0.95);
  let beta = tangent * speed;

  // We integrate backward from the camera, so the photon's actual direction of
  // travel (emitter -> observer) is the reverse of dx/dlambda.
  let deriv = geodesicRHS(xc, pc, a);
  let toObserver = -normalize(deriv.dx);

  let lorentz = inverseSqrt(max(1.0 - speed * speed, 1e-4));
  let doppler = 1.0 / (lorentz * (1.0 - dot(beta, toObserver)));
  let beaming = clamp(pow(max(doppler, 1e-3), 3.0), 0.04, 14.0);

  let emissivity = temperature * beaming;

  // Saturating ramp: cool blue at low intensity, hot white where beaming piles up.
  let t = clamp(emissivity / (emissivity + 1.1), 0.0, 1.0);
  let tint = mix(DISK_COOL, DISK_HOT, t * t);

  // Soft edges so the annulus does not terminate in a hard ring.
  let innerFade = smoothstep(rIsco, rIsco * 1.08, rc);
  let outerFade = 1.0 - smoothstep(rOuter * 0.85, rOuter, rc);

  return tint * emissivity * DISK_BRIGHTNESS * innerFade * outerFade;
}

// ---------------------------------------------------------------------------
// Integration
// ---------------------------------------------------------------------------

fn traceRadiance(origin: vec3f, direction: vec3f) -> vec3f {
  let a = U.params.x;
  let rOuter = U.params.y;
  let rIsco = U.params.z;
  let rPlus = U.params.w;
  let diskEnabled = U.options.x > 0.5;
  let maxSteps = u32(U.options.y);

  var st: State;
  st.x = origin;
  st.p = nullMomentum(origin, direction, a);

  var step: u32 = 0u;
  loop {
    if (step >= maxSteps) {
      break;
    }

    let r = kerrRadius(st.x, a);

    // Captured by the event horizon.
    if (r < rPlus * KERR_HORIZON_PAD) {
      return vec3f(0.0);
    }

    // Escaped: sample the background along the true coordinate velocity rather
    // than the momentum, which still differ slightly at r = 60.
    if (r > KERR_ESCAPE_RADIUS) {
      let deriv = geodesicRHS(st.x, st.p, a);
      return background(normalize(deriv.dx));
    }

    let previous = st;
    st = rk4Step(st, a, adaptiveStep(r));

    // Equatorial crossing between steps. The disk is opaque, so the first
    // crossing along the backward ray is the last emission the observer sees —
    // this is what produces the lensed over-and-under image.
    if (diskEnabled && previous.x.z * st.x.z < 0.0) {
      let t = previous.x.z / (previous.x.z - st.x.z);
      let crossing = mix(previous.x, st.x, t);
      let rc = kerrRadius(crossing, a);
      if (rc >= rIsco && rc <= rOuter) {
        return diskColor(crossing, mix(previous.p, st.p, t), rc, a);
      }
    }

    step = step + 1u;
  }

  // Step budget exhausted. Only reachable deep in the strong-field region where
  // the ray is orbiting near the photon sphere, so treating it as captured is
  // the right call visually.
  return vec3f(0.0);
}

// ---------------------------------------------------------------------------

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let resolution = vec2u(u32(U.frame.y), u32(U.frame.z));
  if (gid.x >= resolution.x || gid.y >= resolution.y) {
    return;
  }

  let frameIndex = u32(U.frame.x);
  let jitter = sampleOffset(gid.xy, frameIndex);

  // Pixel centre plus sub-pixel jitter, mapped to NDC with y up.
  let uv = (vec2f(gid.xy) + jitter) / vec2f(resolution);
  let ndc = vec2f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);

  let tanHalfFov = U.camRight.w;
  let aspect = U.camUp.w;
  let direction = normalize(
    U.camFwd.xyz
      + U.camRight.xyz * (ndc.x * tanHalfFov * aspect)
      + U.camUp.xyz * (ndc.y * tanHalfFov),
  );

  let radiance = traceRadiance(U.camPos.xyz, direction);

  // Progressive blend. At frameIndex 0 the weight is 1, so the first pass after
  // a reset fully overwrites and no explicit clear is needed.
  let coord = vec2i(gid.xy);
  let previous = textureLoad(prevAccum, coord, 0);
  let weight = 1.0 / (f32(frameIndex) + 1.0);
  textureStore(nextAccum, coord, mix(previous, vec4f(radiance, 1.0), weight));
}
