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
  // disk enabled, max integration steps, canvas width, canvas height
  options: vec4f,
  // row offset and row count for this dispatch, unused, unused
  band: vec4f,
  // bloom width, bloom height, threshold, strength
  bloom: vec4f,
}

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var prevAccum: texture_2d<f32>;
@group(0) @binding(2) var nextAccum: texture_storage_2d<rgba16float, write>;

// R2 / Roberts low-discrepancy sequence: the 2D generalisation of the golden ratio.
// g is the plastic number, the real root of x^3 = x + 1.
const R2_A1: f32 = 0.7548776662466927;
const R2_A2: f32 = 0.5698402909980532;

// Five-stop ramp across the observed temperature: near-black ember at the cool
// outer rim, through deep red and orange, into white at the hottest radii, and
// finally blue-white.
//
// The blue end is real physics, not decoration — a relativistic Doppler factor
// shifts observed temperature as T_obs = g * T_emit, so an approaching limb
// genuinely runs bluer. It only appears when the Doppler control is turned up;
// at the default the disk stays in the black-to-white range, which is the look
// the film went for after deliberately suppressing beaming.
const DISK_C0: vec3f = vec3f(0.16, 0.012, 0.004);
const DISK_C1: vec3f = vec3f(0.85, 0.14, 0.012);
const DISK_C2: vec3f = vec3f(1.0, 0.47, 0.09);
const DISK_C3: vec3f = vec3f(1.0, 0.82, 0.46);
const DISK_C4: vec3f = vec3f(0.80, 0.90, 1.0);
/// Scales the emissivity profile so the hot inner edge lands near the top of the
/// tone curve without clipping, leaving the rest of the disk in its linear range.
/// Large because it is normalizing a T^3.4 profile, whose peak is only ~0.09.
/// Cut hard when the striation amplitude dropped. Matching the *mean* was not
/// enough: the striation used to swing the emissivity roughly 2:1, and those
/// dips were carrying the disk's color. Tone mapping desaturates everything
/// bright, so a ring that spends half its width lower on the curve reads as gold
/// while a uniformly bright one reads as white. With the swing gone the level
/// itself has to come down or the whole inner disk washes out.
const DISK_BRIGHTNESS: f32 = 7.0;
/// Emissivity exponent. Stefan-Boltzmann is T^4, but a real disk is not a
/// blackbody at every radius and the flat T^4 tail drops the outer disk into a
/// muddy near-black before the outer rim. Backing off to 3.4 keeps the inner
/// ring dominant while leaving the outer disk legibly amber.
const DISK_EMISSIVITY_POWER: f32 = 3.4;
/// Gamma on the ramp coordinate. Emitted temperature varies slowly near its
/// peak, so a linear mapping spends a wide radial band at the white end of the
/// ramp and washes the color out. Biasing the coordinate downward holds the
/// bulk of the disk in the saturated orange stops and narrows white to a lip.
const DISK_RAMP_GAMMA: f32 = 1.7;
/// Radius, in units of the ISCO, that heavily lensed images fall back to. This
/// is where the emissivity profile peaks, and the higher-order images stack up
/// around the inner ring, so it is the representative value for a band that can
/// no longer be resolved. See the lensing level-of-detail note in diskColor.
const DISK_LENSED_REFERENCE: f32 = 1.45;
/// Radial striation. Deliberately subtle: orbital shear does smear structure
/// into concentric bands, but at any amplitude you can comfortably see, a
/// periodic function of log(r) stops reading as texture and starts reading as
/// contour banding — a rendering artifact rather than a disk.
const DISK_FILAMENT_BASE: f32 = 0.91;
const DISK_FILAMENT_COARSE: f32 = 0.07;
const DISK_FILAMENT_FINE: f32 = 0.035;
/// Phase average of the striation, used as the fallback once radial detail is no
/// longer resolvable. E[((1+sin)/2)^3] = 5/16 and E[((1+sin)/2)^5] = 63/256.
const DISK_FILAMENT_MEAN: f32 =
  DISK_FILAMENT_BASE + DISK_FILAMENT_COARSE * 0.3125 + DISK_FILAMENT_FINE * 0.24609375;

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
/// a per-pixel hash then applies a Cranley-Patterson rotation so neighboring
/// pixels do not march through an identical pattern in lockstep.
///
/// Note for anyone chasing the thin ragged arcs beside the shadow: they are not
/// a defect in this. Measured pixel profiles across one of those arcs are the
/// partially-converged average of an exponentially compressed region — close to
/// the shadow edge the crossing radius sweeps most of the disk within a single
/// pixel, so one pixel is integrating the whole radial profile. It converges
/// like any other high-variance region, just more slowly. Permuting the sample
/// index per pixel was tried here and measured to change nothing.
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

/// One layer of stars on a jittered grid. The 3x3 neighborhood is walked so
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

  // Very nearly black. A bright, busy sky flattens the contrast that makes the
  // disk read as incandescent, so the only ambient light here is a barely
  // perceptible cool cast to keep it from banding to flat zero.
  let axis = clamp(d.z * 0.5 + 0.5, 0.0, 1.0);
  var col = mix(vec3f(0.0016, 0.0022, 0.0038), vec3f(0.0028, 0.0030, 0.0060), axis);

  // Sparse, so individual stars read against the black rather than forming a haze.
  let bright = starLayer(fuv, 46.0, 0.030, 0.0);
  let faint = starLayer(fuv, 115.0, 0.045, 31.0);

  // Stars span their own range of spectral classes rather than all being white.
  let tintSeed = hash1(floor(fuv * 60.0));
  let tint = select(
    mix(vec3f(0.62, 0.76, 1.0), vec3f(0.97, 0.97, 1.0), tintSeed / 0.5),
    mix(vec3f(0.97, 0.97, 1.0), vec3f(1.0, 0.80, 0.62), (tintSeed - 0.5) / 0.5),
    tintSeed > 0.5,
  );

  col += tint * (bright * 0.80 + faint * 0.22);
  return col;
}

// ---------------------------------------------------------------------------
// Accretion disk (stylized — not radiative transfer)
// ---------------------------------------------------------------------------

/// Piecewise ramp over the five disk stops.
///
/// The orange band is deliberately the widest. Tone mapping already desaturates
/// anything bright — a value well above the clip point comes out white whatever
/// its hue — so reaching the pale stops early costs the disk its color twice
/// over. Holding orange until the observed temperature is genuinely high keeps
/// the body of the disk saturated and leaves white as an accent on the hottest
/// radii only.
fn diskRamp(t: f32) -> vec3f {
  let x = clamp(t, 0.0, 1.0);
  if (x < 0.22) {
    return mix(DISK_C0, DISK_C1, x / 0.22);
  }
  if (x < 0.62) {
    return mix(DISK_C1, DISK_C2, (x - 0.22) / 0.40);
  }
  if (x < 0.93) {
    return mix(DISK_C2, DISK_C3, (x - 0.62) / 0.31);
  }
  return mix(DISK_C3, DISK_C4, (x - 0.93) / 0.07);
}

/// Shade an equatorial-plane crossing.
///
/// Temperature follows the r^-0.75 falloff, completed with the standard
/// Shakura-Sunyaev inner-boundary factor (1 - sqrt(r_isco/r))^0.25 so emission
/// goes to zero *at* the ISCO and peaks just outside it, which is what gives the
/// disk a bright ring rather than a flat wash.
///
/// Brightness is not temperature: a thermal emitter radiates roughly as T^3-T^4,
/// so intensity falls far faster than the temperature does. That gap is what
/// separates a hot inner ring from a dim outer disk.
///
/// The Doppler/beaming factor comes from the local prograde Keplerian angular
/// velocity omega = 1/(r^1.5 + a) — crude, but it is what makes one side of the
/// disk visibly brighter than the other.
fn diskColor(xc: vec3f, pc: vec3f, rc: f32, a: f32, imageOrder: f32) -> vec3f {
  let rIsco = U.params.z;
  let rOuter = U.params.y;

  // Lensing level of detail.
  //
  // The direct image of the disk maps screen position to disk radius smoothly.
  // Every higher-order image — light that wound around the hole before reaching
  // the camera — is compressed exponentially, and by the second or third pass
  // the whole radial profile is squeezed into a band thinner than a pixel. Point
  // sampling that is aliasing a signal with no band limit, which is what makes
  // the thin arcs beside the shadow look ragged: it is variance, not geometry.
  // Rendering the disk with a flat color makes those same arcs perfectly smooth.
  //
  // The image order is the number of equatorial crossings the ray has made, so
  // it measures that compression directly and costs a counter. Past the first
  // image the shading is faded to the emission of the bright inner ring, which
  // is what dominates the stack anyway. This is the same trade a mip level
  // makes for a minified texture: the detail is not resolvable, so do not
  // pretend to resolve it.
  let lod = smoothstep(1.0, 3.0, imageOrder);

  let xRaw = max(rc / rIsco, 1.0);
  let x = mix(xRaw, DISK_LENSED_REFERENCE, lod);
  let innerBoundary = pow(max(1.0 - inverseSqrt(x), 0.0), 0.25);
  let temperature = pow(x, -0.75) * innerBoundary;

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

  // How much relativistic beaming to apply, 0 to 1. Interstellar's visual
  // effects team suppressed this outright: the physically correct asymmetry
  // makes one limb far brighter than the other, which reads as an error rather
  // than as physics. At 0 the disk is brightness-symmetric like the film; at 1
  // it is the honest g^3 boost.
  let beamAmount = U.band.z;
  let beaming = mix(
    1.0,
    clamp(pow(max(doppler, 1e-3), 3.0), 0.03, 16.0),
    beamAmount,
  );

  // Orbital shear smears any structure azimuthally, so concentric striation is
  // the physically-motivated texture for a disk. Two octaves at logarithmic
  // spacing, raised to a power so the peaks come out as thin bright filaments
  // rather than broad sinusoidal bands.
  //
  // Highest-frequency term in the shader, so it is the first thing to go as the
  // radial mapping compresses; DISK_FILAMENT_MEAN is its average over phase.
  let lr = log(max(rc, 1e-3));
  let coarse = pow(0.5 + 0.5 * sin(lr * 31.0), 3.0);
  let fine = pow(0.5 + 0.5 * sin(lr * 67.0 + 1.7), 5.0);
  let filaments = mix(
    DISK_FILAMENT_BASE + DISK_FILAMENT_COARSE * coarse + DISK_FILAMENT_FINE * fine,
    DISK_FILAMENT_MEAN,
    lod,
  );

  // Stefan-Boltzmann: a thermal emitter radiates as T^4. Combined with the
  // r^-0.75 temperature law that puts the outer rim at a few percent of the peak,
  // which is what concentrates the disk into a thin bright band with a dim tail
  // instead of a broad filled wedge.
  let emissivity =
    pow(temperature, DISK_EMISSIVITY_POWER) * beaming * DISK_BRIGHTNESS * filaments;

  // Observed temperature, not emitted: the Doppler factor shifts it directly,
  // T_obs = g * T_emit. Blended by the same control as the beaming, so turning
  // Doppler off leaves a purely radial color gradient.
  let shifted = mix(1.0, doppler, beamAmount);
  let observed = clamp(temperature * shifted * 2.0, 0.0, 1.0);
  let tint = diskRamp(pow(observed, DISK_RAMP_GAMMA));

  // Soft edges so the annulus does not terminate in a hard ring.
  let outerFade = 1.0 - smoothstep(rOuter * 0.8, rOuter, rc);

  return tint * emissivity * outerFade;
}

// ---------------------------------------------------------------------------
// Disk geometry
// ---------------------------------------------------------------------------

struct SlabHit {
  hit: bool,
  /// Where along the step the ray met the disk, in [0, 1].
  t: f32,
}

/// Half-thickness at radius r. Proportional to r, which is roughly what a real
/// accretion disk does — it flares outward rather than staying a constant slab.
fn diskHalfThickness(r: f32) -> f32 {
  return U.band.w * r;
}

/// First entry into the disk slab over one integration step.
///
/// Thickness is not decoration. A zero-thickness disk is a mathematically
/// non-band-limited emitter: near the shadow the lensed images of it pile up
/// with infinitely thin bright bands and infinitely thin dark gaps between them,
/// and no finite number of samples per pixel can resolve that. It shows up as a
/// ragged sliver hugging the shadow. Giving the disk real thickness makes those
/// bands finite-width and the gaps fill in, because the slab occults them.
///
/// The test is a sign change in |z| - H(r), the signed distance to the slab
/// surface. planeLimitedStep keeps a step from jumping the midplane, which for a
/// slab of any thickness means it cannot jump the slab either.
///
/// Thickness zero is still supported and reduces exactly to the old midplane
/// crossing, so the control can be taken to zero and this behaves as before.
fn slabEntry(xPrev: vec3f, rPrev: f32, xNext: vec3f, rNext: f32) -> SlabHit {
  var out: SlabHit;
  out.hit = false;
  out.t = 0.0;

  let sPrev = abs(xPrev.z) - diskHalfThickness(rPrev);
  let sNext = abs(xNext.z) - diskHalfThickness(rNext);

  if (sPrev > 0.0 && sNext <= 0.0) {
    out.hit = true;
    out.t = sPrev / max(sPrev - sNext, 1e-8);
  } else if (xPrev.z * xNext.z < 0.0) {
    // Zero thickness, or a step that spanned the whole slab: fall back to the
    // midplane crossing so the disk never becomes invisible.
    out.hit = true;
    out.t = xPrev.z / (xPrev.z - xNext.z);
  }
  return out;
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

  // Carried across iterations rather than recomputed: the slab test needs the
  // radius at both ends of a step, and the far end becomes the near end next
  // time round.
  var r = kerrRadius(st.x, a);

  // How many times the ray has met the equatorial plane. The first meeting that
  // lands on the disk is the direct image, the second is light that wound once
  // around the hole, and so on — so this is the image order, and diskColor uses
  // it to decide how much radial detail is still resolvable. Crossings inside
  // the ISCO count: the ray passed through the disk plane there, it just found
  // no emission, and it wound just as far getting there.
  var crossings = 0.0;

  var step: u32 = 0u;
  loop {
    if (step >= maxSteps) {
      break;
    }

    // Captured by the event horizon.
    if (r < rPlus * KERR_HORIZON_PAD) {
      return vec3f(0.0);
    }

    // The RK4 stage-one derivative, hoisted: the escape branch, the step
    // limiter and the integrator all want it, so evaluating it once here makes
    // the plane limiter free rather than a fifth metric evaluation per step.
    let deriv = geodesicRHS(st.x, st.p, a);

    // Escaped: sample the background along the true coordinate velocity rather
    // than the momentum, which still differ slightly at r = 60.
    if (r > KERR_ESCAPE_RADIUS) {
      return background(normalize(deriv.dx));
    }

    var h = adaptiveStep(r);
    if (diskEnabled) {
      h = planeLimitedStep(h, st.x.z, deriv.dx.z);
    }

    let previous = st;
    let previousR = r;
    st = rk4StepFrom(st, deriv, a, h);
    r = kerrRadius(st.x, a);

    // Where the ray meets the disk. The disk is opaque, so the first hit along
    // the backward ray is the last emission the observer sees — that is what
    // produces the lensed over-and-under image.
    if (diskEnabled) {
      let entry = slabEntry(previous.x, previousR, st.x, r);
      if (entry.hit) {
        crossings += 1.0;
        let crossing = mix(previous.x, st.x, entry.t);
        let rc = kerrRadius(crossing, a);
        if (rc >= rIsco && rc <= rOuter) {
          return diskColor(
            crossing,
            mix(previous.p, st.p, entry.t),
            rc,
            a,
            crossings,
          );
        }
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

  // One sample is spread over several dispatches, a horizontal band at a time.
  // Integrating a full image in one submission takes long enough on modest GPUs
  // to stall compositing and input handling for the whole tab; short dispatches
  // keep the browser responsive at the same total throughput.
  let pixel = vec2u(gid.x, gid.y + u32(U.band.x));
  if (pixel.x >= resolution.x || pixel.y >= resolution.y) {
    return;
  }

  let frameIndex = u32(U.frame.x);
  let jitter = sampleOffset(pixel, frameIndex);

  // Pixel center plus sub-pixel jitter, mapped to NDC with y up.
  let uv = (vec2f(pixel) + jitter) / vec2f(resolution);
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
  let coord = vec2i(pixel);
  let previous = textureLoad(prevAccum, coord, 0);
  let weight = 1.0 / (f32(frameIndex) + 1.0);
  textureStore(nextAccum, coord, mix(previous, vec4f(radiance, 1.0), weight));
}
