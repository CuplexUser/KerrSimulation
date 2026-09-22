// Progressive path-traced Kerr black hole.
//
// Concatenated after kerr_math.wgsl. One jittered sample per pixel per dispatch,
// blended into the running average held in a double-buffered rgba32float texture.
//
// The accumulation is double-buffered rather than read-write because baseline
// WebGPU only permits read_write storage-texture access on r32float/r32uint/r32sint.
//
// It is f32 rather than f16 on purpose. The blend is mix(prev, sample, 1/n), and
// once n reaches a few hundred the increment falls below half an f16 ulp and is
// rounded away: high-variance pixels (the photon ring, stars) stop converging and
// keep whatever noise they had. f32 has headroom far past the sample cap.
//
// Time orientation. Rays are launched future-directed *away* from the camera,
// which is the time reverse of the light that actually arrives. Time reversal
// maps Kerr with spin a to Kerr with spin -a, so the geodesics are integrated
// with traceSpin = -a. That is what puts the flattened edge of the shadow on the
// approaching side of the disk, as it is for real light. The photon's physical
// angular momentum is then minus the traced one. Everything about the disk
// itself — ISCO, orbital velocity, redshift — uses the physical +a.

struct Uniforms {
  // xyz = eye position, w = lens shift x (pan)
  camPos: vec4f,
  // xyz = right basis vector, w = tan(fov/2)
  camRight: vec4f,
  // xyz = up basis vector, w = aspect ratio
  camUp: vec4f,
  // xyz = forward basis vector, w = lens shift y (pan)
  camFwd: vec4f,
  // spin a, disk outer radius, r_isco, r_plus
  params: vec4f,
  // frame index (0-based), resolution x, resolution y, exposure
  frame: vec4f,
  // disk enabled, max integration steps, canvas width, canvas height
  options: vec4f,
  // row offset and row count for this dispatch, Doppler beaming, disk thickness
  band: vec4f,
  // bloom width, bloom height, threshold, strength
  bloom: vec4f,
  // physical shading flag, peak temperature (K), 1 / peak flux, luminance norm
  disk: vec4f,
  // presented width, presented height, dither seed, pixel angle (radians)
  view: vec4f,
}

@group(0) @binding(0) var<uniform> U: Uniforms;
@group(0) @binding(1) var prevAccum: texture_2d<f32>;
@group(0) @binding(2) var nextAccum: texture_storage_2d<rgba32float, write>;

// R2 / Roberts low-discrepancy sequence: the 2D generalization of the golden ratio.
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

fn hash3(p: vec3f) -> vec3f {
  var q = fract(p * vec3f(0.1031, 0.1030, 0.0973));
  q += dot(q, q.yxz + 33.33);
  return fract((q.xxy + q.yxx) * q.zyx);
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
// Color science
// ---------------------------------------------------------------------------

/// CIE XYZ to linear sRGB (D65). Mirrors xyzToLinearSrgb in diskReference.ts.
fn xyzToLinearSrgb(c: vec3f) -> vec3f {
  return vec3f(
    3.2404542 * c.x - 1.5371385 * c.y - 0.4985314 * c.z,
    -0.9692660 * c.x + 1.8760108 * c.y + 0.0415560 * c.z,
    0.0556434 * c.x - 0.2040259 * c.y + 1.0572252 * c.z,
  );
}

/// Piecewise Gaussian lobe of the CIE fit below.
fn lobe(x: f32, mu: f32, below: f32, above: f32) -> f32 {
  let t = (x - mu) / select(above, below, x < mu);
  return exp(-0.5 * t * t);
}

/// Number of wavelength samples across 380-780 nm. Mirrors BLACKBODY_SAMPLES.
const BLACKBODY_SAMPLES: i32 = 20;

/// CIE XYZ of a blackbody at temperature T, by midpoint integration of Planck's
/// law against the Wyman-Sloan-Shirley fit of the CIE 1931 matching functions.
/// Mirrors blackbodyXYZ in diskReference.ts, which is where the renderer gets
/// the normalization that puts the peak temperature at unit luminance.
///
/// Twenty wavelengths is a lot of exps per sample, but it runs once per disk
/// hit — nothing next to the geodesic that found the hit.
fn blackbodyXYZ(temperature: f32) -> vec3f {
  let t = max(temperature, 500.0);
  let step = 400.0 / f32(BLACKBODY_SAMPLES);
  var total = vec3f(0.0);
  for (var i = 0; i < BLACKBODY_SAMPLES; i++) {
    let lambda = 380.0 + (f32(i) + 0.5) * step;
    let micrometers = lambda * 1e-3;
    let m2 = micrometers * micrometers;
    let planck = 1.0 / (m2 * m2 * micrometers * (exp(1.4388e7 / (lambda * t)) - 1.0));
    let matching = vec3f(
      1.056 * lobe(lambda, 599.8, 37.9, 31.0)
        + 0.362 * lobe(lambda, 442.0, 16.0, 26.7)
        - 0.065 * lobe(lambda, 501.1, 20.4, 26.2),
      0.821 * lobe(lambda, 568.8, 46.9, 40.5) + 0.286 * lobe(lambda, 530.9, 16.3, 31.1),
      1.217 * lobe(lambda, 437.0, 11.8, 36.0) + 0.681 * lobe(lambda, 459.0, 26.0, 13.8),
    );
    total += matching * planck;
  }
  return total * step;
}

/// Chromaticity of a blackbody at temperature T, as linear sRGB with unit
/// luminance. Kim et al. (2002) cubic fit to the Planckian locus in CIE xy,
/// valid 1667-25000 K. Cheap enough to evaluate per star, where the full
/// spectral integral would be wasted.
fn planckianRgb(temperature: f32) -> vec3f {
  let t = clamp(temperature, 1667.0, 25000.0);
  let it = 1.0 / t;
  let it2 = it * it;
  let it3 = it2 * it;
  var x: f32;
  if (t < 4000.0) {
    x = -0.2661239e9 * it3 - 0.2343589e6 * it2 + 0.8776956e3 * it + 0.179910;
  } else {
    x = -3.0258469e9 * it3 + 2.1070379e6 * it2 + 0.2226347e3 * it + 0.240390;
  }
  let x2 = x * x;
  let x3 = x2 * x;
  var y: f32;
  if (t < 2222.0) {
    y = -1.1063814 * x3 - 1.34811020 * x2 + 2.18555832 * x - 0.20219683;
  } else if (t < 4000.0) {
    y = -0.9549476 * x3 - 1.37418593 * x2 + 2.09137015 * x - 0.16748867;
  } else {
    y = 3.0817580 * x3 - 5.87338670 * x2 + 3.75112997 * x - 0.37001483;
  }
  return max(xyzToLinearSrgb(vec3f(x / y, 1.0, (1.0 - x - y) / y)), vec3f(0.0));
}

// ---------------------------------------------------------------------------
// Background
// ---------------------------------------------------------------------------

/// Star splat width in traced pixels, measured on the sky. See starLayer.
const STAR_PIXEL_SIGMA: f32 = 0.25;

/// One layer of stars, seeded on a 3D lattice around the celestial sphere.
///
/// Stars used to be hashed on the six faces of a cube, and the neighbor walk
/// never crossed a face edge, so stars were clipped in half along the seams. A
/// 3D lattice has no seams: each cell near the sphere owns at most one star,
/// projected onto the sphere, and the eight cells around the sample point are
/// the only ones close enough to matter.
///
/// Stars are point sources, splatted as a Gaussian scaled to the traced pixel
/// and normalized over solid angle to a fixed flux. A star then carries the same
/// total light at any trace resolution, instead of shrinking far below a pixel
/// and sparkling while the camera moves and the trace goes coarse.
///
/// The width is measured on the sky, at the camera's pixel scale, so it is kept
/// to a quarter of a pixel. From this close the whole view is strongly lensed — the
/// Einstein radius is ~18 degrees at r = 40 — and the shear stretches a sky-
/// space splat tangentially by 2-3x. A wider splat turns every star into a short
/// arc, which is what an extended source really does but not what a star does.
fn starLayer(d: vec3f, density: f32, presence: f32, fluxScale: f32, seed: f32) -> vec3f {
  let p = d * density;
  let base = floor(p - 0.5);
  let sigma = max(U.view.w * STAR_PIXEL_SIGMA, 1e-5);
  let inv2s2 = 1.0 / (2.0 * sigma * sigma);
  let norm = 1.0 / (6.2831853 * sigma * sigma);
  var total = vec3f(0.0);

  for (var i = 0; i < 8; i++) {
    let cell = base + vec3f(f32(i & 1), f32((i >> 1) & 1), f32((i >> 2) & 1));
    let h = hash3(cell + seed);
    if (h.x > presence) {
      continue;
    }
    let jitter = hash3(cell + seed + 17.13);
    let star = normalize(cell + jitter) * density;
    // Angular separation from the chord in lattice units. Taking it from the
    // dot product of two unit vectors would lose it to f32 cancellation.
    let angle = length(p - star) / density;

    // Heavy-tailed brightness: most stars are faint, a few dominate.
    let flux = fluxScale * pow(h.y, 7.0);
    // Temperatures skew cool — K and M stars outnumber the hot blue ones.
    let temperature = mix(2800.0, 14000.0, h.z * h.z);
    total += planckianRgb(temperature) * (flux * norm * exp(-angle * angle * inv2s2));
  }
  return total;
}

fn valueNoise(p: vec3f) -> f32 {
  let i = floor(p);
  let f = fract(p);
  let u = f * f * (3.0 - 2.0 * f);
  return mix(
    mix(
      mix(hash3(i).x, hash3(i + vec3f(1.0, 0.0, 0.0)).x, u.x),
      mix(hash3(i + vec3f(0.0, 1.0, 0.0)).x, hash3(i + vec3f(1.0, 1.0, 0.0)).x, u.x),
      u.y,
    ),
    mix(
      mix(hash3(i + vec3f(0.0, 0.0, 1.0)).x, hash3(i + vec3f(1.0, 0.0, 1.0)).x, u.x),
      mix(hash3(i + vec3f(0.0, 1.0, 1.0)).x, hash3(i + vec3f(1.0, 1.0, 1.0)).x, u.x),
      u.y,
    ),
    u.z,
  );
}

fn fbm(p: vec3f) -> f32 {
  var total = 0.0;
  var amplitude = 0.5;
  var q = p;
  for (var i = 0; i < 4; i++) {
    total += valueNoise(q) * amplitude;
    q = q * 2.03 + 11.7;
    amplitude *= 0.5;
  }
  return total;
}

/// Normal of the galactic plane, tilted well away from the spin axis so the
/// band crosses the default view at an angle instead of running along the disk.
const GALACTIC_POLE: vec3f = vec3f(0.3401, -0.5202, 0.7835);
/// Direction of the galactic center, which lies in the galactic plane.
const GALACTIC_CENTER: vec3f = vec3f(0.9277, 0.0801, -0.3496);

/// A faint Milky Way: a soft band along a great circle, broken up by noise and
/// cut by a darker dust lane, brightening toward the galactic center. Kept far
/// below the disk so it adds depth without flattening the contrast.
fn milkyWay(d: vec3f) -> vec3f {
  let latitude = dot(d, GALACTIC_POLE);
  let band = exp(-latitude * latitude * 38.0);
  if (band < 1e-3) {
    return vec3f(0.0);
  }
  let clouds = fbm(d * 5.0);
  let dust = smoothstep(0.35, 0.75, fbm(d * 9.0 + 3.1)) * exp(-latitude * latitude * 400.0);
  let bulge = 1.0 + 1.6 * pow(max(dot(d, GALACTIC_CENTER), 0.0), 6.0);
  let glow = band * (0.35 + 0.65 * clouds) * (1.0 - 0.7 * dust) * bulge;
  return vec3f(0.0075, 0.0068, 0.0058) * glow;
}

/// Deterministic — required for the accumulation to converge. The per-sample
/// jitter is what antialiases the stars, so this is where progressive
/// refinement is most visible.
fn background(d: vec3f) -> vec3f {
  // Very nearly black. A bright, busy sky flattens the contrast that makes the
  // disk read as incandescent, so the only ambient light here is a barely
  // perceptible cool cast to keep it from banding to flat zero.
  let axis = clamp(d.z * 0.5 + 0.5, 0.0, 1.0);
  var col = mix(vec3f(0.0016, 0.0022, 0.0038), vec3f(0.0028, 0.0030, 0.0060), axis);

  col += milkyWay(d);

  // Sparse, so individual stars read against the black rather than forming a haze.
  col += starLayer(d, 60.0, 0.030, 4.0e-6, 0.0);
  col += starLayer(d, 150.0, 0.045, 0.8e-6, 31.0);
  return col;
}

// ---------------------------------------------------------------------------
// Accretion disk — Cinematic (stylized, not radiative transfer)
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

/// Lensing level of detail.
///
/// The direct image of the disk maps screen position to disk radius smoothly.
/// Every higher-order image — light that wound around the hole before reaching
/// the camera — is compressed exponentially, and by the second or third pass
/// the whole radial profile is squeezed into a band thinner than a pixel. Point
/// sampling that is aliasing a signal with no band limit, which is what makes
/// the thin arcs beside the shadow look ragged: it is variance, not geometry.
/// Rendering the disk with a flat color makes those same arcs perfectly smooth.
///
/// The image order is the number of equatorial crossings the ray has made, so
/// it measures that compression directly and costs a counter. Past the first
/// image the shading is faded to the emission of the bright inner ring, which
/// is what dominates the stack anyway. This is the same trade a mip level
/// makes for a minified texture: the detail is not resolvable, so do not
/// pretend to resolve it.
fn lensingLod(imageOrder: f32) -> f32 {
  return smoothstep(1.0, 3.0, imageOrder);
}

/// Shade an equatorial-plane crossing, Cinematic style.
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
fn diskColor(
  xc: vec3f,
  pc: vec3f,
  rc: f32,
  a: f32,
  traceSpin: f32,
  imageOrder: f32,
) -> vec3f {
  let rIsco = U.params.z;
  let rOuter = U.params.y;
  let lod = lensingLod(imageOrder);

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
  let deriv = geodesicRHS(xc, pc, traceSpin);
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
// Accretion disk — Physical (Novikov-Thorne, exact redshift, blackbody)
// ---------------------------------------------------------------------------

/// Radial profile of the Novikov-Thorne thin disk's emitted flux (Page & Thorne
/// 1974), unnormalized. Mirrors pageThorneFlux in diskReference.ts, which also
/// supplies the peak the renderer divides by.
fn pageThorneFlux(r: f32, a: f32, rIsco: f32) -> f32 {
  let x = sqrt(r);
  let x0 = sqrt(rIsco);
  if (x <= x0) {
    return 0.0;
  }

  let phase = acos(clamp(a, -1.0, 1.0)) / 3.0;
  let third = 1.0471976;
  let x1 = 2.0 * cos(phase - third);
  let x2 = 2.0 * cos(phase + third);
  let x3 = -2.0 * cos(phase);

  let bracket = x - x0 - 1.5 * a * log(x / x0)
    - pageThorneTerm(x, x0, a, x1, x2, x3)
    - pageThorneTerm(x, x0, a, x2, x1, x3)
    - pageThorneTerm(x, x0, a, x3, x1, x2);

  return max(bracket, 0.0) / (x * x * x * x * (x * x * x - 3.0 * x + 2.0 * a));
}

/// One root's logarithmic term. (x_i - a)^2 / x_i is 0/0 at a = 0 for the root
/// at the origin; its limit is zero, so the term is dropped there.
fn pageThorneTerm(x: f32, x0: f32, a: f32, xi: f32, xj: f32, xk: f32) -> f32 {
  if (abs(xi) < 1e-6) {
    return 0.0;
  }
  return 3.0 * (xi - a) * (xi - a) / (xi * (xi - xj) * (xi - xk))
    * log((x - xi) / (x0 - xi));
}

/// Redshift factor g = E_observed / E_emitted for a circular, prograde,
/// Keplerian emitter seen from infinity. Mirrors diskRedshift in
/// diskReference.ts:
///
///   u^t = (r^1.5 + a) / (r^0.75 sqrt(r^1.5 - 3 r^0.5 + 2a))
///   g   = 1 / (u^t (1 - Omega L)),   Omega = 1 / (r^1.5 + a)
///
/// L is the photon's physical angular momentum per unit energy, conserved along
/// the ray, so this one factor carries gravitational redshift, Doppler,
/// transverse Doppler and frame dragging — no local velocity needed.
fn diskRedshift(r: f32, a: f32, photonL: f32) -> f32 {
  let sr = sqrt(r);
  let r15 = r * sr;
  let omega = 1.0 / (r15 + a);
  let ut = (r15 + a) / (sqrt(sr) * sr * sqrt(max(r15 - 3.0 * sr + 2.0 * a, 1e-6)));
  return 1.0 / (ut * max(1.0 - omega * photonL, 1e-6));
}

/// Shade a disk crossing physically.
///
/// A thermal spectrum stays a blackbody under a frequency shift, at the shifted
/// temperature: I_nu,obs = g^3 B_nu/g(T) = B_nu(gT). So both the color and the
/// brightness of a disk element come from T_obs = g * T_emit alone, with
/// T_emit ~ F^(1/4) from the Novikov-Thorne flux. Everything that makes one side
/// bluer and brighter than the other falls out of that; there is nothing to tune.
fn diskPhysical(rc: f32, a: f32, photonL: f32, imageOrder: f32) -> vec3f {
  let rIsco = U.params.z;
  let rOuter = U.params.y;

  // Same level of detail as the Cinematic path: higher-order images fall back
  // to the ring that dominates them. Only the flux is affected; the redshift is
  // smooth across the image and keeps the true radius.
  let rEmit = mix(rc, rIsco * DISK_LENSED_REFERENCE, lensingLod(imageOrder));
  let flux = pageThorneFlux(rEmit, a, rIsco) * U.disk.z;
  let emitted = U.disk.y * pow(max(flux, 0.0), 0.25);
  let observed = emitted * diskRedshift(rc, a, photonL);

  let rgb = max(xyzToLinearSrgb(blackbodyXYZ(observed) * U.disk.w), vec3f(0.0));
  let outerFade = 1.0 - smoothstep(rOuter * 0.8, rOuter, rc);
  return rgb * outerFade;
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
  // Physical spin, for the disk; the geodesics run in its time reverse.
  let a = U.params.x;
  let traceSpin = -a;
  let rOuter = U.params.y;
  let rIsco = U.params.z;
  let rPlus = U.params.w;
  let diskEnabled = U.options.x > 0.5;
  let physical = U.disk.x > 0.5;
  let maxSteps = u32(U.options.y);

  var st: State;
  st.x = origin;
  st.p = nullMomentum(origin, direction, traceSpin);

  // Carried across iterations rather than recomputed: the slab test needs the
  // radius at both ends of a step, and the far end becomes the near end next
  // time round.
  var r = kerrRadius(st.x, traceSpin);

  // How many times the ray has met the equatorial plane. The first meeting that
  // lands on the disk is the direct image, the second is light that wound once
  // around the hole, and so on — so this is the image order, and the shading
  // uses it to decide how much radial detail is still resolvable. Crossings
  // inside the ISCO count: the ray passed through the disk plane there, it just
  // found no emission, and it wound just as far getting there.
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
    let deriv = geodesicRHS(st.x, st.p, traceSpin);

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
    st = rk4StepFrom(st, deriv, traceSpin, h);
    r = kerrRadius(st.x, traceSpin);

    // Where the ray meets the disk. The disk is opaque, so the first hit along
    // the backward ray is the last emission the observer sees — that is what
    // produces the lensed over-and-under image.
    if (diskEnabled) {
      let entry = slabEntry(previous.x, previousR, st.x, r);
      if (entry.hit) {
        crossings += 1.0;
        let crossing = mix(previous.x, st.x, entry.t);
        let rc = kerrRadius(crossing, traceSpin);
        if (rc >= rIsco && rc <= rOuter) {
          let pc = mix(previous.p, st.p, entry.t);
          if (physical) {
            // p_t = -1, so p_phi is already per unit energy. Negated because
            // the traced ray is the time reverse of the real photon.
            return diskPhysical(rc, a, -angularMomentum(crossing, pc), crossings);
          }
          return diskColor(crossing, pc, rc, a, traceSpin, crossings);
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

  // Pan is an off-axis lens shift: the image plane slides under a camera that
  // keeps facing the hole.
  let tanHalfFov = U.camRight.w;
  let aspect = U.camUp.w;
  let direction = normalize(
    U.camFwd.xyz
      + U.camRight.xyz * (ndc.x * tanHalfFov * aspect + U.camPos.w)
      + U.camUp.xyz * (ndc.y * tanHalfFov + U.camFwd.w),
  );

  let radiance = traceRadiance(U.camPos.xyz, direction);

  // Progressive blend. At frameIndex 0 the weight is 1, so the first pass after
  // a reset fully overwrites and no explicit clear is needed.
  let coord = vec2i(pixel);
  let previous = textureLoad(prevAccum, coord, 0);
  let weight = 1.0 / (f32(frameIndex) + 1.0);
  textureStore(nextAccum, coord, mix(previous, vec4f(radiance, 1.0), weight));
}
