/**
 * Physical accretion-disk shading, f64 reference.
 *
 * Mirrors the Physical-mode functions in `src/gpu/shaders/trace.wgsl`
 * (`diskRedshift`, `pageThorneFlux`, `blackbodyXYZ`, `xyzToLinearSrgb`). The
 * renderer also uses it on the CPU to derive per-frame normalizations — the peak
 * of the flux profile and the luminance at the peak temperature — so the shader
 * never has to search for either.
 *
 * Geometric units, G = c = M = 1.
 */

import { iscoRadius } from './kerrReference.ts';

/**
 * Redshift factor g = E_observed / E_emitted for light leaving a circular,
 * prograde, Keplerian emitter in the equatorial plane and reaching a static
 * observer at infinity.
 *
 *   Omega = 1 / (r^1.5 + a)
 *   u^t   = (r^1.5 + a) / (r^0.75 sqrt(r^1.5 - 3 r^0.5 + 2a))     (Bardeen 1972)
 *   g     = 1 / (u^t (1 - Omega L))
 *
 * where L = p_phi / (-p_t) is the photon's conserved angular momentum per unit
 * energy. This single factor carries gravitational redshift, the ordinary and
 * transverse Doppler shifts, and frame dragging. At a = 0 seen face-on (L = 0)
 * it reduces to sqrt(1 - 3/r).
 */
export function diskRedshift(r: number, a: number, photonL: number): number {
  const sr = Math.sqrt(r);
  const r15 = r * sr;
  const omega = 1 / (r15 + a);
  const ut = (r15 + a) / (Math.sqrt(sr) * sr * Math.sqrt(Math.max(r15 - 3 * sr + 2 * a, 1e-6)));
  return 1 / (ut * Math.max(1 - omega * photonL, 1e-6));
}

/**
 * Radial profile of the Novikov-Thorne thin disk's emitted flux, from Page &
 * Thorne (1974). Unnormalized — only its shape matters, since the renderer
 * divides by the peak. Zero at the ISCO, peaks a little outside it, and falls
 * as r^-3 far out.
 *
 * With x = sqrt(r), x0 = sqrt(r_isco), and x1..x3 the roots of x^3 - 3x + 2a:
 *
 *   F ~ 1 / (x^4 (x^3 - 3x + 2a)) * [ x - x0 - 1.5 a ln(x/x0)
 *         - sum_i 3 (x_i - a)^2 / (x_i (x_i - x_j)(x_i - x_k)) ln((x - x_i)/(x0 - x_i)) ]
 */
export function pageThorneFlux(r: number, a: number, rIsco: number): number {
  const x = Math.sqrt(r);
  const x0 = Math.sqrt(rIsco);
  if (x <= x0) return 0;

  const phase = Math.acos(Math.min(Math.max(a, -1), 1)) / 3;
  const x1 = 2 * Math.cos(phase - Math.PI / 3);
  const x2 = 2 * Math.cos(phase + Math.PI / 3);
  const x3 = -2 * Math.cos(phase);

  // (x_i - a)^2 / x_i is 0/0 at a = 0 for the root that sits at the origin;
  // its limit is zero (x2 ~ 2a/3 there), so the term is dropped.
  const term = (xi: number, xj: number, xk: number): number => {
    if (Math.abs(xi) < 1e-6) return 0;
    return (
      ((3 * (xi - a) * (xi - a)) / (xi * (xi - xj) * (xi - xk))) *
      Math.log((x - xi) / (x0 - xi))
    );
  };

  const bracket =
    x -
    x0 -
    1.5 * a * Math.log(x / x0) -
    term(x1, x2, x3) -
    term(x2, x1, x3) -
    term(x3, x1, x2);

  return Math.max(bracket, 0) / (x ** 4 * (x ** 3 - 3 * x + 2 * a));
}

/**
 * Peak of the flux profile over the disk, found by sampling. The peak sits
 * between ~1.2 and ~1.8 r_isco depending on spin; sampling out to 4 r_isco
 * brackets it with room to spare.
 */
export function pageThorneFluxPeak(a: number): number {
  const rIsco = iscoRadius(a);
  let peak = 0;
  for (let i = 1; i <= 512; i++) {
    const r = rIsco * (1 + (3 * i) / 512);
    peak = Math.max(peak, pageThorneFlux(r, a, rIsco));
  }
  return peak;
}

// ---------------------------------------------------------------------------
// Color
// ---------------------------------------------------------------------------

/** Piecewise Gaussian lobe used by the CIE fit below. */
const lobe = (x: number, mu: number, below: number, above: number): number => {
  const t = (x - mu) / (x < mu ? below : above);
  return Math.exp(-0.5 * t * t);
};

/**
 * CIE 1931 2-degree color matching functions, as the multi-lobe Gaussian fit
 * of Wyman, Sloan & Shirley (2013). Wavelength in nanometers.
 */
export function cieMatching(lambda: number): [number, number, number] {
  return [
    1.056 * lobe(lambda, 599.8, 37.9, 31.0) +
      0.362 * lobe(lambda, 442.0, 16.0, 26.7) -
      0.065 * lobe(lambda, 501.1, 20.4, 26.2),
    0.821 * lobe(lambda, 568.8, 46.9, 40.5) + 0.286 * lobe(lambda, 530.9, 16.3, 31.1),
    1.217 * lobe(lambda, 437.0, 11.8, 36.0) + 0.681 * lobe(lambda, 459.0, 26.0, 13.8),
  ];
}

/** Number of wavelength samples across 380-780 nm. Must match trace.wgsl. */
export const BLACKBODY_SAMPLES = 20;

/** hc/k in nm*K. */
const SECOND_RADIATION_CONSTANT = 1.4388e7;

/**
 * CIE XYZ of a blackbody at temperature T (kelvin), by midpoint integration of
 * Planck's law against the matching functions. Arbitrary but consistent scale:
 * the renderer divides by the Y of a reference temperature.
 *
 * Planck in wavelength, with lambda in micrometers so f32 stays in range:
 *   B ~ lambda^-5 / (exp(hc / (lambda k T)) - 1)
 */
export function blackbodyXYZ(temperature: number): [number, number, number] {
  const t = Math.max(temperature, 500);
  const step = 400 / BLACKBODY_SAMPLES;
  let x = 0;
  let y = 0;
  let z = 0;
  for (let i = 0; i < BLACKBODY_SAMPLES; i++) {
    const lambda = 380 + (i + 0.5) * step;
    const micrometers = lambda * 1e-3;
    const planck =
      1 / (micrometers ** 5 * (Math.exp(SECOND_RADIATION_CONSTANT / (lambda * t)) - 1));
    const [cx, cy, cz] = cieMatching(lambda);
    x += planck * cx;
    y += planck * cy;
    z += planck * cz;
  }
  return [x * step, y * step, z * step];
}

/** CIE XYZ to linear sRGB (D65). Out-of-gamut components are left negative. */
export function xyzToLinearSrgb([x, y, z]: [number, number, number]): [number, number, number] {
  return [
    3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
    -0.969266 * x + 1.8760108 * y + 0.041556 * z,
    0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
  ];
}
