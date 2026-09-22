/**
 * Double-precision reference implementation of the Kerr null-geodesic integrator.
 *
 * This is a faithful mirror of `src/gpu/shaders/kerr_math.wgsl` — same metric,
 * same Hamiltonian, same closed-form gradient, same RK4 with the same adaptive
 * step rule — evaluated in f64 instead of f32.
 *
 * It also keeps the central-difference gradient the shader used to run. Pass a
 * numeric `eps` to any of the integrator entry points to select it; that makes
 * it an independent oracle for the analytic derivative, since the two share
 * nothing but the definition of W.
 *
 * It has two consumers:
 *   - `scripts/validate-physics.mts` asserts the invariants under Node.
 *   - `src/gpu/validatePhysics.ts` computes live reference values in the browser
 *     to compare against what the GPU actually produced.
 *
 * Keeping one reference means the in-app harness has no baked-in golden numbers
 * that can drift out of date. If you edit the WGSL, edit this too.
 *
 * Geometric units: G = c = M = 1. Signature -+++.
 */

export type Vec3 = readonly [number, number, number];

export const v3 = (x: number, y: number, z: number): Vec3 => [x, y, z];
export const add = (a: Vec3, b: Vec3): Vec3 => [
  a[0] + b[0],
  a[1] + b[1],
  a[2] + b[2],
];
export const sub = (a: Vec3, b: Vec3): Vec3 => [
  a[0] - b[0],
  a[1] - b[1],
  a[2] - b[2],
];
export const mul = (a: Vec3, s: number): Vec3 => [a[0] * s, a[1] * s, a[2] * s];
export const dot = (a: Vec3, b: Vec3): number =>
  a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
export const length = (a: Vec3): number => Math.sqrt(dot(a, a));
export const normalize = (a: Vec3): Vec3 => mul(a, 1 / length(a));

// Must match the constants at the top of kerr_math.wgsl.
export const STEP_SCALE = 0.045;
export const STEP_MIN = 0.015;
export const STEP_MAX = 0.9;
export const ESCAPE_RADIUS = 60;
/** Central-difference step for the finite-difference oracle only. */
export const GRAD_EPS = 0.0015;
export const HORIZON_PAD = 1.02;
export const PLANE_APPROACH = 0.85;
export const PLANE_STEP_MIN = 0.01;

// ---------------------------------------------------------------------------
// Kerr-Schild metric quantities
// ---------------------------------------------------------------------------

/** Kerr-Schild radius: positive root of r^4 - (rho^2 - a^2) r^2 - a^2 z^2 = 0. */
export function kerrRadius(x: Vec3, a: number): number {
  const rho2 = dot(x, x);
  const a2 = a * a;
  const t = rho2 - a2;
  const r2 = 0.5 * (t + Math.sqrt(Math.max(t * t + 4 * a2 * x[2] * x[2], 0)));
  return Math.sqrt(Math.max(r2, 1e-12));
}

/** Scalar f and the spatial part of the null covector k (k_t = 1). */
export function kerrFK(x: Vec3, a: number): { f: number; k: Vec3 } {
  const r = kerrRadius(x, a);
  const a2 = a * a;
  const r2 = r * r;
  const f = (2 * r2 * r) / (r2 * r2 + a2 * x[2] * x[2]);
  const denom = r2 + a2;
  const k = v3(
    (r * x[0] + a * x[1]) / denom,
    (r * x[1] - a * x[0]) / denom,
    x[2] / r,
  );
  return { f, k };
}

/**
 * The only x-dependent part of H: W(x, p) = -0.5 * f * S^2.
 *
 * Differencing W rather than the full H gives an identical gradient, because the
 * remaining 0.5*(-1 + dot(p,p)) term has zero x-derivative. It matters because the
 * GPU runs this in f32, where subtracting two O(1) Hamiltonians to recover an
 * O(1e-5) difference is catastrophic cancellation. The reference uses the same
 * formulation so both sides stay in lockstep.
 */
export function metricPotential(x: Vec3, p: Vec3, a: number): number {
  const { f, k } = kerrFK(x, a);
  const s = 1 + dot(k, p);
  return -0.5 * f * s * s;
}

/**
 * H(x, p) = 0.5 * (-p_t^2 + dot(p,p) - f * S^2), with p_t = -1 held fixed.
 * The metric is static, so p_t is exactly conserved and is NOT part of the state.
 */
export function hamiltonian(x: Vec3, p: Vec3, a: number): number {
  return 0.5 * (-1 + dot(p, p)) + metricPotential(x, p, a);
}

/** Axial angular momentum p_phi = x*py - y*px, conserved by axisymmetry. */
export const angularMomentum = (x: Vec3, p: Vec3): number =>
  x[0] * p[1] - x[1] * p[0];

/**
 * Geodesic RHS.
 *   dx/dlambda = dH/dp = p - f * S * k                        (exact closed form)
 *   dp/dlambda = -dH/dx = 0.5 S^2 grad f + f S (p . grad k)   (closed form)
 *
 * Mirrors geodesicRHS in kerr_math.wgsl, including its grad r from implicitly
 * differentiating the Kerr-Schild quartic:
 *   grad r = r / D * (x r^2, y r^2, z (r^2 + a^2)),   D = r^4 + a^2 z^2
 *
 * With a numeric `eps` the momentum derivative is instead taken by central
 * differences of W — the shader's former method, kept as a test oracle.
 */
export function geodesicRHS(
  x: Vec3,
  p: Vec3,
  a: number,
  eps?: number,
): { dx: Vec3; dp: Vec3 } {
  if (eps !== undefined) return geodesicRHSFiniteDiff(x, p, a, eps);

  const r = kerrRadius(x, a);
  const a2 = a * a;
  const r2 = r * r;
  const q = r2 + a2;
  const d = r2 * r2 + a2 * x[2] * x[2];

  const f = (2 * r2 * r) / d;
  const k = v3((r * x[0] + a * x[1]) / q, (r * x[1] - a * x[0]) / q, x[2] / r);
  const s = 1 + dot(k, p);

  const gradR = mul(v3(x[0] * r2, x[1] * r2, x[2] * q), r / d);

  const d2 = d * d;
  const gradF = sub(
    mul(gradR, (6 * r2 * d - 8 * r2 * r2 * r2) / d2),
    v3(0, 0, (4 * r2 * r * a2 * x[2]) / d2),
  );

  const throughR =
    (p[0] * x[0] + p[1] * x[1]) / q -
    (2 * r * (p[0] * k[0] + p[1] * k[1])) / q -
    (p[2] * x[2]) / r2;
  const direct = v3((r * p[0] - a * p[1]) / q, (a * p[0] + r * p[1]) / q, p[2] / r);
  const pGradK = add(mul(gradR, throughR), direct);

  return {
    dx: sub(p, mul(k, f * s)),
    dp: add(mul(gradF, 0.5 * s * s), mul(pGradK, f * s)),
  };
}

/** dp/dlambda by central differences of W. The analytic form's oracle. */
export function geodesicRHSFiniteDiff(
  x: Vec3,
  p: Vec3,
  a: number,
  eps: number = GRAD_EPS,
): { dx: Vec3; dp: Vec3 } {
  const { f, k } = kerrFK(x, a);
  const s = 1 + dot(k, p);
  const dx = sub(p, mul(k, f * s));

  const inv2e = 1 / (2 * eps);
  const ex = v3(eps, 0, 0);
  const ey = v3(0, eps, 0);
  const ez = v3(0, 0, eps);
  const dp = v3(
    -(metricPotential(add(x, ex), p, a) - metricPotential(sub(x, ex), p, a)) *
      inv2e,
    -(metricPotential(add(x, ey), p, a) - metricPotential(sub(x, ey), p, a)) *
      inv2e,
    -(metricPotential(add(x, ez), p, a) - metricPotential(sub(x, ez), p, a)) *
      inv2e,
  );
  return { dx, dp };
}

/** One classical RK4 step of the coupled (x, p) system. */
export function rk4Step(
  x: Vec3,
  p: Vec3,
  a: number,
  h: number,
  eps?: number,
): { x: Vec3; p: Vec3 } {
  const k1 = geodesicRHS(x, p, a, eps);
  const k2 = geodesicRHS(
    add(x, mul(k1.dx, h / 2)),
    add(p, mul(k1.dp, h / 2)),
    a,
    eps,
  );
  const k3 = geodesicRHS(
    add(x, mul(k2.dx, h / 2)),
    add(p, mul(k2.dp, h / 2)),
    a,
    eps,
  );
  const k4 = geodesicRHS(add(x, mul(k3.dx, h)), add(p, mul(k3.dp, h)), a, eps);

  const wx = mul(
    add(add(k1.dx, mul(k2.dx, 2)), add(mul(k3.dx, 2), k4.dx)),
    h / 6,
  );
  const wp = mul(
    add(add(k1.dp, mul(k2.dp, 2)), add(mul(k3.dp, 2), k4.dp)),
    h / 6,
  );
  return { x: add(x, wx), p: add(p, wp) };
}

// ---------------------------------------------------------------------------
// Characteristic radii
// ---------------------------------------------------------------------------

/** Outer event horizon. */
export const horizonRadius = (a: number): number => 1 + Math.sqrt(1 - a * a);

/**
 * Prograde ISCO, Bardeen-Press-Teukolsky.
 * NOT the photon-orbit formula 2*(1+cos((2/3)*acos(-a))) — a different radius
 * (4 vs 6 at a=0), and substituting it here is a real, previously-caught bug.
 */
export function iscoRadius(a: number): number {
  const a2 = a * a;
  const z1 = 1 + Math.cbrt(1 - a2) * (Math.cbrt(1 + a) + Math.cbrt(1 - a));
  const z2 = Math.sqrt(3 * a2 + z1 * z1);
  return 3 + z2 - Math.sqrt((3 - z1) * (3 + z1 + 2 * z2));
}

/** Smaller steps near the hole, larger far away. */
export const adaptiveStep = (r: number, stepScale: number = STEP_SCALE): number =>
  Math.min(
    Math.max(r * stepScale, STEP_MIN * (stepScale / STEP_SCALE)),
    STEP_MAX,
  );

/**
 * Shorten a step so it cannot carry the ray through the equatorial plane.
 *
 * Mirrors planeLimitedStep in kerr_math.wgsl. Only the render path uses it —
 * the disk crossing test sees at most one sign change per step, so a ray that
 * enters and leaves the plane within one step is missed and integrates on into
 * the horizon. It is not part of the geodesic itself, so the validation rays do
 * not use it; it only ever makes a step smaller, which cannot hurt conservation.
 */
export function planeLimitedStep(base: number, z: number, dz: number): number {
  const speed = Math.abs(dz);
  if (speed < 1e-6) return base;
  return Math.min(base, Math.max((Math.abs(z) / speed) * PLANE_APPROACH, PLANE_STEP_MIN));
}

/**
 * Scale a unit direction into a null momentum at x, by solving H = 0 for s in p = s*d:
 *   A s^2 + B s + C = 0,  A = 1 - f (k.d)^2,  B = -2 f (k.d),  C = -(1 + f)
 * Flat-space check: f = 0 gives A=1, B=0, C=-1, so s = 1.
 */
export function nullMomentum(x: Vec3, d: Vec3, a: number): Vec3 {
  const { f, k } = kerrFK(x, a);
  const kd = dot(k, d);
  const qa = 1 - f * kd * kd;
  const qb = -2 * f * kd;
  const qc = -(1 + f);
  const disc = Math.sqrt(Math.max(qb * qb - 4 * qa * qc, 0));
  const s = Math.abs(qa) < 1e-12 ? -qc / qb : (-qb + disc) / (2 * qa);
  return mul(d, s);
}

// ---------------------------------------------------------------------------
// Tracing with invariant tracking
// ---------------------------------------------------------------------------

export type Fate = 'maxSteps' | 'captured' | 'escaped';

/** Matches the `fate` encoding in validate.wgsl. */
export const FATE_ORDER: Fate[] = ['maxSteps', 'captured', 'escaped'];

export type TraceResult = {
  fate: Fate;
  steps: number;
  maxAbsH: number;
  maxRelL: number;
  l0: number;
  h0: number;
  minRadius: number;
  samples: { step: number; r: number; h: number; l: number }[];
};

export function traceRay(
  origin: Vec3,
  direction: Vec3,
  a: number,
  maxSteps: number,
  eps?: number,
  sampleEvery = 0,
  stepScale: number = STEP_SCALE,
): TraceResult {
  let x = origin;
  let p = nullMomentum(origin, normalize(direction), a);

  const rPlus = horizonRadius(a);
  const l0 = angularMomentum(x, p);
  const h0 = hamiltonian(x, p, a);
  const lScale = Math.max(Math.abs(l0), 1e-3);

  let maxAbsH = Math.abs(h0);
  let maxRelL = 0;
  let minRadius = kerrRadius(x, a);
  const samples: TraceResult['samples'] = [];

  let fate: Fate = 'maxSteps';
  let step = 0;
  for (; step < maxSteps; step++) {
    const r = kerrRadius(x, a);
    minRadius = Math.min(minRadius, r);

    const h = Math.abs(hamiltonian(x, p, a));
    const l = angularMomentum(x, p);
    maxAbsH = Math.max(maxAbsH, h);
    maxRelL = Math.max(maxRelL, Math.abs(l - l0) / lScale);
    if (sampleEvery > 0 && step % sampleEvery === 0) {
      samples.push({ step, r, h, l });
    }

    if (r < rPlus * HORIZON_PAD) {
      fate = 'captured';
      break;
    }
    if (r > ESCAPE_RADIUS) {
      fate = 'escaped';
      break;
    }

    ({ x, p } = rk4Step(x, p, a, adaptiveStep(r, stepScale), eps));
  }

  return { fate, steps: step, maxAbsH, maxRelL, l0, h0, minRadius, samples };
}

// ---------------------------------------------------------------------------
// Shared ray suite — traced identically on CPU (f64) and GPU (f32)
// ---------------------------------------------------------------------------

export type RaySpec = {
  name: string;
  a: number;
  origin: Vec3;
  direction: Vec3;
  expect?: Fate;
  /**
   * Plunging rays get a looser bound on |H|. Near the horizon f ~ 2/r is O(1)
   * and momenta grow without bound, so RK4 truncation error rises sharply over
   * the final steps. That is integrator stiffness at a coordinate the ray is
   * about to terminate at, not a defect in the RHS — the step-refinement sweep
   * in the Node harness demonstrates it shrinks as O(h^4). These pixels are
   * black regardless.
   */
  hTol?: number;
};

export const REFERENCE_RAYS: RaySpec[] = [
  {
    name: 'a=0     b=8    weak deflection',
    a: 0,
    origin: v3(30, 8, 0),
    direction: v3(-1, 0, 0),
    expect: 'escaped',
  },
  {
    name: 'a=0     b=5.3  near-critical (b_crit=5.196)',
    a: 0,
    origin: v3(30, 5.3, 0),
    direction: v3(-1, 0, 0),
    expect: 'escaped',
  },
  {
    name: 'a=0     b=4    captured',
    a: 0,
    origin: v3(30, 4, 0),
    direction: v3(-1, 0, 0),
    expect: 'captured',
    hTol: 1e-3,
  },
  {
    name: 'a=0.9   b=+8   prograde',
    a: 0.9,
    origin: v3(30, 8, 0),
    direction: v3(-1, 0, 0),
    expect: 'escaped',
  },
  {
    name: 'a=0.9   b=-8   retrograde',
    a: 0.9,
    origin: v3(30, -8, 0),
    direction: v3(-1, 0, 0),
    expect: 'escaped',
  },
  {
    name: 'a=0.9   off-equatorial 3D ray',
    a: 0.9,
    origin: v3(25, 6, 8),
    direction: v3(-1, -0.05, -0.28),
  },
  {
    name: 'a=0.998 near-extremal, off-equatorial',
    a: 0.998,
    origin: v3(28, 7, -6),
    direction: v3(-1, 0, 0.2),
  },
];

export const REFERENCE_MAX_STEPS = 4000;
