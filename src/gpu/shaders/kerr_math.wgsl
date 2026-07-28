// Kerr null-geodesic integration in Kerr-Schild coordinates.
//
// This is the single source of truth for the physics: it is concatenated into
// both the tracing shader and the validation shader, so there is exactly one
// copy of these equations on the GPU.
//
// It mirrors scripts/validate-physics.mts line for line. That script runs the
// same equations in f64 under Node and is checked against known invariants; the
// in-app validation harness then compares this f32 code against those f64
// results. Keep the two in step — if you edit one, edit the other.
//
// Geometric units: G = c = M = 1. Signature -+++.
// Metric: g_uv = eta_uv + f * k_u * k_v, with k eta-null, so the inverse is
// exactly g^uv = eta^uv - f * k^u * k^v (Woodbury).
//
// The metric is static, so p_t is exactly conserved along every geodesic. It is
// fixed at p_t = -1 and is NOT part of the integrated state; the state is just
// position (x, y, z) and spatial momentum (px, py, pz).

const KERR_STEP_SCALE: f32 = 0.045;
const KERR_STEP_MIN: f32 = 0.015;
const KERR_STEP_MAX: f32 = 0.9;
const KERR_ESCAPE_RADIUS: f32 = 60.0;
const KERR_HORIZON_PAD: f32 = 1.02;

// Central-difference step for dH/dx. Pipeline-overridable so the validation
// harness can sweep it on the actual GPU: the value 0.0015 was tuned against an
// f64 prototype, and f32 has a different roundoff/truncation balance, so it is
// worth measuring rather than assuming.
override KERR_GRAD_EPS: f32 = 0.0015;

struct MetricFK {
  f: f32,
  k: vec3f,
}

struct Deriv {
  dx: vec3f,
  dp: vec3f,
}

struct State {
  x: vec3f,
  p: vec3f,
}

/// Kerr-Schild radius: the positive root of r^4 - (rho^2 - a^2) r^2 - a^2 z^2 = 0.
fn kerrRadius(x: vec3f, a: f32) -> f32 {
  let rho2 = dot(x, x);
  let a2 = a * a;
  let t = rho2 - a2;
  let r2 = 0.5 * (t + sqrt(max(t * t + 4.0 * a2 * x.z * x.z, 0.0)));
  return sqrt(max(r2, 1e-12));
}

/// Scalar f and the spatial part of the null covector k (k_t = 1, so k^t = -1).
fn kerrFK(x: vec3f, a: f32) -> MetricFK {
  let r = kerrRadius(x, a);
  let a2 = a * a;
  let r2 = r * r;
  var m: MetricFK;
  m.f = 2.0 * r2 * r / (r2 * r2 + a2 * x.z * x.z);
  let denom = r2 + a2;
  m.k = vec3f(
    (r * x.x + a * x.y) / denom,
    (r * x.y - a * x.x) / denom,
    x.z / r,
  );
  return m;
}

/// The only x-dependent part of H: W(x, p) = -0.5 * f * S^2.
///
/// Differencing W instead of the full H yields an identical gradient, since the
/// remaining 0.5*(-1 + dot(p,p)) term has zero x-derivative. This matters a great
/// deal in f32: subtracting two O(1) Hamiltonians to recover an O(1e-5) difference
/// is catastrophic cancellation, whereas W is small and the subtraction is well
/// conditioned.
fn metricPotential(x: vec3f, p: vec3f, a: f32) -> f32 {
  let m = kerrFK(x, a);
  let s = 1.0 + dot(m.k, p);
  return -0.5 * m.f * s * s;
}

/// H(x, p) = 0.5 * (-p_t^2 + dot(p,p) - f * S^2) with p_t = -1.
/// Zero for null geodesics — this is the constraint the validation harness tracks.
fn hamiltonian(x: vec3f, p: vec3f, a: f32) -> f32 {
  return 0.5 * (-1.0 + dot(p, p)) + metricPotential(x, p, a);
}

/// Axial angular momentum p_phi = x*py - y*px, conserved by axisymmetry.
fn angularMomentum(x: vec3f, p: vec3f) -> f32 {
  return x.x * p.y - x.y * p.x;
}

/// Geodesic RHS.
///   dx/dlambda = dH/dp = p - f * S * k   (exact closed form, g^uv p_v)
///   dp/dlambda = -dH/dx                  (central differences; no autodiff here)
fn geodesicRHS(x: vec3f, p: vec3f, a: f32) -> Deriv {
  let m = kerrFK(x, a);
  let s = 1.0 + dot(m.k, p);

  let e = KERR_GRAD_EPS;
  let inv2e = 1.0 / (2.0 * e);
  let ex = vec3f(e, 0.0, 0.0);
  let ey = vec3f(0.0, e, 0.0);
  let ez = vec3f(0.0, 0.0, e);

  var d: Deriv;
  d.dx = p - m.k * (m.f * s);
  d.dp = vec3f(
    -(metricPotential(x + ex, p, a) - metricPotential(x - ex, p, a)) * inv2e,
    -(metricPotential(x + ey, p, a) - metricPotential(x - ey, p, a)) * inv2e,
    -(metricPotential(x + ez, p, a) - metricPotential(x - ez, p, a)) * inv2e,
  );
  return d;
}

/// One classical RK4 step of the coupled (x, p) system.
fn rk4Step(st: State, a: f32, h: f32) -> State {
  let half = h * 0.5;
  let k1 = geodesicRHS(st.x, st.p, a);
  let k2 = geodesicRHS(st.x + k1.dx * half, st.p + k1.dp * half, a);
  let k3 = geodesicRHS(st.x + k2.dx * half, st.p + k2.dp * half, a);
  let k4 = geodesicRHS(st.x + k3.dx * h, st.p + k3.dp * h, a);

  let sixth = h / 6.0;
  var out: State;
  out.x = st.x + (k1.dx + 2.0 * k2.dx + 2.0 * k3.dx + k4.dx) * sixth;
  out.p = st.p + (k1.dp + 2.0 * k2.dp + 2.0 * k3.dp + k4.dp) * sixth;
  return out;
}

/// Smaller steps near the hole, larger far away.
fn adaptiveStep(r: f32) -> f32 {
  return clamp(r * KERR_STEP_SCALE, KERR_STEP_MIN, KERR_STEP_MAX);
}

/// Outer event horizon.
fn horizonRadius(a: f32) -> f32 {
  return 1.0 + sqrt(max(1.0 - a * a, 0.0));
}

fn cbrtSafe(v: f32) -> f32 {
  return sign(v) * pow(abs(v), 1.0 / 3.0);
}

/// Prograde ISCO, Bardeen-Press-Teukolsky.
/// NOT the photon-orbit formula 2*(1+cos((2/3)*acos(-a))) — that is a different
/// radius (4 vs 6 at a=0) and substituting it here is a real, previously-caught bug.
fn iscoRadius(a: f32) -> f32 {
  let a2 = a * a;
  let z1 = 1.0 + cbrtSafe(1.0 - a2) * (cbrtSafe(1.0 + a) + cbrtSafe(1.0 - a));
  let z2 = sqrt(3.0 * a2 + z1 * z1);
  return 3.0 + z2 - sqrt(max((3.0 - z1) * (3.0 + z1 + 2.0 * z2), 0.0));
}

/// Scale a unit direction into a null momentum at x, by solving H = 0 for s in p = s*d:
///   A s^2 + B s + C = 0,  A = 1 - f (k.d)^2,  B = -2 f (k.d),  C = -(1 + f)
/// Flat-space check: f = 0 gives A=1, B=0, C=-1, so s = 1.
fn nullMomentum(x: vec3f, d: vec3f, a: f32) -> vec3f {
  let m = kerrFK(x, a);
  let kd = dot(m.k, d);
  let qa = 1.0 - m.f * kd * kd;
  let qb = -2.0 * m.f * kd;
  let qc = -(1.0 + m.f);
  let disc = sqrt(max(qb * qb - 4.0 * qa * qc, 0.0));
  var s: f32;
  if (abs(qa) < 1e-9) {
    s = -qc / qb;
  } else {
    s = (-qb + disc) / (2.0 * qa);
  }
  return d * s;
}
