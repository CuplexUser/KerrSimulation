/**
 * Regression tests for the f64 geodesic reference in `src/physics/kerrReference.ts`.
 *
 * `scripts/validate-physics.mts` already exercises this code, but it is a
 * diagnostic report: it prints tables a human reads, and a failure surfaces as
 * one non-zero exit code for the whole file. These are the same invariants
 * expressed as individual assertions, plus the properties that report does not
 * cover — monotonicity in spin, the step-control helpers, and the flat-space
 * limits that pin down sign conventions.
 *
 * Nothing here is a golden number captured from a previous run. Every bound is
 * either analytic (r+ = 2 at a = 0), a published value (ISCO at a = 0.998), or a
 * property that must hold for any correct integrator (H stays zero, L is
 * conserved, truncation error shrinks with the step). A recorded output would
 * lock in whatever the code did on the day it was written, including its bugs.
 */

import { strict as assert } from 'node:assert';
import { describe, test } from 'node:test';

import {
  ESCAPE_RADIUS,
  GRAD_EPS,
  HORIZON_PAD,
  PLANE_APPROACH,
  PLANE_STEP_MIN,
  REFERENCE_MAX_STEPS,
  REFERENCE_RAYS,
  STEP_MAX,
  STEP_MIN,
  STEP_SCALE,
  adaptiveStep,
  angularMomentum,
  dot,
  geodesicRHS,
  hamiltonian,
  horizonRadius,
  iscoRadius,
  kerrRadius,
  length,
  normalize,
  nullMomentum,
  planeLimitedStep,
  rk4Step,
  sub,
  traceRay,
  v3,
  type Vec3,
} from '../src/physics/kerrReference.ts';

/** Spins spanning Schwarzschild to near-extremal, avoiding a = 1 exactly. */
const SPINS = [0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.998];

const traceImpact = (a: number, impact: number) =>
  traceRay(v3(40, impact, 0), v3(-1, 0, 0), a, REFERENCE_MAX_STEPS, GRAD_EPS);

/** The photon-orbit radius, kept here purely to assert the ISCO is not it. */
const photonOrbitRadius = (a: number) =>
  2 * (1 + Math.cos((2 / 3) * Math.acos(-a)));

/** Worst |H| on the plunging ray at a given step scale. */
const plungeResidual = (scale: number) =>
  traceRay(v3(30, 4, 0), v3(-1, 0, 0), 0, 50_000, GRAD_EPS, 0, scale).maxAbsH;

// ---------------------------------------------------------------------------

describe('characteristic radii', () => {
  test('horizon is the analytic Kerr value', () => {
    assert.equal(horizonRadius(0), 2);
    assert.equal(horizonRadius(1), 1);
    for (const a of SPINS) {
      assert.ok(
        Math.abs(horizonRadius(a) - (1 + Math.sqrt(1 - a * a))) < 1e-12,
        `r+ wrong at a=${a}`,
      );
    }
  });

  test('horizon shrinks monotonically with spin, never below M', () => {
    for (let i = 1; i < SPINS.length; i++) {
      assert.ok(
        horizonRadius(SPINS[i]) < horizonRadius(SPINS[i - 1]),
        `r+ did not shrink from a=${SPINS[i - 1]} to a=${SPINS[i]}`,
      );
    }
    for (const a of SPINS) assert.ok(horizonRadius(a) >= 1);
  });

  test('ISCO matches Bardeen-Press-Teukolsky at known spins', () => {
    // 6M at a=0 and 1M at a=1 are exact; the middle values are the standard
    // published prograde figures.
    assert.ok(Math.abs(iscoRadius(0) - 6) < 1e-9, `got ${iscoRadius(0)}`);
    assert.ok(Math.abs(iscoRadius(1) - 1) < 1e-6, `got ${iscoRadius(1)}`);
    assert.ok(Math.abs(iscoRadius(0.5) - 4.233) < 5e-3);
    assert.ok(Math.abs(iscoRadius(0.998) - 1.2367) < 5e-3);
  });

  test('ISCO shrinks monotonically and always lies outside the horizon', () => {
    for (let i = 1; i < SPINS.length; i++) {
      assert.ok(
        iscoRadius(SPINS[i]) < iscoRadius(SPINS[i - 1]),
        `ISCO did not shrink from a=${SPINS[i - 1]} to a=${SPINS[i]}`,
      );
    }
    for (const a of SPINS) {
      assert.ok(
        iscoRadius(a) > horizonRadius(a),
        `ISCO inside the horizon at a=${a}`,
      );
    }
  });

  test('ISCO is not the photon-orbit formula', () => {
    // A previously-caught bug: 2*(1+cos((2/3)*acos(-a))) is a different radius
    // (4 at a=0, or 3 given IEEE's acos(-0) = pi/2). Either way, not 6.
    assert.ok(Math.abs(iscoRadius(0) - photonOrbitRadius(0)) > 1);
  });

  test('Kerr-Schild radius reduces to the Euclidean radius at a = 0', () => {
    for (const x of [v3(3, 4, 0), v3(1, 2, 2), v3(0, 0, 7)] as Vec3[]) {
      assert.ok(Math.abs(kerrRadius(x, 0) - length(x)) < 1e-12);
    }
  });

  test('Kerr-Schild radius is the oblate-spheroidal root at a > 0', () => {
    // On the axis (rho = 0, z) the root is r = |z|; in the equatorial plane it
    // satisfies r^2 = rho^2 - a^2, so a ring singularity of radius a sits at r=0.
    const a = 0.9;
    assert.ok(Math.abs(kerrRadius(v3(0, 0, 5), a) - 5) < 1e-12);
    assert.ok(
      Math.abs(kerrRadius(v3(10, 0, 0), a) - Math.sqrt(100 - a * a)) < 1e-12,
    );
  });
});

// ---------------------------------------------------------------------------

describe('null initial conditions', () => {
  const launches: [number, Vec3][] = [
    [0, v3(30, 8, 0)],
    [0.9, v3(30, 8, 0)],
    [0.9, v3(12, -5, 9)],
    [0.998, v3(15, 3, -4)],
    [0.998, v3(8, 0, 0)],
  ];

  test('nullMomentum solves H = 0 exactly at launch', () => {
    for (const [a, origin] of launches) {
      const d = normalize(sub(v3(0, 0, 0), origin));
      const h = hamiltonian(origin, nullMomentum(origin, d, a), a);
      assert.ok(Math.abs(h) < 1e-14, `a=${a} origin=(${origin.join(',')}): H=${h}`);
    }
  });

  test('momentum stays parallel to the requested direction and forward-going', () => {
    for (const [a, origin] of launches) {
      const d = normalize(sub(v3(0, 0, 0), origin));
      const p = nullMomentum(origin, d, a);
      const s = dot(p, d);
      assert.ok(s > 0, `momentum reversed at a=${a}`);
      // p = s*d, so the component orthogonal to d must vanish.
      const orthogonal = length(sub(p, [d[0] * s, d[1] * s, d[2] * s]));
      assert.ok(orthogonal < 1e-12);
    }
  });

  test('flat-space limit gives a unit momentum', () => {
    const p = nullMomentum(v3(1e7, 0, 0), v3(-1, 0, 0), 0);
    assert.ok(Math.abs(length(p) - 1) < 1e-6, `|p| = ${length(p)}`);
  });
});

// ---------------------------------------------------------------------------

describe('conservation along the reference ray suite', () => {
  for (const ray of REFERENCE_RAYS) {
    test(ray.name, () => {
      const result = traceRay(
        ray.origin,
        ray.direction,
        ray.a,
        REFERENCE_MAX_STEPS,
        GRAD_EPS,
      );

      assert.ok(
        result.maxAbsH < (ray.hTol ?? 1e-5),
        `null constraint drifted: max|H| = ${result.maxAbsH}`,
      );
      assert.ok(
        result.maxRelL < 1e-5,
        `angular momentum drifted: max dL/L = ${result.maxRelL}`,
      );
      if (ray.expect) assert.equal(result.fate, ray.expect);

      // Termination must agree with the radius that triggered it, or the fate
      // encoding the GPU harness compares against is meaningless.
      if (result.fate === 'captured') {
        assert.ok(result.minRadius < horizonRadius(ray.a) * HORIZON_PAD);
      }
      if (result.fate === 'escaped') {
        assert.ok(result.steps < REFERENCE_MAX_STEPS);
      }
    });
  }

  test('every reference ray starts far outside the horizon and inside the escape radius', () => {
    for (const ray of REFERENCE_RAYS) {
      const r = kerrRadius(ray.origin, ray.a);
      assert.ok(r > horizonRadius(ray.a) * HORIZON_PAD, `${ray.name}: starts inside the horizon`);
      assert.ok(r < ESCAPE_RADIUS, `${ray.name}: starts already escaped`);
      assert.ok(length(ray.direction) > 0, `${ray.name}: zero direction`);
      assert.ok(Math.abs(ray.a) < 1, `${ray.name}: spin is not sub-extremal`);
    }
  });
});

// ---------------------------------------------------------------------------

describe('frame dragging', () => {
  /** Closest approach at +b minus that at -b. Zero iff the metric is static. */
  const asymmetry = (a: number, b: number) =>
    traceImpact(a, b).minRadius - traceImpact(a, -b).minRadius;

  test('vanishes exactly at a = 0', () => {
    assert.ok(Math.abs(asymmetry(0, 6)) < 1e-12);
  });

  test('grows with spin', () => {
    assert.ok(Math.abs(asymmetry(0.9, 6)) > Math.abs(asymmetry(0.3, 6)));
  });

  test('prograde photons survive where retrograde ones are captured', () => {
    // Prograde has the smaller critical impact parameter, so at equal |b| it is
    // the counter-rotating ray that falls in. Getting this backwards is the
    // classic sign error in the k vector.
    assert.equal(traceImpact(0.9, 6).fate, 'escaped');
    assert.equal(traceImpact(0.9, -6).fate, 'captured');
    assert.ok(asymmetry(0.9, 6) > 0);
  });

  test('a zero-momentum photon is dragged toward +phi', () => {
    // Launched radially inward in the equatorial plane, so L = 0 exactly. Any
    // transverse motion it picks up is pure frame dragging, and must follow the
    // hole's own rotation sense (+z spin => +phi).
    const origin = v3(20, 0, 0);
    const spun = nullMomentum(origin, v3(-1, 0, 0), 0.9);
    assert.ok(Math.abs(angularMomentum(origin, spun)) < 1e-12);
    assert.ok(geodesicRHS(origin, spun, 0.9).dx[1] > 0);
  });

  test('no dragging at a = 0', () => {
    const origin = v3(20, 0, 0);
    const still = nullMomentum(origin, v3(-1, 0, 0), 0);
    assert.ok(Math.abs(geodesicRHS(origin, still, 0).dx[1]) < 1e-15);
  });
});

// ---------------------------------------------------------------------------

describe('integrator', () => {
  test('reduces to straight-line motion in the flat-space limit', () => {
    // Far from the hole f ~ 2/r is negligible, so one RK4 step must translate
    // the position by h*p and leave the momentum alone.
    const x = v3(1e6, 0, 0);
    const p = nullMomentum(x, v3(0, 1, 0), 0);
    const stepped = rk4Step(x, p, 0, 0.5);
    assert.ok(Math.abs(stepped.x[0] - x[0]) < 1e-3);
    assert.ok(Math.abs(stepped.x[1] - 0.5 * p[1]) < 1e-6);
    assert.ok(length(sub(stepped.p, p)) < 1e-9);
  });

  test('residual shrinks as the finite-difference epsilon shrinks', () => {
    // A sign error in dp/dlambda would be epsilon-independent. Central-difference
    // truncation is O(eps^2) and must fall away.
    const coarse = traceRay(v3(30, 8, 0), v3(-1, 0, 0), 0.9, REFERENCE_MAX_STEPS, 1e-2);
    const fine = traceRay(v3(30, 8, 0), v3(-1, 0, 0), 0.9, REFERENCE_MAX_STEPS, 1e-4);
    assert.ok(
      fine.maxAbsH < coarse.maxAbsH,
      `max|H| did not improve: ${coarse.maxAbsH} -> ${fine.maxAbsH}`,
    );
  });

  test('residual on the plunging ray falls with the step size', () => {
    // The largest residual anywhere in the suite is on the captured ray, where
    // f is O(1) and momenta blow up. Halving the step must cut it hard — RK4
    // truncation is O(h^4) — which is what distinguishes stiffness from a bug.
    const coarse = plungeResidual(STEP_SCALE);
    const fine = plungeResidual(STEP_SCALE / 4);
    assert.ok(
      fine < coarse * 0.25,
      `4x refinement barely helped: ${coarse} -> ${fine}`,
    );
  });

  test('tracing is deterministic', () => {
    const once = traceImpact(0.9, 5.5);
    const twice = traceImpact(0.9, 5.5);
    assert.deepEqual(once, twice);
  });
});

// ---------------------------------------------------------------------------

describe('step control', () => {
  test('adaptiveStep is clamped and linear in between', () => {
    assert.equal(adaptiveStep(0), STEP_MIN);
    assert.equal(adaptiveStep(1e6), STEP_MAX);
    for (const r of [1, 5, 10, 19]) {
      assert.ok(Math.abs(adaptiveStep(r) - r * STEP_SCALE) < 1e-12);
    }
    for (const r of [0, 0.5, 3, 30, 1e6]) {
      const h = adaptiveStep(r);
      assert.ok(h >= STEP_MIN && h <= STEP_MAX, `step out of range at r=${r}`);
    }
  });

  test('adaptiveStep rescales its floor with the step scale', () => {
    // Halving the scale must halve the floor too, or a refinement sweep stalls
    // against a fixed minimum and stops converging.
    assert.ok(Math.abs(adaptiveStep(0, STEP_SCALE / 2) - STEP_MIN / 2) < 1e-12);
  });

  test('planeLimitedStep never lengthens a step', () => {
    for (const z of [-4, -0.2, 0, 0.05, 3]) {
      for (const dz of [-2, -0.01, 0, 0.5, 7]) {
        assert.ok(planeLimitedStep(0.3, z, dz) <= 0.3);
      }
    }
  });

  test('planeLimitedStep stands down when the ray is not approaching the plane', () => {
    assert.equal(planeLimitedStep(0.4, 3, 0), 0.4);
    assert.equal(planeLimitedStep(0.4, 3, 1e-9), 0.4);
  });

  test('a limited step cannot carry the ray through the equatorial plane', () => {
    // The disk test only sees one sign change per step, so the step must stop
    // short of z = 0 — or bottom out at the floor that keeps it from stalling.
    for (const z of [-5, -0.5, -0.02, 0.02, 0.5, 5]) {
      for (const dz of [-3, -0.4, 0.4, 3]) {
        const h = planeLimitedStep(0.9, z, dz);
        assert.ok(
          h * Math.abs(dz) <= Math.abs(z) * PLANE_APPROACH + 1e-12 ||
            h <= PLANE_STEP_MIN,
          `step overshot the plane at z=${z}, dz=${dz}`,
        );
      }
    }
  });
});
