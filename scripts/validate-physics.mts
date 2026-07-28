/**
 * Numerical validation of the Kerr null-geodesic port, run under Node.
 *
 * The maths lives in `src/physics/kerrReference.ts`, which is a faithful f64
 * mirror of `src/gpu/shaders/kerr_math.wgsl` and is also imported by the in-app
 * WGSL harness. This file is just the assertion suite over it.
 *
 * Its whole job is to catch sign errors in the geodesic RHS *before* they are
 * buried in a compute shader where the only symptom is a subtly wrong picture.
 * Two invariants are checked along every traced ray:
 *
 *   H = 0            null condition (Hamiltonian constraint)
 *   L = x*py - y*px  axial angular momentum (metric is axisymmetric)
 *
 * Run with: pnpm validate:physics
 */

import {
  GRAD_EPS,
  REFERENCE_MAX_STEPS,
  REFERENCE_RAYS,
  STEP_SCALE,
  angularMomentum,
  geodesicRHS,
  hamiltonian,
  horizonRadius,
  iscoRadius,
  length,
  normalize,
  nullMomentum,
  sub,
  traceRay,
  v3,
  type TraceResult,
  type Vec3,
} from '../src/physics/kerrReference.ts';

const sci = (n: number, digits = 3): string => n.toExponential(digits);
const fix = (n: number, digits = 6): string => n.toFixed(digits);

let failures = 0;

function check(label: string, ok: boolean, detail: string): void {
  if (!ok) failures++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label.padEnd(46)} ${detail}`);
}

function heading(title: string): void {
  console.log(`\n${title}\n${'-'.repeat(title.length)}`);
}

// ---------------------------------------------------------------------------
// 1. Characteristic radii
// ---------------------------------------------------------------------------

heading('1. Characteristic radii');

check(
  'r_plus(a=0) == 2',
  Math.abs(horizonRadius(0) - 2) < 1e-12,
  `got ${fix(horizonRadius(0), 12)}`,
);
check(
  'r_plus(a=0.998)',
  Math.abs(horizonRadius(0.998) - (1 + Math.sqrt(1 - 0.998 ** 2))) < 1e-12,
  `got ${fix(horizonRadius(0.998), 9)}`,
);
check(
  'r_isco(a=0) == 6  (BPT, not photon orbit)',
  Math.abs(iscoRadius(0) - 6) < 1e-9,
  `got ${fix(iscoRadius(0), 9)}`,
);
check(
  'r_isco(a=0.998) ~ 1.2367 (prograde, near-extremal)',
  Math.abs(iscoRadius(0.998) - 1.2367) < 5e-3,
  `got ${fix(iscoRadius(0.998), 6)}`,
);
check(
  'r_isco(a=0.5) ~ 4.233',
  Math.abs(iscoRadius(0.5) - 4.233) < 5e-3,
  `got ${fix(iscoRadius(0.5), 6)}`,
);
// Guard against the photon-orbit formula being substituted here (a real bug
// caught during the original validation): 2*(1+cos((2/3)*acos(-a))) is 4 at a=0,
// but evaluates to 3 with acos(-0) = pi/2 in IEEE. Either way, not 6.
const photonOrbitAtZero = 2 * (1 + Math.cos((2 / 3) * Math.acos(-0)));
check(
  'ISCO is not the photon-orbit formula',
  Math.abs(iscoRadius(0) - photonOrbitAtZero) > 1,
  `isco=6 vs photon-orbit=${fix(photonOrbitAtZero, 3)}`,
);

// ---------------------------------------------------------------------------
// 2. Null initial condition
// ---------------------------------------------------------------------------

heading('2. Null initial condition (H = 0 at ray launch)');

for (const [a, origin] of [
  [0, v3(30, 8, 0)],
  [0.9, v3(30, 8, 0)],
  [0.9, v3(12, -5, 9)],
  [0.998, v3(15, 3, -4)],
] as [number, Vec3][]) {
  const d = normalize(sub(v3(0, 0, 0), origin));
  const p = nullMomentum(origin, d, a);
  const h = hamiltonian(origin, p, a);
  check(
    `a=${a}, origin=(${origin.join(',')})`,
    Math.abs(h) < 1e-14,
    `H = ${sci(h)}, |p| = ${fix(length(p), 6)}`,
  );
}
{
  // Flat-space limit: at a=0 far from the hole, f -> 0 so |p| -> 1.
  const far = v3(1e7, 0, 0);
  const p = nullMomentum(far, v3(-1, 0, 0), 0);
  check(
    'flat-space limit |p| -> 1',
    Math.abs(length(p) - 1) < 1e-6,
    `|p| = ${fix(length(p), 12)}`,
  );
}

// ---------------------------------------------------------------------------
// 3. Conservation along traced geodesics
// ---------------------------------------------------------------------------

heading('3. Conservation along traced geodesics (eps = 0.0015, RK4)');

const traced = REFERENCE_RAYS.map((ray) => ({
  ray,
  result: traceRay(
    ray.origin,
    ray.direction,
    ray.a,
    REFERENCE_MAX_STEPS,
    GRAD_EPS,
  ),
}));

console.log(
  `  ${'ray'.padEnd(44)} ${'fate'.padEnd(9)} ${'steps'.padStart(5)} ${'r_min'.padStart(7)} ${'max|H|'.padStart(10)} ${'max dL/L'.padStart(10)}`,
);
for (const { ray, result } of traced) {
  console.log(
    `  ${ray.name.padEnd(44)} ${result.fate.padEnd(9)} ${String(result.steps).padStart(5)} ${fix(result.minRadius, 3).padStart(7)} ${sci(result.maxAbsH, 2).padStart(10)} ${sci(result.maxRelL, 2).padStart(10)}`,
  );
}

console.log();
for (const { ray, result } of traced) {
  check(
    `H ~ 0        ${ray.name}`,
    result.maxAbsH < (ray.hTol ?? 1e-5),
    `max|H| = ${sci(result.maxAbsH)}${ray.hTol ? '  (plunging, loosened tol)' : ''}`,
  );
  check(
    `L conserved  ${ray.name}`,
    result.maxRelL < 1e-5,
    `max relative drift = ${sci(result.maxRelL)}`,
  );
  if (ray.expect) {
    check(
      `fate         ${ray.name}`,
      result.fate === ray.expect,
      `expected ${ray.expect}, got ${result.fate}`,
    );
  }
}

// ---------------------------------------------------------------------------
// 4. Finite-difference epsilon sweep
// ---------------------------------------------------------------------------

heading('4. Epsilon sweep — confirms residual is FD truncation, not a sign error');

console.log(
  '  A sign error in dp/dlambda would not shrink as eps -> 0. Truncation error does (O(eps^2)).',
);
console.log(`\n  ${'eps'.padStart(10)} ${'max|H|'.padStart(12)} ${'max dL/L'.padStart(12)}`);
const sweepRay = { a: 0.9, origin: v3(30, 8, 0), direction: v3(-1, 0, 0) };
const sweep: { eps: number; h: number; l: number }[] = [];
for (const eps of [0.01, GRAD_EPS, 1e-4, 1e-5]) {
  const t = traceRay(
    sweepRay.origin,
    sweepRay.direction,
    sweepRay.a,
    REFERENCE_MAX_STEPS,
    eps,
  );
  sweep.push({ eps, h: t.maxAbsH, l: t.maxRelL });
  console.log(
    `  ${sci(eps, 0).padStart(10)} ${sci(t.maxAbsH, 3).padStart(12)} ${sci(t.maxRelL, 3).padStart(12)}`,
  );
}
check(
  '\n  max|H| shrinks as eps decreases',
  sweep[2].h < sweep[0].h,
  `eps=1e-2 -> ${sci(sweep[0].h)},  eps=1e-4 -> ${sci(sweep[2].h)}`,
);

// ---------------------------------------------------------------------------
// 4b. Step-size refinement on the plunging ray
// ---------------------------------------------------------------------------

heading('4b. Step refinement on the captured ray (a=0, b=4)');

console.log(
  '  Its max|H| ~ 3e-5 is the largest residual anywhere. If that were a sign error it would',
);
console.log(
  '  be step-size independent. RK4 truncation is O(h^4), so halving the step must cut it hard.',
);
console.log(`\n  ${'step scale'.padStart(11)} ${'steps'.padStart(6)} ${'max|H|'.padStart(12)}`);
const plunge: { scale: number; h: number }[] = [];
for (const scale of [STEP_SCALE, STEP_SCALE / 2, STEP_SCALE / 4, STEP_SCALE / 8]) {
  const t = traceRay(v3(30, 4, 0), v3(-1, 0, 0), 0, 200_000, GRAD_EPS, 0, scale);
  plunge.push({ scale, h: t.maxAbsH });
  console.log(
    `  ${fix(scale, 6).padStart(11)} ${String(t.steps).padStart(6)} ${sci(t.maxAbsH, 3).padStart(12)}`,
  );
}
const coarsest = plunge[0].h;
const finest = plunge[plunge.length - 1].h;
check(
  'plunging-ray residual falls with step size (truncation, not a sign error)',
  finest < coarsest * 0.25,
  `${sci(coarsest)} -> ${sci(finest)} over 8x refinement`,
);

// ---------------------------------------------------------------------------
// 5. Frame dragging must vanish at a = 0
// ---------------------------------------------------------------------------

heading('5. Frame dragging: prograde/retrograde asymmetry, vanishing at a=0');

console.log(
  '  Rays launched at +b and -b from x=+40 along -x have L = +b*s and -b*s, so +b co-rotates',
);
console.log(
  '  with the hole (prograde). Prograde photons have the SMALLER critical impact parameter, so',
);
console.log(
  '  at equal |b| it is the retrograde ray that is captured first — frame dragging makes a',
);
console.log('  co-rotating photon harder to swallow, not easier.\n');

const traceImpact = (a: number, impact: number): TraceResult =>
  traceRay(v3(40, impact, 0), v3(-1, 0, 0), a, REFERENCE_MAX_STEPS, GRAD_EPS);

/** Asymmetry in closest approach between +b and -b; zero iff the metric is non-rotating. */
const deflectionAsymmetry = (a: number, b: number): number =>
  traceImpact(a, b).minRadius - traceImpact(a, -b).minRadius;

console.log(
  `  ${'a'.padStart(6)} ${'r_min(+b)'.padStart(10)} ${'r_min(-b)'.padStart(10)} ${'difference'.padStart(11)}   fates (pro / retro)`,
);
for (const a of [0, 0.3, 0.6, 0.9, 0.998]) {
  const pro = traceImpact(a, 6);
  const retro = traceImpact(a, -6);
  console.log(
    `  ${fix(a, 3).padStart(6)} ${fix(pro.minRadius, 4).padStart(10)} ${fix(retro.minRadius, 4).padStart(10)} ${fix(pro.minRadius - retro.minRadius, 6).padStart(11)}   ${pro.fate} / ${retro.fate}`,
  );
}

check(
  'asymmetry vanishes exactly at a = 0',
  Math.abs(deflectionAsymmetry(0, 6)) < 1e-12,
  `got ${sci(Math.abs(deflectionAsymmetry(0, 6)))}`,
);
check(
  'asymmetry grows with spin',
  Math.abs(deflectionAsymmetry(0.9, 6)) > Math.abs(deflectionAsymmetry(0.3, 6)),
  `|asym(0.3)| = ${fix(Math.abs(deflectionAsymmetry(0.3, 6)), 4)}, |asym(0.9)| = ${fix(Math.abs(deflectionAsymmetry(0.9, 6)), 4)}`,
);
check(
  'at a=0.9, b=6: retrograde captured, prograde escapes',
  traceImpact(0.9, -6).fate === 'captured' &&
    traceImpact(0.9, 6).fate === 'escaped',
  `prograde=${traceImpact(0.9, 6).fate}, retrograde=${traceImpact(0.9, -6).fate}`,
);
check(
  'prograde stays farther out than retrograde at equal |b|',
  deflectionAsymmetry(0.9, 6) > 0,
  `r_min(+b) - r_min(-b) = ${fix(deflectionAsymmetry(0.9, 6), 6)}`,
);

{
  // Direct check of the drag direction: a photon with exactly zero angular
  // momentum, launched radially inward in the equatorial plane, must acquire
  // motion in +phi (the hole's own rotation sense) once a > 0.
  const origin = v3(20, 0, 0);
  const p = nullMomentum(origin, v3(-1, 0, 0), 0.9);
  const { dx } = geodesicRHS(origin, p, 0.9);
  const l0 = angularMomentum(origin, p);
  check(
    'zero-L photon is dragged toward +phi (spin sense is +z)',
    Math.abs(l0) < 1e-12 && dx[1] > 0,
    `L0 = ${sci(l0)}, dy/dlambda = ${sci(dx[1])}`,
  );
}

// ---------------------------------------------------------------------------
// 6. Sample trace, as in the original JAX validation
// ---------------------------------------------------------------------------

heading('6. Sample trace: a=0.9, b=5.5 (strong deflection)');

const detailed = traceRay(
  v3(30, 5.5, 0),
  v3(-1, 0, 0),
  0.9,
  REFERENCE_MAX_STEPS,
  GRAD_EPS,
  40,
);
console.log(
  `  ${'step'.padStart(5)} ${'r'.padStart(9)} ${'H'.padStart(12)} ${'L'.padStart(14)} ${'dL/L'.padStart(11)}`,
);
for (const s of detailed.samples) {
  const rel = Math.abs(s.l - detailed.l0) / Math.abs(detailed.l0);
  console.log(
    `  ${String(s.step).padStart(5)} ${fix(s.r, 4).padStart(9)} ${sci(s.h, 3).padStart(12)} ${fix(s.l, 9).padStart(14)} ${sci(rel, 2).padStart(11)}`,
  );
}
console.log(`  fate = ${detailed.fate}, L0 = ${fix(detailed.l0, 9)}`);

// ---------------------------------------------------------------------------

heading('Summary');
if (failures === 0) {
  console.log('  ALL CHECKS PASSED — geodesic port is numerically sound.');
  console.log(
    '  The GPU f32 implementation is separately checked against these same rays\n' +
      '  by the in-app harness (pnpm dev, then "Validate physics" in the panel).\n',
  );
} else {
  console.log(`  ${failures} CHECK(S) FAILED.\n`);
  process.exitCode = 1;
}
