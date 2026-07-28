// Physics validation harness.
//
// Concatenated after kerr_math.wgsl. Traces one ray per invocation using the
// exact code the renderer uses, tracking the two invariants that must hold along
// any null geodesic of an axisymmetric static metric:
//
//   H = 0            null condition
//   L = x*py - y*px  axial angular momentum
//
// The host compares these against the f64 reference values produced by
// scripts/validate-physics.mts, so this checks the shipped WGSL rather than just
// checking the shader against itself.

struct RaySpec {
  // xyz = origin, w = spin a
  origin: vec4f,
  // xyz = direction (need not be normalised), w = max steps
  direction: vec4f,
}

struct RayResult {
  maxAbsH: f32,
  maxRelL: f32,
  minRadius: f32,
  steps: f32,
  // 0 = hit step cap, 1 = captured, 2 = escaped
  fate: f32,
  l0: f32,
  h0: f32,
  // iscoRadius(a) evaluated on the GPU, cross-checked against the CPU value
  isco: f32,
}

@group(0) @binding(0) var<storage, read> specs: array<RaySpec>;
@group(0) @binding(1) var<storage, read_write> results: array<RayResult>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&specs)) {
    return;
  }

  let spec = specs[i];
  let a = spec.origin.w;
  let maxSteps = u32(spec.direction.w);

  var st: State;
  st.x = spec.origin.xyz;
  st.p = nullMomentum(st.x, normalize(spec.direction.xyz), a);

  let rPlus = horizonRadius(a);
  let l0 = angularMomentum(st.x, st.p);
  let h0 = hamiltonian(st.x, st.p, a);
  let lScale = max(abs(l0), 1e-3);

  var maxAbsH = abs(h0);
  var maxRelL = 0.0;
  var minRadius = kerrRadius(st.x, a);
  var fate = 0.0;
  var step: u32 = 0u;

  loop {
    if (step >= maxSteps) {
      break;
    }

    let r = kerrRadius(st.x, a);
    minRadius = min(minRadius, r);
    maxAbsH = max(maxAbsH, abs(hamiltonian(st.x, st.p, a)));
    maxRelL = max(maxRelL, abs(angularMomentum(st.x, st.p) - l0) / lScale);

    if (r < rPlus * KERR_HORIZON_PAD) {
      fate = 1.0;
      break;
    }
    if (r > KERR_ESCAPE_RADIUS) {
      fate = 2.0;
      break;
    }

    st = rk4Step(st, a, adaptiveStep(r));
    step = step + 1u;
  }

  var out: RayResult;
  out.maxAbsH = maxAbsH;
  out.maxRelL = maxRelL;
  out.minRadius = minRadius;
  out.steps = f32(step);
  out.fate = fate;
  out.l0 = l0;
  out.h0 = h0;
  out.isco = iscoRadius(a);
  results[i] = out;
}
