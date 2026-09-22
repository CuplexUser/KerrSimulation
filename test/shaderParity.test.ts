/**
 * Contract tests between the WGSL and the TypeScript that mirrors it.
 *
 * Three copies of the same information exist in this project by design:
 *
 *   kerr_math.wgsl        the physics the GPU runs
 *   kerrReference.ts      the f64 mirror the validation harness compares against
 *   uniforms.ts           the byte layout the shaders read their parameters from
 *
 * Every one of those files says "if you edit one, edit the other" — and until
 * now nothing enforced it. A drifted constant or a renamed field does not fail
 * the build, the lint, or the Node physics script. It fails at shader
 * compilation in a browser with a GPU, or worse, it does not fail at all: a
 * uniform slot swapped with its neighbour just renders a subtly wrong picture.
 *
 * These tests read the shader source off disk and check it against the
 * TypeScript, so the drift is caught headlessly in CI.
 */

import { strict as assert } from 'node:assert';
import { describe, test } from 'node:test';

import { traceShader, validateShader } from '../src/gpu/shaders/index.ts';
import { UNIFORM_BYTES, UNIFORM_FLOATS } from '../src/gpu/uniforms.ts';
import {
  ESCAPE_RADIUS,
  FATE_ORDER,
  HORIZON_PAD,
  PLANE_APPROACH,
  PLANE_STEP_MIN,
  STEP_MAX,
  STEP_MIN,
  STEP_SCALE,
} from '../src/physics/kerrReference.ts';
import {
  functionNames,
  identifiers,
  readShader,
  scalarConstants,
  structFields,
  structNames,
} from './support/wgsl.ts';

const kerrMath = readShader('kerr_math');
const trace = readShader('trace');
const present = readShader('present');
const bloom = readShader('bloom');
const validate = readShader('validate');

// ---------------------------------------------------------------------------

describe('kerr_math.wgsl constants mirror kerrReference.ts', () => {
  /** Every KERR_-prefixed shader constant and the export it must equal. */
  const MIRRORED: Record<string, number> = {
    KERR_STEP_SCALE: STEP_SCALE,
    KERR_STEP_MIN: STEP_MIN,
    KERR_STEP_MAX: STEP_MAX,
    KERR_ESCAPE_RADIUS: ESCAPE_RADIUS,
    KERR_HORIZON_PAD: HORIZON_PAD,
    KERR_PLANE_APPROACH: PLANE_APPROACH,
    KERR_PLANE_STEP_MIN: PLANE_STEP_MIN,
  };

  const declared = scalarConstants(kerrMath);

  for (const [wgslName, referenceValue] of Object.entries(MIRRORED)) {
    test(`${wgslName} == ${referenceValue}`, () => {
      assert.ok(declared.has(wgslName), `${wgslName} is not declared in kerr_math.wgsl`);
      assert.equal(
        declared.get(wgslName),
        referenceValue,
        `${wgslName} has drifted from its kerrReference.ts counterpart`,
      );
    });
  }

  test('no shader constant is missing a TypeScript mirror', () => {
    // Catches the other direction: a new tuning constant added to the shader
    // that the f64 reference never learns about, which would silently make the
    // two integrators different algorithms.
    const unmirrored = [...declared.keys()]
      .filter((name) => name.startsWith('KERR_'))
      .filter((name) => !(name in MIRRORED));
    assert.deepEqual(unmirrored, []);
  });
});

// ---------------------------------------------------------------------------

describe('kerr_math.wgsl is the single source of physics for its consumers', () => {
  /**
   * The API trace.wgsl and validate.wgsl reach across for. Neither declares
   * these itself — they are supplied by concatenation — so a rename in
   * kerr_math.wgsl breaks the shipped shader at pipeline-creation time.
   *
   * Boundary names only. `MetricFK`, `Deriv` and `kerrFK` are deliberately
   * absent: consumers never name them, so they are free to change.
   */
  const SHARED_API = [
    'State',
    'kerrRadius',
    'hamiltonian',
    'angularMomentum',
    'geodesicRHS',
    'rk4Step',
    'rk4StepFrom',
    'adaptiveStep',
    'planeLimitedStep',
    'horizonRadius',
    'iscoRadius',
    'nullMomentum',
  ];

  const defined = new Set([...functionNames(kerrMath), ...structNames(kerrMath)]);
  const consumers = { 'trace.wgsl': trace, 'validate.wgsl': validate };

  for (const name of SHARED_API) {
    test(`kerr_math.wgsl defines ${name}`, () => {
      assert.ok(defined.has(name), `${name} is no longer declared in kerr_math.wgsl`);
    });
  }

  test('every shared name is actually used by a consumer', () => {
    // Keeps the list above honest. A name that no consumer references is either
    // dead physics or a stale entry here.
    const used = new Set([...identifiers(trace), ...identifiers(validate)]);
    assert.deepEqual(
      SHARED_API.filter((name) => !used.has(name)),
      [],
    );
  });

  for (const [label, source] of Object.entries(consumers)) {
    test(`${label} redeclares nothing it inherits`, () => {
      const own = new Set([...functionNames(source), ...structNames(source)]);
      const shadowed = SHARED_API.filter((name) => own.has(name));
      assert.deepEqual(shadowed, [], `${label} shadows the shared physics`);
    });

    test(`${label} references only declared KERR_ constants`, () => {
      const declared = scalarConstants(kerrMath);
      const undeclared = [...identifiers(source)]
        .filter((token) => token.startsWith('KERR_'))
        .filter((token) => !declared.has(token));
      assert.deepEqual(undeclared, [], `${label} uses a constant that no longer exists`);
    });
  }

  test('the composed shaders really do prepend the physics', () => {
    for (const [label, composed] of Object.entries({
      traceShader,
      validateShader,
    })) {
      assert.ok(composed.startsWith(kerrMath), `${label} does not lead with kerr_math.wgsl`);
      assert.equal(
        composed.indexOf('fn kerrRadius'),
        composed.lastIndexOf('fn kerrRadius'),
        `${label} contains two copies of the physics`,
      );
    }
  });
});

// ---------------------------------------------------------------------------

describe('uniform buffer layout', () => {
  /**
   * The slot order packUniforms writes, as documented above it. Position in
   * this list is the vec4 index, so an insertion anywhere shifts every field
   * after it — which is exactly the change that renders a plausible-looking but
   * wrong image rather than failing outright.
   */
  const FIELDS = [
    'camPos',
    'camRight',
    'camUp',
    'camFwd',
    'params',
    'frame',
    'options',
    'band',
    'bloom',
    'disk',
    'view',
    'sky',
  ];

  test('trace.wgsl declares the expected fields in the expected order', () => {
    const fields = structFields(trace, 'Uniforms');
    assert.ok(fields, 'trace.wgsl no longer declares a Uniforms struct');
    assert.deepEqual(fields.map((f) => f.name), FIELDS);
  });

  test('present.wgsl declares the identical struct', () => {
    // Both pipelines bind the same buffer, so a field added to one and not the
    // other misreads every slot past the insertion point.
    const traceFields = structFields(trace, 'Uniforms');
    const presentFields = structFields(present, 'Uniforms');
    assert.ok(presentFields, 'present.wgsl no longer declares a Uniforms struct');
    assert.deepEqual(presentFields, traceFields);
  });

  test('bloom.wgsl declares the identical struct', () => {
    const traceFields = structFields(trace, 'Uniforms');
    const bloomFields = structFields(bloom, 'Uniforms');
    assert.ok(bloomFields, 'bloom.wgsl no longer declares a Uniforms struct');
    assert.deepEqual(bloomFields, traceFields);
  });

  test('every field is a vec4f', () => {
    // The padding contract: nothing may be a bare scalar or vec3, or WGSL's
    // uniform-address-space alignment rules silently insert gaps the packer
    // does not account for.
    for (const field of structFields(trace, 'Uniforms') ?? []) {
      assert.equal(field.type, 'vec4f', `${field.name} is not padded to vec4`);
    }
  });

  test('UNIFORM_FLOATS covers the struct exactly', () => {
    const fields = structFields(trace, 'Uniforms') ?? [];
    assert.equal(UNIFORM_FLOATS, fields.length * 4);
    assert.equal(UNIFORM_BYTES, UNIFORM_FLOATS * 4);
    // WebGPU requires uniform buffer bindings to be 16-byte aligned.
    assert.equal(UNIFORM_BYTES % 16, 0);
  });
});

// ---------------------------------------------------------------------------

describe('validation harness layout', () => {
  test('RayResult is the 8 floats the readback stride assumes', () => {
    const fields = structFields(validate, 'RayResult') ?? [];
    assert.equal(fields.length, 8);
    for (const field of fields) assert.equal(field.type, 'f32');
    assert.deepEqual(fields.map((f) => f.name), [
      'maxAbsH',
      'maxRelL',
      'minRadius',
      'steps',
      'fate',
      'l0',
      'h0',
      'isco',
    ]);
  });

  test('RaySpec packs into two vec4s', () => {
    const fields = structFields(validate, 'RaySpec') ?? [];
    assert.deepEqual(fields, [
      { name: 'origin', type: 'vec4f' },
      { name: 'direction', type: 'vec4f' },
    ]);
  });

  test('FATE_ORDER decodes the shader fate encoding', () => {
    // validate.wgsl writes 0 for the step cap, 1 for capture, 2 for escape, and
    // the host indexes FATE_ORDER with that float. Reordering the array
    // relabels every result without any error.
    assert.deepEqual(FATE_ORDER, ['maxSteps', 'captured', 'escaped']);
    assert.match(validate, /r < rPlus \* KERR_HORIZON_PAD\)?\s*\{\s*fate = 1\.0;/);
    assert.match(validate, /r > KERR_ESCAPE_RADIUS\)?\s*\{\s*fate = 2\.0;/);
  });
});
