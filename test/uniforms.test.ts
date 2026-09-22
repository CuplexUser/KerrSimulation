/**
 * Regression tests for uniform packing.
 *
 * `shaderParity.test.ts` checks that the WGSL struct still has the fields this
 * layout assumes. This checks the other half: that each value lands in the slot
 * the shader reads it from. The two together are what make the buffer a
 * contract rather than a convention.
 *
 * A packer bug is uniquely nasty — the GPU cannot detect it, so a swapped pair
 * of slots just renders the wrong picture at full speed.
 */

import { strict as assert } from 'node:assert';
import { describe, test } from 'node:test';

import type { CameraBasis } from '../src/gpu/camera.ts';
import {
  UNIFORM_BYTES,
  UNIFORM_FLOATS,
  packUniforms,
  type RenderParams,
  type UniformInput,
} from '../src/gpu/uniforms.ts';

/** Distinct values everywhere, so a slot mix-up cannot pass by coincidence. */
const BASIS: CameraBasis = {
  eye: [11, 12, 13],
  forward: [21, 22, 23],
  right: [31, 32, 33],
  up: [41, 42, 43],
  tanHalfFov: 0.5,
  shift: [0.25, -0.125],
};

const PARAMS: RenderParams = {
  spin: 0.85,
  diskOuterRadius: 20,
  iscoRadius: 2.5,
  horizonRadius: 1.5,
  diskEnabled: true,
  maxSteps: 450,
  exposure: 1.3,
  bloomThreshold: 0.95,
  bloomStrength: 0.6,
  dopplerBeaming: 0.15,
  diskThickness: 0.02,
  shading: 'physical',
  peakTemperature: 7500,
  fluxPeak: 0.004,
  luminanceNorm: 0.03125,
};

const INPUT: UniformInput = {
  basis: BASIS,
  params: PARAMS,
  frameIndex: 7,
  width: 800,
  height: 400,
  canvasWidth: 1600,
  canvasHeight: 900,
  bandOffset: 64,
  bandHeight: 32,
  bloomWidth: 400,
  bloomHeight: 225,
  presentWidth: 600,
  presentHeight: 300,
  ditherSeed: 9,
};

const pack = (over: Partial<UniformInput> = {}): Float32Array => {
  const target = new Float32Array(UNIFORM_FLOATS);
  packUniforms(target, { ...INPUT, ...over });
  return target;
};

/** Reads one vec4 slot by index, matching how WGSL addresses the struct. */
const vec4 = (data: Float32Array, slot: number): number[] =>
  Array.from(data.slice(slot * 4, slot * 4 + 4));

// ---------------------------------------------------------------------------

describe('packUniforms slot layout', () => {
  const data = pack();

  test('slot 0 — camera position and horizontal lens shift', () => {
    assert.deepEqual(vec4(data, 0), [11, 12, 13, 0.25]);
  });

  test('slot 1 — right basis vector and tan(fov/2)', () => {
    assert.deepEqual(vec4(data, 1), [31, 32, 33, 0.5]);
  });

  test('slot 2 — up basis vector and aspect ratio', () => {
    assert.deepEqual(vec4(data, 2), [41, 42, 43, 2]);
  });

  test('slot 3 — forward basis vector and vertical lens shift', () => {
    assert.deepEqual(vec4(data, 3), [21, 22, 23, -0.125]);
  });

  test('slot 4 — spin, disk edge, ISCO, horizon', () => {
    assert.deepEqual(vec4(data, 4), [Math.fround(0.85), 20, 2.5, 1.5]);
  });

  test('slot 5 — frame index, trace resolution, exposure', () => {
    assert.deepEqual(vec4(data, 5), [7, 800, 400, Math.fround(1.3)]);
  });

  test('slot 6 — disk toggle, step cap, canvas size', () => {
    assert.deepEqual(vec4(data, 6), [1, 450, 1600, 900]);
  });

  test('slot 7 — band range, beaming, disk thickness', () => {
    assert.deepEqual(vec4(data, 7), [64, 32, Math.fround(0.15), Math.fround(0.02)]);
  });

  test('slot 8 — bloom size and tuning', () => {
    assert.deepEqual(vec4(data, 8), [400, 225, Math.fround(0.95), Math.fround(0.6)]);
  });

  test('slot 9 — shading mode, peak temperature, inverse flux peak, luminance norm', () => {
    assert.deepEqual(vec4(data, 9), [1, 7500, 250, 0.03125]);
  });

  test('slot 10 — presented region, dither seed, pixel angle', () => {
    // Pixel angle is the image-plane height over the traced rows: 2 tan(fov/2) / h.
    assert.deepEqual(vec4(data, 10), [600, 300, 9, Math.fround((2 * 0.5) / 400)]);
  });

  test('the right vector is not confused with the forward vector', () => {
    // Slots 1 and 3 are adjacent and both hold a basis vector, which makes them
    // the easiest pair in the buffer to transpose.
    assert.deepEqual(vec4(data, 1).slice(0, 3), [...BASIS.right]);
    assert.deepEqual(vec4(data, 3).slice(0, 3), [...BASIS.forward]);
  });
});

// ---------------------------------------------------------------------------

describe('packUniforms derived values', () => {
  test('aspect ratio comes from the traced resolution, not the canvas', () => {
    // The trace resolution is a fraction of the canvas and changes mid-drag;
    // taking the aspect from the canvas instead would stretch the coarse pass.
    const data = pack({ width: 300, height: 200, canvasWidth: 1600, canvasHeight: 900 });
    assert.equal(data[11], 1.5);
  });

  test('a zero-height target does not produce a non-finite aspect', () => {
    // ResizeObserver reports a zero-height box for a hidden canvas, and Infinity
    // in a uniform poisons every ray in the dispatch.
    const data = pack({ height: 0 });
    assert.equal(data[11], 1);
    assert.ok(data.every(Number.isFinite));
  });

  test('shading is encoded as a float flag', () => {
    assert.equal(pack({ params: { ...PARAMS, shading: 'physical' } })[36], 1);
    assert.equal(pack({ params: { ...PARAMS, shading: 'cinematic' } })[36], 0);
  });

  test('a missing flux peak does not produce a non-finite reciprocal', () => {
    const data = pack({ params: { ...PARAMS, fluxPeak: 0 } });
    assert.ok(data.every(Number.isFinite));
  });

  test('diskEnabled is encoded as a float flag', () => {
    assert.equal(pack({ params: { ...PARAMS, diskEnabled: true } })[24], 1);
    assert.equal(pack({ params: { ...PARAMS, diskEnabled: false } })[24], 0);
  });
});

// ---------------------------------------------------------------------------

describe('packUniforms buffer discipline', () => {
  test('writes every slot', () => {
    const target = new Float32Array(UNIFORM_FLOATS).fill(Number.NaN);
    packUniforms(target, INPUT);
    assert.ok(target.every(Number.isFinite), 'some slot was left unwritten');
  });

  test('writes nothing past the declared size', () => {
    // The scratch array the renderer reuses is exactly UNIFORM_FLOATS long, so
    // an overrun would be a silent truncation on the GPU side.
    const guarded = new Float32Array(UNIFORM_FLOATS + 8).fill(-1);
    packUniforms(guarded, INPUT);
    assert.deepEqual(
      Array.from(guarded.slice(UNIFORM_FLOATS)),
      Array.from({ length: 8 }, () => -1),
    );
  });

  test('is a pure function of its input', () => {
    assert.deepEqual(Array.from(pack()), Array.from(pack()));
  });

  test('UNIFORM_BYTES matches the float count', () => {
    assert.equal(UNIFORM_BYTES, UNIFORM_FLOATS * 4);
    assert.equal(new Float32Array(UNIFORM_FLOATS).byteLength, UNIFORM_BYTES);
  });
});
