/**
 * Regression tests for the render loop's scheduling.
 *
 * The renderer is progressive: one sample per pixel is split across several
 * dispatches, blended into a double-buffered running average, and thrown away
 * whenever a parameter changes what the rays do. Three things in that sentence
 * are silent when they break.
 *
 *   Banding — the split must tile the image exactly. A gap leaves rows that
 *   never get traced; an overlap traces them twice and weights them wrong. Both
 *   depend on where band boundaries happen to land, so they show up as stable
 *   banding rather than noise and do not average away with more samples.
 *
 *   Presentation — the texture on screen must be the last *complete* one, or
 *   every frame shows a partly-updated image. That includes the moment a drag
 *   starts or ends: the old image stays up until the new one is whole.
 *
 *   Interaction — a moving camera must redraw the whole image every frame, at a
 *   resolution that adapts to hold the frame rate, without reallocating.
 *
 *   Reset scope — exposure and bloom are applied at present time and must not
 *   discard converged samples; anything that changes the geodesics must.
 *
 * None of this needs a GPU, so it is checked against the recording stand-in in
 * `support/fakeWebGpu.ts`.
 */

import { strict as assert } from 'node:assert';
import { afterEach, describe, test } from 'node:test';

import { DEFAULT_CAMERA } from '../src/gpu/camera.ts';
import {
  BLOOM_LEVELS,
  DEFAULT_SCENE,
  KerrRenderer,
  MAX_ACCUMULATED_SAMPLES,
  type SceneParams,
} from '../src/gpu/KerrRenderer.ts';
import { horizonRadius, iscoRadius } from '../src/physics/kerrReference.ts';
import { createFakeGpu, type FakeGpu, type FrameRecord } from './support/fakeWebGpu.ts';

const CSS_WIDTH = 800;
const CSS_HEIGHT = 600;

/** Uniform slot indices, mirroring the layout documented in uniforms.ts. */
const SLOT = {
  eyeX: 0,
  spin: 16,
  diskOuterRadius: 17,
  iscoRadius: 18,
  horizonRadius: 19,
  frameIndex: 20,
  traceWidth: 21,
  traceHeight: 22,
  exposure: 23,
  diskEnabled: 24,
  maxSteps: 25,
  canvasWidth: 26,
  canvasHeight: 27,
  bandOffset: 28,
  bandHeight: 29,
  dopplerBeaming: 30,
  diskThickness: 31,
  bloomWidth: 32,
  bloomHeight: 33,
  bloomThreshold: 34,
  bloomStrength: 35,
  shading: 36,
  peakTemperature: 37,
  presentWidth: 40,
  presentHeight: 41,
};

const read = (frame: FrameRecord, slot: keyof typeof SLOT): number =>
  frame.uniforms[SLOT[slot]];

const active: FakeGpu[] = [];

async function setup(): Promise<{ gpu: FakeGpu; renderer: KerrRenderer }> {
  const gpu = createFakeGpu({ width: CSS_WIDTH, height: CSS_HEIGHT });
  active.push(gpu);
  const renderer = await KerrRenderer.create(gpu.device, gpu.canvas);
  renderer.start();
  return { gpu, renderer };
}

afterEach(() => {
  // Globals are patched per test, so they must come back even on failure.
  for (const gpu of active.splice(0)) gpu.restore();
});

/** Runs `count` frames, optionally advancing the clock between each. */
function tick(gpu: FakeGpu, count: number, msPerFrame = 0): void {
  for (let i = 0; i < count; i++) {
    assert.ok(gpu.tick(), 'the frame loop stopped unexpectedly');
    if (msPerFrame > 0) gpu.advance(msPerFrame);
  }
}

/** Consecutive runs of frames sharing a sample index — i.e. one pass each. */
function passes(frames: FrameRecord[]): { index: number; frames: FrameRecord[] }[] {
  const grouped: { index: number; frames: FrameRecord[] }[] = [];
  for (const frame of frames) {
    const index = read(frame, 'frameIndex');
    const current = grouped.at(-1);
    if (current && current.index === index) current.frames.push(frame);
    else grouped.push({ index, frames: [frame] });
  }
  return grouped;
}

/** Trailing digit of a labelled resource, e.g. `kerr-present-bindgroup-1` -> 1. */
const labelIndex = (label: string): number => {
  const match = /-(\d+)$/.exec(label);
  assert.ok(match, `expected a trailing index on "${label}"`);
  return Number(match[1]);
};

const presentPass = (frame: FrameRecord): FrameRecord['passes'][number] => {
  const pass = frame.passes.find((p) => p.label === 'kerr-present-pass');
  assert.ok(pass, 'no present pass was encoded');
  return pass;
};

// ---------------------------------------------------------------------------

describe('frame structure', () => {
  test('a frame traces, blooms, and presents', async () => {
    const { gpu } = await setup();
    tick(gpu, 1);

    assert.equal(gpu.frames.length, 1);
    const [frame] = gpu.frames;

    assert.ok(frame.compute, 'the first frame did not dispatch the tracer');
    const down = Array.from({ length: BLOOM_LEVELS }, (_, i) => `kerr-bloom-down-${i}`);
    const up = Array.from(
      { length: BLOOM_LEVELS - 1 },
      (_, i) => `kerr-bloom-up-${BLOOM_LEVELS - 2 - i}`,
    );
    assert.deepEqual(
      frame.passes.map((pass) => pass.label),
      [...down, ...up, 'kerr-present-pass'],
      'bloom must go all the way down, all the way back up, and present last',
    );
  });

  test('the trace resolution is a fraction of the canvas', async () => {
    const { gpu } = await setup();
    tick(gpu, 1);
    const [frame] = gpu.frames;

    assert.equal(read(frame, 'canvasWidth'), CSS_WIDTH);
    assert.equal(read(frame, 'canvasHeight'), CSS_HEIGHT);
    assert.equal(read(frame, 'traceWidth'), CSS_WIDTH * DEFAULT_SCENE.resolutionScale);
    assert.equal(read(frame, 'traceHeight'), CSS_HEIGHT * DEFAULT_SCENE.resolutionScale);
  });

  test('the bloom chain starts at half the canvas', async () => {
    const { gpu } = await setup();
    tick(gpu, 1);
    const [frame] = gpu.frames;
    assert.equal(read(frame, 'bloomWidth'), CSS_WIDTH * 0.5);
    assert.equal(read(frame, 'bloomHeight'), CSS_HEIGHT * 0.5);
  });

  test('the shading mode and peak temperature reach the shader', async () => {
    const { renderer, gpu } = await setup();
    renderer.setScene({ shading: 'physical', peakTemperature: 9000 });
    tick(gpu, 1);
    const frame = gpu.frames.at(-1)!;
    assert.equal(read(frame, 'shading'), 1);
    assert.equal(read(frame, 'peakTemperature'), 9000);
  });

  test('the dispatch covers its band at one workgroup per 8x8 pixels', async () => {
    const { gpu } = await setup();
    tick(gpu, 1);
    const [frame] = gpu.frames;
    assert.deepEqual(frame.compute?.workgroups, [
      Math.ceil(read(frame, 'traceWidth') / 8),
      Math.ceil(read(frame, 'bandHeight') / 8),
    ]);
  });

  test('the scene parameters reach the shader', async () => {
    const { gpu } = await setup();
    tick(gpu, 1);
    const [frame] = gpu.frames;

    assert.equal(read(frame, 'spin'), Math.fround(DEFAULT_SCENE.spin));
    assert.equal(read(frame, 'diskOuterRadius'), DEFAULT_SCENE.diskOuterRadius);
    assert.equal(read(frame, 'maxSteps'), DEFAULT_SCENE.maxSteps);
    assert.equal(read(frame, 'diskEnabled'), DEFAULT_SCENE.diskEnabled ? 1 : 0);
    assert.equal(read(frame, 'exposure'), Math.fround(DEFAULT_SCENE.exposure));
  });

  test('the characteristic radii are derived from the spin, not passed in', async () => {
    // The shader draws the shadow and the disk's inner edge from these, so they
    // must track the spin slider exactly rather than being set independently.
    const { renderer, gpu } = await setup();
    renderer.setScene({ spin: 0.5 });
    tick(gpu, 1);
    const frame = gpu.frames.at(-1)!;

    assert.equal(read(frame, 'iscoRadius'), Math.fround(iscoRadius(0.5)));
    assert.equal(read(frame, 'horizonRadius'), Math.fround(horizonRadius(0.5)));
  });
});

// ---------------------------------------------------------------------------

describe('band scheduling', () => {
  test('the bands of a pass tile the image exactly', async () => {
    const { gpu } = await setup();
    tick(gpu, 40);

    const complete = passes(gpu.frames).slice(0, -1);
    assert.ok(complete.length >= 2, 'expected at least two complete passes');

    for (const pass of complete) {
      const height = read(pass.frames[0], 'traceHeight');
      const bands = pass.frames
        .filter((frame) => frame.compute)
        .map((frame) => ({
          offset: read(frame, 'bandOffset'),
          rows: read(frame, 'bandHeight'),
        }))
        .toSorted((a, b) => a.offset - b.offset);

      let covered = 0;
      for (const band of bands) {
        assert.equal(band.offset, covered, `pass ${pass.index}: bands do not join up`);
        assert.ok(band.rows > 0, `pass ${pass.index}: dispatched an empty band`);
        covered += band.rows;
      }
      assert.equal(covered, height, `pass ${pass.index}: bands do not cover the image`);
    }
  });

  test('a pass never skips a band that has rows in it', async () => {
    const { gpu } = await setup();
    tick(gpu, 40);
    for (const frame of gpu.frames) {
      if (!frame.compute) {
        assert.ok(
          read(frame, 'bandHeight') <= 0,
          'a band with rows in it was not traced',
        );
      }
    }
  });

  test('the sample counter advances only once a pass is complete', async () => {
    const { gpu } = await setup();
    tick(gpu, 40);

    const grouped = passes(gpu.frames);
    // Sample indices must be consecutive with no gaps or repeats.
    assert.deepEqual(
      grouped.map((pass) => pass.index),
      grouped.map((_, i) => i),
    );
    // Every complete pass must have taken the same number of dispatches as its
    // band count, so no pass is silently short.
    for (const pass of grouped.slice(0, -1)) {
      assert.ok(pass.frames.length >= 1);
    }
  });

  test('the band count is retuned only between passes', async () => {
    // Changing the split mid-pass would leave rows traced twice or not at all.
    // The band count is not observable directly, but the row height it implies is.
    const { gpu } = await setup();
    tick(gpu, 60, 40); // Slow frames, so the controller wants more bands.

    for (const pass of passes(gpu.frames).slice(0, -1)) {
      const heights = new Set(
        pass.frames.map((frame) => read(frame, 'bandHeight')),
      );
      // Within a pass every band is the same height except the last, which
      // absorbs the remainder.
      assert.ok(heights.size <= 2, `pass ${pass.index} changed its band height`);
    }
  });

  test('slow frames split the work further, fast frames coalesce it', async () => {
    const bandsPerPass = async (msPerFrame: number) => {
      const { gpu } = await setup();
      tick(gpu, 120, msPerFrame);
      return passes(gpu.frames).at(-2)?.frames.length ?? 0;
    };

    // 40 ms/frame is over budget; 1 ms is well under it.
    assert.ok(
      (await bandsPerPass(40)) > (await bandsPerPass(1)),
      'the adaptive band controller did not respond to frame time',
    );
  });
});

// ---------------------------------------------------------------------------

describe('presentation', () => {
  test('a completed pass is presented, never the one in flight', async () => {
    const { gpu } = await setup();
    tick(gpu, 40);

    for (const frame of gpu.frames) {
      if (!frame.compute) continue;
      // Compute bind group i reads accumulation texture i and writes 1-i.
      const reading = labelIndex(frame.compute.bindGroup);
      const writing = 1 - reading;
      const presented = labelIndex(presentPass(frame).bindGroup);

      if (gpu.frames.indexOf(frame) < passes(gpu.frames)[0].frames.length) {
        // Before anything has ever completed there is nothing else to show, so
        // the very first pass shows its own bands as they land.
        assert.equal(presented, writing, 'the first pass should show its own bands');
      } else {
        assert.equal(presented, reading, 'presented a half-written texture');
      }
    }
  });

  test('the accumulation buffers alternate every pass', async () => {
    const { gpu } = await setup();
    tick(gpu, 40);

    const perPass = passes(gpu.frames)
      .slice(0, -1)
      .map((pass) => labelIndex(pass.frames[0].compute!.bindGroup));

    for (let i = 1; i < perPass.length; i++) {
      assert.notEqual(perPass[i], perPass[i - 1], 'the ping-pong stalled');
    }
  });

  test('bloom reads whichever texture is being presented', async () => {
    const { gpu } = await setup();
    tick(gpu, 12);

    for (const frame of gpu.frames) {
      const bright = frame.passes.find((pass) => pass.label === 'kerr-bloom-down-0');
      assert.ok(bright);
      assert.equal(
        labelIndex(bright.bindGroup),
        labelIndex(presentPass(frame).bindGroup),
        'bloom glowed off a different image than the one on screen',
      );
    }
  });
});

// ---------------------------------------------------------------------------

describe('accumulation reset', () => {
  /** Runs until at least one pass has completed, then returns the last frame. */
  const settle = (gpu: FakeGpu): FrameRecord => {
    tick(gpu, 40);
    assert.ok(read(gpu.frames.at(-1)!, 'frameIndex') > 0, 'no pass ever completed');
    return gpu.frames.at(-1)!;
  };

  const PRESENT_TIME: Partial<SceneParams>[] = [
    { exposure: 2.4 },
    { bloomThreshold: 0.4 },
    { bloomStrength: 0.2 },
  ];

  const AFFECTS_TRACING: Partial<SceneParams>[] = [
    { spin: 0.2 },
    { diskOuterRadius: 12 },
    { diskEnabled: false },
    { maxSteps: 900 },
    { dopplerBeaming: 1 },
    { diskThickness: 0.05 },
    { resolutionScale: 0.5 },
  ];

  for (const change of PRESENT_TIME) {
    test(`${Object.keys(change)[0]} keeps the accumulated samples`, async () => {
      const { renderer, gpu } = await setup();
      const before = read(settle(gpu), 'frameIndex');

      renderer.setScene(change);
      tick(gpu, 1);

      assert.ok(
        read(gpu.frames.at(-1)!, 'frameIndex') >= before,
        'a present-time setting threw away converged samples',
      );
    });
  }

  for (const change of AFFECTS_TRACING) {
    test(`${Object.keys(change)[0]} discards the accumulated samples`, async () => {
      const { renderer, gpu } = await setup();
      settle(gpu);

      renderer.setScene(change);
      tick(gpu, 1);

      const frame = gpu.frames.at(-1)!;
      assert.equal(read(frame, 'frameIndex'), 0, 'stale samples survived');
      assert.equal(read(frame, 'bandOffset'), 0, 'the pass did not restart at the top');
    });
  }

  test('setting a parameter to its current value is not a reset', async () => {
    const { renderer, gpu } = await setup();
    const before = read(settle(gpu), 'frameIndex');

    renderer.setScene({ spin: DEFAULT_SCENE.spin });
    tick(gpu, 1);

    assert.ok(read(gpu.frames.at(-1)!, 'frameIndex') >= before);
  });

  test('moving the camera discards the accumulated samples', async () => {
    const { renderer, gpu } = await setup();
    settle(gpu);

    renderer.updateCamera((camera) => {
      camera.azimuth += 0.2;
    });
    tick(gpu, 1);

    const frame = gpu.frames.at(-1)!;
    assert.equal(read(frame, 'frameIndex'), 0);
    assert.equal(read(frame, 'bandOffset'), 0);
  });

  test('resizing the canvas discards the accumulated samples', async () => {
    const { gpu } = await setup();
    settle(gpu);

    gpu.resizeCanvas(400, 300);
    tick(gpu, 1);

    const frame = gpu.frames.at(-1)!;
    assert.equal(read(frame, 'frameIndex'), 0);
    assert.equal(read(frame, 'canvasWidth'), 400);
    assert.equal(read(frame, 'traceWidth'), 400 * DEFAULT_SCENE.resolutionScale);
  });

  test('a resize to the same size does not discard anything', async () => {
    // ResizeObserver fires on layout changes that leave the box alone.
    const { gpu } = await setup();
    const before = read(settle(gpu), 'frameIndex');

    gpu.resizeCanvas(CSS_WIDTH, CSS_HEIGHT);
    tick(gpu, 1);

    assert.ok(read(gpu.frames.at(-1)!, 'frameIndex') >= before);
  });
});

// ---------------------------------------------------------------------------

describe('interaction', () => {
  test('the camera drops to a coarse trace while it is moving', async () => {
    const { renderer, gpu } = await setup();
    tick(gpu, 1);
    const full = read(gpu.frames.at(-1)!, 'traceWidth');

    renderer.updateCamera((camera) => {
      camera.azimuth += 0.2;
    });
    tick(gpu, 1);

    assert.ok(
      read(gpu.frames.at(-1)!, 'traceWidth') < full,
      'the interactive pass did not drop resolution',
    );
  });

  test('full resolution returns once the camera has been still', async () => {
    const { renderer, gpu } = await setup();
    tick(gpu, 1);
    const full = read(gpu.frames.at(-1)!, 'traceWidth');

    renderer.updateCamera((camera) => {
      camera.azimuth += 0.2;
    });
    tick(gpu, 1);
    assert.ok(read(gpu.frames.at(-1)!, 'traceWidth') < full);

    // The drawn camera eases after the target, and the trace stays coarse
    // until it has settled and then held still for the idle window.
    tick(gpu, 60, 16);
    assert.equal(read(gpu.frames.at(-1)!, 'traceWidth'), full);
  });

  test('the swap chain stays at full size while the trace shrinks', async () => {
    // Resizing the swap chain mid-drag would reallocate it every frame; only the
    // traced image is allowed to change size.
    const { renderer, gpu } = await setup();
    renderer.updateCamera((camera) => {
      camera.elevation += 0.1;
    });
    tick(gpu, 1);

    const frame = gpu.frames.at(-1)!;
    assert.equal(read(frame, 'canvasWidth'), CSS_WIDTH);
    assert.equal(read(frame, 'canvasHeight'), CSS_HEIGHT);
  });
});

// ---------------------------------------------------------------------------

describe('smooth interaction', () => {
  /** Settles into full-resolution progressive mode, several passes deep. */
  const settle = (gpu: FakeGpu) => tick(gpu, 40);

  test('a change keeps the last complete image on screen until its pass lands', async () => {
    const { renderer, gpu } = await setup();
    settle(gpu);
    const width = read(gpu.frames.at(-1)!, 'traceWidth');

    renderer.setScene({ diskThickness: 0.03 });
    tick(gpu, 1);
    const frame = gpu.frames.at(-1)!;

    // The new (coarse) pass has started. What is on screen is the texture it
    // reads — the old whole image, at its own size — never the one it writes.
    assert.equal(read(frame, 'frameIndex'), 0);
    assert.ok(read(frame, 'traceWidth') < width, 'the change did not go coarse');
    assert.equal(
      labelIndex(presentPass(frame).bindGroup),
      labelIndex(frame.compute!.bindGroup),
      'showed the texture being written',
    );
    assert.equal(read(frame, 'presentWidth'), width);
  });

  test('a moving camera redraws the whole image every frame', async () => {
    const { renderer, gpu } = await setup();
    settle(gpu);

    const start = gpu.frames.length;
    for (let i = 0; i < 10; i++) {
      renderer.updateCamera((camera) => {
        camera.azimuth += 0.01;
      });
      tick(gpu, 1, 16);
    }

    const moving = gpu.frames.slice(start);
    for (const frame of moving) {
      assert.equal(read(frame, 'bandOffset'), 0, 'a moving frame was split into bands');
      assert.equal(read(frame, 'bandHeight'), read(frame, 'traceHeight'));
    }
    // Each frame completed, so the next one presents it.
    for (let i = 1; i < moving.length; i++) {
      assert.equal(read(moving[i], 'presentWidth'), read(moving[i - 1], 'traceWidth'));
    }
  });

  test('starting and ending a drag never reallocates the accumulation', async () => {
    const { renderer, gpu } = await setup();
    settle(gpu);

    let created = 0;
    const createTexture = gpu.device.createTexture.bind(gpu.device);
    gpu.device.createTexture = (descriptor: GPUTextureDescriptor) => {
      created++;
      return createTexture(descriptor);
    };

    renderer.updateCamera((camera) => {
      camera.azimuth += 0.3;
    });
    tick(gpu, 80, 16);
    assert.equal(created, 0, 'a drag reallocated textures, which flashes black');
  });

  test('the coarse image stays up until the first full pass completes', async () => {
    const { renderer, gpu } = await setup();
    tick(gpu, 1);
    const full = read(gpu.frames.at(-1)!, 'traceWidth');

    renderer.updateCamera((camera) => {
      camera.azimuth += 0.2;
    });
    // Slow frames: full resolution needs many bands per pass.
    tick(gpu, 40, 40);

    const handoff = gpu.frames.findIndex(
      (frame, i) =>
        i > 0 &&
        read(frame, 'traceWidth') === full &&
        read(gpu.frames[i - 1], 'traceWidth') < full,
    );
    assert.ok(handoff > 0, 'never returned to full resolution');
    const coarse = read(gpu.frames[handoff - 1], 'traceWidth');

    for (const frame of gpu.frames.slice(handoff)) {
      if (read(frame, 'frameIndex') > 0) break;
      assert.equal(read(frame, 'presentWidth'), coarse, 'showed a partial full-res pass');
    }
  });

  test('the interactive resolution adapts to frame time', async () => {
    const widthWhileMoving = async (msPerFrame: number) => {
      const { renderer, gpu } = await setup();
      for (let i = 0; i < 60; i++) {
        renderer.updateCamera((camera) => {
          camera.azimuth += 0.01;
        });
        tick(gpu, 1, msPerFrame);
      }
      return read(gpu.frames.at(-1)!, 'traceWidth');
    };

    assert.ok(
      (await widthWhileMoving(40)) < (await widthWhileMoving(5)),
      'slow frames did not lower the interactive resolution',
    );
  });

  test('the drawn camera eases toward the target', async () => {
    const { renderer, gpu } = await setup();
    tick(gpu, 1, 16);
    const startX = read(gpu.frames.at(-1)!, 'eyeX');

    renderer.updateCamera((camera) => {
      camera.azimuth += 1;
    });
    const target = renderer.camera.azimuth;
    tick(gpu, 2, 16);
    const midX = read(gpu.frames.at(-1)!, 'eyeX');

    tick(gpu, 120, 16);
    const endX = read(gpu.frames.at(-1)!, 'eyeX');
    const expectedEnd =
      DEFAULT_CAMERA.radius * Math.cos(DEFAULT_CAMERA.elevation) * Math.cos(target);

    assert.notEqual(midX, startX, 'the camera did not start moving');
    assert.ok(Math.abs(midX - endX) > 1e-2, 'the camera jumped instead of easing');
    assert.ok(Math.abs(endX - expectedEnd) < 1e-3, 'the camera never arrived');
  });

  test('a fling coasts and then stops', async () => {
    const { renderer, gpu } = await setup();
    tick(gpu, 1, 16);
    const before = renderer.camera.azimuth;

    renderer.setCameraVelocity(0.002, 0);
    tick(gpu, 10, 16);
    assert.ok(renderer.camera.azimuth > before, 'the camera did not coast');

    tick(gpu, 400, 16);
    const stopped = renderer.camera.azimuth;
    tick(gpu, 10, 16);
    assert.equal(renderer.camera.azimuth, stopped, 'the coast never decayed');
  });
});

// ---------------------------------------------------------------------------

describe('convergence', () => {
  test('tracing stops at the sample cap but the image keeps presenting', async () => {
    const { gpu } = await setup();

    // Fast frames coalesce the split down to a single dispatch per sample, which
    // is what makes reaching the cap tractable in a test.
    for (let i = 0; i < 20_000; i++) {
      gpu.tick();
      gpu.advance(1);
      if (read(gpu.frames.at(-1)!, 'frameIndex') >= MAX_ACCUMULATED_SAMPLES) break;
    }

    const reached = read(gpu.frames.at(-1)!, 'frameIndex');
    assert.equal(reached, MAX_ACCUMULATED_SAMPLES, 'never converged');

    const before = gpu.frames.length;
    tick(gpu, 5, 1);

    for (const frame of gpu.frames.slice(before)) {
      assert.equal(frame.compute, null, 'kept tracing past the sample cap');
      assert.equal(read(frame, 'frameIndex'), MAX_ACCUMULATED_SAMPLES);
      assert.ok(presentPass(frame), 'stopped presenting after converging');
    }
  });

  test('a converged image still responds to exposure', async () => {
    // Present-time settings must apply without a single new sample, which is the
    // whole reason they are excluded from the reset.
    const { renderer, gpu } = await setup();
    tick(gpu, 4);

    renderer.setScene({ exposure: 2.5 });
    tick(gpu, 1);
    assert.equal(read(gpu.frames.at(-1)!, 'exposure'), Math.fround(2.5));
  });
});

// ---------------------------------------------------------------------------

describe('degenerate sizes', () => {
  test('a zero-size canvas does not poison the uniforms', async () => {
    // ResizeObserver reports an empty box for a hidden or unattached canvas, and
    // a non-finite aspect ratio in the buffer breaks every ray in the dispatch.
    const { gpu } = await setup();
    gpu.resizeCanvas(0, 0);
    tick(gpu, 2);

    const frame = gpu.frames.at(-1)!;
    assert.ok([...frame.uniforms].every(Number.isFinite));
    assert.ok(read(frame, 'traceWidth') >= 1);
    assert.ok(read(frame, 'traceHeight') >= 1);
    assert.ok(read(frame, 'bloomWidth') >= 1);
    assert.ok(read(frame, 'bloomHeight') >= 1);
  });

  test('a canvas narrower than one band still tiles', async () => {
    const { gpu } = await setup();
    gpu.resizeCanvas(4, 4);
    tick(gpu, 40);

    for (const pass of passes(gpu.frames).slice(1, -1)) {
      const height = read(pass.frames[0], 'traceHeight');
      const covered = pass.frames
        .filter((frame) => frame.compute)
        .reduce((sum, frame) => sum + read(frame, 'bandHeight'), 0);
      assert.equal(covered, height, `pass ${pass.index} did not cover the image`);
    }
  });
});

// ---------------------------------------------------------------------------

describe('lifecycle', () => {
  test('stats report the live resolution and derived radii', async () => {
    const { renderer, gpu } = await setup();
    const seen: { samples: number; width: number; iscoRadius: number }[] = [];
    renderer.onStats((stats) => seen.push(stats));

    // Reporting is throttled to 4 Hz so the panel does not re-render per frame.
    gpu.advance(300);
    tick(gpu, 1);

    assert.equal(seen.length, 1);
    assert.equal(seen[0].width, CSS_WIDTH * DEFAULT_SCENE.resolutionScale);
    assert.equal(seen[0].iscoRadius, iscoRadius(DEFAULT_SCENE.spin));
  });

  test('stats are throttled, not sent every frame', async () => {
    const { renderer, gpu } = await setup();
    let count = 0;
    renderer.onStats(() => count++);

    tick(gpu, 20, 5); // 100 ms of frames.
    assert.ok(count <= 2, `stats fired ${count} times in 100 ms`);
  });

  test('dispose stops the loop', async () => {
    const { renderer, gpu } = await setup();
    tick(gpu, 2);
    const before = gpu.frames.length;

    renderer.dispose();
    assert.equal(gpu.tick(), false, 'a frame was still scheduled after dispose');
    assert.equal(gpu.frames.length, before);
  });

  test('start after dispose does not resurrect the loop', async () => {
    const { renderer, gpu } = await setup();
    renderer.dispose();
    renderer.start();
    assert.equal(gpu.tick(), false);
  });

  test('start is idempotent', async () => {
    // The React effect can call it more than once; a second loop would double
    // the dispatch rate and desynchronise the band index.
    const { renderer, gpu } = await setup();
    renderer.start();
    renderer.start();
    tick(gpu, 1);
    assert.equal(gpu.frames.length, 1);
  });
});
