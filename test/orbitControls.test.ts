/**
 * Regression tests for pointer orbit and wheel zoom.
 *
 * Two things here are easy to get backwards and impossible to notice in a diff:
 * the sign of each drag axis (direct manipulation means the scene follows the
 * cursor, so azimuth moves *against* dx) and the wheel direction. Both are
 * pinned down below.
 *
 * The third is teardown. These listeners are attached in a React effect, and a
 * cleanup that misses one leaves a detached renderer being driven by a live
 * canvas across a StrictMode remount.
 */

import { strict as assert } from 'node:assert';
import { describe, test } from 'node:test';

import {
  DEFAULT_CAMERA,
  MAX_ELEVATION,
  MAX_RADIUS,
  MIN_RADIUS,
  type CameraState,
} from '../src/gpu/camera.ts';
import type { KerrRenderer } from '../src/gpu/KerrRenderer.ts';
import { attachOrbitControls } from '../src/ui/orbitControls.ts';
import { FakeCanvas, FakePointerEvent, FakeWheelEvent } from './support/fakeCanvas.ts';

/** A renderer stand-in that records how often the camera was touched. */
function harness(camera: Partial<CameraState> = {}) {
  const state: CameraState = { ...DEFAULT_CAMERA, ...camera };
  let updates = 0;

  const canvas = new FakeCanvas();
  const renderer = {
    updateCamera(mutate: (c: CameraState) => void) {
      updates++;
      mutate(state);
    },
  };

  const detach = attachOrbitControls(
    canvas as unknown as HTMLCanvasElement,
    renderer as unknown as KerrRenderer,
  );

  return {
    canvas,
    detach,
    camera: state,
    get updates() {
      return updates;
    },
    /** Press, move to each point in turn, release. */
    drag(points: [number, number][], pointerId = 1) {
      canvas.dispatchEvent(
        new FakePointerEvent('pointerdown', { pointerId, clientX: 0, clientY: 0 }),
      );
      for (const [clientX, clientY] of points) {
        canvas.dispatchEvent(
          new FakePointerEvent('pointermove', { pointerId, clientX, clientY }),
        );
      }
      canvas.dispatchEvent(new FakePointerEvent('pointerup', { pointerId }));
    },
    wheel(deltaY: number, deltaMode = 0) {
      const event = new FakeWheelEvent({ deltaY, deltaMode });
      canvas.dispatchEvent(event);
      return event;
    },
  };
}

// ---------------------------------------------------------------------------

describe('orbit drag', () => {
  test('the scene follows the cursor', () => {
    // Direct manipulation: dragging right must swing the camera the other way,
    // so the object under the pointer travels with it.
    const right = harness({ azimuth: 1 });
    right.drag([[100, 0]]);
    assert.ok(right.camera.azimuth < 1, 'dragging right did not decrease azimuth');

    const left = harness({ azimuth: 1 });
    left.drag([[-100, 0]]);
    assert.ok(left.camera.azimuth > 1, 'dragging left did not increase azimuth');
  });

  test('dragging down raises the elevation', () => {
    const down = harness({ elevation: 0 });
    down.drag([[0, 100]]);
    assert.ok(down.camera.elevation > 0);

    const up = harness({ elevation: 0 });
    up.drag([[0, -100]]);
    assert.ok(up.camera.elevation < 0);
  });

  test('motion is measured incrementally, not from the press point', () => {
    // Each move is relative to the previous one, so a path and a single jump to
    // the same endpoint must agree.
    const stepwise = harness({ azimuth: 1, elevation: 0 });
    stepwise.drag([
      [30, 10],
      [60, 20],
      [90, 30],
    ]);

    const direct = harness({ azimuth: 1, elevation: 0 });
    direct.drag([[90, 30]]);

    assert.ok(Math.abs(stepwise.camera.azimuth - direct.camera.azimuth) < 1e-12);
    assert.ok(Math.abs(stepwise.camera.elevation - direct.camera.elevation) < 1e-12);
  });

  test('elevation stays clamped however far the drag goes', () => {
    const rig = harness({ elevation: 0 });
    rig.drag([[0, 100_000]]);
    assert.equal(rig.camera.elevation, MAX_ELEVATION);

    const other = harness({ elevation: 0 });
    other.drag([[0, -100_000]]);
    assert.equal(other.camera.elevation, -MAX_ELEVATION);
  });

  test('the radius is untouched by dragging', () => {
    const rig = harness({ radius: 30 });
    rig.drag([[120, -80]]);
    assert.equal(rig.camera.radius, 30);
  });

  test('the drag class is added on press and removed on release', () => {
    const rig = harness();
    rig.canvas.dispatchEvent(new FakePointerEvent('pointerdown'));
    assert.ok(rig.canvas.classList.contains('is-dragging'));
    assert.ok(rig.canvas.captured.has(1));

    rig.canvas.dispatchEvent(new FakePointerEvent('pointerup'));
    assert.ok(!rig.canvas.classList.contains('is-dragging'));
    assert.ok(!rig.canvas.captured.has(1));
  });

  test('a cancelled pointer ends the drag too', () => {
    // Losing the pointer to a browser gesture fires pointercancel, not pointerup.
    // Missing it strands the canvas in its dragging state forever.
    const rig = harness();
    rig.canvas.dispatchEvent(new FakePointerEvent('pointerdown'));
    rig.canvas.dispatchEvent(new FakePointerEvent('pointercancel'));
    assert.ok(!rig.canvas.classList.contains('is-dragging'));

    rig.canvas.dispatchEvent(new FakePointerEvent('pointermove', { clientX: 50 }));
    assert.equal(rig.updates, 0);
  });
});

// ---------------------------------------------------------------------------

describe('orbit drag ignores what it should', () => {
  test('non-primary buttons do not start a drag', () => {
    const rig = harness();
    rig.canvas.dispatchEvent(new FakePointerEvent('pointerdown', { button: 2 }));
    rig.canvas.dispatchEvent(new FakePointerEvent('pointermove', { clientX: 100 }));
    assert.equal(rig.updates, 0);
  });

  test('hovering without pressing does nothing', () => {
    const rig = harness();
    rig.canvas.dispatchEvent(new FakePointerEvent('pointermove', { clientX: 100 }));
    assert.equal(rig.updates, 0);
  });

  test('a second pointer cannot hijack an active drag', () => {
    // Touch devices deliver a full event stream per finger. Without the id
    // check, a second finger's absolute coordinates read as one enormous delta.
    const rig = harness({ azimuth: 1 });
    rig.canvas.dispatchEvent(
      new FakePointerEvent('pointerdown', { pointerId: 1, clientX: 0, clientY: 0 }),
    );
    rig.canvas.dispatchEvent(
      new FakePointerEvent('pointermove', { pointerId: 2, clientX: 500, clientY: 500 }),
    );
    assert.equal(rig.updates, 0);
    assert.equal(rig.camera.azimuth, 1);
  });

  test('releasing a different pointer does not end the drag', () => {
    const rig = harness();
    rig.canvas.dispatchEvent(new FakePointerEvent('pointerdown', { pointerId: 1 }));
    rig.canvas.dispatchEvent(new FakePointerEvent('pointerup', { pointerId: 9 }));
    assert.ok(rig.canvas.classList.contains('is-dragging'));
  });
});

// ---------------------------------------------------------------------------

describe('wheel zoom', () => {
  test('scrolling down pulls back, scrolling up moves in', () => {
    const out = harness({ radius: 30 });
    out.wheel(100);
    assert.ok(out.camera.radius > 30, 'positive deltaY did not zoom out');

    const inward = harness({ radius: 30 });
    inward.wheel(-100);
    assert.ok(inward.camera.radius < 30, 'negative deltaY did not zoom in');
  });

  test('zoom is multiplicative, so it feels the same at any distance', () => {
    // Exponential in deltaY: two equal notches equal one double-sized notch.
    const twice = harness({ radius: 20 });
    twice.wheel(50);
    twice.wheel(50);

    const once = harness({ radius: 20 });
    once.wheel(100);

    assert.ok(Math.abs(twice.camera.radius - once.camera.radius) < 1e-9);
  });

  test('the radius is clamped at both ends', () => {
    const far = harness({ radius: 30 });
    for (let i = 0; i < 200; i++) far.wheel(100);
    assert.equal(far.camera.radius, MAX_RADIUS);

    const near = harness({ radius: 30 });
    for (let i = 0; i < 200; i++) near.wheel(-100);
    assert.equal(near.camera.radius, MIN_RADIUS);
  });

  test('delta modes are normalised to pixels', () => {
    // Firefox reports lines (mode 1) and paged devices report pages (mode 2).
    // Without the scale a line-mode wheel barely moves at all.
    const pixels = harness({ radius: 30 });
    pixels.wheel(16);

    const lines = harness({ radius: 30 });
    lines.wheel(1, 1);

    assert.ok(Math.abs(pixels.camera.radius - lines.camera.radius) < 1e-9);
  });

  test('an unknown delta mode falls back to unscaled', () => {
    const rig = harness({ radius: 30 });
    rig.wheel(10, 99);
    const baseline = harness({ radius: 30 });
    baseline.wheel(10);
    assert.equal(rig.camera.radius, baseline.camera.radius);
  });

  test('the page does not scroll behind the canvas', () => {
    const rig = harness();
    assert.ok(rig.wheel(100).defaultPrevented);
  });

  test('the azimuth and elevation are untouched by zooming', () => {
    const rig = harness({ azimuth: 1.2, elevation: 0.3 });
    rig.wheel(-100);
    assert.equal(rig.camera.azimuth, 1.2);
    assert.equal(rig.camera.elevation, 0.3);
  });
});

// ---------------------------------------------------------------------------

describe('teardown', () => {
  test('detaching removes every listener', () => {
    // A leaked listener survives a StrictMode remount and keeps driving a
    // disposed renderer.
    const rig = harness({ azimuth: 1, radius: 30 });
    rig.detach();

    rig.drag([[100, 100]]);
    rig.wheel(100);

    assert.equal(rig.updates, 0);
    assert.equal(rig.camera.azimuth, 1);
    assert.equal(rig.camera.radius, 30);
    assert.ok(!rig.canvas.classList.contains('is-dragging'));
  });

  test('detaching mid-drag is safe', () => {
    const rig = harness({ azimuth: 1 });
    rig.canvas.dispatchEvent(new FakePointerEvent('pointerdown'));
    rig.detach();
    rig.canvas.dispatchEvent(new FakePointerEvent('pointermove', { clientX: 100 }));
    assert.equal(rig.camera.azimuth, 1);
  });
});
