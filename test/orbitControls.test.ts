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
 *
 * Pan, pinch, fling and the keyboard map are covered too, and CONTROL_HELP is
 * checked against the bindings so the help panel cannot drift from them.
 */

import { strict as assert } from 'node:assert';
import { describe, test } from 'node:test';

import {
  DEFAULT_CAMERA,
  MAX_ELEVATION,
  MAX_RADIUS,
  MAX_PAN,
  MIN_RADIUS,
  type CameraState,
} from '../src/gpu/camera.ts';
import { attachOrbitControls, CONTROL_HELP } from '../src/ui/orbitControls.ts';
import {
  FakeCanvas,
  FakeKeyEvent,
  FakePointerEvent,
  FakeWheelEvent,
} from './support/fakeCanvas.ts';

/** A renderer stand-in that records how often the camera was touched. */
function harness(camera: Partial<CameraState> = {}) {
  const state: CameraState = { ...DEFAULT_CAMERA, ...camera };
  let updates = 0;
  const velocity = { azimuth: 0, elevation: 0 };

  const canvas = new FakeCanvas();
  const keys = new EventTarget();
  const renderer = {
    updateCamera(mutate: (c: CameraState) => void) {
      updates++;
      mutate(state);
    },
    setCameraVelocity(azimuth: number, elevation: number) {
      velocity.azimuth = azimuth;
      velocity.elevation = elevation;
    },
  };

  const detach = attachOrbitControls(canvas as unknown as HTMLCanvasElement, renderer, keys);

  return {
    canvas,
    detach,
    camera: state,
    velocity,
    get updates() {
      return updates;
    },
    key(key: string, shiftKey = false) {
      const event = new FakeKeyEvent(key, { shiftKey });
      keys.dispatchEvent(event);
      return event;
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
  test('the back and forward buttons do not start a drag', () => {
    for (const button of [3, 4]) {
      const rig = harness();
      rig.canvas.dispatchEvent(new FakePointerEvent('pointerdown', { button }));
      rig.canvas.dispatchEvent(new FakePointerEvent('pointermove', { clientX: 100 }));
      assert.equal(rig.updates, 0);
    }
  });

  test('the context menu is suppressed, since right-drag pans', () => {
    const rig = harness();
    const event = new Event('contextmenu', { cancelable: true });
    rig.canvas.dispatchEvent(event);
    assert.ok(event.defaultPrevented);
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

describe('pan', () => {
  const panDrag = (
    init: { button?: number; shiftKey?: boolean },
    to: [number, number],
    over: Partial<CameraState> = {},
  ) => {
    const rig = harness(over);
    rig.canvas.dispatchEvent(new FakePointerEvent('pointerdown', init));
    rig.canvas.dispatchEvent(
      new FakePointerEvent('pointermove', { clientX: to[0], clientY: to[1] }),
    );
    rig.canvas.dispatchEvent(new FakePointerEvent('pointerup'));
    return rig;
  };

  test('shift-drag, right-drag and middle-drag pan instead of orbiting', () => {
    for (const init of [{ shiftKey: true }, { button: 2 }, { button: 1 }]) {
      const rig = panDrag(init, [60, 0]);
      assert.equal(rig.camera.azimuth, DEFAULT_CAMERA.azimuth, 'a pan also orbited');
      assert.notEqual(rig.camera.panX, 0, `${JSON.stringify(init)} did not pan`);
    }
  });

  test('the image follows the cursor', () => {
    // Dragging right slides the view window left over the scene, so the scene
    // moves right with the pointer; dragging down moves it down.
    const rig = panDrag({ button: 2 }, [60, 30]);
    assert.ok(rig.camera.panX < 0);
    assert.ok(rig.camera.panY > 0);
  });

  test('a half-height drag pans by one half-screen', () => {
    // The fallback height is 600 px, and pan is measured in half-heights.
    const rig = panDrag({ button: 2 }, [0, 300]);
    assert.ok(Math.abs(rig.camera.panY - 1) < 1e-12);
  });

  test('pan is clamped', () => {
    const rig = panDrag({ button: 2 }, [-100_000, 100_000]);
    assert.equal(rig.camera.panX, MAX_PAN);
    assert.equal(rig.camera.panY, MAX_PAN);
  });
});

// ---------------------------------------------------------------------------

describe('fling', () => {
  const fling = (releaseDelay: number) => {
    const rig = harness();
    rig.canvas.dispatchEvent(new FakePointerEvent('pointerdown', { timeStamp: 0 }));
    for (let i = 1; i <= 5; i++) {
      rig.canvas.dispatchEvent(
        new FakePointerEvent('pointermove', { clientX: i * 10, timeStamp: i * 16 }),
      );
    }
    rig.canvas.dispatchEvent(
      new FakePointerEvent('pointerup', { timeStamp: 80 + releaseDelay }),
    );
    return rig;
  };

  test('releasing a moving drag hands its velocity to the renderer', () => {
    const rig = fling(10);
    // Dragging right decreases azimuth, so it coasts the same way.
    assert.ok(rig.velocity.azimuth < 0);
    assert.equal(rig.velocity.elevation, 0);
  });

  test('pausing before release does not fling', () => {
    const rig = fling(500);
    assert.equal(rig.velocity.azimuth, 0);
  });

  test('pressing again stops a coast', () => {
    const rig = fling(10);
    rig.canvas.dispatchEvent(new FakePointerEvent('pointerdown', { timeStamp: 200 }));
    assert.equal(rig.velocity.azimuth, 0);
  });
});

// ---------------------------------------------------------------------------

describe('pinch zoom', () => {
  test('spreading two fingers zooms in, pinching zooms out', () => {
    const pinch = (from: number, to: number) => {
      const rig = harness({ radius: 30 });
      const touch = (type: string, pointerId: number, clientX: number) =>
        rig.canvas.dispatchEvent(
          new FakePointerEvent(type, { pointerId, pointerType: 'touch', clientX }),
        );
      touch('pointerdown', 1, 0);
      touch('pointerdown', 2, from);
      touch('pointermove', 2, to);
      return rig.camera.radius;
    };

    assert.ok(pinch(100, 200) < 30, 'spreading did not zoom in');
    assert.ok(pinch(200, 100) > 30, 'pinching did not zoom out');
    assert.ok(Math.abs(pinch(100, 200) - 15) < 1e-9, 'zoom is not inverse to the spread');
  });

  test('a second mouse pointer is not a pinch', () => {
    const rig = harness({ radius: 30 });
    rig.canvas.dispatchEvent(new FakePointerEvent('pointerdown', { pointerId: 1 }));
    rig.canvas.dispatchEvent(new FakePointerEvent('pointerdown', { pointerId: 2 }));
    rig.canvas.dispatchEvent(
      new FakePointerEvent('pointermove', { pointerId: 2, clientX: 300 }),
    );
    assert.equal(rig.camera.radius, 30);
  });
});

// ---------------------------------------------------------------------------

describe('keyboard', () => {
  test('arrows orbit the camera the way they point', () => {
    const rig = harness({ azimuth: 1, elevation: 0 });
    rig.key('ArrowRight');
    assert.ok(rig.camera.azimuth > 1);
    rig.key('ArrowUp');
    assert.ok(rig.camera.elevation > 0);

    const other = harness({ azimuth: 1, elevation: 0 });
    other.key('a');
    other.key('s');
    assert.ok(other.camera.azimuth < 1);
    assert.ok(other.camera.elevation < 0);
  });

  test('shift+arrows pan', () => {
    const rig = harness();
    rig.key('ArrowRight', true);
    rig.key('ArrowUp', true);
    assert.ok(rig.camera.panX > 0);
    assert.ok(rig.camera.panY > 0);
    assert.equal(rig.camera.azimuth, DEFAULT_CAMERA.azimuth);
  });

  test('+ and - zoom, and stay clamped', () => {
    const rig = harness({ radius: 30 });
    rig.key('+');
    assert.ok(rig.camera.radius < 30);
    for (let i = 0; i < 100; i++) rig.key('-');
    assert.equal(rig.camera.radius, MAX_RADIUS);
  });

  test('R resets the view', () => {
    const rig = harness({ azimuth: 2, elevation: 0.5, radius: 12, panX: 0.4 });
    rig.key('r');
    assert.deepEqual({ ...rig.camera }, DEFAULT_CAMERA);
  });

  test('bound keys are consumed and unbound keys are not', () => {
    const rig = harness();
    assert.ok(rig.key('ArrowLeft').defaultPrevented);
    const unbound = rig.key('q');
    assert.ok(!unbound.defaultPrevented);
    assert.equal(rig.updates, 1);
  });

  test('keys typed into a form control are left alone', () => {
    // A focused slider uses the arrows itself. In the page the event reaches
    // the window from the input; here the target itself stands in for it.
    const keys = Object.assign(new EventTarget(), { tagName: 'INPUT' });
    let updates = 0;
    attachOrbitControls(
      new FakeCanvas() as unknown as HTMLCanvasElement,
      { updateCamera: () => updates++, setCameraVelocity: () => {} },
      keys,
    );
    keys.dispatchEvent(new FakeKeyEvent('ArrowRight'));
    assert.equal(updates, 0);
  });

  test('the help panel lists every key the handler binds', () => {
    // Guards the reference against drifting from the code.
    const listed = CONTROL_HELP.flatMap((group) => group.items)
      .map((item) => `${item.keys.join(' ')} ${item.action}`)
      .join(' ');
    for (const token of ['←', '→', '↑', '↓', 'A, D', 'W, S', 'Shift + arrows', '+', 'Page Up', 'R', 'Home']) {
      assert.ok(listed.includes(token), `help does not mention ${token}`);
    }
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
    rig.key('ArrowLeft');

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
