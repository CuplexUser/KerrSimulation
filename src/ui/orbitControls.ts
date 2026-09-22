/**
 * Pointer, wheel, touch and keyboard camera controls.
 *
 * These write the renderer's *target* camera directly rather than going through
 * React state: a drag fires pointermove at display rate, and a re-render per
 * event would be both wasteful and janky. The renderer eases the drawn camera
 * toward the target, so every control here can move in plain steps and still
 * look continuous on screen.
 *
 *   drag                      orbit
 *   shift+drag, right-drag,   pan
 *   middle-drag
 *   wheel, pinch              zoom
 *   arrows / WASD             orbit
 *   shift+arrows              pan
 *   + / -                     zoom
 *   R or Home                 reset the view
 *
 * CONTROL_HELP below is the same list as data, for the help panel.
 */

import {
  clampElevation,
  clampPan,
  clampRadius,
  DEFAULT_CAMERA,
  type CameraState,
} from '../gpu/camera.ts';
import type { KerrRenderer } from '../gpu/KerrRenderer.ts';

const ORBIT_SPEED = 0.006;
const ZOOM_SPEED = 0.0012;
/** Wheel deltas arrive in wildly different units across browsers and devices. */
const DELTA_MODE_SCALE = [1, 16, 400];

/** Per key press (and per auto-repeat while held). */
const KEY_ORBIT_STEP = 0.08;
const KEY_PAN_STEP = 0.08;
const KEY_ZOOM_FACTOR = 1.12;

/**
 * A release counts as a fling only if the pointer was still moving this
 * recently; otherwise the user stopped before letting go and nothing coasts.
 */
const FLING_WINDOW_MS = 80;
/** Weight of the newest sample in the drag-velocity average. */
const VELOCITY_SMOOTHING = 0.5;
/** Fallback when the canvas reports no height (headless tests). */
const FALLBACK_HEIGHT = 600;

type Renderer = Pick<KerrRenderer, 'updateCamera' | 'setCameraVelocity'>;

export type ControlHelp = { keys: string[]; action: string }[];

/** Every binding this module installs, for the help panel. */
export const CONTROL_HELP: { group: string; items: ControlHelp }[] = [
  {
    group: 'Mouse and touch',
    items: [
      { keys: ['Drag'], action: 'Orbit around the hole — release while moving to coast' },
      { keys: ['Shift + drag', 'Right-drag', 'Middle-drag'], action: 'Pan the view' },
      { keys: ['Scroll'], action: 'Zoom in and out' },
      { keys: ['Pinch'], action: 'Zoom on a touch screen' },
    ],
  },
  {
    group: 'Keyboard',
    items: [
      { keys: ['←', '→'], action: 'Orbit left and right (also A, D)' },
      { keys: ['↑', '↓'], action: 'Orbit up and down (also W, S)' },
      { keys: ['Shift + arrows'], action: 'Pan the view' },
      { keys: ['+', '−'], action: 'Zoom in and out (also Page Up, Page Down)' },
      { keys: ['R', 'Home'], action: 'Reset the view' },
      { keys: ['?', 'H'], action: 'Show or hide this help' },
      { keys: ['Esc'], action: 'Close this help' },
    ],
  },
];

const FORM_TAGS = new Set(['INPUT', 'TEXTAREA', 'SELECT']);

/**
 * Whether a key event came from a form control, whose keys keep their own
 * meaning — a focused slider uses the arrows itself. Structural rather than an
 * instanceof check, so it runs headlessly too.
 */
export function isFormControl(target: EventTarget | null): boolean {
  if (!target) return false;
  if ('isContentEditable' in target && target.isContentEditable === true) return true;
  return 'tagName' in target && typeof target.tagName === 'string' && FORM_TAGS.has(target.tagName);
}

/** The fields of a keydown this module reads, or null for anything else. */
function readKey(
  event: Event,
): { key: string; shift: boolean; modified: boolean } | null {
  if (!('key' in event) || typeof event.key !== 'string') return null;
  const flag = (name: string) => name in event && Reflect.get(event, name) === true;
  return {
    key: event.key,
    shift: flag('shiftKey'),
    modified: flag('ctrlKey') || flag('metaKey') || flag('altKey'),
  };
}

// Right-drag pans, so the context menu would get in the way.
const suppressContextMenu = (event: Event) => event.preventDefault();

type CameraEdit = (camera: CameraState) => void;

const orbitBy =
  (azimuth: number, elevation: number): CameraEdit =>
  (camera) => {
    camera.azimuth += azimuth;
    camera.elevation = clampElevation(camera.elevation + elevation);
  };

const panBy =
  (x: number, y: number): CameraEdit =>
  (camera) => {
    camera.panX = clampPan(camera.panX + x);
    camera.panY = clampPan(camera.panY + y);
  };

const zoomBy =
  (factor: number): CameraEdit =>
  (camera) => {
    camera.radius = clampRadius(camera.radius * factor);
  };

const resetView: CameraEdit = (camera) => {
  Object.assign(camera, DEFAULT_CAMERA);
};

export function attachOrbitControls(
  canvas: HTMLCanvasElement,
  renderer: Renderer,
  keyTarget: EventTarget | null = globalThis.window ?? null,
): () => void {
  type Gesture = 'orbit' | 'pan';
  const pointers = new Map<number, { x: number; y: number }>();
  let gesture: Gesture = 'orbit';
  let pinchDistance = 0;

  /** Smoothed drag velocity, radians per millisecond, and when it was sampled. */
  let velocity = { azimuth: 0, elevation: 0 };
  let lastMoveAt = 0;

  const height = () => canvas.clientHeight || FALLBACK_HEIGHT;

  const pinchSpan = (): number => {
    const [a, b] = [...pointers.values()];
    return Math.hypot(a.x - b.x, a.y - b.y);
  };

  const onPointerDown = (event: PointerEvent) => {
    // Primary starts an orbit (or a pan with Shift); middle and right pan.
    if (event.button !== 0 && event.button !== 1 && event.button !== 2) return;
    if (pointers.size >= 2) return;
    if (pointers.size === 1 && event.pointerType !== 'touch') return;

    pointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.add('is-dragging');

    // Grabbing the scene stops any coasting from the last fling.
    renderer.setCameraVelocity(0, 0);
    velocity = { azimuth: 0, elevation: 0 };

    if (pointers.size === 2) {
      pinchDistance = pinchSpan();
      return;
    }
    gesture = event.button !== 0 || event.shiftKey ? 'pan' : 'orbit';
    lastMoveAt = event.timeStamp;
  };

  const onPointerMove = (event: PointerEvent) => {
    const last = pointers.get(event.pointerId);
    if (!last) return;
    const dx = event.clientX - last.x;
    const dy = event.clientY - last.y;
    last.x = event.clientX;
    last.y = event.clientY;

    if (pointers.size === 2) {
      // Pinch: the radius scales inversely with the finger spread.
      const span = pinchSpan();
      if (pinchDistance > 0 && span > 0) {
        const ratio = pinchDistance / span;
        renderer.updateCamera((camera) => {
          camera.radius = clampRadius(camera.radius * ratio);
        });
      }
      pinchDistance = span;
      return;
    }

    if (gesture === 'pan') {
      // Direct manipulation: the image follows the cursor. Pan is measured in
      // half-screen heights, so one full-height drag moves it by two.
      const scale = 2 / height();
      renderer.updateCamera((camera) => {
        camera.panX = clampPan(camera.panX - dx * scale);
        camera.panY = clampPan(camera.panY + dy * scale);
      });
      return;
    }

    // Direct manipulation: the scene follows the cursor.
    const dAzimuth = -dx * ORBIT_SPEED;
    const dElevation = dy * ORBIT_SPEED;
    renderer.updateCamera((camera) => {
      camera.azimuth += dAzimuth;
      camera.elevation = clampElevation(camera.elevation + dElevation);
    });

    const dt = event.timeStamp - lastMoveAt;
    if (dt > 0) {
      const k = VELOCITY_SMOOTHING;
      velocity = {
        azimuth: velocity.azimuth * (1 - k) + (dAzimuth / dt) * k,
        elevation: velocity.elevation * (1 - k) + (dElevation / dt) * k,
      };
    }
    lastMoveAt = event.timeStamp;
  };

  const endDrag = (event: PointerEvent) => {
    if (!pointers.has(event.pointerId)) return;
    pointers.delete(event.pointerId);
    canvas.releasePointerCapture?.(event.pointerId);

    if (pointers.size === 1) {
      // Pinch ended with one finger still down: carry on as an orbit from it.
      gesture = 'orbit';
      velocity = { azimuth: 0, elevation: 0 };
      pinchDistance = 0;
      return;
    }
    canvas.classList.remove('is-dragging');

    const flung =
      gesture === 'orbit' &&
      event.type === 'pointerup' &&
      event.timeStamp - lastMoveAt < FLING_WINDOW_MS;
    if (flung) {
      renderer.setCameraVelocity(velocity.azimuth, velocity.elevation);
    }
    velocity = { azimuth: 0, elevation: 0 };
  };

  const onWheel = (event: WheelEvent) => {
    event.preventDefault();
    const scale = DELTA_MODE_SCALE[event.deltaMode] ?? 1;
    renderer.updateCamera((camera) => {
      camera.radius = clampRadius(
        camera.radius * Math.exp(event.deltaY * scale * ZOOM_SPEED),
      );
    });
  };

  const onKeyDown = (event: Event) => {
    const pressed = readKey(event);
    if (!pressed || pressed.modified || event.defaultPrevented) return;
    if (isFormControl(event.target)) return;

    const move = keyAction(pressed.key, pressed.shift);
    if (!move) return;
    event.preventDefault();
    renderer.setCameraVelocity(0, 0);
    renderer.updateCamera(move);
  };

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener('wheel', onWheel, { passive: false });
  canvas.addEventListener('contextmenu', suppressContextMenu);
  keyTarget?.addEventListener('keydown', onKeyDown);

  return () => {
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerup', endDrag);
    canvas.removeEventListener('pointercancel', endDrag);
    canvas.removeEventListener('wheel', onWheel);
    canvas.removeEventListener('contextmenu', suppressContextMenu);
    keyTarget?.removeEventListener('keydown', onKeyDown);
  };
}

/** The camera change a key asks for, or null if the key is not bound. */
function keyAction(key: string, shift: boolean): CameraEdit | null {
  switch (key) {
    case 'ArrowLeft':
    case 'a':
    case 'A':
      return shift ? panBy(-KEY_PAN_STEP, 0) : orbitBy(-KEY_ORBIT_STEP, 0);
    case 'ArrowRight':
    case 'd':
    case 'D':
      return shift ? panBy(KEY_PAN_STEP, 0) : orbitBy(KEY_ORBIT_STEP, 0);
    case 'ArrowUp':
    case 'w':
    case 'W':
      return shift ? panBy(0, KEY_PAN_STEP) : orbitBy(0, KEY_ORBIT_STEP);
    case 'ArrowDown':
    case 's':
    case 'S':
      return shift ? panBy(0, -KEY_PAN_STEP) : orbitBy(0, -KEY_ORBIT_STEP);
    case '+':
    case '=':
    case 'PageUp':
      return zoomBy(1 / KEY_ZOOM_FACTOR);
    case '-':
    case '_':
    case 'PageDown':
      return zoomBy(KEY_ZOOM_FACTOR);
    case 'r':
    case 'R':
    case 'Home':
      return resetView;
    default:
      return null;
  }
}
