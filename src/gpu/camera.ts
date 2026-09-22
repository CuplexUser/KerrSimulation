/**
 * Orbit camera. The spin axis is +z, so elevation is measured from the
 * equatorial plane — elevation 0 views the disk edge-on.
 *
 * The basis is built on the CPU and handed to the shader as three vectors, which
 * keeps camera semantics in TypeScript where they are easy to reason about and
 * leaves the WGSL to do nothing but ray generation.
 */

import { horizonRadius } from '../physics/kerrReference.ts';

export type CameraState = {
  azimuth: number;
  elevation: number;
  radius: number;
  fovDegrees: number;
  /**
   * Pan, as an off-axis lens shift in units of the half-height of the image
   * plane: 1 moves the view center by half a screen. The eye stays put and keeps
   * facing the hole, so panning reframes the picture without changing which
   * light reaches the camera — no approximation enters the physics.
   */
  panX: number;
  panY: number;
};

export type CameraBasis = {
  eye: [number, number, number];
  forward: [number, number, number];
  right: [number, number, number];
  up: [number, number, number];
  tanHalfFov: number;
  /** Lens shift in image-plane units (tanHalfFov already applied). */
  shift: [number, number];
};

/** Keeps the up-vector construction from degenerating at the poles. */
export const MAX_ELEVATION = Math.PI / 2 - 0.02;

/**
 * Zoom limits. The inner limit is deliberately well outside the ergosphere
 * (which reaches r = 2 in the equatorial plane at any spin): the renderer uses
 * the coordinate-camera convention, whose aberration only becomes noticeable
 * close in, and no static observer exists inside the ergosphere at all.
 */
export const MIN_RADIUS = 6;
export const MAX_RADIUS = 55;

/** Pan limit, in half-screens. Far enough to frame the disk edge, no further. */
export const MAX_PAN = 1.5;

/**
 * Almost exactly edge-on, and far enough out that the disk sits inside the field
 * rather than filling it.
 *
 * The shallow elevation is doing most of the visual work: it is what collapses
 * the disk into a thin plane and lets the lensed far side arc up over the shadow
 * as a separate band. Tilt much above ~10 degrees and it reads as a thick wedge
 * instead.
 */
export const DEFAULT_CAMERA: CameraState = {
  azimuth: 0.6,
  elevation: 0.055,
  radius: 40,
  fovDegrees: 45,
  panX: 0,
  panY: 0,
};

export const clampElevation = (e: number): number =>
  Math.min(Math.max(e, -MAX_ELEVATION), MAX_ELEVATION);

export const clampRadius = (r: number): number =>
  Math.min(Math.max(r, MIN_RADIUS), MAX_RADIUS);

export const clampPan = (p: number): number => Math.min(Math.max(p, -MAX_PAN), MAX_PAN);

type Vec3 = [number, number, number];

const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

const normalize = (a: Vec3): Vec3 => {
  const n = Math.hypot(a[0], a[1], a[2]) || 1;
  return [a[0] / n, a[1] / n, a[2] / n];
};

export function cameraBasis(camera: CameraState): CameraBasis {
  const elevation = clampElevation(camera.elevation);
  const ce = Math.cos(elevation);
  const se = Math.sin(elevation);
  const ca = Math.cos(camera.azimuth);
  const sa = Math.sin(camera.azimuth);

  const eye: Vec3 = [
    camera.radius * ce * ca,
    camera.radius * ce * sa,
    camera.radius * se,
  ];

  // Always looking at the hole.
  const forward = normalize([-eye[0], -eye[1], -eye[2]]);
  const worldUp: Vec3 = [0, 0, 1];
  const right = normalize(cross(forward, worldUp));
  const up = cross(right, forward);

  const tanHalfFov = Math.tan((camera.fovDegrees * Math.PI) / 360);
  return {
    eye,
    forward,
    right,
    up,
    tanHalfFov,
    shift: [camera.panX * tanHalfFov, camera.panY * tanHalfFov],
  };
}

/** Time constant of the camera's approach to its target, in milliseconds. */
export const CAMERA_EASE_MS = 70;

/** Below this every component counts as having arrived. */
const SETTLE_EPSILON = 1e-4;

/**
 * Advance `current` toward `target` over `dtMs`, frame-rate independently.
 *
 * Input writes the target; the renderer draws the current state. Easing the
 * gap exponentially turns a notched wheel's discrete jumps and a mouse's
 * jittery deltas into continuous motion. Radius eases in log space so a zoom
 * feels the same speed at any distance.
 *
 * Returns true while the camera is still moving, and snaps exactly onto the
 * target once it is close enough that another step would be invisible.
 */
export function easeCamera(current: CameraState, target: CameraState, dtMs: number): boolean {
  const k = 1 - Math.exp(-Math.max(dtMs, 0) / CAMERA_EASE_MS);
  const logRadius = Math.log(current.radius);
  const logTarget = Math.log(target.radius);

  const gaps = [
    target.azimuth - current.azimuth,
    target.elevation - current.elevation,
    logTarget - logRadius,
    target.panX - current.panX,
    target.panY - current.panY,
    target.fovDegrees - current.fovDegrees,
  ];
  if (gaps.every((gap) => Math.abs(gap) < SETTLE_EPSILON)) {
    const moved = gaps.some((gap) => gap !== 0);
    Object.assign(current, target);
    return moved;
  }

  current.azimuth += gaps[0] * k;
  current.elevation += gaps[1] * k;
  current.radius = Math.exp(logRadius + gaps[2] * k);
  current.panX += gaps[3] * k;
  current.panY += gaps[4] * k;
  current.fovDegrees += gaps[5] * k;
  return true;
}

/**
 * Smallest camera radius that still keeps the horizon comfortably in view.
 * Used to stop the zoom limit from being meaningless at high spin.
 */
export const safeMinRadius = (spin: number): number =>
  Math.max(MIN_RADIUS, horizonRadius(spin) * 2.5);
