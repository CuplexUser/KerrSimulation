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
};

export type CameraBasis = {
  eye: [number, number, number];
  forward: [number, number, number];
  right: [number, number, number];
  up: [number, number, number];
  tanHalfFov: number;
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

/**
 * Near-equatorial and far enough out that a 14 M disk sits comfortably inside a
 * 45-degree field rather than filling it. The shallow elevation is what makes
 * the lensed far side of the disk arc over the shadow.
 */
export const DEFAULT_CAMERA: CameraState = {
  azimuth: 0.6,
  elevation: 0.16,
  radius: 32,
  fovDegrees: 45,
};

export const clampElevation = (e: number): number =>
  Math.min(Math.max(e, -MAX_ELEVATION), MAX_ELEVATION);

export const clampRadius = (r: number): number =>
  Math.min(Math.max(r, MIN_RADIUS), MAX_RADIUS);

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

  return {
    eye,
    forward,
    right,
    up,
    tanHalfFov: Math.tan((camera.fovDegrees * Math.PI) / 360),
  };
}

/**
 * Smallest camera radius that still keeps the horizon comfortably in view.
 * Used to stop the zoom limit from being meaningless at high spin.
 */
export const safeMinRadius = (spin: number): number =>
  Math.max(MIN_RADIUS, horizonRadius(spin) * 2.5);
