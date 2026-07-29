/**
 * Regression tests for the orbit camera basis.
 *
 * The shader does nothing but generate rays from these three vectors, so a
 * handedness flip or a non-orthogonal basis here mirrors or skews the entire
 * image — and a mirrored black hole still looks like a black hole. The
 * orientation checks below are the ones that would notice.
 */

import { strict as assert } from 'node:assert';
import { describe, test } from 'node:test';

import {
  DEFAULT_CAMERA,
  MAX_ELEVATION,
  MAX_RADIUS,
  MIN_RADIUS,
  cameraBasis,
  clampElevation,
  clampRadius,
  safeMinRadius,
  type CameraState,
} from '../src/gpu/camera.ts';
import { horizonRadius } from '../src/physics/kerrReference.ts';

type Vec3 = readonly [number, number, number];

const dot = (a: Vec3, b: Vec3) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a: Vec3) => Math.hypot(a[0], a[1], a[2]);
const cross = (a: Vec3, b: Vec3): Vec3 => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0],
];

const camera = (over: Partial<CameraState> = {}): CameraState => ({
  ...DEFAULT_CAMERA,
  ...over,
});

/** Azimuth/elevation pairs covering all four quadrants and both poles. */
const ORIENTATIONS: [number, number][] = [
  [0, 0],
  [0.6, 0.055],
  [Math.PI / 2, 0.4],
  [Math.PI, -0.4],
  [-2.3, 1.2],
  [4.7, -1.2],
  [0.3, MAX_ELEVATION],
  [0.3, -MAX_ELEVATION],
];

// ---------------------------------------------------------------------------

describe('cameraBasis', () => {
  test('is orthonormal at every orientation', () => {
    for (const [azimuth, elevation] of ORIENTATIONS) {
      const { forward, right, up } = cameraBasis(camera({ azimuth, elevation }));
      const at = `azimuth=${azimuth}, elevation=${elevation}`;

      for (const [label, axis] of Object.entries({ forward, right, up })) {
        assert.ok(Math.abs(norm(axis) - 1) < 1e-12, `${label} is not unit at ${at}`);
      }
      assert.ok(Math.abs(dot(forward, right)) < 1e-12, `forward/right not perpendicular at ${at}`);
      assert.ok(Math.abs(dot(forward, up)) < 1e-12, `forward/up not perpendicular at ${at}`);
      assert.ok(Math.abs(dot(right, up)) < 1e-12, `right/up not perpendicular at ${at}`);
    }
  });

  test('is right-handed', () => {
    // right x up == -forward for a right-handed (right, up, -forward) frame.
    // If this flips, the image is mirrored and nothing else complains.
    for (const [azimuth, elevation] of ORIENTATIONS) {
      const { forward, right, up } = cameraBasis(camera({ azimuth, elevation }));
      const handedness = cross(right, up);
      for (let i = 0; i < 3; i++) {
        assert.ok(
          Math.abs(handedness[i] + forward[i]) < 1e-12,
          `handedness flipped at azimuth=${azimuth}, elevation=${elevation}`,
        );
      }
    }
  });

  test('always looks at the hole from the requested distance', () => {
    for (const radius of [MIN_RADIUS, 12, 40, MAX_RADIUS]) {
      const { eye, forward } = cameraBasis(camera({ radius }));
      assert.ok(Math.abs(norm(eye) - radius) < 1e-9);
      // Forward points from the eye back to the origin.
      for (let i = 0; i < 3; i++) {
        assert.ok(Math.abs(forward[i] + eye[i] / radius) < 1e-12);
      }
    }
  });

  test('elevation is measured from the equatorial plane', () => {
    // Zero elevation must put the eye in the disk plane — this is what makes
    // the default view edge-on, and the reason elevation is not polar angle.
    const flat = cameraBasis(camera({ elevation: 0 }));
    assert.equal(flat.eye[2], 0);
    // ...and up points along the spin axis there.
    assert.ok(Math.abs(flat.up[2] - 1) < 1e-12);

    const raised = cameraBasis(camera({ elevation: 0.5, radius: 40 }));
    assert.ok(Math.abs(raised.eye[2] - 40 * Math.sin(0.5)) < 1e-9);
  });

  test('azimuth sweeps the eye around the spin axis', () => {
    const start = cameraBasis(camera({ azimuth: 0, elevation: 0, radius: 20 }));
    const quarter = cameraBasis(camera({ azimuth: Math.PI / 2, elevation: 0, radius: 20 }));
    assert.ok(Math.abs(start.eye[0] - 20) < 1e-9 && Math.abs(start.eye[1]) < 1e-9);
    assert.ok(Math.abs(quarter.eye[0]) < 1e-9 && Math.abs(quarter.eye[1] - 20) < 1e-9);
  });

  test('clamps elevation rather than degenerating at the poles', () => {
    // The up vector is built from cross(forward, +z), which collapses if the
    // camera ever reaches the axis exactly.
    for (const elevation of [Math.PI / 2, -Math.PI / 2, 12, -12]) {
      const { right, up } = cameraBasis(camera({ elevation }));
      assert.ok(Math.abs(norm(right) - 1) < 1e-12, `right degenerated at elevation=${elevation}`);
      assert.ok(Math.abs(norm(up) - 1) < 1e-12, `up degenerated at elevation=${elevation}`);
    }
  });

  test('tanHalfFov is the half-angle tangent', () => {
    assert.ok(Math.abs(cameraBasis(camera({ fovDegrees: 90 })).tanHalfFov - 1) < 1e-12);
    assert.ok(
      Math.abs(cameraBasis(camera({ fovDegrees: 45 })).tanHalfFov - Math.tan(Math.PI / 8)) < 1e-12,
    );
  });
});

// ---------------------------------------------------------------------------

describe('clamps', () => {
  test('clampElevation keeps the camera off the poles', () => {
    assert.equal(clampElevation(10), MAX_ELEVATION);
    assert.equal(clampElevation(-10), -MAX_ELEVATION);
    assert.equal(clampElevation(0.3), 0.3);
    assert.ok(MAX_ELEVATION < Math.PI / 2, 'MAX_ELEVATION reaches the pole');
  });

  test('clampRadius keeps the camera in the supported shell', () => {
    assert.equal(clampRadius(0), MIN_RADIUS);
    assert.equal(clampRadius(1e6), MAX_RADIUS);
    assert.equal(clampRadius(20), 20);
    assert.ok(MIN_RADIUS < MAX_RADIUS);
  });

  test('the inner zoom limit clears the ergosphere', () => {
    // No static observer exists inside the ergosphere, which reaches r = 2 in
    // the equatorial plane at any spin, so the coordinate camera must stay out.
    assert.ok(MIN_RADIUS > 2);
  });

  test('the default camera sits inside its own limits', () => {
    assert.ok(DEFAULT_CAMERA.radius >= MIN_RADIUS && DEFAULT_CAMERA.radius <= MAX_RADIUS);
    assert.equal(clampElevation(DEFAULT_CAMERA.elevation), DEFAULT_CAMERA.elevation);
    assert.ok(DEFAULT_CAMERA.fovDegrees > 0 && DEFAULT_CAMERA.fovDegrees < 180);
  });

  test('the default view is close enough to edge-on to keep the disk thin', () => {
    // Above roughly 10 degrees the disk reads as a wedge and the lensed far side
    // stops arcing over the shadow as a separate band.
    assert.ok(Math.abs(DEFAULT_CAMERA.elevation) < (10 * Math.PI) / 180);
  });

  test('safeMinRadius never lets the horizon crowd the frame', () => {
    for (const spin of [0, 0.5, 0.85, 0.998, 1]) {
      const r = safeMinRadius(spin);
      assert.ok(r >= MIN_RADIUS, `below the hard floor at a=${spin}`);
      assert.ok(r >= horizonRadius(spin) * 2.5, `too close to the horizon at a=${spin}`);
      assert.ok(r <= MAX_RADIUS, `unusable zoom range at a=${spin}`);
    }
  });
});
