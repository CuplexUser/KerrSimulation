/**
 * Pointer-drag orbit and wheel zoom.
 *
 * These mutate the renderer's camera directly rather than going through React
 * state: a drag fires pointermove at display rate, and a re-render per event
 * would be both wasteful and janky. Every mutation resets accumulation.
 */

import { clampElevation, clampRadius } from '../gpu/camera.ts';
import type { KerrRenderer } from '../gpu/KerrRenderer.ts';

const ORBIT_SPEED = 0.006;
const ZOOM_SPEED = 0.0012;
/** Wheel deltas arrive in wildly different units across browsers and devices. */
const DELTA_MODE_SCALE = [1, 16, 400];

export function attachOrbitControls(
  canvas: HTMLCanvasElement,
  renderer: KerrRenderer,
): () => void {
  let activePointer: number | null = null;
  let lastX = 0;
  let lastY = 0;

  const onPointerDown = (event: PointerEvent) => {
    if (event.button !== 0) return;
    activePointer = event.pointerId;
    lastX = event.clientX;
    lastY = event.clientY;
    canvas.setPointerCapture(event.pointerId);
    canvas.classList.add('is-dragging');
  };

  const onPointerMove = (event: PointerEvent) => {
    if (event.pointerId !== activePointer) return;
    const dx = event.clientX - lastX;
    const dy = event.clientY - lastY;
    lastX = event.clientX;
    lastY = event.clientY;

    // Direct manipulation: the scene follows the cursor.
    renderer.updateCamera((camera) => {
      camera.azimuth -= dx * ORBIT_SPEED;
      camera.elevation = clampElevation(camera.elevation + dy * ORBIT_SPEED);
    });
  };

  const endDrag = (event: PointerEvent) => {
    if (event.pointerId !== activePointer) return;
    activePointer = null;
    canvas.releasePointerCapture?.(event.pointerId);
    canvas.classList.remove('is-dragging');
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

  canvas.addEventListener('pointerdown', onPointerDown);
  canvas.addEventListener('pointermove', onPointerMove);
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener('wheel', onWheel, { passive: false });

  return () => {
    canvas.removeEventListener('pointerdown', onPointerDown);
    canvas.removeEventListener('pointermove', onPointerMove);
    canvas.removeEventListener('pointerup', endDrag);
    canvas.removeEventListener('pointercancel', endDrag);
    canvas.removeEventListener('wheel', onWheel);
  };
}
