/**
 * The three radii that determine everything on screen, on one axis.
 *
 * Spacing is sqrt(r) rather than linear: the interesting structure all lives
 * between r = 1 and r = 10, which a linear axis would crush against the left
 * edge. Raising the spin visibly slides both the horizon and the ISCO inward,
 * which is the point.
 */

import { MAX_RADIUS } from '../gpu/camera.ts';

type Props = {
  horizonRadius: number;
  iscoRadius: number;
  diskOuterRadius: number;
  cameraRadius: number;
};

const TICKS = [1, 2, 5, 10, 20, 40];
const position = (r: number): number =>
  Math.sqrt(Math.max(r, 0) / MAX_RADIUS) * 100;

export function RadialScale({
  horizonRadius,
  iscoRadius,
  diskOuterRadius,
  cameraRadius,
}: Props) {
  const markers = [
    { key: 'horizon', label: 'r₊', value: horizonRadius, className: 'is-horizon' },
    { key: 'isco', label: 'ISCO', value: iscoRadius, className: 'is-isco' },
    { key: 'outer', label: 'disk', value: diskOuterRadius, className: 'is-outer' },
    { key: 'camera', label: 'eye', value: cameraRadius, className: 'is-camera' },
  ];

  return (
    <figure className="radial-scale">
      <div className="radial-scale__track" aria-hidden="true">
        {/* Shadow region: everything inside the horizon. */}
        <span
          className="radial-scale__horizon-fill"
          style={{ width: `${position(horizonRadius)}%` }}
        />
        {/* The disk annulus, from ISCO out to the configured edge. */}
        <span
          className="radial-scale__disk-band"
          style={{
            left: `${position(iscoRadius)}%`,
            width: `${position(diskOuterRadius) - position(iscoRadius)}%`,
          }}
        />
        {markers.map((marker) => (
          <span
            key={marker.key}
            className={`radial-scale__marker ${marker.className}`}
            style={{ left: `${position(marker.value)}%` }}
          >
            <span className="radial-scale__marker-label">{marker.label}</span>
          </span>
        ))}
      </div>

      <div className="radial-scale__ticks" aria-hidden="true">
        {TICKS.map((tick) => (
          <span key={tick} style={{ left: `${position(tick)}%` }}>
            {tick}
          </span>
        ))}
      </div>

      <figcaption className="radial-scale__caption">
        Radius in <em>M</em> — horizon {horizonRadius.toFixed(3)}, ISCO{' '}
        {iscoRadius.toFixed(3)}, disk edge {diskOuterRadius.toFixed(1)}, camera{' '}
        {cameraRadius.toFixed(1)}
      </figcaption>
    </figure>
  );
}
