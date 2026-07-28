import { useCallback } from 'react';
import {
  MAX_ACCUMULATED_SAMPLES,
  type RendererStats,
  type SceneParams,
} from '../gpu/KerrRenderer.ts';
import { RadialScale } from './RadialScale.tsx';

type NumericKey =
  | 'spin'
  | 'diskOuterRadius'
  | 'resolutionScale'
  | 'exposure'
  | 'bloomStrength';

type Props = {
  scene: SceneParams;
  stats: RendererStats | null;
  cameraRadius: number;
  onNumericChange: (key: NumericKey, value: number) => void;
  onDiskToggle: (enabled: boolean) => void;
  children?: React.ReactNode;
};

// Hoisted so they are not reallocated on every render.
const formatSpin = (v: number): string => v.toFixed(3);
const formatRadius = (v: number): string => `${v.toFixed(1)} M`;
const formatPercent = (v: number): string => `${Math.round(v * 100)}%`;
const formatMultiplier = (v: number): string => `${v.toFixed(2)}×`;

type SliderProps = {
  name: NumericKey;
  label: string;
  hint: string;
  min: number;
  max: number;
  step: number;
  value: number;
  format: (value: number) => string;
  onChange: (key: NumericKey, value: number) => void;
};

function Slider({
  name,
  label,
  hint,
  min,
  max,
  step,
  value,
  format,
  onChange,
}: SliderProps) {
  const handleChange = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      onChange(name, event.target.valueAsNumber);
    },
    [name, onChange],
  );

  return (
    <div className="field">
      <label className="field__head" htmlFor={`slider-${name}`}>
        <span className="field__label">{label}</span>
        <output className="field__value">{format(value)}</output>
      </label>
      <input
        id={`slider-${name}`}
        className="field__range"
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={handleChange}
      />
      <p className="field__hint">{hint}</p>
    </div>
  );
}

export function Controls({
  scene,
  stats,
  cameraRadius,
  onNumericChange,
  onDiskToggle,
  children,
}: Props) {
  const handleDiskToggle = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      onDiskToggle(event.target.checked);
    },
    [onDiskToggle],
  );

  const samples = stats?.samples ?? 0;
  const progress = Math.min(samples / MAX_ACCUMULATED_SAMPLES, 1) * 100;

  return (
    <aside className="panel">
      <header className="panel__head">
        <h1 className="panel__title">Kerr</h1>
        <p className="panel__subtitle">
          Null geodesics through the Kerr metric, refined one sample at a time.
        </p>
      </header>

      <section className="panel__section">
        <div className="convergence">
          <div className="convergence__row">
            <span className="convergence__label">Samples</span>
            <span className="convergence__count">
              {samples.toLocaleString()}
              <span className="convergence__total">
                / {MAX_ACCUMULATED_SAMPLES.toLocaleString()}
              </span>
            </span>
          </div>
          <div
            className="convergence__bar"
            role="progressbar"
            aria-valuenow={samples}
            aria-valuemin={0}
            aria-valuemax={MAX_ACCUMULATED_SAMPLES}
            aria-label="Samples accumulated"
          >
            <span style={{ width: `${progress}%` }} />
          </div>
          <p className="convergence__state">
            {stats?.interacting
              ? 'Tracing at reduced resolution while you move.'
              : stats?.converged
                ? 'Converged. Move the camera to start over.'
                : 'Refining while the camera holds still.'}
          </p>
        </div>
      </section>

      <section className="panel__section">
        <RadialScale
          horizonRadius={stats?.horizonRadius ?? 2}
          iscoRadius={stats?.iscoRadius ?? 6}
          diskOuterRadius={scene.diskOuterRadius}
          cameraRadius={cameraRadius}
        />
      </section>

      <section className="panel__section">
        <Slider
          name="spin"
          label="Spin a/M"
          hint="Frame dragging skews the shadow. Zero is Schwarzschild."
          min={0}
          max={0.998}
          step={0.001}
          value={scene.spin}
          format={formatSpin}
          onChange={onNumericChange}
        />
        <Slider
          name="diskOuterRadius"
          label="Disk edge"
          hint="Outer rim of the accretion disk. The inner rim is pinned to the ISCO."
          min={4}
          max={40}
          step={0.5}
          value={scene.diskOuterRadius}
          format={formatRadius}
          onChange={onNumericChange}
        />
        <Slider
          name="resolutionScale"
          label="Resolution"
          hint="Lower this if the first frame after a drag feels sluggish."
          min={0.4}
          max={1}
          step={0.05}
          value={scene.resolutionScale}
          format={formatPercent}
          onChange={onNumericChange}
        />
        <Slider
          name="exposure"
          label="Exposure"
          hint="Applied at tone mapping, so it does not reset the accumulation."
          min={0.2}
          max={3}
          step={0.05}
          value={scene.exposure}
          format={formatMultiplier}
          onChange={onNumericChange}
        />
        <Slider
          name="bloomStrength"
          label="Glow"
          hint="Light bleeding off the hottest part of the disk."
          min={0}
          max={2}
          step={0.05}
          value={scene.bloomStrength}
          format={formatMultiplier}
          onChange={onNumericChange}
        />

        <label className="toggle" htmlFor="toggle-disk">
          <input
            id="toggle-disk"
            type="checkbox"
            checked={scene.diskEnabled}
            onChange={handleDiskToggle}
          />
          <span className="toggle__box" aria-hidden="true" />
          <span className="toggle__text">
            <span className="toggle__label">Show accretion disk</span>
            <span className="toggle__hint">
              Turn it off to see the bare shadow and photon ring.
            </span>
          </span>
        </label>
      </section>

      {children}

      <footer className="panel__foot">
        <dl className="legend">
          <div>
            <dt>Orbit</dt>
            <dd>drag</dd>
          </div>
          <div>
            <dt>Zoom</dt>
            <dd>scroll</dd>
          </div>
          <div>
            <dt>Render</dt>
            <dd>{stats ? `${stats.width}×${stats.height}` : '—'}</dd>
          </div>
          <div>
            <dt>Split</dt>
            <dd>{stats ? `${stats.bandCount}×` : '—'}</dd>
          </div>
        </dl>
      </footer>
    </aside>
  );
}
