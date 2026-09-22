import { useCallback } from 'react';
import {
  MAX_ACCUMULATED_SAMPLES,
  type RendererStats,
  type SceneParams,
} from '../gpu/KerrRenderer.ts';
import type { Shading } from '../gpu/uniforms.ts';
import { RadialScale } from './RadialScale.tsx';
import { usePanelLayout } from './usePanelLayout.ts';

type NumericKey =
  | 'spin'
  | 'diskOuterRadius'
  | 'resolutionScale'
  | 'exposure'
  | 'bloomStrength'
  | 'dopplerBeaming'
  | 'diskThickness'
  | 'peakTemperature';

type Props = {
  scene: SceneParams;
  stats: RendererStats | null;
  cameraRadius: number;
  onNumericChange: (key: NumericKey, value: number) => void;
  onDiskToggle: (enabled: boolean) => void;
  onShadingChange: (shading: Shading) => void;
  onRestoreDefaults: () => void;
  onShowHelp: () => void;
  children?: React.ReactNode;
};

// Hoisted so they are not reallocated on every render.
const formatSpin = (v: number): string => v.toFixed(3);
const formatRadius = (v: number): string => `${v.toFixed(1)} M`;
const formatPercent = (v: number): string => `${Math.round(v * 100)}%`;
const formatMultiplier = (v: number): string => `${v.toFixed(2)}×`;
const formatKelvin = (v: number): string => `${Math.round(v).toLocaleString()} K`;

const SHADING_OPTIONS: { value: Shading; label: string }[] = [
  { value: 'cinematic', label: 'Cinematic' },
  { value: 'physical', label: 'Physical' },
];

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

type SegmentProps = {
  value: Shading;
  label: string;
  active: boolean;
  onSelect: (value: Shading) => void;
};

function Segment({ value, label, active, onSelect }: SegmentProps) {
  const handleClick = useCallback(() => onSelect(value), [onSelect, value]);
  return (
    <button
      type="button"
      role="radio"
      aria-checked={active}
      className={`segmented__option${active ? ' is-active' : ''}`}
      onClick={handleClick}
    >
      {label}
    </button>
  );
}

export function Controls({
  scene,
  stats,
  cameraRadius,
  onNumericChange,
  onDiskToggle,
  onShadingChange,
  onRestoreDefaults,
  onShowHelp,
  children,
}: Props) {
  const handleDiskToggle = useCallback(
    (event: React.ChangeEvent<HTMLInputElement>) => {
      onDiskToggle(event.target.checked);
    },
    [onDiskToggle],
  );

  const {
    panelRef,
    floating,
    collapsed,
    gestureMode,
    style,
    startMove,
    startResize,
    onPointerMove,
    endGesture,
    toggleCollapsed,
    resetLayout,
  } = usePanelLayout();

  // The collapse control sits inside the drag handle, so it has to opt out of
  // both gestures the header owns.
  const swallowGesture = useCallback((event: React.SyntheticEvent) => {
    event.stopPropagation();
  }, []);

  const samples = stats?.samples ?? 0;
  const progress = Math.min(samples / MAX_ACCUMULATED_SAMPLES, 1) * 100;

  return (
    <aside
      ref={panelRef}
      className={`panel${collapsed ? ' is-collapsed' : ''}${floating ? ' is-floating' : ''}`}
      style={style}
      data-gesture={gestureMode ?? undefined}
      onPointerMove={onPointerMove}
      onPointerUp={endGesture}
      onPointerCancel={endGesture}
    >
      <header
        className="panel__head"
        onPointerDown={startMove}
        onDoubleClick={toggleCollapsed}
      >
        <div className="panel__heading">
          <h1 className="panel__title">Kerr</h1>
          {collapsed ? (
            <p className="panel__summary">
              <span className="panel__summary-count">
                {samples.toLocaleString()}
              </span>
              <span className="panel__summary-unit">samples</span>
            </p>
          ) : (
            <p className="panel__subtitle">
              Null geodesics through the Kerr metric, refined one sample at a
              time.
            </p>
          )}
        </div>
        <button
          type="button"
          className="panel__collapse"
          onClick={toggleCollapsed}
          onPointerDown={swallowGesture}
          onDoubleClick={swallowGesture}
          aria-expanded={!collapsed}
          aria-label={collapsed ? 'Expand controls' : 'Collapse controls'}
          title={collapsed ? 'Expand' : 'Collapse'}
        >
          <span className="panel__chevron" aria-hidden="true" />
        </button>
      </header>

      <div className="panel__scroll">
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
          <div className="section__head">
            <h2 className="section__title">Parameters</h2>
            <button
              type="button"
              className="section__action"
              onClick={onRestoreDefaults}
              title="Restore every control and the camera to their starting values"
            >
              Restore defaults
            </button>
          </div>

          <div className="field">
            <span className="field__head">
              <span className="field__label" id="shading-label">
                Shading
              </span>
            </span>
            <div className="segmented" role="radiogroup" aria-labelledby="shading-label">
              {SHADING_OPTIONS.map((option) => (
                <Segment
                  key={option.value}
                  value={option.value}
                  label={option.label}
                  active={scene.shading === option.value}
                  onSelect={onShadingChange}
                />
              ))}
            </div>
            <p className="field__hint">
              {scene.shading === 'physical'
                ? 'Novikov-Thorne disk, blackbody color, and the exact redshift: gravity, Doppler and frame dragging in one factor. Nothing is tuned.'
                : 'The film look: a hand-tuned color ramp with beaming mostly suppressed.'}
            </p>
          </div>

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
            name="diskThickness"
            label="Disk thickness"
            hint="Half-thickness as a fraction of radius. At zero the disk is a mathematical plane, and its lensed images beside the shadow are thinner than a pixel."
            min={0}
            max={0.12}
            step={0.005}
            value={scene.diskThickness}
            format={formatPercent}
            onChange={onNumericChange}
          />
          {scene.shading === 'physical' ? (
            <Slider
              name="peakTemperature"
              label="Peak temperature"
              hint="Emitted temperature at the hottest ring. What you see is shifted by g on the way out: bluer and brighter where the disk comes toward you."
              min={2500}
              max={20000}
              step={100}
              value={scene.peakTemperature}
              format={formatKelvin}
              onChange={onNumericChange}
            />
          ) : (
            <Slider
              name="dopplerBeaming"
              label="Doppler"
              hint="How much relativistic beaming to show. Films suppress it — the real asymmetry looks like a bug."
              min={0}
              max={1}
              step={0.01}
              value={scene.dopplerBeaming}
              format={formatPercent}
              onChange={onNumericChange}
            />
          )}
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
      </div>

      <footer className="panel__foot">
        <dl className="legend">
          <div>
            <dt>Render</dt>
            <dd>{stats ? `${stats.width}×${stats.height}` : '—'}</dd>
          </div>
          <div>
            <dt>Split</dt>
            <dd>{stats ? `${stats.bandCount}×` : '—'}</dd>
          </div>
        </dl>
        <button
          type="button"
          className="panel__help"
          onClick={onShowHelp}
          title="Mouse, touch and keyboard controls (?)"
        >
          Controls <kbd>?</kbd>
        </button>
        {floating ? (
          <button type="button" className="panel__reset" onClick={resetLayout}>
            Reset panel
          </button>
        ) : null}
      </footer>

      {floating && !collapsed ? (
        <span
          className="panel__grip"
          onPointerDown={startResize}
          role="separator"
          aria-label="Resize controls"
        />
      ) : null}
    </aside>
  );
}
