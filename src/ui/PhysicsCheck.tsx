/**
 * Renders the GPU-vs-reference validation report in the panel.
 *
 * The results go in the DOM rather than only the console so the check is a
 * visible part of the app, not something you need devtools to read. The console
 * still gets the full tables.
 */

import type { ValidationReport } from '../gpu/validatePhysics.ts';

type Props = {
  status: 'idle' | 'running' | 'done';
  report: ValidationReport | null;
  error: string | null;
  onRun: () => void;
};

const sci = (n: number): string => n.toExponential(1);

export function PhysicsCheck({ status, report, error, onRun }: Props) {
  return (
    <section className="panel__section physics" data-testid="physics-check">
      <div className="section__head">
        <h2 className="section__title">Physics check</h2>
        <button
          type="button"
          className="section__action"
          onClick={onRun}
          disabled={status === 'running'}
        >
          {status === 'running' ? 'Running…' : 'Run'}
        </button>
      </div>

      <p className="physics__blurb">
        Traces a fixed ray suite through the shipped WGSL and compares it against
        the double-precision reference in{' '}
        <code>src/physics/kerrReference.ts</code>. Both sides are computed fresh —
        there are no stored expected values.
      </p>

      {error ? (
        <p className="physics__error" data-testid="physics-error">
          {error}
        </p>
      ) : null}

      {report ? (
        <div
          className="physics__report"
          data-testid="physics-report"
          data-passed={String(report.passed)}
        >
          <p
            className={`physics__verdict ${report.passed ? 'is-pass' : 'is-fail'}`}
          >
            {report.passed
              ? 'Pass — GPU agrees with the reference'
              : 'Fail — GPU disagrees with the reference'}
          </p>

          <ul className="physics__rays">
            {report.comparisons.map((c) => (
              <li key={c.name} className={c.ok ? 'is-pass' : 'is-fail'}>
                <span className="physics__ray-name">{c.name.trim()}</span>
                <span className="physics__ray-detail">
                  {c.gpu.fate} · r_min {c.gpu.minRadius.toFixed(3)} vs{' '}
                  {c.cpu.minRadius.toFixed(3)} · max|H| {sci(c.gpu.maxAbsH)} ·
                  max ΔL/L {sci(c.gpu.maxRelL)}
                </span>
              </li>
            ))}
          </ul>

          <dl className="physics__facts">
            <div>
              <dt>ISCO, GPU vs CPU</dt>
              <dd>{sci(report.iscoMaxDiff)}</dd>
            </div>
            <div>
              <dt>Gradient ε in use</dt>
              <dd>{report.configuredEps}</dd>
            </div>
            <div>
              <dt>Best ε measured here</dt>
              <dd>{report.recommendedEps}</dd>
            </div>
          </dl>

          <p className="physics__note">
            ε sweep on this GPU:{' '}
            {report.epsilonSweep
              .map((s) => `${s.eps}→${sci(s.worstAbsH)}`)
              .join('  ')}
          </p>
        </div>
      ) : null}
    </section>
  );
}
