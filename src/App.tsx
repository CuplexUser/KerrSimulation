import { useCallback, useEffect, useRef, useState } from 'react';
import { DEFAULT_CAMERA } from './gpu/camera.ts';
import { initWebGPU } from './gpu/device.ts';
import {
  DEFAULT_SCENE,
  KerrRenderer,
  type RendererStats,
  type SceneParams,
} from './gpu/KerrRenderer.ts';
import {
  validatePhysicsOnGpu,
  type ValidationReport,
} from './gpu/validatePhysics.ts';
import type { Shading } from './gpu/uniforms.ts';
import { Controls } from './ui/Controls.tsx';
import { HelpPanel } from './ui/HelpPanel.tsx';
import { attachOrbitControls, isFormControl } from './ui/orbitControls.ts';
import { PhysicsCheck } from './ui/PhysicsCheck.tsx';
import { Unsupported } from './ui/Unsupported.tsx';

type Status =
  | { kind: 'starting' }
  | { kind: 'ready' }
  | { kind: 'unsupported'; reason: string };

type NumericKey =
  | 'spin'
  | 'diskOuterRadius'
  | 'resolutionScale'
  | 'exposure'
  | 'bloomStrength'
  | 'dopplerBeaming'
  | 'diskThickness'
  | 'peakTemperature';

export default function App() {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<KerrRenderer | null>(null);

  const [status, setStatus] = useState<Status>({ kind: 'starting' });
  const [scene, setScene] = useState<SceneParams>(DEFAULT_SCENE);
  const [stats, setStats] = useState<RendererStats | null>(null);
  const [cameraRadius, setCameraRadius] = useState(DEFAULT_CAMERA.radius);
  const [helpOpen, setHelpOpen] = useState(false);

  const [checkStatus, setCheckStatus] = useState<'idle' | 'running' | 'done'>(
    'idle',
  );
  const [report, setReport] = useState<ValidationReport | null>(null);
  const [checkError, setCheckError] = useState<string | null>(null);

  // Device and renderer lifecycle. Runs once; the canceled flag guards against
  // StrictMode's double-invoke in development.
  useEffect(() => {
    let canceled = false;
    let detachControls: (() => void) | undefined;

    const runPhysicsCheck = async (device: GPUDevice) => {
      setCheckStatus('running');
      setCheckError(null);
      try {
        const result = await validatePhysicsOnGpu(device);
        if (!canceled) setReport(result);
      } catch (error) {
        if (!canceled) {
          setCheckError(
            error instanceof Error ? error.message : 'The physics check failed.',
          );
        }
      } finally {
        if (!canceled) setCheckStatus('done');
      }
    };

    const start = async () => {
      const init = await initWebGPU();
      if (canceled) return;

      if (!init.ok) {
        setStatus({ kind: 'unsupported', reason: init.reason });
        return;
      }

      const canvas = canvasRef.current;
      if (!canvas) return;

      const { device } = init.context;
      device.addEventListener('uncapturederror', (event) => {
        console.error('WebGPU error:', event.error);
      });

      const watchForDeviceLoss = async () => {
        const info = await device.lost;
        if (canceled) return;
        setStatus({
          kind: 'unsupported',
          reason: `The GPU device was lost (${info.reason}): ${info.message}`,
        });
      };
      void watchForDeviceLoss();

      try {
        const renderer = await KerrRenderer.create(device, canvas);
        if (canceled) {
          renderer.dispose();
          return;
        }

        rendererRef.current = renderer;
        renderer.onStats((next) => {
          setStats(next);
          setCameraRadius(renderer.camera.radius);
        });
        detachControls = attachOrbitControls(canvas, renderer);
        renderer.start();
        setStatus({ kind: 'ready' });

        if (new URLSearchParams(globalThis.location.search).has('validate')) {
          void runPhysicsCheck(device);
        }
      } catch (error) {
        setStatus({
          kind: 'unsupported',
          reason:
            error instanceof Error
              ? error.message
              : 'The renderer failed to start.',
        });
      }
    };

    void start();

    return () => {
      canceled = true;
      detachControls?.();
      rendererRef.current?.dispose();
      rendererRef.current = null;
    };
  }, []);

  // Push UI parameters down to the renderer.
  useEffect(() => {
    rendererRef.current?.setScene(scene);
  }, [scene]);

  const handleNumericChange = useCallback((key: NumericKey, value: number) => {
    setScene((current) => ({ ...current, [key]: value }));
  }, []);

  const handleDiskToggle = useCallback((diskEnabled: boolean) => {
    setScene((current) => ({ ...current, diskEnabled }));
  }, []);

  const handleShadingChange = useCallback((shading: Shading) => {
    setScene((current) => ({ ...current, shading }));
  }, []);

  const showHelp = useCallback(() => setHelpOpen(true), []);
  const hideHelp = useCallback(() => setHelpOpen(false), []);

  // ? and H toggle the controls reference, Escape closes it. The camera keys
  // are handled by orbitControls; these are the only app-level shortcuts.
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.ctrlKey || event.metaKey || event.altKey) return;
      if (isFormControl(event.target)) return;

      if (event.key === '?' || event.key === 'h' || event.key === 'H') {
        event.preventDefault();
        setHelpOpen((open) => !open);
      } else if (event.key === 'Escape') {
        setHelpOpen(false);
      }
    };
    globalThis.addEventListener('keydown', onKeyDown);
    return () => globalThis.removeEventListener('keydown', onKeyDown);
  }, []);

  // Scene and camera together — "defaults" means the view you started with, and
  // restoring the sliders while leaving the camera somewhere else would only be
  // half the job. updateCamera also resets the accumulation, which is required
  // anyway since the image is no longer valid for the new parameters.
  const handleRestoreDefaults = useCallback(() => {
    setScene(DEFAULT_SCENE);
    rendererRef.current?.setCameraVelocity(0, 0);
    rendererRef.current?.updateCamera((camera) => {
      Object.assign(camera, DEFAULT_CAMERA);
    });
  }, []);

  const handleRunCheck = useCallback(() => {
    const renderer = rendererRef.current;
    if (!renderer) return;

    const run = async () => {
      setCheckStatus('running');
      setCheckError(null);
      try {
        setReport(await validatePhysicsOnGpu(renderer.device));
      } catch (error) {
        setCheckError(
          error instanceof Error ? error.message : 'The physics check failed.',
        );
      } finally {
        setCheckStatus('done');
      }
    };
    void run();
  }, []);

  if (status.kind === 'unsupported') {
    return <Unsupported reason={status.reason} />;
  }

  return (
    <main className="stage">
      <canvas
        ref={canvasRef}
        className="stage__canvas"
        aria-label="Kerr black hole render"
      />
      {status.kind === 'starting' ? (
        <p className="stage__booting">Starting the GPU…</p>
      ) : null}
      <Controls
        scene={scene}
        stats={stats}
        cameraRadius={cameraRadius}
        onNumericChange={handleNumericChange}
        onDiskToggle={handleDiskToggle}
        onShadingChange={handleShadingChange}
        onRestoreDefaults={handleRestoreDefaults}
        onShowHelp={showHelp}
      >
        <PhysicsCheck
          status={checkStatus}
          report={report}
          error={checkError}
          onRun={handleRunCheck}
        />
      </Controls>
      <button
        type="button"
        className="stage__help"
        onClick={showHelp}
        aria-label="Show controls"
        title="Controls (?)"
      >
        ?
      </button>
      {helpOpen ? <HelpPanel onClose={hideHelp} /> : null}
    </main>
  );
}
