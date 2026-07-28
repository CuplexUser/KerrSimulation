/** WebGPU adapter/device acquisition, with the failure modes spelled out. */

export type GpuContext = {
  adapter: GPUAdapter;
  device: GPUDevice;
};

export type GpuInitResult =
  | { ok: true; context: GpuContext }
  | { ok: false; reason: string };

export async function initWebGPU(): Promise<GpuInitResult> {
  if (!navigator.gpu) {
    return {
      ok: false,
      reason:
        'This browser does not expose the WebGPU API. Chrome or Edge 113+ on a machine with a hardware GPU is required.',
    };
  }

  let adapter: GPUAdapter | null = null;
  try {
    adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });
  } catch (error) {
    return { ok: false, reason: `Requesting a GPU adapter failed: ${describe(error)}` };
  }

  if (!adapter) {
    return {
      ok: false,
      reason:
        'No WebGPU adapter is available. The GPU may be blocklisted, or hardware acceleration may be disabled in browser settings.',
    };
  }

  try {
    const device = await adapter.requestDevice({ label: 'kerr-device' });
    return { ok: true, context: { adapter, device } };
  } catch (error) {
    return { ok: false, reason: `Requesting a GPU device failed: ${describe(error)}` };
  }
}

const describe = (error: unknown): string =>
  error instanceof Error ? error.message : String(error);
