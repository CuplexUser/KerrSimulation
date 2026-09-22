/**
 * A recording stand-in for WebGPU, enough to drive `KerrRenderer` under Node.
 *
 * None of it computes anything — the point is not to render, it is to observe
 * *scheduling*. Every uniform write and every pass is logged and grouped by
 * queue submission, so a test can read back exactly which rows each dispatch
 * covered, which accumulation texture was presented, and when the sample counter
 * advanced. That is where the renderer's subtle logic lives, and it is entirely
 * separable from whether a GPU exists.
 *
 * Resource objects are labelled the way the renderer labels them, so the
 * otherwise-opaque bind groups identify themselves in the log. `kerr-compute-
 * bindgroup-1`, for instance, is the one that reads accumulation texture 1 and
 * writes texture 0.
 *
 * The clock is fake and starts frozen. The renderer makes three decisions from
 * `performance.now()` — coarse-vs-full trace resolution, the adaptive band split,
 * and stats throttling — and a frozen clock pins all three so tests can move one
 * at a time.
 */

import { UNIFORM_FLOATS } from '../../src/gpu/uniforms.ts';

export type PassRecord = { label: string; bindGroup: string };

export type FrameRecord = {
  /** Snapshot of the uniform buffer as this frame was submitted. */
  uniforms: Float32Array;
  /** Null when the frame skipped tracing, i.e. converged or an empty band. */
  compute: { bindGroup: string; workgroups: [number, number] } | null;
  passes: PassRecord[];
};

type Labelled = { label?: string };

export type FakeGpu = {
  device: GPUDevice;
  canvas: HTMLCanvasElement;
  frames: FrameRecord[];
  /** Advance one animation frame. Returns false if the loop is not running. */
  tick: () => boolean;
  /** Advance the fake clock, in milliseconds. */
  advance: (ms: number) => void;
  /** Fire the observer the renderer registered on the canvas. */
  resizeCanvas: (width: number, height: number) => void;
  restore: () => void;
};

const BUFFER_USAGE = {
  MAP_READ: 1,
  MAP_WRITE: 2,
  COPY_SRC: 4,
  COPY_DST: 8,
  INDEX: 16,
  VERTEX: 32,
  UNIFORM: 64,
  STORAGE: 128,
  INDIRECT: 256,
  QUERY_RESOLVE: 512,
};

const TEXTURE_USAGE = {
  COPY_SRC: 1,
  COPY_DST: 2,
  TEXTURE_BINDING: 4,
  STORAGE_BINDING: 8,
  RENDER_ATTACHMENT: 16,
};

const SHADER_STAGE = { VERTEX: 1, FRAGMENT: 2, COMPUTE: 4 };

/** Pipelines only need to hand back a layout the bind groups can be built from. */
const fakePipeline = (label: string | undefined) => ({
  label,
  getBindGroupLayout: (index: number) => ({ label: `${label}-layout-${index}` }),
});

/** Replaces a global, leaving it configurable so it can be restored. */
const define = (name: string, value: unknown): void => {
  Object.defineProperty(globalThis, name, {
    value,
    writable: true,
    configurable: true,
  });
};

export function createFakeGpu(
  options: { width?: number; height?: number; devicePixelRatio?: number } = {},
): FakeGpu {
  let cssWidth = options.width ?? 800;
  let cssHeight = options.height ?? 600;

  const frames: FrameRecord[] = [];
  let uniforms = new Float32Array(UNIFORM_FLOATS);
  let pending: FrameRecord | null = null;

  const frame = (): FrameRecord => {
    pending ??= { uniforms: new Float32Array(UNIFORM_FLOATS), compute: null, passes: [] };
    return pending;
  };

  // -- clock ----------------------------------------------------------------

  let now = 0;
  const realPerformance = globalThis.performance;

  // -- animation frames -----------------------------------------------------

  let scheduled: (() => void) | null = null;
  const realRaf = globalThis.requestAnimationFrame;
  const realCancelRaf = globalThis.cancelAnimationFrame;

  // -- resize observation ---------------------------------------------------

  let observerCallback: (() => void) | null = null;
  const realResizeObserver = globalThis.ResizeObserver;
  const realDpr = globalThis.devicePixelRatio;

  // -- device ---------------------------------------------------------------

  const encoder = {
    label: 'encoder',
    beginComputePass(descriptor: Labelled = {}) {
      const record = frame();
      let bindGroup = '';
      let workgroups: [number, number] = [0, 0];
      return {
        label: descriptor.label,
        setPipeline() {},
        setBindGroup(_index: number, group: Labelled) {
          bindGroup = group.label ?? '';
        },
        dispatchWorkgroups(x: number, y = 1) {
          workgroups = [x, y];
        },
        end() {
          record.compute = { bindGroup, workgroups };
        },
      };
    },
    beginRenderPass(descriptor: Labelled = {}) {
      const record = frame();
      const pass: PassRecord = { label: descriptor.label ?? '', bindGroup: '' };
      return {
        setPipeline() {},
        setBindGroup(_index: number, group: Labelled) {
          pass.bindGroup = group.label ?? '';
        },
        setBlendConstant() {},
        draw() {},
        end() {
          record.passes.push(pass);
        },
      };
    },
    copyBufferToBuffer() {},
    finish() {
      return { label: 'command-buffer' };
    },
  };

  const device = {
    label: 'fake-device',
    limits: { maxTextureDimension2D: 8192 },
    features: new Set<string>(),

    createShaderModule: (descriptor: Labelled & { code: string }) => ({
      label: descriptor.label,
      code: descriptor.code,
    }),
    createBindGroupLayout: (descriptor: Labelled) => ({ label: descriptor.label }),
    createPipelineLayout: (descriptor: Labelled) => ({ label: descriptor.label }),
    createBindGroup: (descriptor: Labelled) => ({ label: descriptor.label }),
    createSampler: (descriptor: Labelled = {}) => ({ label: descriptor.label }),

    createComputePipelineAsync: (descriptor: Labelled) =>
      Promise.resolve(fakePipeline(descriptor.label)),
    createRenderPipelineAsync: (descriptor: Labelled) =>
      Promise.resolve(fakePipeline(descriptor.label)),

    createBuffer: (descriptor: Labelled & { size: number }) => ({
      label: descriptor.label,
      size: descriptor.size,
      destroy() {},
    }),

    createTexture: (
      descriptor: Labelled & { size: { width: number; height: number } },
    ) => ({
      label: descriptor.label,
      width: descriptor.size.width,
      height: descriptor.size.height,
      createView: () => ({ label: `${descriptor.label}-view` }),
      destroy() {},
    }),

    createCommandEncoder: (descriptor: Labelled = {}) => ({
      ...encoder,
      label: descriptor.label,
    }),

    queue: {
      writeBuffer(_buffer: unknown, _offset: number, data: Float32Array) {
        uniforms = new Float32Array(data);
      },
      submit() {
        const record = frame();
        record.uniforms = new Float32Array(uniforms);
        frames.push(record);
        pending = null;
      },
    },
  };

  // -- canvas ---------------------------------------------------------------

  const context = {
    configure() {},
    unconfigure() {},
    getCurrentTexture: () => ({ createView: () => ({ label: 'swapchain-view' }) }),
  };

  const canvas = {
    width: 0,
    height: 0,
    getContext: (kind: string) => (kind === 'webgpu' ? context : null),
    getBoundingClientRect: () => ({
      width: cssWidth,
      height: cssHeight,
      top: 0,
      left: 0,
      right: cssWidth,
      bottom: cssHeight,
      x: 0,
      y: 0,
    }),
  };

  // -- globals --------------------------------------------------------------

  const realNavigatorGpu = Reflect.get(globalThis.navigator ?? {}, 'gpu');

  function install(): void {
    define('GPUBufferUsage', BUFFER_USAGE);
    define('GPUTextureUsage', TEXTURE_USAGE);
    define('GPUShaderStage', SHADER_STAGE);
    define('performance', { now: () => now });
    define('devicePixelRatio', options.devicePixelRatio ?? 1);

    define('requestAnimationFrame', (callback: () => void) => {
      scheduled = callback;
      return 1;
    });
    define('cancelAnimationFrame', () => {
      scheduled = null;
    });

    define(
      'ResizeObserver',
      class {
        constructor(callback: () => void) {
          observerCallback = callback;
        }
        observe() {}
        unobserve() {}
        disconnect() {
          observerCallback = null;
        }
      },
    );

    Object.defineProperty(globalThis.navigator, 'gpu', {
      value: { getPreferredCanvasFormat: () => 'bgra8unorm' },
      writable: true,
      configurable: true,
    });
  }

  function restore(): void {
    define('performance', realPerformance);
    define('devicePixelRatio', realDpr);
    define('requestAnimationFrame', realRaf);
    define('cancelAnimationFrame', realCancelRaf);
    define('ResizeObserver', realResizeObserver);
    Object.defineProperty(globalThis.navigator, 'gpu', {
      value: realNavigatorGpu,
      writable: true,
      configurable: true,
    });
  }

  install();

  return {
    device: device as unknown as GPUDevice,
    canvas: canvas as unknown as HTMLCanvasElement,
    frames,

    tick() {
      const callback = scheduled;
      if (!callback) return false;
      scheduled = null;
      callback();
      return true;
    },

    advance(ms: number) {
      now += ms;
    },

    resizeCanvas(width: number, height: number) {
      cssWidth = width;
      cssHeight = height;
      observerCallback?.();
    },

    restore,
  };
}
