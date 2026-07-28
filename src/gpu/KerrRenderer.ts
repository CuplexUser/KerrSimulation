/**
 * Owns the WebGPU device state and the render loop.
 *
 * Per frame:
 *   1. compute pass  — one jittered sample per pixel, blended into the running
 *                      average (reads accumulation A, writes accumulation B)
 *   2. render pass   — tone maps B onto the swap chain
 *   3. swap A and B
 *
 * The loop lives here rather than in React on purpose: camera drags run at
 * pointer-event rate and must not trigger re-renders. React pushes parameters in;
 * a throttled callback reports statistics back out.
 */

import { iscoRadius, horizonRadius } from '../physics/kerrReference.ts';
import { cameraBasis, type CameraState, DEFAULT_CAMERA } from './camera.ts';
import { presentShader, traceShader } from './shaders/index.ts';
import { UNIFORM_BYTES, UNIFORM_FLOATS, packUniforms, type RenderParams } from './uniforms.ts';

export type SceneParams = {
  spin: number;
  diskOuterRadius: number;
  diskEnabled: boolean;
  resolutionScale: number;
  maxSteps: number;
  exposure: number;
};

export const DEFAULT_SCENE: SceneParams = {
  spin: 0.85,
  diskOuterRadius: 14,
  diskEnabled: true,
  resolutionScale: 1,
  maxSteps: 450,
  exposure: 1.2,
};

/** Past this the image has converged; stop dispatching and just re-present. */
export const MAX_ACCUMULATED_SAMPLES = 1024;

/**
 * While the camera is moving, trace at this fraction of the full resolution.
 * The first frame after any camera change is a single noisy sample regardless,
 * so spending full resolution on it buys nothing and costs interactivity.
 */
const INTERACTIVE_SCALE = 0.5;

/** How long after the last camera change to stay in the coarse mode. */
const INTERACTION_IDLE_MS = 180;

export type RendererStats = {
  samples: number;
  width: number;
  height: number;
  iscoRadius: number;
  horizonRadius: number;
  converged: boolean;
  interacting: boolean;
};

const ACCUMULATION_FORMAT: GPUTextureFormat = 'rgba16float';

export class KerrRenderer {
  readonly device: GPUDevice;

  #canvas: HTMLCanvasElement;
  #context: GPUCanvasContext;
  #canvasFormat: GPUTextureFormat;

  #computePipeline: GPUComputePipeline;
  #presentPipeline: GPURenderPipeline;
  #uniformBuffer: GPUBuffer;
  #uniformScratch = new Float32Array(UNIFORM_FLOATS);

  /** Double-buffered accumulation. Index 0 is read, index 1 is written. */
  #accumTextures: GPUTexture[] = [];
  #computeBindGroups: GPUBindGroup[] = [];
  #presentBindGroups: GPUBindGroup[] = [];
  #sampler: GPUSampler;
  #pingPong = 0;

  #camera: CameraState = { ...DEFAULT_CAMERA };
  #scene: SceneParams = { ...DEFAULT_SCENE };

  #frameIndex = 0;
  /** Resolution the compute pass traces at. */
  #width = 1;
  #height = 1;
  /** Swap-chain resolution, which the trace resolution is a fraction of. */
  #canvasWidth = 1;
  #canvasHeight = 1;
  #interacting = false;
  #lastInteractionAt = 0;
  #rafHandle = 0;
  #disposed = false;
  #resizeObserver: ResizeObserver | null = null;

  #onStats: ((stats: RendererStats) => void) | null = null;
  #lastStatsAt = 0;

  private constructor(
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    context: GPUCanvasContext,
    canvasFormat: GPUTextureFormat,
    computePipeline: GPUComputePipeline,
    presentPipeline: GPURenderPipeline,
  ) {
    this.device = device;
    this.#canvas = canvas;
    this.#context = context;
    this.#canvasFormat = canvasFormat;
    this.#computePipeline = computePipeline;
    this.#presentPipeline = presentPipeline;

    this.#uniformBuffer = device.createBuffer({
      label: 'kerr-uniforms',
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    // Linear so the coarse interactive pass upscales smoothly instead of
    // showing blocky texels while you drag.
    this.#sampler = device.createSampler({
      label: 'kerr-accumulation-sampler',
      magFilter: 'linear',
      minFilter: 'linear',
    });
  }

  static async create(
    device: GPUDevice,
    canvas: HTMLCanvasElement,
  ): Promise<KerrRenderer> {
    const context = canvas.getContext('webgpu');
    if (!context) {
      throw new Error('Could not acquire a WebGPU canvas context.');
    }

    const canvasFormat = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device,
      format: canvasFormat,
      alphaMode: 'opaque',
    });

    const traceModule = device.createShaderModule({
      label: 'kerr-trace',
      code: traceShader,
    });
    const presentModule = device.createShaderModule({
      label: 'kerr-present',
      code: presentShader,
    });

    const [computePipeline, presentPipeline] = await Promise.all([
      device.createComputePipelineAsync({
        label: 'kerr-trace-pipeline',
        layout: 'auto',
        compute: { module: traceModule, entryPoint: 'main' },
      }),
      device.createRenderPipelineAsync({
        label: 'kerr-present-pipeline',
        layout: 'auto',
        vertex: { module: presentModule, entryPoint: 'vs' },
        fragment: {
          module: presentModule,
          entryPoint: 'fs',
          targets: [{ format: canvasFormat }],
        },
        primitive: { topology: 'triangle-list' },
      }),
    ]);

    const renderer = new KerrRenderer(
      device,
      canvas,
      context,
      canvasFormat,
      computePipeline,
      presentPipeline,
    );
    renderer.#observeResize();
    return renderer;
  }

  // -------------------------------------------------------------------------
  // Public surface
  // -------------------------------------------------------------------------

  get camera(): CameraState {
    return this.#camera;
  }

  get canvasFormat(): GPUTextureFormat {
    return this.#canvasFormat;
  }

  /** Discards accumulated samples; the next dispatch fully overwrites. */
  resetAccumulation(): void {
    this.#frameIndex = 0;
  }

  /**
   * Mutating the camera in place is intentional — orbit drags update this at
   * pointer-event rate and must not go through React state.
   */
  updateCamera(mutate: (camera: CameraState) => void): void {
    mutate(this.#camera);
    this.resetAccumulation();

    // Drop to the coarse trace resolution for the duration of the gesture. The
    // frame loop restores full resolution once the camera has been still for
    // INTERACTION_IDLE_MS.
    this.#lastInteractionAt = performance.now();
    if (!this.#interacting) {
      this.#interacting = true;
      this.#resize();
    }
  }

  setScene(scene: Partial<SceneParams>): void {
    const previous = this.#scene;
    const next = { ...previous, ...scene };
    this.#scene = next;

    if (next.resolutionScale !== previous.resolutionScale) {
      this.#resize();
      return;
    }

    // Exposure is applied at tone-map time, so changing it must not throw away
    // samples that are already converged. Everything else changes what the rays
    // actually do.
    const affectsTracing =
      next.spin !== previous.spin ||
      next.diskOuterRadius !== previous.diskOuterRadius ||
      next.diskEnabled !== previous.diskEnabled ||
      next.maxSteps !== previous.maxSteps;

    if (affectsTracing) {
      this.resetAccumulation();
    }
  }

  onStats(callback: (stats: RendererStats) => void): void {
    this.#onStats = callback;
  }

  start(): void {
    if (this.#rafHandle !== 0 || this.#disposed) return;
    const tick = () => {
      if (this.#disposed) return;
      this.#renderFrame();
      this.#rafHandle = requestAnimationFrame(tick);
    };
    this.#rafHandle = requestAnimationFrame(tick);
  }

  dispose(): void {
    this.#disposed = true;
    if (this.#rafHandle !== 0) {
      cancelAnimationFrame(this.#rafHandle);
      this.#rafHandle = 0;
    }
    this.#resizeObserver?.disconnect();
    this.#resizeObserver = null;
    for (const texture of this.#accumTextures) texture.destroy();
    this.#accumTextures = [];
    this.#uniformBuffer.destroy();
  }

  // -------------------------------------------------------------------------
  // Sizing
  // -------------------------------------------------------------------------

  #observeResize(): void {
    this.#resizeObserver = new ResizeObserver(() => this.#resize());
    this.#resizeObserver.observe(this.#canvas);
    this.#resize();
  }

  #resize(): void {
    const dpr = Math.min(globalThis.devicePixelRatio || 1, 2);
    const rect = this.#canvas.getBoundingClientRect();
    const limit = this.device.limits.maxTextureDimension2D;

    const clamp = (value: number) =>
      Math.max(1, Math.min(Math.floor(value), limit));

    // The canvas always stays at full device resolution; only the traced image
    // shrinks. That keeps the swap chain stable while the trace resolution
    // changes underneath it.
    const canvasWidth = clamp(rect.width * dpr);
    const canvasHeight = clamp(rect.height * dpr);

    const scale =
      this.#scene.resolutionScale * (this.#interacting ? INTERACTIVE_SCALE : 1);
    const width = clamp(canvasWidth * scale);
    const height = clamp(canvasHeight * scale);

    if (
      width === this.#width &&
      height === this.#height &&
      canvasWidth === this.#canvasWidth &&
      canvasHeight === this.#canvasHeight &&
      this.#accumTextures.length > 0
    ) {
      return;
    }

    this.#width = width;
    this.#height = height;
    this.#canvasWidth = canvasWidth;
    this.#canvasHeight = canvasHeight;
    this.#canvas.width = canvasWidth;
    this.#canvas.height = canvasHeight;

    for (const texture of this.#accumTextures) texture.destroy();

    this.#accumTextures = [0, 1].map((i) =>
      this.device.createTexture({
        label: `kerr-accumulation-${i}`,
        size: { width, height },
        format: ACCUMULATION_FORMAT,
        usage:
          GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      }),
    );

    // Two bind groups, pre-built: A reads 0 and writes 1, B reads 1 and writes 0.
    // Swapping an index each frame keeps the loop allocation-free.
    this.#computeBindGroups = [0, 1].map((i) =>
      this.device.createBindGroup({
        label: `kerr-compute-bindgroup-${i}`,
        layout: this.#computePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.#uniformBuffer } },
          { binding: 1, resource: this.#accumTextures[i].createView() },
          { binding: 2, resource: this.#accumTextures[1 - i].createView() },
        ],
      }),
    );

    this.#presentBindGroups = [0, 1].map((i) =>
      this.device.createBindGroup({
        label: `kerr-present-bindgroup-${i}`,
        layout: this.#presentPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.#accumTextures[i].createView() },
          { binding: 1, resource: { buffer: this.#uniformBuffer } },
          { binding: 2, resource: this.#sampler },
        ],
      }),
    );

    this.resetAccumulation();
  }

  // -------------------------------------------------------------------------
  // Frame
  // -------------------------------------------------------------------------

  #renderParams(): RenderParams {
    return {
      spin: this.#scene.spin,
      diskOuterRadius: this.#scene.diskOuterRadius,
      iscoRadius: iscoRadius(this.#scene.spin),
      horizonRadius: horizonRadius(this.#scene.spin),
      diskEnabled: this.#scene.diskEnabled,
      maxSteps: this.#scene.maxSteps,
      exposure: this.#scene.exposure,
    };
  }

  #renderFrame(): void {
    if (this.#accumTextures.length < 2) return;

    // Camera has been still long enough — go back to full resolution.
    if (
      this.#interacting &&
      performance.now() - this.#lastInteractionAt > INTERACTION_IDLE_MS
    ) {
      this.#interacting = false;
      this.#resize();
    }

    const converged = this.#frameIndex >= MAX_ACCUMULATED_SAMPLES;

    packUniforms(this.#uniformScratch, {
      basis: cameraBasis(this.#camera),
      params: this.#renderParams(),
      frameIndex: this.#frameIndex,
      width: this.#width,
      height: this.#height,
      canvasWidth: this.#canvasWidth,
      canvasHeight: this.#canvasHeight,
    });
    this.device.queue.writeBuffer(this.#uniformBuffer, 0, this.#uniformScratch);

    const encoder = this.device.createCommandEncoder({ label: 'kerr-frame' });

    // The texture holding the newest accumulation. Normally the compute pass is
    // about to write it; once converged we simply present what is already there.
    let presentIndex = this.#pingPong;

    if (!converged) {
      const pass = encoder.beginComputePass({ label: 'kerr-trace-pass' });
      pass.setPipeline(this.#computePipeline);
      pass.setBindGroup(0, this.#computeBindGroups[this.#pingPong]);
      pass.dispatchWorkgroups(
        Math.ceil(this.#width / 8),
        Math.ceil(this.#height / 8),
      );
      pass.end();
      presentIndex = 1 - this.#pingPong;
    }

    const renderPass = encoder.beginRenderPass({
      label: 'kerr-present-pass',
      colorAttachments: [
        {
          view: this.#context.getCurrentTexture().createView(),
          loadOp: 'clear',
          storeOp: 'store',
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
        },
      ],
    });
    renderPass.setPipeline(this.#presentPipeline);
    renderPass.setBindGroup(0, this.#presentBindGroups[presentIndex]);
    renderPass.draw(3);
    renderPass.end();

    this.device.queue.submit([encoder.finish()]);

    if (!converged) {
      this.#pingPong = 1 - this.#pingPong;
      this.#frameIndex++;
    }

    this.#reportStats(converged);
  }

  /** Throttled so the panel updates without re-rendering React every frame. */
  #reportStats(converged: boolean): void {
    const now = performance.now();
    if (now - this.#lastStatsAt < 250) return;
    this.#lastStatsAt = now;

    this.#onStats?.({
      samples: this.#frameIndex,
      width: this.#width,
      height: this.#height,
      iscoRadius: iscoRadius(this.#scene.spin),
      horizonRadius: horizonRadius(this.#scene.spin),
      converged,
      interacting: this.#interacting,
    });
  }
}
