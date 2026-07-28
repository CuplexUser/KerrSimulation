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
import { bloomShader, presentShader, traceShader } from './shaders/index.ts';
import { UNIFORM_BYTES, UNIFORM_FLOATS, packUniforms, type RenderParams } from './uniforms.ts';

export type SceneParams = {
  spin: number;
  diskOuterRadius: number;
  diskEnabled: boolean;
  resolutionScale: number;
  maxSteps: number;
  exposure: number;
  bloomThreshold: number;
  bloomStrength: number;
};

export const DEFAULT_SCENE: SceneParams = {
  spin: 0.85,
  diskOuterRadius: 14,
  diskEnabled: true,
  // Not 1.0: integrated geodesics are expensive enough that full device
  // resolution is punishing on integrated GPUs. The image still converges to a
  // clean result while the camera is still — raise this if your GPU has room.
  resolutionScale: 0.75,
  maxSteps: 450,
  exposure: 1.2,
  // High enough that only the disk and the brightest stars glow. Lower it and
  // the whole starfield blooms, which mostly just reveals the grid the stars sit on.
  bloomThreshold: 1.1,
  bloomStrength: 0.9,
};

/** Bloom runs at this fraction of the canvas — a wide, soft glow needs no detail. */
const BLOOM_SCALE = 0.25;

/** Past this the image has converged; stop dispatching and just re-present. */
export const MAX_ACCUMULATED_SAMPLES = 1024;

/**
 * While the camera is moving, trace at this fraction of the full resolution —
 * about a sixth of the pixels. The first frame after any camera change is a
 * single noisy sample regardless, so spending full resolution on it buys nothing
 * and costs interactivity.
 *
 * This matters more than it looks: each RK4 step evaluates the metric ~28 times
 * (four stages, each taking a central difference over three axes), so the shader
 * is heavy enough that pixel count dominates everything else.
 */
const INTERACTIVE_SCALE = 0.4;

/** How long after the last camera change to stay in the coarse mode. */
const INTERACTION_IDLE_MS = 180;

/**
 * Frame-time budget for the adaptive band controller, in milliseconds.
 *
 * Integrating a whole image in a single dispatch can occupy the GPU for close to
 * a second on an integrated part. A submission that long stalls compositing and
 * input handling for the entire tab — the page reads as frozen and slider drags
 * queue up behind it. Splitting one sample across several short dispatches keeps
 * total throughput the same while leaving the browser responsive between them.
 */
const TARGET_FRAME_MS = 16;
const FRAME_MS_TOO_SLOW = 24;
const FRAME_MS_TOO_FAST = 10;
const MAX_BANDS = 256;

export type RendererStats = {
  samples: number;
  width: number;
  height: number;
  iscoRadius: number;
  horizonRadius: number;
  converged: boolean;
  interacting: boolean;
  /** Dispatches per sample, auto-tuned to keep the browser responsive. */
  bandCount: number;
};

const ACCUMULATION_FORMAT: GPUTextureFormat = 'rgba16float';

type BloomPipelines = {
  bright: GPURenderPipeline;
  blurH: GPURenderPipeline;
  blurV: GPURenderPipeline;
  /** Shared across all three stages, so one bind group serves any of them. */
  bindGroupLayout: GPUBindGroupLayout;
};

export class KerrRenderer {
  readonly device: GPUDevice;

  #canvas: HTMLCanvasElement;
  #context: GPUCanvasContext;
  #canvasFormat: GPUTextureFormat;

  #computePipeline: GPUComputePipeline;
  #presentPipeline: GPURenderPipeline;
  #bloomPipelines: BloomPipelines;
  #uniformBuffer: GPUBuffer;
  #uniformScratch = new Float32Array(UNIFORM_FLOATS);

  /** Double-buffered accumulation. Index 0 is read, index 1 is written. */
  #accumTextures: GPUTexture[] = [];
  #computeBindGroups: GPUBindGroup[] = [];
  #presentBindGroups: GPUBindGroup[] = [];
  #sampler: GPUSampler;
  #pingPong = 0;

  /** Bloom chain: [0] holds the bright pass and the final blur, [1] is scratch. */
  #bloomTextures: GPUTexture[] = [];
  #bloomBindGroups: GPUBindGroup[] = [];
  #brightBindGroups: GPUBindGroup[] = [];
  #bloomWidth = 1;
  #bloomHeight = 1;

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

  /** One sample is traced as `#bandCount` successive dispatches. */
  #bandCount = 8;
  #bandIndex = 0;
  #frameMsEma = TARGET_FRAME_MS;
  #lastFrameAt = 0;

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
    bloomPipelines: BloomPipelines,
  ) {
    this.device = device;
    this.#canvas = canvas;
    this.#context = context;
    this.#canvasFormat = canvasFormat;
    this.#computePipeline = computePipeline;
    this.#presentPipeline = presentPipeline;
    this.#bloomPipelines = bloomPipelines;

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
    const bloomModule = device.createShaderModule({
      label: 'kerr-bloom',
      code: bloomShader,
    });

    // An explicit layout, shared by all three bloom stages. `layout: 'auto'`
    // mints a fresh bind group layout per pipeline that is deliberately not
    // interchangeable with any other, so a bind group built for the horizontal
    // blur would be rejected by the vertical one.
    const bloomBindGroupLayout = device.createBindGroupLayout({
      label: 'kerr-bloom-bindgroup-layout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'float' },
        },
        {
          binding: 1,
          visibility: GPUShaderStage.FRAGMENT,
          buffer: { type: 'uniform' },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.FRAGMENT,
          sampler: { type: 'filtering' },
        },
      ],
    });
    const bloomPipelineLayout = device.createPipelineLayout({
      label: 'kerr-bloom-pipeline-layout',
      bindGroupLayouts: [bloomBindGroupLayout],
    });

    const bloomStage = (label: string, entryPoint: string) =>
      device.createRenderPipelineAsync({
        label,
        layout: bloomPipelineLayout,
        vertex: { module: bloomModule, entryPoint: 'vs' },
        fragment: {
          module: bloomModule,
          entryPoint,
          targets: [{ format: ACCUMULATION_FORMAT }],
        },
        primitive: { topology: 'triangle-list' },
      });

    const [computePipeline, presentPipeline, bright, blurH, blurV] =
      await Promise.all([
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
        bloomStage('kerr-bloom-bright', 'fsBright'),
        bloomStage('kerr-bloom-blur-h', 'fsBlurH'),
        bloomStage('kerr-bloom-blur-v', 'fsBlurV'),
      ]);

    const renderer = new KerrRenderer(
      device,
      canvas,
      context,
      canvasFormat,
      computePipeline,
      presentPipeline,
      { bright, blurH, blurV, bindGroupLayout: bloomBindGroupLayout },
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

  /** Discards accumulated samples; the next pass fully overwrites. */
  resetAccumulation(): void {
    this.#frameIndex = 0;
    this.#bandIndex = 0;
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
    // Exposure and bloom are applied at present time, so they must not throw
    // away samples that are already converged.
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
    for (const texture of this.#bloomTextures) texture.destroy();
    this.#accumTextures = [];
    this.#bloomTextures = [];
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

    this.#buildBloomResources();

    this.#presentBindGroups = [0, 1].map((i) =>
      this.device.createBindGroup({
        label: `kerr-present-bindgroup-${i}`,
        layout: this.#presentPipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: this.#accumTextures[i].createView() },
          { binding: 1, resource: { buffer: this.#uniformBuffer } },
          { binding: 2, resource: this.#sampler },
          { binding: 3, resource: this.#bloomTextures[0].createView() },
        ],
      }),
    );

    this.resetAccumulation();
  }

  #buildBloomResources(): void {
    for (const texture of this.#bloomTextures) texture.destroy();

    this.#bloomWidth = Math.max(1, Math.floor(this.#canvasWidth * BLOOM_SCALE));
    this.#bloomHeight = Math.max(
      1,
      Math.floor(this.#canvasHeight * BLOOM_SCALE),
    );

    this.#bloomTextures = [0, 1].map((i) =>
      this.device.createTexture({
        label: `kerr-bloom-${i}`,
        size: { width: this.#bloomWidth, height: this.#bloomHeight },
        format: ACCUMULATION_FORMAT,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      }),
    );

    // Bright pass reads whichever accumulation texture is being presented.
    this.#brightBindGroups = [0, 1].map((i) =>
      this.device.createBindGroup({
        label: `kerr-bright-bindgroup-${i}`,
        layout: this.#bloomPipelines.bindGroupLayout,
        entries: [
          { binding: 0, resource: this.#accumTextures[i].createView() },
          { binding: 1, resource: { buffer: this.#uniformBuffer } },
          { binding: 2, resource: this.#sampler },
        ],
      }),
    );

    // Blur passes read one bloom texture and write the other.
    this.#bloomBindGroups = [0, 1].map((i) =>
      this.device.createBindGroup({
        label: `kerr-bloom-bindgroup-${i}`,
        layout: this.#bloomPipelines.bindGroupLayout,
        entries: [
          { binding: 0, resource: this.#bloomTextures[i].createView() },
          { binding: 1, resource: { buffer: this.#uniformBuffer } },
          { binding: 2, resource: this.#sampler },
        ],
      }),
    );
  }

  /**
   * Bright-pass, blur across, blur down. Runs every frame off the presented
   * accumulation texture; three passes at a sixteenth of the pixels is nothing
   * next to integrating geodesics.
   */
  #encodeBloom(encoder: GPUCommandEncoder, presentIndex: number): void {
    const stage = (
      label: string,
      pipeline: GPURenderPipeline,
      bindGroup: GPUBindGroup,
      target: GPUTexture,
    ) => {
      const pass = encoder.beginRenderPass({
        label,
        colorAttachments: [
          {
            view: target.createView(),
            loadOp: 'clear',
            storeOp: 'store',
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
          },
        ],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.draw(3);
      pass.end();
    };

    stage(
      'kerr-bloom-bright',
      this.#bloomPipelines.bright,
      this.#brightBindGroups[presentIndex],
      this.#bloomTextures[0],
    );
    stage(
      'kerr-bloom-blur-h',
      this.#bloomPipelines.blurH,
      this.#bloomBindGroups[0],
      this.#bloomTextures[1],
    );
    // Lands back in texture 0, which is what the present pass samples.
    stage(
      'kerr-bloom-blur-v',
      this.#bloomPipelines.blurV,
      this.#bloomBindGroups[1],
      this.#bloomTextures[0],
    );
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
      bloomThreshold: this.#scene.bloomThreshold,
      bloomStrength: this.#scene.bloomStrength,
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

    this.#trackFrameTime();

    const converged = this.#frameIndex >= MAX_ACCUMULATED_SAMPLES;

    // Rows covered by this dispatch. The last band absorbs the remainder.
    const bandHeight = Math.ceil(this.#height / this.#bandCount);
    const bandOffset = this.#bandIndex * bandHeight;
    const bandRows = Math.min(bandHeight, this.#height - bandOffset);

    packUniforms(this.#uniformScratch, {
      basis: cameraBasis(this.#camera),
      params: this.#renderParams(),
      frameIndex: this.#frameIndex,
      width: this.#width,
      height: this.#height,
      canvasWidth: this.#canvasWidth,
      canvasHeight: this.#canvasHeight,
      bandOffset,
      bandHeight: bandRows,
      bloomWidth: this.#bloomWidth,
      bloomHeight: this.#bloomHeight,
    });
    this.device.queue.writeBuffer(this.#uniformBuffer, 0, this.#uniformScratch);

    const encoder = this.device.createCommandEncoder({ label: 'kerr-frame' });

    if (!converged && bandRows > 0) {
      const pass = encoder.beginComputePass({ label: 'kerr-trace-pass' });
      pass.setPipeline(this.#computePipeline);
      pass.setBindGroup(0, this.#computeBindGroups[this.#pingPong]);
      pass.dispatchWorkgroups(
        Math.ceil(this.#width / 8),
        Math.ceil(bandRows / 8),
      );
      pass.end();
    }

    // #pingPong indexes the texture holding the last *complete* accumulation, so
    // present it and the image never shows a half-finished pass. The exception is
    // the very first pass after a reset, where there is no previous image worth
    // showing — there, present the in-progress texture so bands appear as they
    // land instead of leaving the viewer staring at the pre-reset frame.
    const destination = 1 - this.#pingPong;
    const presentIndex = this.#frameIndex === 0 ? destination : this.#pingPong;

    this.#encodeBloom(encoder, presentIndex);

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
      this.#bandIndex++;
      if (this.#bandIndex >= this.#bandCount) {
        // Pass complete: the destination now holds a whole sample.
        this.#bandIndex = 0;
        this.#pingPong = destination;
        this.#frameIndex++;
        this.#adaptBandCount();
      }
    }

    this.#reportStats(converged);
  }

  #trackFrameTime(): void {
    const now = performance.now();
    if (this.#lastFrameAt !== 0) {
      const delta = now - this.#lastFrameAt;
      // Ignore long gaps from a backgrounded tab, which would otherwise spike
      // the band count on return.
      if (delta < 500) {
        this.#frameMsEma = this.#frameMsEma * 0.85 + delta * 0.15;
      }
    }
    this.#lastFrameAt = now;
  }

  /**
   * Retune the split so each dispatch stays near the frame budget. Only ever
   * called between passes, since changing the band count mid-pass would leave
   * rows either traced twice or not at all.
   */
  #adaptBandCount(): void {
    if (this.#frameMsEma > FRAME_MS_TOO_SLOW && this.#bandCount < MAX_BANDS) {
      this.#bandCount = Math.min(MAX_BANDS, Math.ceil(this.#bandCount * 1.5));
      this.#frameMsEma = TARGET_FRAME_MS;
    } else if (this.#frameMsEma < FRAME_MS_TOO_FAST && this.#bandCount > 1) {
      this.#bandCount = Math.max(1, Math.floor(this.#bandCount / 1.5));
      this.#frameMsEma = TARGET_FRAME_MS;
    }
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
      bandCount: this.#bandCount,
    });
  }
}
