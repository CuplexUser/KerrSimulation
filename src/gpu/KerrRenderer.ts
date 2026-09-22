/**
 * Owns the WebGPU device state and the render loop.
 *
 * Per frame:
 *   1. compute pass  — one jittered sample per pixel for one band of the image,
 *                      blended into the running average (reads the displayed
 *                      accumulation texture, writes the other one)
 *   2. bloom chain   — bright pass, downsample, upsample
 *   3. render pass   — tone maps the last *complete* image onto the swap chain
 *
 * The loop lives here rather than in React on purpose: camera drags run at
 * pointer-event rate and must not trigger re-renders. React pushes parameters in;
 * a throttled callback reports statistics back out.
 *
 * Smoothness rules, each of which fixes an artifact that used to be visible:
 *
 *   - The screen only ever shows a complete pass. A change mid-pass abandons it
 *     and starts a new one in the *other* texture, while the old image stays up.
 *     Showing a pass while it was in flight let new bands tear against stale or
 *     black rows.
 *   - The accumulation textures are allocated once, at full trace size. The
 *     coarse interactive trace uses a region of them, so starting and ending a
 *     drag never reallocates — reallocating used to flash black.
 *   - While anything is moving, a whole coarse image is traced every frame, and
 *     its resolution adapts to hold the frame rate. Splitting a moving image
 *     into bands meant only the top band was ever refreshed.
 *   - Input moves a target camera; the drawn camera eases toward it, so wheel
 *     notches and mouse jitter turn into continuous motion.
 */

import { blackbodyXYZ, pageThorneFluxPeak } from '../physics/diskReference.ts';
import { iscoRadius, horizonRadius } from '../physics/kerrReference.ts';
import {
  cameraBasis,
  clampElevation,
  type CameraState,
  DEFAULT_CAMERA,
  easeCamera,
} from './camera.ts';
import { bloomShader, presentShader, traceShader } from './shaders/index.ts';
import {
  UNIFORM_BYTES,
  UNIFORM_FLOATS,
  packUniforms,
  type RenderParams,
  type Shading,
} from './uniforms.ts';

export type SceneParams = {
  spin: number;
  diskOuterRadius: number;
  diskEnabled: boolean;
  resolutionScale: number;
  maxSteps: number;
  exposure: number;
  bloomThreshold: number;
  bloomStrength: number;
  dopplerBeaming: number;
  diskThickness: number;
  /** Cinematic is the stylized film look; Physical is Novikov-Thorne + exact g. */
  shading: Shading;
  /** Physical mode: emitted temperature at the peak of the flux profile, K. */
  peakTemperature: number;
};

export const DEFAULT_SCENE: SceneParams = {
  spin: 0.85,
  diskOuterRadius: 20,
  diskEnabled: true,
  // Not 1.0: integrated geodesics are expensive enough that full device
  // resolution is punishing on integrated GPUs. The image still converges to a
  // clean result while the camera is still — raise this if your GPU has room.
  resolutionScale: 0.75,
  maxSteps: 450,
  exposure: 1.3,
  // High enough that only the disk and the brightest stars glow. Lower it and
  // the whole starfield blooms.
  // Bloom is additive, so a bright ring around a dark shadow bleeds inward.
  // Keep the threshold high and the strength moderate or the shadow — the one
  // thing that must stay black — fills with haze.
  bloomThreshold: 0.95,
  bloomStrength: 0.6,
  // Defaults to mostly suppressed, matching how the disk is usually depicted.
  // Push it to 1 for the physically honest asymmetry.
  dopplerBeaming: 0.15,
  // Half-thickness as a fraction of radius. Small — the disk should still read
  // as thin — but not zero: a mathematically flat disk produces lensed images
  // thinner than a pixel next to the shadow, which no sample count can resolve.
  // See slabEntry in trace.wgsl.
  diskThickness: 0.0,
  shading: 'cinematic',
  // Warm white at the hottest ring. Real stellar-mass disks peak in X-rays;
  // this is scaled to where the color is visible.
  peakTemperature: 7500,
};

/** Levels in the bloom chain; level i is the canvas at 1 / 2^(i+1). */
export const BLOOM_LEVELS = 6;
/**
 * Weight of each deeper (twice as wide) bloom level relative to the one above
 * it. Below 1 so the glow has a core and a soft tail rather than a flat haze.
 * present.wgsl divides the geometric sum back out — keep them in step.
 */
export const BLOOM_UPSAMPLE_WEIGHT = 0.6;

/** Past this the image has converged; stop dispatching and just re-present. */
export const MAX_ACCUMULATED_SAMPLES = 1024;

/**
 * While anything is moving, trace a whole image per frame at this fraction of
 * the full trace resolution, adapted to frame time between these bounds.
 *
 * Each RK4 step evaluates the metric four times and a ray takes hundreds of
 * steps, so pixel count dominates the frame cost; resolution is the only lever
 * that scales with it.
 */
export const INTERACTIVE_SCALE_MIN = 0.2;
export const INTERACTIVE_SCALE_MAX = 0.6;
const INTERACTIVE_SCALE_INITIAL = 0.4;
const INTERACTIVE_MS_TOO_SLOW = 22;
const INTERACTIVE_MS_TOO_FAST = 14;

/** How long after the last change to stay in the coarse mode. */
const INTERACTION_IDLE_MS = 180;

/** Time constant of a released drag's coasting, in milliseconds. */
const INERTIA_DECAY_MS = 320;
/** Below this angular speed (radians per ms) coasting stops. */
const INERTIA_STOP = 2e-5;

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

/**
 * f32 so the running average keeps converging past a few hundred samples; f16
 * rounds the 1/n increment away. Not filterable, so it is read with textureLoad.
 */
const ACCUMULATION_FORMAT: GPUTextureFormat = 'rgba32float';
const BLOOM_FORMAT: GPUTextureFormat = 'rgba16float';

/** Extent of bloom level `level`, which halves the canvas level + 1 times. */
const bloomLevelSize = (extent: number, level: number): number =>
  Math.max(1, Math.floor(extent / 2 ** (level + 1)));

type BloomPipelines = {
  first: GPURenderPipeline;
  down: GPURenderPipeline;
  up: GPURenderPipeline;
  /** Reads the unfilterable accumulation texture. */
  firstLayout: GPUBindGroupLayout;
  /** Reads a filterable bloom level. */
  levelLayout: GPUBindGroupLayout;
};

export class KerrRenderer {
  readonly device: GPUDevice;

  #canvas: HTMLCanvasElement;
  #context: GPUCanvasContext;
  #canvasFormat: GPUTextureFormat;

  #computePipeline: GPUComputePipeline;
  #computeLayout: GPUBindGroupLayout;
  #presentPipeline: GPURenderPipeline;
  #presentLayout: GPUBindGroupLayout;
  #bloomPipelines: BloomPipelines;
  #uniformBuffer: GPUBuffer;
  #uniformScratch = new Float32Array(UNIFORM_FLOATS);

  /** Double-buffered accumulation. `#pingPong` indexes the displayed one. */
  #accumTextures: GPUTexture[] = [];
  #computeBindGroups: GPUBindGroup[] = [];
  #presentBindGroups: GPUBindGroup[] = [];
  #sampler: GPUSampler;
  #pingPong = 0;

  #bloomTextures: GPUTexture[] = [];
  #bloomViews: GPUTextureView[] = [];
  /** Bright pass off accumulation texture i. */
  #bloomFirstBindGroups: GPUBindGroup[] = [];
  /** Group i reads bloom level i. */
  #bloomLevelBindGroups: GPUBindGroup[] = [];
  #bloomWidth = 1;
  #bloomHeight = 1;

  /** Where input has asked the camera to be. */
  #target: CameraState = { ...DEFAULT_CAMERA };
  /** What is being drawn, easing toward the target. */
  #view: CameraState = { ...DEFAULT_CAMERA };
  /** Coasting after a released drag, in radians per millisecond. */
  #velocity = { azimuth: 0, elevation: 0 };

  #scene: SceneParams = { ...DEFAULT_SCENE };
  #fluxPeak = pageThorneFluxPeak(DEFAULT_SCENE.spin);
  #luminanceNorm = 1 / blackbodyXYZ(DEFAULT_SCENE.peakTemperature)[1];

  /** Samples blended into the pass in flight. */
  #frameIndex = 0;
  /** Set by anything that invalidates the image; starts a new pass. */
  #restart = true;

  /** Size of the accumulation textures: the full trace resolution. */
  #texWidth = 1;
  #texHeight = 1;
  /** Region traced by the pass in flight. */
  #width = 1;
  #height = 1;
  /** Region of the displayed texture holding a complete image. */
  #displayWidth = 1;
  #displayHeight = 1;
  /** False until a pass has completed into the current textures. */
  #hasDisplay = false;
  /** Swap-chain resolution, which the trace resolution is a fraction of. */
  #canvasWidth = 1;
  #canvasHeight = 1;

  #interacting = false;
  #lastInteractionAt = Number.NEGATIVE_INFINITY;
  #interactiveScale = INTERACTIVE_SCALE_INITIAL;

  /** One full-resolution sample is traced as `#bandCount` successive dispatches. */
  #bandCount = 8;
  #bandIndex = 0;
  #frameMsEma = TARGET_FRAME_MS;
  #lastFrameAt = 0;
  #presentCount = 0;

  #rafHandle = 0;
  #disposed = false;
  #resizeObserver: ResizeObserver | null = null;

  #onStats: ((stats: RendererStats) => void) | null = null;
  #lastStatsAt = Number.NEGATIVE_INFINITY;

  private constructor(
    device: GPUDevice,
    canvas: HTMLCanvasElement,
    context: GPUCanvasContext,
    canvasFormat: GPUTextureFormat,
    compute: { pipeline: GPUComputePipeline; layout: GPUBindGroupLayout },
    present: { pipeline: GPURenderPipeline; layout: GPUBindGroupLayout },
    bloomPipelines: BloomPipelines,
  ) {
    this.device = device;
    this.#canvas = canvas;
    this.#context = context;
    this.#canvasFormat = canvasFormat;
    this.#computePipeline = compute.pipeline;
    this.#computeLayout = compute.layout;
    this.#presentPipeline = present.pipeline;
    this.#presentLayout = present.layout;
    this.#bloomPipelines = bloomPipelines;

    this.#uniformBuffer = device.createBuffer({
      label: 'kerr-uniforms',
      size: UNIFORM_BYTES,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.#sampler = device.createSampler({
      label: 'kerr-bloom-sampler',
      magFilter: 'linear',
      minFilter: 'linear',
      addressModeU: 'clamp-to-edge',
      addressModeV: 'clamp-to-edge',
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

    // Explicit layouts throughout. The accumulation texture is rgba32float,
    // which must be bound as unfilterable, and `layout: 'auto'` would also mint
    // per-pipeline layouts that refuse each other's bind groups.
    const computeLayout = device.createBindGroupLayout({
      label: 'kerr-compute-layout',
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } },
        {
          binding: 1,
          visibility: GPUShaderStage.COMPUTE,
          texture: { sampleType: 'unfilterable-float' },
        },
        {
          binding: 2,
          visibility: GPUShaderStage.COMPUTE,
          storageTexture: { access: 'write-only', format: ACCUMULATION_FORMAT },
        },
      ],
    });

    const presentLayout = device.createBindGroupLayout({
      label: 'kerr-present-layout',
      entries: [
        {
          binding: 0,
          visibility: GPUShaderStage.FRAGMENT,
          texture: { sampleType: 'unfilterable-float' },
        },
        { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
      ],
    });

    const bloomLayout = (label: string, sampleType: GPUTextureSampleType) =>
      device.createBindGroupLayout({
        label,
        entries: [
          { binding: 0, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType } },
          { binding: 1, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
          { binding: 2, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
        ],
      });
    const bloomFirstLayout = bloomLayout('kerr-bloom-first-layout', 'unfilterable-float');
    const bloomLevelLayout = bloomLayout('kerr-bloom-level-layout', 'float');

    const pipelineLayout = (label: string, layout: GPUBindGroupLayout) =>
      device.createPipelineLayout({ label, bindGroupLayouts: [layout] });

    const bloomStage = (
      label: string,
      entryPoint: string,
      layout: GPUBindGroupLayout,
      blend?: GPUBlendState,
    ) =>
      device.createRenderPipelineAsync({
        label,
        layout: pipelineLayout(`${label}-layout`, layout),
        vertex: { module: bloomModule, entryPoint: 'vs' },
        fragment: {
          module: bloomModule,
          entryPoint,
          targets: [{ format: BLOOM_FORMAT, blend }],
        },
        primitive: { topology: 'triangle-list' },
      });

    // Upsampling adds the wider level into the one above it, scaled by the
    // blend constant so deeper levels contribute geometrically less.
    const additive: GPUBlendState = {
      color: { srcFactor: 'constant', dstFactor: 'one', operation: 'add' },
      alpha: { srcFactor: 'zero', dstFactor: 'one', operation: 'add' },
    };

    const [computePipeline, presentPipeline, first, down, up] = await Promise.all([
      device.createComputePipelineAsync({
        label: 'kerr-trace-pipeline',
        layout: pipelineLayout('kerr-trace-pipeline-layout', computeLayout),
        compute: { module: traceModule, entryPoint: 'main' },
      }),
      device.createRenderPipelineAsync({
        label: 'kerr-present-pipeline',
        layout: pipelineLayout('kerr-present-pipeline-layout', presentLayout),
        vertex: { module: presentModule, entryPoint: 'vs' },
        fragment: {
          module: presentModule,
          entryPoint: 'fs',
          targets: [{ format: canvasFormat }],
        },
        primitive: { topology: 'triangle-list' },
      }),
      bloomStage('kerr-bloom-first', 'fsDownsampleFirst', bloomFirstLayout),
      bloomStage('kerr-bloom-down', 'fsDownsample', bloomLevelLayout),
      bloomStage('kerr-bloom-up', 'fsUpsample', bloomLevelLayout, additive),
    ]);

    const renderer = new KerrRenderer(
      device,
      canvas,
      context,
      canvasFormat,
      { pipeline: computePipeline, layout: computeLayout },
      { pipeline: presentPipeline, layout: presentLayout },
      {
        first,
        down,
        up,
        firstLayout: bloomFirstLayout,
        levelLayout: bloomLevelLayout,
      },
    );
    renderer.#observeResize();
    return renderer;
  }

  // -------------------------------------------------------------------------
  // Public surface
  // -------------------------------------------------------------------------

  /** The camera input is steering toward. The drawn camera eases after it. */
  get camera(): CameraState {
    return this.#target;
  }

  get canvasFormat(): GPUTextureFormat {
    return this.#canvasFormat;
  }

  /** Discards accumulated samples; the next frame starts a fresh pass. */
  resetAccumulation(): void {
    this.#restart = true;
  }

  /**
   * Mutating the target camera in place is intentional — orbit drags update it
   * at pointer-event rate and must not go through React state.
   */
  updateCamera(mutate: (camera: CameraState) => void): void {
    mutate(this.#target);
    this.#target.elevation = clampElevation(this.#target.elevation);
    this.#markInteraction();
  }

  /**
   * Set the camera coasting, in radians per millisecond. A released drag hands
   * over its final velocity here; pressing again stops it with zeros.
   */
  setCameraVelocity(azimuth: number, elevation: number): void {
    this.#velocity.azimuth = azimuth;
    this.#velocity.elevation = elevation;
  }

  setScene(scene: Partial<SceneParams>): void {
    const previous = this.#scene;
    const next = { ...previous, ...scene };
    this.#scene = next;

    if (next.spin !== previous.spin) {
      this.#fluxPeak = pageThorneFluxPeak(next.spin);
    }
    if (next.peakTemperature !== previous.peakTemperature) {
      this.#luminanceNorm = 1 / blackbodyXYZ(next.peakTemperature)[1];
    }

    if (next.resolutionScale !== previous.resolutionScale) {
      this.#resize();
      return;
    }

    // Exposure and bloom are applied at present time, so they must not throw
    // away samples that are already converged. Everything else changes what the
    // rays actually do, and the accumulated image is no longer valid.
    const affectsTracing =
      next.spin !== previous.spin ||
      next.diskOuterRadius !== previous.diskOuterRadius ||
      next.diskEnabled !== previous.diskEnabled ||
      next.maxSteps !== previous.maxSteps ||
      next.dopplerBeaming !== previous.dopplerBeaming ||
      next.diskThickness !== previous.diskThickness ||
      next.shading !== previous.shading ||
      next.peakTemperature !== previous.peakTemperature;

    // A slider drag is an interaction like a camera drag: preview it coarse and
    // live, then refine once it stops.
    if (affectsTracing) this.#markInteraction();
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

  /**
   * Reallocate for a new canvas size or resolution scale. Interaction never
   * comes through here — the coarse trace is a region of these same textures.
   */
  #resize(): void {
    const dpr = Math.min(globalThis.devicePixelRatio || 1, 2);
    const rect = this.#canvas.getBoundingClientRect();
    const limit = this.device.limits.maxTextureDimension2D;

    const clamp = (value: number) =>
      Math.max(1, Math.min(Math.floor(value), limit));

    const canvasWidth = clamp(rect.width * dpr);
    const canvasHeight = clamp(rect.height * dpr);
    const texWidth = clamp(canvasWidth * this.#scene.resolutionScale);
    const texHeight = clamp(canvasHeight * this.#scene.resolutionScale);

    if (
      texWidth === this.#texWidth &&
      texHeight === this.#texHeight &&
      canvasWidth === this.#canvasWidth &&
      canvasHeight === this.#canvasHeight &&
      this.#accumTextures.length > 0
    ) {
      return;
    }

    this.#texWidth = texWidth;
    this.#texHeight = texHeight;
    this.#canvasWidth = canvasWidth;
    this.#canvasHeight = canvasHeight;
    this.#canvas.width = canvasWidth;
    this.#canvas.height = canvasHeight;

    for (const texture of this.#accumTextures) texture.destroy();

    this.#accumTextures = [0, 1].map((i) =>
      this.device.createTexture({
        label: `kerr-accumulation-${i}`,
        size: { width: texWidth, height: texHeight },
        format: ACCUMULATION_FORMAT,
        usage:
          GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING,
      }),
    );

    // Two bind groups, pre-built: group i reads texture i and writes 1 - i.
    // Swapping an index each pass keeps the loop allocation-free.
    this.#computeBindGroups = [0, 1].map((i) =>
      this.device.createBindGroup({
        label: `kerr-compute-bindgroup-${i}`,
        layout: this.#computeLayout,
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
        layout: this.#presentLayout,
        entries: [
          { binding: 0, resource: this.#accumTextures[i].createView() },
          { binding: 1, resource: { buffer: this.#uniformBuffer } },
          { binding: 2, resource: this.#sampler },
          { binding: 3, resource: this.#bloomViews[0] },
        ],
      }),
    );

    // Fresh textures hold nothing worth showing.
    this.#hasDisplay = false;
    this.#restart = true;
  }

  #buildBloomResources(): void {
    for (const texture of this.#bloomTextures) texture.destroy();

    this.#bloomWidth = bloomLevelSize(this.#canvasWidth, 0);
    this.#bloomHeight = bloomLevelSize(this.#canvasHeight, 0);

    this.#bloomTextures = Array.from({ length: BLOOM_LEVELS }, (_, i) =>
      this.device.createTexture({
        label: `kerr-bloom-${i}`,
        size: {
          width: bloomLevelSize(this.#canvasWidth, i),
          height: bloomLevelSize(this.#canvasHeight, i),
        },
        format: BLOOM_FORMAT,
        usage:
          GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
      }),
    );
    this.#bloomViews = this.#bloomTextures.map((texture) => texture.createView());

    this.#bloomFirstBindGroups = [0, 1].map((i) =>
      this.device.createBindGroup({
        label: `kerr-bloom-first-bindgroup-${i}`,
        layout: this.#bloomPipelines.firstLayout,
        entries: [
          { binding: 0, resource: this.#accumTextures[i].createView() },
          { binding: 1, resource: { buffer: this.#uniformBuffer } },
          { binding: 2, resource: this.#sampler },
        ],
      }),
    );

    this.#bloomLevelBindGroups = this.#bloomViews.map((view, i) =>
      this.device.createBindGroup({
        label: `kerr-bloom-level-bindgroup-${i}`,
        layout: this.#bloomPipelines.levelLayout,
        entries: [
          { binding: 0, resource: view },
          { binding: 1, resource: { buffer: this.#uniformBuffer } },
          { binding: 2, resource: this.#sampler },
        ],
      }),
    );
  }

  /**
   * Bright pass into level 0, downsample to the smallest level, then upsample
   * back, adding each level into the one above. Runs every frame off the
   * presented image; the whole chain touches about a third of the canvas's
   * pixels, which is nothing next to integrating geodesics.
   */
  #encodeBloom(encoder: GPUCommandEncoder, presentIndex: number): void {
    const stage = (
      label: string,
      pipeline: GPURenderPipeline,
      bindGroup: GPUBindGroup,
      level: number,
      accumulate: boolean,
    ) => {
      const pass = encoder.beginRenderPass({
        label,
        colorAttachments: [
          {
            view: this.#bloomViews[level],
            loadOp: accumulate ? 'load' : 'clear',
            storeOp: 'store',
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
          },
        ],
      });
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      if (accumulate) {
        const w = BLOOM_UPSAMPLE_WEIGHT;
        pass.setBlendConstant({ r: w, g: w, b: w, a: w });
      }
      pass.draw(3);
      pass.end();
    };

    stage(
      'kerr-bloom-down-0',
      this.#bloomPipelines.first,
      this.#bloomFirstBindGroups[presentIndex],
      0,
      false,
    );
    for (let level = 1; level < BLOOM_LEVELS; level++) {
      stage(
        `kerr-bloom-down-${level}`,
        this.#bloomPipelines.down,
        this.#bloomLevelBindGroups[level - 1],
        level,
        false,
      );
    }
    // Lands back in level 0, which is what the present pass samples.
    for (let level = BLOOM_LEVELS - 2; level >= 0; level--) {
      stage(
        `kerr-bloom-up-${level}`,
        this.#bloomPipelines.up,
        this.#bloomLevelBindGroups[level + 1],
        level,
        true,
      );
    }
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
      dopplerBeaming: this.#scene.dopplerBeaming,
      diskThickness: this.#scene.diskThickness,
      shading: this.#scene.shading,
      peakTemperature: this.#scene.peakTemperature,
      fluxPeak: this.#fluxPeak,
      luminanceNorm: this.#luminanceNorm,
    };
  }

  #markInteraction(): void {
    this.#lastInteractionAt = performance.now();
    this.#restart = true;
  }

  /** Coasting and easing. Any visible camera motion counts as interaction. */
  #advanceCamera(dt: number): void {
    const velocity = this.#velocity;
    if (velocity.azimuth !== 0 || velocity.elevation !== 0) {
      this.#target.azimuth += velocity.azimuth * dt;
      this.#target.elevation = clampElevation(
        this.#target.elevation + velocity.elevation * dt,
      );
      const decay = Math.exp(-dt / INERTIA_DECAY_MS);
      velocity.azimuth *= decay;
      velocity.elevation *= decay;
      if (Math.hypot(velocity.azimuth, velocity.elevation) < INERTIA_STOP) {
        velocity.azimuth = 0;
        velocity.elevation = 0;
      }
    }

    if (easeCamera(this.#view, this.#target, dt)) {
      this.#markInteraction();
    }
  }

  #renderFrame(): void {
    if (this.#accumTextures.length < 2) return;

    const now = performance.now();
    const dt = this.#lastFrameAt === 0 ? 0 : Math.min(now - this.#lastFrameAt, 250);
    this.#trackFrameTime(now);
    this.#advanceCamera(dt);

    const interacting = now - this.#lastInteractionAt <= INTERACTION_IDLE_MS;
    if (interacting !== this.#interacting) {
      // Changing resolution invalidates the pass in flight.
      this.#interacting = interacting;
      this.#restart = true;
    }

    if (this.#restart) {
      this.#restart = false;
      if (interacting) this.#adaptInteractiveScale();
      const scale = interacting ? this.#interactiveScale : 1;
      this.#width = Math.max(1, Math.min(this.#texWidth, Math.floor(this.#texWidth * scale)));
      this.#height = Math.max(1, Math.min(this.#texHeight, Math.floor(this.#texHeight * scale)));
      this.#frameIndex = 0;
      this.#bandIndex = 0;
    }

    // A moving image is traced whole every frame; a still one in bands.
    const bandCount = this.#interacting ? 1 : this.#bandCount;
    const converged = this.#frameIndex >= MAX_ACCUMULATED_SAMPLES;

    // Rows covered by this dispatch. The last band absorbs the remainder.
    const bandHeight = Math.ceil(this.#height / bandCount);
    const bandOffset = this.#bandIndex * bandHeight;
    const bandRows = Math.min(bandHeight, this.#height - bandOffset);

    // #pingPong indexes the last *complete* image. Before any pass has
    // completed into these textures there is nothing to show but the pass in
    // flight, so show that and let bands appear as they land.
    const destination = 1 - this.#pingPong;
    const presentIndex = this.#hasDisplay ? this.#pingPong : destination;

    packUniforms(this.#uniformScratch, {
      basis: cameraBasis(this.#view),
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
      presentWidth: this.#hasDisplay ? this.#displayWidth : this.#width,
      presentHeight: this.#hasDisplay ? this.#displayHeight : this.#height,
      ditherSeed: this.#presentCount++ % 4096,
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
      if (this.#bandIndex >= bandCount) {
        // Pass complete: the destination now holds a whole image, and it is
        // what the screen shows from the next frame on.
        this.#bandIndex = 0;
        this.#pingPong = destination;
        this.#displayWidth = this.#width;
        this.#displayHeight = this.#height;
        this.#hasDisplay = true;
        this.#frameIndex++;
        if (!this.#interacting) this.#adaptBandCount();
      }
    }

    this.#reportStats(converged);
  }

  #trackFrameTime(now: number): void {
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
   * Hold the interactive frame rate by trading resolution. Remembered across
   * gestures, so the second drag starts where the first one settled.
   */
  #adaptInteractiveScale(): void {
    if (this.#frameMsEma > INTERACTIVE_MS_TOO_SLOW) {
      this.#interactiveScale = Math.max(INTERACTIVE_SCALE_MIN, this.#interactiveScale * 0.92);
    } else if (this.#frameMsEma < INTERACTIVE_MS_TOO_FAST) {
      this.#interactiveScale = Math.min(INTERACTIVE_SCALE_MAX, this.#interactiveScale * 1.04);
    }
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
      bandCount: this.#interacting ? 1 : this.#bandCount,
    });
  }
}
