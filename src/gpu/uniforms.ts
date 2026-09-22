/**
 * Uniform buffer packing.
 *
 * Everything is padded to vec4 so WGSL's uniform-address-space alignment rules
 * cannot bite. This layout is the contract with the `Uniforms` struct declared in
 * trace.wgsl, present.wgsl and bloom.wgsl — change one, change them all.
 */

import type { CameraBasis } from './camera.ts';

export const UNIFORM_FLOATS = 48;
export const UNIFORM_BYTES = UNIFORM_FLOATS * 4;

export type Shading = 'cinematic' | 'physical';

/**
 * What the escaped rays land on. Stars is the film look; the others are there
 * to make the lensing legible. A sparse field of point sources hides most of
 * the map — the sky has to have structure for its distortion to show.
 */
export type Background = 'stars' | 'galaxy' | 'grid' | 'checker';

/** Index the shader switches on. Order is the contract with background() in trace.wgsl. */
export const BACKGROUND_INDEX: Record<Background, number> = {
  stars: 0,
  galaxy: 1,
  grid: 2,
  checker: 3,
};

export type RenderParams = {
  spin: number;
  diskOuterRadius: number;
  iscoRadius: number;
  horizonRadius: number;
  diskEnabled: boolean;
  maxSteps: number;
  exposure: number;
  bloomThreshold: number;
  bloomStrength: number;
  /** 0 = brightness-symmetric disk (the film look), 1 = full relativistic g^3. */
  dopplerBeaming: number;
  /** Disk half-thickness as a fraction of radius. 0 is a mathematical plane. */
  diskThickness: number;
  shading: Shading;
  background: Background;
  /** Physical mode: emitted temperature at the flux peak, in kelvin. */
  peakTemperature: number;
  /** Physical mode: peak of the Page-Thorne flux profile at this spin. */
  fluxPeak: number;
  /** Physical mode: 1 / luminance of a blackbody at the peak temperature. */
  luminanceNorm: number;
};

export type UniformInput = {
  basis: CameraBasis;
  params: RenderParams;
  frameIndex: number;
  /** Resolution the compute pass traces at — may be coarser than the canvas. */
  width: number;
  height: number;
  /** Swap-chain size, used by the present pass to map fragments to UVs. */
  canvasWidth: number;
  canvasHeight: number;
  /** Row range this dispatch covers. One sample is spread over several bands. */
  bandOffset: number;
  bandHeight: number;
  /** Size of the first bloom level. */
  bloomWidth: number;
  bloomHeight: number;
  /**
   * Region of the accumulation texture holding the image on screen. It can
   * differ from the traced size: the last complete (possibly coarse) image
   * stays on screen while a new pass is in flight.
   */
  presentWidth: number;
  presentHeight: number;
  /** Changes every presented frame, so the output dither does not freeze. */
  ditherSeed: number;
};

/**
 * Writes into the caller's scratch array so the render loop allocates nothing.
 *
 *   camPos   vec4  xyz eye        w lens shift x
 *   camRight vec4  xyz            w tanHalfFov
 *   camUp    vec4  xyz            w aspect
 *   camFwd   vec4  xyz            w lens shift y
 *   params   vec4  spin, rOuter, rIsco, rPlus
 *   frame    vec4  frameIndex, traceResX, traceResY, exposure
 *   options  vec4  diskEnabled, maxSteps, canvasW, canvasH
 *   band     vec4  bandOffset, bandHeight, dopplerBeaming, diskThickness
 *   bloom    vec4  bloomW, bloomH, threshold, strength
 *   disk     vec4  physical shading flag, peakTemperature, 1/fluxPeak, luminanceNorm
 *   view     vec4  presentW, presentH, ditherSeed, pixel angle (radians)
 *   sky      vec4  background index, unused, unused, unused
 */
export function packUniforms(target: Float32Array, input: UniformInput): void {
  const {
    basis,
    params,
    frameIndex,
    width,
    height,
    canvasWidth,
    canvasHeight,
    bandOffset,
    bandHeight,
    bloomWidth,
    bloomHeight,
    presentWidth,
    presentHeight,
    ditherSeed,
  } = input;

  target[0] = basis.eye[0];
  target[1] = basis.eye[1];
  target[2] = basis.eye[2];
  target[3] = basis.shift[0];

  target[4] = basis.right[0];
  target[5] = basis.right[1];
  target[6] = basis.right[2];
  target[7] = basis.tanHalfFov;

  target[8] = basis.up[0];
  target[9] = basis.up[1];
  target[10] = basis.up[2];
  target[11] = height === 0 ? 1 : width / height;

  target[12] = basis.forward[0];
  target[13] = basis.forward[1];
  target[14] = basis.forward[2];
  target[15] = basis.shift[1];

  target[16] = params.spin;
  target[17] = params.diskOuterRadius;
  target[18] = params.iscoRadius;
  target[19] = params.horizonRadius;

  target[20] = frameIndex;
  target[21] = width;
  target[22] = height;
  target[23] = params.exposure;

  target[24] = params.diskEnabled ? 1 : 0;
  target[25] = params.maxSteps;
  target[26] = canvasWidth;
  target[27] = canvasHeight;

  target[28] = bandOffset;
  target[29] = bandHeight;
  target[30] = params.dopplerBeaming;
  target[31] = params.diskThickness;

  target[32] = bloomWidth;
  target[33] = bloomHeight;
  target[34] = params.bloomThreshold;
  target[35] = params.bloomStrength;

  target[36] = params.shading === 'physical' ? 1 : 0;
  target[37] = params.peakTemperature;
  target[38] = params.fluxPeak > 0 ? 1 / params.fluxPeak : 0;
  target[39] = params.luminanceNorm;

  target[40] = presentWidth;
  target[41] = presentHeight;
  target[42] = ditherSeed;
  // Angular size of one traced pixel. Stars are splatted at no less than this,
  // so their brightness does not depend on the trace resolution.
  target[43] = height === 0 ? 0 : (2 * basis.tanHalfFov) / height;

  target[44] = BACKGROUND_INDEX[params.background];
  target[45] = 0;
  target[46] = 0;
  target[47] = 0;
}
