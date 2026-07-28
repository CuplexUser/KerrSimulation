/**
 * Uniform buffer packing.
 *
 * Everything is padded to vec4 so WGSL's uniform-address-space alignment rules
 * cannot bite. This layout is the contract with the `Uniforms` struct declared in
 * both trace.wgsl and present.wgsl — change one, change all three.
 */

import type { CameraBasis } from './camera.ts';

export const UNIFORM_FLOATS = 36;
export const UNIFORM_BYTES = UNIFORM_FLOATS * 4;

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
  /** Size of the bloom chain textures. */
  bloomWidth: number;
  bloomHeight: number;
};

/**
 * Writes into the caller's scratch array so the render loop allocates nothing.
 *
 *   camPos   vec4  xyz eye        w unused
 *   camRight vec4  xyz            w tanHalfFov
 *   camUp    vec4  xyz            w aspect
 *   camFwd   vec4  xyz            w unused
 *   params   vec4  spin, rOuter, rIsco, rPlus
 *   frame    vec4  frameIndex, traceResX, traceResY, exposure
 *   options  vec4  diskEnabled, maxSteps, canvasW, canvasH
 *   band     vec4  bandOffset, bandHeight, dopplerBeaming, unused
 *   bloom    vec4  bloomW, bloomH, threshold, strength
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
  } = input;

  target[0] = basis.eye[0];
  target[1] = basis.eye[1];
  target[2] = basis.eye[2];
  target[3] = 0;

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
  target[15] = 0;

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
  target[31] = 0;

  target[32] = bloomWidth;
  target[33] = bloomHeight;
  target[34] = params.bloomThreshold;
  target[35] = params.bloomStrength;
}
