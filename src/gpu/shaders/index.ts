/**
 * Shader composition.
 *
 * `kerr_math.wgsl` holds the physics and is prepended to every shader that needs
 * it, so the tracing pipeline and the validation harness provably run identical
 * code. Vite's `?raw` import handles the loading — no plugin required.
 */

import kerrMath from './kerr_math.wgsl?raw';
import presentSource from './present.wgsl?raw';
import traceSource from './trace.wgsl?raw';
import validateSource from './validate.wgsl?raw';

export const traceShader = `${kerrMath}\n${traceSource}`;
export const validateShader = `${kerrMath}\n${validateSource}`;
export const presentShader = presentSource;
