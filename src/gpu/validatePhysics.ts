/**
 * In-app physics validation.
 *
 * Runs the *shipped* WGSL (`kerr_math.wgsl`, via `validate.wgsl`) over the same
 * ray suite as the Node harness, then compares the f32 GPU results against f64
 * reference values computed live in the browser from `src/physics/kerrReference.ts`.
 *
 * There are no baked-in golden numbers: both sides are computed fresh, so this
 * cannot silently go stale when the physics is edited. What it proves that the
 * Node script cannot is that the actual shader text is correct — a transcription
 * slip between the TypeScript reference and the WGSL would show up here as a
 * disagreement on fate, closest approach, or a conserved quantity.
 */

import {
  FATE_ORDER,
  REFERENCE_MAX_STEPS,
  REFERENCE_RAYS,
  iscoRadius,
  normalize,
  traceRay,
  type Fate,
} from '../physics/kerrReference.ts';
import { validateShader } from './shaders/index.ts';

const RESULT_FLOATS = 8;
const SPEC_FLOATS = 8;

const sci = (n: number): string => n.toExponential(2);

/** Tolerances for f32-vs-f64 agreement. */
const TOL = {
  /**
   * f32 roundoff random-walks over hundreds of RK4 steps. The gradient is
   * analytic now, so this is looser than it needs to be — a geometry error
   * would still blow past it by orders of magnitude.
   */
  maxAbsH: 5e-3,
  maxRelL: 5e-3,
  /** Closest approach, relative. Chaotic near the critical impact parameter. */
  minRadius: 0.02,
  isco: 1e-4,
};

export type RayComparison = {
  name: string;
  gpu: { maxAbsH: number; maxRelL: number; minRadius: number; steps: number; fate: Fate; l0: number };
  cpu: { maxAbsH: number; maxRelL: number; minRadius: number; steps: number; fate: Fate; l0: number };
  fateAgrees: boolean;
  radiusRelDiff: number;
  ok: boolean;
};

export type ValidationReport = {
  passed: boolean;
  comparisons: RayComparison[];
  iscoMaxDiff: number;
};

type RawResult = {
  maxAbsH: number;
  maxRelL: number;
  minRadius: number;
  steps: number;
  fate: Fate;
  l0: number;
  h0: number;
  isco: number;
};

/** Dispatch the validation shader once and read the per-ray results back. */
async function runOnGpu(device: GPUDevice): Promise<RawResult[]> {
  const rayCount = REFERENCE_RAYS.length;

  const specData = new Float32Array(rayCount * SPEC_FLOATS);
  REFERENCE_RAYS.forEach((ray, i) => {
    const d = normalize(ray.direction);
    specData.set(
      [
        ray.origin[0],
        ray.origin[1],
        ray.origin[2],
        ray.a,
        d[0],
        d[1],
        d[2],
        REFERENCE_MAX_STEPS,
      ],
      i * SPEC_FLOATS,
    );
  });

  const specBuffer = device.createBuffer({
    label: 'validate-specs',
    size: specData.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(specBuffer, 0, specData);

  const resultBytes = rayCount * RESULT_FLOATS * 4;
  const resultBuffer = device.createBuffer({
    label: 'validate-results',
    size: resultBytes,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const readbackBuffer = device.createBuffer({
    label: 'validate-readback',
    size: resultBytes,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const module = device.createShaderModule({
    label: 'validate-shader',
    code: validateShader,
  });
  const pipeline = await device.createComputePipelineAsync({
    label: 'validate-pipeline',
    layout: 'auto',
    compute: { module, entryPoint: 'main' },
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: specBuffer } },
      { binding: 1, resource: { buffer: resultBuffer } },
    ],
  });

  const encoder = device.createCommandEncoder({ label: 'validate-encoder' });
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.dispatchWorkgroups(Math.ceil(rayCount / 64));
  pass.end();
  encoder.copyBufferToBuffer(resultBuffer, 0, readbackBuffer, 0, resultBytes);
  device.queue.submit([encoder.finish()]);

  await readbackBuffer.mapAsync(GPUMapMode.READ);
  const view = new Float32Array(readbackBuffer.getMappedRange().slice(0));
  readbackBuffer.unmap();

  specBuffer.destroy();
  resultBuffer.destroy();
  readbackBuffer.destroy();

  return REFERENCE_RAYS.map((_, i) => {
    const o = i * RESULT_FLOATS;
    return {
      maxAbsH: view[o],
      maxRelL: view[o + 1],
      minRadius: view[o + 2],
      steps: view[o + 3],
      fate: FATE_ORDER[view[o + 4]] ?? 'maxSteps',
      l0: view[o + 5],
      h0: view[o + 6],
      isco: view[o + 7],
    };
  });
}

export async function validatePhysicsOnGpu(
  device: GPUDevice,
): Promise<ValidationReport> {
  const gpuResults = await runOnGpu(device);

  const cpuResults = REFERENCE_RAYS.map((ray) =>
    traceRay(ray.origin, ray.direction, ray.a, REFERENCE_MAX_STEPS),
  );

  const comparisons: RayComparison[] = REFERENCE_RAYS.map((ray, i) => {
    const gpu = gpuResults[i];
    const cpu = cpuResults[i];
    const radiusRelDiff =
      Math.abs(gpu.minRadius - cpu.minRadius) / Math.max(cpu.minRadius, 1e-6);
    const fateAgrees = gpu.fate === cpu.fate;
    const hTol = ray.hTol ? Math.max(ray.hTol, TOL.maxAbsH) : TOL.maxAbsH;

    return {
      name: ray.name,
      gpu: {
        maxAbsH: gpu.maxAbsH,
        maxRelL: gpu.maxRelL,
        minRadius: gpu.minRadius,
        steps: gpu.steps,
        fate: gpu.fate,
        l0: gpu.l0,
      },
      cpu: {
        maxAbsH: cpu.maxAbsH,
        maxRelL: cpu.maxRelL,
        minRadius: cpu.minRadius,
        steps: cpu.steps,
        fate: cpu.fate,
        l0: cpu.l0,
      },
      fateAgrees,
      radiusRelDiff,
      ok:
        fateAgrees &&
        gpu.maxAbsH < hTol &&
        gpu.maxRelL < TOL.maxRelL &&
        radiusRelDiff < TOL.minRadius,
    };
  });

  const iscoMaxDiff = Math.max(
    ...REFERENCE_RAYS.map((ray, i) =>
      Math.abs(gpuResults[i].isco - iscoRadius(ray.a)),
    ),
  );

  const report: ValidationReport = {
    passed: comparisons.every((c) => c.ok) && iscoMaxDiff < TOL.isco,
    comparisons,
    iscoMaxDiff,
  };

  logReport(report);
  return report;
}

function logReport(report: ValidationReport): void {
  console.group(
    `%cKerr physics validation — ${report.passed ? 'PASS' : 'FAIL'}`,
    `font-weight:bold;color:${report.passed ? '#4ade80' : '#f87171'}`,
  );
  console.log(
    'GPU (f32, shipped WGSL) vs CPU (f64 reference, src/physics/kerrReference.ts)',
  );
  console.table(
    report.comparisons.map((c) => ({
      ray: c.name,
      ok: c.ok ? 'PASS' : 'FAIL',
      'fate gpu': c.gpu.fate,
      'fate cpu': c.cpu.fate,
      'r_min gpu': c.gpu.minRadius.toFixed(4),
      'r_min cpu': c.cpu.minRadius.toFixed(4),
      'r_min rel diff': sci(c.radiusRelDiff),
      'max|H| gpu': sci(c.gpu.maxAbsH),
      'max|H| cpu': sci(c.cpu.maxAbsH),
      'max dL/L gpu': sci(c.gpu.maxRelL),
    })),
  );
  console.log(
    `ISCO agreement (GPU iscoRadius vs CPU): max abs diff = ${sci(report.iscoMaxDiff)}`,
  );

  console.groupEnd();
}
