import type Network from '../network';
import { canUseGPU } from './network.gpu.capability';
import { batchActivate } from './network.gpu.batched';
import { SUPPORTED_ACTIVATION_INDICES } from './network.gpu.kernel';

/**
 * Options controlling batch evaluation in the racing-curriculum worker seam.
 */
export interface RacingBatchOptions {
  /**
   * Minimum number of networks required before the GPU batch path is chosen.
   * Smaller batches always fall back to per-network CPU activation.
   *
   * @default 8
   */
  gpuBatchThreshold?: number;
}

/**
 * Default threshold below which the per-network CPU path is preferred.
 *
 * The racing-curriculum worker can override this through
 * `RacingBatchOptions.gpuBatchThreshold`; the default is sized for a
 * generation large enough that GPU dispatch overhead is amortized.
 */
const DEFAULT_GPU_BATCH_THRESHOLD = 8;

/**
 * Activation indices that the current placeholder GPU kernel understands.
 *
 * Reused from the kernel contract so the racing eligibility check stays in
 * sync with the batched GPU seam.
 */
const SUPPORTED_ACTIVATIONS = new Set<number>(
  SUPPORTED_ACTIVATION_INDICES as unknown as number[],
);

/**
 * Check whether a supplied GPU device is present and has not been lost.
 *
 * See the matching helper in `network.gpu.fallback` for the rationale:
 * real devices report loss asynchronously through `device.lost`, while this
 * predicate gives a synchronous yes/no answer for the current call site.
 *
 * @param device - Device to inspect, or null/undefined when WebGPU is absent.
 * @returns True when the device is present and not marked lost.
 */
function isDeviceUsable(
  device: GPUDevice | null | undefined,
): device is GPUDevice {
  if (device === null || device === undefined) {
    return false;
  }

  const maybeLost = device as unknown as { __lost?: boolean };
  return maybeLost.__lost !== true;
}

/**
 * Decide whether the racing generation can use the batched GPU path.
 *
 * All of the following must hold:
 * 1. The batch size is strictly greater than the configured threshold.
 * 2. Every network in the batch is structurally GPU-eligible.
 *
 * Callers must already have verified the device is usable with
 * `isDeviceUsable` before invoking this predicate.
 *
 * @param networks - Generation of networks to evaluate.
 * @param device - Verified WebGPU device.
 * @param threshold - Minimum batch size that justifies GPU dispatch.
 * @returns True when the batched GPU path should be used.
 */
function shouldUseGPUPath(
  networks: Network[],
  device: GPUDevice,
  threshold: number,
): boolean {
  if (networks.length <= threshold) {
    return false;
  }

  return networks.every((network) =>
    canUseGPU(network, device, SUPPORTED_ACTIVATIONS),
  );
}

/**
 * Fall back to per-network CPU activation and stack the results.
 *
 * The output matrix is laid out row-major, with one row per network. Each row
 * contains the output values returned by `network.activate()` for the
 * corresponding input slice.
 *
 * @param networks - Networks to evaluate in CPU mode.
 * @param inputMatrix - Flattened row-major input matrix.
 * @returns Row-major Float32Array of stacked network outputs.
 */
function evaluateOnCPU(
  networks: Network[],
  inputMatrix: Float32Array,
): Float32Array {
  if (networks.length === 0) {
    return new Float32Array(0);
  }

  const inputCount = networks[0].input;
  const outputCount = networks[0].output;
  const outputs = new Float32Array(networks.length * outputCount);

  for (let i = 0; i < networks.length; i++) {
    const start = i * inputCount;
    const slice = inputMatrix.subarray(start, start + inputCount);
    const row = networks[i].activate(slice);

    for (let j = 0; j < row.length; j++) {
      outputs[i * outputCount + j] = row[j];
    }
  }

  return outputs;
}

/**
 * Racing-curriculum generation evaluation seam.
 *
 * Evaluates a generation of controller networks against a row-major input
 * matrix. Uses the batched GPU path when the batch size is above the threshold,
 * every network is GPU-eligible, and a valid device is supplied; otherwise falls
 * back to per-network CPU `network.activate()` calls.
 *
 * @param networks - One network per car / genome in the generation.
 * @param inputMatrix - Flattened row-major inputs, length
 *   `networks.length * networks[0].input`.
 * @param device - WebGPU device, or null when GPU inference is unavailable.
 * @param options - Threshold and policy options.
 * @returns Promise resolving to a row-major output matrix of length
 *   `networks.length * networks[0].output`.
 */
export async function evaluateRacingGeneration(
  networks: Network[],
  inputMatrix: Float32Array,
  device: GPUDevice | null,
  options?: RacingBatchOptions,
): Promise<Float32Array> {
  const threshold = options?.gpuBatchThreshold ?? DEFAULT_GPU_BATCH_THRESHOLD;

  if (isDeviceUsable(device) && shouldUseGPUPath(networks, device, threshold)) {
    const { outputs } = await batchActivate(device, networks, inputMatrix);
    return outputs;
  }

  return evaluateOnCPU(networks, inputMatrix);
}

/**
 * Single concurrent racing evaluation request.
 */
export interface RacingAgentRequest {
  network: Network;
  inputs: Float32Array | number[];
}

/**
 * Evaluate many racing agents in parallel on the GPU.
 *
 * This is a stub export that satisfies the compile-time contract for the red
 * testing phase. The implementation will dispatch all requests concurrently,
 * avoid pipeline collisions when the same network instance appears multiple
 * times, and return one output array per request.
 *
 * @param device - WebGPU device used to run the concurrent dispatch.
 * @param requests - One request per agent to evaluate.
 * @param options - Threshold and policy options.
 * @returns Promise resolving to one output array per request, in request order.
 */
export async function evaluateConcurrentRacingAgents(
  device: GPUDevice,
  requests: RacingAgentRequest[],
  options?: RacingBatchOptions,
): Promise<Float32Array[]> {
  void device;
  void requests;
  void options;
  throw new Error('evaluateConcurrentRacingAgents: not implemented');
}
