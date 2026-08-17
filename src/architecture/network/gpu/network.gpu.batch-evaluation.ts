import type Network from '../network';
import { activateGPUWithFreshState } from './network.gpu.activate';
import { batchActivate } from './network.gpu.batched';
import { canUseGPU } from './network.gpu.capability';
import { SUPPORTED_ACTIVATION_INDICES } from './network.gpu.kernel';

/**
 * Options controlling batch evaluation in the worker seam for the GPU path.
 */
export interface BatchEvaluationOptions {
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
 * The worker seam can override this through
 * `BatchEvaluationOptions.gpuBatchThreshold`; the default is sized for a
 * batch generation large enough that GPU dispatch overhead is amortized.
 */
const DEFAULT_GPU_BATCH_THRESHOLD = 8;

/**
 * Activation indices that the current placeholder GPU kernel understands.
 *
 * Reused from the kernel contract so the batch eligibility check stays in
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
 * Decide whether the batch generation can use the batched GPU path.
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
 * Batch generation evaluation seam.
 *
 * Evaluates a generation of controller networks against a row-major input
 * matrix. Uses the batched GPU path when the batch size is above the threshold,
 * every network is GPU-eligible, and a valid device is supplied; otherwise falls
 * back to per-network CPU `network.activate()` calls.
 *
 * @param networks - One network per genome in the generation.
 * @param inputMatrix - Flattened row-major inputs, length
 *   `networks.length * networks[0].input`.
 * @param device - WebGPU device, or null when GPU inference is unavailable.
 * @param options - Threshold and policy options.
 * @returns Promise resolving to a row-major output matrix of length
 *   `networks.length * networks[0].output`.
 */
export async function evaluateBatchGeneration(
  networks: Network[],
  inputMatrix: Float32Array,
  device: GPUDevice | null,
  options?: BatchEvaluationOptions,
): Promise<Float32Array> {
  const threshold = options?.gpuBatchThreshold ?? DEFAULT_GPU_BATCH_THRESHOLD;

  if (isDeviceUsable(device) && shouldUseGPUPath(networks, device, threshold)) {
    const { outputs } = await batchActivate(device, networks, inputMatrix);
    return outputs;
  }

  return evaluateOnCPU(networks, inputMatrix);
}

/**
 * Single concurrent agent evaluation request submitted to the batch GPU path.
 */
export interface AgentEvaluationRequest {
  network: Network;
  inputs: Float32Array | number[];
}

/**
 * Evaluate many agents in parallel on the GPU.
 *
 * Each request receives its own fresh GPU buffer set and bind group, so the
 * same `Network` instance can appear in multiple requests with different inputs
 * without any read/write collision. Pipelines are still shared through the
 * per-device pipeline cache, so identical topologies compile once regardless of
 * how the requests are interleaved.
 *
 * @param device - WebGPU device used to run the concurrent dispatch.
 * @param requests - One request per agent to evaluate.
 * @param options - Threshold and policy options (currently unused; reserved for
 *   future batch-size tuning).
 * @returns Promise resolving to one output array per request, in request order.
 */
export async function evaluateConcurrentAgents(
  device: GPUDevice,
  requests: AgentEvaluationRequest[],
  options?: BatchEvaluationOptions,
): Promise<Float32Array[]> {
  void options;

  if (requests.length === 0) {
    return [];
  }

  return Promise.all(
    requests.map((request) =>
      activateGPUWithFreshState(device, request.network, request.inputs),
    ),
  );
}
