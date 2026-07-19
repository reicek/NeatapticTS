/**
 * Generic GPU auto-enable helpers for the acceleration layer.
 *
 * `shouldAutoEnableGpu` decides whether a network is large enough to benefit
 * from WebGPU offload, using node-count and batch-parallel thresholds that are
 * configurable through {@link AccelerationConfig}. `autoEnableGpu` performs the
 * actual probe: it checks eligibility, confirms that a WebGPU surface is
 * available, and requests a device via the shared
 * {@link requestGPUDevice} helper.
 *
 * These helpers are environment-agnostic wrappers around the WebGPU probe. They
 * gracefully fall back to CPU when GPU support is missing, disabled, or when the
 * network is too small to justify the GPU setup cost.
 *
 * Background reading:
 * - WebGPU is described in [WebGPU (Wikipedia)](https://en.wikipedia.org/wiki/WebGPU).
 * - The W3C WebGPU specification is the authoritative reference:
 *   [WebGPU API](https://www.w3.org/TR/webgpu/).
 */

import { DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD } from './acceleration.constants';
import { resolveAccelerationConfig } from './acceleration.config';
import type {
  AccelerationConfig,
  VariantEvaluationNetwork,
  VariantScorer,
  WeightVariant,
  WeightVariantInputs,
  WeightVariantTarget,
} from './acceleration.types';
import { requestGPUDevice } from './acceleration.gpu.device';

export type { AccelerationConfig } from './acceleration.types';

/**
 * Result of a GPU auto-enable attempt.
 *
 * The result surface is intentionally flat and serialisation-friendly: it tells
 * the caller whether the GPU was enabled, gives the acquired device (if any),
 * records whether the caller was notified that a GPU exists but was not enabled,
 * and explains the outcome with a human-readable reason.
 */
export interface GpuAutoEnableResult {
  /** Whether GPU acceleration was successfully enabled. */
  readonly enabled: boolean;

  /** WebGPU device when auto-enable succeeded; `null` otherwise. */
  readonly gpuDevice: GPUDevice | null;

  /**
   * Whether the caller was notified that a GPU is available but was not
   * auto-enabled (for example, the network is below the threshold).
   */
  readonly notified: boolean;

  /** Human-readable explanation of the result. */
  readonly reason: string;
}

/**
 * Parameters accepted by {@link autoEnableGpu}.
 */
export interface AutoEnableGpuOptions {
  /** Current network node count for a single evaluation. */
  nodeCount: number;

  /** Number of networks evaluated in parallel (batch mode). Defaults to 0. */
  batchParallelCount?: number;

  /** Optional config overrides for thresholds and disable flags. */
  config?: Partial<AccelerationConfig>;
}

/**
 * Determine whether GPU acceleration should be auto-enabled.
 *
 * The decision is `true` when either:
 * - `nodeCount` meets or exceeds the configured (or default) node threshold, OR
 * - `batchParallelCount` meets or exceeds the configured (or default) batch
 *   threshold.
 *
 * The decision is `false` when `disableGPU` is `true`, regardless of other
 * inputs.
 *
 * @param nodeCount - Current network node count.
 * @param batchParallelCount - Number of networks evaluated in parallel.
 *   Defaults to 0.
 * @param config - Optional overrides for thresholds and disable flag.
 * @returns `true` when GPU acceleration should be auto-enabled.
 *
 * @example
 * ```ts
 * if (shouldAutoEnableGpu(2048)) {
 *   console.log('Network large enough for GPU offload');
 * }
 * ```
 */
export function shouldAutoEnableGpu(
  nodeCount: number,
  batchParallelCount = 0,
  config: Partial<AccelerationConfig> = {},
): boolean {
  const resolved = resolveAccelerationConfig(
    config,
  ) as Required<AccelerationConfig>;

  if (resolved.disableGPU) {
    return false;
  }

  return (
    nodeCount >= resolved.gpuNodeThreshold ||
    batchParallelCount >= resolved.gpuBatchParallelThreshold
  );
}

/**
 * Check whether a WebGPU surface is available in the current environment.
 *
 * @returns `true` when `navigator.gpu` is present.
 */
function isGpuSurfaceAvailable(): boolean {
  return typeof navigator !== 'undefined' && !!navigator.gpu;
}

/**
 * Attempt to auto-enable GPU acceleration for the given network parameters.
 *
 * When the network is eligible (per {@link shouldAutoEnableGpu}) and a WebGPU
 * surface is available, this function requests a GPU device via
 * {@link requestGPUDevice} and returns it in the result. When the GPU is
 * available but the network is below threshold, the result carries
 * `notified: true` so the caller knows GPU is an option for larger networks.
 * When the GPU is unavailable, disabled, or the device request fails, the
 * result gracefully falls back to `enabled: false` with a human-readable reason.
 *
 * @param options - Network parameters and optional config overrides.
 * @returns Auto-enable result describing the outcome.
 *
 * @example
 * ```ts
 * const result = await autoEnableGpu({ nodeCount: 2048 });
 * if (result.enabled) {
 *   network.gpuDevice = result.gpuDevice;
 * }
 * ```
 */
export async function autoEnableGpu(
  options: AutoEnableGpuOptions,
): Promise<GpuAutoEnableResult> {
  const {
    nodeCount,
    batchParallelCount = 0,
    config: partialConfig = {},
  } = options;
  const config = resolveAccelerationConfig(
    partialConfig,
  ) as Required<AccelerationConfig>;

  // Step 1: Respect explicit disable override
  if (config.disableGPU) {
    return {
      enabled: false,
      gpuDevice: null,
      notified: false,
      reason: 'GPU disabled by configuration override',
    };
  }

  // Step 2: Check eligibility against thresholds
  const eligible = shouldAutoEnableGpu(nodeCount, batchParallelCount, config);
  const gpuAvailable = isGpuSurfaceAvailable();

  // Step 3: Notify when GPU is available but the network is below threshold
  if (!eligible) {
    if (gpuAvailable) {
      return {
        enabled: false,
        gpuDevice: null,
        notified: true,
        reason: 'GPU available but network is below the auto-enable threshold',
      };
    }

    return {
      enabled: false,
      gpuDevice: null,
      notified: false,
      reason: 'Network below auto-enable threshold and GPU not available',
    };
  }

  // Step 4: Fall back when eligible but GPU is unavailable
  if (!gpuAvailable) {
    return {
      enabled: false,
      gpuDevice: null,
      notified: false,
      reason: 'WebGPU not available in this environment',
    };
  }

  // Step 5: Request a GPU device
  const device = await requestGPUDevice().catch(() => null);

  if (device) {
    return {
      enabled: true,
      gpuDevice: device,
      notified: false,
      reason: 'GPU auto-enabled',
    };
  }

  return {
    enabled: false,
    gpuDevice: null,
    notified: false,
    reason: 'GPU device request failed',
  };
}

/**
 * Evaluate a batch of weight variants on the WebGPU backend.
 *
 * For each variant the helper applies the signed delta, dispatches a GPU forward
 * pass for every input sample, scores the stacked outputs, and restores the
 * original connection weight. The supplied WebGPU device is bound to the network
 * surface for the duration of the evaluation so the GPU activation path can be
 * forced with `{ useGPU: true }`.
 *
 * @param network - Network surface to evaluate. Must expose a connection list
 *   and an `activate` method that supports the `{ useGPU: true }` option.
 * @param variants - Candidate weight perturbations for this batch.
 * @param inputs - Input batch.
 * @param target - Target output vector.
 * @param scorer - Scoring function.
 * @param device - Live WebGPU device to use for the forward passes.
 * @returns Per-variant scores in the same order as the input `variants` array.
 *
 * @example
 * ```ts
 * const scores = await evaluateWeightVariantsOnGpu(
 *   network,
 *   [{ weightIndex: 0, delta: 0.05 }],
 *   [[0.5, 0.5]],
 *   [1.0],
 *   DEFAULT_VARIANT_SCORER,
 *   device,
 * );
 * console.log(scores); // [-0.25]
 * ```
 */
export async function evaluateWeightVariantsOnGpu(
  network: VariantEvaluationNetwork,
  variants: readonly WeightVariant[],
  inputs: WeightVariantInputs,
  target: WeightVariantTarget,
  scorer: VariantScorer,
  device: GPUDevice,
): Promise<number[]> {
  const previousDevice = network.gpuDevice;
  network.gpuDevice = device;

  const scores: number[] = [];
  try {
    for (const variant of variants) {
      scores.push(
        await evaluateVariantOnGpu(network, variant, inputs, target, scorer),
      );
    }
    return scores;
  } finally {
    network.gpuDevice = previousDevice;
  }
}

/**
 * Evaluate a single variant on the WebGPU backend.
 *
 * @param network - Network surface to evaluate.
 * @param variant - Weight perturbation to apply.
 * @param inputs - Input batch.
 * @param target - Target output vector.
 * @param scorer - Scoring function.
 * @returns Score for this variant.
 * @internal
 */
async function evaluateVariantOnGpu(
  network: VariantEvaluationNetwork,
  variant: WeightVariant,
  inputs: WeightVariantInputs,
  target: WeightVariantTarget,
  scorer: VariantScorer,
): Promise<number> {
  const connection = network.connections[variant.weightIndex];
  const originalWeight = connection?.weight;

  if (connection !== undefined) {
    connection.weight += variant.delta;
  }

  try {
    const outputs: number[][] = [];
    // Small networks pay a higher round-trip cost than CPU activation; only
    // force the GPU path when the network is large enough to benefit.
    const useGpuForActivation =
      network.nodes.length >= DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD;
    for (const input of inputs) {
      const raw = await network.activate(
        input,
        useGpuForActivation ? { useGPU: true } : undefined,
      );
      outputs.push([...raw]);
    }
    return scorer(outputs, target);
  } finally {
    if (connection !== undefined && originalWeight !== undefined) {
      connection.weight = originalWeight;
    }
  }
}
