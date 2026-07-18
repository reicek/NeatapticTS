/**
 * Configuration builder for the generic network acceleration layer.
 *
 * This module resolves a partial user config into a complete
 * {@link AccelerationConfig} by filling in environment-agnostic defaults. It
 * does not perform environment detection; that responsibility lives in the
 * acceleration detection module.
 */

import {
  DEFAULT_ACCELERATION_GPU_BATCH_PARALLEL_THRESHOLD,
  DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
  DEFAULT_ACCELERATION_MAX_WORKERS,
  DEFAULT_ACCELERATION_PARALLEL_VARIANT_COUNT,
  DEFAULT_ACCELERATION_WORKER_MIN_CORES,
} from './acceleration.constants';
import type {
  AccelerationConfig,
  AccelerationMode,
  AccelerationStatus,
  BackendMode,
} from './acceleration.types';

export type {
  AccelerationConfig,
  AccelerationMode,
  AccelerationStatus,
  BackendMode,
};

export {
  DEFAULT_ACCELERATION_GPU_BATCH_PARALLEL_THRESHOLD,
  DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
  DEFAULT_ACCELERATION_MAX_WORKERS,
  DEFAULT_ACCELERATION_PARALLEL_VARIANT_COUNT,
  DEFAULT_ACCELERATION_WORKER_MIN_CORES,
} from './acceleration.constants';

/**
 * Resolve a partial acceleration config into a complete config object.
 *
 * Unspecified fields are filled with conservative defaults. `stageVariantCounts`
 * is forwarded verbatim when supplied so NGE lifecycle consumers can read
 * per-stage variant counts from the resolved config. The returned config still
 * expresses caller intent; the final runtime backend selection is made later by
 * environment detection and the backend policy.
 *
 * @param partial - Optional user overrides. An empty object produces the default
 *   configuration.
 * @returns A complete acceleration config with all fields populated.
 *
 * @example
 * ```ts
 * const config = resolveAccelerationConfig({
 *   backend: 'gpu',
 *   parallelVariantCount: 256,
 *   stageVariantCounts: { baby: 256 },
 * });
 * console.log(config.backend); // 'gpu'
 * console.log(config.gpuNodeThreshold); // 1024
 * console.log(config.parallelVariantCount); // 256
 * console.log(config.stageVariantCounts?.baby); // 256
 * ```
 */
export function resolveAccelerationConfig(
  partial: Partial<AccelerationConfig> = {},
): AccelerationConfig {
  return {
    gpuNodeThreshold:
      partial.gpuNodeThreshold ?? DEFAULT_ACCELERATION_GPU_NODE_THRESHOLD,
    gpuBatchParallelThreshold:
      partial.gpuBatchParallelThreshold ??
      DEFAULT_ACCELERATION_GPU_BATCH_PARALLEL_THRESHOLD,
    workerMinCores:
      partial.workerMinCores ?? DEFAULT_ACCELERATION_WORKER_MIN_CORES,
    maxWorkers: partial.maxWorkers ?? DEFAULT_ACCELERATION_MAX_WORKERS,
    disableGPU: partial.disableGPU ?? false,
    disableWorkers: partial.disableWorkers ?? false,
    backend: partial.backend ?? 'auto',
    parallelVariantCount:
      partial.parallelVariantCount ??
      DEFAULT_ACCELERATION_PARALLEL_VARIANT_COUNT,
    ...(partial.stageVariantCounts
      ? { stageVariantCounts: partial.stageVariantCounts }
      : {}),
  };
}
