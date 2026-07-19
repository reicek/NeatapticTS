/**
 * NGE juvenile lifecycle acceleration policy.
 *
 * Maps each NGE lifecycle stage (`embryo`, `baby`, `juvenile`, `adult`,
 * `equilibrium`) to a stage-specific {@link AccelerationConfig}. The policy keeps
 * NGE's developmental preferences explicit and separate from the generic
 * acceleration layer so the same `src/acceleration/` machinery can be reused by
 * non-NGE callers.
 *
 * The defaults encode the explore–exploit curve of the NGE lifecycle:
 * - **Embryo/baby** use a modest variant-friendly batch size and rely on CPU
 *   by default because networks are tiny and setup overhead dominates.
 * - **Juvenile/adult** may opt into `auto` backend selection once the network is
 *   large enough for GPU or worker batching to matter.
 * - **Equilibrium** falls back to CPU to keep inference deterministic and
 *   lightweight while the network is no longer structurally evolving.
 *
 * Background reading:
 * - The explore–exploit tradeoff:
 *   [Wikipedia — Exploration-exploitation dilemma](https://en.wikipedia.org/wiki/Exploration_exploitation_dilemma).
 */

import type { AccelerationConfig } from '../../acceleration/acceleration.types';
import type { NgeLifecycleStage } from './neat.nge-juvenile.lifecycle-stages';

/**
 * Build the default NGE juvenile lifecycle acceleration policy.
 *
 * The returned policy is a pure, deterministic factory: every call produces an
 * equivalent fresh object. Callers can override individual stage fields by
 * cloning the returned map and editing the desired stage.
 *
 * @returns A record mapping each NGE lifecycle stage to an `AccelerationConfig`.
 *
 * @example
 * ```ts
 * const policy = buildJuvenileLifecyclePolicy();
 * console.log(policy.stages.baby.backend); // 'cpu'
 * console.log(policy.stages.adult.backend); // 'auto'
 * ```
 */
export function buildJuvenileLifecyclePolicy(): {
  stages: Record<NgeLifecycleStage, AccelerationConfig>;
} {
  const stages: Record<NgeLifecycleStage, AccelerationConfig> = {
    embryo: {
      backend: 'cpu',
      gpuNodeThreshold: 256,
      gpuBatchParallelThreshold: 8,
      workerMinCores: 2,
      maxWorkers: 2,
      disableGPU: true,
      disableWorkers: true,
    },
    baby: {
      backend: 'cpu',
      gpuNodeThreshold: 512,
      gpuBatchParallelThreshold: 8,
      workerMinCores: 2,
      maxWorkers: 2,
      disableGPU: true,
      disableWorkers: true,
    },
    juvenile: {
      backend: 'auto',
      gpuNodeThreshold: 1_024,
      gpuBatchParallelThreshold: 8,
      workerMinCores: 2,
      maxWorkers: 4,
      disableGPU: false,
      disableWorkers: false,
    },
    adult: {
      backend: 'auto',
      gpuNodeThreshold: 1_024,
      gpuBatchParallelThreshold: 8,
      workerMinCores: 2,
      maxWorkers: 4,
      disableGPU: false,
      disableWorkers: false,
    },
    equilibrium: {
      backend: 'cpu',
      gpuNodeThreshold: 1_024,
      gpuBatchParallelThreshold: 8,
      workerMinCores: 2,
      maxWorkers: 4,
      disableGPU: true,
      disableWorkers: true,
    },
  };

  return { stages };
}
