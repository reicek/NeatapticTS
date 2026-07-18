/**
 * Generic acceleration auto-enable orchestrator.
 *
 * `autoEnableAcceleration` combines the GPU and worker auto-enable helpers into a
 * single unified decision. It returns an {@link AccelerationStatus} with the
 * selected mode, per-backend reports, CPU fallback, and gap reasons. It honours
 * config overrides (`disableGPU`, `disableWorkers`) and emits observer telemetry
 * when a backend is selected.
 */

import { resolveAccelerationConfig } from './acceleration.config';
import { autoEnableGpu } from './acceleration.gpu';
import { autoEnableWorker } from './acceleration.workers';
import type { AccelerationObserver } from './acceleration.observer';
import type {
  AccelerationConfig,
  AccelerationMode,
  AccelerationStatus,
  GPUStatus,
  WorkerStatus,
} from './acceleration.types';

/**
 * Parameters accepted by {@link autoEnableAcceleration}.
 */
export interface AutoEnableAccelerationOptions {
  /** Current network node count for a single evaluation. */
  nodeCount: number;

  /** Number of networks evaluated in parallel (batch mode). Defaults to 0. */
  batchParallelCount?: number;

  /** Optional config overrides for thresholds, caps, and disable flags. */
  config?: Partial<AccelerationConfig>;

  /** Optional observer that receives backend-change events. */
  observer?: AccelerationObserver;
}

/**
 * Acquire a GPU device and worker count, then choose the strongest available
 * backend.
 *
 * The precedence order is GPU, then worker, then CPU. Config overrides are
 * respected by the underlying helpers, so `disableGPU` or `disableWorkers`
 * prevent the corresponding backend from being selected. When an observer is
 * supplied, an `onBackendChange` event is emitted with the selected backend.
 *
 * Backend selection and any fallback reasons are also written to the console
 * as diagnostic output, which makes the chosen path visible in browser demos
 * and integration logs without requiring an observer.
 *
 * @param options - Network parameters, optional config overrides, and observer.
 * @returns A unified acceleration status describing the selected backend.
 *
 * @example
 * ```ts
 * const status = await autoEnableAcceleration({ nodeCount: 2048 });
 * console.log(status.mode); // 'gpu', 'worker', or 'cpu'
 * ```
 */
export async function autoEnableAcceleration(
  options: AutoEnableAccelerationOptions,
): Promise<AccelerationStatus> {
  const {
    nodeCount,
    batchParallelCount = 0,
    config: partialConfig = {},
    observer,
  } = options;

  const config = resolveAccelerationConfig(partialConfig);

  const gpuResult = await autoEnableGpu({
    nodeCount,
    batchParallelCount,
    config,
  });

  const workerResult = await autoEnableWorker({
    nodeCount,
    batchParallelCount,
    config,
  });

  const gpu: GPUStatus = {
    available: gpuResult.enabled,
    reason: gpuResult.reason,
    device: gpuResult.gpuDevice,
  };

  const worker: WorkerStatus = {
    available: workerResult.enabled,
    count: workerResult.workerCount,
    reason: workerResult.reason,
  };

  const mode: AccelerationMode = gpu.available
    ? 'gpu'
    : worker.available
      ? 'worker'
      : 'cpu';

  if (mode === 'gpu') {
    console.log('[NeatapticTS Acceleration] Backend selected: gpu');
  } else if (mode === 'worker') {
    console.log(
      '[NeatapticTS Acceleration] GPU not available, falling back to: worker',
    );
    console.log(`[NeatapticTS Acceleration] Fallback reason: ${gpu.reason}`);
  } else {
    console.log('[NeatapticTS Acceleration] Falling back to: cpu');
    const reasons = [gpu.reason, worker.reason].filter(Boolean).join('; ');
    if (reasons) {
      console.log(`[NeatapticTS Acceleration] Fallback reasons: ${reasons}`);
    }
  }

  if (observer && typeof observer.onBackendChange === 'function') {
    observer.onBackendChange({
      previous: null,
      current: mode,
      reason: `Acceleration backend selected: ${mode}`,
      timestamp: Date.now(),
    });
  }

  return {
    mode,
    gpu,
    worker,
    cpu: { available: true },
  };
}
