/**
 * Environment detection for the generic network acceleration layer.
 *
 * `detectAcceleration()` probes the host for WebGPU and worker-thread support,
 * applies the caller's configuration, and returns a structured
 * {@link AccelerationStatus} that explains which backends are available and
 * why any backend was disabled.
 *
 * Detection is intentionally synchronous: it only inspects facts that are
 * immediately observable (`navigator.gpu` presence, `hardwareConcurrency`,
 * `crossOriginIsolated`). GPU adapter resolution is asynchronous and can only
 * be confirmed at activation time; the detection surface therefore reports
 * availability when the WebGPU API is present and not explicitly disabled.
 *
 * Background reading:
 * - WebGPU is described in
 *   [WebGPU (Wikipedia)](https://en.wikipedia.org/wiki/WebGPU).
 * - Web Workers and `navigator.hardwareConcurrency` are described in
 *   [Web worker (Wikipedia)](https://en.wikipedia.org/wiki/Web_worker).
 * - Cross-origin isolation and COOP/COEP are documented by MDN:
 *   [Window.crossOriginIsolated](https://developer.mozilla.org/en-US/docs/Web/API/Window/crossOriginIsolated)
 *   and
 *   [COOP and COEP](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/SharedArrayBuffer/Planned_changes).
 */

import { resolveAccelerationConfig } from './acceleration.config';
import type {
  AccelerationConfig,
  AccelerationMode,
  AccelerationStatus,
  CPUStatus,
  GPUStatus,
  WorkerStatus,
} from './acceleration.types';

export type {
  AccelerationConfig,
  AccelerationMode,
  AccelerationStatus,
} from './acceleration.types';

/** Read a value from `globalThis` in a way that is safe in Node and browsers. */
function readGlobal<T>(key: string): T | undefined {
  return (globalThis as unknown as Record<string, T | undefined>)[key];
}

/** Build the GPU availability report for the current environment. */
function detectGPU(
  config: Required<AccelerationConfig>,
  nodeCount: number,
): GPUStatus {
  if (config.disableGPU) {
    return {
      available: false,
      reason: 'GPU disabled by configuration override',
    };
  }

  if (nodeCount < config.gpuNodeThreshold) {
    return {
      available: false,
      reason: `Network below GPU node threshold (${config.gpuNodeThreshold})`,
    };
  }

  const navigatorLike = readGlobal<Navigator>('navigator');
  const gpu =
    navigatorLike && (navigatorLike as unknown as { gpu?: unknown }).gpu;

  if (
    gpu &&
    typeof (gpu as { requestAdapter?: unknown }).requestAdapter === 'function'
  ) {
    return {
      available: true,
      reason: 'WebGPU adapter API available and ready',
    };
  }

  return {
    available: false,
    reason: 'WebGPU adapter API not available',
  };
}

/** Build the worker availability report for the current environment. */
function detectWorker(
  config: Required<AccelerationConfig>,
  crossOriginIsolated: boolean,
  hardwareConcurrency: number,
): WorkerStatus {
  if (config.disableWorkers) {
    return {
      available: false,
      count: 0,
      reason: 'Workers disabled by configuration override',
    };
  }

  if (!crossOriginIsolated) {
    return {
      available: false,
      count: 0,
      reason: 'Workers require cross-origin isolation (COOP/COEP)',
    };
  }

  if (hardwareConcurrency < config.workerMinCores) {
    return {
      available: false,
      count: 0,
      reason: `Insufficient logical cores (${hardwareConcurrency} < ${config.workerMinCores})`,
    };
  }

  const count = Math.min(
    config.maxWorkers,
    Math.max(1, hardwareConcurrency - 1),
  );

  return {
    available: true,
    count,
    reason: `Workers available (${count} worker${count === 1 ? '' : 's'})`,
  };
}

/** Build the CPU fallback report. */
function detectCPU(): CPUStatus {
  return {
    available: true,
    reason: 'CPU fallback always available',
  };
}

/** Select the canonical acceleration mode from the capability reports. */
function selectMode(gpu: GPUStatus, worker: WorkerStatus): AccelerationMode {
  if (gpu.available) {
    return 'gpu';
  }
  if (worker.available) {
    return 'worker';
  }
  return 'cpu';
}

/** Collect human-readable reasons for every unavailable backend. */
function collectGapReasons(gpu: GPUStatus, worker: WorkerStatus): string[] {
  const reasons: string[] = [];

  if (!gpu.available && gpu.reason) {
    reasons.push(gpu.reason);
  }

  if (!worker.available && worker.reason) {
    reasons.push(worker.reason);
  }

  return reasons;
}

/**
 * Probe the runtime environment and return a structured acceleration status.
 *
 * This function resolves a partial config into a full config, then inspects
 * `navigator.gpu`, `navigator.hardwareConcurrency`, and the global
 * `crossOriginIsolated` flag to decide whether GPU and worker backends are
 * eligible. CPU is always reported as available so callers have a safe
 * fallback. The returned `gapReasons` array explains why any backend is
 * disabled, making backend-selection diagnostics transparent.
 *
 * Background on the APIs used for detection:
 * - [WebGPU (Wikipedia)](https://en.wikipedia.org/wiki/WebGPU)
 * - [Web worker (Wikipedia)](https://en.wikipedia.org/wiki/Web_worker)
 * - [Window.crossOriginIsolated (MDN)](https://developer.mozilla.org/en-US/docs/Web/API/Window/crossOriginIsolated)
 *
 * @param partial - Optional user overrides. An empty object produces the
 *   default configuration.
 * @param nodeCount - Number of nodes in the network being evaluated; used to
 *   decide whether the GPU backend is worth considering.
 * @returns A structured status with mode, per-backend reports, CPU fallback,
 *   and gap reasons.
 *
 * @example
 * ```ts
 * const status = detectAcceleration({}, 2048);
 * console.log(status.mode); // 'gpu', 'worker', or 'cpu'
 * console.log(status.gapReasons); // reasons for disabled backends
 * ```
 */
export function detectAcceleration(
  partial: Partial<AccelerationConfig> = {},
  nodeCount: number = 0,
): AccelerationStatus {
  const config = resolveAccelerationConfig(
    partial,
  ) as Required<AccelerationConfig>;
  const crossOriginIsolated =
    readGlobal<boolean>('crossOriginIsolated') === true;
  const navigatorLike = readGlobal<Navigator>('navigator');
  const hardwareConcurrency =
    navigatorLike && typeof navigatorLike.hardwareConcurrency === 'number'
      ? navigatorLike.hardwareConcurrency
      : 1;

  const gpu = detectGPU(config, nodeCount);
  const worker = detectWorker(config, crossOriginIsolated, hardwareConcurrency);
  const cpu = detectCPU();
  const mode = selectMode(gpu, worker);
  const gapReasons = collectGapReasons(gpu, worker);

  return {
    mode,
    gpu,
    worker,
    cpu,
    gapReasons,
  };
}
