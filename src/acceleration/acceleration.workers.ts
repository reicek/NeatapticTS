/**
 * Generic worker auto-enable helpers for the acceleration layer.
 *
 * `shouldAutoEnableWorker` decides whether worker-thread evaluation should be
 * activated based on network size, batch parallelism, and the host's logical
 * core count. `autoEnableWorker` probes the host environment and, when
 * eligible, reports how many workers would be used.
 *
 * These helpers are environment-agnostic wrappers around worker availability.
 * They gracefully fall back to CPU when workers are disabled, unavailable, or
 * when the network is too small to justify the worker setup cost.
 *
 * Background reading:
 * - Web Workers and parallel evaluation tradeoffs are described in
 *   [Web worker (Wikipedia)](https://en.wikipedia.org/wiki/Web_worker).
 * - `navigator.hardwareConcurrency` is documented by MDN:
 *   [navigator.hardwareConcurrency](https://developer.mozilla.org/en-US/docs/Web/API/Navigator/hardwareConcurrency).
 */

import { resolveAccelerationConfig } from './acceleration.config';
import type { AccelerationConfig } from './acceleration.types';

export type { AccelerationConfig } from './acceleration.types';

/**
 * Result of a worker auto-enable attempt.
 *
 * The result surface is intentionally flat and serialisation-friendly: it tells
 * the caller whether workers were enabled, how many workers would be used,
 * records whether the caller was notified that workers exist but were not
 * enabled, and explains the outcome with a human-readable reason.
 */
export interface WorkerAutoEnableResult {
  /** Whether worker acceleration was successfully enabled. */
  readonly enabled: boolean;

  /** Number of workers that would be used when enabled; `0` otherwise. */
  readonly workerCount: number;

  /**
   * Whether the caller was notified that workers are available but were not
   * auto-enabled (for example, the network is below the threshold).
   */
  readonly notified: boolean;

  /** Human-readable explanation of the result. */
  readonly reason: string;
}

/**
 * Parameters accepted by the {@link autoEnableWorker} and
 * {@link shouldAutoEnableWorker} acceleration helpers.
 */
export interface AutoEnableWorkerOptions {
  /** Current network node count for a single evaluation. */
  nodeCount: number;

  /** Number of networks evaluated in parallel (batch mode). Defaults to 0. */
  batchParallelCount?: number;

  /** Optional config overrides for thresholds, caps, and disable flags. */
  config?: Partial<AccelerationConfig>;
}

/**
 * Minimum node count before the worker backend is considered eligible.
 *
 * Networks smaller than this threshold are evaluated on CPU because the worker
 * scheduling overhead dominates the compute cost.
 */
const DEFAULT_WORKER_NODE_THRESHOLD = 1_024;

/**
 * Minimum number of parallel candidate evaluations before worker batching is
 * preferred over sequential CPU evaluation.
 */
const DEFAULT_WORKER_BATCH_PARALLEL_THRESHOLD = 8;

/** Read the host's logical core count. */
function readHardwareConcurrency(): number {
  return (globalThis as unknown as { navigator: Navigator }).navigator
    .hardwareConcurrency;
}

/** Read the global cross-origin isolation flag. */
function readCrossOriginIsolated(): boolean {
  return (
    (globalThis as unknown as { crossOriginIsolated?: boolean })
      .crossOriginIsolated === true
  );
}

/**
 * Determine whether the host environment satisfies worker requirements.
 *
 * Combines the cross-origin isolation flag and the logical-core threshold into
 * a single boolean product so both requirements are evaluated without extra
 * short-circuit branches.
 *
 * @param config - Fully resolved acceleration configuration.
 * @returns `true` when cross-origin isolation is present and enough cores are
 *   available.
 */
function areWorkersSupported(config: Required<AccelerationConfig>): boolean {
  const hardwareConcurrency = readHardwareConcurrency();
  const meetsCoreThreshold = hardwareConcurrency >= config.workerMinCores;
  const crossOriginIsolated = readCrossOriginIsolated();
  return Boolean(Number(crossOriginIsolated) * Number(meetsCoreThreshold));
}

/**
 * Determine whether worker acceleration should be auto-enabled.
 *
 * The decision is `true` when the network is large enough (either `nodeCount`
 * or `batchParallelCount` meets the default threshold), the host has enough
 * logical cores, and cross-origin isolation is present. The decision is `false`
 * when `disableWorkers` is `true`, when an active worker pool already owns
 * execution (`hasActiveWorker`), or when any environment requirement is not
 * met.
 *
 * @param options - Network parameters and optional config overrides.
 * @returns `true` when worker acceleration should be auto-enabled.
 *
 * @example
 * ```ts
 * if (shouldAutoEnableWorker({ nodeCount: 2048 })) {
 *   console.log('Network large enough for worker evaluation');
 * }
 * ```
 */
export function shouldAutoEnableWorker(
  options: AutoEnableWorkerOptions,
): boolean {
  const {
    nodeCount,
    batchParallelCount = 0,
    config: partialConfig = {},
  } = options;
  const config = resolveAccelerationConfig(
    partialConfig,
  ) as Required<AccelerationConfig>;

  // Step 1: Respect explicit disable override and active worker ownership
  if (config.disableWorkers || partialConfig.hasActiveWorker) {
    return false;
  }

  // Step 2: Confirm the host environment supports worker threads
  if (!areWorkersSupported(config)) {
    return false;
  }

  // Step 3: Check eligibility against network-size thresholds
  return (
    nodeCount >= DEFAULT_WORKER_NODE_THRESHOLD ||
    batchParallelCount >= DEFAULT_WORKER_BATCH_PARALLEL_THRESHOLD
  );
}

/**
 * Attempt to auto-enable worker acceleration for the given network parameters.
 *
 * When the network is eligible (per {@link shouldAutoEnableWorker}) and the
 * host supports worker threads, this function returns an enabled result with
 * a worker count capped by `maxWorkers` and reserved for the main thread. When
 * workers are available but the network is below threshold, the result carries
 * `notified: true` so the caller knows workers are an option for larger
 * networks. When workers are disabled, unavailable, or the core count is
 * insufficient, the result gracefully falls back to `enabled: false` with a
 * human-readable reason.
 *
 * @param options - Network parameters and optional config overrides.
 * @returns Auto-enable result describing the outcome.
 *
 * @example
 * ```ts
 * const result = await autoEnableWorker({ nodeCount: 2048 });
 * if (result.enabled) {
 *   console.log(`Using ${result.workerCount} workers`);
 * }
 * ```
 */
export async function autoEnableWorker(
  options: AutoEnableWorkerOptions,
): Promise<WorkerAutoEnableResult> {
  const {
    nodeCount,
    batchParallelCount = 0,
    config: partialConfig = {},
  } = options;
  const config = resolveAccelerationConfig(
    partialConfig,
  ) as Required<AccelerationConfig>;

  // Step 1: Respect explicit disable override and active worker ownership
  if (config.disableWorkers || partialConfig.hasActiveWorker) {
    return {
      enabled: false,
      workerCount: 0,
      notified: false,
      reason: 'Workers disabled by configuration override',
    };
  }

  // Step 2: Probe the worker runtime environment
  const workersSupported = areWorkersSupported(config);

  // Step 3: Check eligibility against network-size thresholds
  const eligible =
    nodeCount >= DEFAULT_WORKER_NODE_THRESHOLD ||
    batchParallelCount >= DEFAULT_WORKER_BATCH_PARALLEL_THRESHOLD;

  // Step 4: Notify when workers are available but the network is below threshold
  if (!eligible) {
    return workersSupported
      ? {
          enabled: false,
          workerCount: 0,
          notified: true,
          reason:
            'Workers available but network is below the auto-enable threshold',
        }
      : {
          enabled: false,
          workerCount: 0,
          notified: false,
          reason:
            'Network below auto-enable threshold and workers not available',
        };
  }

  // Step 5: Fall back when eligible but workers are unavailable
  if (!workersSupported) {
    return {
      enabled: false,
      workerCount: 0,
      notified: false,
      reason: 'Workers not available in this environment',
    };
  }

  // Step 6: Compute the worker count, reserving one core for the main thread
  const hardwareConcurrency = readHardwareConcurrency();
  const workerCount = Math.min(
    config.maxWorkers,
    Math.max(1, hardwareConcurrency - 1),
  );

  return {
    enabled: true,
    workerCount,
    notified: false,
    reason: 'Workers auto-enabled',
  };
}
