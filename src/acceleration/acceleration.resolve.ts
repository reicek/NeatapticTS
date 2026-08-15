/**
 * Acceleration-mode resolution for the generic network acceleration layer.
 *
 * `resolveAccelerationMode` turns a detected {@link AccelerationStatus} into the
 * runtime-active mode. It applies explicit caller preferences, honours an
 * active worker pool, and emits backend-change telemetry when the final mode
 * differs from the originally detected mode.
 *
 * Resolution rules, in order:
 * - An explicit `backend: 'cpu'` or `backend: 'gpu'` always wins.
 * - `backend: 'auto'` uses the mode reported by environment detection.
 * - When no backend override is given, an active worker pool (`hasActiveWorker`)
 *   takes precedence over a GPU backend.
 * - Otherwise the strongest available backend is selected: GPU, then worker,
 *   then CPU.
 *
 * CPU is always reported as available in the returned status, and the
 * capability reports from the input status are preserved so diagnostics remain
 * transparent.
 */

import type {
  AccelerationMode,
  AccelerationStatus,
  BackendMode,
} from './acceleration.types';
import type {
  AccelerationBackendChangeEvent,
  AccelerationObserver,
} from './acceleration.observer';

/** Options that influence the resolved acceleration mode selection from detected status. */
export interface ResolveAccelerationModeOptions {
  /** Explicit backend preference; defaults to auto-selection. */
  backend?: BackendMode;

  /** True when an active worker pool currently owns execution. */
  hasActiveWorker?: boolean;

  /** Optional observer that receives backend-change events. */
  observer?: AccelerationObserver;
}

/** Collect human-readable reasons from the capability reports. */
function collectGapReasons(status: AccelerationStatus): string[] {
  const reasons: string[] = [];

  if (!status.gpu.available && status.gpu.reason) {
    reasons.push(status.gpu.reason);
  }

  if (!status.worker.available && status.worker.reason) {
    reasons.push(status.worker.reason);
  }

  return reasons;
}

/** Choose the resolved acceleration mode from status and caller options. */
function chooseMode(
  status: AccelerationStatus,
  options: ResolveAccelerationModeOptions,
): AccelerationMode {
  const { backend, hasActiveWorker } = options;

  if (backend === 'cpu') {
    return 'cpu';
  }

  if (backend === 'gpu') {
    return 'gpu';
  }

  if (backend === 'worker') {
    return 'worker';
  }

  if (backend === 'auto') {
    return status.mode;
  }

  if (hasActiveWorker && status.worker.available) {
    return 'worker';
  }

  if (status.gpu.available) {
    return 'gpu';
  }

  if (status.worker.available) {
    return 'worker';
  }

  return 'cpu';
}

/** Build a backend-change event when the resolved mode differs. */
function buildChangeEvent(
  status: AccelerationStatus,
  resolvedMode: AccelerationMode,
): AccelerationBackendChangeEvent {
  return {
    previous: status.mode,
    current: resolvedMode,
    reason: `Resolved acceleration backend changed from ${status.mode} to ${resolvedMode}`,
    timestamp:
      typeof performance !== 'undefined' &&
      typeof performance.now === 'function'
        ? performance.now()
        : Date.now(),
  };
}

/**
 * Resolve the active acceleration mode from a detected status and caller options.
 *
 * The function preserves the GPU and worker capability reports, ensures CPU is
 * always reported as available, and emits an `onBackendChange` event through the
 * observer when the final mode differs from the detected mode.
 *
 * @param status - Detected acceleration status with capability reports.
 * @param options - Optional resolution controls.
 * @returns A new {@link AccelerationStatus} with the resolved active mode.
 *
 * @example
 * ```ts
 * const resolved = resolveAccelerationMode(detected, {
 *   hasActiveWorker: true,
 * });
 * console.log(resolved.mode); // 'worker' when workers own execution
 * ```
 */
export function resolveAccelerationMode(
  status: AccelerationStatus,
  options: ResolveAccelerationModeOptions = {},
): AccelerationStatus {
  const resolvedMode = chooseMode(status, options);

  if (
    resolvedMode !== status.mode &&
    options.observer &&
    typeof options.observer.onBackendChange === 'function'
  ) {
    options.observer.onBackendChange(buildChangeEvent(status, resolvedMode));
  }

  return {
    mode: resolvedMode,
    gpu: status.gpu,
    worker: status.worker,
    cpu: status.cpu ?? {
      available: true,
      reason: 'CPU fallback always available',
    },
    gapReasons: status.gapReasons ?? collectGapReasons(status),
  };
}
