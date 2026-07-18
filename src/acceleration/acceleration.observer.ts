/**
 * Observer and rolling-report surface for the generic network acceleration layer.
 *
 * This module defines a small, injectable callback interface for acceleration
 * lifecycle events. Callers that want telemetry supply an
 * {@link AccelerationObserver}; callers that do not care pass
 * {@link NoopAccelerationObserver}. The report type gives consumers a stable,
 * in-memory shape for tracking backend time, transitions, fallback events, and
 * telemetry without committing to a persistence format.
 */

import type {
  AccelerationBackendChangeEvent,
  AccelerationMode,
  AccelerationObserver,
  AccelerationTelemetryEvent,
} from './acceleration.types';

export type {
  AccelerationBackendChangeEvent,
  AccelerationFallbackEvent,
  AccelerationObserver,
  AccelerationTelemetryEvent,
} from './acceleration.types';

/**
 * Safe empty default observer.
 *
 * Use this when a caller does not provide an observer so the acceleration layer
 * can unconditionally invoke callbacks without null-checking at every call
 * site. It is intentionally a plain object with no-op implementations of all
 * optional callbacks — in this case, an empty object, because every callback is
 * optional.
 */
export const NoopAccelerationObserver: AccelerationObserver = {};

/**
 * Create an observer that caches the most recently reported backend mode.
 *
 * The returned observer updates an internal slot whenever `onBackendChange`
 * fires. Callers can read that slot synchronously through `getBackend`, which is
 * useful for UI surfaces (such as the racing-curriculum HUD chip) that need to
 * display the backend chosen by the actual async evaluator without repeating
 * the expensive environment probe on every frame.
 *
 * @returns An observer and a synchronous getter for the cached backend mode.
 *
 * @example
 * ```ts
 * const { observer, getBackend } = createBackendCacheObserver();
 * await autoEnableAcceleration({ nodeCount: 2048, observer });
 * console.log(getBackend()); // 'gpu', 'worker', or 'cpu'
 * ```
 */
export function createBackendCacheObserver(): {
  observer: AccelerationObserver;
  getBackend: () => AccelerationMode | null;
} {
  let backend: AccelerationMode | null = null;

  return {
    observer: {
      onBackendChange: (event) => {
        backend = event.current;
      },
    },
    getBackend: () => backend,
  };
}

/**
 * Rolling in-memory report owned by an observer instance or acceleration context.
 *
 * This type does not implement persistence; it is the shared shape that
 * reporters and dashboards can consume. Optional fields allow callers to
 * attach richer diagnostics without requiring every producer to populate them.
 */
export interface AccelerationReport {
  /** Cumulative milliseconds spent in each backend. */
  timeInBackendMs: Record<AccelerationMode, number>;

  /** Ordered history of backend transitions. */
  transitions: AccelerationBackendChangeEvent[];

  /** Total number of fallback events observed. */
  fallbackCount: number;

  /** Ordered history of telemetry events. */
  telemetry: AccelerationTelemetryEvent[];

  /** Optional pooled-buffer hit ratio, when available. */
  poolHitRatio?: number;

  /** Optional worker queue depth snapshot. */
  workerQueueDepth?: number;

  /** Optional throughput estimate in inferences per second. */
  inferencesPerSecond?: number;

  /** Optional bytes currently held in pooled buffers. */
  bytesPooled?: number;

  /** Optional micro-benchmark results used for backend selection. */
  benchmarkResults?: Array<{ backend: AccelerationMode; medianMs: number }>;
}
