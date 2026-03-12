import {
  createMinorGcObserver,
  resolveEventsPerMinute,
  resolveHudUpdatesPerSecond,
  trimSamplesToWindow,
} from '../browser-entry.telemetry.utils';
import {
  FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION,
  FLAPPY_HUD_OFF_TEXT,
  FLAPPY_HUD_UPDATES_WINDOW_MS,
  FLAPPY_HUD_ZERO_DECIMAL_TEXT,
  FLAPPY_MINOR_GC_WINDOW_MS,
} from '../../constants/constants';
import type { PlaybackFrameStats } from '../browser-entry.types';

/**
 * Runtime telemetry helpers for live browser HUD updates.
 *
 * The runtime tracks a small rolling window of operational signals such as HUD
 * update frequency and minor GC activity. These are not part of the simulation
 * itself; they are observability features for understanding how expensive the
 * browser playback loop is.
 */

/**
 * Runtime telemetry mutable state used for rolling HUD metrics.
 *
 * The state keeps timestamp windows rather than pre-aggregated counters so the
 * HUD can report smoothed recent rates instead of lifetime totals.
 */
export interface RuntimeTelemetryState {
  hudUpdateTimestampsMs: number[];
  minorGcTimestampsMs: number[];
  gcObserver?: PerformanceObserver;
}

/**
 * Creates telemetry state and attaches optional minor-GC observer.
 *
 * Instrumentation is feature-gated so the demo can run in a low-noise mode when
 * telemetry is not desired.
 *
 * @returns Initialized telemetry state.
 */
export function createRuntimeTelemetryState(): RuntimeTelemetryState {
  const minorGcTimestampsMs: number[] = [];
  return {
    hudUpdateTimestampsMs: [],
    minorGcTimestampsMs,
    gcObserver: FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION
      ? createMinorGcObserver(minorGcTimestampsMs)
      : undefined,
  };
}

/**
 * Disconnects runtime telemetry observers.
 *
 * This is part of runtime teardown and prevents instrumentation observers from
 * lingering after the demo has stopped.
 *
 * @param telemetryState - Runtime telemetry state.
 * @returns Nothing.
 */
export function disconnectRuntimeTelemetry(
  telemetryState: RuntimeTelemetryState,
): void {
  telemetryState.gcObserver?.disconnect();
}

/**
 * Resolves default telemetry HUD values used before first playback updates.
 *
 * The initial values make the instrumentation section self-describing even
 * before the first playback frame arrives.
 *
 * @returns Initial telemetry field values.
 */
export function resolveInitialRuntimeTelemetryHudValues(): {
  telemetryHeader: string;
  telemetryActivationsPerFrame: string;
  telemetrySimulationStepsPerRaf: string;
  telemetryHudUpdatesPerSecond: string;
  telemetryMinorGcPerMinute: string;
} {
  return {
    telemetryHeader: 'Instrumentation',
    telemetryActivationsPerFrame: FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION
      ? FLAPPY_HUD_ZERO_DECIMAL_TEXT
      : FLAPPY_HUD_OFF_TEXT,
    telemetrySimulationStepsPerRaf: FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION
      ? FLAPPY_HUD_ZERO_DECIMAL_TEXT
      : FLAPPY_HUD_OFF_TEXT,
    telemetryHudUpdatesPerSecond: FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION
      ? FLAPPY_HUD_ZERO_DECIMAL_TEXT
      : FLAPPY_HUD_OFF_TEXT,
    telemetryMinorGcPerMinute: FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION
      ? FLAPPY_HUD_ZERO_DECIMAL_TEXT
      : FLAPPY_HUD_OFF_TEXT,
  };
}

/**
 * Resolves per-frame telemetry HUD values and updates rolling windows.
 *
 * On each published playback frame, the runtime folds the new telemetry sample
 * into rolling windows and emits human-readable HUD strings.
 *
 * @param frameStats - Playback frame stats for the current frame.
 * @param telemetryState - Runtime telemetry mutable state.
 * @returns Formatted telemetry HUD values for this frame.
 */
export function resolveRuntimeTelemetryHudValues(
  frameStats: PlaybackFrameStats,
  telemetryState: RuntimeTelemetryState,
): {
  telemetryActivationsPerFrame: string;
  telemetrySimulationStepsPerRaf: string;
  telemetryHudUpdatesPerSecond: string;
  telemetryMinorGcPerMinute: string;
} {
  if (FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION) {
    const nowMs = performance.now();
    telemetryState.hudUpdateTimestampsMs.push(nowMs);
    trimSamplesToWindow(
      telemetryState.hudUpdateTimestampsMs,
      FLAPPY_HUD_UPDATES_WINDOW_MS,
      nowMs,
    );
    trimSamplesToWindow(
      telemetryState.minorGcTimestampsMs,
      FLAPPY_MINOR_GC_WINDOW_MS,
      nowMs,
    );

    return {
      telemetryActivationsPerFrame:
        frameStats.activationCallsPerFrame.toFixed(2),
      telemetrySimulationStepsPerRaf:
        frameStats.simulationStepsPerRaf.toFixed(2),
      telemetryHudUpdatesPerSecond: resolveHudUpdatesPerSecond(
        telemetryState.hudUpdateTimestampsMs,
      ).toFixed(2),
      telemetryMinorGcPerMinute: resolveEventsPerMinute(
        telemetryState.minorGcTimestampsMs,
      ).toFixed(2),
    };
  }

  return {
    telemetryActivationsPerFrame: FLAPPY_HUD_OFF_TEXT,
    telemetrySimulationStepsPerRaf: FLAPPY_HUD_OFF_TEXT,
    telemetryHudUpdatesPerSecond: FLAPPY_HUD_OFF_TEXT,
    telemetryMinorGcPerMinute: FLAPPY_HUD_OFF_TEXT,
  };
}
