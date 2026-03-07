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
 * Runtime telemetry mutable state used for rolling HUD metrics.
 */
export interface RuntimeTelemetryState {
  hudUpdateTimestampsMs: number[];
  minorGcTimestampsMs: number[];
  gcObserver?: PerformanceObserver;
}

/**
 * Creates telemetry state and attaches optional minor-GC observer.
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
