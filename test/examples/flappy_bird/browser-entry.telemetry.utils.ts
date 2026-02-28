import {
  FLAPPY_HUD_UPDATES_WINDOW_SECONDS,
  FLAPPY_MINOR_GC_WINDOW_MS,
} from './browser-entry.constants';

/**
 * Trims timestamp samples to a sliding time window.
 *
 * @param samples - Mutable timestamp buffer.
 * @param windowMs - Window width in milliseconds.
 * @param nowMs - Current timestamp.
 * @returns Nothing.
 */
export function trimSamplesToWindow(
  samples: number[],
  windowMs: number,
  nowMs: number,
): void {
  const minimumTimestampMs = nowMs - windowMs;
  while (samples.length > 0 && samples[0] < minimumTimestampMs) {
    samples.shift();
  }
}

/**
 * Resolves HUD updates per second from the latest sample window.
 *
 * @param samples - HUD update timestamps.
 * @returns Updates-per-second estimate.
 */
export function resolveHudUpdatesPerSecond(samples: number[]): number {
  return samples.length / FLAPPY_HUD_UPDATES_WINDOW_SECONDS;
}

/**
 * Resolves events per minute from the latest sample window.
 *
 * @param samples - Event timestamps.
 * @returns Events-per-minute estimate.
 */
export function resolveEventsPerMinute(samples: number[]): number {
  return samples.length;
}

/**
 * Creates a PerformanceObserver that tracks minor GC events when supported.
 *
 * @param minorGcTimestampsMs - Mutable minor-GC timestamp buffer.
 * @returns Observer when supported; otherwise `undefined`.
 */
export function createMinorGcObserver(
  minorGcTimestampsMs: number[],
): PerformanceObserver | undefined {
  if (typeof PerformanceObserver === 'undefined') {
    return undefined;
  }

  try {
    const supportedEntryTypes =
      typeof PerformanceObserver.supportedEntryTypes === 'object'
        ? PerformanceObserver.supportedEntryTypes
        : [];
    if (!supportedEntryTypes.includes('gc')) {
      return undefined;
    }

    const observer = new PerformanceObserver((list) => {
      const nowMs = performance.now();
      list.getEntries().forEach((entry) => {
        const runtimeEntry = entry as PerformanceEntry & {
          detail?: { kind?: number };
        };
        if (runtimeEntry.detail?.kind === 1) {
          minorGcTimestampsMs.push(nowMs);
        }
      });

      trimSamplesToWindow(
        minorGcTimestampsMs,
        FLAPPY_MINOR_GC_WINDOW_MS,
        nowMs,
      );
    });

    observer.observe({ entryTypes: ['gc'] as string[] });
    return observer;
  } catch {
    return undefined;
  }
}
