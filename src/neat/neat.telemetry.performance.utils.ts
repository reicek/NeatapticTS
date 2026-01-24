import type { NeatOptions } from './neat.types';
import type {
  TelemetryDiversityOptions,
  TelemetryEntryRecord,
} from './neat.telemetry.types';

/**
 * Attach performance stats when configured.
 *
 * @param telemetryContext - Neat-like context with performance data.
 * @param telemetryOptions - Options controlling performance telemetry.
 * @param entry - Telemetry entry to update.
 */
export function applyPerformanceStats(
  telemetryContext: {
    _lastEvalDuration?: number;
    _lastEvolveDuration?: number;
  },
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  entry: TelemetryEntryRecord,
): void {
  // Step 1: Skip if performance telemetry is disabled.
  if (!telemetryOptions.telemetry?.performance) return;

  // Step 2: Attach duration metrics.
  entry.perf = {
    evalMs: telemetryContext._lastEvalDuration,
    evolveMs: telemetryContext._lastEvolveDuration,
  };
}
