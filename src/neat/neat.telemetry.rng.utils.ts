import type { NeatOptions } from './neat.types';
import type {
  TelemetryDiversityOptions,
  TelemetryEntryRecord,
} from './neat.telemetry.types';

/**
 * Attach RNG state when configured.
 *
 * @param telemetryContext - Neat-like context with RNG state.
 * @param telemetryOptions - Options controlling RNG telemetry.
 * @param entry - Telemetry entry to update.
 */
export function applyRngState(
  telemetryContext: { _rngState?: unknown },
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  entry: TelemetryEntryRecord,
): void {
  // Step 1: Attach RNG state only when enabled and present.
  if (telemetryOptions.rngState && telemetryContext._rngState !== undefined)
    entry.rng = telemetryContext._rngState as number | undefined;
}
