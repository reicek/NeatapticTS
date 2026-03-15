import type Network from '../../../architecture/network';
import type { NeatControllerForEvolution } from '../evolve.types';

/**
 * Compute diversity stats safely if the hook exists.
 * @param internal - NEAT controller instance.
 * @returns void.
 */
export function computeDiversityStatsSafely(
  internal: NeatControllerForEvolution,
): void {
  // Step 1: Invoke optional telemetry hook.
  try {
    internal._computeDiversityStats?.();
  } catch {
    // Empty catch: diversity stats computation is optional.
  }
}

/**
 * Record telemetry if enabled.
 * @param internal - NEAT controller instance.
 * @param snapshot - Best network snapshot for the generation.
 * @returns void.
 */
export async function recordTelemetryIfEnabled(
  internal: NeatControllerForEvolution,
  snapshot: Network,
): Promise<void> {
  // Step 1: Exit if telemetry disabled.
  if (!internal.options.telemetry?.enabled) return;
  // Step 2: Build and record telemetry entry.
  const telemetry = await import('../../telemetry/recorder/telemetry.recorder');
  const entry = telemetry.buildTelemetryEntry.call(
    internal as never,
    snapshot as never,
  );
  telemetry.recordTelemetryEntry.call(internal as never, entry);
}
