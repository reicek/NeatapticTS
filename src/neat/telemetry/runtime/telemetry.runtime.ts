import type { TelemetryEntry } from '../../shared/neat.shared.types';
import type {
  TelemetryBufferContext,
  TelemetryStreamOptions,
} from '../types/telemetry.types';

/**
 * Write-side telemetry runtime helpers shared by the recorder path.
 *
 * This chapter owns the tiny mechanics that make telemetry recording safe at
 * runtime: initialize the buffer on demand, stream entries without breaking the
 * evolution loop, and keep the in-memory history bounded. Keeping those helpers
 * together gives the telemetry subtree a direct internal chapter for the
 * "record and stream" path instead of leaving that write-side behavior in a
 * flat root utility file.
 *
 * The key idea is containment. The recorder chapter decides what a telemetry
 * entry means; this runtime layer decides how that entry can be stored and
 * observed safely while evolution is still in progress.
 * By isolating buffer creation, callback delivery, and trimming here, the
 * telemetry pipeline can remain useful to dashboards and experiments without
 * letting diagnostics logic leak back into the evolutionary control flow.
 */

/**
 * Ensure the telemetry buffer is initialized before a generation snapshot is recorded.
 *
 * This helper keeps telemetry opt-in and cheap. Runs that never inspect or
 * export telemetry do not need an eager buffer allocation, while runs that do
 * record telemetry can ask for a stable mutable array at the moment they need
 * it.
 *
 * @param telemetryContext - Neat-like context holding the mutable telemetry buffer.
 * @returns Mutable telemetry buffer used for in-memory history.
 *
 * @example
 * ```ts
 * const telemetryBuffer = ensureTelemetryBuffer(neat);
 * telemetryBuffer.push(entry);
 * ```
 */
export function ensureTelemetryBuffer(
  telemetryContext: TelemetryBufferContext,
): TelemetryEntry[] {
  // Step 1: Initialize the buffer lazily so lightweight runs do not pay setup cost upfront.
  if (!telemetryContext._telemetry) telemetryContext._telemetry = [];

  // Step 2: Return the live buffer reference used by the current controller.
  return telemetryContext._telemetry;
}

/**
 * Stream a telemetry entry when the host enables runtime callbacks.
 *
 * Callback failures are intentionally swallowed so diagnostics hooks cannot
 * destabilize an evolution run.
 * This is the main runtime safety boundary between telemetry observers and the
 * search loop: visibility is allowed, but observer failures must never become
 * control-flow failures for evolution itself.
 *
 * @param telemetryContext - Neat-like context with optional telemetry stream settings.
 * @param telemetryEntry - Entry to forward to the configured stream callback.
 * @returns Nothing. The helper only invokes the callback when the runtime opts in.
 *
 * @example
 * ```ts
 * safelyStreamTelemetryEntry(
 *   { options: { telemetryStream: { enabled: true, onEntry: console.log } } },
 *   entry,
 * );
 * ```
 */
export function safelyStreamTelemetryEntry(
  telemetryContext: { options?: TelemetryStreamOptions },
  telemetryEntry: TelemetryEntry,
): void {
  // Step 1: Resolve the optional stream callback from runtime options.
  const telemetryStream = telemetryContext.options?.telemetryStream;

  // Step 2: Exit early when streaming is disabled or the callback is absent.
  if (
    !telemetryStream?.enabled ||
    typeof telemetryStream.onEntry !== 'function'
  ) {
    return;
  }

  // Step 3: Invoke the callback without letting observer failures disrupt evolution.
  try {
    telemetryStream.onEntry(telemetryEntry);
  } catch {
    // ignored: telemetry stream callback errors should not disrupt evolution
  }
}

/**
 * Trim the telemetry buffer to its configured maximum size.
 *
 * Telemetry is useful precisely because it can span many generations, but an
 * unbounded in-memory history would turn that visibility into a memory leak.
 * This helper enforces the retention contract by keeping only the most recent
 * slice of the run.
 *
 * @param telemetryBufferRef - Buffer to trim in place.
 * @param maxEntries - Maximum number of recent entries to keep.
 * @returns Nothing. Older entries are dropped from the front of the buffer.
 *
 * @example
 * ```ts
 * trimTelemetryBuffer(neat._telemetry ?? [], 500);
 * ```
 */
export function trimTelemetryBuffer(
  telemetryBufferRef: TelemetryEntry[],
  maxEntries: number,
): void {
  // Step 1: Drop the oldest entry once the runtime history exceeds the cap.
  if (telemetryBufferRef.length > maxEntries) telemetryBufferRef.shift();
}
