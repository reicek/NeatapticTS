import type { TelemetryEntry } from './neat.types';
import type {
  TelemetryBufferContext,
  TelemetryStreamOptions,
} from './neat.telemetry.types';

/**
 * Ensure the telemetry buffer is initialized.
 *
 * @param telemetryContext - Neat-like context holding telemetry buffer.
 * @returns A mutable telemetry buffer.
 */
export function ensureTelemetryBuffer(
  telemetryContext: TelemetryBufferContext,
): TelemetryEntry[] {
  // Step 1: Initialize buffer if missing.
  if (!telemetryContext._telemetry) telemetryContext._telemetry = [];
  // Step 2: Return the buffer reference.
  return telemetryContext._telemetry;
}

/**
 * Stream telemetry entry when a stream callback is configured.
 *
 * @param telemetryContext - Neat-like context with stream settings.
 * @param telemetryEntry - Entry to stream.
 */
export function safelyStreamTelemetryEntry(
  telemetryContext: { options?: TelemetryStreamOptions },
  telemetryEntry: TelemetryEntry,
): void {
  // Step 1: Check if streaming is enabled and the callback is valid.
  try {
    const telemetryStream = telemetryContext.options?.telemetryStream;
    if (
      telemetryStream?.enabled &&
      typeof telemetryStream.onEntry === 'function'
    ) {
      telemetryStream.onEntry(telemetryEntry);
    }
  } catch {
    // ignored: telemetry stream callback errors should not disrupt evolution
  }
}

/**
 * Trim the telemetry buffer to a maximum size.
 *
 * @param telemetryBufferRef - Buffer to trim in-place.
 * @param maxEntries - Maximum entries to keep.
 */
export function trimTelemetryBuffer(
  telemetryBufferRef: TelemetryEntry[],
  maxEntries: number,
): void {
  // Step 1: Drop oldest entries when over limit.
  if (telemetryBufferRef.length > maxEntries) telemetryBufferRef.shift();
}
