import type { TelemetryEntry } from '../../../shared/neat.shared.types';
import {
  clearTelemetryBuffer,
  getTelemetryBuffer,
} from '../../accessors/telemetry.accessors';
import {
  exportTelemetryCSV as exportTelemetryCsvImpl,
  exportTelemetryJSONL as exportTelemetryJsonlImpl,
} from '../../exports/telemetry.exports';

/**
 * Narrow telemetry-facade host surface required by the buffer/export chapter.
 *
 * This chapter exists to keep the root telemetry facade focused on public
 * orchestration while the mechanics for reading, clearing, and serializing the
 * telemetry buffer live together in one small boundary.
 */
export interface TelemetryFacadeBufferHost {
  _telemetry?: TelemetryEntry[];
}

/**
 * Return the in-memory telemetry buffer.
 *
 * This chapter groups the lowest-level telemetry reads with the export helpers
 * so callers can treat "inspect the buffer" and "serialize the buffer" as one
 * concept cluster inside the broader telemetry facade.
 *
 * @param host - `Neat` instance storing generation telemetry snapshots.
 * @returns Telemetry entries captured so far, or an empty array when telemetry
 * is not initialized.
 *
 * @example
 * ```ts
 * const recentTelemetry = getTelemetry(neat);
 * console.log(recentTelemetry.at(-1)?.gen);
 * ```
 */
export function getTelemetry(
  host: TelemetryFacadeBufferHost,
): TelemetryEntry[] {
  return getTelemetryBuffer(host);
}

/**
 * Export telemetry as JSON Lines so logs can stream into files or post-processors.
 *
 * @param host - `Neat` instance whose telemetry buffer should be serialized.
 * @returns JSONL payload with one telemetry object per line.
 */
export function exportTelemetryJSONL(host: TelemetryFacadeBufferHost): string {
  return exportTelemetryJsonlImpl.call(host as never);
}

/**
 * Export recent telemetry entries as CSV for quick spreadsheet inspection.
 *
 * @param host - `Neat` instance whose telemetry buffer should be exported.
 * @param maxEntries - Maximum number of recent entries to include.
 * @returns CSV string containing the requested telemetry window.
 */
export function exportTelemetryCSV(
  host: TelemetryFacadeBufferHost,
  maxEntries: number = 500,
): string {
  return exportTelemetryCsvImpl.call(host as never, maxEntries);
}

/**
 * Clear cached telemetry entries.
 *
 * @param host - `Neat` instance whose telemetry buffer should be reset.
 * @returns Nothing. The helper mutates the host buffer in place.
 */
export function clearTelemetry(host: TelemetryFacadeBufferHost): void {
  clearTelemetryBuffer(host);
}
