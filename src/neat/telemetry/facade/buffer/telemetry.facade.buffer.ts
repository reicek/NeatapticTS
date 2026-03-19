/**
 * Telemetry-buffer inspection and export helpers inside the public facade.
 *
 * This chapter is the shortest path into the raw telemetry stream that a NEAT
 * run accumulates over time. When a caller asks "what just happened across the
 * last few generations?", this is the boundary that answers.
 *
 * The helper set stays intentionally small:
 *
 * - `getTelemetry()` returns the in-memory generation snapshots,
 * - `exportTelemetryCSV()` turns a recent window into a table for human review,
 * - `exportTelemetryJSONL()` turns the same evidence into a script-friendly
 *   streaming format,
 * - `clearTelemetry()` resets the observation window without touching the rest
 *   of the controller state.
 *
 * Read this chapter after the root telemetry facade when the remaining question
 * is specifically about recent-generation evidence rather than species,
 * multi-objective tradeoffs, or derived diversity snapshots.
 *
 * ```mermaid
 * flowchart TD
 *   Buffer[In-memory telemetry buffer] --> Inspect[getTelemetry()<br/>recent generation entries]
 *   Buffer --> Csv[exportTelemetryCSV()<br/>spreadsheet review]
 *   Buffer --> Jsonl[exportTelemetryJSONL()<br/>script or file export]
 *   Buffer --> Clear[clearTelemetry()<br/>fresh observation window]
 * ```
 */
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
 *
 * Only the telemetry array itself is required here because this subchapter is
 * deliberately about raw recorded entries, not about species registries,
 * diversity caches, or objective metadata.
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
 * Prefer this when telemetry is leaving the process boundary. JSONL keeps the
 * append-and-pipe workflow simple: one generation snapshot per line, easy to
 * write to disk, ingest from scripts, or scan in notebooks.
 *
 * @param host - `Neat` instance whose telemetry buffer should be serialized.
 * @returns JSONL payload with one telemetry object per line.
 *
 * @example
 * ```ts
 * const jsonl = exportTelemetryJSONL(neat);
 * console.log(jsonl.split('\n').at(0));
 * ```
 */
export function exportTelemetryJSONL(host: TelemetryFacadeBufferHost): string {
  return exportTelemetryJsonlImpl.call(host as never);
}

/**
 * Export recent telemetry entries as CSV for quick spreadsheet inspection.
 *
 * Use this when the consumer is a person first. CSV makes it easy to open a
 * recent telemetry window in a spreadsheet or quick table view without having
 * to parse nested JSON structures.
 *
 * @param host - `Neat` instance whose telemetry buffer should be exported.
 * @param maxEntries - Maximum number of recent entries to include.
 * @returns CSV string containing the requested telemetry window.
 *
 * @example
 * ```ts
 * const csv = exportTelemetryCSV(neat, 100);
 * console.log(csv.split('\n').slice(0, 3).join('\n'));
 * ```
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
 * Reach for this when you want a fresh telemetry observation window between
 * experiment phases without rebuilding the rest of the controller.
 *
 * @param host - `Neat` instance whose telemetry buffer should be reset.
 * @returns Nothing. The helper mutates the host buffer in place.
 */
export function clearTelemetry(host: TelemetryFacadeBufferHost): void {
  clearTelemetryBuffer(host);
}
