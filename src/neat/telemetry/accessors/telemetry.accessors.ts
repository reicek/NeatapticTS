import type { TelemetryEntry } from '../../shared/neat.shared.types';
import type { DiversityStats } from '../../diversity/core/diversity.types';

/**
 * Shared telemetry accessors used by both the internal telemetry chapters and
 * the already-folderized public telemetry facade.
 *
 * This chapter keeps the smallest read-only host helpers together: buffer
 * reads, objective-event snapshots, lineage sampling defaults, cached diversity
 * reads, and coarse performance snapshots. Those helpers are intentionally
 * lower-level than the facade, but they are stable enough to deserve a direct
 * telemetry chapter of their own instead of staying as a flat root utility
 * file.
 *
 * The design goal is separation of audience:
 * - `facade/` answers end-user questions in `Neat` vocabulary
 * - `accessors/` answers internal telemetry questions with tiny, reusable read models
 *
 * That split matters because many telemetry chapters need the same safe reads
 * without needing the entire public inspection surface. By keeping the
 * accessors small and deterministic, the recorder, runtime, exports, and
 * facade layers can all reuse one consistent view of telemetry state.
 *
 * Read this chapter when you want to understand the lowest-level read path in
 * telemetry: how the code grabs the current buffer, snapshots recent objective
 * events, samples lineage metadata without exporting a full genealogy, and
 * exposes cached diversity and coarse timing data.
 */

/**
 * Default limit for lineage snapshots to avoid large payloads.
 *
 * The lineage view is meant for inspection, tests, and compact summaries, not
 * for exporting the full ancestry of every genome in a large population. This
 * default keeps snapshots small enough to log or render quickly while still
 * showing inheritance patterns near the front of the population.
 */
export const LINEAGE_SNAPSHOT_DEFAULT_LIMIT = 20;

/**
 * Minimal host surface needed by telemetry accessors.
 *
 * This interface is intentionally narrow. It describes the smallest slice of a
 * `Neat`-like host that the accessor helpers need in order to read telemetry
 * state safely, which keeps these helpers reusable across internal modules and
 * avoids coupling them to the full controller implementation.
 */
export interface TelemetryAccessorHost {
  _telemetry?: TelemetryEntry[];
  _objectiveEvents?: { gen: number; type: 'add' | 'remove'; key: string }[];
  _diversityStats?: DiversityStats;
  _lastEvalDuration?: number;
  _lastEvolveDuration?: number;
}

/**
 * Return the telemetry buffer, defaulting to an empty array when missing.
 *
 * This is the lowest-level read for recent telemetry history. It gives callers
 * a stable way to inspect the current in-memory buffer without forcing every
 * consumer to reimplement null checks.
 *
 * @param host - Telemetry host that may or may not have initialized its buffer.
 * @returns Telemetry entries recorded so far, or an empty array when telemetry
 * has not been started yet.
 * @example
 * const recentEntries = getTelemetryBuffer(neat).slice(-3);
 * console.log(recentEntries.length);
 */
export function getTelemetryBuffer(
  host: TelemetryAccessorHost,
): TelemetryEntry[] {
  return host._telemetry ?? [];
}

/**
 * Clear the telemetry buffer in place.
 *
 * This is intentionally tiny and mutation-oriented because several higher-level
 * helpers need one consistent reset primitive when starting a fresh observation
 * window.
 *
 * @param host - Telemetry host whose in-memory buffer should be reset.
 * @returns Nothing. The host buffer is replaced with an empty array.
 */
export function clearTelemetryBuffer(host: TelemetryAccessorHost): void {
  host._telemetry = [];
}

/**
 * Return a shallow copy of recent objective events.
 *
 * Objective events explain how the active objective set changed over time. The
 * accessor returns a copy so readers can inspect that lifecycle without
 * accidentally mutating the host's event log.
 *
 * @param host - Telemetry host storing objective add/remove events.
 * @returns Shallow copy of the current objective event list.
 * @example
 * const events = getObjectiveEventsSnapshot(neat);
 * console.table(events);
 */
export function getObjectiveEventsSnapshot(
  host: TelemetryAccessorHost,
): { gen: number; type: 'add' | 'remove'; key: string }[] {
  return (host._objectiveEvents ?? []).slice();
}

/**
 * Snapshot lineage metadata for the first `limit` genomes.
 *
 * The accessor deliberately returns only `{ id, parents }` pairs. That keeps
 * lineage snapshots lightweight enough for telemetry entries, tests, and quick
 * debugging while still exposing the key teaching signal: which genomes share
 * ancestry and how recent offspring connect back to parents.
 *
 * @param population - Population slice containing optional runtime lineage metadata.
 * @param limit - Maximum number of genomes to sample from the front of the population.
 * @returns Compact lineage entries with genome ids and parent ids.
 * @example
 * const lineage = buildLineageSnapshot(neat.population, 5);
 * console.log(lineage.map((entry) => entry.parents));
 */
export function buildLineageSnapshot(
  population: Array<{ _id?: number; _parents?: number[] }>,
  limit: number = LINEAGE_SNAPSHOT_DEFAULT_LIMIT,
): { id: number; parents: number[] }[] {
  return population.slice(0, limit).map((genome) => ({
    id: genome._id ?? -1,
    parents: Array.isArray(genome._parents) ? genome._parents.slice() : [],
  }));
}

/**
 * Read cached diversity statistics.
 *
 * This accessor does not compute diversity on demand. Its job is only to expose
 * the last cached snapshot so callers can distinguish between "no diversity has
 * been computed yet" and "diversity was computed and here is the result."
 *
 * @param host - Telemetry host exposing an optional cached diversity snapshot.
 * @returns Cached diversity statistics when available, otherwise `undefined`.
 */
export function getCachedDiversityStats(
  host: TelemetryAccessorHost,
): DiversityStats | undefined {
  return host._diversityStats;
}

/**
 * Snapshot performance timings for evaluation and evolution steps.
 *
 * These timings are intentionally coarse. They are useful for telemetry reads,
 * simple dashboards, and regression tests that need to know which phase was
 * recently expensive, but they are not meant to replace a profiler trace.
 *
 * @param host - Telemetry host storing the last evaluation and evolution durations.
 * @returns Object containing the most recent evaluation and evolution timings.
 * @example
 * const timings = getPerformanceStatsSnapshot(neat);
 * console.log(timings.lastEvalMs, timings.lastEvolveMs);
 */
export function getPerformanceStatsSnapshot(host: TelemetryAccessorHost) {
  return {
    lastEvalMs: host._lastEvalDuration,
    lastEvolveMs: host._lastEvolveDuration,
  };
}
