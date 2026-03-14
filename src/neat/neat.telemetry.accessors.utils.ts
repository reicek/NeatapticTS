import type { TelemetryEntry } from './neat.types';
import type { DiversityStats } from './diversity/core/diversity.types';

/** Default limit for lineage snapshots to avoid large payloads. */
export const LINEAGE_SNAPSHOT_DEFAULT_LIMIT = 20;

/** Minimal host surface needed by telemetry accessors. */
export interface TelemetryAccessorHost {
  _telemetry?: TelemetryEntry[];
  _objectiveEvents?: { gen: number; type: 'add' | 'remove'; key: string }[];
  _diversityStats?: DiversityStats;
  _lastEvalDuration?: number;
  _lastEvolveDuration?: number;
}

/**
 * Return the telemetry buffer, defaulting to an empty array when missing.
 */
export function getTelemetryBuffer(
  host: TelemetryAccessorHost,
): TelemetryEntry[] {
  return host._telemetry ?? [];
}

/**
 * Clear the telemetry buffer in place.
 */
export function clearTelemetryBuffer(host: TelemetryAccessorHost): void {
  host._telemetry = [];
}

/**
 * Return a shallow copy of recent objective events.
 */
export function getObjectiveEventsSnapshot(
  host: TelemetryAccessorHost,
): { gen: number; type: 'add' | 'remove'; key: string }[] {
  return (host._objectiveEvents ?? []).slice();
}

/**
 * Snapshot lineage metadata for the first `limit` genomes.
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
 */
export function getCachedDiversityStats(
  host: TelemetryAccessorHost,
): DiversityStats | undefined {
  return host._diversityStats;
}

/**
 * Snapshot performance timings for evaluation and evolution steps.
 */
export function getPerformanceStatsSnapshot(host: TelemetryAccessorHost) {
  return {
    lastEvalMs: host._lastEvalDuration,
    lastEvolveMs: host._lastEvolveDuration,
  };
}
