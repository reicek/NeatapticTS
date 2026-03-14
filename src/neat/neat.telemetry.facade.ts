/**
 * Public read-heavy facade helpers for Neat telemetry, objectives, and archive inspection.
 *
 * This module groups the parts of the Neat surface that mainly expose existing
 * state rather than drive evolution. Keeping them here lets [src/neat.ts](src/neat.ts)
 * stay orchestration-first while generated docs still show one place where a
 * reader can inspect telemetry, lineage, species history, Pareto fronts, and
 * cached diversity snapshots.
 */
import type Network from '../architecture/network';
import { clearObjectives, registerObjective } from './objectives/objectives';
import {
  getSpeciesHistory as getSpeciesHistoryImpl,
  getSpeciesStats as getSpeciesStatsImpl,
} from './species/species';
import {
  exportTelemetryCSV as exportTelemetryCsvImpl,
  exportTelemetryJSONL as exportTelemetryJsonlImpl,
  exportSpeciesHistoryCSV as exportSpeciesHistoryCsvImpl,
} from './neat.telemetry.exports';
import { readOperatorStats } from './neat.telemetry.operator.utils';
import {
  buildLineageSnapshot,
  clearTelemetryBuffer,
  getCachedDiversityStats,
  getObjectiveEventsSnapshot,
  getPerformanceStatsSnapshot,
  getTelemetryBuffer,
  LINEAGE_SNAPSHOT_DEFAULT_LIMIT,
} from './neat.telemetry.accessors.utils';
import type { DiversityStats } from './diversity/diversity';
import type {
  ObjectiveDescriptor,
  GenomeLike,
  OperatorStatsRecord,
  ParetoArchiveEntry,
  SpeciesHistoryEntry,
  TelemetryEntry,
} from './neat.types';
import {
  buildMultiObjectiveMetrics,
  DEFAULT_MAX_PARETO_FRONTS,
  DEFAULT_PARETO_ARCHIVE_JSONL_MAX,
  DEFAULT_PARETO_ARCHIVE_MAX_ENTRIES,
  exportParetoArchiveJsonl,
  reconstructParetoFronts,
  sliceParetoArchive,
} from './neat.multiobjective.metrics.utils';
import {
  exportSpeciesHistoryJsonl,
  SPECIES_HISTORY_JSONL_MAX_DEFAULT,
} from './species/history/species.history';
import {
  getNoveltyArchiveSize as getNoveltyArchiveSizeHelper,
  resetNoveltyArchive as resetNoveltyArchiveHelper,
} from './neat.novelty.utils';

/**
 * Narrow `Neat` surface needed by the public telemetry, objective, and archive
 * facade methods.
 *
 * This interface exists so the public [src/neat.ts](src/neat.ts) facade can
 * delegate read-heavy diagnostics and export helpers into one focused module
 * without exposing the entire controller implementation. The host shape keeps
 * the contract small: population snapshots, telemetry caches, objective
 * accessors, and archive buffers.
 */
export interface NeatTelemetryFacadeHost {
  population: Network[];
  options: {
    multiObjective?: {
      enabled?: boolean;
      objectives?: unknown[];
    };
  };
  _operatorStats: Map<string, OperatorStatsRecord>;
  _paretoArchive: ParetoArchiveEntry[];
  _paretoObjectivesArchive: number[][];
  _speciesHistory: SpeciesHistoryEntry[];
  _telemetry?: TelemetryEntry[];
  _objectiveEvents?: { gen: number; type: 'add' | 'remove'; key: string }[];
  _diversityStats?: DiversityStats;
  _lastEvalDuration?: number;
  _lastEvolveDuration?: number;
  _getObjectives(): ObjectiveDescriptor[];
  _computeDiversityStats(): DiversityStats;
}

/**
 * Return just the registered objective keys in stable order.
 *
 * This is the shortest inspection surface for tests and quick diagnostics that
 * only need to confirm which objectives are active, not the full descriptor
 * payload.
 *
 * @param host - `Neat` instance exposing objective descriptors.
 * @returns Ordered list of active objective keys.
 */
export function getObjectiveKeys(host: NeatTelemetryFacadeHost): string[] {
  return host._getObjectives().map((objective) => objective.key);
}

/**
 * Return the in-memory telemetry buffer.
 *
 * @param host - `Neat` instance storing generation telemetry snapshots.
 * @returns Telemetry entries captured so far, or an empty array when telemetry
 * is not initialized.
 */
export function getTelemetry(host: NeatTelemetryFacadeHost): TelemetryEntry[] {
  return getTelemetryBuffer(host);
}

/**
 * Export telemetry as JSON Lines so logs can stream into files or post-processors.
 *
 * @param host - `Neat` instance whose telemetry buffer should be serialized.
 * @returns JSONL payload with one telemetry object per line.
 */
export function exportTelemetryJSONL(host: NeatTelemetryFacadeHost): string {
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
  host: NeatTelemetryFacadeHost,
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
export function clearTelemetry(host: NeatTelemetryFacadeHost): void {
  clearTelemetryBuffer(host);
}

/**
 * Return a compact view of active objective descriptors.
 *
 * The full objective descriptor includes accessors and internal metadata. This
 * read model trims that down to the pieces most useful in UI surfaces and
 * debugging output: the key and whether the objective is minimized or
 * maximized.
 *
 * @param host - `Neat` instance exposing objective descriptors.
 * @returns Compact objective summaries in evaluation order.
 */
export function getObjectives(
  host: NeatTelemetryFacadeHost,
): { key: string; direction: 'max' | 'min' }[] {
  return host._getObjectives().map((objective) => ({
    key: objective.key,
    direction: objective.direction,
  }));
}

/**
 * Register or replace a custom objective.
 *
 * @param host - `Neat` instance whose multi-objective registry should change.
 * @param key - Unique objective key.
 * @param direction - Whether lower or higher values are considered better.
 * @param accessor - Function that reads the objective value from a genome.
 * @returns Nothing. The objective registry on `host` is updated in place.
 */
export function registerTelemetryObjective(
  host: NeatTelemetryFacadeHost,
  key: string,
  direction: 'min' | 'max',
  accessor: (genome: GenomeLike) => number,
): void {
  registerObjective.call(host as never, key, direction, accessor);
}

/**
 * Remove all registered custom objectives so only the default objective path remains.
 *
 * @param host - `Neat` instance whose objective registry should be cleared.
 * @returns Nothing. The helper mutates the objective registry in place.
 */
export function clearTelemetryObjectives(host: NeatTelemetryFacadeHost): void {
  clearObjectives.call(host as never);
}

/**
 * Snapshot recent objective add/remove events for telemetry consumers.
 *
 * @param host - `Neat` instance storing objective lifecycle events.
 * @returns Shallow copy of the recorded objective events.
 */
export function getObjectiveEvents(host: NeatTelemetryFacadeHost): {
  gen: number;
  type: 'add' | 'remove';
  key: string;
}[] {
  return getObjectiveEventsSnapshot(host);
}

/**
 * Return a compact lineage sample for the first genomes in the current population.
 *
 * This is meant for inspection and teaching, not for full genealogy export. It
 * keeps payloads small by clipping the returned slice while still showing which
 * genomes share parents.
 *
 * @param host - `Neat` instance whose population lineage should be sampled.
 * @param limit - Maximum number of genomes to include in the snapshot.
 * @returns Array of `{ id, parents }` lineage entries.
 */
export function getLineageSnapshot(
  host: NeatTelemetryFacadeHost,
  limit: number = LINEAGE_SNAPSHOT_DEFAULT_LIMIT,
): { id: number; parents: number[] }[] {
  return buildLineageSnapshot(
    host.population as Array<{ _id?: number; _parents?: number[] }>,
    limit,
  );
}

/**
 * Export species history as CSV rows.
 *
 * @param host - `Neat` instance whose species history should be exported.
 * @param maxEntries - Maximum number of recent history entries to include.
 * @returns CSV payload for offline species analysis.
 */
export function exportSpeciesHistoryCSV(
  host: NeatTelemetryFacadeHost,
  maxEntries: number = 200,
): string {
  return exportSpeciesHistoryCsvImpl.call(host as never, maxEntries);
}

/**
 * Export species history as JSON Lines.
 *
 * @param host - `Neat` instance whose species history should be serialized.
 * @param maxEntries - Maximum number of recent history entries to include.
 * @returns JSONL payload describing recent species history snapshots.
 */
export function exportSpeciesHistoryJSONL(
  host: NeatTelemetryFacadeHost,
  maxEntries: number = SPECIES_HISTORY_JSONL_MAX_DEFAULT,
): string {
  return exportSpeciesHistoryJsonl(host._speciesHistory, maxEntries);
}

/**
 * Return a concise summary for each current species.
 *
 * @param host - `Neat` instance whose live species registry should be summarized.
 * @returns Array of current species summaries.
 */
export function getSpeciesStats(host: NeatTelemetryFacadeHost): {
  id: number;
  size: number;
  bestScore: number;
  lastImproved: number;
}[] {
  return getSpeciesStatsImpl.call(host as never);
}

/**
 * Return recorded species history, lazily backfilling extended metrics when enabled.
 *
 * @param host - `Neat` instance storing species history snapshots.
 * @returns Historical species entries for each recorded generation.
 */
export function getSpeciesHistory(
  host: NeatTelemetryFacadeHost,
): SpeciesHistoryEntry[] {
  return getSpeciesHistoryImpl.call(host as never) as SpeciesHistoryEntry[];
}

/**
 * Return the current novelty archive size.
 *
 * @param host - `Neat` instance tracking novelty behavior descriptors.
 * @returns Number of archived novelty descriptors.
 */
export function getNoveltyArchiveSize(host: NeatTelemetryFacadeHost): number {
  return getNoveltyArchiveSizeHelper(host as never);
}

/**
 * Build compact multi-objective metrics for the current population snapshot.
 *
 * @param host - `Neat` instance whose population should be summarized.
 * @returns Rank, crowding, score, and size metrics per genome.
 */
export function getMultiObjectiveMetrics(host: NeatTelemetryFacadeHost): {
  rank: number;
  crowding: number;
  score: number;
  nodes: number;
  connections: number;
}[] {
  return buildMultiObjectiveMetrics(host.population);
}

/**
 * Return aggregated mutation/operator statistics.
 *
 * @param host - `Neat` instance recording operator attempts and successes.
 * @returns Operator summaries suitable for dashboards and debugging.
 */
export function getOperatorStats(host: NeatTelemetryFacadeHost): {
  name: string;
  success: number;
  attempts: number;
}[] {
  return readOperatorStats(host._operatorStats);
}

/**
 * Reconstruct Pareto fronts from current rank annotations.
 *
 * @param host - `Neat` instance whose population should be partitioned.
 * @param maxFronts - Maximum number of fronts to reconstruct.
 * @returns Pareto fronts ordered from best to worst.
 */
export function getParetoFronts(
  host: NeatTelemetryFacadeHost,
  maxFronts: number = DEFAULT_MAX_PARETO_FRONTS,
): Network[][] {
  return reconstructParetoFronts(
    host.population,
    maxFronts,
    Boolean(host.options.multiObjective?.enabled),
  );
}

/**
 * Return the most recent Pareto archive entries.
 *
 * @param host - `Neat` instance storing archived Pareto metadata.
 * @param maxEntries - Maximum number of archive entries to return.
 * @returns Slice of the recent Pareto archive.
 */
export function getParetoArchive(
  host: NeatTelemetryFacadeHost,
  maxEntries: number = DEFAULT_PARETO_ARCHIVE_MAX_ENTRIES,
): ParetoArchiveEntry[] {
  return sliceParetoArchive(host._paretoArchive, maxEntries);
}

/**
 * Export recent Pareto archive entries as JSON Lines.
 *
 * @param host - `Neat` instance storing Pareto objective snapshots.
 * @param maxEntries - Maximum number of entries to serialize.
 * @returns JSONL payload for recent Pareto archive entries.
 */
export function exportParetoFrontJSONL(
  host: NeatTelemetryFacadeHost,
  maxEntries: number = DEFAULT_PARETO_ARCHIVE_JSONL_MAX,
): string {
  return exportParetoArchiveJsonl(host._paretoObjectivesArchive, maxEntries);
}

/**
 * Return coarse timing metrics for the last evaluation and evolution passes.
 *
 * @param host - `Neat` instance tracking performance timings.
 * @returns Snapshot of the last evaluation and evolution durations.
 */
export function getPerformanceStats(host: NeatTelemetryFacadeHost) {
  return getPerformanceStatsSnapshot(host);
}

/**
 * Return cached diversity metrics, computing a fallback snapshot when needed.
 *
 * This keeps the public facade resilient: callers can always ask for diversity
 * stats even before a full metrics pass has run.
 *
 * @param host - `Neat` instance exposing cached diversity state.
 * @returns Diversity metrics for the current population.
 */
export function getDiversityStats(
  host: NeatTelemetryFacadeHost,
): DiversityStats {
  if (!host._diversityStats) {
    return host._computeDiversityStats();
  }

  return (
    getCachedDiversityStats(host) ??
    buildEmptyDiversityStats(host.population.length)
  );
}

/**
 * Clear the novelty archive.
 *
 * @param host - `Neat` instance whose novelty archive should be reset.
 * @returns Nothing. The archive is mutated in place.
 */
export function resetNoveltyArchive(host: NeatTelemetryFacadeHost): void {
  resetNoveltyArchiveHelper(host as never);
}

/**
 * Clear the Pareto archive metadata stored on the host.
 *
 * @param host - `Neat` instance whose Pareto archive should be emptied.
 * @returns Nothing. The archive buffer is reset in place.
 */
export function clearParetoArchive(host: NeatTelemetryFacadeHost): void {
  host._paretoArchive = [];
}

function buildEmptyDiversityStats(populationSize: number): DiversityStats {
  return {
    lineageMeanDepth: 0,
    lineageMeanPairDist: 0,
    meanNodes: 0,
    meanConns: 0,
    nodeVar: 0,
    connVar: 0,
    meanCompat: 0,
    graphletEntropy: 0,
    population: populationSize,
  };
}
