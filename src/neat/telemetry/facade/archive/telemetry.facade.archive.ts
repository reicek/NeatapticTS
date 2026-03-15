import type Network from '../../../../architecture/network';
import type { NeatLikeWithObjectives } from '../../../objectives/core/objectives.types';
import type { ParetoArchiveEntry } from '../../../shared/neat.shared.types';
import {
  buildMultiObjectiveMetrics,
  DEFAULT_MAX_PARETO_FRONTS,
  DEFAULT_PARETO_ARCHIVE_JSONL_MAX,
  DEFAULT_PARETO_ARCHIVE_MAX_ENTRIES,
  exportParetoArchiveJsonl,
  reconstructParetoFronts,
  sliceParetoArchive,
} from '../../../multiobjective/metrics/multiobjective.metrics';

/**
 * Narrow telemetry-facade host surface required by the archive chapter.
 *
 * This chapter groups the public multi-objective inspection helpers so the
 * root telemetry facade can treat Pareto fronts, archive snapshots, and their
 * compact derived summaries as one concept cluster.
 */
export interface TelemetryFacadeArchiveHost extends NeatLikeWithObjectives {
  population: Network[];
  _paretoArchive: ParetoArchiveEntry[];
  _paretoObjectivesArchive: number[][];
}

/**
 * Build compact multi-objective metrics for the current population snapshot.
 *
 * This chapter keeps the highest-level Pareto inspection helpers together so a
 * caller can move from per-genome rank summaries to reconstructed fronts and
 * archived vectors without leaving the same conceptual boundary.
 *
 * @param host - `Neat` instance whose population should be summarized.
 * @returns Rank, crowding, score, and size metrics per genome.
 */
export function getMultiObjectiveMetrics(host: TelemetryFacadeArchiveHost): {
  rank: number;
  crowding: number;
  score: number;
  nodes: number;
  connections: number;
}[] {
  return buildMultiObjectiveMetrics(host.population);
}

/**
 * Reconstruct Pareto fronts from current rank annotations.
 *
 * @param host - `Neat` instance whose population should be partitioned.
 * @param maxFronts - Maximum number of fronts to reconstruct.
 * @returns Pareto fronts ordered from best to worst.
 */
export function getParetoFronts(
  host: TelemetryFacadeArchiveHost,
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
  host: TelemetryFacadeArchiveHost,
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
  host: TelemetryFacadeArchiveHost,
  maxEntries: number = DEFAULT_PARETO_ARCHIVE_JSONL_MAX,
): string {
  return exportParetoArchiveJsonl(host._paretoObjectivesArchive, maxEntries);
}

/**
 * Clear the Pareto archive metadata stored on the host.
 *
 * @param host - `Neat` instance whose Pareto archive should be emptied.
 * @returns Nothing. The archive buffer is reset in place.
 */
export function clearParetoArchive(host: TelemetryFacadeArchiveHost): void {
  host._paretoArchive = [];
}
