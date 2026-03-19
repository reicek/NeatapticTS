/**
 * Pareto-front and archive inspection helpers inside the public telemetry facade.
 *
 * This chapter is the multi-objective read-side companion to the broader
 * telemetry facade. Once a run starts ranking genomes by Pareto dominance,
 * callers usually need one of three views:
 *
 * - a compact per-genome metrics table for quick inspection,
 * - reconstructed live fronts from the current population,
 * - archived objective snapshots that can be exported or reviewed later.
 *
 * The helpers stay together because those views answer the same practical
 * question from different distances: what tradeoff structure is the controller
 * currently seeing, and what evidence has it kept around from previous steps?
 *
 * Read this chapter after the root telemetry facade when the remaining question
 * is specifically about multi-objective ranking. The root surface shows where
 * objective inspection lives overall; this file narrows that map to the
 * archive- and Pareto-oriented read helpers.
 *
 * ```mermaid
 * flowchart TD
 *   Population[Current population] --> Metrics[getMultiObjectiveMetrics()<br/>compact per-genome view]
 *   Population --> Fronts[getParetoFronts()<br/>live reconstructed fronts]
 *   Archive[Stored Pareto archive] --> Slice[getParetoArchive()<br/>recent snapshots]
 *   Slice --> Export[exportParetoFrontJSONL()<br/>portable archive export]
 *   Archive --> Clear[clearParetoArchive()<br/>reset archive state]
 * ```
 */
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
 *
 * The host contract combines the live population with the stored Pareto
 * archive because callers often need to compare current fronts against the
 * snapshots that survived earlier generations.
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
 *
 * @example
 * ```ts
 * const metrics = getMultiObjectiveMetrics(neat);
 * console.table(metrics.slice(0, 5));
 * ```
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
 * Use this when you want the live frontier grouping itself rather than a flat
 * metrics table. The helper rebuilds the front structure from the population's
 * current multi-objective annotations, which makes it useful for dashboards,
 * tests, or teaching material that needs to show how genomes separate into
 * dominance layers.
 *
 * @param host - `Neat` instance whose population should be partitioned.
 * @param maxFronts - Maximum number of fronts to reconstruct.
 * @returns Pareto fronts ordered from best to worst.
 *
 * @example
 * ```ts
 * const fronts = getParetoFronts(neat, 3);
 * console.log(fronts.map((front) => front.length));
 * ```
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
 * This is the historical companion to {@link getParetoFronts}. Instead of
 * reconstructing the current live fronts, it slices the archive the controller
 * has already decided to retain for later inspection or export.
 *
 * @param host - `Neat` instance storing archived Pareto metadata.
 * @param maxEntries - Maximum number of archive entries to return.
 * @returns Slice of the recent Pareto archive.
 *
 * @example
 * ```ts
 * const recentArchive = getParetoArchive(neat, 25);
 * console.log(recentArchive.length);
 * ```
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
 * Prefer this when archive inspection is leaving the process boundary. JSONL is
 * easy to append to files, load into notebooks, or post-process with simple
 * scripts while preserving one archived snapshot per line.
 *
 * @param host - `Neat` instance storing Pareto objective snapshots.
 * @param maxEntries - Maximum number of entries to serialize.
 * @returns JSONL payload for recent Pareto archive entries.
 *
 * @example
 * ```ts
 * const archiveJsonl = exportParetoFrontJSONL(neat, 100);
 * console.log(archiveJsonl.split('\n').at(0));
 * ```
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
 * Reach for this when a caller wants a fresh archive observation window
 * without resetting the rest of the telemetry system.
 *
 * @param host - `Neat` instance whose Pareto archive should be emptied.
 * @returns Nothing. The archive buffer is reset in place.
 */
export function clearParetoArchive(host: TelemetryFacadeArchiveHost): void {
  host._paretoArchive = [];
}
