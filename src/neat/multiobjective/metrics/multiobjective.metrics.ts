import type Network from '../../../architecture/network/network';

type MultiObjectiveAnnotatedNetwork = Network & {
  _moRank?: number;
  _moCrowd?: number;
};

/**
 * Read-heavy Pareto metrics and archive access helpers.
 *
 * This chapter owns the small helpers that summarize multi-objective state
 * after ranking has already happened: compact per-genome metrics,
 * reconstructed Pareto front views, bounded archive slices, and JSONL export
 * of archived objective vectors.
 *
 * The boundary is intentionally read-side only. The ranking chapters decide
 * `_moRank`, `_moCrowd`, frontier membership, and archive contents earlier in
 * the evolve loop. This file exists for the next question: once that evidence
 * has been written onto genomes and archive arrays, how should inspection code
 * read it back in a compact, stable way?
 *
 * The helpers fall into four small families:
 * - per-genome metrics for dashboards and quick diagnostics,
 * - front reconstruction from stored rank annotations,
 * - bounded archive slicing for recent-history views,
 * - JSONL export for downstream tooling.
 *
 * Read this chapter when the missing question is "how do I inspect or export
 * the current multi-objective state without re-running ranking?" Read
 * `multiobjective/` or `archive/` first if the missing context is how that
 * state was produced.
 */

/**
 * Default number of Pareto fronts returned by accessors.
 *
 * Read helpers stay deliberately bounded by default so inspection callers get a
 * useful frontier summary without accidentally materializing every tail front.
 */
export const DEFAULT_MAX_PARETO_FRONTS = 3;

/**
 * Default slice size when reading Pareto archive entries.
 *
 * This favors recent history, which is usually the most relevant window for
 * telemetry panels or interactive inspection.
 */
export const DEFAULT_PARETO_ARCHIVE_MAX_ENTRIES = 50;

/**
 * Default slice size when exporting Pareto archive as JSONL.
 *
 * Export uses a slightly larger default window than in-memory reads so offline
 * tooling can inspect a broader recent history without requiring the full
 * archive.
 */
export const DEFAULT_PARETO_ARCHIVE_JSONL_MAX = 100;

/**
 * Build lightweight multi-objective metrics for each genome in the population.
 *
 * This helper turns the transient `_moRank` and `_moCrowd` annotations into a
 * compact read model that can be rendered directly in telemetry, debug tables,
 * or quick assertions. It deliberately mixes multi-objective evidence with a
 * few structural summary fields so callers can inspect competitive position and
 * genome size in one pass.
 *
 * @param population - Ranked population with optional multi-objective
 * annotations.
 * @returns Compact metrics aligned with the current population order.
 */
export function buildMultiObjectiveMetrics(population: Network[]) {
  return population.map((genome) => ({
    rank: (genome as MultiObjectiveAnnotatedNetwork)._moRank ?? 0,
    crowding: (genome as MultiObjectiveAnnotatedNetwork)._moCrowd ?? 0,
    score: genome.score || 0,
    nodes: genome.nodes.length,
    connections: genome.connections.length,
  }));
}

/**
 * Reconstruct Pareto fronts from stored rank annotations.
 *
 * This helper is for read-time reconstruction, not ranking-time discovery.
 * Instead of rerunning dominance and crowding, it groups genomes by their
 * stored `_moRank` values and returns the leading fronts up to `maxFronts`.
 *
 * When multi-objective mode is disabled, the function falls back to one front
 * containing the whole population so callers can keep a uniform read path.
 *
 * @param population - Current population with optional `_moRank` annotations.
 * @param maxFronts - Maximum number of fronts to reconstruct.
 * @param isMultiObjectiveEnabled - Whether the controller is currently using
 * multi-objective ranking.
 * @returns Reconstructed fronts in ascending rank order.
 */
export function reconstructParetoFronts(
  population: Network[],
  maxFronts: number = DEFAULT_MAX_PARETO_FRONTS,
  isMultiObjectiveEnabled: boolean,
): Network[][] {
  if (!isMultiObjectiveEnabled) return [[...population]];

  const paretoFronts: Network[][] = [];
  for (let frontIndex = 0; frontIndex < maxFronts; frontIndex++) {
    const frontMembers = population.filter(
      (genome) =>
        ((genome as MultiObjectiveAnnotatedNetwork)._moRank ?? 0) ===
        frontIndex,
    );
    if (!frontMembers.length) break;
    paretoFronts.push(frontMembers);
  }
  return paretoFronts;
}

/**
 * Return the most recent Pareto archive entries up to the provided limit.
 *
 * Archive reads are intentionally recent-first in spirit: callers usually want
 * the freshest frontier history rather than the earliest snapshots from a long
 * run. This helper therefore keeps the slicing rule explicit and reusable.
 *
 * @param archive - Archive collection ordered from oldest to newest.
 * @param maxEntries - Maximum number of recent entries to keep.
 * @returns A trailing slice containing at most `maxEntries` items.
 */
export function sliceParetoArchive<T>(
  archive: T[],
  maxEntries: number = DEFAULT_PARETO_ARCHIVE_MAX_ENTRIES,
): T[] {
  return archive.slice(-maxEntries);
}

/**
 * Export a Pareto archive slice as JSON Lines.
 *
 * JSONL keeps each archived snapshot on its own line, which makes the output
 * easy to stream, diff, or feed into external tooling without inventing another
 * archive-specific export format.
 *
 * @param archive - Archive collection ordered from oldest to newest.
 * @param maxEntries - Maximum number of recent entries to export.
 * @returns Newline-delimited JSON for the selected archive window.
 */
export function exportParetoArchiveJsonl(
  archive: unknown[],
  maxEntries: number = DEFAULT_PARETO_ARCHIVE_JSONL_MAX,
): string {
  const archiveSlice = archive.slice(-maxEntries);
  return archiveSlice.map((entry) => JSON.stringify(entry)).join('\n');
}
