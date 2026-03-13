import type Network from '../architecture/network';

type MultiObjectiveAnnotatedNetwork = Network & {
  _moRank?: number;
  _moCrowd?: number;
};

/** Default number of Pareto fronts returned by accessors. */
export const DEFAULT_MAX_PARETO_FRONTS = 3;

/** Default slice size when reading Pareto archive entries. */
export const DEFAULT_PARETO_ARCHIVE_MAX_ENTRIES = 50;

/** Default slice size when exporting Pareto archive as JSONL. */
export const DEFAULT_PARETO_ARCHIVE_JSONL_MAX = 100;

/**
 * Build lightweight multi-objective metrics for each genome in the population.
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
 */
export function sliceParetoArchive<T>(
  archive: T[],
  maxEntries: number = DEFAULT_PARETO_ARCHIVE_MAX_ENTRIES,
): T[] {
  return archive.slice(-maxEntries);
}

/**
 * Export a Pareto archive slice as JSON Lines.
 */
export function exportParetoArchiveJsonl(
  archive: unknown[],
  maxEntries: number = DEFAULT_PARETO_ARCHIVE_JSONL_MAX,
): string {
  const archiveSlice = archive.slice(-maxEntries);
  return archiveSlice.map((entry) => JSON.stringify(entry)).join('\n');
}
