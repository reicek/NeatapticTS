import Network from '../architecture/network';
import {
  calculateDiversityStats,
  calculateStructuralEntropy,
  type CompatComputer,
  type DiversityStats,
  type GenomeWithMetrics,
} from './neat.diversity.utils';

export type { DiversityStats } from './neat.diversity.utils';
export {
  MAX_COMPATIBILITY_SAMPLE,
  MAX_LINEAGE_PAIR_SAMPLE,
} from './neat.diversity.utils';

/**
 * Compute the Shannon-style entropy of a network's out-degree distribution.
 * This is a lightweight, approximate structural dispersion metric used to
 * characterise how 'spread out' connections are across nodes.
 *
 * Educational note: structural entropy here is simply H = -sum(p_i log p_i)
 * over the normalized out-degree histogram. It does not measure information
 * content of weights or dynamics, but provides a quick structural fingerprint.
 *
 * @example
 * // network-like object shape expected by this helper:
 * // const net = { nodes: [ { connections: { out: [] } }, ... ] };
 * // const h = structuralEntropy(net);
 */
export function structuralEntropy(graph: Network): number {
  return calculateStructuralEntropy(graph);
}

/**
 * Compute diversity statistics for a NEAT population.
 * This is a pure helper used by reporting and diagnostics. It intentionally
 * samples pairwise computations to keep cost bounded for large populations.
 *
 * Notes for documentation:
 * - Lineage metrics rely on genomes exposing a numeric `_depth` property.
 * - Compatibility distances are computed via the provided compatComputer
 *   which mirrors legacy code and may use historical marker logic.
 *
 * @param population - array of genome-like objects (nodes, connections, optional _depth)
 * @param compatibilityComputer - object exposing _compatibilityDistance(a,b)
 * @returns DiversityStats object with all computed aggregates, or undefined if input empty
 *
 * @example
 * const stats = computeDiversityStats(population, compatImpl);
 * console.log(`Mean nodes: ${stats?.meanNodes}`);
 */
export function computeDiversityStats(
  population: GenomeWithMetrics[],
  compatibilityComputer: CompatComputer,
): DiversityStats | undefined {
  return calculateDiversityStats(population, compatibilityComputer);
}
