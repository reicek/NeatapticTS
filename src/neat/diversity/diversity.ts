import Network from '../../architecture/network';
import {
  calculateDiversityStats,
  calculateStructuralEntropy,
  MAX_COMPATIBILITY_SAMPLE,
  MAX_LINEAGE_PAIR_SAMPLE,
} from './core/diversity.core';
import type {
  CompatComputer,
  DiversityStats,
  GenomeWithMetrics,
} from './core/diversity.types';

export type { DiversityStats } from './core/diversity.types';
export {
  MAX_COMPATIBILITY_SAMPLE,
  MAX_LINEAGE_PAIR_SAMPLE,
} from './core/diversity.core';

/**
 * Diversity-reporting helpers for NEAT populations.
 *
 * This root diversity chapter stays compact on purpose: it surfaces the two
 * public read models first, then points readers to `core/` for the sampled
 * aggregation mechanics and narrow telemetry types.
 *
 * - `core/` explains the sampling limits, structural entropy helpers, and the
 *   population metrics used by telemetry and diagnostics.
 */

/**
 * Compute the Shannon-style entropy of a network's out-degree distribution.
 *
 * Structural entropy here is a lightweight topology fingerprint: it measures
 * how evenly outgoing connections are distributed across nodes. It does not
 * inspect weights or recurrent dynamics, so it works well as a cheap structural
 * diversity signal.
 *
 * @param graph - Network to summarize structurally.
 * @returns Shannon-style entropy of the out-degree distribution.
 */
export function structuralEntropy(graph: Network): number {
  return calculateStructuralEntropy(graph);
}

/**
 * Compute sampled diversity statistics for a NEAT population.
 *
 * The helper intentionally samples pairwise lineage and compatibility work so
 * large populations can still produce telemetry without quadratic blowups.
 *
 * @param population - Population genomes exposing nodes, connections, and optional lineage depth.
 * @param compatibilityComputer - Compatibility-distance provider used for pair sampling.
 * @returns Aggregate diversity statistics or `undefined` when the population is empty.
 */
export function computeDiversityStats(
  population: GenomeWithMetrics[],
  compatibilityComputer: CompatComputer,
): DiversityStats | undefined {
  return calculateDiversityStats(population, compatibilityComputer);
}
