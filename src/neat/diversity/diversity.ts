import Network from '../../architecture/network/network';
import {
  calculateDiversityStats,
  calculateStructuralEntropy,
} from './core/diversity.core';
import type {
  CompatComputer,
  DiversityStats,
  GenomeWithMetrics,
} from './core/diversity.types';

/**
 * Controller-facing diversity summary returned by sampled population analysis.
 *
 * Read this object as one compact report with four complementary lenses:
 * lineage spread, structural size, compatibility separation, and entropy.
 * Telemetry and diagnostics consumers usually treat the fields as trend
 * signals across generations rather than as exact whole-population proofs.
 *
 * - `lineageMeanDepth` and `lineageMeanPairDist` summarize ancestry spread when
 *   genomes expose lineage depth metadata.
 * - `meanNodes`, `meanConns`, `nodeVar`, and `connVar` summarize how large and
 *   how uneven the current topologies have become.
 * - `meanCompat` summarizes sampled genetic separation using the controller's
 *   compatibility metric.
 * - `graphletEntropy` adds a cheap topology-shape signal by measuring how
 *   evenly outgoing edges are distributed inside sampled networks.
 */
export type { DiversityStats } from './core/diversity.types';

/**
 * Maximum number of genomes sampled for compatibility-distance comparisons.
 *
 * This cap exists so controller telemetry can ask, "How genetically separated
 * is this population right now?" without paying the full quadratic cost of
 * comparing every genome to every other genome in large runs.
 */
export { MAX_COMPATIBILITY_SAMPLE } from './core/diversity.core';

/**
 * Maximum number of lineage-depth values sampled for pairwise lineage spread.
 *
 * This keeps lineage-distance reporting cheap while still giving diagnostics a
 * stable coarse signal about whether ancestry depth is converging or staying
 * spread out across the current population.
 */
export { MAX_LINEAGE_PAIR_SAMPLE } from './core/diversity.core';

/**
 * Diversity-reporting helpers for NEAT populations.
 *
 * The diversity boundary answers one controller-facing question: "How varied is
 * the current population, and varied in what sense?" Rather than expose a long
 * list of low-level folds, the root chapter keeps two public read models in
 * view: one for a single network's structural shape and one for a whole
 * population summary that telemetry, diagnostics, and debugging tools can
 * compare across generations.
 *
 * The summary is intentionally sampled rather than exhaustive. Diversity reads
 * are meant to stay cheap enough to run during telemetry capture, so the root
 * API favors stable trend signals over perfect all-pairs precision.
 *
 * Read the chapter in this order:
 *
 * - `structuralEntropy()` for the single-network topology fingerprint.
 * - `computeDiversityStats()` for the controller-facing population summary.
 * - `core/` when you need the heavier aggregation mechanics, sampling details,
 *   or the narrow contracts behind the root helpers.
 *
 * ```mermaid
 * flowchart TD
 *   Population[Population genomes] --> Lineage[Sample lineage depth spread]
 *   Population --> Structure[Measure node and connection size]
 *   Population --> Compatibility[Sample compatibility distance]
 *   Population --> Entropy[Sample structural entropy]
 *   Lineage --> Summary[DiversityStats summary]
 *   Structure --> Summary
 *   Compatibility --> Summary
 *   Entropy --> Summary
 *   Summary --> Consumers[Telemetry and diagnostics consumers]
 * ```
 *
 * The root chapter stays compact on purpose. `core/` owns the reusable sampling
 * and aggregation mechanics, while this file stays focused on the controller's
 * public read flow and the meaning of the resulting summary.
 */

/**
 * Compute the Shannon-style entropy of a network's out-degree distribution.
 *
 * Structural entropy here is a lightweight topology fingerprint: it measures
 * how evenly outgoing connections are distributed across nodes. It does not
 * inspect weights or recurrent dynamics, so it works well as a cheap structural
 * diversity signal.
 *
 * Use this when you want to compare the shape of individual networks or add one
 * more structural signal beside raw node and connection counts. Higher values
 * generally mean connectivity is spread across more nodes instead of being
 * concentrated into a few hubs.
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
 * This is the controller-facing population read: it blends four evidence
 * families into one compact summary that is cheap enough to reuse during
 * telemetry capture and post-run diagnostics.
 *
 * - lineage metrics estimate how far ancestry depth has spread or collapsed,
 * - structural size metrics summarize topology growth and unevenness,
 * - compatibility sampling estimates genetic separation across the population,
 * - entropy adds a topology-shape signal that raw size counts cannot express.
 *
 * The helper intentionally samples pairwise lineage and compatibility work so
 * large populations can still produce diversity telemetry without quadratic
 * blowups. Interpret the returned object as a bounded trend report: it is best
 * for comparing generations, spotting collapse, or validating that speciation
 * and mutation pressure are still producing variety.
 *
 * @example
 * ```ts
 * const diversity = computeDiversityStats(neat.population, neat);
 *
 * if (diversity) {
 *   console.log(diversity.meanCompat, diversity.graphletEntropy);
 * }
 * ```
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

/**
 * Build a zeroed diversity snapshot when no sampled metrics exist yet.
 *
 * This helper gives controller facades and diagnostics a safe fallback object
 * whose shape matches ordinary diversity output without pretending that real
 * real sampling work has happened yet.
 *
 * @param populationSize - Population size to echo into the empty snapshot.
 * @returns Diversity stats object with zeroed aggregates.
 */
export function buildEmptyDiversityStats(
  populationSize: number,
): DiversityStats {
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
