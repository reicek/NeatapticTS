import Network from '../../../architecture/network';
import type {
  CompatComputer,
  DiversityStats,
  GenomeWithMetrics,
} from './diversity.types';

/**
 * Maximum lineage sample size for pairwise depth comparisons.
 *
 * Lineage spread is useful for telemetry, but full all-pairs ancestry distance
 * becomes expensive quickly. This cap keeps the lineage side of the report
 * bounded while still surfacing whether ancestry depth is bunching up or
 * staying distributed.
 */
export const MAX_LINEAGE_PAIR_SAMPLE = 30;

/**
 * Maximum population sample size for compatibility comparisons.
 *
 * Compatibility distance is the most obviously quadratic part of the diversity
 * report. Sampling lets the controller estimate genetic separation cheaply
 * enough to keep diversity reporting on the hot path for telemetry.
 */
export const MAX_COMPATIBILITY_SAMPLE = 25;

/**
 * Diversity-statistics mechanics used by telemetry and diagnostics.
 *
 * This core chapter assembles one bounded diversity report from four
 * complementary measurements:
 *
 * - lineage depth spread explains how ancestry is fanning out or collapsing,
 * - node and connection counts describe topology size and uneven growth,
 * - sampled compatibility distance estimates genetic separation,
 * - structural entropy captures topology shape beyond raw size alone.
 *
 * The helpers stay deliberately cheap. Instead of running exhaustive all-pairs
 * analysis, they sample the expensive comparisons and fold the results into a
 * report that is accurate enough for telemetry trends, diagnostics, and
 * generation-over-generation comparisons.
 */

/**
 * Compute the Shannon-style entropy of a network's out-degree distribution.
 *
 * This is the chapter's shape metric. Two genomes can share similar node and
 * connection counts while still distributing edges very differently, so the
 * entropy read adds a lightweight topology fingerprint beside the raw size
 * aggregates.
 *
 * @param graph - Network instance to evaluate.
 * @returns Shannon-style entropy value.
 */
export function calculateStructuralEntropy(graph: Network): number {
  // Step 1: Collect out-degree counts for each node in the graph.
  const outDegreeCounts = graph.nodes.map(
    (node) => node.connections.out.length,
  );

  // Step 2: Normalize the counts into a probability distribution.
  const totalCount = outDegreeCounts.reduce(
    (accumulator, count) => accumulator + count,
    0,
  );
  const degreeProbabilities = outDegreeCounts
    .map((count) => count / (totalCount || 1))
    .filter((probability) => probability > 0);

  // Step 3: Fold the probabilities into Shannon entropy.
  return degreeProbabilities.reduce(
    (accumulator, probability) =>
      accumulator - probability * Math.log(probability),
    0,
  );
}

/**
 * Compute diversity statistics for a NEAT population.
 *
 * This is the orchestration step for the bounded diversity report. It projects
 * lineage depths, structure counts, sampled compatibility distances, and per-
 * genome structural entropy into one object that callers can log, chart, or
 * compare across generations.
 *
 * The returned metrics are intentionally read-oriented rather than prescriptive.
 * They help answer questions such as whether species are collapsing toward a
 * narrow family of similar genomes or whether structural experimentation is
 * still producing spread.
 *
 * @param population - Population genomes exposing nodes, connections, and optional `_depth`.
 * @param compatibilityComputer - Object exposing `_compatibilityDistance(a, b)`.
 * @returns Diversity report or `undefined` when the population is empty.
 *
 * @example
 * ```ts
 * const diversity = calculateDiversityStats(neat.population, neat);
 *
 * if (diversity) {
 *   console.log(diversity.lineageMeanPairDist, diversity.graphletEntropy);
 * }
 * ```
 */
export function calculateDiversityStats(
  population: GenomeWithMetrics[],
  compatibilityComputer: CompatComputer,
): DiversityStats | undefined {
  // Step 1: Exit early when no genomes are available.
  if (!population.length) {
    return undefined;
  }

  // Step 2: Collect lineage and structural size projections.
  const lineageDepths = population.flatMap((genome) =>
    typeof genome._depth === 'number' ? [genome._depth] : [],
  );
  const nodeCounts = population.map((genome) => genome.nodes.length);
  const connectionCounts = population.map(
    (genome) => genome.connections.length,
  );

  // Step 3: Aggregate lineage and structural size metrics.
  const lineageMeanDepth = mean(lineageDepths);
  const lineageMeanPairDist = computeMeanAbsolutePairDistance(
    lineageDepths,
    MAX_LINEAGE_PAIR_SAMPLE,
  );
  const meanNodes = mean(nodeCounts);
  const meanConns = mean(connectionCounts);
  const nodeVar = variance(nodeCounts);
  const connVar = variance(connectionCounts);

  // Step 4: Aggregate sampled compatibility and entropy metrics.
  const meanCompat = computeMeanCompatibilityDistance(
    population,
    compatibilityComputer,
    MAX_COMPATIBILITY_SAMPLE,
  );
  const graphletEntropy = mean(
    population.map((genome) => calculateStructuralEntropy(genome as Network)),
  );

  // Step 5: Return the assembled diversity report.
  return {
    lineageMeanDepth,
    lineageMeanPairDist,
    meanNodes,
    meanConns,
    nodeVar,
    connVar,
    meanCompat,
    graphletEntropy,
    population: population.length,
  };
}

/**
 * Compute the arithmetic mean of a numeric array.
 *
 * The diversity report uses this helper for the direct summary columns such as
 * average lineage depth, average node count, average connection count, and the
 * mean entropy across genomes.
 *
 * @param values - Values to average.
 * @returns Arithmetic mean, or `0` when the array is empty.
 */
function mean(values: number[]): number {
  return values.length
    ? values.reduce((total, value) => total + value, 0) / values.length
    : 0;
}

/**
 * Compute the population variance of a numeric array.
 *
 * Variance complements the raw averages by showing whether the population is
 * staying structurally tight or spreading into a wider range of topology sizes.
 *
 * @param values - Values to evaluate.
 * @returns Population variance, or `0` when the array is empty.
 */
function variance(values: number[]): number {
  if (!values.length) {
    return 0;
  }

  const meanValue = mean(values);
  return mean(values.map((value) => (value - meanValue) * (value - meanValue)));
}

/**
 * Compute the mean absolute pairwise distance across a sampled value list.
 *
 * The diversity report uses this for lineage depth because the question is not
 * just "how deep are genomes on average?" but also "how far apart are sampled
 * genomes along that depth axis?" Sampling keeps that pairwise comparison
 * cheap enough for repeated telemetry reads.
 *
 * @param values - Values to compare.
 * @param sampleLimit - Maximum number of sampled values to include.
 * @returns Mean absolute pair distance across the sampled values.
 */
function computeMeanAbsolutePairDistance(
  values: number[],
  sampleLimit: number,
): number {
  // Step 1: Sample the input values to keep the computation bounded.
  const sampledValues = values.slice(0, sampleLimit);

  // Step 2: Walk unique pairs and accumulate absolute differences.
  let distanceSum = 0;
  let pairCount = 0;

  for (
    let outerValueIndex = 0;
    outerValueIndex < sampledValues.length;
    outerValueIndex += 1
  ) {
    for (
      let innerValueIndex = outerValueIndex + 1;
      innerValueIndex < sampledValues.length;
      innerValueIndex += 1
    ) {
      distanceSum += Math.abs(
        sampledValues[outerValueIndex] - sampledValues[innerValueIndex],
      );
      pairCount += 1;
    }
  }

  return pairCount ? distanceSum / pairCount : 0;
}

/**
 * Compute the mean compatibility distance across sampled genome pairs.
 *
 * Compatibility is the most expensive dimension in the report because it needs
 * host-provided pairwise comparisons. This helper bounds the work by sampling
 * a prefix of genomes, then averaging the pair distances so callers get a
 * stable separation trend instead of an exhaustive matrix.
 *
 * @param genomes - Population genomes to compare.
 * @param compatibilityComputer - Compatibility-distance provider.
 * @param sampleLimit - Maximum number of genomes to include.
 * @returns Mean compatibility distance across the sampled pairs.
 */
function computeMeanCompatibilityDistance(
  genomes: GenomeWithMetrics[],
  compatibilityComputer: CompatComputer,
  sampleLimit: number,
): number {
  // Step 1: Sample the genomes to keep pairwise work bounded.
  const sampledGenomes = genomes.slice(0, sampleLimit);

  // Step 2: Walk unique genome pairs and accumulate compatibility distance.
  let distanceSum = 0;
  let pairCount = 0;

  for (
    let outerGenomeIndex = 0;
    outerGenomeIndex < sampledGenomes.length;
    outerGenomeIndex += 1
  ) {
    for (
      let innerGenomeIndex = outerGenomeIndex + 1;
      innerGenomeIndex < sampledGenomes.length;
      innerGenomeIndex += 1
    ) {
      distanceSum += compatibilityComputer._compatibilityDistance(
        sampledGenomes[outerGenomeIndex],
        sampledGenomes[innerGenomeIndex],
      );
      pairCount += 1;
    }
  }

  return pairCount ? distanceSum / pairCount : 0;
}
