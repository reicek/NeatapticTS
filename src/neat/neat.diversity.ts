import Network from '../architecture/network';

/**
 * Diversity statistics returned by computeDiversityStats.
 * Each field represents an aggregate metric for a NEAT population.
 */
export interface DiversityStats {
  /** Mean depth of lineages in the population (if genomes expose _depth). */
  lineageMeanDepth: number;
  /** Mean pairwise absolute difference between lineage depths (sampled). */
  lineageMeanPairDist: number;
  /** Mean number of nodes across genomes in the population. */
  meanNodes: number;
  /** Mean number of connections across genomes in the population. */
  meanConns: number;
  /** Variance of node counts across the population. */
  nodeVar: number;
  /** Variance of connection counts across the population. */
  connVar: number;
  /** Mean compatibility distance across a sampled subset of genome pairs. */
  meanCompat: number;
  /** Mean structural entropy (graphlet entropy) across genomes. */
  graphletEntropy: number;
  /** Population size (number of genomes given). */
  population: number;
}

/** const JSDoc short descriptions above each constant */
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
 * // const h = structuralEntropy(net as any);
 */
export function structuralEntropy(graph: Network): number {
  // method steps descriptions
  // 1) Collect out-degree for each node
  /** Array of out-degrees for each node in the network. */
  const outDegrees: number[] = graph.nodes.map(
    (node: any) =>
      // each node exposes connections.out array in current architecture
      node.connections.out.length
  );

  // 2) Normalize degrees to a probability distribution
  /** Sum of all out-degrees (used for normalization; guarded to avoid 0). */
  const totalOut = outDegrees.reduce((acc, v) => acc + v, 0) || 1;
  /** Probability mass per node computed from out-degree. */
  const probabilities = outDegrees
    .map((d) => d / totalOut)
    .filter((p) => p > 0);

  // 3) Compute Shannon entropy H = -sum p log p
  /** Accumulator for entropy value. */
  let entropy = 0;
  for (const p of probabilities) {
    entropy -= p * Math.log(p);
  }
  return entropy;
}

/**
 * Minimal interface that provides a compatibility distance function.
 * Implementors should expose a compatible signature with legacy NEAT code.
 */
interface CompatComputer {
  /**
   * Compute a compatibility (distance) value between two genomes.
   * @param a - first genome-like object
   * @param b - second genome-like object
   * @returns non-negative numeric distance (higher = more different)
   */
  _compatibilityDistance(a: any, b: any): number;
}

/**
 * Compute the arithmetic mean of a numeric array. Returns 0 for empty arrays.
 * Extracted as a helper so it can be documented/tested independently.
 */
function arrayMean(values: number[]): number {
  /** Guard: return 0 when there are no values */
  if (!values.length) return 0;
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

/**
 * Compute the variance (population variance) of a numeric array.
 * Returns 0 for empty arrays. Uses arrayMean internally.
 */
function arrayVariance(values: number[]): number {
  if (!values.length) return 0;
  const m = arrayMean(values);
  return arrayMean(values.map((v) => (v - m) * (v - m)));
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
  population: any[],
  compatibilityComputer: CompatComputer
): DiversityStats | undefined {
  // Early exit: empty population
  if (!population.length) return undefined;

  // === Lineage depth metrics ===
  // method steps descriptions
  // 1) Collect lineage depths when available (_depth is optional)
  /** Collected lineage depths from genomes that expose a numeric `_depth`. */
  const lineageDepths: number[] = [];
  for (const genome of population) {
    if (typeof (genome as any)._depth === 'number') {
      lineageDepths.push((genome as any)._depth);
    }
  }

  // 2) Compute mean lineage depth
  /** Mean depth across available lineage depths. */
  const lineageMeanDepth = arrayMean(lineageDepths);

  // 3) Compute sampled pairwise absolute depth differences (bounded 30x30)
  /** Sum of absolute differences between sampled lineage depth pairs. */
  let depthPairAbsDiffSum = 0;
  /** Count of depth pairs sampled. */
  let depthPairCount = 0;
  for (let i = 0; i < lineageDepths.length && i < 30; i++) {
    for (let j = i + 1; j < lineageDepths.length && j < 30; j++) {
      depthPairAbsDiffSum += Math.abs(lineageDepths[i] - lineageDepths[j]);
      depthPairCount++;
    }
  }
  /** Mean absolute pairwise lineage depth distance (sampled). */
  const lineageMeanPairDist = depthPairCount
    ? depthPairAbsDiffSum / depthPairCount
    : 0;

  // === Structural size metrics (nodes / connections) ===
  // method steps descriptions
  // 1) Map genomes to node & connection counts
  /** Node counts per genome in the provided population. */
  const nodeCounts = population.map((g) => g.nodes.length);
  /** Connection counts per genome in the provided population. */
  const connectionCounts = population.map((g) => g.connections.length);

  // 2) Compute means and variances
  /** Mean number of nodes across the population. */
  const meanNodes = arrayMean(nodeCounts);
  /** Mean number of connections across the population. */
  const meanConns = arrayMean(connectionCounts);
  /** Variance of node counts across the population. */
  const nodeVar = arrayVariance(nodeCounts);
  /** Variance of connection counts across the population. */
  const connVar = arrayVariance(connectionCounts);

  // === Compatibility sampling ===
  // method steps descriptions
  // Sample pairwise compatibility distances up to 25x25 to limit cost.
  /** Sum of compatibility distances across sampled pairs. */
  let compatSum = 0;
  /** Number of compatibility pairs measured. */
  let compatPairCount = 0;
  for (let i = 0; i < population.length && i < 25; i++) {
    for (let j = i + 1; j < population.length && j < 25; j++) {
      compatSum += compatibilityComputer._compatibilityDistance(
        population[i],
        population[j]
      );
      compatPairCount++;
    }
  }
  /** Mean compatibility (distance) across sampled pairs. */
  const meanCompat = compatPairCount ? compatSum / compatPairCount : 0;

  // === Graphlet / structural entropy ===
  // method steps descriptions
  // Compute structuralEntropy per genome and average the results.
  /** Mean structural entropy across the population. */
  const graphletEntropy = arrayMean(
    population.map((g) => structuralEntropy(g as Network))
  );

  // Final aggregated result
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
