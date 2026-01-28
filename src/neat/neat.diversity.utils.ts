import Network from '../architecture/network';

/**
 * Minimal node interface with connections.
 */
export interface NodeWithConnections {
  connections: {
    out: unknown[];
  };
}

/**
 * Minimal genome interface for diversity computations.
 */
export interface GenomeWithMetrics {
  nodes: NodeWithConnections[];
  connections: unknown[];
  _depth?: number;
}

/**
 * Minimal interface that provides a compatibility distance function.
 * Implementors should expose a compatible signature with legacy NEAT code.
 */
export interface CompatComputer {
  /**
   * Compute a compatibility (distance) value between two genomes.
   * @param a - first genome-like object
   * @param b - second genome-like object
   * @returns non-negative numeric distance (higher = more different)
   */
  _compatibilityDistance(a: GenomeWithMetrics, b: GenomeWithMetrics): number;
}

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

/** Maximum lineage sample size for pairwise depth comparisons. */
export const MAX_LINEAGE_PAIR_SAMPLE = 30;

/** Maximum population sample size for compatibility comparisons. */
export const MAX_COMPATIBILITY_SAMPLE = 25;

/**
 * Compute the Shannon-style entropy of a network's out-degree distribution.
 * @param graph - Network instance to evaluate.
 * @returns Shannon-style entropy value.
 */
export function calculateStructuralEntropy(graph: Network): number {
  /** Out-degree counts per node in the graph. */
  const outDegreeCounts = collectOutDegreeCounts(graph);
  /** Normalized probability mass over out-degree counts. */
  const degreeProbabilities = normalizeToProbabilities(outDegreeCounts);
  /** Shannon entropy computed from the probability mass. */
  const entropy = computeShannonEntropy(degreeProbabilities);

  return entropy;

  /**
   * Collect out-degree counts for each node in the graph.
   * @param network - Network instance with node connections.
   * @returns Array of out-degree counts per node.
   */
  function collectOutDegreeCounts(network: Network): number[] {
    // Step 1: Project each node into its out-degree count.
    return network.nodes.map(
      (node) =>
        // each node exposes connections.out array in current architecture
        node.connections.out.length,
    );
  }

  /**
   * Normalize counts into a probability distribution.
   * @param counts - Non-negative counts to normalize.
   * @returns Probability values that sum to 1 (for non-empty inputs).
   */
  function normalizeToProbabilities(counts: number[]): number[] {
    // Step 1: Compute a safe denominator (avoid divide by zero).
    const totalCount = sumCounts(counts) || 1;
    // Step 2: Normalize and remove zeros.
    return counts
      .map((count) => count / totalCount)
      .filter((probability) => probability > 0);
  }

  /**
   * Compute Shannon entropy H = -sum(p log p).
   * @param probabilities - Probability distribution over outcomes.
   * @returns Shannon entropy value.
   */
  function computeShannonEntropy(probabilities: number[]): number {
    // Step 1: Fold probabilities into a single entropy value.
    return probabilities.reduce(
      (accumulator, probability) =>
        accumulator - probability * Math.log(probability),
      0,
    );
  }

  /**
   * Sum counts with a stable numeric fold.
   * @param counts - Values to sum.
   * @returns Sum of the provided counts.
   */
  function sumCounts(counts: number[]): number {
    // Step 1: Fold values into a total sum.
    return counts.reduce((accumulator, value) => accumulator + value, 0);
  }
}

/**
 * Compute the arithmetic mean of a numeric array. Returns 0 for empty arrays.
 * @param values - Values to average.
 * @returns Arithmetic mean of the values.
 */
function arrayMean(values: number[]): number {
  /** Guard: return 0 when there are no values */
  if (!values.length) return 0;
  return values.reduce((total, value) => total + value, 0) / values.length;
}

/**
 * Compute the variance (population variance) of a numeric array.
 * Returns 0 for empty arrays. Uses arrayMean internally.
 * @param values - Values to evaluate.
 * @returns Population variance.
 */
function arrayVariance(values: number[]): number {
  if (!values.length) return 0;
  const meanValue = arrayMean(values);
  return arrayMean(
    values.map((value) => (value - meanValue) * (value - meanValue)),
  );
}

/**
 * Compute diversity statistics for a NEAT population.
 * @param population - array of genome-like objects (nodes, connections, optional _depth)
 * @param compatibilityComputer - object exposing _compatibilityDistance(a,b)
 * @returns DiversityStats object with all computed aggregates, or undefined if input empty
 */
export function calculateDiversityStats(
  population: GenomeWithMetrics[],
  compatibilityComputer: CompatComputer,
): DiversityStats | undefined {
  // Early exit: empty population
  if (!population.length) return undefined;
  // Step 1: Collect lineage depths for genomes that expose `_depth`.
  const lineageDepths = collectLineageDepths(population);
  // Step 2: Aggregate lineage metrics.
  const lineageMeanDepth = arrayMean(lineageDepths);
  const lineageMeanPairDist = computeMeanAbsolutePairDistance(
    lineageDepths,
    MAX_LINEAGE_PAIR_SAMPLE,
  );

  // Step 3: Map structural size metrics.
  const nodeCounts = mapNodeCounts(population);
  const connectionCounts = mapConnectionCounts(population);
  // Step 4: Aggregate structural size metrics.
  const meanNodes = arrayMean(nodeCounts);
  const meanConns = arrayMean(connectionCounts);
  const nodeVar = arrayVariance(nodeCounts);
  const connVar = arrayVariance(connectionCounts);

  // Step 5: Aggregate compatibility metrics.
  const meanCompat = computeMeanCompatibilityDistance(
    population,
    compatibilityComputer,
    MAX_COMPATIBILITY_SAMPLE,
  );

  // Step 6: Aggregate graphlet entropy.
  const graphletEntropy = computeMeanGraphletEntropy(population);

  return buildDiversityStats({
    lineageMeanDepth,
    lineageMeanPairDist,
    meanNodes,
    meanConns,
    nodeVar,
    connVar,
    meanCompat,
    graphletEntropy,
    population: population.length,
  });

  /**
   * Collect lineage depths from genomes that expose `_depth`.
   * @param genomes - Population genomes to scan for depths.
   * @returns Array of lineage depths.
   */
  function collectLineageDepths(genomes: GenomeWithMetrics[]): number[] {
    // Step 1: Filter and project depths into a list.
    return genomes.flatMap((genome) =>
      typeof genome._depth === 'number' ? [genome._depth] : [],
    );
  }

  /**
   * Map genomes to node counts.
   * @param genomes - Population genomes to count nodes from.
   * @returns Array of node counts per genome.
   */
  function mapNodeCounts(genomes: GenomeWithMetrics[]): number[] {
    // Step 1: Project each genome into its node count.
    return genomes.map((genome) => genome.nodes.length);
  }

  /**
   * Map genomes to connection counts.
   * @param genomes - Population genomes to count connections from.
   * @returns Array of connection counts per genome.
   */
  function mapConnectionCounts(genomes: GenomeWithMetrics[]): number[] {
    // Step 1: Project each genome into its connection count.
    return genomes.map((genome) => genome.connections.length);
  }

  /**
   * Compute mean absolute pairwise distance for sampled values.
   * @param values - Values to compare.
   * @param sampleLimit - Max number of values to include.
   * @returns Mean absolute pairwise distance across the sample.
   */
  function computeMeanAbsolutePairDistance(
    values: number[],
    sampleLimit: number,
  ): number {
    // Step 1: Sample the values to keep the computation bounded.
    const sampledValues = values.slice(0, sampleLimit);
    // Step 2: Fold pairwise differences into an average.
    const { sum, count } = sumPairwiseAbsoluteDifferences(sampledValues);
    return count ? sum / count : 0;
  }

  /**
   * Compute mean compatibility distance for sampled genomes.
   * @param genomes - Population genomes to compare.
   * @param compatComputer - Compatibility distance provider.
   * @param sampleLimit - Max number of genomes to include.
   * @returns Mean compatibility distance across the sample.
   */
  function computeMeanCompatibilityDistance(
    genomes: GenomeWithMetrics[],
    compatComputer: CompatComputer,
    sampleLimit: number,
  ): number {
    // Step 1: Sample the genomes to keep the computation bounded.
    const sampledGenomes = genomes.slice(0, sampleLimit);
    // Step 2: Fold pairwise compatibility values into an average.
    const { sum, count } = sumPairwiseCompatibilityDistances(
      sampledGenomes,
      compatComputer,
    );
    return count ? sum / count : 0;
  }

  /**
   * Compute mean structural entropy across the population.
   * @param genomes - Population genomes to score.
   * @returns Mean structural entropy.
   */
  function computeMeanGraphletEntropy(genomes: GenomeWithMetrics[]): number {
    // Step 1: Map genomes to structural entropy values.
    const entropies = genomes.map((genome) =>
      calculateStructuralEntropy(genome as Network),
    );
    // Step 2: Aggregate with a mean fold.
    return arrayMean(entropies);
  }

  /**
   * Sum pairwise absolute differences for a list of values.
   * @param values - Sampled values to compare.
   * @returns Accumulator with sum and count.
   */
  function sumPairwiseAbsoluteDifferences(values: number[]): {
    sum: number;
    count: number;
  } {
    // Step 1: Iterate unique value pairs and accumulate differences.
    let sum = 0;
    let count = 0;
    for (let outerIndex = 0; outerIndex < values.length; outerIndex += 1) {
      for (
        let innerIndex = outerIndex + 1;
        innerIndex < values.length;
        innerIndex += 1
      ) {
        sum += Math.abs(values[outerIndex] - values[innerIndex]);
        count += 1;
      }
    }
    return { sum, count };
  }

  /**
   * Sum pairwise compatibility distances for sampled genomes.
   * @param genomes - Sampled genomes to compare.
   * @param compatComputer - Compatibility distance provider.
   * @returns Accumulator with sum and count.
   */
  function sumPairwiseCompatibilityDistances(
    genomes: GenomeWithMetrics[],
    compatComputer: CompatComputer,
  ): { sum: number; count: number } {
    // Step 1: Iterate unique genome pairs and accumulate distances.
    let sum = 0;
    let count = 0;
    for (let outerIndex = 0; outerIndex < genomes.length; outerIndex += 1) {
      for (
        let innerIndex = outerIndex + 1;
        innerIndex < genomes.length;
        innerIndex += 1
      ) {
        sum += compatComputer._compatibilityDistance(
          genomes[outerIndex],
          genomes[innerIndex],
        );
        count += 1;
      }
    }
    return { sum, count };
  }

  /**
   * Build a DiversityStats object with all aggregate metrics.
   * @param stats - Aggregate values to return.
   * @returns DiversityStats instance.
   */
  function buildDiversityStats(stats: DiversityStats): DiversityStats {
    // Step 1: Return the computed stats object.
    return stats;
  }
}
