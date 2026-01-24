import { EPSILON } from './neat.constants';

/**
 * Read a cached entropy value if it exists and belongs to the current
 * generation.
 *
 * @param generation - Current generation number.
 * @param entropyGraph - Genome-like graph object.
 * @returns Cached entropy number, or undefined when not available.
 */
export function getCachedEntropy(
  generation: number | undefined,
  entropyGraph: Record<string, unknown>,
): number | undefined {
  // Step 1: Accept cache only when generation matches.
  if (entropyGraph._entropyGen !== generation) return;
  // Step 2: Accept cache only when value is numeric.
  if (typeof entropyGraph._entropyVal !== 'number') return;
  // Step 3: Return cached entropy.
  return entropyGraph._entropyVal as number;
}

/**
 * Compute per-node degree counts for enabled connections.
 *
 * @param entropyGraph - Genome-like graph object.
 * @returns Map geneId -> degree count.
 */
export function computeDegreeCounts(entropyGraph: {
  nodes: Array<{ geneId: number }>;
  connections: Array<{
    from: { geneId: number };
    to: { geneId: number };
    enabled: boolean;
  }>;
}): Record<number, number> {
  // Step 1: Seed degree counts with all node ids at 0.
  const degreeCounts: Record<number, number> = {};
  for (const node of entropyGraph.nodes) degreeCounts[node.geneId] = 0;

  // Step 2: Add degree contributions from enabled connections.
  for (const connection of entropyGraph.connections) {
    if (!connection.enabled) continue;
    const sourceGeneId = connection.from.geneId;
    const targetGeneId = connection.to.geneId;
    if (degreeCounts[sourceGeneId] !== undefined) degreeCounts[sourceGeneId]!++;
    if (degreeCounts[targetGeneId] !== undefined) degreeCounts[targetGeneId]!++;
  }

  // Step 3: Return the degree count table.
  return degreeCounts;
}

/**
 * Build a histogram of degree frequencies from a degree-count table.
 *
 * @param counts - Map geneId -> degree count.
 * @returns Map degree -> number of nodes with that degree.
 */
export function buildDegreeHistogram(
  counts: Record<number, number>,
): Record<number, number> {
  // Step 1: Convert node degrees into degree-frequency buckets.
  const degreeHistogram: Record<number, number> = {};
  for (const degree of Object.values(counts)) {
    degreeHistogram[degree] = (degreeHistogram[degree] ?? 0) + 1;
  }
  // Step 2: Return the histogram.
  return degreeHistogram;
}

/**
 * Compute entropy from a degree-frequency histogram.
 *
 * @param histogram - Map degree -> number of nodes.
 * @param totalNodes - Total node count used to normalize into probabilities.
 * @returns Entropy value (non-negative).
 */
export function computeEntropyFromHistogram(
  histogram: Record<number, number>,
  totalNodes: number,
): number {
  // Step 1: Fold histogram into entropy scalar.
  let entropy = 0;
  for (const [degree, frequency] of Object.entries(histogram)) {
    void degree;
    const probability = frequency / totalNodes;
    if (probability > 0)
      entropy -= probability * Math.log(probability + EPSILON);
  }
  // Step 2: Return the computed entropy.
  return entropy;
}

/**
 * Cache an entropy value for the current generation on the graph object.
 *
 * @param generation - Current generation number.
 * @param entropyGraph - Genome-like graph object.
 * @param entropyValue - Entropy value to cache.
 */
export function setCachedEntropy(
  generation: number | undefined,
  entropyGraph: Record<string, unknown>,
  entropyValue: number,
): void {
  // Step 1: Tag cache with generation.
  entropyGraph._entropyGen = generation;
  // Step 2: Store computed entropy.
  entropyGraph._entropyVal = entropyValue;
}
