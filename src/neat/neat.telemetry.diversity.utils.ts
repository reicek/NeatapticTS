import type { NeatOptions, NodeLike } from './neat.types';
import type {
  TelemetryDiversityOptions,
  TelemetryGenome,
} from './neat.telemetry.types';

/**
 * Apply fast-mode tuning to diversity sampling and novelty defaults.
 *
 * @param telemetryContext - Context object storing fast-mode tuning flag.
 * @param telemetryOptions - Options with diversity and novelty settings.
 */
export function applyFastModeDefaults(
  telemetryContext: { _fastModeTuned?: boolean },
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
): void {
  // Step 1: Skip when fast mode is disabled or already tuned.
  if (!telemetryOptions.fastMode) return;
  if (telemetryContext._fastModeTuned) return;

  // Step 2: Ensure default sampling values are populated.
  const diversityMetrics = telemetryOptions.diversityMetrics;
  if (diversityMetrics) {
    if (diversityMetrics.pairSample == null) diversityMetrics.pairSample = 20;
    if (diversityMetrics.graphletSample == null)
      diversityMetrics.graphletSample = 30;
  }

  // Step 3: Ensure novelty neighborhood defaults are populated.
  if (telemetryOptions.novelty?.enabled && telemetryOptions.novelty.k == null)
    telemetryOptions.novelty.k = 5;

  // Step 4: Mark tuning as applied.
  telemetryContext._fastModeTuned = true;
}

/**
 * Compute pairwise compatibility statistics via sampling.
 *
 * @param genomes - Population snapshot.
 * @param size - Population size.
 * @param pairSampleCount - Number of pairs to sample.
 * @param rngFactoryFn - RNG factory returning a uniform random function.
 * @param compatibilityDistance - Optional compatibility distance function.
 * @returns Mean and variance of sampled compatibilities.
 */
export function computeCompatibilityStats(
  genomes: TelemetryGenome[],
  size: number,
  pairSampleCount: number,
  rngFactoryFn: () => () => number,
  compatibilityDistance?: (a: TelemetryGenome, b: TelemetryGenome) => number,
): { meanCompat: number; varCompat: number } {
  // Step 1: Accumulate sampled distances.
  let compatibilitySum = 0;
  let compatibilitySumSq = 0;
  let compatibilityCount = 0;
  for (let sampleIndex = 0; sampleIndex < pairSampleCount; sampleIndex++) {
    if (size < 2) break;
    const rng = rngFactoryFn();
    const firstIndex = Math.floor(rng() * size);
    let secondIndex = Math.floor(rng() * size);
    if (secondIndex === firstIndex) secondIndex = (secondIndex + 1) % size;
    const distance = compatibilityDistance
      ? compatibilityDistance(genomes[firstIndex], genomes[secondIndex])
      : 0;
    compatibilitySum += distance;
    compatibilitySumSq += distance * distance;
    compatibilityCount++;
  }

  // Step 2: Derive mean and variance from the sampled distances.
  const meanCompat = compatibilityCount
    ? compatibilitySum / compatibilityCount
    : 0;
  const varCompat = compatibilityCount
    ? Math.max(
        0,
        compatibilitySumSq / compatibilityCount - meanCompat * meanCompat,
      )
    : 0;

  // Step 3: Return the computed stats.
  return { meanCompat, varCompat };
}

/**
 * Compute structural entropy mean and variance across the population.
 *
 * @param genomes - Population snapshot.
 * @param structuralEntropyFn - Function to compute entropy for a genome.
 * @returns Mean and variance of entropy values.
 */
export function computeEntropyStats(
  genomes: TelemetryGenome[],
  structuralEntropyFn: (genome: TelemetryGenome) => number,
): { meanEntropy: number; varEntropy: number } {
  // Step 1: Compute entropy per genome.
  const entropies = genomes.map((genome) => structuralEntropyFn(genome));

  // Step 2: Compute mean entropy.
  const meanEntropy =
    entropies.reduce((sum, value) => sum + value, 0) / (entropies.length || 1);

  // Step 3: Compute variance of entropy.
  const varEntropy = entropies.length
    ? entropies.reduce(
        (sum, value) => sum + (value - meanEntropy) * (value - meanEntropy),
        0,
      ) / entropies.length
    : 0;

  // Step 4: Return the entropy stats.
  return { meanEntropy, varEntropy };
}

/**
 * Sample graphlet motifs and compute entropy over their edge counts.
 *
 * @param genomes - Population snapshot.
 * @param size - Population size.
 * @param graphletSampleCount - Number of graphlets to sample.
 * @param rngFactoryFn - RNG factory returning a uniform random function.
 * @returns Graphlet entropy value.
 */
export function computeGraphletEntropy(
  genomes: TelemetryGenome[],
  size: number,
  graphletSampleCount: number,
  rngFactoryFn: () => () => number,
): number {
  // Step 1: Initialize motif buckets for 0..3 edges.
  const motifCounts = [0, 0, 0, 0];

  // Step 2: Sample graphlets.
  for (let sampleIndex = 0; sampleIndex < graphletSampleCount; sampleIndex++) {
    if (size === 0) break;
    const rng = rngFactoryFn();
    const genome = genomes[Math.floor(rng() * size)];
    if (!genome) break;
    if (genome.nodes.length < 3) continue;

    const selectedNodeIndices = pickDistinctIndices(
      genome.nodes.length,
      3,
      rng,
    );
    const selectedNodes = selectedNodeIndices.map(
      (nodeIndex) => genome.nodes[nodeIndex],
    );
    const edgeCount = countEnabledEdges(genome, selectedNodes);
    motifCounts[edgeCount]++;
  }

  // Step 3: Convert motif counts to entropy.
  const totalMotifs = motifCounts.reduce((sum, value) => sum + value, 0) || 1;
  let graphletEntropy = 0;
  for (let motifIndex = 0; motifIndex < motifCounts.length; motifIndex++) {
    const probability = motifCounts[motifIndex] / totalMotifs;
    if (probability > 0) graphletEntropy -= probability * Math.log(probability);
  }

  // Step 4: Return the graphlet entropy value.
  return graphletEntropy;
}

/**
 * Pick a fixed number of distinct random indices.
 *
 * @param upperBound - Exclusive upper bound for random indices.
 * @param count - Number of distinct indices to pick.
 * @param rng - RNG function returning values in [0,1).
 * @returns Array of distinct indices.
 */
export function pickDistinctIndices(
  upperBound: number,
  count: number,
  rng: () => number,
): number[] {
  // Step 1: Gather unique indices using a set.
  const selectedIndices = new Set<number>();
  while (selectedIndices.size < count)
    selectedIndices.add(Math.floor(rng() * upperBound));

  // Step 2: Return indices as an array.
  return Array.from(selectedIndices);
}

/**
 * Count enabled edges between the selected nodes in a genome.
 *
 * @param genome - Genome with connections to inspect.
 * @param selectedNodes - Nodes forming the graphlet sample.
 * @returns Edge count capped at 3.
 */
export function countEnabledEdges(
  genome: TelemetryGenome,
  selectedNodes: NodeLike[],
): number {
  // Step 1: Count enabled edges among selected nodes.
  let edgeCount = 0;
  for (const connection of genome.connections) {
    if (!connection.enabled) continue;
    if (
      selectedNodes.includes(connection.from) &&
      selectedNodes.includes(connection.to)
    ) {
      edgeCount++;
    }
  }

  // Step 2: Cap the edge count at 3.
  return Math.min(edgeCount, 3);
}
