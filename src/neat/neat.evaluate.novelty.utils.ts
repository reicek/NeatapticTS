import type {
  GenomeForEvaluation,
  NeatControllerForEval,
} from './neat.evaluate.utils.types';
import {
  NOVELTY_ARCHIVE_CAP,
  NOVELTY_DEFAULT_BLEND,
  NOVELTY_DEFAULT_NEIGHBORS,
} from './neat.evaluate.constants.utils';

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 * @returns void.
 */
export function runNoveltyBlendAndArchive(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
): void {
  // Step 1: Guard and evaluate novelty if enabled.
  try {
    const noveltyOptions = evaluationOptions.novelty;
    if (
      !noveltyOptions?.enabled ||
      typeof noveltyOptions.descriptor !== 'function'
    ) {
      return;
    }

    const kNeighbors = getNoveltyNeighborCount(noveltyOptions);
    const blendFactor = getNoveltyBlendFactor(noveltyOptions);
    const descriptors = buildNoveltyDescriptors(controller, noveltyOptions);
    const distanceMatrix = buildDistanceMatrix(descriptors);
    applyNoveltyToPopulation(
      controller,
      descriptors,
      distanceMatrix,
      kNeighbors,
      blendFactor,
      noveltyOptions,
    );
  } catch {
    // Intentionally ignore novelty computation errors to allow evaluation to continue
  }
}

/**
 * @param noveltyOptions - Novelty configuration.
 * @returns Number of neighbors to consider.
 */
function getNoveltyNeighborCount(
  noveltyOptions: NonNullable<NeatControllerForEval['options']['novelty']>,
): number {
  // Step 1: Enforce at least one neighbor.
  return Math.max(1, noveltyOptions.k || NOVELTY_DEFAULT_NEIGHBORS);
}

/**
 * @param noveltyOptions - Novelty configuration.
 * @returns Blend factor for novelty vs. fitness.
 */
function getNoveltyBlendFactor(
  noveltyOptions: NonNullable<NeatControllerForEval['options']['novelty']>,
): number {
  // Step 1: Default to a moderate blend if not provided.
  return noveltyOptions.blendFactor ?? NOVELTY_DEFAULT_BLEND;
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param noveltyOptions - Novelty configuration.
 * @returns Descriptor vectors for each genome.
 */
function buildNoveltyDescriptors(
  controller: NeatControllerForEval,
  noveltyOptions: NonNullable<NeatControllerForEval['options']['novelty']>,
): number[][] {
  // Step 1: Map genomes through the descriptor function.
  return controller.population.map((genome) => {
    try {
      return noveltyOptions.descriptor?.(genome) ?? [];
    } catch {
      return [];
    }
  });
}

/**
 * @param descriptors - Descriptor vectors.
 * @returns Distance matrix.
 */
function buildDistanceMatrix(descriptors: number[][]): number[][] {
  // Step 1: Compute distances between descriptor vectors.
  return descriptors.map((rowDescriptor, rowIndex) =>
    descriptors.map((columnDescriptor, columnIndex) =>
      computeDescriptorDistance(
        rowDescriptor,
        columnDescriptor,
        rowIndex === columnIndex,
      ),
    ),
  );
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param descriptors - Descriptor vectors for each genome.
 * @param distanceMatrix - Distance matrix.
 * @param kNeighbors - Neighbor count.
 * @param blendFactor - Blend factor.
 * @param noveltyOptions - Novelty configuration.
 * @returns void.
 */
function applyNoveltyToPopulation(
  controller: NeatControllerForEval,
  descriptors: number[][],
  distanceMatrix: number[][],
  kNeighbors: number,
  blendFactor: number,
  noveltyOptions: NonNullable<NeatControllerForEval['options']['novelty']>,
): void {
  // Step 1: Compute novelty and blend it into scores.
  controller.population.forEach((genome, genomeIndex) => {
    const novelty = computeNoveltyScore(
      distanceMatrix[genomeIndex],
      kNeighbors,
    );
    genome._novelty = novelty;
    blendNoveltyIntoScore(genome, novelty, blendFactor);
    addGenomeToNoveltyArchive(
      controller,
      descriptors[genomeIndex],
      novelty,
      noveltyOptions,
    );
  });
}

/**
 * @param distanceRow - Distance values for a single genome.
 * @param kNeighbors - Neighbor count.
 * @returns Novelty score.
 */
function computeNoveltyScore(
  distanceRow: number[],
  kNeighbors: number,
): number {
  // Step 1: Average the k nearest neighbors.
  const sortedDistances = distanceRow.toSorted(
    (leftDistance, rightDistance) => leftDistance - rightDistance,
  );
  const neighbors = sortedDistances.slice(1, kNeighbors + 1);
  if (neighbors.length === 0) return 0;
  return (
    neighbors.reduce((accumulated, value) => accumulated + value, 0) /
    neighbors.length
  );
}

/**
 * @param genome - Genome to update.
 * @param novelty - Novelty value.
 * @param blendFactor - Blend factor.
 * @returns void.
 */
function blendNoveltyIntoScore(
  genome: GenomeForEvaluation,
  novelty: number,
  blendFactor: number,
): void {
  // Step 1: Blend novelty into score when a numeric score is present.
  if (typeof genome.score !== 'number') return;
  genome.score =
    (1 - blendFactor) * (genome.score ?? 0) + blendFactor * novelty;
}

/**
 * @param controller - NEAT controller instance for evaluation.
 * @param descriptor - Genome descriptor.
 * @param novelty - Novelty score.
 * @param noveltyOptions - Novelty configuration.
 * @returns void.
 */
function addGenomeToNoveltyArchive(
  controller: NeatControllerForEval,
  descriptor: number[],
  novelty: number,
  noveltyOptions: NonNullable<NeatControllerForEval['options']['novelty']>,
): void {
  // Step 1: Ensure archive exists.
  if (!controller._noveltyArchive) controller._noveltyArchive = [];
  // Step 2: Evaluate archive eligibility.
  const archiveAddThreshold = noveltyOptions.archiveAddThreshold ?? Infinity;
  const shouldAdd =
    noveltyOptions.archiveAddThreshold === 0 || novelty > archiveAddThreshold;
  if (!shouldAdd) return;
  // Step 3: Append while under the cap.
  if (controller._noveltyArchive.length < NOVELTY_ARCHIVE_CAP) {
    controller._noveltyArchive.push({ desc: descriptor, novelty });
  }
}

/**
 * @param left - Left descriptor.
 * @param right - Right descriptor.
 * @param isSame - Whether the descriptors are the same index.
 * @returns Euclidean distance.
 */
function computeDescriptorDistance(
  leftDescriptor: number[],
  rightDescriptor: number[],
  isSame: boolean,
): number {
  // Step 1: Return zero for identical indices.
  if (isSame) return 0;
  // Step 2: Compute Euclidean distance on the common prefix.
  const commonLength = Math.min(leftDescriptor.length, rightDescriptor.length);
  const squaredSum = leftDescriptor
    .slice(0, commonLength)
    .reduce((accumulated, leftValue, index) => {
      const delta = leftValue - (rightDescriptor[index] ?? 0);
      return accumulated + delta * delta;
    }, 0);
  return Math.sqrt(squaredSum);
}
