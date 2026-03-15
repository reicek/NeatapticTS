import type {
  GenomeForEvaluation,
  NeatControllerForEval,
} from '../shared/evaluate.types';
import {
  NOVELTY_ARCHIVE_CAP,
  NOVELTY_DEFAULT_BLEND,
  NOVELTY_DEFAULT_NEIGHBORS,
} from '../shared/evaluate.constants';

/**
 * Novelty-search helpers for the NEAT evaluate chapter.
 *
 * This chapter keeps the behavior-descriptor path together: collect
 * descriptors, compute pairwise distances, score novelty against nearby
 * neighbors, blend novelty into the current fitness score, and append archive
 * entries when they beat the configured threshold.
 */

/**
 * Compute novelty, blend it into scores, and update the novelty archive.
 *
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 */
export function runNoveltyBlendAndArchive(
  controller: NeatControllerForEval,
  evaluationOptions: NeatControllerForEval['options'],
): void {
  // Step 1: Skip safely when novelty search is disabled or misconfigured.
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
    // Novelty search is optional; errors should not stop evaluation.
  }
}

/**
 * Resolve the number of nearest neighbors used for novelty scoring.
 *
 * @param noveltyOptions - Novelty configuration.
 * @returns Neighbor count clamped to at least one.
 */
function getNoveltyNeighborCount(
  noveltyOptions: NonNullable<NeatControllerForEval['options']['novelty']>,
): number {
  return Math.max(1, noveltyOptions.k || NOVELTY_DEFAULT_NEIGHBORS);
}

/**
 * Resolve the novelty-vs-fitness blend factor.
 *
 * @param noveltyOptions - Novelty configuration.
 * @returns Blend factor used when a genome already has a numeric score.
 */
function getNoveltyBlendFactor(
  noveltyOptions: NonNullable<NeatControllerForEval['options']['novelty']>,
): number {
  return noveltyOptions.blendFactor ?? NOVELTY_DEFAULT_BLEND;
}

/**
 * Build behavior descriptors for the current population.
 *
 * @param controller - NEAT controller instance for evaluation.
 * @param noveltyOptions - Novelty configuration.
 * @returns Descriptor vectors aligned with population order.
 */
function buildNoveltyDescriptors(
  controller: NeatControllerForEval,
  noveltyOptions: NonNullable<NeatControllerForEval['options']['novelty']>,
): number[][] {
  return controller.population.map((genome) => {
    try {
      return noveltyOptions.descriptor?.(genome) ?? [];
    } catch {
      return [];
    }
  });
}

/**
 * Build the full pairwise distance matrix for the descriptor set.
 *
 * @param descriptors - Descriptor vectors for the current population.
 * @returns Dense distance matrix aligned with population order.
 */
function buildDistanceMatrix(descriptors: number[][]): number[][] {
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
 * Apply novelty scores and archive writes across the population.
 *
 * @param controller - NEAT controller instance for evaluation.
 * @param descriptors - Descriptor vectors for each genome.
 * @param distanceMatrix - Dense distance matrix.
 * @param kNeighbors - Number of nearest neighbors to average.
 * @param blendFactor - Novelty-vs-fitness blend factor.
 * @param noveltyOptions - Novelty configuration.
 */
function applyNoveltyToPopulation(
  controller: NeatControllerForEval,
  descriptors: number[][],
  distanceMatrix: number[][],
  kNeighbors: number,
  blendFactor: number,
  noveltyOptions: NonNullable<NeatControllerForEval['options']['novelty']>,
): void {
  // Step 1: Score novelty, blend it into fitness, and update the archive.
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
 * Compute a novelty score from one row of the distance matrix.
 *
 * @param distanceRow - Distance values for a single genome.
 * @param kNeighbors - Number of nearest neighbors to average.
 * @returns Novelty score for the genome.
 */
function computeNoveltyScore(
  distanceRow: number[],
  kNeighbors: number,
): number {
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
 * Blend novelty into the genome score when the score already exists.
 *
 * @param genome - Genome to update.
 * @param novelty - Computed novelty value.
 * @param blendFactor - Blend factor for novelty versus fitness.
 */
function blendNoveltyIntoScore(
  genome: GenomeForEvaluation,
  novelty: number,
  blendFactor: number,
): void {
  if (typeof genome.score !== 'number') return;
  genome.score =
    (1 - blendFactor) * (genome.score ?? 0) + blendFactor * novelty;
}

/**
 * Add a descriptor to the novelty archive when it exceeds the threshold.
 *
 * @param controller - NEAT controller instance for evaluation.
 * @param descriptor - Behavior descriptor for the current genome.
 * @param novelty - Computed novelty score.
 * @param noveltyOptions - Novelty configuration.
 */
function addGenomeToNoveltyArchive(
  controller: NeatControllerForEval,
  descriptor: number[],
  novelty: number,
  noveltyOptions: NonNullable<NeatControllerForEval['options']['novelty']>,
): void {
  // Step 1: Ensure the backing archive exists.
  if (!controller._noveltyArchive) controller._noveltyArchive = [];

  // Step 2: Respect the archive threshold before appending.
  const archiveAddThreshold = noveltyOptions.archiveAddThreshold ?? Infinity;
  const shouldAdd =
    noveltyOptions.archiveAddThreshold === 0 || novelty > archiveAddThreshold;
  if (!shouldAdd) return;

  // Step 3: Append only while the bounded archive still has room.
  if (controller._noveltyArchive.length < NOVELTY_ARCHIVE_CAP) {
    controller._noveltyArchive.push({ desc: descriptor, novelty });
  }
}

/**
 * Compute the Euclidean distance between two descriptors.
 *
 * @param leftDescriptor - Left descriptor vector.
 * @param rightDescriptor - Right descriptor vector.
 * @param isSame - Whether both descriptors belong to the same genome index.
 * @returns Euclidean distance across the shared prefix.
 */
function computeDescriptorDistance(
  leftDescriptor: number[],
  rightDescriptor: number[],
  isSame: boolean,
): number {
  // Step 1: Keep self-distance fixed at zero.
  if (isSame) return 0;

  // Step 2: Compute Euclidean distance across the shared descriptor prefix.
  const commonLength = Math.min(leftDescriptor.length, rightDescriptor.length);
  const squaredSum = leftDescriptor
    .slice(0, commonLength)
    .reduce((accumulated, leftValue, index) => {
      const delta = leftValue - (rightDescriptor[index] ?? 0);
      return accumulated + delta * delta;
    }, 0);

  return Math.sqrt(squaredSum);
}
