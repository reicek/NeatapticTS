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
 *
 * The boundary is intentionally evidence-focused. By the time these helpers
 * run, the fitness stage has already produced fresh scores or at least
 * prepared the population for scoring. Novelty does not replace evaluation with
 * an entirely separate optimization loop. Instead it adds one more kind of
 * evidence: how behaviorally unusual each genome looks relative to its nearby
 * peers.
 *
 * Read this chapter when you want to answer questions such as:
 * - Why does novelty stay a separate evaluation stage instead of widening into
 *   objective management or multi-objective ranking?
 * - How do descriptor building, distance-matrix construction, and neighbor
 *   scoring fit together?
 * - What does blend factor actually mean when a genome already has a fitness
 *   score?
 * - Why is archive admission threshold-based instead of recording every
 *   descriptor forever?
 *
 * The mental model is a five-step evidence loop:
 * 1. build one behavior descriptor per genome,
 * 2. compute pairwise distances across that descriptor set,
 * 3. average the nearest-neighbor distances into a novelty score,
 * 4. blend novelty into the current score when a numeric fitness score exists,
 * 5. append only sufficiently novel descriptors to the bounded archive.
 *
 * The preserved assumptions matter as much as the additional evidence. This
 * boundary does not reorder the population, rebuild species, or register new
 * objectives. It annotates each genome with novelty evidence and optionally
 * blends that evidence into the current score so later tuning, selection, and
 * speciation reads can still reason from one stable evaluated population.
 *
 * ```mermaid
 * flowchart TD
 *   Population[Freshly scored population] --> Descriptors[Build one behavior descriptor per genome]
 *   Descriptors --> Distances[Compute dense descriptor distance matrix]
 *   Distances --> Novelty[Average nearest-neighbor distances into novelty scores]
 *   Novelty --> Blend[Blend novelty into existing fitness scores]
 *   Blend --> Archive[Append only descriptors above archive threshold]
 *   Archive --> Next[Later tuning and selection read enriched scores]
 * ```
 */

/**
 * Compute novelty, blend it into scores, and update the novelty archive.
 *
 * This is the controller-facing entrypoint for the novelty stage. It behaves
 * like best-effort evidence enrichment after the core fitness path has run.
 * If novelty is disabled or misconfigured, evaluation can safely continue with
 * the base score evidence alone.
 *
 * The helper preserves several important controller assumptions:
 * - the current population order is left intact,
 * - current species membership is left intact,
 * - no objective-registration policy is changed here,
 * - only novelty evidence, optional score blending, and archive state are
 *   updated.
 *
 * @param controller - NEAT controller instance for evaluation.
 * @param evaluationOptions - Options object for the current evaluation pass.
 *
 * @example
 * ```ts
 * runNoveltyBlendAndArchive(controller, controller.options);
 * console.log(controller.population[0]?._novelty);
 * ```
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
 * Smaller neighbor counts make novelty more sensitive to local behavioral
 * differences, while larger counts smooth that signal across a wider portion of
 * the current population.
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
 * A factor of `0` keeps the existing fitness score unchanged. A factor of `1`
 * makes novelty fully replace it when a numeric score exists. Values in between
 * turn novelty into a partial exploratory bonus instead of a hard override.
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
 * Descriptor building is isolated here so the rest of the chapter can treat
 * novelty as a pure data pipeline: descriptor first, distance second, scoring
 * third. Descriptor failures degrade to an empty vector instead of failing the
 * entire evaluation pass.
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
 * The matrix is dense and population-order aligned so later helpers can keep
 * the scoring flow simple: each genome reads one row, drops its self-distance,
 * and averages the nearest neighbors.
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
 * This helper is the fold stage of the novelty pipeline. It translates the
 * descriptor-distance evidence into per-genome novelty annotations, then
 * optionally blends that evidence into numeric scores and records sufficiently
 * novel descriptors for future exploration pressure.
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
 * Novelty is defined here as the average distance to the nearest neighbors
 * after excluding the genome's self-distance. Higher values mean the genome is
 * behaving in a less crowded region of descriptor space.
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
 * This stage intentionally does not invent a base score when one is missing.
 * Novelty acts as a companion signal to the existing evaluation path, not as a
 * universal replacement for every scoring mode.
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
 * The archive is intentionally bounded and selective. Threshold-based admission
 * keeps it focused on descriptors that are genuinely unusual enough to be worth
 * remembering, while the cap prevents novelty exploration from turning into an
 * unbounded memory sink.
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
 * Distance is computed across the shared descriptor prefix so callers can keep
 * using practical descriptor functions even when they occasionally emit vectors
 * of uneven length. Self-distance is fixed at zero to keep later neighbor
 * ranking deterministic.
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
