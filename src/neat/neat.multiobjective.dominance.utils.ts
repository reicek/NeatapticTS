import type { ObjectiveDescriptor } from './neat.multiobjective.utils.types';

/**
 * Determines whether vector A Pareto-dominates vector B.
 *
 * A dominates B iff:
 * - A is **no worse** than B in every objective (respecting each objective’s
 *   direction: maximize/minimize), and
 * - A is **strictly better** in at least one objective.
 *
 * Assumptions:
 * - `valuesA` and `valuesB` are aligned and have the same length.
 * - `descriptors` provides a descriptor for each objective index.
 * - If a descriptor has no `direction`, it defaults to `'max'`.
 *
 * @param valuesA - Objective values for candidate A.
 * @param valuesB - Objective values for candidate B.
 * @param descriptors - Objective descriptors defining direction semantics.
 * @returns `true` if A dominates B; otherwise `false`.
 *
 * @example
 * ```ts
 * // Maximize accuracy, minimize latency:
 * vectorDominates([0.9, 120], [0.9, 150], [
 *   { accessor: () => 0, direction: 'max' },
 *   { accessor: () => 0, direction: 'min' },
 * ]);
 * // => true (equal accuracy, lower latency)
 * ```
 */
export function vectorDominates(
  valuesA: number[],
  valuesB: number[],
  descriptors: ObjectiveDescriptor[],
): boolean {
  const objectiveCount = valuesA.length;
  let hasStrictImprovement = false;

  // Step 1: compare all objectives with their directional semantics.
  for (
    let objectiveIndex = 0;
    objectiveIndex < objectiveCount;
    objectiveIndex++
  ) {
    const direction = resolveObjectiveDirection(descriptors, objectiveIndex);
    const candidateValue = valuesA[objectiveIndex];
    const opponentValue = valuesB[objectiveIndex];
    const comparison = compareObjectiveValues(
      direction,
      candidateValue,
      opponentValue,
    );

    if (comparison.isDominated) return false;
    hasStrictImprovement = updateStrictImprovement(
      hasStrictImprovement,
      comparison.isStrictlyBetter,
    );
  }

  return hasStrictImprovement;
}

/**
 * Resolves the objective direction for a given objective index.
 *
 * If a descriptor omits `direction`, it is treated as maximization.
 *
 * @param descriptors - Objective descriptors.
 * @param objectiveIndex - Objective index.
 * @returns Normalized objective direction.
 */
function resolveObjectiveDirection(
  descriptors: ObjectiveDescriptor[],
  objectiveIndex: number,
): 'max' | 'min' {
  // Step 1: resolve to default direction when unspecified.
  return descriptors[objectiveIndex].direction ?? 'max';
}

/**
 * Compares a candidate and opponent value for a single objective.
 *
 * This does not compute full Pareto dominance; it returns per-objective flags
 * used by the vector-level dominance check.
 *
 * @param direction - Objective direction.
 * @param candidateValue - Candidate objective value.
 * @param opponentValue - Opponent objective value.
 * @returns Comparison flags for this objective.
 */
function compareObjectiveValues(
  direction: 'max' | 'min',
  candidateValue: number,
  opponentValue: number,
): { isDominated: boolean; isStrictlyBetter: boolean } {
  // Step 1: evaluate whether candidate is dominated for this objective.
  const isDominated = isCandidateDominatedByObjective(
    direction,
    candidateValue,
    opponentValue,
  );
  if (isDominated) {
    return { isDominated: true, isStrictlyBetter: false };
  }

  // Step 2: check if candidate strictly improves for this objective.
  const isStrictlyBetter = isCandidateStrictlyBetterForObjective(
    direction,
    candidateValue,
    opponentValue,
  );
  return { isDominated: false, isStrictlyBetter };
}

/**
 * Checks if the candidate is worse than the opponent for a single objective.
 *
 * For dominance, being worse on any objective makes the candidate unable to
 * dominate the opponent.
 *
 * @param direction - Objective direction.
 * @param candidateValue - Candidate objective value.
 * @param opponentValue - Opponent objective value.
 * @returns `true` if the candidate is dominated for this objective.
 */
function isCandidateDominatedByObjective(
  direction: 'max' | 'min',
  candidateValue: number,
  opponentValue: number,
): boolean {
  // Step 1: apply direction-specific dominance check.
  if (direction === 'max') return candidateValue < opponentValue;
  return candidateValue > opponentValue;
}

/**
 * Checks if the candidate is strictly better than the opponent for a single
 * objective.
 *
 * Strict improvement in at least one objective is required for Pareto
 * dominance when the candidate is not worse in any objective.
 *
 * @param direction - Objective direction.
 * @param candidateValue - Candidate objective value.
 * @param opponentValue - Opponent objective value.
 * @returns `true` if the candidate is strictly better for this objective.
 */
function isCandidateStrictlyBetterForObjective(
  direction: 'max' | 'min',
  candidateValue: number,
  opponentValue: number,
): boolean {
  // Step 1: apply direction-specific strict comparison.
  if (direction === 'max') return candidateValue > opponentValue;
  return candidateValue < opponentValue;
}

/**
 * Accumulates whether the candidate has any strict improvement across
 * objectives.
 *
 * @param hasStrictImprovement - Current strict-improvement flag.
 * @param isStrictlyBetter - Whether the candidate strictly improves on the
 * current objective.
 * @returns Updated strict-improvement flag.
 */
function updateStrictImprovement(
  hasStrictImprovement: boolean,
  isStrictlyBetter: boolean,
): boolean {
  // Step 1: accumulate strict improvement flag.
  if (hasStrictImprovement) return true;
  return isStrictlyBetter;
}

/**
 * Dominance bookkeeping structures for fast non-dominated sorting.
 *
 * These structures are typically produced once per generation (from the values
 * matrix) and then consumed to build Pareto fronts.
 */
export type DominanceState = {
  /** Number of genomes that dominate genome i. */
  dominationCounts: number[];

  /** For each genome i, the indices of genomes that i dominates. */
  dominatedIndicesByIndex: number[][];

  /** Indices of the first front (domination count == 0). */
  firstFrontIndices: number[];
};

/**
 * Builds dominance bookkeeping structures used by fast non-dominated sorting.
 *
 * This computes (pairwise):
 * - `dominationCounts[i]`: how many genomes dominate genome `i`.
 * - `dominatedIndicesByIndex[i]`: which genomes are dominated by genome `i`.
 * - `firstFrontIndices`: genomes with `dominationCounts[i] === 0`.
 *
 * Complexity:
 * - Time: $O(n^2 \cdot m)$ where $n$ is population size and $m$ is objective
 *   count.
 * - Space: $O(n^2)$ in the worst case for the dominated adjacency lists.
 *
 * Assumptions:
 * - Each row in `valuesMatrixInput` is a vector aligned with `descriptors`.
 * - Genome ordering in later steps is expected to match the matrix ordering.
 *
 * @param valuesMatrixInput - Matrix of objective values (row = genome).
 * @param descriptors - Objective descriptors (direction semantics).
 * @returns Dominance bookkeeping structures for ranking.
 */
export function buildDominanceState(
  valuesMatrixInput: number[][],
  descriptors: ObjectiveDescriptor[],
): DominanceState {
  const populationSize = valuesMatrixInput.length;
  const dominanceState = createEmptyDominanceState(populationSize);
  const candidateIndices = buildIndexRange(populationSize);

  // Step 1: compare every pair once for dominance relationships.
  for (const candidateIndex of candidateIndices) {
    // Step 1.1: apply pairwise dominance updates for the candidate.
    updateDominanceForCandidate(
      dominanceState,
      valuesMatrixInput,
      descriptors,
      candidateIndex,
      candidateIndices,
    );
    // Step 1.2: collect candidates with zero domination count.
    if (isNonDominatedCandidate(dominanceState, candidateIndex)) {
      dominanceState.firstFrontIndices.push(candidateIndex);
    }
  }

  return dominanceState;
}

/**
 * Creates an empty dominance state container sized to the population.
 *
 * @param populationSize - Number of genomes.
 * @returns An initialized dominance state with zeroed counts.
 */
function createEmptyDominanceState(populationSize: number): DominanceState {
  // Step 1: initialize dominance bookkeeping collections.
  return {
    dominationCounts: new Array(populationSize).fill(0),
    dominatedIndicesByIndex: Array.from({ length: populationSize }, () => []),
    firstFrontIndices: [],
  };
}

/**
 * Builds a stable index range for iterating the population.
 *
 * @param populationSize - Number of genomes.
 * @returns Array of indices `0..populationSize-1`.
 */
function buildIndexRange(populationSize: number): number[] {
  // Step 1: create a stable index range.
  return Array.from({ length: populationSize }, (_, index) => index);
}

/**
 * Updates dominance bookkeeping for a candidate against all opponents.
 *
 * This iterates every opponent index and applies a pairwise dominance update.
 * Self-comparisons are ignored.
 *
 * @param dominanceState - Dominance bookkeeping.
 * @param valuesMatrixInput - Matrix of objective values.
 * @param descriptors - Objective descriptors.
 * @param candidateIndex - Candidate genome index.
 * @param candidateIndices - Indices to compare against.
 */
function updateDominanceForCandidate(
  dominanceState: DominanceState,
  valuesMatrixInput: number[][],
  descriptors: ObjectiveDescriptor[],
  candidateIndex: number,
  candidateIndices: number[],
): void {
  // Step 1: compare candidate against all opponents.
  for (const opponentIndex of candidateIndices) {
    // Step 1.1: apply one pairwise comparison.
    applyPairwiseDominance(
      dominanceState,
      valuesMatrixInput,
      descriptors,
      candidateIndex,
      opponentIndex,
    );
  }
}

/**
 * Applies a single pairwise dominance update between candidate and opponent.
 *
 * If the candidate dominates the opponent, the opponent index is appended to
 * `dominatedIndicesByIndex[candidateIndex]`. If the candidate is dominated by
 * the opponent, `dominationCounts[candidateIndex]` is incremented.
 *
 * @param dominanceState - Dominance bookkeeping.
 * @param valuesMatrixInput - Matrix of objective values.
 * @param descriptors - Objective descriptors.
 * @param candidateIndex - Candidate genome index.
 * @param opponentIndex - Opponent genome index.
 */
function applyPairwiseDominance(
  dominanceState: DominanceState,
  valuesMatrixInput: number[][],
  descriptors: ObjectiveDescriptor[],
  candidateIndex: number,
  opponentIndex: number,
): void {
  if (shouldSkipSelfComparison(candidateIndex, opponentIndex)) return;

  // Step 1: resolve objective vectors for this pair.
  const candidateVector = valuesMatrixInput[candidateIndex];
  const opponentVector = valuesMatrixInput[opponentIndex];

  // Step 2: determine dominance outcome for the pair.
  const outcome = resolveDominanceOutcome(
    candidateVector,
    opponentVector,
    descriptors,
  );

  // Step 3: update bookkeeping based on outcome.
  if (outcome === 'dominates') {
    dominanceState.dominatedIndicesByIndex[candidateIndex].push(opponentIndex);
    return;
  }
  if (outcome === 'dominated') {
    dominanceState.dominationCounts[candidateIndex]++;
  }
}

/**
 * Determines whether a pairwise comparison should be skipped.
 *
 * Currently this skips only self-comparisons.
 *
 * @param candidateIndex - Candidate genome index.
 * @param opponentIndex - Opponent genome index.
 * @returns `true` if the pair should be skipped.
 */
function shouldSkipSelfComparison(
  candidateIndex: number,
  opponentIndex: number,
): boolean {
  // Step 1: skip self comparison.
  return candidateIndex === opponentIndex;
}

/**
 * Resolves dominance outcome between two objective vectors.
 *
 * Outcome meanings:
 * - `'dominates'`: candidate dominates opponent.
 * - `'dominated'`: candidate is dominated by opponent.
 * - `'indifferent'`: neither dominates the other.
 *
 * @param candidateVector - Candidate objective values.
 * @param opponentVector - Opponent objective values.
 * @param descriptors - Objective descriptors.
 * @returns Dominance outcome between candidate and opponent.
 */
function resolveDominanceOutcome(
  candidateVector: number[],
  opponentVector: number[],
  descriptors: ObjectiveDescriptor[],
): 'dominates' | 'dominated' | 'indifferent' {
  // Step 1: check if candidate dominates opponent.
  if (vectorDominates(candidateVector, opponentVector, descriptors)) {
    return 'dominates';
  }
  // Step 2: check if opponent dominates candidate.
  if (vectorDominates(opponentVector, candidateVector, descriptors)) {
    return 'dominated';
  }
  // Step 3: neither dominates the other.
  return 'indifferent';
}

/**
 * Determines whether a candidate has zero domination count.
 *
 * @param dominanceState - Dominance bookkeeping.
 * @param candidateIndex - Candidate genome index.
 * @returns `true` if the candidate is currently non-dominated.
 */
function isNonDominatedCandidate(
  dominanceState: DominanceState,
  candidateIndex: number,
): boolean {
  // Step 1: check candidate domination count.
  return dominanceState.dominationCounts[candidateIndex] === 0;
}
