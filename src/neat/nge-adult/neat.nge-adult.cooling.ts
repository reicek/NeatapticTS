import {
  NGE_ADULT_DEFAULT_GROWTH_COOLING_FACTOR,
  NGE_ADULT_DEFAULT_GROWTH_FOCUS_FLOOR,
} from './neat.nge-adult.constants';

/**
 * Budget split emitted by the adult cooling policy for one focus score.
 */
export interface AdultGrowthCoolingDecision {
  /** Whether adult growth remains in the cooled regime. */
  growthCoolingActive: boolean;
  /** Fraction of the current morph budget still available for growth. */
  growthBudgetFraction: number;
  /** Fraction of the current morph budget reserved for prune and compact. */
  pruneCompactBudgetFraction: number;
}

/**
 * Ranked adult prune candidate evaluated before compact morphs are ever considered.
 */
export interface AdultPruneCandidate {
  /** Stable candidate identifier used by later adult morph planners. */
  candidateId: string;
  /** Relative edge-length score used by adult prune preference rules. */
  edgeLength: number;
  /** Whether the edge crosses adult module boundaries. */
  isInterModule: boolean;
  /** Whether the edge is protected because it is still reward-critical. */
  isRewardCritical: boolean;
}

/**
 * Local dominance decision produced by the adult prune and compact arbitration step.
 */
export interface AdultPruneCompactDecision {
  /** Ordered adult morph kinds that currently dominate the cooled budget. */
  dominantMorphKinds: ReadonlyArray<'edgePrune' | 'compact'>;
  /** Selected non-critical prune candidate, if one survives arbitration. */
  selectedPruneCandidateId: string | null;
}

/**
 * Resolve the current adult cooling decision for one normalized focus score.
 *
 * Growth cooling is active only when the focus score is above the floor and the
 * residual growth budget is non-zero. A zero cooling factor or a sub-floor focus score
 * fully suppresses adult growth and leaves the entire morph budget for prune and
 * compact actions.
 *
 * @param focusScore - Normalized adult focus score for the active module.
 * @param growthCoolingFactor - Residual growth fraction preserved by adult cooling.
 * @param focusFloor - Minimum focus score that keeps residual growth alive.
 * @returns Cooling decision carrying the active flag and budget split for the cycle.
 *
 * @example
 * ```ts
 * const decision = resolveGrowthCoolingDecision(0.82, 0.1, 0.6);
 * console.log(decision.growthCoolingActive); // true
 * console.log(decision.growthBudgetFraction); // 0.1
 * ```
 */
export function resolveGrowthCoolingDecision(
  focusScore: number,
  growthCoolingFactor = NGE_ADULT_DEFAULT_GROWTH_COOLING_FACTOR,
  focusFloor = NGE_ADULT_DEFAULT_GROWTH_FOCUS_FLOOR,
): AdultGrowthCoolingDecision {
  const growthBudgetFraction =
    focusScore >= focusFloor ? growthCoolingFactor : 0;

  return {
    growthCoolingActive: growthBudgetFraction > 0,
    growthBudgetFraction,
    pruneCompactBudgetFraction: 1 - growthBudgetFraction,
  };
}

/**
 * Arbitrate whether adult prune and compact dominate the cooled morph budget.
 *
 * Selects the highest-priority non-reward-critical prune candidate, preferring
 * inter-module edges and longer edges, then resolves which morph kinds are eligible
 * for the cooled budget based on candidate and compact availability.
 *
 * @param candidates - Ranked prune candidates visible to the local adult module.
 * @param compactEligible - Whether compact still has headroom in the current window.
 * @returns Prune-compact decision naming the dominant morph kinds and selected candidate.
 *
 * @example
 * ```ts
 * const decision = arbitratePruneCompactDominance(
 *   [{ candidateId: 'e1', edgeLength: 2, isInterModule: true, isRewardCritical: false }],
 *   true,
 * );
 * console.log(decision.dominantMorphKinds); // ['edgePrune', 'compact']
 * ```
 */
export function arbitratePruneCompactDominance(
  candidates: readonly AdultPruneCandidate[],
  compactEligible: boolean,
): AdultPruneCompactDecision {
  const selectedPruneCandidate = candidates
    .filter((candidate) => !candidate.isRewardCritical)
    .toSorted((leftCandidate, rightCandidate) => {
      const interModuleDelta =
        Number(rightCandidate.isInterModule) -
        Number(leftCandidate.isInterModule);

      if (interModuleDelta !== 0) {
        return interModuleDelta;
      }

      const edgeLengthDelta =
        rightCandidate.edgeLength - leftCandidate.edgeLength;

      if (edgeLengthDelta !== 0) {
        return edgeLengthDelta;
      }

      return leftCandidate.candidateId.localeCompare(
        rightCandidate.candidateId,
      );
    })
    .at(0);

  return {
    dominantMorphKinds: resolveDominantMorphKinds(
      selectedPruneCandidate !== undefined,
      compactEligible,
    ),
    selectedPruneCandidateId: selectedPruneCandidate?.candidateId ?? null,
  };
}

function resolveDominantMorphKinds(
  pruneEligible: boolean,
  compactEligible: boolean,
): ReadonlyArray<'edgePrune' | 'compact'> {
  if (pruneEligible && compactEligible) {
    return ['edgePrune', 'compact'];
  }

  if (pruneEligible) {
    return ['edgePrune'];
  }

  if (compactEligible) {
    return ['compact'];
  }

  return [];
}
