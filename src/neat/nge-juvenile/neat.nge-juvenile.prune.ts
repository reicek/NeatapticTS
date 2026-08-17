import { NGE_JUVENILE_DEFAULT_MIN_EDGE_FLOOR } from './neat.nge-juvenile.constants';
import {
  NgeJuvenile_BudgetError,
  NgeJuvenile_MorphError,
} from './neat.nge-juvenile.errors';
import type {
  NgeHysteresisState,
  NgeJuvenilePhaseConfig,
  NgeMorphDelta,
  NgePruneBudget,
  NgePruneCandidate,
} from './neat.nge-juvenile.types';

type NgePruneMorphKind = 'edgePrune' | 'compact';

/**
 * Check whether juvenile prune or compact actions may commit in the current window.
 *
 * @param hysteresis - Current prune-side hysteresis state tracked across windows.
 * @param config - Resolved juvenile-phase configuration.
 * @returns `true` when the underuse streak is satisfied and cooldown is clear.
 */
export function canPruneNow(
  hysteresis: NgeHysteresisState,
  config: NgeJuvenilePhaseConfig,
): boolean {
  return (
    hysteresis.pruneUnderuseWindowCount >= config.hysteresisWindowCount &&
    hysteresis.cooldownWindowsRemaining === 0
  );
}

/**
 * Advance the prune-side hysteresis counters for one evaluation window and return fresh state.
 *
 * @param hysteresis - Previous hysteresis state.
 * @param isUnderuseWindow - Whether the current window carried prune evidence.
 * @returns A fresh hysteresis state with the prune streak and cooldown advanced.
 */
export function advancePruneHysteresis(
  hysteresis: NgeHysteresisState,
  isUnderuseWindow: boolean,
): NgeHysteresisState {
  return {
    ...hysteresis,
    pruneUnderuseWindowCount: isUnderuseWindow
      ? hysteresis.pruneUnderuseWindowCount + 1
      : 0,
    cooldownWindowsRemaining: Math.max(
      hysteresis.cooldownWindowsRemaining - 1,
      0,
    ),
  };
}

/**
 * Commit one prune-side hysteresis update after a validated morph is applied.
 *
 * @param hysteresis - Previous hysteresis state.
 * @param morphKind - Concrete prune or compact morph kind that committed.
 * @param config - Resolved juvenile-phase configuration.
 * @returns A fresh hysteresis state ready for the next cooldown window.
 */
export function commitPrune(
  hysteresis: NgeHysteresisState,
  morphKind: NgePruneMorphKind,
  config: NgeJuvenilePhaseConfig,
): NgeHysteresisState {
  return {
    ...hysteresis,
    lastMorphKind: morphKind,
    cooldownWindowsRemaining: config.cooldownWindowCount,
    pruneUnderuseWindowCount: 0,
  };
}

/**
 * Select the highest-priority non-exempt prune candidate for one module, sorted by wiring cost.
 *
 * @param candidates - Caller-supplied candidate edges for one prune pass.
 * @param budget - DNA floors and permanent prune exemptions for the module.
 * @returns The highest-priority non-exempt candidate, ordered by wiring cost then edge length.
 * @throws {NgeJuvenile_MorphError} When no non-exempt prune candidates remain after filtering.
 */
export function selectPruneCandidate(
  candidates: readonly NgePruneCandidate[],
  budget: NgePruneBudget,
): NgePruneCandidate {
  const eligibleCandidates = candidates
    .filter(
      ({ candidateId }) => !budget.costExemptEdgeIds.includes(candidateId),
    )
    .toSorted((leftCandidate, rightCandidate) => {
      const wiringCostDelta =
        rightCandidate.wiringCost - leftCandidate.wiringCost;

      if (wiringCostDelta !== 0) {
        return wiringCostDelta;
      }

      const edgeLengthDelta =
        rightCandidate.edgeLength - leftCandidate.edgeLength;

      if (edgeLengthDelta !== 0) {
        return edgeLengthDelta;
      }

      return leftCandidate.candidateId.localeCompare(
        rightCandidate.candidateId,
      );
    });

  if (eligibleCandidates.length === 0) {
    throw new NgeJuvenile_MorphError(
      'No non-exempt prune candidates remain after cost-exempt filtering.',
    );
  }

  return eligibleCandidates[0];
}

/**
 * Plan one dry-run edge-prune delta for a single module, respecting cost-exempt edges and DNA floor.
 *
 * @param moduleId - Module receiving the planned edge prune.
 * @param candidate - Candidate edge selected for dry-run pruning.
 * @param budget - DNA floors and prune exemptions for the module.
 * @returns One dry-run edge-prune delta.
 * @throws {NgeJuvenile_MorphError} When the candidate edge is permanently cost-exempt.
 * @throws {NgeJuvenile_BudgetError} When pruning would violate the effective edge floor.
 */
export function planEdgePrune(
  moduleId: string,
  candidate: NgePruneCandidate,
  budget: NgePruneBudget,
): NgeMorphDelta {
  if (budget.costExemptEdgeIds.includes(candidate.candidateId)) {
    throw new NgeJuvenile_MorphError(
      `Edge ${candidate.candidateId} is permanently exempt from pruning.`,
    );
  }

  const effectiveMinEdges = Math.max(
    budget.minEdges,
    NGE_JUVENILE_DEFAULT_MIN_EDGE_FLOOR,
  );

  if (budget.currentEdgeCount <= effectiveMinEdges) {
    throw new NgeJuvenile_BudgetError(
      `Edge pruning would violate the edge floor for ${moduleId}.`,
    );
  }

  return {
    kind: 'edgePrune',
    targetModuleId: moduleId,
    detail: {
      candidateId: candidate.candidateId,
      wiringCost: candidate.wiringCost,
      edgeLength: candidate.edgeLength,
      currentEdgeCount: budget.currentEdgeCount,
    },
    wiringCostDelta: -candidate.wiringCost,
  };
}

/**
 * Plan one dry-run compact delta for a single module, verifying the node floor before returning.
 *
 * @param moduleId - Module receiving the planned compact action.
 * @param budget - DNA floors and current structural counts for the module.
 * @returns One dry-run compact delta.
 * @throws {NgeJuvenile_BudgetError} When compaction would violate the configured node floor.
 */
export function planCompact(
  moduleId: string,
  budget: NgePruneBudget,
): NgeMorphDelta {
  if (budget.currentNodeCount <= budget.minNodes) {
    throw new NgeJuvenile_BudgetError(
      `Compaction would violate the node floor for ${moduleId}.`,
    );
  }

  return {
    kind: 'compact',
    targetModuleId: moduleId,
    detail: {
      currentNodeCount: budget.currentNodeCount,
      currentWiringCost: budget.currentWiringCost,
    },
    wiringCostDelta: 0,
  };
}

/**
 * Re-validate one dry-run prune delta against the current structural floors.
 *
 * @param delta - Planned morph delta to validate.
 * @param budget - DNA floors and current structural counts for the module.
 * @throws {NgeJuvenile_BudgetError} When the planned delta would drop below a structural floor.
 */
export function validatePruneDelta(
  delta: NgeMorphDelta,
  budget: NgePruneBudget,
): void {
  switch (delta.kind) {
    case 'edgePrune': {
      const effectiveMinEdges = Math.max(
        budget.minEdges,
        NGE_JUVENILE_DEFAULT_MIN_EDGE_FLOOR,
      );

      if (budget.currentEdgeCount + delta.wiringCostDelta < effectiveMinEdges) {
        throw new NgeJuvenile_BudgetError(
          `Edge pruning would drop below the edge floor for ${delta.targetModuleId}.`,
        );
      }

      return;
    }
    case 'compact':
      if (budget.currentNodeCount <= budget.minNodes) {
        throw new NgeJuvenile_BudgetError(
          `Compaction would drop below the node floor for ${delta.targetModuleId}.`,
        );
      }

      return;
    case 'edgeDensify':
    case 'slotExpand':
    case 'nodeAdd':
      return;
  }

  /* istanbul ignore next -- compile-time exhaustiveness guard for future morph kinds */
  const exhaustiveMorphKind: never = delta.kind;
  /* istanbul ignore next -- compile-time exhaustiveness guard for future morph kinds */
  throw new NgeJuvenile_MorphError(
    `Unhandled prune delta kind: ${String(exhaustiveMorphKind)}`,
  );
}

/**
 * Plan all eligible dry-run prune deltas for one module in prune-before-compact order.
 *
 * @param moduleId - Module receiving all planned prune-side actions.
 * @param budget - DNA floors, exemptions, and current structural counts.
 * @param candidates - Caller-supplied edge candidates ranked locally within the module.
 * @param config - Resolved juvenile-phase configuration.
 * @param hysteresis - Current prune-side hysteresis state.
 * @returns Zero or more validated dry-run prune deltas in priority order.
 * @throws {Error} When an unexpected non-budget, non-morph error is raised by a sub-step.
 */
export function planPruneMorphs(
  moduleId: string,
  budget: NgePruneBudget,
  candidates: readonly NgePruneCandidate[],
  config: NgeJuvenilePhaseConfig,
  hysteresis: NgeHysteresisState,
): NgeMorphDelta[] {
  // Step 1: Exit immediately until the prune hysteresis gate opens.
  if (!canPruneNow(hysteresis, config)) {
    return [];
  }

  const plannedDeltas: NgeMorphDelta[] = [];

  // Step 2: Try the highest-priority edge prune first.
  try {
    const pruneCandidate = selectPruneCandidate(candidates, budget);
    const edgePruneDelta = planEdgePrune(moduleId, pruneCandidate, budget);
    validatePruneDelta(edgePruneDelta, budget);
    plannedDeltas.push(edgePruneDelta);
  } catch (error) {
    if (
      !(error instanceof NgeJuvenile_BudgetError) &&
      !(error instanceof NgeJuvenile_MorphError)
    ) {
      throw error;
    }
  }

  // Step 3: Try the compact action second.
  try {
    const compactDelta = planCompact(moduleId, budget);
    validatePruneDelta(compactDelta, budget);
    plannedDeltas.push(compactDelta);
  } catch (error) {
    if (!(error instanceof NgeJuvenile_BudgetError)) {
      throw error;
    }
  }

  // Step 4: Return the dry-run delta plan without mutating any inputs.
  return plannedDeltas;
}
