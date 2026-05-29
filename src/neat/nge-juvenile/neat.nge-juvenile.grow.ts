import {
  NGE_JUVENILE_DEFAULT_EDGE_DENSIFICATION_COUNT,
  NGE_JUVENILE_DEFAULT_NODE_REWARD_DELTA_FLOOR,
  NGE_JUVENILE_DEFAULT_SLOT_EXPANSION_COUNT,
} from './neat.nge-juvenile.constants';
import {
  NgeJuvenile_BudgetError,
  NgeJuvenile_MorphError,
} from './neat.nge-juvenile.errors';
import type {
  NgeFocusScore,
  NgeGrowthBudget,
  NgeHysteresisState,
  NgeJuvenilePhaseConfig,
  NgeModuleMetricsSnapshot,
  NgeMorphDelta,
} from './neat.nge-juvenile.types';

type NgeGrowthMorphKind = 'edgeDensify' | 'slotExpand' | 'nodeAdd';

/**
 * Check whether juvenile growth may commit in the current window.
 *
 * @param hysteresis - Current hysteresis state tracked across windows.
 * @param config - Resolved juvenile-phase configuration.
 * @returns `true` when the positive-focus streak is satisfied and cooldown is clear.
 */
export function canGrowNow(
  hysteresis: NgeHysteresisState,
  config: NgeJuvenilePhaseConfig,
): boolean {
  return (
    hysteresis.growthPositiveWindowCount >= config.hysteresisWindowCount &&
    hysteresis.cooldownWindowsRemaining === 0
  );
}

/**
 * Advance the growth-side hysteresis counters for one evaluation window and return fresh state.
 *
 * @param hysteresis - Previous hysteresis state.
 * @param isPositiveFocusWindow - Whether the current window carried positive focus evidence.
 * @returns A fresh hysteresis state with the growth streak and cooldown advanced.
 */
export function advanceGrowthHysteresis(
  hysteresis: NgeHysteresisState,
  isPositiveFocusWindow: boolean,
): NgeHysteresisState {
  return {
    ...hysteresis,
    growthPositiveWindowCount: isPositiveFocusWindow
      ? hysteresis.growthPositiveWindowCount + 1
      : 0,
    cooldownWindowsRemaining: Math.max(
      hysteresis.cooldownWindowsRemaining - 1,
      0,
    ),
  };
}

/**
 * Commit one growth-side hysteresis update after a validated morph is applied.
 *
 * @param hysteresis - Previous hysteresis state.
 * @param morphKind - Concrete growth morph kind that committed.
 * @param config - Resolved juvenile-phase configuration.
 * @returns A fresh hysteresis state ready for the next cooldown window.
 */
export function commitGrowth(
  hysteresis: NgeHysteresisState,
  morphKind: NgeGrowthMorphKind,
  config: NgeJuvenilePhaseConfig,
): NgeHysteresisState {
  return {
    ...hysteresis,
    lastMorphKind: morphKind,
    cooldownWindowsRemaining: config.cooldownWindowCount,
    growthPositiveWindowCount: 0,
  };
}

/**
 * Plan one local edge-densification delta for a single module, validating the DNA edge budget.
 *
 * @param moduleId - Module receiving the planned densification.
 * @param budget - DNA-configured growth caps and current live counts.
 * @param focusScore - Focus score for the target module.
 * @returns One dry-run edge densification delta.
 */
export function planEdgeDensification(
  moduleId: string,
  budget: NgeGrowthBudget,
  focusScore: NgeFocusScore,
): NgeMorphDelta {
  if (
    budget.currentEdgeCount + NGE_JUVENILE_DEFAULT_EDGE_DENSIFICATION_COUNT >
    budget.maxEdges
  ) {
    throw new NgeJuvenile_BudgetError(
      `Edge densification would exceed the edge budget for ${moduleId}.`,
    );
  }

  return {
    kind: 'edgeDensify',
    targetModuleId: moduleId,
    detail: {
      currentEdgeCount: budget.currentEdgeCount,
      proposedAdditions: NGE_JUVENILE_DEFAULT_EDGE_DENSIFICATION_COUNT,
      normalizedFocusScore: focusScore.normalizedScore,
    },
    wiringCostDelta: NGE_JUVENILE_DEFAULT_EDGE_DENSIFICATION_COUNT,
  };
}

/**
 * Plan one local episodic-slot expansion delta for a single module.
 *
 * Phase B uses `metrics.utilization` as the hit-rate proxy until a dedicated episodic
 * hit-rate field lands in a later step.
 *
 * @param moduleId - Module receiving the planned slot expansion.
 * @param hitRate - Episodic hit-rate proxy for the target module.
 * @param budget - DNA-configured growth caps and current live counts.
 * @param focusScore - Focus score for the target module.
 * @param config - Resolved juvenile-phase configuration.
 * @returns One dry-run slot expansion delta.
 */
export function planSlotExpansion(
  moduleId: string,
  hitRate: number,
  budget: NgeGrowthBudget,
  focusScore: NgeFocusScore,
  config: NgeJuvenilePhaseConfig,
): NgeMorphDelta {
  if (hitRate < config.episodicHitRateThreshold) {
    throw new NgeJuvenile_MorphError(
      `Slot expansion requires hit rate >= ${config.episodicHitRateThreshold} for ${moduleId}.`,
    );
  }

  if (focusScore.normalizedScore <= 0) {
    throw new NgeJuvenile_MorphError(
      `Slot expansion requires positive focus for ${moduleId}.`,
    );
  }

  if (
    budget.currentEpisodicSlotCount +
      NGE_JUVENILE_DEFAULT_SLOT_EXPANSION_COUNT >
    budget.maxEpisodicSlots
  ) {
    throw new NgeJuvenile_BudgetError(
      `Slot expansion would exceed the episodic slot budget for ${moduleId}.`,
    );
  }

  return {
    kind: 'slotExpand',
    targetModuleId: moduleId,
    detail: {
      currentSlotCount: budget.currentEpisodicSlotCount,
      proposedAdditions: NGE_JUVENILE_DEFAULT_SLOT_EXPANSION_COUNT,
      hitRate,
      hitRateSource: 'metrics.utilization',
    },
    wiringCostDelta: NGE_JUVENILE_DEFAULT_SLOT_EXPANSION_COUNT,
  };
}

/**
 * Plan one rare evidence-gated node-addition delta for a single module.
 *
 * @param moduleId - Module receiving the planned node addition.
 * @param budget - DNA-configured growth caps and current live counts.
 * @param rewardDelta - Measured reward delta acting as the positive-evidence signal.
 * @returns One dry-run node-addition delta.
 */
export function planNodeAddition(
  moduleId: string,
  budget: NgeGrowthBudget,
  rewardDelta: number,
): NgeMorphDelta {
  if (rewardDelta <= NGE_JUVENILE_DEFAULT_NODE_REWARD_DELTA_FLOOR) {
    throw new NgeJuvenile_MorphError(
      `Node addition requires rewardDelta > ${NGE_JUVENILE_DEFAULT_NODE_REWARD_DELTA_FLOOR} for ${moduleId}.`,
    );
  }

  if (budget.currentNodeCount + 1 > budget.maxNodes) {
    throw new NgeJuvenile_BudgetError(
      `Node addition would exceed the node budget for ${moduleId}.`,
    );
  }

  return {
    kind: 'nodeAdd',
    targetModuleId: moduleId,
    detail: {
      currentNodeCount: budget.currentNodeCount,
      rewardDelta,
    },
    wiringCostDelta: 0,
  };
}

/**
 * Re-validate one dry-run morph delta against the current structural budget.
 *
 * @param delta - Planned morph delta to validate.
 * @param budget - DNA-configured growth caps and current live counts.
 */
export function validateMorphDelta(
  delta: NgeMorphDelta,
  budget: NgeGrowthBudget,
): void {
  switch (delta.kind) {
    case 'edgeDensify':
      if (budget.currentEdgeCount + delta.wiringCostDelta > budget.maxEdges) {
        throw new NgeJuvenile_BudgetError(
          `Edge densification would exceed the edge budget for ${delta.targetModuleId}.`,
        );
      }

      return;
    case 'slotExpand':
      if (
        budget.currentEpisodicSlotCount + delta.wiringCostDelta >
        budget.maxEpisodicSlots
      ) {
        throw new NgeJuvenile_BudgetError(
          `Slot expansion would exceed the episodic slot budget for ${delta.targetModuleId}.`,
        );
      }

      return;
    case 'nodeAdd':
      if (budget.currentNodeCount + 1 > budget.maxNodes) {
        throw new NgeJuvenile_BudgetError(
          `Node addition would exceed the node budget for ${delta.targetModuleId}.`,
        );
      }

      return;
    case 'edgePrune':
    case 'compact':
      return;
  }

  /* istanbul ignore next -- compile-time exhaustiveness guard for future morph kinds */
  const exhaustiveMorphKind: never = delta.kind;
  /* istanbul ignore next -- compile-time exhaustiveness guard for future morph kinds */
  throw new NgeJuvenile_MorphError(
    `Unhandled morph delta kind: ${String(exhaustiveMorphKind)}`,
  );
}

/**
 * Plan all eligible local growth deltas for one module in edge-first priority order.
 *
 * @param moduleId - Module receiving all planned local growth actions.
 * @param focusScore - Focus score for the target module.
 * @param metrics - Module metrics whose utilization and reward delta drive eligibility.
 * @param budget - DNA-configured growth caps and current live counts.
 * @param config - Resolved juvenile-phase configuration.
 * @param hysteresis - Current growth-side hysteresis state.
 * @returns Zero or more validated dry-run morph deltas in edge-first priority order.
 */
export function planGrowthMorphs(
  moduleId: string,
  focusScore: NgeFocusScore,
  metrics: NgeModuleMetricsSnapshot,
  budget: NgeGrowthBudget,
  config: NgeJuvenilePhaseConfig,
  hysteresis: NgeHysteresisState,
): NgeMorphDelta[] {
  // Step 1: Exit immediately until the growth hysteresis gate opens.
  if (!canGrowNow(hysteresis, config)) {
    return [];
  }

  const plannedDeltas: NgeMorphDelta[] = [];

  // Step 2: Try the preferred edge-densification increment first.
  try {
    const edgeDelta = planEdgeDensification(moduleId, budget, focusScore);
    validateMorphDelta(edgeDelta, budget);
    plannedDeltas.push(edgeDelta);
  } catch (error) {
    if (!(error instanceof NgeJuvenile_BudgetError)) {
      throw error;
    }
  }

  // Step 3: Try episodic-slot expansion when the hit-rate proxy and focus permit it.
  try {
    const slotDelta = planSlotExpansion(
      moduleId,
      metrics.utilization,
      budget,
      focusScore,
      config,
    );
    validateMorphDelta(slotDelta, budget);
    plannedDeltas.push(slotDelta);
  } catch (error) {
    if (
      !(error instanceof NgeJuvenile_BudgetError) &&
      !(error instanceof NgeJuvenile_MorphError)
    ) {
      throw error;
    }
  }

  // Step 4: Try the rare evidence-gated node addition last.
  try {
    const nodeDelta = planNodeAddition(moduleId, budget, metrics.rewardDelta);
    validateMorphDelta(nodeDelta, budget);
    plannedDeltas.push(nodeDelta);
  } catch (error) {
    if (
      !(error instanceof NgeJuvenile_BudgetError) &&
      !(error instanceof NgeJuvenile_MorphError)
    ) {
      throw error;
    }
  }

  // Step 5: Return the dry-run delta plan without mutating any inputs.
  return plannedDeltas;
}
