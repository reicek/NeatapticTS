import {
  NGE_JUVENILE_DEFAULT_NODE_GROWTH_SIGNAL_FLOOR,
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

/**
 * Growth-side morph kinds that the juvenile planner can emit and the lifecycle
 * can commit. Edge densify is the preferred fast path; slot expansion and node
 * addition are rarer, higher-cost growth actions.
 */
export type NgeGrowthMorphKind = 'edgeDensify' | 'slotExpand' | 'nodeAdd';

/**
 * Check whether juvenile growth may commit in the current window.
 *
 * The gate opens only after `hysteresisWindowCount` consecutive windows have
 * carried positive focus evidence *and* the previous growth cooldown has
 * expired. Once growth commits, `commitGrowth` resets the streak and starts a
 * new cooldown, so two morphs cannot fire back-to-back without fresh evidence.
 *
 * @param hysteresis - Current hysteresis state tracked across windows.
 * @param config - Resolved juvenile-phase configuration.
 * @returns `true` when the positive-focus streak is satisfied and cooldown is clear.
 *
 * @example
 * ```ts
 * const hysteresis = { growthPositiveWindowCount: 3, cooldownWindowsRemaining: 0 };
 * const config = resolveFocusConfig({ hysteresisWindowCount: 3 });
 * console.log(canGrowNow(hysteresis, config)); // true
 * ```
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
  config: NgeJuvenilePhaseConfig,
): NgeMorphDelta {
  const proposedAdditions = config.edgeDensificationCount;

  if (budget.currentEdgeCount + proposedAdditions > budget.maxEdges) {
    throw new NgeJuvenile_BudgetError(
      `Edge densification would exceed the edge budget for ${moduleId}.`,
    );
  }

  return {
    kind: 'edgeDensify',
    targetModuleId: moduleId,
    detail: {
      currentEdgeCount: budget.currentEdgeCount,
      proposedAdditions,
      normalizedFocusScore: focusScore.normalizedScore,
    },
    wiringCostDelta: proposedAdditions,
  };
}

/**
 * Plan one local episodic-slot expansion delta for a single module.
 *
 * This planner uses `metrics.utilization` as the episodic hit-rate proxy until
 * a dedicated hit-rate metric is added to the module snapshot.
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
 * Compute the composite node-growth signal from a focus score using the same
 * normalized metric weights that produced the raw focus score. The signal is in
 * [-1, 1] and replaces the old raw-reward-delta gate.
 *
 * @param score - Focus score for the target module.
 * @param config - Resolved juvenile configuration carrying focus weights.
 * @returns Scalar growth signal; values above the configured floor open the gate.
 */
function computeNodeGrowthSignal(
  score: NgeFocusScore,
  config: NgeJuvenilePhaseConfig,
): number {
  const weights = config.focusWeights;

  return (
    score.normalizedUtilization * weights.w_u +
    score.normalizedRewardDelta * weights.w_r +
    score.normalizedNovelty * weights.w_n +
    score.normalizedStabilityAge * weights.w_s -
    score.normalizedWiringCost * weights.w_c
  );
}

/**
 * Plan one rare evidence-gated node-addition delta for a single module.
 *
 * Eligibility is now driven by the composite focus-derived growth signal rather
 * than raw reward delta alone. The planned insertion count honors the DNA
 * `nodeAdditionCount` and available node budget.
 *
 * @param moduleId - Module receiving the planned node addition.
 * @param budget - DNA-configured growth caps and current live counts.
 * @param score - Focus score carrying normalized metrics and the growth flag.
 * @param config - Resolved juvenile configuration.
 * @returns One dry-run node-addition delta.
 */
export function planNodeAddition(
  moduleId: string,
  budget: NgeGrowthBudget,
  score: NgeFocusScore,
  config: NgeJuvenilePhaseConfig,
): NgeMorphDelta {
  if (!score.supportsGrowth) {
    throw new NgeJuvenile_MorphError(
      `Node addition requires a growth-supporting focus score for ${moduleId}.`,
    );
  }

  const growthSignal = computeNodeGrowthSignal(score, config);
  if (
    growthSignal <=
    (config.nodeGrowthSignalFloor ??
      NGE_JUVENILE_DEFAULT_NODE_GROWTH_SIGNAL_FLOOR)
  ) {
    throw new NgeJuvenile_MorphError(
      `Node addition requires growthSignal > ${config.nodeGrowthSignalFloor} for ${moduleId}.`,
    );
  }

  const requestedCount = Math.max(
    1,
    Math.floor(config.nodeAdditionCount * (1 + growthSignal)),
  );
  const availableHeadroom = budget.maxNodes - budget.currentNodeCount;
  const count = Math.min(requestedCount, Math.max(1, availableHeadroom));

  if (count <= 0 || budget.currentNodeCount + count > budget.maxNodes) {
    throw new NgeJuvenile_BudgetError(
      `Node addition would exceed the node budget for ${moduleId}.`,
    );
  }

  return {
    kind: 'nodeAdd',
    targetModuleId: moduleId,
    detail: {
      currentNodeCount: budget.currentNodeCount,
      growthSignal,
      proposedAdditions: count,
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
    case 'nodeAdd': {
      const nodeAdditions = (delta.detail.proposedAdditions as number) ?? 1;
      if (budget.currentNodeCount + nodeAdditions > budget.maxNodes) {
        throw new NgeJuvenile_BudgetError(
          `Node addition would exceed the node budget for ${delta.targetModuleId}.`,
        );
      }

      return;
    }
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
 * The planner tries densification first, slot expansion second, and node
 * addition last. Each candidate is validated against the supplied DNA budget
 * before it is returned. If the hysteresis gate is closed, the function returns
 * an empty array without throwing.
 *
 * @param moduleId - Module receiving all planned local growth actions.
 * @param focusScore - Focus score for the target module.
 * @param metrics - Module metrics whose utilization and reward delta drive eligibility.
 * @param budget - DNA-configured growth caps and current live counts.
 * @param config - Resolved juvenile-phase configuration.
 * @param hysteresis - Current growth-side hysteresis state.
 * @returns Zero or more validated dry-run morph deltas in edge-first priority order.
 *
 * @example
 * ```ts
 * const deltas = planGrowthMorphs('policy', focus, metrics, budget, config, hysteresis);
 * console.log(deltas.map((d) => d.kind)); // ['edgeDensify'] (or [] when gated)
 * ```
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
    const edgeDelta = planEdgeDensification(
      moduleId,
      budget,
      focusScore,
      config,
    );
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
    const nodeDelta = planNodeAddition(moduleId, budget, focusScore, config);
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
