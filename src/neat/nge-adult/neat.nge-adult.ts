import type {
  AdultGrowthCoolingDecision,
  AdultPruneCandidate,
  AdultPruneCompactDecision,
} from './neat.nge-adult.cooling';
import {
  arbitratePruneCompactDominance,
  resolveGrowthCoolingDecision,
} from './neat.nge-adult.cooling';
import {
  advanceGainStabilityRecord,
  detectEquilibriumCandidate,
} from './neat.nge-adult.equilibrium';
import {
  advancePlateauRecord,
  computeMarginalReturn,
} from './neat.nge-adult.plateau';
import type { AdultState, EquilibriumCandidate } from './neat.nge-adult.types';

/**
 * Runtime input bag consumed by one adult-phase orchestration step call.
 */
export interface AdvanceAdultStateInput {
  /** Whether the adult phase is active for the current NGE run. */
  adultPhaseEnabled: boolean;
  /** Persistent adult state carried into the current cycle. */
  adultState: AdultState;
  /** Whether compact still has headroom in the current window. */
  compactEligible: boolean;
  /** Current normalized adult focus score. */
  focusScore: number;
  /** Latest adult neuromodulator gain measurement. */
  gainMeasurement: number;
  /** Ranked adult prune candidates visible in the current window. */
  pruneCandidates: readonly AdultPruneCandidate[];
  /** Latest normalized reward delta observed in the current window. */
  rewardDelta: number;
  /** Number of structural edits that produced the observed reward delta. */
  structuralEditCount: number;
  /** Stable adult-zone identifier under evaluation. */
  zoneId: string;
}

/**
 * Top-level adult orchestration result returned for one full evaluation cycle.
 */
export interface AdvanceAdultStateResult {
  /** Updated adult state after the current cycle. */
  adultState: AdultState;
  /** Adult cooling decision for the current cycle, when resolved. */
  coolingDecision: AdultGrowthCoolingDecision | null;
  /** Current equilibrium candidate event for downstream assimilation. */
  equilibriumCandidate: EquilibriumCandidate | null;
  /** Reward improvement normalized by the current edit batch size. */
  marginalReturn: number | null;
  /** Adult prune-versus-compact arbitration decision for the current cycle. */
  pruneCompactDecision: AdultPruneCompactDecision | null;
}

/**
 * Advance one complete owner-local adult optimization and equilibrium detection cycle.
 *
 * @param input - Runtime inputs for the current adult cycle.
 * @returns The composed adult transition for the current cycle.
 */
export function advanceAdultState(
  input: AdvanceAdultStateInput,
): AdvanceAdultStateResult {
  // Step 1: Advance the owner-local plateau, gain, cooling, and prune/compact decisions.
  const plateauRecord = advancePlateauRecord(
    input.adultState.plateauRecord,
    input.rewardDelta,
  );
  const gainStabilityRecord = advanceGainStabilityRecord(
    input.adultState.gainStabilityRecord,
    input.gainMeasurement,
  );
  const coolingDecision = resolveGrowthCoolingDecision(input.focusScore);
  const pruneCompactDecision = arbitratePruneCompactDominance(
    input.pruneCandidates,
    input.compactEligible,
  );

  // Step 2: Resolve the current marginal return and equilibrium emission.
  const marginalReturn = computeMarginalReturn(
    input.rewardDelta,
    input.structuralEditCount,
  );
  const equilibriumCandidate = detectEquilibriumCandidate(
    input.zoneId,
    plateauRecord,
    gainStabilityRecord,
  );

  // Step 3: Fold the updated owner-local state and transition outputs.
  return {
    adultState: {
      plateauRecord,
      gainStabilityRecord,
      equilibriumCandidate: {
        zoneId: input.zoneId,
        isGainStable: gainStabilityRecord.isStable,
        isPlateau: plateauRecord.isStagnant,
      },
      growthCoolingActive: coolingDecision.growthCoolingActive,
      cycleCount: input.adultState.cycleCount + 1,
    },
    coolingDecision,
    equilibriumCandidate,
    marginalReturn,
    pruneCompactDecision,
  };
}
