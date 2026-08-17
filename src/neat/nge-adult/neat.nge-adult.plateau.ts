import { NGE_ADULT_DEFAULT_MARGINAL_EPSILON } from './neat.nge-adult.constants';
import { NgeAdult_PlateauError } from './neat.nge-adult.errors';
import type { PlateauRecord } from './neat.nge-adult.types';

/**
 * Advances one rolling plateau record with a new adult reward delta.
 *
 * @param plateauRecord - Existing reward-delta evidence for the active adult zone.
 * @param rewardDelta - Latest normalized reward delta observed for the current evaluation window.
 * @param marginalEpsilon - Smallest improvement that still counts as meaningful positive return.
 * @returns Updated plateau state with the retained reward window and stagnation decision.
 */
export function advancePlateauRecord(
  plateauRecord: PlateauRecord,
  rewardDelta: number,
  marginalEpsilon = NGE_ADULT_DEFAULT_MARGINAL_EPSILON,
): PlateauRecord {
  const plateauWindow = plateauRecord.windowSize;
  const retainedRewardDeltas = [
    ...plateauRecord.rewardDeltas,
    rewardDelta,
  ].slice(-plateauWindow);
  const hasFullPlateauWindow = retainedRewardDeltas.length === plateauWindow;
  const isStagnant =
    hasFullPlateauWindow &&
    retainedRewardDeltas.every(
      (retainedRewardDelta) => retainedRewardDelta <= marginalEpsilon,
    );

  return {
    windowSize: plateauWindow,
    rewardDeltas: retainedRewardDeltas,
    isStagnant,
  };
}

/**
 * Resolves one marginal-return score for the current adult structural edit batch.
 *
 * @param rewardDelta - Normalized reward improvement produced by the current edit batch.
 * @param structuralEditCount - Number of structural edits responsible for the observed reward delta.
 * @returns Reward improvement normalized by the structural edit count.
 * @throws {NgeAdult_PlateauError} When the structural edit count is zero or negative.
 */
export function computeMarginalReturn(
  rewardDelta: number,
  structuralEditCount: number,
): number {
  if (structuralEditCount <= 0) {
    throw new NgeAdult_PlateauError(
      'Adult marginal return requires at least one structural edit.',
    );
  }

  return rewardDelta / structuralEditCount;
}
