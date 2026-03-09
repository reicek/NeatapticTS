import type Network from '../../../../src/architecture/network';
import type { INetwork } from '../interfaces';
import { NetworkRefinement } from '../networkRefinement';
import type {
  MazeEvolutionCurriculumPhaseOutcome,
  MazeEvolutionRunResult,
} from './evolutionEngine.types';

/**
 * Resolve the stable curriculum-facing outcome of one maze evolution phase.
 *
 * @remarks
 * Browser-entry and curriculum-style tests should consume this helper instead
 * of re-implementing solve-threshold checks or winner carry-over refinement in
 * host-specific files. That keeps curriculum progression separate from the
 * engine-owned interpretation of a completed phase result.
 *
 * @param evolutionResult - Stable engine result returned by `EvolutionEngine.runMazeEvolution()`.
 * @param previousBestNetwork - Previously carried winner used when the latest phase has no replacement.
 * @param minProgressToPass - Progress threshold required before curriculum should advance.
 * @returns Shared curriculum outcome describing progress, solve state, and next carry-over seed.
 *
 * @example
 * ```ts
 * const phaseOutcome = resolveMazeEvolutionPhaseOutcome(result, previousBest, 95);
 * if (phaseOutcome.solved) {
 *   previousBest = phaseOutcome.nextBestNetwork;
 * }
 * ```
 */
export const resolveMazeEvolutionPhaseOutcome = (
  evolutionResult: MazeEvolutionRunResult,
  previousBestNetwork: INetwork | undefined,
  minProgressToPass: number,
): MazeEvolutionCurriculumPhaseOutcome => {
  const progress = evolutionResult.bestResult?.progress;

  return {
    result: evolutionResult,
    progress,
    solved: hasMazeEvolutionReachedCurriculumThreshold(
      progress,
      minProgressToPass,
    ),
    nextBestNetwork: refineMazeEvolutionCarryOverNetwork(
      (evolutionResult.bestNetwork as unknown as INetwork | undefined) ??
        undefined,
      previousBestNetwork,
    ),
  };
};

/**
 * Determine whether a completed phase should advance the surrounding curriculum.
 *
 * @param progress - Progress reported by the best run result.
 * @param minProgressToPass - Minimum progress percentage required to advance.
 * @returns Whether the phase counts as curriculum-complete.
 */
export const hasMazeEvolutionReachedCurriculumThreshold = (
  progress: unknown,
  minProgressToPass: number,
): boolean => {
  return typeof progress === 'number' && progress >= minProgressToPass;
};

/**
 * Refine the winning network before seeding the next curriculum phase.
 *
 * @param bestNetwork - Network returned by the latest evolution phase.
 * @param previousBestNetwork - Previously carried curriculum seed.
 * @returns Refined winner or the best available carry-over network.
 */
export const refineMazeEvolutionCarryOverNetwork = (
  bestNetwork: INetwork | undefined,
  previousBestNetwork: INetwork | undefined,
): INetwork | undefined => {
  if (!bestNetwork) {
    return previousBestNetwork;
  }

  try {
    const refinedNetwork = NetworkRefinement.refineWinnerWithBackprop(
      bestNetwork as unknown as Network,
    );
    return (refinedNetwork as unknown as INetwork | undefined) ?? bestNetwork;
  } catch {
    return bestNetwork;
  }
};
