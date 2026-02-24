import Network from '../../network';
import type {
  EvolutionLoopState,
  EvolutionSettings,
  EvolveOptions,
  NeatRuntime,
} from '../network.types';
import { computeComplexityPenalty } from './network.evolve.fitness.utils';
import {
  DISABLED_TARGET_ERROR,
  MAX_CONSECUTIVE_INVALID_ERRORS,
} from './network.evolve.utils.types';

/**
 * Run core evolution loop until stop condition is met.
 *
 * @param neatInstance - Active NEAT instance.
 * @param resolvedSettings - Scalar evolution settings.
 * @param targetError - Effective target error (-1 means disabled).
 * @param iterations - Optional max iteration count.
 * @returns Loop result snapshot.
 */
export async function runEvolutionLoop(
  neatInstance: NeatRuntime,
  resolvedSettings: EvolutionSettings,
  targetError: number,
  iterations: number | undefined,
): Promise<{ error: number; bestGenome: Network | undefined }> {
  const loopState = createInitialLoopState();
  const iterationsSpecified = typeof iterations === 'number';

  while (
    shouldContinueEvolution(
      loopState.currentError,
      targetError,
      iterationsSpecified,
      neatInstance.generation,
      iterations,
    )
  ) {
    const evolvedGenome = await neatInstance.evolve();
    applyEvolutionStep(loopState, evolvedGenome, resolvedSettings.growth);

    if (shouldAbortForInvalidErrors(loopState)) {
      break;
    }

    runScheduleCallbackSafely(
      resolvedSettings.schedule,
      neatInstance.generation,
      loopState.bestFitness,
      loopState.currentError,
    );
  }

  return {
    error: loopState.currentError,
    bestGenome: loopState.bestGenome,
  };
}

/**
 * Determine whether evolution loop should continue.
 *
 * @param currentError - Current derived error value.
 * @param targetError - Effective target error (-1 means disabled).
 * @param iterationsSpecified - Whether iterations limit is active.
 * @param currentGeneration - Current NEAT generation index.
 * @param maxIterations - Maximum iteration limit.
 * @returns True when loop should continue.
 */
function shouldContinueEvolution(
  currentError: number,
  targetError: number,
  iterationsSpecified: boolean,
  currentGeneration: number,
  maxIterations: number | undefined,
): boolean {
  const targetConditionMet =
    targetError !== DISABLED_TARGET_ERROR && currentError <= targetError;
  if (targetConditionMet) return false;
  if (!iterationsSpecified) return true;
  return currentGeneration < (maxIterations ?? 0);
}

/**
 * Creates initial loop state snapshot.
 *
 * @returns Initial loop state.
 */
function createInitialLoopState(): EvolutionLoopState {
  return {
    currentError: Infinity,
    bestFitness: -Infinity,
    bestGenome: undefined,
    consecutiveInvalidErrorCount: 0,
  };
}

/**
 * Applies one evolve() result to loop state.
 *
 * @param state - Mutable loop state.
 * @param evolvedGenome - Genome returned by NEAT evolve step.
 * @param growth - Complexity growth scalar.
 * @returns Nothing.
 */
function applyEvolutionStep(
  state: EvolutionLoopState,
  evolvedGenome: Network,
  growth: number,
): void {
  const evolvedFitness = evolvedGenome.score ?? -Infinity;
  state.currentError = deriveErrorFromFitness(
    evolvedFitness,
    evolvedGenome,
    growth,
  );

  const bestSnapshot = updateBestGenomeIfImproved(
    state.bestFitness,
    state.bestGenome,
    evolvedFitness,
    evolvedGenome,
  );
  state.bestFitness = bestSnapshot.bestFitness;
  state.bestGenome = bestSnapshot.bestGenome;
  state.consecutiveInvalidErrorCount = updateInvalidErrorCounter(
    state.consecutiveInvalidErrorCount,
    state.currentError,
  );
}

/**
 * Determines whether loop must abort due to invalid-error streak.
 *
 * @param state - Mutable loop state.
 * @returns True when invalid-error threshold is reached.
 */
function shouldAbortForInvalidErrors(state: EvolutionLoopState): boolean {
  return state.consecutiveInvalidErrorCount >= MAX_CONSECUTIVE_INVALID_ERRORS;
}

/**
 * Derive error from fitness by inverting score composition.
 *
 * @param fitness - Fitness value from fittest genome.
 * @param genome - Fittest genome.
 * @param growth - Complexity growth scalar.
 * @returns Derived error value.
 */
function deriveErrorFromFitness(
  fitness: number,
  genome: Network,
  growth: number,
): number {
  return -(fitness - computeComplexityPenalty(genome, growth)) || Infinity;
}

/**
 * Update best fitness/genome snapshot when improved.
 *
 * @param currentBestFitness - Current best fitness.
 * @param currentBestGenome - Current best genome.
 * @param candidateFitness - Candidate fitness.
 * @param candidateGenome - Candidate genome.
 * @returns Updated best snapshot.
 */
function updateBestGenomeIfImproved(
  currentBestFitness: number,
  currentBestGenome: Network | undefined,
  candidateFitness: number,
  candidateGenome: Network,
): { bestFitness: number; bestGenome: Network | undefined } {
  if (candidateFitness <= currentBestFitness) {
    return {
      bestFitness: currentBestFitness,
      bestGenome: currentBestGenome,
    };
  }

  return { bestFitness: candidateFitness, bestGenome: candidateGenome };
}

/**
 * Update invalid-error counter.
 *
 * @param currentCount - Current consecutive invalid-error count.
 * @param currentError - Current derived error value.
 * @returns Updated guard state.
 */
function updateInvalidErrorCounter(
  currentCount: number,
  currentError: number,
): number {
  if (Number.isFinite(currentError) && !Number.isNaN(currentError)) {
    return 0;
  }

  return currentCount + 1;
}

/**
 * Run schedule callback if schedule trigger is reached.
 *
 * @param scheduleConfig - Optional schedule configuration.
 * @param generation - Current generation.
 * @param bestFitness - Current best fitness.
 * @param error - Current error.
 * @returns Nothing.
 */
function runScheduleCallbackSafely(
  scheduleConfig: EvolveOptions['schedule'],
  generation: number,
  bestFitness: number,
  error: number,
): void {
  if (!scheduleConfig) return;
  if (generation % scheduleConfig.iterations !== 0) return;

  try {
    scheduleConfig.function({
      fitness: bestFitness,
      error,
      iteration: generation,
    });
  } catch {
    // Ignore schedule callback errors
  }
}
