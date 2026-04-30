/**
 * Generation-planning helpers for staged evaluation, curriculum difficulty, and
 * mutation cooling.
 *
 * The trainer does not decide stage budgets ad hoc inside the evolution loop.
 * Instead, each generation resolves one explicit plan that says which shared
 * seeds to use, how hard the environment should currently be, and how much
 * mutation pressure should remain.
 *
 * Generation planning map:
 * ```mermaid
 * flowchart TB
 *     Generation["generationIndex"] --> Mutation["resolveMutationSchedule()\nrate + amount"]
 *     Generation --> Difficulty["resolveCurriculumDifficultyScale()\ncourse difficulty"]
 *     Generation --> QuickSeeds["quick shared seeds"]
 *     Generation --> FullSeeds["full-stage shared seeds"]
 *     Generation --> ReevalSeeds["reevaluation shared seeds"]
 *     Difficulty --> QuickOptions["createQuickRolloutOptions()"]
 *     Difficulty --> FullOptions["createFullRolloutOptions()"]
 *     Difficulty --> ReevalOptions["createReevaluationRolloutOptions()"]
 *     Mutation --> Plan["FlappyGenerationEvaluationPlan"]
 *     QuickSeeds --> Plan
 *     FullSeeds --> Plan
 *     ReevalSeeds --> Plan
 *     QuickOptions --> Plan
 *     FullOptions --> Plan
 *     ReevalOptions --> Plan
 * ```
 */
import type { FlappyRolloutOptions } from '../flappyEvaluation';
import { FLAPPY_MAX_FRAMES_PER_EPISODE } from '../constants/constants';
import {
  clampValue,
  interpolateValue,
} from '../flappy.simulation.shared.utils';
import { createXorshift32 } from '../rng';
import {
  FLAPPY_TRAINER_FULL_ROLLOUT_EARLY_TERMINATION_CONSECUTIVE_FRAMES,
  FLAPPY_TRAINER_FULL_ROLLOUT_EARLY_TERMINATION_GRACE_FRAMES,
  FLAPPY_TRAINER_FULL_ROLLOUT_PIPE_PROGRESS_TARGET,
  FLAPPY_TRAINER_MUTATION_AMOUNT_END,
  FLAPPY_TRAINER_MUTATION_AMOUNT_START,
  FLAPPY_TRAINER_MUTATION_ANNEAL_GENERATIONS,
  FLAPPY_TRAINER_MUTATION_RATE_END,
  FLAPPY_TRAINER_MUTATION_RATE_START,
  FLAPPY_TRAINER_QUICK_ROLLOUT_EARLY_TERMINATION_CONSECUTIVE_FRAMES,
  FLAPPY_TRAINER_QUICK_ROLLOUT_EARLY_TERMINATION_GRACE_FRAMES,
  FLAPPY_TRAINER_QUICK_ROLLOUT_MAX_FRAMES,
  FLAPPY_TRAINER_QUICK_ROLLOUT_PIPE_PROGRESS_TARGET,
  FLAPPY_TRAINER_RECURRENT_FULL_ROLLOUT_EARLY_TERMINATION_GRACE_FRAMES,
  FLAPPY_TRAINER_RECURRENT_QUICK_ROLLOUT_EARLY_TERMINATION_GRACE_FRAMES,
} from './trainer.constants';
import type { FlappyGenerationEvaluationPlan } from './trainer.types';

/**
 * Mutation schedule used by generation planning and outer loop logging.
 *
 * These two numbers are treated as a single policy decision because the trainer
 * cools both the frequency and the size of mutations together.
 */
export interface FlappyMutationSchedule {
  mutationRate: number;
  mutationAmount: number;
}

/**
 * Resolves all per-generation evaluation controls.
 *
 * Think of this as the trainer's "generation contract": the rest of the system
 * can ask for one object and receive a fully prepared set of seeds, rollout
 * options, and annealed mutation values.
 *
 * @param generationIndex - Zero-based generation index.
 * @returns Full staged evaluation plan for the generation.
 */
export function resolveGenerationEvaluationPlan(
  generationIndex: number,
  isRecurrent: boolean = false,
): FlappyGenerationEvaluationPlan {
  const mutationSchedule = resolveMutationSchedule(generationIndex);
  const difficultyScale = resolveCurriculumDifficultyScale(generationIndex);

  return {
    generationIndex,
    mutationRate: mutationSchedule.mutationRate,
    mutationAmount: mutationSchedule.mutationAmount,
    difficultyScale,
    quickSeeds: buildSharedSeedBatch(generationIndex, 0x41a7, 3),
    fullSeeds: buildSharedSeedBatch(generationIndex, 0x7d2b, 8),
    reevaluationSeeds: buildSharedSeedBatch(generationIndex, 0xb8f3, 32),
    quickRolloutOptions: createQuickRolloutOptions(difficultyScale, isRecurrent),
    fullRolloutOptions: createFullRolloutOptions(difficultyScale, isRecurrent),
    reevaluationRolloutOptions:
      createReevaluationRolloutOptions(difficultyScale),
  };
}

/**
 * Resolve a smooth mutation annealing schedule.
 *
 * Early generations mutate more aggressively so the population can search the
 * space broadly. Later generations cool down so the trainer can refine useful
 * structures rather than constantly replacing them.
 *
 * @param generationIndex - Zero-based generation index.
 * @returns Mutation rate and mutation amount for this generation.
 */
export function resolveMutationSchedule(
  generationIndex: number,
): FlappyMutationSchedule {
  const annealProgress = clampValue(
    generationIndex / FLAPPY_TRAINER_MUTATION_ANNEAL_GENERATIONS,
    0,
    1,
  );
  return {
    mutationRate: interpolateValue(
      FLAPPY_TRAINER_MUTATION_RATE_START,
      FLAPPY_TRAINER_MUTATION_RATE_END,
      annealProgress,
    ),
    mutationAmount: interpolateValue(
      FLAPPY_TRAINER_MUTATION_AMOUNT_START,
      FLAPPY_TRAINER_MUTATION_AMOUNT_END,
      annealProgress,
    ),
  };
}

/**
 * Builds quick-screen rollout options.
 *
 * The quick stage is a cheap gate. It favors speed and comparability over fully
 * trusted estimates because weak genomes only need enough evidence to be ruled
 * out early.
 *
 * @param difficultyScale - Difficulty scale for this generation.
 * @returns Quick stage rollout options.
 */
function createQuickRolloutOptions(
  difficultyScale: number,
  isRecurrent: boolean,
): FlappyRolloutOptions {
  return {
    difficultyScale,
    maxFrames: FLAPPY_TRAINER_QUICK_ROLLOUT_MAX_FRAMES,
    enableEarlyTermination: true,
    earlyTerminationGraceFrames: isRecurrent
      ? FLAPPY_TRAINER_RECURRENT_QUICK_ROLLOUT_EARLY_TERMINATION_GRACE_FRAMES
      : FLAPPY_TRAINER_QUICK_ROLLOUT_EARLY_TERMINATION_GRACE_FRAMES,
    earlyTerminationConsecutiveFrames:
      FLAPPY_TRAINER_QUICK_ROLLOUT_EARLY_TERMINATION_CONSECUTIVE_FRAMES,
    normalizeFitness: true,
    pipeProgressTarget: FLAPPY_TRAINER_QUICK_ROLLOUT_PIPE_PROGRESS_TARGET,
  };
}

/**
 * Builds full-stage rollout options.
 *
 * This stage gives stronger candidates a longer, stricter test so the trainer
 * can refine the leaderboard before committing to expensive reevaluation.
 *
 * @param difficultyScale - Difficulty scale for this generation.
 * @returns Full stage rollout options.
 */
function createFullRolloutOptions(
  difficultyScale: number,
  isRecurrent: boolean,
): FlappyRolloutOptions {
  return {
    difficultyScale,
    maxFrames: FLAPPY_MAX_FRAMES_PER_EPISODE,
    enableEarlyTermination: true,
    earlyTerminationGraceFrames: isRecurrent
      ? FLAPPY_TRAINER_RECURRENT_FULL_ROLLOUT_EARLY_TERMINATION_GRACE_FRAMES
      : FLAPPY_TRAINER_FULL_ROLLOUT_EARLY_TERMINATION_GRACE_FRAMES,
    earlyTerminationConsecutiveFrames:
      FLAPPY_TRAINER_FULL_ROLLOUT_EARLY_TERMINATION_CONSECUTIVE_FRAMES,
    normalizeFitness: true,
    pipeProgressTarget: FLAPPY_TRAINER_FULL_ROLLOUT_PIPE_PROGRESS_TARGET,
  };
}

/**
 * Builds high-confidence reevaluation rollout options.
 *
 * Reevaluation deliberately disables early termination so the strongest
 * candidates are judged on a more faithful, less shortcut-heavy comparison.
 *
 * @param difficultyScale - Difficulty scale for this generation.
 * @returns Reevaluation stage rollout options.
 */
function createReevaluationRolloutOptions(
  difficultyScale: number,
): FlappyRolloutOptions {
  return {
    difficultyScale,
    maxFrames: FLAPPY_MAX_FRAMES_PER_EPISODE,
    enableEarlyTermination: false,
    normalizeFitness: true,
    pipeProgressTarget: FLAPPY_TRAINER_FULL_ROLLOUT_PIPE_PROGRESS_TARGET,
  };
}

/**
 * Resolve curriculum difficulty scale for the current generation.
 *
 * The course starts gentle, ramps through the middle generations, and then caps
 * at full difficulty once the population has had time to discover viable flight.
 *
 * @param generationIndex - Zero-based generation index.
 * @returns Difficulty scale in [0, 1].
 */
function resolveCurriculumDifficultyScale(generationIndex: number): number {
  if (generationIndex < 25) return 0;
  if (generationIndex >= 95) return 1;
  return (generationIndex - 25) / 70;
}

/**
 * Build deterministic shared seeds for one generation stage.
 *
 * Shared seeds are what make same-generation comparisons fair: genomes face the
 * same sampled worlds instead of winning because they happened to get a kinder
 * random rollout.
 *
 * @param generationIndex - Zero-based generation index.
 * @param stageSalt - Constant stage-specific salt.
 * @param seedCount - Number of seeds to produce.
 * @returns Deterministic shared seed list.
 */
function buildSharedSeedBatch(
  generationIndex: number,
  stageSalt: number,
  seedCount: number,
): number[] {
  const mixedSeed = mixSeed(generationIndex, stageSalt);
  const deterministicRandom = createXorshift32(mixedSeed);

  return Array.from({ length: seedCount }, () =>
    deterministicRandom.nextInt(1, 0x7fff_ffff),
  );
}

/**
 * Mixes generation and stage salts into a deterministic uint32 RNG seed.
 *
 * The small mixing pipeline spreads nearby generation numbers apart so adjacent
 * stages and generations do not accidentally reuse overly correlated seed sets.
 *
 * @param generationIndex - Current generation index.
 * @param stageSalt - Stage-specific salt.
 * @returns Mixed uint32 seed.
 */
function mixSeed(generationIndex: number, stageSalt: number): number {
  let seed = (generationIndex >>> 0) ^ (stageSalt >>> 0) ^ 0x9e3779b9;
  seed ^= seed >>> 16;
  seed = Math.imul(seed, 0x85ebca6b);
  seed ^= seed >>> 13;
  seed = Math.imul(seed, 0xc2b2ae35);
  seed ^= seed >>> 16;
  return seed >>> 0;
}
