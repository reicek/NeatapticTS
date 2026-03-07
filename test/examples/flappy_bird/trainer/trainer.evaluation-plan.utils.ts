import type { FlappyRolloutOptions } from '../flappyEvaluation';
import { FLAPPY_MAX_FRAMES_PER_EPISODE } from '../constants/constants';
import {
  clampValue,
  interpolateValue,
} from '../flappy.simulation.shared.utils';
import { createXorshift32 } from '../rng';
import type { FlappyGenerationEvaluationPlan } from './trainer.types';

/**
 * Mutation schedule used by generation planning and outer loop logging.
 */
export interface FlappyMutationSchedule {
  mutationRate: number;
  mutationAmount: number;
}

/**
 * Resolves all per-generation evaluation controls.
 *
 * @param generationIndex - Zero-based generation index.
 * @returns Full staged evaluation plan for the generation.
 */
export function resolveGenerationEvaluationPlan(
  generationIndex: number,
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
    quickRolloutOptions: createQuickRolloutOptions(difficultyScale),
    fullRolloutOptions: createFullRolloutOptions(difficultyScale),
    reevaluationRolloutOptions:
      createReevaluationRolloutOptions(difficultyScale),
  };
}

/**
 * Resolve a smooth mutation annealing schedule.
 *
 * @param generationIndex - Zero-based generation index.
 * @returns Mutation rate and mutation amount for this generation.
 */
export function resolveMutationSchedule(
  generationIndex: number,
): FlappyMutationSchedule {
  const annealProgress = clampValue(generationIndex / 120, 0, 1);
  return {
    mutationRate: interpolateValue(0.7, 0.25, annealProgress),
    mutationAmount: interpolateValue(2, 1, annealProgress),
  };
}

/**
 * Builds quick-screen rollout options.
 *
 * @param difficultyScale - Difficulty scale for this generation.
 * @returns Quick stage rollout options.
 */
function createQuickRolloutOptions(
  difficultyScale: number,
): FlappyRolloutOptions {
  return {
    difficultyScale,
    maxFrames: 1_500,
    enableEarlyTermination: true,
    earlyTerminationGraceFrames: 120,
    earlyTerminationConsecutiveFrames: 18,
    normalizeFitness: true,
    pipeProgressTarget: 12,
  };
}

/**
 * Builds full-stage rollout options.
 *
 * @param difficultyScale - Difficulty scale for this generation.
 * @returns Full stage rollout options.
 */
function createFullRolloutOptions(
  difficultyScale: number,
): FlappyRolloutOptions {
  return {
    difficultyScale,
    maxFrames: FLAPPY_MAX_FRAMES_PER_EPISODE,
    enableEarlyTermination: true,
    earlyTerminationGraceFrames: 220,
    earlyTerminationConsecutiveFrames: 28,
    normalizeFitness: true,
    pipeProgressTarget: 20,
  };
}

/**
 * Builds high-confidence reevaluation rollout options.
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
    pipeProgressTarget: 20,
  };
}

/**
 * Resolve curriculum difficulty scale for the current generation.
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
