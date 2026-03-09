/**
 * Rollout-internal type contracts.
 *
 * This file will host runtime-only rollout types that should not widen the
 * public evaluation-level API surface.
 */
import { createSharedObservationMemoryState } from '../../flappy.simulation.shared.utils';
import { createXorshift32 } from '../../rng';
import type { FlappyGameState } from '../../flappyEnvironment.ts';

/**
 * Immutable rollout options normalized into execution-safe ranges.
 */
export type RolloutEpisodeContext = {
  seed: number;
  difficultyScale: number;
  maxFramesPerEpisode: number;
  earlyTerminationGraceFrames: number;
  earlyTerminationConsecutiveFrames: number;
  enableEarlyTermination: boolean;
  normalizeFitness: boolean;
  pipeProgressTarget: number | undefined;
};

/**
 * Mutable runtime state accumulated while one rollout episode executes.
 */
export type RolloutEpisodeRuntimeState = {
  rng: ReturnType<typeof createXorshift32>;
  state: FlappyGameState;
  observationMemoryState: ReturnType<typeof createSharedObservationMemoryState>;
  denseShapingFitness: number;
  unrecoverableFrameCount: number;
};

/**
 * Fitness-channel breakdown used to compose the public episode result.
 */
export type RolloutFitnessBreakdown = {
  survivalFitness: number;
  pipePassFitness: number;
  denseShapingFitness: number;
  terminalShapingFitness: number;
};

/**
 * Per-frame dense shaping channels resolved from consecutive observations.
 */
export type DenseShapingRewardComponents = {
  nextGapAlignmentReward: number;
  approachProgressReward: number;
  centeringProgressReward: number;
  clearanceReward: number;
  secondGapAlignmentReward: number;
  velocityStabilityReward: number;
};
