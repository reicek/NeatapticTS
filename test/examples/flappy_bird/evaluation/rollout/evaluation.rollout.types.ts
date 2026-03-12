/**
 * Rollout-internal type contracts.
 *
 * This file will host runtime-only rollout types that should not widen the
 * public evaluation-level API surface.
 *
 * That separation keeps the public evaluation API compact even as rollout
 * internals become more detailed.
 */
import { createSharedObservationMemoryState } from '../../flappy.simulation.shared.utils';
import { createXorshift32 } from '../../rng';
import type { FlappyGameState } from '../../flappyEnvironment.ts';

/**
 * Immutable rollout options normalized into execution-safe ranges.
 *
 * Every field here is ready for direct use inside the episode loop.
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
 *
 * This is the mutable side of the rollout: world state, RNG, temporal memory,
 * and the counters accumulated during execution.
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
 *
 * Named channels make reward design easier to audit than a single opaque number.
 */
export type RolloutFitnessBreakdown = {
  survivalFitness: number;
  pipePassFitness: number;
  denseShapingFitness: number;
  terminalShapingFitness: number;
};

/**
 * Per-frame dense shaping channels resolved from consecutive observations.
 *
 * The shaping system rewards more than survival: it also tracks approach,
 * centering, clearance, and stable motion.
 */
export type DenseShapingRewardComponents = {
  nextGapAlignmentReward: number;
  approachProgressReward: number;
  centeringProgressReward: number;
  clearanceReward: number;
  secondGapAlignmentReward: number;
  velocityStabilityReward: number;
};
