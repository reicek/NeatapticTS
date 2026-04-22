/**
 * Rollout-internal type contracts.
 *
 * These runtime-only types are the private vocabulary of one rollout episode.
 * They keep the public evaluation API compact while still giving the rollout
 * loop explicit names for the data it carries between phases.
 *
 * Read them as three layers:
 *
 * - `RolloutEpisodeContext`: immutable, normalized configuration.
 * - `RolloutEpisodeRuntimeState`: mutable execution state.
 * - fitness and shaping types: named reward channels used during folding.
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
 * This is the mutable side of the rollout: world state, RNG, shared
 * observation-memory compatibility state, and the counters accumulated during
 * execution.
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
  velocityStabilityReward: number;
};
