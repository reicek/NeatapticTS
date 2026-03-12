import type { FlappyGameState } from '../flappyEnvironment.ts';

/**
 * Minimal network contract required by Flappy evaluation.
 *
 * The evaluation layer depends only on activation plus an optional stable id
 * used for deterministic seed mixing.
 */
export interface FlappyNetworkLike {
  activate(inputs: number[]): number[] | number;
  _id?: number;
}

/**
 * Runtime controls for one rollout evaluation.
 *
 * This is the public control surface for evaluation callers. The rollout layer
 * later normalizes these options into execution-safe context values.
 */
export interface FlappyRolloutOptions {
  /** Optional deterministic seed. Defaults to mixed genome id. */
  seed?: number;

  /** Difficulty scale used by curriculum scheduling in [0, 1]. */
  difficultyScale?: number;

  /** Per-rollout frame cap. Defaults to `FLAPPY_MAX_FRAMES_PER_EPISODE`. */
  maxFrames?: number;

  /** Enable heuristic early termination for clearly unrecoverable starts. */
  enableEarlyTermination?: boolean;

  /** Frames to wait before early-termination logic activates. */
  earlyTerminationGraceFrames?: number;

  /** Required consecutive unrecoverable frames to stop the rollout. */
  earlyTerminationConsecutiveFrames?: number;

  /** Enable normalized, capped fitness composition for stable ranking. */
  normalizeFitness?: boolean;

  /** Pipe count target used to normalize progress. */
  pipeProgressTarget?: number;
}

/**
 * Summary metrics for a single Flappy episode rollout.
 *
 * The result intentionally keeps both a single scalar `fitness` and the channel
 * breakdown that produced it, which makes reward debugging much easier.
 */
export interface FlappyEpisodeResult {
  /** Number of simulation frames survived. */
  framesSurvived: number;

  /** Number of pipes successfully passed. */
  pipesPassed: number;

  /** Whether the episode ended by collision/out-of-bounds/timeout. */
  done: boolean;

  /** Termination reason (when done). */
  doneReason?: FlappyGameState['doneReason'];

  /** Fitness value used by NEAT selection (higher is better). */
  fitness: number;

  /** Decomposed fitness channels used to build `fitness`. */
  fitnessBreakdown: {
    survival: number;
    pipeProgress: number;
    denseShaping: number;
    terminalShaping: number;
  };
}

/**
 * Aggregate statistics from evaluating one network across shared seeds.
 *
 * These statistics are the trainer-facing view of evaluation quality: mean,
 * median, $p90$, stability, and average gameplay progress.
 */
export interface FlappySeedBatchEvaluation {
  seedCount: number;
  meanFitness: number;
  medianFitness: number;
  p90Fitness: number;
  fitnessStdDev: number;
  robustFitness: number;
  meanPipesPassed: number;
  meanFramesSurvived: number;
}
