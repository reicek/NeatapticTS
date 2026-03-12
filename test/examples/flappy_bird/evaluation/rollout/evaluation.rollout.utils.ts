/**
 * Rollout shaping and result helpers.
 *
 * This file will host rollout-local fitness composition, shaping utilities,
 * and terminal result assembly helpers.
 *
 * Educational note:
 * The rollout subsystem separates simulation from scoring on purpose. The
 * services file determines what happened; this file determines how that episode
 * should be interpreted as fitness.
 */
import {
  FLAPPY_FITNESS_ALIGNMENT_WEIGHT_PER_FRAME,
  FLAPPY_FITNESS_APPROACH_PROGRESS_WEIGHT,
  FLAPPY_FITNESS_BONUS_PER_PIPE,
  FLAPPY_FITNESS_CENTERING_PROGRESS_WEIGHT,
  FLAPPY_FITNESS_CLEARANCE_WEIGHT_PER_FRAME,
  FLAPPY_FITNESS_SECOND_GAP_ALIGNMENT_WEIGHT_PER_FRAME,
  FLAPPY_FITNESS_STABLE_VELOCITY_WEIGHT_PER_FRAME,
  FLAPPY_FITNESS_SURVIVAL_WEIGHT,
  FLAPPY_FITNESS_TERMINAL_ALIGNMENT_BONUS_WEIGHT,
  FLAPPY_FITNESS_TERMINAL_PROGRESS_BONUS_WEIGHT,
} from '../../constants/constants';
import {
  FLAPPY_EVALUATION_DEFAULT_PIPE_PROGRESS_TARGET,
  FLAPPY_EVALUATION_DENSE_SHAPING_FRAMES_NORMALIZER,
  FLAPPY_EVALUATION_NORMALIZED_DENSE_WEIGHT,
  FLAPPY_EVALUATION_NORMALIZED_PROGRESS_WEIGHT,
  FLAPPY_EVALUATION_NORMALIZED_SURVIVAL_WEIGHT,
  FLAPPY_EVALUATION_NORMALIZED_TERMINAL_WEIGHT,
  FLAPPY_EVALUATION_UNRECOVERABLE_ABOVE_GAP_DELTA,
  FLAPPY_EVALUATION_UNRECOVERABLE_BELOW_GAP_DELTA,
  FLAPPY_EVALUATION_UNRECOVERABLE_CLEARANCE_THRESHOLD,
  FLAPPY_EVALUATION_UNRECOVERABLE_FALLING_VELOCITY,
  FLAPPY_EVALUATION_UNRECOVERABLE_RISING_VELOCITY,
} from '../evaluation.constants';
import {
  getFlappyObservationFeatures,
  type FlappyGameState,
  type FlappyObservationFeatures,
} from '../../flappyEnvironment.ts';
import { clampValue } from '../../flappy.simulation.shared.utils';
import {
  FLAPPY_ROLLOUT_MIN_MAX_FRAMES,
  FLAPPY_ROLLOUT_ZERO_FITNESS,
} from './evaluation.rollout.constants';
import type {
  DenseShapingRewardComponents,
  RolloutEpisodeContext,
  RolloutEpisodeRuntimeState,
  RolloutFitnessBreakdown,
} from './evaluation.rollout.types';
import type { FlappyEpisodeResult } from '../evaluation.types';

/**
 * Composes the final rollout result from the terminal game state.
 *
 * This is the final fold step for rollout execution: internal counters and
 * shaping channels become the public `FlappyEpisodeResult` consumed by training
 * and reporting.
 *
 * @param rolloutEpisodeContext - Normalized rollout configuration.
 * @param rolloutEpisodeRuntimeState - Mutable runtime state.
 * @returns Episode result details.
 */
export function composeRolloutEpisodeResult(
  rolloutEpisodeContext: RolloutEpisodeContext,
  rolloutEpisodeRuntimeState: RolloutEpisodeRuntimeState,
): FlappyEpisodeResult {
  // Step 1: Resolve fitness-channel breakdown values from the final game state.
  const framesSurvived = rolloutEpisodeRuntimeState.state.frameIndex;
  const pipesPassed = rolloutEpisodeRuntimeState.state.pipesPassed;
  const rolloutFitnessBreakdown = resolveRolloutFitnessBreakdown(
    rolloutEpisodeContext,
    rolloutEpisodeRuntimeState,
    framesSurvived,
    pipesPassed,
  );

  // Step 2: Compose normalized or raw fitness from the resolved breakdown.
  const fitness = rolloutEpisodeContext.normalizeFitness
    ? composeNormalizedFitness(
        framesSurvived,
        pipesPassed,
        rolloutFitnessBreakdown.denseShapingFitness,
        rolloutFitnessBreakdown.terminalShapingFitness,
        rolloutEpisodeContext.maxFramesPerEpisode,
        rolloutEpisodeContext.pipeProgressTarget,
      )
    : resolveUnnormalizedRolloutFitness(rolloutFitnessBreakdown);

  // Step 3: Return the public episode result payload.
  return {
    framesSurvived,
    pipesPassed,
    done: rolloutEpisodeRuntimeState.state.done,
    doneReason: rolloutEpisodeRuntimeState.state.doneReason,
    fitness,
    fitnessBreakdown: {
      survival: rolloutFitnessBreakdown.survivalFitness,
      pipeProgress: rolloutFitnessBreakdown.pipePassFitness,
      denseShaping: rolloutFitnessBreakdown.denseShapingFitness,
      terminalShaping: rolloutFitnessBreakdown.terminalShapingFitness,
    },
  };
}

/**
 * Computes dense reward shaping from consecutive observations.
 *
 * Dense shaping rewards incremental improvement throughout an episode instead of
 * paying out only at the end, which gives evolution a more informative signal.
 *
 * @param previousFeatures - Observation before stepping the environment.
 * @param currentFeatures - Observation after stepping the environment.
 * @returns Per-step shaped reward.
 */
export function computeDenseShapingReward(
  previousFeatures: FlappyObservationFeatures,
  currentFeatures: FlappyObservationFeatures,
): number {
  // Step 1: Resolve each shaping component from the consecutive observations.
  const denseShapingRewardComponents = resolveDenseShapingRewardComponents(
    previousFeatures,
    currentFeatures,
  );

  // Step 2: Sum the shaping channels into one per-frame reward.
  return (
    denseShapingRewardComponents.nextGapAlignmentReward +
    denseShapingRewardComponents.approachProgressReward +
    denseShapingRewardComponents.centeringProgressReward +
    denseShapingRewardComponents.clearanceReward +
    denseShapingRewardComponents.secondGapAlignmentReward +
    denseShapingRewardComponents.velocityStabilityReward
  );
}

/**
 * Detects trajectories that are usually irrecoverable in early warmup.
 *
 * The heuristic focuses on obvious early failures, where spending more rollout
 * budget is least informative.
 *
 * @param observationFeatures - Post-step observation features.
 * @returns Whether the current trajectory appears unrecoverable.
 */
export function isBirdLikelyUnrecoverable(
  observationFeatures: FlappyObservationFeatures,
): boolean {
  const farOutsideGap =
    observationFeatures.normalizedNextGapClearance <
    FLAPPY_EVALUATION_UNRECOVERABLE_CLEARANCE_THRESHOLD;
  const belowGapAndStillFalling =
    observationFeatures.normalizedDeltaToNextGap >
      FLAPPY_EVALUATION_UNRECOVERABLE_BELOW_GAP_DELTA &&
    observationFeatures.normalizedVelocity >
      FLAPPY_EVALUATION_UNRECOVERABLE_FALLING_VELOCITY;
  const aboveGapAndStillRising =
    observationFeatures.normalizedDeltaToNextGap <
      FLAPPY_EVALUATION_UNRECOVERABLE_ABOVE_GAP_DELTA &&
    observationFeatures.normalizedVelocity <
      FLAPPY_EVALUATION_UNRECOVERABLE_RISING_VELOCITY;

  return farOutsideGap && (belowGapAndStillFalling || aboveGapAndStillRising);
}

/**
 * Resolves the raw fitness channels from the final episode state.
 *
 * Separating raw channels from final composition makes reward rebalancing much
 * easier to reason about.
 *
 * @param rolloutEpisodeContext - Normalized rollout configuration.
 * @param rolloutEpisodeRuntimeState - Mutable runtime state.
 * @param framesSurvived - Final frame count.
 * @param pipesPassed - Final pipe-pass count.
 * @returns Fitness-channel breakdown.
 */
function resolveRolloutFitnessBreakdown(
  rolloutEpisodeContext: RolloutEpisodeContext,
  rolloutEpisodeRuntimeState: RolloutEpisodeRuntimeState,
  framesSurvived: number,
  pipesPassed: number,
): RolloutFitnessBreakdown {
  // Step 1: Resolve the direct survival and progress channels.
  const survivalFitness = framesSurvived * FLAPPY_FITNESS_SURVIVAL_WEIGHT;
  const pipePassFitness = pipesPassed * FLAPPY_FITNESS_BONUS_PER_PIPE;

  // Step 2: Resolve the terminal shaping bonus from the final observation.
  const terminalShapingFitness = computeTerminalShapingFitness(
    rolloutEpisodeRuntimeState.state,
    rolloutEpisodeContext.difficultyScale,
  );

  return {
    survivalFitness,
    pipePassFitness,
    denseShapingFitness: rolloutEpisodeRuntimeState.denseShapingFitness,
    terminalShapingFitness,
  };
}

/**
 * Resolves raw fitness by summing every fitness channel.
 *
 * This is the legacy unnormalized objective. The normalized path below caps
 * channels so no single term dominates the whole score.
 *
 * @param rolloutFitnessBreakdown - Fitness-channel breakdown.
 * @returns Raw unnormalized fitness.
 */
function resolveUnnormalizedRolloutFitness(
  rolloutFitnessBreakdown: RolloutFitnessBreakdown,
): number {
  // Step 1: Sum all channels directly for the legacy unnormalized objective.
  return (
    rolloutFitnessBreakdown.survivalFitness +
    rolloutFitnessBreakdown.pipePassFitness +
    rolloutFitnessBreakdown.denseShapingFitness +
    rolloutFitnessBreakdown.terminalShapingFitness
  );
}

/**
 * Resolves every dense-shaping reward component from consecutive observations.
 *
 * If you want background reading, the Wikipedia article on "reward shaping" is
 * a good high-level companion concept for why these components exist.
 *
 * @param previousFeatures - Observation before stepping the environment.
 * @param currentFeatures - Observation after stepping the environment.
 * @returns Dense-shaping reward components.
 */
function resolveDenseShapingRewardComponents(
  previousFeatures: FlappyObservationFeatures,
  currentFeatures: FlappyObservationFeatures,
): DenseShapingRewardComponents {
  // Step 1: Reward alignment with the next immediate pipe gap.
  const nextGapAlignmentReward =
    Math.max(
      FLAPPY_ROLLOUT_ZERO_FITNESS,
      FLAPPY_ROLLOUT_MIN_MAX_FRAMES -
        Math.abs(currentFeatures.normalizedDeltaToNextGap),
    ) * FLAPPY_FITNESS_ALIGNMENT_WEIGHT_PER_FRAME;

  // Step 2: Reward forward progress toward the next pipe.
  const approachProgressReward =
    Math.max(
      FLAPPY_ROLLOUT_ZERO_FITNESS,
      previousFeatures.normalizedDistanceToNextPipe -
        currentFeatures.normalizedDistanceToNextPipe,
    ) * FLAPPY_FITNESS_APPROACH_PROGRESS_WEIGHT;

  // Step 3: Reward reduction in absolute next-gap centering error.
  const centeringProgressReward =
    Math.max(
      FLAPPY_ROLLOUT_ZERO_FITNESS,
      Math.abs(previousFeatures.normalizedDeltaToNextGap) -
        Math.abs(currentFeatures.normalizedDeltaToNextGap),
    ) * FLAPPY_FITNESS_CENTERING_PROGRESS_WEIGHT;

  // Step 4: Reward maintaining positive clearance around the next gap.
  const clearanceReward =
    Math.max(
      FLAPPY_ROLLOUT_ZERO_FITNESS,
      currentFeatures.normalizedNextGapClearance,
    ) * FLAPPY_FITNESS_CLEARANCE_WEIGHT_PER_FRAME;

  // Step 5: Reward alignment with the second upcoming gap to encourage stability.
  const secondGapAlignmentReward =
    Math.max(
      FLAPPY_ROLLOUT_ZERO_FITNESS,
      FLAPPY_ROLLOUT_MIN_MAX_FRAMES -
        Math.abs(currentFeatures.normalizedDeltaToSecondGap),
    ) * FLAPPY_FITNESS_SECOND_GAP_ALIGNMENT_WEIGHT_PER_FRAME;

  // Step 6: Reward stable velocity magnitudes that avoid extreme oscillation.
  const velocityStabilityReward =
    Math.max(
      FLAPPY_ROLLOUT_ZERO_FITNESS,
      FLAPPY_ROLLOUT_MIN_MAX_FRAMES -
        Math.abs(currentFeatures.normalizedVelocity),
    ) * FLAPPY_FITNESS_STABLE_VELOCITY_WEIGHT_PER_FRAME;

  return {
    nextGapAlignmentReward,
    approachProgressReward,
    centeringProgressReward,
    clearanceReward,
    secondGapAlignmentReward,
    velocityStabilityReward,
  };
}

/**
 * Adds small terminal bonuses from final progress/alignment signals.
 *
 * Terminal bonuses refine the final ranking, but they are intentionally smaller
 * than the main survival and pipe-progress channels.
 *
 * @param episodeState - Final rollout state.
 * @param difficultyScale - Active rollout difficulty scale.
 * @returns Terminal shaping reward.
 */
function computeTerminalShapingFitness(
  episodeState: FlappyGameState,
  difficultyScale: number,
): number {
  const finalObservationFeatures = getFlappyObservationFeatures(
    episodeState,
    difficultyScale,
  );
  const finalAlignment =
    1 - Math.abs(finalObservationFeatures.normalizedDeltaToNextGap);
  const finalProgress =
    1 -
    Math.max(
      0,
      Math.min(1, finalObservationFeatures.normalizedDistanceToNextPipe),
    );

  const terminalAlignmentBonus =
    Math.max(0, finalAlignment) *
    FLAPPY_FITNESS_TERMINAL_ALIGNMENT_BONUS_WEIGHT;
  const terminalProgressBonus =
    Math.max(0, finalProgress) * FLAPPY_FITNESS_TERMINAL_PROGRESS_BONUS_WEIGHT;

  return terminalAlignmentBonus + terminalProgressBonus;
}

/**
 * Normalize and cap fitness channels so no single reward term dominates.
 *
 * Educational note:
 * Channel normalization is a pragmatic way to keep the objective balanced across
 * episodes of different lengths and levels of progress.
 *
 * @param framesValue - Frames survived for the episode.
 * @param pipesPassedValue - Pipes passed during the episode.
 * @param denseShapingValue - Accumulated dense shaping reward.
 * @param terminalShapingValue - Terminal shaping reward.
 * @param maxFramesValue - Frame budget used for the episode.
 * @param pipeProgressTarget - Optional target used to normalize pipe progress.
 * @returns Normalized composite fitness.
 */
function composeNormalizedFitness(
  framesValue: number,
  pipesPassedValue: number,
  denseShapingValue: number,
  terminalShapingValue: number,
  maxFramesValue: number,
  pipeProgressTarget: number | undefined,
): number {
  const effectivePipeProgressTarget = Math.max(
    1,
    Math.trunc(
      pipeProgressTarget ?? FLAPPY_EVALUATION_DEFAULT_PIPE_PROGRESS_TARGET,
    ),
  );
  const normalizedSurvival = clampValue(
    framesValue / Math.max(1, maxFramesValue),
    0,
    1,
  );
  const normalizedPipeProgress = clampValue(
    pipesPassedValue / effectivePipeProgressTarget,
    0,
    1,
  );
  const normalizedDenseShaping = clampValue(
    denseShapingValue /
      Math.max(
        1,
        framesValue * FLAPPY_EVALUATION_DENSE_SHAPING_FRAMES_NORMALIZER,
      ),
    0,
    1,
  );
  const normalizedTerminalShaping = clampValue(
    terminalShapingValue /
      Math.max(
        1,
        FLAPPY_FITNESS_TERMINAL_ALIGNMENT_BONUS_WEIGHT +
          FLAPPY_FITNESS_TERMINAL_PROGRESS_BONUS_WEIGHT,
      ),
    0,
    1,
  );

  return (
    normalizedSurvival *
      FLAPPY_EVALUATION_NORMALIZED_SURVIVAL_WEIGHT *
      FLAPPY_FITNESS_SURVIVAL_WEIGHT +
    normalizedPipeProgress * FLAPPY_EVALUATION_NORMALIZED_PROGRESS_WEIGHT +
    normalizedDenseShaping * FLAPPY_EVALUATION_NORMALIZED_DENSE_WEIGHT +
    normalizedTerminalShaping * FLAPPY_EVALUATION_NORMALIZED_TERMINAL_WEIGHT
  );
}
