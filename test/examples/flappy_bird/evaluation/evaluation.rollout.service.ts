import {
  FLAPPY_CONTROL_SUBSTEPS_PER_FRAME,
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
  FLAPPY_MAX_FRAMES_PER_EPISODE,
} from '../constants/constants';
import {
  FLAPPY_EVALUATION_DEFAULT_DIFFICULTY_SCALE,
  FLAPPY_EVALUATION_DEFAULT_EARLY_TERMINATION_CONSECUTIVE_FRAMES,
  FLAPPY_EVALUATION_DEFAULT_EARLY_TERMINATION_GRACE_FRAMES,
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
} from './evaluation.constants';
import {
  createInitialFlappyState,
  getFlappyObservationFeatures,
  stepFlappyStateWithControlSubsteps,
  type FlappyGameState,
  type FlappyObservationFeatures,
} from '../flappyEnvironment.ts';
import {
  clampValue,
  commitSharedObservationMemoryStep,
  createSharedObservationMemoryState,
  resolveFlapDecision,
  resolveTemporalObservationVector,
} from '../flappy.simulation.shared.utils';
import { createXorshift32 } from '../rng';
import { mixGenomeEvaluationSeed } from './evaluation.seed.utils';
import type {
  FlappyEpisodeResult,
  FlappyNetworkLike,
  FlappyRolloutOptions,
} from './evaluation.types';

/**
 * Roll out an episode and return details.
 *
 * @param network - Genome/network to evaluate.
 * @param rolloutOptions - Optional rollout controls.
 * @returns Episode result details.
 */
export function rolloutEpisode(
  network: FlappyNetworkLike,
  rolloutOptions: FlappyRolloutOptions = {},
): FlappyEpisodeResult {
  const genomeId = network._id ?? 0;
  const seed = rolloutOptions.seed ?? mixGenomeEvaluationSeed(genomeId);
  const difficultyScale = clampValue(
    rolloutOptions.difficultyScale ??
      FLAPPY_EVALUATION_DEFAULT_DIFFICULTY_SCALE,
    0,
    1,
  );
  const maxFramesPerEpisode = Math.max(
    1,
    Math.min(
      FLAPPY_MAX_FRAMES_PER_EPISODE,
      Math.trunc(rolloutOptions.maxFrames ?? FLAPPY_MAX_FRAMES_PER_EPISODE),
    ),
  );
  const earlyTerminationGraceFrames = Math.max(
    0,
    Math.trunc(
      rolloutOptions.earlyTerminationGraceFrames ??
        FLAPPY_EVALUATION_DEFAULT_EARLY_TERMINATION_GRACE_FRAMES,
    ),
  );
  const earlyTerminationConsecutiveFrames = Math.max(
    1,
    Math.trunc(
      rolloutOptions.earlyTerminationConsecutiveFrames ??
        FLAPPY_EVALUATION_DEFAULT_EARLY_TERMINATION_CONSECUTIVE_FRAMES,
    ),
  );
  const rng = createXorshift32(seed);

  const state = createInitialFlappyState(rng);
  const observationMemoryState = createSharedObservationMemoryState();
  let denseShapingFitness = 0;
  let unrecoverableFrameCount = 0;

  while (!state.done && state.frameIndex < maxFramesPerEpisode) {
    const previousObservationFeatures = getFlappyObservationFeatures(
      state,
      difficultyScale,
    );
    stepFlappyStateWithControlSubsteps(
      state,
      rng,
      () => {
        const observationFeatures = getFlappyObservationFeatures(
          state,
          difficultyScale,
        );
        const observation = resolveTemporalObservationVector(
          observationFeatures,
          observationMemoryState,
        );
        const outputs = network.activate(observation);
        const shouldFlap = resolveFlapDecision(outputs);
        commitSharedObservationMemoryStep(
          observationMemoryState,
          observationFeatures,
          shouldFlap,
        );
        return shouldFlap;
      },
      difficultyScale,
      FLAPPY_CONTROL_SUBSTEPS_PER_FRAME,
    );

    const currentObservationFeatures = getFlappyObservationFeatures(
      state,
      difficultyScale,
    );
    denseShapingFitness += computeDenseShapingReward(
      previousObservationFeatures,
      currentObservationFeatures,
    );

    if (rolloutOptions.enableEarlyTermination === true) {
      const birdLikelyUnrecoverable = isBirdLikelyUnrecoverable(
        currentObservationFeatures,
      );
      const earlyTerminationEligible =
        state.pipesPassed === 0 &&
        state.frameIndex >= earlyTerminationGraceFrames;

      unrecoverableFrameCount =
        earlyTerminationEligible && birdLikelyUnrecoverable
          ? unrecoverableFrameCount + 1
          : 0;

      if (unrecoverableFrameCount >= earlyTerminationConsecutiveFrames) {
        state.done = true;
        state.doneReason = 'collision';
      }
    }
  }

  if (!state.done && state.frameIndex >= maxFramesPerEpisode) {
    state.done = true;
    state.doneReason = 'timeout';
  }

  const framesSurvived = state.frameIndex;
  const pipesPassed = state.pipesPassed;
  const survivalFitness = framesSurvived * FLAPPY_FITNESS_SURVIVAL_WEIGHT;
  const pipePassFitness = pipesPassed * FLAPPY_FITNESS_BONUS_PER_PIPE;
  const terminalShapingFitness = computeTerminalShapingFitness(
    state,
    difficultyScale,
  );
  const fitness =
    rolloutOptions.normalizeFitness === true
      ? composeNormalizedFitness(
          framesSurvived,
          pipesPassed,
          denseShapingFitness,
          terminalShapingFitness,
          maxFramesPerEpisode,
          rolloutOptions.pipeProgressTarget,
        )
      : survivalFitness +
        pipePassFitness +
        denseShapingFitness +
        terminalShapingFitness;

  return {
    framesSurvived,
    pipesPassed,
    done: state.done,
    doneReason: state.doneReason,
    fitness,
    fitnessBreakdown: {
      survival: survivalFitness,
      pipeProgress: pipePassFitness,
      denseShaping: denseShapingFitness,
      terminalShaping: terminalShapingFitness,
    },
  };
}

/**
 * Computes dense reward shaping from consecutive observations.
 *
 * @param previousFeatures - Observation before stepping the environment.
 * @param currentFeatures - Observation after stepping the environment.
 * @returns Per-step shaped reward.
 */
function computeDenseShapingReward(
  previousFeatures: FlappyObservationFeatures,
  currentFeatures: FlappyObservationFeatures,
): number {
  const nextGapAlignment =
    1 - Math.abs(currentFeatures.normalizedDeltaToNextGap);
  const nextGapAlignmentReward =
    Math.max(0, nextGapAlignment) * FLAPPY_FITNESS_ALIGNMENT_WEIGHT_PER_FRAME;

  const distanceImprovement = Math.max(
    0,
    previousFeatures.normalizedDistanceToNextPipe -
      currentFeatures.normalizedDistanceToNextPipe,
  );
  const approachProgressReward =
    distanceImprovement * FLAPPY_FITNESS_APPROACH_PROGRESS_WEIGHT;

  const previousAbsoluteGapError = Math.abs(
    previousFeatures.normalizedDeltaToNextGap,
  );
  const currentAbsoluteGapError = Math.abs(
    currentFeatures.normalizedDeltaToNextGap,
  );
  const centeringImprovement = Math.max(
    0,
    previousAbsoluteGapError - currentAbsoluteGapError,
  );
  const centeringProgressReward =
    centeringImprovement * FLAPPY_FITNESS_CENTERING_PROGRESS_WEIGHT;

  const positiveClearance = Math.max(
    0,
    currentFeatures.normalizedNextGapClearance,
  );
  const clearanceReward =
    positiveClearance * FLAPPY_FITNESS_CLEARANCE_WEIGHT_PER_FRAME;

  const secondGapAlignment =
    1 - Math.abs(currentFeatures.normalizedDeltaToSecondGap);
  const secondGapAlignmentReward =
    Math.max(0, secondGapAlignment) *
    FLAPPY_FITNESS_SECOND_GAP_ALIGNMENT_WEIGHT_PER_FRAME;

  const velocityStability = 1 - Math.abs(currentFeatures.normalizedVelocity);
  const velocityStabilityReward =
    Math.max(0, velocityStability) *
    FLAPPY_FITNESS_STABLE_VELOCITY_WEIGHT_PER_FRAME;

  return (
    nextGapAlignmentReward +
    approachProgressReward +
    centeringProgressReward +
    clearanceReward +
    secondGapAlignmentReward +
    velocityStabilityReward
  );
}

/**
 * Adds small terminal bonuses from final progress/alignment signals.
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

/**
 * Detects trajectories that are usually irrecoverable in early warmup.
 */
function isBirdLikelyUnrecoverable(
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
