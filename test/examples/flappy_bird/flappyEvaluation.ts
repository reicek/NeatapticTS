import {
  FLAPPY_CONTROL_SUBSTEPS_PER_FRAME,
  FLAPPY_FITNESS_ALIGNMENT_WEIGHT_PER_FRAME,
  FLAPPY_FITNESS_APPROACH_PROGRESS_WEIGHT,
  FLAPPY_FITNESS_BONUS_PER_PIPE,
  FLAPPY_FITNESS_CENTERING_PROGRESS_WEIGHT,
  FLAPPY_FITNESS_CLEARANCE_WEIGHT_PER_FRAME,
  FLAPPY_FITNESS_SURVIVAL_WEIGHT,
  FLAPPY_MAX_FRAMES_PER_EPISODE,
  FLAPPY_FITNESS_SECOND_GAP_ALIGNMENT_WEIGHT_PER_FRAME,
  FLAPPY_FITNESS_STABLE_VELOCITY_WEIGHT_PER_FRAME,
  FLAPPY_FITNESS_TERMINAL_ALIGNMENT_BONUS_WEIGHT,
  FLAPPY_FITNESS_TERMINAL_PROGRESS_BONUS_WEIGHT,
} from './constants.ts';
import {
  createInitialFlappyState,
  getFlappyObservation,
  getFlappyObservationFeatures,
  type FlappyObservationFeatures,
  stepFlappyStateWithControlSubsteps,
  type FlappyGameState,
} from './flappyEnvironment.ts';
import {
  clampValue,
  resolveFlapDecision,
} from './flappy.simulation.shared.utils.ts';
import { createXorshift32 } from './rng.ts';

/** Minimal network contract required by Flappy evaluation. */
export interface FlappyNetworkLike {
  activate(inputs: number[]): number[] | number;
  _id?: number;
}

/**
 * Runtime controls for one rollout evaluation.
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

/** Summary metrics for a single Flappy episode rollout. */
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

/**
 * Evaluate a network on a single deterministic Flappy Bird episode.
 *
 * This is intentionally simple: one episode per genome, seeded from the genome id.
 * That keeps evaluation cheap and makes the demo a good stress-test of evolution
 * operators (mutation/crossover/speciation) rather than a benchmark.
 *
 * @param network - Genome/network to evaluate.
 * @param rolloutOptions - Optional rollout controls.
 * @returns Fitness score (higher is better).
 */
export function evaluateFlappyFitness(
  network: FlappyNetworkLike,
  rolloutOptions: FlappyRolloutOptions = {},
): number {
  return rolloutEpisode(network, rolloutOptions).fitness;
}

/**
 * Evaluate a network on a shared batch of deterministic seeds.
 *
 * This is used by the trainer to reduce per-genome luck and produce
 * stable generation-to-generation rankings.
 *
 * @param network - Genome/network to evaluate.
 * @param sharedSeeds - Shared deterministic seeds used for all genomes.
 * @param rolloutOptions - Optional rollout controls.
 * @returns Robust aggregate metrics for selection/ranking.
 */
export function evaluateFlappyFitnessAcrossSeeds(
  network: FlappyNetworkLike,
  sharedSeeds: readonly number[],
  rolloutOptions: FlappyRolloutOptions = {},
): FlappySeedBatchEvaluation {
  const episodeResults = sharedSeeds.map((seedValue) =>
    rolloutEpisode(network, {
      ...rolloutOptions,
      seed: seedValue,
    }),
  );
  const rolloutFitnessValues = episodeResults.map(
    (rolloutResult) => rolloutResult.fitness,
  );

  const fitnessMean = computeMean(rolloutFitnessValues);
  const fitnessStdDev = computePopulationStandardDeviation(
    rolloutFitnessValues,
    fitnessMean,
  );

  const meanPipesPassed = computeMean(
    episodeResults.map((episodeResult) => episodeResult.pipesPassed),
  );
  const meanFramesSurvived = computeMean(
    episodeResults.map((episodeResult) => episodeResult.framesSurvived),
  );

  return {
    seedCount: sharedSeeds.length,
    meanFitness: fitnessMean,
    medianFitness: computePercentile(rolloutFitnessValues, 0.5),
    p90Fitness: computePercentile(rolloutFitnessValues, 0.9),
    fitnessStdDev,
    robustFitness: fitnessMean - fitnessStdDev * 0.35,
    meanPipesPassed,
    meanFramesSurvived,
  };
}

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
  const seed = rolloutOptions.seed ?? mixSeed(genomeId);
  const difficultyScale = clampValue(rolloutOptions.difficultyScale ?? 1, 0, 1);
  const maxFramesPerEpisode = Math.max(
    1,
    Math.min(
      FLAPPY_MAX_FRAMES_PER_EPISODE,
      Math.trunc(rolloutOptions.maxFrames ?? FLAPPY_MAX_FRAMES_PER_EPISODE),
    ),
  );
  const earlyTerminationGraceFrames = Math.max(
    0,
    Math.trunc(rolloutOptions.earlyTerminationGraceFrames ?? 160),
  );
  const earlyTerminationConsecutiveFrames = Math.max(
    1,
    Math.trunc(rolloutOptions.earlyTerminationConsecutiveFrames ?? 24),
  );
  const rng = createXorshift32(seed);

  const state = createInitialFlappyState(rng);
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
        const observation = getFlappyObservation(state, difficultyScale);
        const outputs = network.activate(observation);
        return resolveFlapDecision(outputs);
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
  const terminalShapingFitness = computeTerminalShapingFitness(state);
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

  /**
   * Computes dense reward shaping from consecutive observations.
   *
   * This gives the policy a gradient before it can reliably pass pipes.
   *
   * @param previousFeatures - Observation before stepping the environment.
   * @param currentFeatures - Observation after stepping the environment.
   * @returns Per-step shaped reward.
   */
  function computeDenseShapingReward(
    previousFeatures: FlappyObservationFeatures,
    currentFeatures: FlappyObservationFeatures,
  ): number {
    // Step 1: Reward direct alignment with the active gap.
    const nextGapAlignment =
      1 - Math.abs(currentFeatures.normalizedDeltaToNextGap);
    const nextGapAlignmentReward =
      Math.max(0, nextGapAlignment) * FLAPPY_FITNESS_ALIGNMENT_WEIGHT_PER_FRAME;

    // Step 2: Reward moving closer horizontally to the next pipe.
    const distanceImprovement = Math.max(
      0,
      previousFeatures.normalizedDistanceToNextPipe -
        currentFeatures.normalizedDistanceToNextPipe,
    );
    const approachProgressReward =
      distanceImprovement * FLAPPY_FITNESS_APPROACH_PROGRESS_WEIGHT;

    // Step 3: Reward reducing vertical offset to the active gap center.
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

    // Step 4: Reward staying inside the corridor with positive clearance.
    const positiveClearance = Math.max(
      0,
      currentFeatures.normalizedNextGapClearance,
    );
    const clearanceReward =
      positiveClearance * FLAPPY_FITNESS_CLEARANCE_WEIGHT_PER_FRAME;

    // Step 5: Add look-ahead pressure toward the second upcoming gap.
    const secondGapAlignment =
      1 - Math.abs(currentFeatures.normalizedDeltaToSecondGap);
    const secondGapAlignmentReward =
      Math.max(0, secondGapAlignment) *
      FLAPPY_FITNESS_SECOND_GAP_ALIGNMENT_WEIGHT_PER_FRAME;

    // Step 6: Reward stable, controllable vertical speed.
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
   * @returns Terminal shaping reward.
   */
  function computeTerminalShapingFitness(
    episodeState: FlappyGameState,
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
      Math.max(0, finalProgress) *
      FLAPPY_FITNESS_TERMINAL_PROGRESS_BONUS_WEIGHT;

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
      Math.trunc(pipeProgressTarget ?? 20),
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
      denseShapingValue / Math.max(1, framesValue * 4.5),
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
      normalizedSurvival * 2_600 * FLAPPY_FITNESS_SURVIVAL_WEIGHT +
      normalizedPipeProgress * 1_400 +
      normalizedDenseShaping * 1_200 +
      normalizedTerminalShaping * 600
    );
  }

  /**
   * Detects trajectories that are usually irrecoverable in early warmup.
   */
  function isBirdLikelyUnrecoverable(
    observationFeatures: FlappyObservationFeatures,
  ): boolean {
    const farOutsideGap = observationFeatures.normalizedNextGapClearance < -0.7;
    const belowGapAndStillFalling =
      observationFeatures.normalizedDeltaToNextGap > 0.45 &&
      observationFeatures.normalizedVelocity > 0.5;
    const aboveGapAndStillRising =
      observationFeatures.normalizedDeltaToNextGap < -0.45 &&
      observationFeatures.normalizedVelocity < -0.5;

    return farOutsideGap && (belowGapAndStillFalling || aboveGapAndStillRising);
  }
}

/**
 * Mix a small integer seed into a reasonable uint32 RNG seed.
 *
 * @param genomeId - Genome id from NEAT bookkeeping.
 * @returns uint32 seed.
 */
function mixSeed(genomeId: number): number {
  // A tiny avalanching mix (not crypto; just to spread ids).
  let seed = (genomeId >>> 0) ^ 0x9e3779b9;
  seed ^= seed >>> 16;
  seed = Math.imul(seed, 0x85ebca6b);
  seed ^= seed >>> 13;
  seed = Math.imul(seed, 0xc2b2ae35);
  seed ^= seed >>> 16;
  return seed >>> 0;
}

/**
 * @param values - Numeric samples.
 * @returns Arithmetic mean.
 */
function computeMean(values: readonly number[]): number {
  if (values.length === 0) return 0;
  return (
    values.reduce((accumulator, value) => accumulator + value, 0) /
    values.length
  );
}

/**
 * @param values - Numeric samples.
 * @param meanValue - Precomputed mean.
 * @returns Population standard deviation.
 */
function computePopulationStandardDeviation(
  values: readonly number[],
  meanValue: number,
): number {
  if (values.length === 0) return 0;
  const variance =
    values.reduce((accumulator, value) => {
      const delta = value - meanValue;
      return accumulator + delta * delta;
    }, 0) / values.length;
  return Math.sqrt(Math.max(0, variance));
}

/**
 * @param values - Numeric samples.
 * @param percentile - Percentile in [0, 1].
 * @returns Percentile value via nearest-rank interpolation.
 */
function computePercentile(
  values: readonly number[],
  percentile: number,
): number {
  if (values.length === 0) return 0;
  const sortedValues = values.toSorted(
    (leftValue, rightValue) => leftValue - rightValue,
  );
  const clampedPercentile = clampValue(percentile, 0, 1);
  const rawIndex = clampedPercentile * (sortedValues.length - 1);
  const lowerIndex = Math.floor(rawIndex);
  const upperIndex = Math.ceil(rawIndex);
  const interpolation = rawIndex - lowerIndex;

  const lowerValue = sortedValues[lowerIndex] ?? sortedValues[0] ?? 0;
  const upperValue = sortedValues[upperIndex] ?? lowerValue;
  return lowerValue + (upperValue - lowerValue) * interpolation;
}

/**
 * @param value - Scalar input.
 * @param min - Lower bound.
 * @param max - Upper bound.
 * @returns Clamped scalar.
 */
function clampValue(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}
