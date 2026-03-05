import {
  FLAPPY_BIRD_RADIUS_PX,
  FLAPPY_BIRD_X_PX,
  FLAPPY_FLAP_VELOCITY_PX_PER_FRAME,
  FLAPPY_GRAVITY_PX_PER_FRAME2,
  FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
  FLAPPY_MIN_PIPE_RECOVERY_FRAMES,
  FLAPPY_PIPE_GAP_PX,
  FLAPPY_PIPE_WIDTH_PX,
  FLAPPY_WORLD_HEIGHT_PX,
} from '../constants/constants';
import { FLAPPY_SHARED_DEFAULT_NORMALIZATION_EPSILON } from './simulation-shared.constants';
import type {
  SharedObservationFeatures,
  SharedObservationInput,
  SharedPipeLike,
} from './simulation-shared.types';

/**
 * Resolves the next two upcoming pipes in front of the bird.
 *
 * @param pipes - Current pipe list.
 * @param birdCenterXPx - Bird center x-position.
 * @param birdRadiusPx - Bird radius.
 * @param pipeWidthPx - Pipe width.
 * @returns Tuple of first and second upcoming pipes.
 */
export function resolveUpcomingPipes(
  pipes: SharedPipeLike[],
  birdCenterXPx = FLAPPY_BIRD_X_PX,
  birdRadiusPx = FLAPPY_BIRD_RADIUS_PX,
  pipeWidthPx = FLAPPY_PIPE_WIDTH_PX,
): [SharedPipeLike | undefined, SharedPipeLike | undefined] {
  const birdFrontX = birdCenterXPx - birdRadiusPx;
  const upcomingPipes = pipes.filter(
    (pipe) => pipe.xPx + pipeWidthPx >= birdFrontX,
  );
  return [upcomingPipes[0], upcomingPipes[1]];
}

/**
 * Builds the shared normalized observation feature set consumed by policies.
 *
 * @param input - Observation input bundle.
 * @returns Structured observation features.
 */
export function resolveObservationFeatures(
  input: SharedObservationInput,
): SharedObservationFeatures {
  /**
   * Effective world height used for all vertical normalization.
   *
   * Why it matters: scaling by a stable height keeps feature magnitudes
   * comparable across episodes and prevents value drift when display sizes vary.
   */
  const worldHeightPx = input.worldHeightPx ?? FLAPPY_WORLD_HEIGHT_PX;
  /**
   * Maximum downward speed used to normalize velocity-like channels.
   *
   * Why it matters: velocity channels become dimensionless and bounded in [-1, 1],
   * making policy optimization numerically easier.
   */
  const maxFallSpeedPxPerFrame =
    input.maxFallSpeedPxPerFrame ?? FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME;
  /** Bird center x-position used to measure forward distances to pipes. */
  const birdCenterXPx = input.birdCenterXPx ?? FLAPPY_BIRD_X_PX;
  /** Bird collision radius used by upcoming-pipe selection and entry/exit timing. */
  const birdRadiusPx = input.birdRadiusPx ?? FLAPPY_BIRD_RADIUS_PX;
  /** Pipe width used for distance-to-entry/exit and visibility checks. */
  const pipeWidthPx = input.pipeWidthPx ?? FLAPPY_PIPE_WIDTH_PX;
  /** Default gap size fallback when no upcoming pipe is available yet. */
  const defaultGapSizePx = input.defaultGapSizePx ?? FLAPPY_PIPE_GAP_PX;
  /**
   * Small denominator guard used in time/urgency calculations.
   *
   * Why it matters: avoids division by zero and explosive values when
   * speed or time-to-event become extremely small.
   */
  const normalizationEpsilon =
    input.normalizationEpsilon ?? FLAPPY_SHARED_DEFAULT_NORMALIZATION_EPSILON;

  /** First and second upcoming pipes used for near-term and lookahead guidance. */
  const [nextPipe, secondPipe] = resolveUpcomingPipes(
    input.pipes,
    birdCenterXPx,
    birdRadiusPx,
    pipeWidthPx,
  );
  /** Bird y-position normalized to [0, 1]. */
  const normalizedBirdY = clamp01(input.birdYPx / worldHeightPx);
  /** Bird vertical velocity normalized to [-1, 1]. */
  const normalizedVelocity = clamp(
    input.velocityYPxPerFrame / maxFallSpeedPxPerFrame,
    -1,
    1,
  );

  /** Forward distance from bird to the next pipe corridor (pixels). */
  const distanceToNextPipePx = nextPipe
    ? nextPipe.xPx + pipeWidthPx - birdCenterXPx
    : input.visibleWorldWidthPx;
  /** Next-pipe forward distance normalized to [0, 1]. */
  const normalizedDistanceToNextPipe = clamp01(
    distanceToNextPipePx / input.visibleWorldWidthPx,
  );

  /** Next gap center y-position (fallback to mid-world when absent). */
  const nextGapCenterYPx = nextPipe?.gapCenterYPx ?? worldHeightPx * 0.5;
  /** Half-gap height for clearance and boundary calculations. */
  const nextGapHalfPx = (nextPipe?.gapSizePx ?? defaultGapSizePx) * 0.5;
  /** Normalized top boundary of the next gap opening. */
  const normalizedNextGapTop = clamp01(
    (nextGapCenterYPx - nextGapHalfPx) / worldHeightPx,
  );
  /** Normalized bottom boundary of the next gap opening. */
  const normalizedNextGapBottom = clamp01(
    (nextGapCenterYPx + nextGapHalfPx) / worldHeightPx,
  );
  /** Signed bird-to-next-gap-center vertical offset normalized to [-1, 1]. */
  const normalizedDeltaToNextGap = clamp(
    (input.birdYPx - nextGapCenterYPx) / worldHeightPx,
    -1,
    1,
  );

  /** Forward distance from bird to the second upcoming pipe (pixels). */
  const distanceToSecondPipePx = secondPipe
    ? secondPipe.xPx + pipeWidthPx - birdCenterXPx
    : input.visibleWorldWidthPx;
  /** Second-pipe forward distance normalized to [0, 1]. */
  const normalizedDistanceToSecondPipe = clamp01(
    distanceToSecondPipePx / input.visibleWorldWidthPx,
  );

  /** Second gap center y-position (fallback to next center when absent). */
  const secondGapCenterYPx = secondPipe?.gapCenterYPx ?? nextGapCenterYPx;
  /** Signed bird-to-second-gap-center vertical offset normalized to [-1, 1]. */
  const normalizedDeltaToSecondGap = clamp(
    (input.birdYPx - secondGapCenterYPx) / worldHeightPx,
    -1,
    1,
  );

  /** Estimated frames until reaching the next pipe (continuous-time approximation). */
  const estimatedFramesToNextPipe =
    distanceToNextPipePx /
    Math.max(normalizationEpsilon, input.difficultyProfile.pipeSpeedPxPerFrame);
  /** Urgency-like proximity to next pipe in [0, 1], where 1 means imminent. */
  const normalizedTimeToNextPipe = clamp01(
    1 -
      estimatedFramesToNextPipe / Math.max(1, input.activeSpawnIntervalFrames),
  );

  /** Absolute vertical distance from bird to next gap center (pixels). */
  const distanceToNextGapCenterPx = Math.abs(input.birdYPx - nextGapCenterYPx);
  /** Signed clearance relative to half-gap height, normalized to [-1, 1]. */
  const normalizedNextGapClearance = clamp(
    (nextGapHalfPx - distanceToNextGapCenterPx) / Math.max(1, nextGapHalfPx),
    -1,
    1,
  );

  /** Horizontal distance to the gap entry plane (pixels). */
  const distanceToGapEntryPx = nextPipe
    ? nextPipe.xPx - (birdCenterXPx + birdRadiusPx)
    : input.visibleWorldWidthPx;
  /** Horizontal distance to the gap exit plane (pixels). */
  const distanceToGapExitPx = nextPipe
    ? nextPipe.xPx + pipeWidthPx - (birdCenterXPx - birdRadiusPx)
    : input.visibleWorldWidthPx;

  /** Frames until gap entry; clamped to non-negative values. */
  const framesToGapEntry = Math.max(
    0,
    distanceToGapEntryPx /
      Math.max(
        normalizationEpsilon,
        input.difficultyProfile.pipeSpeedPxPerFrame,
      ),
  );
  /**
   * Frames until gap exit; never smaller than entry time.
   *
   * Why it matters: captures tube traversal duration, which is crucial for
   * wide-pipe timing and potential two-flap planning.
   */
  const framesToGapExit = Math.max(
    framesToGapEntry,
    distanceToGapExitPx /
      Math.max(
        normalizationEpsilon,
        input.difficultyProfile.pipeSpeedPxPerFrame,
      ),
  );

  /** Frames-to-entry normalized to [0, 1] using spawn-interval scale. */
  const normalizedFramesToGapEntry = clamp01(
    framesToGapEntry / Math.max(1, input.activeSpawnIntervalFrames),
  );
  /** Frames-to-exit normalized to [0, 1] using spawn-interval scale. */
  const normalizedFramesToGapExit = clamp01(
    framesToGapExit / Math.max(1, input.activeSpawnIntervalFrames),
  );

  /** Required mean vertical velocity to center at next-pipe contact window. */
  const requiredVerticalVelocityToNextGapPxPerFrame = clamp(
    (nextGapCenterYPx - input.birdYPx) / Math.max(1, estimatedFramesToNextPipe),
    -maxFallSpeedPxPerFrame,
    maxFallSpeedPxPerFrame,
  );
  /** Required velocity-to-next-gap channel normalized to [-1, 1]. */
  const normalizedRequiredVerticalVelocityToNextGap = clamp(
    requiredVerticalVelocityToNextGapPxPerFrame / maxFallSpeedPxPerFrame,
    -1,
    1,
  );

  /** Required mean vertical velocity to center precisely at gap entry. */
  const requiredVerticalVelocityAtGapEntryPxPerFrame = clamp(
    (nextGapCenterYPx - input.birdYPx) / Math.max(1, framesToGapEntry),
    -maxFallSpeedPxPerFrame,
    maxFallSpeedPxPerFrame,
  );
  /** Required mean vertical velocity to remain centered by gap exit. */
  const requiredVerticalVelocityAtGapExitPxPerFrame = clamp(
    (nextGapCenterYPx - input.birdYPx) / Math.max(1, framesToGapExit),
    -maxFallSpeedPxPerFrame,
    maxFallSpeedPxPerFrame,
  );
  /** Entry-target velocity normalized to [-1, 1]. */
  const normalizedRequiredVerticalVelocityAtGapEntry = clamp(
    requiredVerticalVelocityAtGapEntryPxPerFrame / maxFallSpeedPxPerFrame,
    -1,
    1,
  );
  /** Exit-target velocity normalized to [-1, 1]. */
  const normalizedRequiredVerticalVelocityAtGapExit = clamp(
    requiredVerticalVelocityAtGapExitPxPerFrame / maxFallSpeedPxPerFrame,
    -1,
    1,
  );

  /**
   * Centering urgency in [0, 1]: large offset with little time-to-entry yields
   * stronger urgency.
   */
  const normalizedEntryUrgency = clamp(
    Math.abs(normalizedDeltaToNextGap) /
      Math.max(normalizationEpsilon, normalizedFramesToGapEntry),
    0,
    1,
  );

  /** Predicted entry y-position if no flap occurs before entry. */
  const predictedBirdYAtEntryWithoutFlap = resolvePredictedBirdYAtFrames(
    input.birdYPx,
    input.velocityYPxPerFrame,
    framesToGapEntry,
  );
  /** Predicted entry y-position if one immediate flap occurs before entry. */
  const predictedBirdYAtEntryWithOneFlap = resolvePredictedBirdYAtFrames(
    input.birdYPx,
    FLAPPY_FLAP_VELOCITY_PX_PER_FRAME,
    framesToGapEntry,
  );
  /** Entry centering error (pixels) under no-flap trajectory. */
  const noFlapEntryErrorPx = Math.abs(
    predictedBirdYAtEntryWithoutFlap - nextGapCenterYPx,
  );
  /** Entry centering error (pixels) under one-flap trajectory. */
  const oneFlapEntryErrorPx = Math.abs(
    predictedBirdYAtEntryWithOneFlap - nextGapCenterYPx,
  );
  /** Best (minimum) entry error among one-step action hypotheses. */
  const minimalEntryErrorPx = Math.min(noFlapEntryErrorPx, oneFlapEntryErrorPx);
  /** Binary reachability feature: 1 when entry center is recoverable within <=1 flap. */
  const normalizedOneFlapReachabilityAtGapEntry =
    minimalEntryErrorPx <= nextGapHalfPx ? 1 : 0;

  /** Signed centerline transition from next gap to second gap, normalized to [-1, 1]. */
  const normalizedNextToSecondGapTransition = clamp(
    (secondGapCenterYPx - nextGapCenterYPx) / worldHeightPx,
    -1,
    1,
  );

  return {
    normalizedBirdY,
    normalizedVelocity,
    normalizedDistanceToNextPipe,
    normalizedDeltaToNextGap,
    normalizedNextGapTop,
    normalizedNextGapBottom,
    normalizedDistanceToSecondPipe,
    normalizedDeltaToSecondGap,
    normalizedTimeToNextPipe,
    normalizedNextGapClearance,
    normalizedRequiredVerticalVelocityToNextGap,
    normalizedNextToSecondGapTransition,
    normalizedFramesToGapEntry,
    normalizedFramesToGapExit,
    normalizedRequiredVerticalVelocityAtGapEntry,
    normalizedRequiredVerticalVelocityAtGapExit,
    normalizedEntryUrgency,
    normalizedOneFlapReachabilityAtGapEntry,
  };
}

/**
 * Converts observation features to the canonical 12-value network input vector.
 *
 * @param features - Structured feature object.
 * @returns Ordered feature vector.
 */
export function resolveObservationVectorFromFeatures(
  features: SharedObservationFeatures,
): number[] {
  return [
    features.normalizedBirdY,
    features.normalizedVelocity,
    features.normalizedDistanceToNextPipe,
    features.normalizedDeltaToNextGap,
    features.normalizedNextGapTop,
    features.normalizedNextGapBottom,
    features.normalizedDistanceToSecondPipe,
    features.normalizedDeltaToSecondGap,
    features.normalizedTimeToNextPipe,
    features.normalizedNextGapClearance,
    features.normalizedRequiredVerticalVelocityToNextGap,
    features.normalizedNextToSecondGapTransition,
  ];
}

/**
 * Resolves the compact core vector used for temporal stacking.
 *
 * The core intentionally keeps directly observed kinematic/geometric channels
 * and drops derived one-step predictors that become redundant once temporal
 * context is available.
 *
 * @param features - Structured observation features.
 * @returns Core per-frame vector.
 */
export function resolveCoreObservationVectorFromFeatures(
  features: SharedObservationFeatures,
): number[] {
  return [
    features.normalizedBirdY,
    features.normalizedVelocity,
    features.normalizedDistanceToNextPipe,
    features.normalizedDeltaToNextGap,
    features.normalizedNextGapTop,
    features.normalizedNextGapBottom,
    features.normalizedDistanceToSecondPipe,
    features.normalizedDeltaToSecondGap,
    features.normalizedNextGapClearance,
    features.normalizedRequiredVerticalVelocityToNextGap,
    features.normalizedEntryUrgency,
    features.normalizedOneFlapReachabilityAtGapEntry,
  ];
}

/**
 * Predicts bird y-position after a frame horizon with constant gravity.
 *
 * This lightweight estimate is used for reachability signals only.
 *
 * @param startYPx - Current bird y-position.
 * @param initialVerticalVelocityPxPerFrame - Initial vertical velocity.
 * @param frameHorizon - Predicted horizon in simulation frames.
 * @returns Predicted y-position.
 */
function resolvePredictedBirdYAtFrames(
  startYPx: number,
  initialVerticalVelocityPxPerFrame: number,
  frameHorizon: number,
): number {
  const safeFrameHorizon = Math.max(0, frameHorizon);
  return (
    startYPx +
    initialVerticalVelocityPxPerFrame * safeFrameHorizon +
    0.5 * FLAPPY_GRAVITY_PX_PER_FRAME2 * safeFrameHorizon * safeFrameHorizon
  );
}

/**
 * Clamps a numeric value to the inclusive `[min, max]` interval.
 *
 * @param value - Candidate value.
 * @param min - Inclusive lower bound.
 * @param max - Inclusive upper bound.
 * @returns Clamped value.
 */
function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/**
 * Clamps a numeric value to the inclusive `[0, 1]` interval.
 *
 * @param value - Candidate value.
 * @returns Value clamped between 0 and 1.
 */
function clamp01(value: number): number {
  return clamp(value, 0, 1);
}
