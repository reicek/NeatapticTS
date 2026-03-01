import {
  FLAPPY_BIRD_RADIUS_PX,
  FLAPPY_BIRD_X_PX,
  FLAPPY_DIFFICULTY_RAMP_PIPES,
  FLAPPY_FLAP_VELOCITY_PX_PER_FRAME,
  FLAPPY_GRAVITY_PX_PER_FRAME2,
  FLAPPY_MEMORY_ACTION_WINDOW_STEPS,
  FLAPPY_MEMORY_CORE_FEATURE_COUNT,
  FLAPPY_MEMORY_STACKED_FRAME_COUNT,
  FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
  FLAPPY_MIN_PIPE_RECOVERY_FRAMES,
  FLAPPY_PIPE_GAP_CENTER_MAX_DELTA_PX,
  FLAPPY_PIPE_GAP_CENTER_MAX_Y_PX,
  FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX,
  FLAPPY_PIPE_GAP_MIN_PX,
  FLAPPY_PIPE_GAP_PX,
  FLAPPY_PIPE_GAP_RANDOM_JITTER_PX,
  FLAPPY_PIPE_GAP_SHRINK_PER_PIPE_PX,
  FLAPPY_PIPE_GAP_START_MULTIPLIER,
  FLAPPY_PIPE_SPAWN_INTERVAL_FRAMES,
  FLAPPY_PIPE_SPAWN_INTERVAL_MIN_FRAMES,
  FLAPPY_PIPE_SPAWN_INTERVAL_SHRINK_PER_PIPE_FRAMES,
  FLAPPY_PIPE_SPAWN_INTERVAL_START_MULTIPLIER,
  FLAPPY_PIPE_SPEED_MAX_PX_PER_FRAME,
  FLAPPY_PIPE_SPEED_PX_PER_FRAME,
  FLAPPY_PIPE_WIDTH_PX,
  FLAPPY_WORLD_HEIGHT_PX,
  FLAPPY_WORLD_WIDTH_PX,
} from './constants';

/** Minimal deterministic random contract used by shared spawn helpers. */
export interface SharedRngLike {
  /**
   * Returns integer in `[min, max)`.
   *
   * @param min - Inclusive lower bound.
   * @param max - Exclusive upper bound.
   * @returns Pseudo-random integer.
   */
  nextInt(min: number, max: number): number;
}

/** Common pipe shape consumed by observation helpers. */
export interface SharedPipeLike {
  /** Horizontal left position in world pixels. */
  xPx: number;

  /** Gap center y position in world pixels. */
  gapCenterYPx: number;

  /** Gap size in world pixels. */
  gapSizePx: number;
}

/** Shared runtime difficulty profile used by browser and environment simulators. */
export interface SharedDifficultyProfile {
  /** Active target gap size. */
  pipeGapPx: number;

  /** Active pipe horizontal speed. */
  pipeSpeedPxPerFrame: number;

  /** Active spawn interval in simulation frames. */
  pipeSpawnIntervalFrames: number;
}

/** Structured observation features for network input. */
export interface SharedObservationFeatures {
  /** Bird y position normalized to [0, 1]. */
  normalizedBirdY: number;

  /** Bird vertical velocity normalized to [-1, 1]. */
  normalizedVelocity: number;

  /** Distance to next pipe normalized to [0, 1]. */
  normalizedDistanceToNextPipe: number;

  /** Delta to next gap center normalized to [-1, 1]. */
  normalizedDeltaToNextGap: number;

  /** Next gap top normalized to [0, 1]. */
  normalizedNextGapTop: number;

  /** Next gap bottom normalized to [0, 1]. */
  normalizedNextGapBottom: number;

  /** Distance to second pipe normalized to [0, 1]. */
  normalizedDistanceToSecondPipe: number;

  /** Delta to second gap center normalized to [-1, 1]. */
  normalizedDeltaToSecondGap: number;

  /** Time-to-next-pipe closeness normalized to [0, 1]. */
  normalizedTimeToNextPipe: number;

  /** Signed next-gap clearance in [-1, 1]. */
  normalizedNextGapClearance: number;

  /** Required vertical velocity to center next gap normalized to [-1, 1]. */
  normalizedRequiredVerticalVelocityToNextGap: number;

  /** Transition between next and second gap centers normalized to [-1, 1]. */
  normalizedNextToSecondGapTransition: number;

  /** Frames-to-gap-entry estimate normalized to [0, 1]. */
  normalizedFramesToGapEntry: number;

  /** Frames-to-gap-exit estimate normalized to [0, 1]. */
  normalizedFramesToGapExit: number;

  /** Required vertical velocity at gap entry normalized to [-1, 1]. */
  normalizedRequiredVerticalVelocityAtGapEntry: number;

  /** Required vertical velocity at gap exit normalized to [-1, 1]. */
  normalizedRequiredVerticalVelocityAtGapExit: number;

  /** Urgency signal for recovering to center before entry normalized to [0, 1]. */
  normalizedEntryUrgency: number;

  /** Reachability signal indicating whether entry can be recovered in <=1 flap. */
  normalizedOneFlapReachabilityAtGapEntry: number;
}

/**
 * Mutable temporal memory attached to one policy-controlled bird.
 *
 * The memory stores recent core observation frames and recent action history,
 * allowing feedforward policies to consume short-term context without adding
 * recurrent connections.
 */
export interface SharedObservationMemoryState {
  /** Previous core frames, newest-first, excluding the current frame. */
  previousCoreObservationFrames: number[][];

  /** Recent flap actions, newest-first, encoded as `1` (flap) or `0` (no flap). */
  recentFlapActions: number[];
}

/** Input shape for observation-feature synthesis. */
export interface SharedObservationInput {
  /** Bird vertical position in world pixels. */
  birdYPx: number;

  /** Bird vertical velocity in world pixels/frame. */
  velocityYPxPerFrame: number;

  /** Current pipe list. */
  pipes: SharedPipeLike[];

  /** Active visible width used for distance normalization. */
  visibleWorldWidthPx: number;

  /** Active difficulty profile. */
  difficultyProfile: SharedDifficultyProfile;

  /** Active spawn interval. */
  activeSpawnIntervalFrames: number;

  /** Default gap size used when no next pipe exists. */
  defaultGapSizePx?: number;

  /** Bird center x-position used for upcoming-pipe resolution. */
  birdCenterXPx?: number;

  /** Bird collision radius used for upcoming-pipe resolution. */
  birdRadiusPx?: number;

  /** Pipe width used for distance and upcoming-pipe calculations. */
  pipeWidthPx?: number;

  /** World height used for normalization. */
  worldHeightPx?: number;

  /** Maximum fall speed used for velocity normalization. */
  maxFallSpeedPxPerFrame?: number;

  /** Division guard used for time-to-next-pipe estimate. */
  normalizationEpsilon?: number;
}

/**
 * Creates an empty temporal observation memory state.
 *
 * @returns Fresh mutable memory buffers for one bird/controller.
 */
export function createSharedObservationMemoryState(): SharedObservationMemoryState {
  return {
    previousCoreObservationFrames: [],
    recentFlapActions: [],
  };
}

/**
 * Resolves adaptive difficulty profile from passed-pipe progress.
 *
 * @param pipesPassed - Number of passed pipes.
 * @param difficultyScale - Curriculum scale in `[0, 1]`.
 * @returns Active difficulty profile.
 */
export function resolveAdaptiveDifficultyProfile(
  pipesPassed: number,
  difficultyScale = 1,
): SharedDifficultyProfile {
  const normalizedDifficultyScale = clamp(difficultyScale, 0, 1);
  const normalizedDifficultyProgress = clamp(
    (pipesPassed / Math.max(1, FLAPPY_DIFFICULTY_RAMP_PIPES)) *
      normalizedDifficultyScale,
    0,
    1,
  );

  return {
    pipeGapPx: Math.round(
      interpolateValue(
        FLAPPY_PIPE_GAP_PX,
        FLAPPY_PIPE_GAP_MIN_PX,
        normalizedDifficultyProgress,
      ),
    ),
    pipeSpeedPxPerFrame: interpolateValue(
      FLAPPY_PIPE_SPEED_PX_PER_FRAME,
      FLAPPY_PIPE_SPEED_MAX_PX_PER_FRAME,
      normalizedDifficultyProgress,
    ),
    pipeSpawnIntervalFrames: Math.round(
      interpolateValue(
        FLAPPY_PIPE_SPAWN_INTERVAL_FRAMES,
        FLAPPY_PIPE_SPAWN_INTERVAL_MIN_FRAMES,
        normalizedDifficultyProgress,
      ),
    ),
  };
}

/**
 * Samples a random gap center y-position.
 *
 * @param rng - Deterministic RNG.
 * @returns Sampled y-position.
 */
export function sampleGapCenterY(rng: SharedRngLike): number {
  return rng.nextInt(
    FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX,
    FLAPPY_PIPE_GAP_CENTER_MAX_Y_PX,
  );
}

/**
 * Resolves next gap center with bounded per-pipe delta.
 *
 * @param previousGapCenterYPx - Previous spawn gap center.
 * @param rng - Deterministic RNG.
 * @returns Next gap center y-position.
 */
export function resolveNextSpawnGapCenterY(
  previousGapCenterYPx: number,
  rng: SharedRngLike,
): number {
  const sampledGapCenterYPx = sampleGapCenterY(rng);
  const minimumGapCenterYPx = Math.max(
    FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX,
    previousGapCenterYPx - FLAPPY_PIPE_GAP_CENTER_MAX_DELTA_PX,
  );
  const maximumGapCenterYPx = Math.min(
    FLAPPY_PIPE_GAP_CENTER_MAX_Y_PX,
    previousGapCenterYPx + FLAPPY_PIPE_GAP_CENTER_MAX_DELTA_PX,
  );
  return clamp(sampledGapCenterYPx, minimumGapCenterYPx, maximumGapCenterYPx);
}

/**
 * Resolves next spawn gap size using progressive shrink and jitter.
 *
 * @param previousSpawnGapPx - Previous spawn gap size.
 * @param difficultyProfile - Active difficulty profile.
 * @param rng - Deterministic RNG.
 * @returns Next spawn gap size.
 */
export function resolveNextSpawnGapSize(
  previousSpawnGapPx: number | undefined,
  difficultyProfile: SharedDifficultyProfile,
  rng: SharedRngLike,
): number {
  const hardestGapPx = difficultyProfile.pipeGapPx;
  const initialWideGapPx = Math.round(
    hardestGapPx * FLAPPY_PIPE_GAP_START_MULTIPLIER,
  );

  const progressiveGapPx =
    previousSpawnGapPx == null
      ? initialWideGapPx
      : Math.max(
          hardestGapPx,
          previousSpawnGapPx - FLAPPY_PIPE_GAP_SHRINK_PER_PIPE_PX,
        );

  const randomJitterPx = rng.nextInt(
    -FLAPPY_PIPE_GAP_RANDOM_JITTER_PX,
    FLAPPY_PIPE_GAP_RANDOM_JITTER_PX + 1,
  );
  const randomizedGapPx = progressiveGapPx + randomJitterPx;

  return Math.round(clamp(randomizedGapPx, hardestGapPx, initialWideGapPx));
}

/**
 * Resolves next spawn interval using progressive shrink.
 *
 * @param previousSpawnIntervalFrames - Previous spawn interval.
 * @param difficultyProfile - Active difficulty profile.
 * @returns Next spawn interval in frames.
 */
export function resolveNextSpawnIntervalFrames(
  previousSpawnIntervalFrames: number | undefined,
  difficultyProfile: SharedDifficultyProfile,
): number {
  const hardestIntervalFrames = difficultyProfile.pipeSpawnIntervalFrames;
  const initialWideIntervalFrames = Math.round(
    hardestIntervalFrames * FLAPPY_PIPE_SPAWN_INTERVAL_START_MULTIPLIER,
  );

  const progressiveIntervalFrames =
    previousSpawnIntervalFrames == null
      ? initialWideIntervalFrames
      : Math.max(
          hardestIntervalFrames,
          previousSpawnIntervalFrames -
            FLAPPY_PIPE_SPAWN_INTERVAL_SHRINK_PER_PIPE_FRAMES,
        );

  return Math.round(
    clamp(
      progressiveIntervalFrames,
      hardestIntervalFrames,
      initialWideIntervalFrames,
    ),
  );
}

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
  const normalizationEpsilon = input.normalizationEpsilon ?? 0.001;

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
 * Builds the temporal policy input vector (stacked observation + action memory).
 *
 * Output layout:
 * 1) current core observation frame
 * 2) previous core frames (newest to oldest) with zero padding
 * 3) last-action channel
 * 4) recent flap-rate channel over a fixed window
 *
 * @param features - Structured observation features for the current decision step.
 * @param observationMemoryState - Mutable temporal memory for the active bird.
 * @returns Ordered temporal input vector for policy activation.
 */
export function resolveTemporalObservationVector(
  features: SharedObservationFeatures,
  observationMemoryState: SharedObservationMemoryState,
): number[] {
  const currentCoreObservationFrame =
    resolveCoreObservationVectorFromFeatures(features);
  const stackedFrames = [
    currentCoreObservationFrame,
    ...resolvePreviousCoreFramesWithPadding(observationMemoryState),
  ];
  const flattenedStackedFrames = stackedFrames.flat();
  const lastActionChannel = observationMemoryState.recentFlapActions[0] ?? 0;
  const recentFlapRateChannel =
    observationMemoryState.recentFlapActions.length === 0
      ? 0
      : observationMemoryState.recentFlapActions.reduce(
          (actionSum, actionValue) => actionSum + actionValue,
          0,
        ) / observationMemoryState.recentFlapActions.length;

  return [...flattenedStackedFrames, lastActionChannel, recentFlapRateChannel];
}

/**
 * Commits one observation-action step into temporal memory.
 *
 * @param observationMemoryState - Mutable temporal memory for the active bird.
 * @param features - Structured observation features used for the decision.
 * @param didFlap - Decision taken at this step.
 * @returns Nothing.
 */
export function commitSharedObservationMemoryStep(
  observationMemoryState: SharedObservationMemoryState,
  features: SharedObservationFeatures,
  didFlap: boolean,
): void {
  // Step 1: Persist current core frame for future stacked observations.
  const currentCoreObservationFrame =
    resolveCoreObservationVectorFromFeatures(features);
  const previousFrameCapacity = Math.max(
    0,
    FLAPPY_MEMORY_STACKED_FRAME_COUNT - 1,
  );
  observationMemoryState.previousCoreObservationFrames = [
    currentCoreObservationFrame,
    ...observationMemoryState.previousCoreObservationFrames,
  ].slice(0, previousFrameCapacity);

  // Step 2: Persist latest action into a fixed-size action history window.
  const actionScalar = didFlap ? 1 : 0;
  observationMemoryState.recentFlapActions = [
    actionScalar,
    ...observationMemoryState.recentFlapActions,
  ].slice(0, FLAPPY_MEMORY_ACTION_WINDOW_STEPS);
}

/**
 * Resolves previous core frames (newest-first) with deterministic zero padding.
 *
 * @param observationMemoryState - Mutable temporal memory for the active bird.
 * @returns Previous core frame list with fixed target length.
 */
function resolvePreviousCoreFramesWithPadding(
  observationMemoryState: SharedObservationMemoryState,
): number[][] {
  const previousFrameTargetCount = Math.max(
    0,
    FLAPPY_MEMORY_STACKED_FRAME_COUNT - 1,
  );
  const previousCoreFrames =
    observationMemoryState.previousCoreObservationFrames
      .slice(0, previousFrameTargetCount)
      .map((coreFrame) =>
        coreFrame.length === FLAPPY_MEMORY_CORE_FEATURE_COUNT
          ? coreFrame
          : resolveZeroCoreObservationFrame(),
      );

  while (previousCoreFrames.length < previousFrameTargetCount) {
    previousCoreFrames.push(resolveZeroCoreObservationFrame());
  }

  return previousCoreFrames;
}

/**
 * Builds a zero-valued core frame with canonical length.
 *
 * @returns Zero core frame.
 */
function resolveZeroCoreObservationFrame(): number[] {
  return Array.from({ length: FLAPPY_MEMORY_CORE_FEATURE_COUNT }, () => 0);
}

/**
 * Resolves flap/no-flap decision from network outputs.
 *
 * @param rawOutputs - Activation output payload.
 * @param flapThreshold - Scalar threshold for single-output policies.
 * @returns True when flap should trigger.
 */
export function resolveFlapDecision(
  rawOutputs: unknown,
  flapThreshold = 0.5,
): boolean {
  if (
    Array.isArray(rawOutputs) &&
    typeof rawOutputs[0] === 'number' &&
    typeof rawOutputs[1] === 'number'
  ) {
    return rawOutputs[1] > rawOutputs[0];
  }

  if (Array.isArray(rawOutputs) && typeof rawOutputs[0] === 'number') {
    return rawOutputs[0] > flapThreshold;
  }

  return typeof rawOutputs === 'number' ? rawOutputs > flapThreshold : false;
}

/**
 * Clamps a numeric value to the inclusive `[min, max]` interval.
 *
 * @param value - Candidate value.
 * @param min - Inclusive lower bound.
 * @param max - Inclusive upper bound.
 * @returns Clamped value.
 */
export function clampValue(value: number, min: number, max: number): number {
  return clamp(value, min, max);
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
export function clamp01(value: number): number {
  return clamp(value, 0, 1);
}

/**
 * Linear interpolation helper.
 *
 * @param startValue - Start value at progress `0`.
 * @param endValue - End value at progress `1`.
 * @param progress - Normalized interpolation progress.
 * @returns Interpolated value.
 */
export function interpolateValue(
  startValue: number,
  endValue: number,
  progress: number,
): number {
  return startValue + (endValue - startValue) * progress;
}
