import {
  FLAPPY_BIRD_RADIUS_PX,
  FLAPPY_BIRD_X_PX,
  FLAPPY_DIFFICULTY_RAMP_PIPES,
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
  const worldHeightPx = input.worldHeightPx ?? FLAPPY_WORLD_HEIGHT_PX;
  const maxFallSpeedPxPerFrame =
    input.maxFallSpeedPxPerFrame ?? FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME;
  const birdCenterXPx = input.birdCenterXPx ?? FLAPPY_BIRD_X_PX;
  const birdRadiusPx = input.birdRadiusPx ?? FLAPPY_BIRD_RADIUS_PX;
  const pipeWidthPx = input.pipeWidthPx ?? FLAPPY_PIPE_WIDTH_PX;
  const defaultGapSizePx = input.defaultGapSizePx ?? FLAPPY_PIPE_GAP_PX;
  const normalizationEpsilon = input.normalizationEpsilon ?? 0.001;

  const [nextPipe, secondPipe] = resolveUpcomingPipes(
    input.pipes,
    birdCenterXPx,
    birdRadiusPx,
    pipeWidthPx,
  );
  const normalizedBirdY = clamp01(input.birdYPx / worldHeightPx);
  const normalizedVelocity = clamp(
    input.velocityYPxPerFrame / maxFallSpeedPxPerFrame,
    -1,
    1,
  );

  const distanceToNextPipePx = nextPipe
    ? nextPipe.xPx + pipeWidthPx - birdCenterXPx
    : input.visibleWorldWidthPx;
  const normalizedDistanceToNextPipe = clamp01(
    distanceToNextPipePx / input.visibleWorldWidthPx,
  );

  const nextGapCenterYPx = nextPipe?.gapCenterYPx ?? worldHeightPx * 0.5;
  const nextGapHalfPx = (nextPipe?.gapSizePx ?? defaultGapSizePx) * 0.5;
  const normalizedNextGapTop = clamp01(
    (nextGapCenterYPx - nextGapHalfPx) / worldHeightPx,
  );
  const normalizedNextGapBottom = clamp01(
    (nextGapCenterYPx + nextGapHalfPx) / worldHeightPx,
  );
  const normalizedDeltaToNextGap = clamp(
    (input.birdYPx - nextGapCenterYPx) / worldHeightPx,
    -1,
    1,
  );

  const distanceToSecondPipePx = secondPipe
    ? secondPipe.xPx + pipeWidthPx - birdCenterXPx
    : input.visibleWorldWidthPx;
  const normalizedDistanceToSecondPipe = clamp01(
    distanceToSecondPipePx / input.visibleWorldWidthPx,
  );

  const secondGapCenterYPx = secondPipe?.gapCenterYPx ?? nextGapCenterYPx;
  const normalizedDeltaToSecondGap = clamp(
    (input.birdYPx - secondGapCenterYPx) / worldHeightPx,
    -1,
    1,
  );

  const estimatedFramesToNextPipe =
    distanceToNextPipePx /
    Math.max(normalizationEpsilon, input.difficultyProfile.pipeSpeedPxPerFrame);
  const normalizedTimeToNextPipe = clamp01(
    1 -
      estimatedFramesToNextPipe / Math.max(1, input.activeSpawnIntervalFrames),
  );

  const distanceToNextGapCenterPx = Math.abs(input.birdYPx - nextGapCenterYPx);
  const normalizedNextGapClearance = clamp(
    (nextGapHalfPx - distanceToNextGapCenterPx) / Math.max(1, nextGapHalfPx),
    -1,
    1,
  );

  const requiredVerticalVelocityToNextGapPxPerFrame = clamp(
    (nextGapCenterYPx - input.birdYPx) / Math.max(1, estimatedFramesToNextPipe),
    -maxFallSpeedPxPerFrame,
    maxFallSpeedPxPerFrame,
  );
  const normalizedRequiredVerticalVelocityToNextGap = clamp(
    requiredVerticalVelocityToNextGapPxPerFrame / maxFallSpeedPxPerFrame,
    -1,
    1,
  );

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
