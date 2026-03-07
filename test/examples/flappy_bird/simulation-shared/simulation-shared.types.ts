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
