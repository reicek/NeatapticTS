import {
  FLAPPY_CONTROL_SUBSTEPS_PER_FRAME,
  FLAPPY_DIFFICULTY_RAMP_PIPES,
  FLAPPY_BIRD_RADIUS_PX,
  FLAPPY_BIRD_X_PX,
  FLAPPY_FLAP_VELOCITY_PX_PER_FRAME,
  FLAPPY_GRAVITY_PX_PER_FRAME2,
  FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
  FLAPPY_MAX_FRAMES_PER_EPISODE,
  FLAPPY_PIPE_GAP_RANDOM_JITTER_PX,
  FLAPPY_PIPE_GAP_SHRINK_PER_PIPE_PX,
  FLAPPY_PIPE_GAP_START_MULTIPLIER,
  FLAPPY_PIPE_GAP_MIN_PX,
  FLAPPY_PIPE_GAP_CENTER_MAX_Y_PX,
  FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX,
  FLAPPY_PIPE_GAP_CENTER_MAX_DELTA_PX,
  FLAPPY_PIPE_GAP_PX,
  FLAPPY_PIPE_SPAWN_INTERVAL_MIN_FRAMES,
  FLAPPY_PIPE_SPAWN_INTERVAL_SHRINK_PER_PIPE_FRAMES,
  FLAPPY_PIPE_SPAWN_INTERVAL_START_MULTIPLIER,
  FLAPPY_PIPE_SPAWN_INTERVAL_FRAMES,
  FLAPPY_PIPE_SPEED_MAX_PX_PER_FRAME,
  FLAPPY_PIPE_SPEED_PX_PER_FRAME,
  FLAPPY_PIPE_WIDTH_PX,
  FLAPPY_WORLD_HEIGHT_PX,
  FLAPPY_WORLD_WIDTH_PX,
} from './constants.ts';
import type { FlappyRng } from './rng.ts';

/** Pipe obstacle definition. */
export interface FlappyPipe {
  /** Horizontal position of the left edge (pixels). */
  xPx: number;

  /** Vertical center of the opening/gap (pixels). */
  gapCenterYPx: number;

  /** Vertical gap size used by this specific pipe (pixels). */
  gapSizePx: number;

  /** Whether the bird has already been credited for passing this pipe. */
  passed: boolean;
}

/** Bird kinematic state. */
export interface FlappyBird {
  /** Vertical position (pixels). */
  yPx: number;

  /** Vertical velocity (pixels/frame). */
  velocityYPxPerFrame: number;
}

/** Full simulation state (single episode). */
export interface FlappyGameState {
  /** Current frame counter (0-based). */
  frameIndex: number;

  /** Bird state. */
  bird: FlappyBird;

  /** Pipe obstacles ordered by increasing x (left to right). */
  pipes: FlappyPipe[];

  /** Gap size of the most recently spawned pipe (pixels). */
  lastSpawnedPipeGapPx: number;

  /** Gap-center y value of the most recently spawned pipe (pixels). */
  lastSpawnedPipeGapCenterYPx: number;

  /** Spawn interval used for the most recently spawned pipe (frames). */
  lastSpawnedPipeSpawnIntervalFrames: number;

  /** Countdown until the next pipe spawn (frames). */
  framesUntilNextPipeSpawn: number;

  /** Total number of pipes passed. */
  pipesPassed: number;

  /** Whether the episode has terminated. */
  done: boolean;

  /**
   * Termination reason (diagnostic only).
   * - `collision`: hit a pipe
   * - `out_of_bounds`: hit floor/ceiling
   * - `timeout`: exceeded max frame limit
   */
  doneReason?: 'collision' | 'out_of_bounds' | 'timeout';
}

/**
 * Structured observation features used to build the neural-network input vector.
 */
export interface FlappyObservationFeatures {
  /** Bird y position normalized to [0, 1]. */
  normalizedBirdY: number;

  /** Bird vertical velocity normalized to [-1, 1]. */
  normalizedVelocity: number;

  /** Distance from bird to next pipe normalized to [0, 1]. */
  normalizedDistanceToNextPipe: number;

  /** Delta from bird y to next gap center normalized to [-1, 1]. */
  normalizedDeltaToNextGap: number;

  /** Next pipe gap top normalized to [0, 1]. */
  normalizedNextGapTop: number;

  /** Next pipe gap bottom normalized to [0, 1]. */
  normalizedNextGapBottom: number;

  /** Distance from bird to second upcoming pipe normalized to [0, 1]. */
  normalizedDistanceToSecondPipe: number;

  /** Delta from bird y to second upcoming gap center normalized to [-1, 1]. */
  normalizedDeltaToSecondGap: number;

  /** Time-to-next-pipe closeness in [0, 1] (1 means imminent). */
  normalizedTimeToNextPipe: number;

  /** Signed clearance to next gap in [-1, 1] (positive inside gap). */
  normalizedNextGapClearance: number;

  /** Required vertical velocity toward next gap center normalized to [-1, 1]. */
  normalizedRequiredVerticalVelocityToNextGap: number;

  /** Next-to-second gap center transition normalized to [-1, 1]. */
  normalizedNextToSecondGapTransition: number;
}

/**
 * Difficulty scale used by the curriculum scheduler.
 *
 * - `0` means easiest profile (wide gaps, slower pipes).
 * - `1` means fully adaptive profile based on passed pipes.
 */
export type FlappyDifficultyScale = number;

/**
 * Create a fresh Flappy Bird episode state.
 *
 * @param rng - Random source used to generate pipes.
 * @returns Initial state.
 */
export function createInitialFlappyState(rng: FlappyRng): FlappyGameState {
  const initialGapCenter = sampleGapCenterY(rng);
  const initialDifficultyProfile = resolveDifficultyProfile(0);
  const initialGapSizePx = resolveNextSpawnGapSize(
    undefined,
    initialDifficultyProfile,
    rng,
  );
  const initialSpawnIntervalFrames = resolveNextSpawnIntervalFrames(
    undefined,
    initialDifficultyProfile,
  );

  return {
    frameIndex: 0,
    bird: {
      yPx: FLAPPY_WORLD_HEIGHT_PX * 0.5,
      velocityYPxPerFrame: 0,
    },
    pipes: [
      {
        xPx: FLAPPY_WORLD_WIDTH_PX + FLAPPY_PIPE_WIDTH_PX,
        gapCenterYPx: initialGapCenter,
        gapSizePx: initialGapSizePx,
        passed: false,
      },
    ],
    lastSpawnedPipeGapPx: initialGapSizePx,
    lastSpawnedPipeGapCenterYPx: initialGapCenter,
    lastSpawnedPipeSpawnIntervalFrames: initialSpawnIntervalFrames,
    framesUntilNextPipeSpawn: initialSpawnIntervalFrames,
    pipesPassed: 0,
    done: false,
  };
}

/**
 * Advance the simulation by one frame.
 *
 * @param state - Mutable state object to update in-place.
 * @param rng - Random source used to spawn pipes.
 * @param flap - If true, applies an upward velocity impulse.
 */
export function stepFlappyState(
  state: FlappyGameState,
  rng: FlappyRng,
  flap: boolean,
  difficultyScale: FlappyDifficultyScale = 1,
): void {
  stepFlappyStateWithControlSubsteps(
    state,
    rng,
    () => flap,
    difficultyScale,
    1,
  );
}

/**
 * Advance one logical frame using multiple control/physics substeps.
 *
 * This allows policies to react multiple times before `frameIndex` advances,
 * improving responsiveness in high-difficulty scenarios.
 *
 * @param state - Mutable state object to update in-place.
 * @param rng - Random source used to spawn pipes.
 * @param shouldFlapForSubstep - Callback deciding flap action per substep.
 * @param difficultyScale - Curriculum difficulty scale in [0, 1].
 * @param controlSubstepsPerFrame - Number of substeps to run this frame.
 * @returns Nothing.
 */
export function stepFlappyStateWithControlSubsteps(
  state: FlappyGameState,
  rng: FlappyRng,
  shouldFlapForSubstep: () => boolean,
  difficultyScale: FlappyDifficultyScale = 1,
  controlSubstepsPerFrame: number = FLAPPY_CONTROL_SUBSTEPS_PER_FRAME,
): void {
  if (state.done) return;

  const difficultyProfile = resolveDifficultyProfile(
    state.pipesPassed,
    difficultyScale,
  );
  const substepCount = Math.max(1, Math.trunc(controlSubstepsPerFrame));
  const substepDelta = 1 / substepCount;

  // Step 1: Run high-frequency control/physics substeps.
  for (
    let controlSubstepIndex = 0;
    controlSubstepIndex < substepCount && !state.done;
    controlSubstepIndex++
  ) {
    const flap = shouldFlapForSubstep();
    if (flap) {
      state.bird.velocityYPxPerFrame = FLAPPY_FLAP_VELOCITY_PX_PER_FRAME;
    }

    state.bird.velocityYPxPerFrame = clamp(
      state.bird.velocityYPxPerFrame + FLAPPY_GRAVITY_PX_PER_FRAME2 * substepDelta,
      -Infinity,
      FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
    );
    state.bird.yPx += state.bird.velocityYPxPerFrame * substepDelta;

    for (const pipe of state.pipes) {
      pipe.xPx -= difficultyProfile.pipeSpeedPxPerFrame * substepDelta;
    }
    state.pipes = state.pipes.filter(
      (pipe) => pipe.xPx + FLAPPY_PIPE_WIDTH_PX > 0,
    );

    state.framesUntilNextPipeSpawn -= substepDelta;
    if (state.framesUntilNextPipeSpawn <= 0) {
      const nextGapSizePx = resolveNextSpawnGapSize(
        state.lastSpawnedPipeGapPx,
        difficultyProfile,
        rng,
      );
      const nextSpawnIntervalFrames = resolveNextSpawnIntervalFrames(
        state.lastSpawnedPipeSpawnIntervalFrames,
        difficultyProfile,
      );
      const nextGapCenterYPx = resolveNextSpawnGapCenterY(
        state.lastSpawnedPipeGapCenterYPx,
        rng,
      );
      state.pipes.push({
        xPx: FLAPPY_WORLD_WIDTH_PX + FLAPPY_PIPE_WIDTH_PX,
        gapCenterYPx: nextGapCenterYPx,
        gapSizePx: nextGapSizePx,
        passed: false,
      });
      state.lastSpawnedPipeGapPx = nextGapSizePx;
      state.lastSpawnedPipeGapCenterYPx = nextGapCenterYPx;
      state.lastSpawnedPipeSpawnIntervalFrames = nextSpawnIntervalFrames;
      state.framesUntilNextPipeSpawn += nextSpawnIntervalFrames;
    }

    if (state.bird.yPx - FLAPPY_BIRD_RADIUS_PX <= 0) {
      state.done = true;
      state.doneReason = 'out_of_bounds';
    } else if (
      state.bird.yPx + FLAPPY_BIRD_RADIUS_PX >= FLAPPY_WORLD_HEIGHT_PX
    ) {
      state.done = true;
      state.doneReason = 'out_of_bounds';
    }

    if (!state.done) {
      const birdLeft = FLAPPY_BIRD_X_PX - FLAPPY_BIRD_RADIUS_PX;
      const birdRight = FLAPPY_BIRD_X_PX + FLAPPY_BIRD_RADIUS_PX;

      for (const pipe of state.pipes) {
        const pipeLeft = pipe.xPx;
        const pipeRight = pipe.xPx + FLAPPY_PIPE_WIDTH_PX;

        const overlapsHorizontally =
          birdRight >= pipeLeft && birdLeft <= pipeRight;
        if (overlapsHorizontally) {
          const gapHalf = pipe.gapSizePx * 0.5;
          const gapTop = pipe.gapCenterYPx - gapHalf;
          const gapBottom = pipe.gapCenterYPx + gapHalf;

          const birdTop = state.bird.yPx - FLAPPY_BIRD_RADIUS_PX;
          const birdBottom = state.bird.yPx + FLAPPY_BIRD_RADIUS_PX;

          const insideGap = birdTop >= gapTop && birdBottom <= gapBottom;
          if (!insideGap) {
            state.done = true;
            state.doneReason = 'collision';
            break;
          }
        }

        if (!pipe.passed && pipeRight < FLAPPY_BIRD_X_PX) {
          pipe.passed = true;
          state.pipesPassed++;
        }
      }
    }
  }

  // Step 2: Frame accounting and timeout.
  state.frameIndex++;
  if (!state.done && state.frameIndex >= FLAPPY_MAX_FRAMES_PER_EPISODE) {
    state.done = true;
    state.doneReason = 'timeout';
  }

  /**
   * @param value - Value to clamp.
   * @param min - Lower bound.
   * @param max - Upper bound.
   */
  function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
  }
}

/**
 * Resolves adaptive difficulty parameters from current progress.
 *
 * Difficulty ramps with passed pipes: starts easier and trends toward baseline/challenging values.
 *
 * @param pipesPassed - Number of pipes passed by the active bird.
 * @param difficultyScale - Curriculum difficulty scale in [0, 1].
 * @returns Runtime difficulty profile.
 */
function resolveDifficultyProfile(
  pipesPassed: number,
  difficultyScale: FlappyDifficultyScale = 1,
): {
  pipeGapPx: number;
  pipeSpeedPxPerFrame: number;
  pipeSpawnIntervalFrames: number;
} {
  const normalizedDifficultyScale = clampValue(difficultyScale, 0, 1);
  const normalizedDifficultyProgress = clampValue(
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
 * Linearly interpolates between two scalar values.
 *
 * @param startValue - Value at progress = 0.
 * @param endValue - Value at progress = 1.
 * @param progress - Interpolation progress in [0, 1].
 * @returns Interpolated value.
 */
function interpolateValue(
  startValue: number,
  endValue: number,
  progress: number,
): number {
  return startValue + (endValue - startValue) * progress;
}

/**
 * Clamps a scalar to the provided range.
 *
 * @param value - Input value.
 * @param min - Lower bound.
 * @param max - Upper bound.
 * @returns Clamped value.
 */
function clampValue(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

/**
 * Resolve the next spawned pipe gap with progressive wide-to-hard shrink and jitter.
 *
 * @param previousSpawnGapPx - Most recently spawned gap size.
 * @param difficultyProfile - Current adaptive profile.
 * @param rng - Random source.
 * @returns Gap size for the next pipe.
 */
function resolveNextSpawnGapSize(
  previousSpawnGapPx: number | undefined,
  difficultyProfile: {
    pipeGapPx: number;
    pipeSpeedPxPerFrame: number;
    pipeSpawnIntervalFrames: number;
  },
  rng: FlappyRng,
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

  return Math.round(clampValue(randomizedGapPx, hardestGapPx, initialWideGapPx));
}

/**
 * Resolve the next spawned pipe interval with progressive wide-to-hard shrink.
 *
 * @param previousSpawnIntervalFrames - Most recently spawned interval.
 * @param difficultyProfile - Current adaptive profile.
 * @returns Spawn interval for the next pipe.
 */
function resolveNextSpawnIntervalFrames(
  previousSpawnIntervalFrames: number | undefined,
  difficultyProfile: {
    pipeGapPx: number;
    pipeSpeedPxPerFrame: number;
    pipeSpawnIntervalFrames: number;
  },
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
    clampValue(
      progressiveIntervalFrames,
      hardestIntervalFrames,
      initialWideIntervalFrames,
    ),
  );
}

/**
 * Generate the network observation vector for the current state.
 *
 * Observation (12 numbers):
 *  1) bird y position normalized to [0, 1]
 *  2) bird vertical velocity normalized to [-1, 1]
 *  3) distance to next pipe normalized to [0, 1]
 *  4) delta (bird y - gap center y) normalized to [-1, 1]
 *  5) next pipe gap top normalized to [0, 1]
 *  6) next pipe gap bottom normalized to [0, 1]
 *  7) distance to second pipe normalized to [0, 1]
 *  8) delta to second gap center normalized to [-1, 1]
 *  9) time-to-next-pipe closeness normalized to [0, 1]
 * 10) signed clearance relative to next gap normalized to [-1, 1]
 * 11) required vertical velocity toward next gap center normalized to [-1, 1]
 * 12) gap-center transition (next to second) normalized to [-1, 1]
 *
 * @param state - Current state.
 * @param difficultyScale - Curriculum difficulty scale in [0, 1].
 * @returns Input vector for the neural network.
 */
export function getFlappyObservation(
  state: FlappyGameState,
  difficultyScale: FlappyDifficultyScale = 1,
): number[] {
  const observationFeatures = getFlappyObservationFeatures(
    state,
    difficultyScale,
  );
  return [
    observationFeatures.normalizedBirdY,
    observationFeatures.normalizedVelocity,
    observationFeatures.normalizedDistanceToNextPipe,
    observationFeatures.normalizedDeltaToNextGap,
    observationFeatures.normalizedNextGapTop,
    observationFeatures.normalizedNextGapBottom,
    observationFeatures.normalizedDistanceToSecondPipe,
    observationFeatures.normalizedDeltaToSecondGap,
    observationFeatures.normalizedTimeToNextPipe,
    observationFeatures.normalizedNextGapClearance,
    observationFeatures.normalizedRequiredVerticalVelocityToNextGap,
    observationFeatures.normalizedNextToSecondGapTransition,
  ];
}

/**
 * Resolve structured observation features for policy input and reward shaping.
 *
 * @param state - Current state.
 * @param difficultyScale - Curriculum difficulty scale in [0, 1].
 * @returns Named feature object.
 */
export function getFlappyObservationFeatures(
  state: FlappyGameState,
  difficultyScale: FlappyDifficultyScale = 1,
): FlappyObservationFeatures {
  const [nextPipe, secondPipe] = findUpcomingPipes(state.pipes);
  const birdYPx = state.bird.yPx;

  const normalizedBirdY = clamp01(birdYPx / FLAPPY_WORLD_HEIGHT_PX);
  const normalizedVelocity = clamp(
    state.bird.velocityYPxPerFrame / FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
    -1,
    1,
  );

  const distanceToNextPipePx = nextPipe
    ? nextPipe.xPx + FLAPPY_PIPE_WIDTH_PX - FLAPPY_BIRD_X_PX
    : FLAPPY_WORLD_WIDTH_PX;
  const normalizedDistanceToNextPipe = clamp01(
    distanceToNextPipePx / FLAPPY_WORLD_WIDTH_PX,
  );

  const nextGapCenterYPx = nextPipe?.gapCenterYPx ?? FLAPPY_WORLD_HEIGHT_PX * 0.5;
  const nextGapHalfPx = (nextPipe?.gapSizePx ?? FLAPPY_PIPE_GAP_PX) * 0.5;
  const normalizedNextGapTop = clamp01(
    (nextGapCenterYPx - nextGapHalfPx) / FLAPPY_WORLD_HEIGHT_PX,
  );
  const normalizedNextGapBottom = clamp01(
    (nextGapCenterYPx + nextGapHalfPx) / FLAPPY_WORLD_HEIGHT_PX,
  );
  const normalizedDeltaToNextGap = clamp(
    (birdYPx - nextGapCenterYPx) / FLAPPY_WORLD_HEIGHT_PX,
    -1,
    1,
  );

  const distanceToSecondPipePx = secondPipe
    ? secondPipe.xPx + FLAPPY_PIPE_WIDTH_PX - FLAPPY_BIRD_X_PX
    : FLAPPY_WORLD_WIDTH_PX;
  const normalizedDistanceToSecondPipe = clamp01(
    distanceToSecondPipePx / FLAPPY_WORLD_WIDTH_PX,
  );

  const secondGapCenterYPx = secondPipe?.gapCenterYPx ?? nextGapCenterYPx;
  const normalizedDeltaToSecondGap = clamp(
    (birdYPx - secondGapCenterYPx) / FLAPPY_WORLD_HEIGHT_PX,
    -1,
    1,
  );

  const difficultyProfile = resolveDifficultyProfile(
    state.pipesPassed,
    difficultyScale,
  );
  const activeSpawnIntervalFrames = Math.max(
    1,
    state.lastSpawnedPipeSpawnIntervalFrames,
  );
  const estimatedFramesToNextPipe =
    distanceToNextPipePx / Math.max(0.001, difficultyProfile.pipeSpeedPxPerFrame);
  const normalizedTimeToNextPipe = clamp01(
    1 - estimatedFramesToNextPipe / activeSpawnIntervalFrames,
  );

  const distanceToNextGapCenterPx = Math.abs(birdYPx - nextGapCenterYPx);
  const normalizedNextGapClearance = clamp(
    (nextGapHalfPx - distanceToNextGapCenterPx) / Math.max(1, nextGapHalfPx),
    -1,
    1,
  );

  const requiredVerticalVelocityToNextGapPxPerFrame = clamp(
    (nextGapCenterYPx - birdYPx) / Math.max(1, estimatedFramesToNextPipe),
    -FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
    FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
  );
  const normalizedRequiredVerticalVelocityToNextGap = clamp(
    requiredVerticalVelocityToNextGapPxPerFrame /
      FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
    -1,
    1,
  );

  const normalizedNextToSecondGapTransition = clamp(
    (secondGapCenterYPx - nextGapCenterYPx) / FLAPPY_WORLD_HEIGHT_PX,
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

  function clamp01(value: number): number {
    return Math.min(1, Math.max(0, value));
  }

  function clamp(value: number, min: number, max: number): number {
    return Math.min(max, Math.max(min, value));
  }

  function findUpcomingPipes(
    pipes: FlappyPipe[],
  ): [FlappyPipe | undefined, FlappyPipe | undefined] {
    const birdFrontX = FLAPPY_BIRD_X_PX - FLAPPY_BIRD_RADIUS_PX;
    const upcomingPipes = pipes.filter(
      (pipe) => pipe.xPx + FLAPPY_PIPE_WIDTH_PX >= birdFrontX,
    );
    return [upcomingPipes[0], upcomingPipes[1]];
  }
}

/**
 * Sample a gap center height inside configured bounds.
 *
 * @param rng - Random source.
 * @returns Gap center y coordinate (pixels).
 */
export function sampleGapCenterY(rng: FlappyRng): number {
  return rng.nextInt(
    FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX,
    FLAPPY_PIPE_GAP_CENTER_MAX_Y_PX,
  );
}

/**
 * Resolves next gap-center y while limiting abrupt consecutive transitions.
 *
 * @param previousGapCenterYPx - Previous spawned gap-center y value.
 * @param rng - Random source.
 * @returns Next gap-center y constrained by transition and world bounds.
 */
export function resolveNextSpawnGapCenterY(
  previousGapCenterYPx: number,
  rng: FlappyRng,
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
  return clampValue(
    sampledGapCenterYPx,
    minimumGapCenterYPx,
    maximumGapCenterYPx,
  );
}
