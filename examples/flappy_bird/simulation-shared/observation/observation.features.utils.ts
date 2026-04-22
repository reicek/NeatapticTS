import {
  FLAPPY_BIRD_RADIUS_PX,
  FLAPPY_BIRD_X_PX,
  FLAPPY_FLAP_VELOCITY_PX_PER_FRAME,
  FLAPPY_GRAVITY_PX_PER_FRAME2,
  FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
  FLAPPY_PIPE_GAP_PX,
  FLAPPY_PIPE_WIDTH_PX,
  FLAPPY_WORLD_HEIGHT_PX,
} from '../../constants/constants';
import { FLAPPY_SHARED_DEFAULT_NORMALIZATION_EPSILON } from '../simulation-shared.constants';
import type {
  SharedObservationFeatures,
  SharedObservationInput,
  SharedPipeLike,
} from '../simulation-shared.types';

/**
 * Resolves the next two upcoming pipes in front of the bird.
 *
 * The shared simulation helpers sometimes need the first two obstacles even
 * though the current controller contract only reads the next immediate gap.
 * Keeping this helper small and explicit makes it easy for callers to choose
 * how much near-future geometry they actually want.
 *
 * @param pipes - Current pipe list.
 * @param birdCenterXPx - Bird center x-position.
 * @param birdRadiusPx - Bird radius.
 * @param pipeWidthPx - Pipe width.
 * @returns Tuple of first and second upcoming pipes.
 * @example
 * ```ts
 * const [nextPipe, secondPipe] = resolveUpcomingPipes(pipes);
 * ```
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
 * Educational note:
 * This helper stays focused on semantic feature assembly only. Projection into
 * the canonical network vectors now lives in the neighboring vector module so
 * observation policy and network-shape concerns can evolve independently.
 *
 * The features deliberately mix two kinds of control signal plus a small set of
 * shaping-oriented derived hints:
 * 1. Current state, such as bird height and vertical velocity.
 * 2. Immediate next-gap geometry, such as distance, offset, and corridor
 *    bounds.
 *
 * This is a compact example of feature engineering for control. Instead of
 * asking NEAT to rediscover basic geometry from raw sensory input, the example
 * hands the network semantically meaningful signals and lets evolution focus on
 * policy search.
 *
 * For broader context, the Wikipedia article on "feature engineering" is a
 * good companion reference.
 *
 * @param input - Observation input bundle.
 * @returns Structured observation features.
 * @example
 * ```ts
 * const features = resolveObservationFeatures({
 *   birdYPx: 120,
 *   velocityYPxPerFrame: 2,
 *   pipes,
 *   visibleWorldWidthPx: 640,
 *   difficultyProfile,
 *   activeSpawnIntervalFrames: 90,
 * });
 *
 * if (features.normalizedEntryUrgency > 0.8) {
 *   // The bird is misaligned and running out of time to recover.
 * }
 * ```
 */
export function resolveObservationFeatures(
  input: SharedObservationInput,
): SharedObservationFeatures {
  // Step 1: Resolve normalization and geometry defaults.
  const worldHeightPx = input.worldHeightPx ?? FLAPPY_WORLD_HEIGHT_PX;
  const maxFallSpeedPxPerFrame =
    input.maxFallSpeedPxPerFrame ?? FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME;
  const birdCenterXPx = input.birdCenterXPx ?? FLAPPY_BIRD_X_PX;
  const birdRadiusPx = input.birdRadiusPx ?? FLAPPY_BIRD_RADIUS_PX;
  const pipeWidthPx = input.pipeWidthPx ?? FLAPPY_PIPE_WIDTH_PX;
  const defaultGapSizePx = input.defaultGapSizePx ?? FLAPPY_PIPE_GAP_PX;
  const normalizationEpsilon =
    input.normalizationEpsilon ?? FLAPPY_SHARED_DEFAULT_NORMALIZATION_EPSILON;

  // Step 2: Resolve next-pipe geometry used by all near-term features.
  const [nextPipe] = resolveUpcomingPipes(
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

  // Step 3: Resolve timing, clearance, and controllability signals.
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

  const distanceToGapEntryPx = nextPipe
    ? nextPipe.xPx - (birdCenterXPx + birdRadiusPx)
    : input.visibleWorldWidthPx;
  const distanceToGapExitPx = nextPipe
    ? nextPipe.xPx + pipeWidthPx - (birdCenterXPx - birdRadiusPx)
    : input.visibleWorldWidthPx;

  const framesToGapEntry = Math.max(
    0,
    distanceToGapEntryPx /
      Math.max(
        normalizationEpsilon,
        input.difficultyProfile.pipeSpeedPxPerFrame,
      ),
  );
  const framesToGapExit = Math.max(
    framesToGapEntry,
    distanceToGapExitPx /
      Math.max(
        normalizationEpsilon,
        input.difficultyProfile.pipeSpeedPxPerFrame,
      ),
  );

  const normalizedFramesToGapEntry = clamp01(
    framesToGapEntry / Math.max(1, input.activeSpawnIntervalFrames),
  );
  const normalizedFramesToGapExit = clamp01(
    framesToGapExit / Math.max(1, input.activeSpawnIntervalFrames),
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

  const requiredVerticalVelocityAtGapEntryPxPerFrame = clamp(
    (nextGapCenterYPx - input.birdYPx) / Math.max(1, framesToGapEntry),
    -maxFallSpeedPxPerFrame,
    maxFallSpeedPxPerFrame,
  );
  const requiredVerticalVelocityAtGapExitPxPerFrame = clamp(
    (nextGapCenterYPx - input.birdYPx) / Math.max(1, framesToGapExit),
    -maxFallSpeedPxPerFrame,
    maxFallSpeedPxPerFrame,
  );
  const normalizedRequiredVerticalVelocityAtGapEntry = clamp(
    requiredVerticalVelocityAtGapEntryPxPerFrame / maxFallSpeedPxPerFrame,
    -1,
    1,
  );
  const normalizedRequiredVerticalVelocityAtGapExit = clamp(
    requiredVerticalVelocityAtGapExitPxPerFrame / maxFallSpeedPxPerFrame,
    -1,
    1,
  );

  const normalizedEntryUrgency = clamp(
    Math.abs(normalizedDeltaToNextGap) /
      Math.max(normalizationEpsilon, normalizedFramesToGapEntry),
    0,
    1,
  );

  const predictedBirdYAtEntryWithoutFlap = resolvePredictedBirdYAtFrames(
    input.birdYPx,
    input.velocityYPxPerFrame,
    framesToGapEntry,
  );
  const predictedBirdYAtEntryWithOneFlap = resolvePredictedBirdYAtFrames(
    input.birdYPx,
    FLAPPY_FLAP_VELOCITY_PX_PER_FRAME,
    framesToGapEntry,
  );
  const noFlapEntryErrorPx = Math.abs(
    predictedBirdYAtEntryWithoutFlap - nextGapCenterYPx,
  );
  const oneFlapEntryErrorPx = Math.abs(
    predictedBirdYAtEntryWithOneFlap - nextGapCenterYPx,
  );
  const minimalEntryErrorPx = Math.min(noFlapEntryErrorPx, oneFlapEntryErrorPx);
  const normalizedOneFlapReachabilityAtGapEntry =
    minimalEntryErrorPx <= nextGapHalfPx ? 1 : 0;

  // Step 4: Return the structured feature record consumed by both runtimes.
  return {
    normalizedBirdY,
    normalizedVelocity,
    normalizedDistanceToNextPipe,
    normalizedDeltaToNextGap,
    normalizedNextGapTop,
    normalizedNextGapBottom,
    normalizedTimeToNextPipe,
    normalizedNextGapClearance,
    normalizedRequiredVerticalVelocityToNextGap,
    normalizedFramesToGapEntry,
    normalizedFramesToGapExit,
    normalizedRequiredVerticalVelocityAtGapEntry,
    normalizedRequiredVerticalVelocityAtGapExit,
    normalizedEntryUrgency,
    normalizedOneFlapReachabilityAtGapEntry,
  };
}

/**
 * Predicts bird y-position after a frame horizon with constant gravity.
 *
 * This is a deliberately cheap forward model. It ignores richer control
 * sequences and simply answers: "where would the bird be after N frames under
 * the current physics assumption?" That small prediction is enough to build the
 * reachability and urgency features used by the controller.
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
 * Clamps a numeric value to the inclusive [min, max] interval.
 *
 * Observation synthesis normalizes many raw measurements, so this helper keeps
 * derived channels inside their documented ranges.
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
 * Clamps a numeric value to the inclusive [0, 1] interval.
 *
 * This is used for channels that are naturally interpreted as normalized
 * proportions or bounded progress values.
 *
 * @param value - Candidate value.
 * @returns Value clamped between 0 and 1.
 */
function clamp01(value: number): number {
  return clamp(value, 0, 1);
}
