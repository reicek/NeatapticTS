import {
  FLAPPY_FLAP_THRESHOLD,
  FLAPPY_NORMALIZATION_EPSILON,
} from '../constants/constants';
import {
  FLAPPY_BIRD_RADIUS_PX,
  FLAPPY_BIRD_X_PX,
  FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
  FLAPPY_PIPE_GAP_PX,
  FLAPPY_PIPE_WIDTH_PX,
} from '../constants/constants';
import {
  commitSharedObservationMemoryStep,
  type SharedObservationFeatures,
  type SharedObservationMemoryState,
  resolveFlapDecision as resolveSharedFlapDecision,
  resolveObservationFeatures,
  resolveTemporalObservationVector,
  resolveUpcomingPipes as resolveSharedUpcomingPipes,
} from '../flappy.simulation.shared.utils';
import type {
  BrowserDifficultyProfile,
  BrowserPopulationBirdLike,
  BrowserPopulationPipeLike,
} from './browser-entry.types';

/**
 * Builds the normalized observation vector consumed by bird networks.
 *
 * @param birdYPx - Bird y position.
 * @param velocityYPxPerFrame - Bird vertical velocity.
 * @param pipes - Current pipe list.
 * @param visibleWorldWidthPx - Current visible world width.
 * @param worldHeightPx - Current world height used for normalization and bounds.
 * @param difficultyProfile - Active difficulty profile.
 * @param activeSpawnIntervalFrames - Current spawn interval.
 * @param observationMemoryState - Shared compatibility memory state kept alongside browser decisions.
 * @returns Ordered normalized observation vector.
 */
export function resolveObservationVector(
  birdYPx: number,
  velocityYPxPerFrame: number,
  pipes: BrowserPopulationPipeLike[],
  visibleWorldWidthPx: number,
  worldHeightPx: number,
  difficultyProfile: BrowserDifficultyProfile,
  activeSpawnIntervalFrames: number,
  observationMemoryState: SharedObservationMemoryState,
): {
  observationVector: number[];
  observationFeatures: SharedObservationFeatures;
} {
  const observationFeatures = resolveObservationFeatures({
    birdYPx,
    velocityYPxPerFrame,
    pipes,
    visibleWorldWidthPx,
    difficultyProfile,
    activeSpawnIntervalFrames,
    defaultGapSizePx: FLAPPY_PIPE_GAP_PX,
    birdCenterXPx: FLAPPY_BIRD_X_PX,
    birdRadiusPx: FLAPPY_BIRD_RADIUS_PX,
    pipeWidthPx: FLAPPY_PIPE_WIDTH_PX,
    worldHeightPx,
    maxFallSpeedPxPerFrame: FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
    normalizationEpsilon: FLAPPY_NORMALIZATION_EPSILON,
  });

  return {
    observationVector: resolveTemporalObservationVector(
      observationFeatures,
      observationMemoryState,
    ),
    observationFeatures,
  };
}

/**
 * Commits one browser decision step into the shared compatibility memory surface.
 *
 * @param observationMemoryState - Mutable memory state for one bird.
 * @param observationFeatures - Structured features used for this decision.
 * @param shouldFlap - Action selected by the policy.
 * @returns Nothing.
 */
export function commitObservationMemoryStep(
  observationMemoryState: SharedObservationMemoryState,
  observationFeatures: SharedObservationFeatures,
  shouldFlap: boolean,
): void {
  commitSharedObservationMemoryStep(
    observationMemoryState,
    observationFeatures,
    shouldFlap,
  );
}

/**
 * Resolves the next two upcoming pipes in front of the bird.
 *
 * @param pipes - Current pipe list.
 * @returns Tuple of first and second upcoming pipes.
 */
export function resolveUpcomingPipes(
  pipes: BrowserPopulationPipeLike[],
): [
  BrowserPopulationPipeLike | undefined,
  BrowserPopulationPipeLike | undefined,
] {
  return resolveSharedUpcomingPipes(
    pipes,
    FLAPPY_BIRD_X_PX,
    FLAPPY_BIRD_RADIUS_PX,
    FLAPPY_PIPE_WIDTH_PX,
  );
}

/**
 * Resolves flap/no-flap decision from network outputs.
 *
 * @param rawOutputs - Activation output payload.
 * @returns True when flap should trigger.
 */
export function resolveFlapDecision(rawOutputs: unknown): boolean {
  return resolveSharedFlapDecision(rawOutputs, FLAPPY_FLAP_THRESHOLD);
}

/**
 * Counts birds that are still alive.
 *
 * @param birds - Population birds.
 * @returns Alive bird count.
 */
export function resolveAliveBirdCount(
  birds: BrowserPopulationBirdLike[],
): number {
  return birds.filter((bird) => !bird.done).length;
}

/**
 * Resolves leading pipes-passed score in the population.
 *
 * @param birds - Population birds.
 * @returns Maximum pipes passed.
 */
export function resolveLeaderPipesPassed(
  birds: BrowserPopulationBirdLike[],
): number {
  return birds.reduce(
    (leaderPipesPassed, bird) => Math.max(leaderPipesPassed, bird.pipesPassed),
    0,
  );
}

/**
 * Resolves winner index for current frame.
 *
 * @param birds - Population birds.
 * @param includeAliveOnly - When true, ignores dead birds.
 * @returns Winner index, or `-1` when unavailable.
 */
export function resolveFramePrimaryWinnerIndex(
  birds: BrowserPopulationBirdLike[],
  includeAliveOnly: boolean,
): number {
  // Step 1: Collect winner candidates, optionally excluding eliminated birds.
  const candidateIndexes = birds
    .map((bird, birdIndex) => ({ bird, birdIndex }))
    .filter(({ bird }) => (includeAliveOnly ? !bird.done : true));

  // Step 2: Return sentinel when no candidates are available.
  if (candidateIndexes.length === 0) {
    return -1;
  }

  // Step 3: Keep only birds tied on the highest pipes-passed score.
  const maxPipesPassed = candidateIndexes.reduce(
    (bestPipesPassed, { bird }) => Math.max(bestPipesPassed, bird.pipesPassed),
    Number.NEGATIVE_INFINITY,
  );

  const filteredByPipes = candidateIndexes.filter(
    ({ bird }) => bird.pipesPassed >= maxPipesPassed,
  );

  // Step 4: Tie-break by frames survived, then pipes passed, then stable index order.
  const winnerEntry = filteredByPipes.toSorted((leftEntry, rightEntry) => {
    if (rightEntry.bird.framesSurvived !== leftEntry.bird.framesSurvived) {
      return rightEntry.bird.framesSurvived - leftEntry.bird.framesSurvived;
    }
    if (rightEntry.bird.pipesPassed !== leftEntry.bird.pipesPassed) {
      return rightEntry.bird.pipesPassed - leftEntry.bird.pipesPassed;
    }
    return leftEntry.birdIndex - rightEntry.birdIndex;
  })[0];

  // Step 5: Return winner index with safe fallback.
  return winnerEntry?.birdIndex ?? -1;
}

/**
 * Checks whether at least one bird remains alive.
 *
 * @param birds - Population birds.
 * @returns True when any bird is alive.
 */
export function hasAliveBirds(birds: BrowserPopulationBirdLike[]): boolean {
  return birds.some((bird) => !bird.done);
}
