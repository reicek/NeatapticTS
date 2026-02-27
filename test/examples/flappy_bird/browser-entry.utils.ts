import {
  FLAPPY_DIFFICULTY_RAMP_PIPES,
  FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
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
  FLAPPY_BIRD_X_PX,
  FLAPPY_BIRD_RADIUS_PX,
} from './constants';

export interface BrowserPopulationBirdLike {
  done: boolean;
  pipesPassed: number;
  framesSurvived: number;
}

export interface BrowserPopulationPipeLike {
  xPx: number;
  gapCenterYPx: number;
  gapSizePx: number;
}

export interface BrowserDifficultyProfile {
  pipeGapPx: number;
  pipeSpeedPxPerFrame: number;
  pipeSpawnIntervalFrames: number;
}

export interface ViewportInfo {
  offsetXPx: number;
  offsetYPx: number;
  scale: number;
}

export interface RngLike {
  nextInt: (min: number, max: number) => number;
}

export function resolveWorldViewport(canvas: HTMLCanvasElement): ViewportInfo {
  const scale = canvas.height / FLAPPY_WORLD_HEIGHT_PX;
  const contentHeightPx = FLAPPY_WORLD_HEIGHT_PX * scale;

  return {
    offsetXPx: 0,
    offsetYPx: (canvas.height - contentHeightPx) * 0.5,
    scale,
  };
}

export function resolveVisibleWorldWidthPx(canvas: HTMLCanvasElement): number {
  const viewportScale = canvas.height / Math.max(1, FLAPPY_WORLD_HEIGHT_PX);
  return canvas.width / Math.max(0.001, viewportScale);
}

export function resolvePipeSpawnXPx(visibleWorldWidthPx: number): number {
  return visibleWorldWidthPx + FLAPPY_PIPE_WIDTH_PX;
}

export function resolveObservationVector(
  birdYPx: number,
  velocityYPxPerFrame: number,
  pipes: BrowserPopulationPipeLike[],
  visibleWorldWidthPx: number,
  difficultyProfile: BrowserDifficultyProfile,
  activeSpawnIntervalFrames: number,
): number[] {
  const [nextPipe, secondPipe] = resolveUpcomingPipes(pipes);
  const normalizedBirdY = clamp01(birdYPx / FLAPPY_WORLD_HEIGHT_PX);
  const normalizedVelocity = clamp(
    velocityYPxPerFrame / FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
    -1,
    1,
  );
  const distanceToPipePx = nextPipe
    ? nextPipe.xPx + FLAPPY_PIPE_WIDTH_PX - FLAPPY_BIRD_X_PX
    : visibleWorldWidthPx;
  const normalizedDistance = clamp01(distanceToPipePx / visibleWorldWidthPx);
  const gapCenterYPx = nextPipe?.gapCenterYPx ?? FLAPPY_WORLD_HEIGHT_PX * 0.5;
  const gapHalfPx = (nextPipe?.gapSizePx ?? FLAPPY_PIPE_GAP_PX) * 0.5;
  const gapTopNormalized = clamp01(
    (gapCenterYPx - gapHalfPx) / FLAPPY_WORLD_HEIGHT_PX,
  );
  const gapBottomNormalized = clamp01(
    (gapCenterYPx + gapHalfPx) / FLAPPY_WORLD_HEIGHT_PX,
  );
  const normalizedDeltaToGap = clamp(
    (birdYPx - gapCenterYPx) / FLAPPY_WORLD_HEIGHT_PX,
    -1,
    1,
  );

  const distanceToSecondPipePx = secondPipe
    ? secondPipe.xPx + FLAPPY_PIPE_WIDTH_PX - FLAPPY_BIRD_X_PX
    : visibleWorldWidthPx;
  const normalizedDistanceToSecondPipe = clamp01(
    distanceToSecondPipePx / visibleWorldWidthPx,
  );

  const secondGapCenterYPx = secondPipe?.gapCenterYPx ?? gapCenterYPx;
  const normalizedDeltaToSecondGap = clamp(
    (birdYPx - secondGapCenterYPx) / FLAPPY_WORLD_HEIGHT_PX,
    -1,
    1,
  );

  const estimatedFramesToNextPipe =
    distanceToPipePx / Math.max(0.001, difficultyProfile.pipeSpeedPxPerFrame);
  const normalizedTimeToNextPipe = clamp01(
    1 - estimatedFramesToNextPipe / Math.max(1, activeSpawnIntervalFrames),
  );

  const distanceToGapCenterPx = Math.abs(birdYPx - gapCenterYPx);
  const normalizedNextGapClearance = clamp(
    (gapHalfPx - distanceToGapCenterPx) / Math.max(1, gapHalfPx),
    -1,
    1,
  );

  const requiredVerticalVelocityToNextGapPxPerFrame = clamp(
    (gapCenterYPx - birdYPx) / Math.max(1, estimatedFramesToNextPipe),
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
    (secondGapCenterYPx - gapCenterYPx) / FLAPPY_WORLD_HEIGHT_PX,
    -1,
    1,
  );

  return [
    normalizedBirdY,
    normalizedVelocity,
    normalizedDistance,
    normalizedDeltaToGap,
    gapTopNormalized,
    gapBottomNormalized,
    normalizedDistanceToSecondPipe,
    normalizedDeltaToSecondGap,
    normalizedTimeToNextPipe,
    normalizedNextGapClearance,
    normalizedRequiredVerticalVelocityToNextGap,
    normalizedNextToSecondGapTransition,
  ];
}

export function resolveUpcomingPipes(
  pipes: BrowserPopulationPipeLike[],
): [BrowserPopulationPipeLike | undefined, BrowserPopulationPipeLike | undefined] {
  const birdFrontX = FLAPPY_BIRD_X_PX - FLAPPY_BIRD_RADIUS_PX;
  const upcomingPipes = pipes.filter(
    (pipe) => pipe.xPx + FLAPPY_PIPE_WIDTH_PX >= birdFrontX,
  );
  return [upcomingPipes[0], upcomingPipes[1]];
}

export function resolveFlapDecision(rawOutputs: unknown): boolean {
  if (
    Array.isArray(rawOutputs) &&
    typeof rawOutputs[0] === 'number' &&
    typeof rawOutputs[1] === 'number'
  ) {
    return rawOutputs[1] > rawOutputs[0];
  }

  if (Array.isArray(rawOutputs) && typeof rawOutputs[0] === 'number') {
    return rawOutputs[0] > 0.5;
  }
  return typeof rawOutputs === 'number' ? rawOutputs > 0.5 : false;
}

export function resolveAliveBirdCount(birds: BrowserPopulationBirdLike[]): number {
  return birds.filter((bird) => !bird.done).length;
}

export function resolveLeaderPipesPassed(
  birds: BrowserPopulationBirdLike[],
): number {
  return birds.reduce(
    (leaderPipesPassed, bird) => Math.max(leaderPipesPassed, bird.pipesPassed),
    0,
  );
}

export function resolveFramePrimaryWinnerIndex(
  birds: BrowserPopulationBirdLike[],
  includeAliveOnly: boolean,
): number {
  const candidateIndexes = birds
    .map((bird, birdIndex) => ({ bird, birdIndex }))
    .filter(({ bird }) => (includeAliveOnly ? !bird.done : true));

  if (candidateIndexes.length === 0) {
    return -1;
  }

  const maxPipesPassed = candidateIndexes.reduce(
    (bestPipesPassed, { bird }) => Math.max(bestPipesPassed, bird.pipesPassed),
    Number.NEGATIVE_INFINITY,
  );

  const filteredByPipes = candidateIndexes.filter(
    ({ bird }) => bird.pipesPassed >= maxPipesPassed,
  );

  const winnerEntry = filteredByPipes.toSorted((leftEntry, rightEntry) => {
    if (rightEntry.bird.framesSurvived !== leftEntry.bird.framesSurvived) {
      return rightEntry.bird.framesSurvived - leftEntry.bird.framesSurvived;
    }
    if (rightEntry.bird.pipesPassed !== leftEntry.bird.pipesPassed) {
      return rightEntry.bird.pipesPassed - leftEntry.bird.pipesPassed;
    }
    return leftEntry.birdIndex - rightEntry.birdIndex;
  })[0];

  return winnerEntry?.birdIndex ?? -1;
}

export function hasAliveBirds(birds: BrowserPopulationBirdLike[]): boolean {
  return birds.some((bird) => !bird.done);
}

export function sampleGapCenterY(rng: RngLike): number {
  return rng.nextInt(
    FLAPPY_PIPE_GAP_CENTER_MIN_Y_PX,
    FLAPPY_PIPE_GAP_CENTER_MAX_Y_PX,
  );
}

export function resolveNextSpawnGapCenterY(
  previousGapCenterYPx: number,
  rng: RngLike,
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

export function createBirdColor(birdIndex: number, totalBirds: number): string {
  const neonBirdPalette = [
    '#00e5ff',
    '#00ff66',
    '#ff9a2e',
    '#00b7ff',
    '#ff5cff',
    '#9fffff',
    '#a6ff00',
    '#ff4a8d',
  ];
  if (totalBirds <= 0) return neonBirdPalette[0];
  return neonBirdPalette[birdIndex % neonBirdPalette.length];
}

export function resolveNextSpawnGapSize(
  previousSpawnGapPx: number | undefined,
  difficultyProfile: BrowserDifficultyProfile,
  rng: RngLike,
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

export function resolveNextSpawnIntervalFrames(
  previousSpawnIntervalFrames: number | undefined,
  difficultyProfile: BrowserDifficultyProfile,
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

export function resolveDifficultyProfile(pipesPassed: number): BrowserDifficultyProfile {
  const normalizedDifficultyProgress = clamp(
    pipesPassed / Math.max(1, FLAPPY_DIFFICULTY_RAMP_PIPES),
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

export function interpolateValue(
  startValue: number,
  endValue: number,
  progress: number,
): number {
  return startValue + (endValue - startValue) * progress;
}

export function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

export function clamp01(value: number): number {
  return clamp(value, 0, 1);
}
