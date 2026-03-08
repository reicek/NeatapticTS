import { clamp } from '../browser-entry/browser-entry.math.utils';
import {
  commitObservationMemoryStep,
  resolveFlapDecision,
  resolveObservationVector,
} from '../browser-entry/browser-entry.observation.utils';
import {
  resolveNextSpawnGapCenterY,
  resolveNextSpawnGapSize,
  resolveNextSpawnIntervalFrames,
} from '../browser-entry/browser-entry.spawn.utils';
import type { RngLike } from '../browser-entry/browser-entry.types';
import { resolvePipeSpawnXPx } from '../browser-entry/browser-entry.viewport.utils';
import type { SharedDifficultyProfile } from '../flappy.simulation.shared.utils';
import { FLAPPY_BIRD_VIEWPORT_X_RATIO } from '../constants/constants';
import {
  FLAPPY_BIRD_RADIUS_PX,
  FLAPPY_BIRD_X_PX,
  FLAPPY_PIPE_COLLISION_ENTRANCE_EXPAND_PX,
  FLAPPY_PIPE_COLLISION_SIDE_EXPAND_PX,
  FLAPPY_CONTROL_SUBSTEPS_PER_FRAME,
  FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION,
  FLAPPY_FLAP_VELOCITY_PX_PER_FRAME,
  FLAPPY_GRAVITY_PX_PER_FRAME2,
  FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
  FLAPPY_PIPE_WIDTH_PX,
} from '../constants/constants';
import type {
  WorkerPlaybackState,
  WorkerPopulationBird,
  WorkerPopulationPipe,
} from './flappy-evolution-worker.types';
import type { WorkerPlaybackFrameContext } from './flappy-evolution-worker.simulation.types';

/**
 * Advances the whole population simulation by one logical frame.
 *
 * @param renderState - Mutable simulation state.
 * @param rng - Deterministic random source for spawn variation.
 * @param difficultyProfile - Active dynamic difficulty profile.
 * @returns Number of policy activation calls made in this frame.
 */
export function stepWorkerPopulationFrame(
  renderState: WorkerPlaybackState,
  rng: RngLike,
  difficultyProfile: SharedDifficultyProfile,
): number {
  const controlSubstepCount = Math.max(1, FLAPPY_CONTROL_SUBSTEPS_PER_FRAME);
  const frameContext: WorkerPlaybackFrameContext = {
    renderState,
    rng,
    difficultyProfile,
    controlSubstepDelta: 1 / controlSubstepCount,
    cameraLeftXPx: resolveCameraLeftXPx(renderState.visibleWorldWidthPx),
    birdLeftXPx: FLAPPY_BIRD_X_PX - FLAPPY_BIRD_RADIUS_PX,
    birdRightXPx: FLAPPY_BIRD_X_PX + FLAPPY_BIRD_RADIUS_PX,
  };
  let activationCallsThisFrame = 0;

  // Step 1: Advance the per-frame survival counters for living birds.
  incrementLivingBirdFrameCounters(renderState);

  for (
    let controlSubstepIndex = 0;
    controlSubstepIndex < controlSubstepCount;
    controlSubstepIndex++
  ) {
    // Step 2: Resolve one control/physics/collision substep.
    activationCallsThisFrame += runWorkerPopulationControlSubstep(frameContext);
  }

  // Step 3: Publish the completed logical frame index.
  renderState.frameIndex += 1;
  return activationCallsThisFrame;
}

/**
 * Resolves the current left-edge of the visible world in world-space pixels.
 *
 * @param visibleWorldWidthPx - Current visible world width.
 * @returns Left edge x-position in world coordinates.
 */
function resolveCameraLeftXPx(visibleWorldWidthPx: number): number {
  const clampedVisibleWorldWidthPx = Math.max(1, visibleWorldWidthPx);
  const desiredBirdScreenXPx =
    clampedVisibleWorldWidthPx * FLAPPY_BIRD_VIEWPORT_X_RATIO;
  return FLAPPY_BIRD_X_PX - desiredBirdScreenXPx;
}

/**
 * Increments survival counters for birds that remain active at frame start.
 *
 * @param renderState - Mutable playback state.
 * @returns Nothing.
 */
function incrementLivingBirdFrameCounters(
  renderState: WorkerPlaybackState,
): void {
  renderState.birds.forEach((bird) => {
    if (bird.done) {
      return;
    }

    bird.framesSurvived += 1;
  });
}

/**
 * Advances one control substep of the worker playback simulation.
 *
 * @param frameContext - Shared frame context for this logical frame.
 * @returns Number of activation calls performed in the substep.
 */
function runWorkerPopulationControlSubstep(
  frameContext: WorkerPlaybackFrameContext,
): number {
  const activationCallsThisSubstep = resolveBirdControlActions(frameContext);

  advanceBirdPhysics(frameContext);
  advancePipes(frameContext);
  spawnPipeIfNeeded(frameContext);
  resolveBirdTerminationAndProgress(frameContext);

  return activationCallsThisSubstep;
}

/**
 * Runs policy evaluation and commits the resulting observation memory updates.
 *
 * @param frameContext - Shared frame context for this logical frame.
 * @returns Number of activation calls performed in the substep.
 */
function resolveBirdControlActions(
  frameContext: WorkerPlaybackFrameContext,
): number {
  const { difficultyProfile, renderState } = frameContext;
  let activationCallsThisSubstep = 0;

  renderState.birds.forEach((bird) => {
    if (bird.done) {
      return;
    }

    const observation = resolveObservationVector(
      bird.yPx,
      bird.velocityYPxPerFrame,
      renderState.pipes,
      renderState.visibleWorldWidthPx,
      renderState.visibleWorldHeightPx,
      difficultyProfile,
      renderState.lastSpawnedPipeSpawnIntervalFrames,
      bird.observationMemoryState,
    );
    const outputs = bird.network.activate(observation.observationVector);
    if (FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION) {
      activationCallsThisSubstep += 1;
    }

    const shouldFlap = resolveFlapDecision(outputs);
    commitObservationMemoryStep(
      bird.observationMemoryState,
      observation.observationFeatures,
      shouldFlap,
    );

    if (shouldFlap) {
      bird.velocityYPxPerFrame = FLAPPY_FLAP_VELOCITY_PX_PER_FRAME;
    }
  });

  return activationCallsThisSubstep;
}

/**
 * Integrates bird velocity and vertical motion for one control substep.
 *
 * @param frameContext - Shared frame context for this logical frame.
 * @returns Nothing.
 */
function advanceBirdPhysics(frameContext: WorkerPlaybackFrameContext): void {
  const { controlSubstepDelta, renderState } = frameContext;

  renderState.birds.forEach((bird) => {
    if (bird.done) {
      return;
    }

    bird.velocityYPxPerFrame = clamp(
      bird.velocityYPxPerFrame +
        FLAPPY_GRAVITY_PX_PER_FRAME2 * controlSubstepDelta,
      -Infinity,
      FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
    );
    bird.yPx += bird.velocityYPxPerFrame * controlSubstepDelta;
  });
}

/**
 * Advances all visible pipes and culls those that have left the camera window.
 *
 * @param frameContext - Shared frame context for this logical frame.
 * @returns Nothing.
 */
function advancePipes(frameContext: WorkerPlaybackFrameContext): void {
  const { controlSubstepDelta, difficultyProfile, renderState, cameraLeftXPx } =
    frameContext;

  renderState.pipes.forEach((pipe) => {
    pipe.xPx -= difficultyProfile.pipeSpeedPxPerFrame * controlSubstepDelta;
  });

  renderState.pipes = renderState.pipes.filter(
    (pipe) => pipe.xPx + FLAPPY_PIPE_WIDTH_PX > cameraLeftXPx,
  );
}

/**
 * Spawns a new pipe when the substep budget crosses the spawn boundary.
 *
 * @param frameContext - Shared frame context for this logical frame.
 * @returns Nothing.
 */
function spawnPipeIfNeeded(frameContext: WorkerPlaybackFrameContext): void {
  const { controlSubstepDelta, difficultyProfile, renderState, rng } =
    frameContext;

  renderState.framesUntilNextPipeSpawn -= controlSubstepDelta;
  if (renderState.framesUntilNextPipeSpawn > 0) {
    return;
  }

  const nextGapSizePx = resolveNextSpawnGapSize(
    renderState.lastSpawnedPipeGapPx,
    difficultyProfile,
    rng,
  );
  const nextSpawnIntervalFrames = resolveNextSpawnIntervalFrames(
    renderState.lastSpawnedPipeSpawnIntervalFrames,
    difficultyProfile,
  );
  const nextGapCenterYPx = resolveNextSpawnGapCenterY(
    renderState.lastSpawnedPipeGapCenterYPx,
    rng,
    renderState.visibleWorldHeightPx,
  );

  renderState.pipes.push({
    id: renderState.nextPipeId++,
    xPx: resolvePipeSpawnXPx(renderState.visibleWorldWidthPx),
    gapCenterYPx: nextGapCenterYPx,
    gapSizePx: nextGapSizePx,
  });
  renderState.lastSpawnedPipeGapPx = nextGapSizePx;
  renderState.lastSpawnedPipeGapCenterYPx = nextGapCenterYPx;
  renderState.lastSpawnedPipeSpawnIntervalFrames = nextSpawnIntervalFrames;
  renderState.framesUntilNextPipeSpawn += nextSpawnIntervalFrames;
}

/**
 * Resolves bird deaths and passed-pipe progress after motion is applied.
 *
 * @param frameContext - Shared frame context for this logical frame.
 * @returns Nothing.
 */
function resolveBirdTerminationAndProgress(
  frameContext: WorkerPlaybackFrameContext,
): void {
  const { renderState } = frameContext;

  renderState.birds.forEach((bird) => {
    if (bird.done) {
      return;
    }

    if (resolveBirdOutOfBounds(bird, renderState.visibleWorldHeightPx)) {
      bird.done = true;
      bird.doneReason = 'out_of_bounds';
      return;
    }

    for (const pipe of renderState.pipes) {
      if (resolveBirdCollisionAgainstPipe(bird, pipe, frameContext)) {
        bird.done = true;
        bird.doneReason = 'collision';
        break;
      }

      commitPassedPipeProgress(bird, pipe);
    }
  });
}

/**
 * Resolves whether a bird has exceeded the vertical play area.
 *
 * @param bird - Mutable bird state.
 * @param visibleWorldHeightPx - Current visible world height.
 * @returns `true` when the bird is outside the vertical bounds.
 */
function resolveBirdOutOfBounds(
  bird: WorkerPopulationBird,
  visibleWorldHeightPx: number,
): boolean {
  const birdTop = bird.yPx - FLAPPY_BIRD_RADIUS_PX;
  const birdBottom = bird.yPx + FLAPPY_BIRD_RADIUS_PX;

  return birdTop <= 0 || birdBottom >= visibleWorldHeightPx;
}

/**
 * Resolves whether a bird collides with one pipe corridor during this substep.
 *
 * @param bird - Mutable bird state.
 * @param pipe - Pipe candidate to test.
 * @param frameContext - Shared frame context for this logical frame.
 * @returns `true` when the bird overlaps the pipe body instead of the gap.
 */
function resolveBirdCollisionAgainstPipe(
  bird: WorkerPopulationBird,
  pipe: WorkerPopulationPipe,
  frameContext: WorkerPlaybackFrameContext,
): boolean {
  const { birdLeftXPx, birdRightXPx } = frameContext;
  const birdTop = bird.yPx - FLAPPY_BIRD_RADIUS_PX;
  const birdBottom = bird.yPx + FLAPPY_BIRD_RADIUS_PX;
  const pipeLeft = pipe.xPx - FLAPPY_PIPE_COLLISION_SIDE_EXPAND_PX;
  const pipeRight =
    pipe.xPx + FLAPPY_PIPE_WIDTH_PX + FLAPPY_PIPE_COLLISION_SIDE_EXPAND_PX;
  const overlapsHorizontally =
    birdRightXPx >= pipeLeft && birdLeftXPx <= pipeRight;

  if (!overlapsHorizontally) {
    return false;
  }

  const gapHalf = pipe.gapSizePx * 0.5;
  const gapTop =
    pipe.gapCenterYPx - gapHalf + FLAPPY_PIPE_COLLISION_ENTRANCE_EXPAND_PX;
  const gapBottom =
    pipe.gapCenterYPx + gapHalf - FLAPPY_PIPE_COLLISION_ENTRANCE_EXPAND_PX;

  return !(birdTop >= gapTop && birdBottom <= gapBottom);
}

/**
 * Commits one passed-pipe progress increment for a bird when eligible.
 *
 * @param bird - Mutable bird state.
 * @param pipe - Pipe candidate to mark as passed.
 * @returns Nothing.
 */
function commitPassedPipeProgress(
  bird: WorkerPopulationBird,
  pipe: WorkerPopulationPipe,
): void {
  const pipeRight =
    pipe.xPx + FLAPPY_PIPE_WIDTH_PX + FLAPPY_PIPE_COLLISION_SIDE_EXPAND_PX;
  if (pipeRight >= FLAPPY_BIRD_X_PX || bird.passedPipeIds.has(pipe.id)) {
    return;
  }

  bird.passedPipeIds.add(pipe.id);
  bird.pipesPassed += 1;
}
