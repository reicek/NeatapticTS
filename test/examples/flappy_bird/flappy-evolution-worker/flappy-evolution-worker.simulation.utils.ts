import type Network from '../../../../src/architecture/network';
import { clamp } from '../browser-entry/browser-entry.math.utils';
import {
  commitObservationMemoryStep,
  resolveFlapDecision,
  resolveObservationVector,
} from '../browser-entry/browser-entry.observation.utils';
import {
  createBirdColor,
  resolveDifficultyProfile,
  resolveNextSpawnGapCenterY,
  resolveNextSpawnGapSize,
  resolveNextSpawnIntervalFrames,
  sampleGapCenterY,
} from '../browser-entry/browser-entry.spawn.utils';
import type { RngLike } from '../browser-entry/browser-entry.types';
import { resolvePipeSpawnXPx } from '../browser-entry/browser-entry.viewport.utils';
import { createSharedObservationMemoryState } from '../flappy.simulation.shared.utils';
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
import type { WorkerPlaybackState } from './flappy-evolution-worker.types';

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
 * Creates initial playback state for a population of networks.
 *
 * @param networks - Population to visualize.
 * @param rng - Deterministic random source.
 * @param initialVisibleWorldWidthPx - Initial viewport width from host.
 * @param initialVisibleWorldHeightPx - Initial viewport height from host.
 * @returns Fresh mutable playback state.
 */
export function createWorkerPopulationRenderState(
  networks: Network[],
  rng: RngLike,
  initialVisibleWorldWidthPx: number,
  initialVisibleWorldHeightPx: number,
): WorkerPlaybackState {
  const initialDifficultyProfile = resolveDifficultyProfile(0);
  const initialGapCenterYPx = sampleGapCenterY(
    rng,
    initialVisibleWorldHeightPx,
  );
  const initialGapSizePx = resolveNextSpawnGapSize(
    undefined,
    initialDifficultyProfile,
    rng,
  );
  const initialSpawnIntervalFrames = resolveNextSpawnIntervalFrames(
    undefined,
    initialDifficultyProfile,
  );

  const birds = networks.map((network, networkIndex) => ({
    network,
    color: createBirdColor(networkIndex, networks.length),
    observationMemoryState: createSharedObservationMemoryState(),
    yPx: initialVisibleWorldHeightPx * 0.5,
    velocityYPxPerFrame: 0,
    pipesPassed: 0,
    framesSurvived: 0,
    passedPipeIds: new Set<number>(),
    done: false,
  }));

  return {
    frameIndex: 0,
    visibleWorldWidthPx: initialVisibleWorldWidthPx,
    visibleWorldHeightPx: initialVisibleWorldHeightPx,
    nextPipeId: 2,
    lastSpawnedPipeGapPx: initialGapSizePx,
    lastSpawnedPipeGapCenterYPx: initialGapCenterYPx,
    lastSpawnedPipeSpawnIntervalFrames: initialSpawnIntervalFrames,
    framesUntilNextPipeSpawn: initialSpawnIntervalFrames,
    pipes: [
      {
        id: 1,
        xPx: resolvePipeSpawnXPx(initialVisibleWorldWidthPx),
        gapCenterYPx: initialGapCenterYPx,
        gapSizePx: initialGapSizePx,
      },
    ],
    birds,
  };
}

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
  difficultyProfile: {
    pipeGapPx: number;
    pipeSpeedPxPerFrame: number;
    pipeSpawnIntervalFrames: number;
  },
): number {
  const controlSubstepCount = Math.max(1, FLAPPY_CONTROL_SUBSTEPS_PER_FRAME);
  const controlSubstepDelta = 1 / controlSubstepCount;
  let activationCallsThisFrame = 0;

  renderState.birds.forEach((bird) => {
    if (bird.done) return;
    bird.framesSurvived += 1;
  });

  for (
    let controlSubstepIndex = 0;
    controlSubstepIndex < controlSubstepCount;
    controlSubstepIndex++
  ) {
    renderState.birds.forEach((bird) => {
      if (bird.done) return;

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
      const outputs = bird.network.activate(
        observation.observationVector,
      ) as unknown;
      if (FLAPPY_ENABLE_RUNTIME_INSTRUMENTATION) {
        activationCallsThisFrame += 1;
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

    renderState.birds.forEach((bird) => {
      if (bird.done) return;

      bird.velocityYPxPerFrame = clamp(
        bird.velocityYPxPerFrame +
          FLAPPY_GRAVITY_PX_PER_FRAME2 * controlSubstepDelta,
        -Infinity,
        FLAPPY_MAX_FALL_SPEED_PX_PER_FRAME,
      );
      bird.yPx += bird.velocityYPxPerFrame * controlSubstepDelta;
    });

    renderState.pipes.forEach((pipe) => {
      pipe.xPx -= difficultyProfile.pipeSpeedPxPerFrame * controlSubstepDelta;
    });

    const cameraLeftXPx = resolveCameraLeftXPx(renderState.visibleWorldWidthPx);
    renderState.pipes = renderState.pipes.filter(
      (pipe) => pipe.xPx + FLAPPY_PIPE_WIDTH_PX > cameraLeftXPx,
    );

    renderState.framesUntilNextPipeSpawn -= controlSubstepDelta;
    if (renderState.framesUntilNextPipeSpawn <= 0) {
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

    renderState.birds.forEach((bird) => {
      if (bird.done) return;

      const birdTop = bird.yPx - FLAPPY_BIRD_RADIUS_PX;
      const birdBottom = bird.yPx + FLAPPY_BIRD_RADIUS_PX;

      if (birdTop <= 0 || birdBottom >= renderState.visibleWorldHeightPx) {
        bird.done = true;
        bird.doneReason = 'out_of_bounds';
        return;
      }

      const birdLeft = FLAPPY_BIRD_X_PX - FLAPPY_BIRD_RADIUS_PX;
      const birdRight = FLAPPY_BIRD_X_PX + FLAPPY_BIRD_RADIUS_PX;

      for (const pipe of renderState.pipes) {
        const pipeLeft = pipe.xPx - FLAPPY_PIPE_COLLISION_SIDE_EXPAND_PX;
        const pipeRight =
          pipe.xPx +
          FLAPPY_PIPE_WIDTH_PX +
          FLAPPY_PIPE_COLLISION_SIDE_EXPAND_PX;
        const overlapsHorizontally =
          birdRight >= pipeLeft && birdLeft <= pipeRight;

        if (overlapsHorizontally) {
          const gapHalf = pipe.gapSizePx * 0.5;
          const gapTop =
            pipe.gapCenterYPx -
            gapHalf +
            FLAPPY_PIPE_COLLISION_ENTRANCE_EXPAND_PX;
          const gapBottom =
            pipe.gapCenterYPx +
            gapHalf -
            FLAPPY_PIPE_COLLISION_ENTRANCE_EXPAND_PX;
          const isInsideGap = birdTop >= gapTop && birdBottom <= gapBottom;

          if (!isInsideGap) {
            bird.done = true;
            bird.doneReason = 'collision';
            break;
          }
        }

        if (pipeRight < FLAPPY_BIRD_X_PX && !bird.passedPipeIds.has(pipe.id)) {
          bird.passedPipeIds.add(pipe.id);
          bird.pipesPassed += 1;
        }
      }
    });
  }

  renderState.frameIndex += 1;
  return activationCallsThisFrame;
}
