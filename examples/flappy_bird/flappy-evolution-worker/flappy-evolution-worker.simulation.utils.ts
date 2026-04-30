import type Network from '../../../src/architecture/network';
import {
  resolveNextSpawnGapSize,
  resolveNextSpawnIntervalFrames,
  sampleGapCenterY,
} from '../browser-entry/browser-entry.spawn.utils';
import type { RngLike } from '../browser-entry/browser-entry.types';
import { resolvePipeSpawnXPx } from '../browser-entry/browser-entry.viewport.utils';
import {
  createSharedObservationMemoryState,
  resolveAdaptiveDifficultyProfile,
} from '../flappy.simulation.shared.utils';
import type { WorkerPlaybackState } from './flappy-evolution-worker.types';

/**
 * Creates initial playback state for a population of networks.
 *
 * Educational note:
 * This is the ownership boundary for worker playback initialization. The frame
 * simulation service mutates the returned state on every step, but only this
 * helper decides how a fresh population is placed into the world at time zero.
 *
 * @example
 * ```ts
 * const playbackState = createWorkerPopulationRenderState(
 *   currentPopulation,
 *   rng,
 *   1280,
 *   720,
 * );
 * ```
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
  const initialDifficultyProfile = resolveAdaptiveDifficultyProfile(0, 1);
  const initialGapSizePx = resolveNextSpawnGapSize(
    undefined,
    initialDifficultyProfile,
    rng,
  );
  const initialGapCenterYPx = sampleGapCenterY(
    rng,
    initialGapSizePx,
    initialVisibleWorldHeightPx,
  );
  const initialSpawnIntervalFrames = resolveNextSpawnIntervalFrames(
    undefined,
    initialDifficultyProfile,
  );

  const birds = networks.map((network) => {
    // Step 1: Reset carried state before the browser starts a fresh playback session.
    network.clear?.();

    // Step 2: Seed fresh per-bird world and observation-memory state.
    return {
      network,
      observationMemoryState: createSharedObservationMemoryState(),
      yPx: initialVisibleWorldHeightPx * 0.5,
      velocityYPxPerFrame: 0,
      pipesPassed: 0,
      framesSurvived: 0,
      passedPipeIds: new Set<number>(),
      done: false,
    };
  });

  return {
    frameIndex: 0,
    cumulativePipeTravelPx: 0,
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
