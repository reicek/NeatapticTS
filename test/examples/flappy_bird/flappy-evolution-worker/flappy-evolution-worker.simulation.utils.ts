import type Network from '../../../../src/architecture/network';
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
