/**
 * Compatibility facade for the shared Flappy Bird simulation helpers.
 *
 * Educational note:
 * The Flappy example uses the same observation, spawning, math, and difficulty
 * logic in multiple places: Node-side evaluation, browser playback, and the
 * worker runtime. This facade keeps those shared concepts discoverable from the
 * root folder while the real implementation lives under `simulation-shared/`.
 *
 * If you are new to the example, start here to see the cross-cutting helpers,
 * then follow the re-export targets into the focused submodules.
 */
export { resolveAdaptiveDifficultyProfile } from './simulation-shared/simulation-shared.difficulty.utils';
export { resolveFlapDecision } from './simulation-shared/simulation-shared.control.utils';
export {
  resolveCoreObservationVectorFromFeatures,
  resolveObservationFeatures,
  resolveObservationVectorFromFeatures,
  resolveUpcomingPipes,
} from './simulation-shared/simulation-shared.observation.utils';
export {
  commitSharedObservationMemoryStep,
  createSharedObservationMemoryState,
  resolveTemporalObservationVector,
} from './simulation-shared/simulation-shared.memory.utils';
export {
  clamp01,
  clampValue,
  interpolateValue,
} from './simulation-shared/simulation-shared.math.utils';
export {
  computeMean,
  computePercentile,
  computePopulationStandardDeviation,
} from './simulation-shared/simulation-shared.statistics.utils';
export {
  resolveNextSpawnGapCenterY,
  resolveNextSpawnGapSize,
  resolveNextSpawnIntervalFrames,
  sampleGapCenterY,
} from './simulation-shared/simulation-shared.spawn.utils';
export type {
  SharedDifficultyProfile,
  SharedObservationFeatures,
  SharedObservationInput,
  SharedObservationMemoryState,
  SharedPipeLike,
  SharedRngLike,
} from './simulation-shared/simulation-shared.types';
