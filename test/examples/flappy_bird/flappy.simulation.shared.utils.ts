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
