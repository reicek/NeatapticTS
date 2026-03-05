import type { FlappyRng } from './rng';
import {
  resolveNextSpawnGapCenterY as resolveSharedNextSpawnGapCenterY,
  sampleGapCenterY as sampleSharedGapCenterY,
} from './flappy.simulation.shared.utils';

export type {
  FlappyBird,
  FlappyDifficultyScale,
  FlappyGameState,
  FlappyObservationFeatures,
  FlappyPipe,
} from './environment/environment.types';
export { createInitialFlappyState } from './environment/environment.state.service';
export {
  stepFlappyState,
  stepFlappyStateWithControlSubsteps,
} from './environment/environment.step.service';
export {
  getFlappyObservation,
  getFlappyObservationFeatures,
} from './environment/environment.observation.utils';

/**
 * Sample a gap center height inside configured bounds.
 *
 * @param rng - Random source.
 * @returns Gap center y coordinate (pixels).
 */
export function sampleGapCenterY(rng: FlappyRng): number {
  return sampleSharedGapCenterY(rng);
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
  return resolveSharedNextSpawnGapCenterY(previousGapCenterYPx, rng);
}
