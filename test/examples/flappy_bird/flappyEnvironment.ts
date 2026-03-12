/**
 * Public environment facade for the Flappy Bird example.
 *
 * Educational note:
 * This file presents the Flappy world as a small environment API: create state,
 * step it forward, and read observations. That makes it the most convenient
 * entrypoint when you want to treat the game as a control problem rather than a
 * rendering demo.
 *
 * For background reading, the Wikipedia article on "Markov decision process" is
 * a useful mental model for why the example separates state transition,
 * observation extraction, and action selection.
 */
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
 * This root-level helper exists as a convenience export. The underlying spawn
 * logic lives in the shared simulation layer so browser playback and evaluation
 * use the same world-generation rules.
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
 * Keeping consecutive gap centers locally smooth prevents the environment from
 * generating visually unfair jumps that a feed-forward policy could not react to
 * consistently.
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
