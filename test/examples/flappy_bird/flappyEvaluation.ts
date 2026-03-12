/**
 * Public evaluation facade for the Flappy Bird example.
 *
 * Educational note:
 * The trainer and worker both need a compact way to score policies. This facade
 * exposes the evaluation surface without forcing callers to know the internal
 * rollout and seed-batching module layout.
 *
 * The important idea is variance control: the example compares genomes across
 * shared random seeds so evolution is rewarded for genuinely better behavior,
 * not for getting an unusually lucky pipe sequence.
 */
export {
  evaluateFlappyFitness,
  evaluateFlappyFitnessAcrossSeeds,
} from './evaluation/evaluation.fitness.utils';
export { rolloutEpisode } from './evaluation/evaluation.rollout.service';
export type {
  FlappyEpisodeResult,
  FlappyNetworkLike,
  FlappyRolloutOptions,
  FlappySeedBatchEvaluation,
} from './evaluation/evaluation.types';
