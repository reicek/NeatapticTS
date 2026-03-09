/**
 * Public rollout compatibility facade.
 *
 * Keeping this file at the evaluation layer preserves the established import
 * path while the actual rollout orchestration lives behind the dedicated
 * rollout-owned module boundary.
 */
export { rolloutEpisode } from './rollout/evaluation.rollout.service';
