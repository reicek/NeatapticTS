/**
 * Public rollout compatibility facade.
 *
 * Keeping this file at the evaluation layer preserves the established import
 * path while the actual rollout orchestration lives behind the dedicated
 * rollout-owned module boundary.
 *
 * This is the public evaluation-layer shelf for callers that should not need to
 * know about the rollout subfolder layout.
 *
 * Minimal usage sketch:
 * ```ts
 * const result = rolloutEpisode(network, {
 *   seed: 123,
 *   normalizeFitness: true,
 * });
 * ```
 */
export { rolloutEpisode } from './rollout/evaluation.rollout.service';
