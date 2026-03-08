/**
 * Rollout orchestration module.
 *
 * This file will host the internal rollout orchestration entry while the
 * public evaluation-level service remains a stable compatibility facade.
 */
import {
  createRolloutEpisodeRuntimeState,
  finalizeRolloutEpisodeState,
  resolveRolloutEpisodeContext,
  runRolloutEpisodeLoop,
} from './evaluation.rollout.services';
import { composeRolloutEpisodeResult } from './evaluation.rollout.utils';
import type {
  FlappyEpisodeResult,
  FlappyNetworkLike,
  FlappyRolloutOptions,
} from '../evaluation.types';

/**
 * Roll out an episode and return details.
 *
 * @param network - Genome/network to evaluate.
 * @param rolloutOptions - Optional rollout controls.
 * @returns Episode result details.
 */
export function rolloutEpisode(
  network: FlappyNetworkLike,
  rolloutOptions: FlappyRolloutOptions = {},
): FlappyEpisodeResult {
  // Step 1: Resolve rollout configuration and initialize mutable runtime state.
  const rolloutEpisodeContext = resolveRolloutEpisodeContext(
    network,
    rolloutOptions,
  );
  const rolloutEpisodeRuntimeState = createRolloutEpisodeRuntimeState(
    rolloutEpisodeContext,
  );

  // Step 2: Simulate frames until the episode terminates or the frame budget is exhausted.
  runRolloutEpisodeLoop(
    network,
    rolloutEpisodeContext,
    rolloutEpisodeRuntimeState,
  );

  // Step 3: Apply timeout termination when the loop exhausted the frame budget.
  finalizeRolloutEpisodeState(
    rolloutEpisodeContext,
    rolloutEpisodeRuntimeState,
  );

  // Step 4: Compose the episode result from the final state and accumulated fitness channels.
  return composeRolloutEpisodeResult(
    rolloutEpisodeContext,
    rolloutEpisodeRuntimeState,
  );
}
