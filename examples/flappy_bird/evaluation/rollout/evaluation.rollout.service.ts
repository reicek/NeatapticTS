/**
 * Rollout orchestration module.
 *
 * This file will host the internal rollout orchestration entry while the
 * public evaluation-level service remains a stable compatibility facade.
 *
 * Educational note:
 * A rollout is one deterministic episode for one policy under one seed. This
 * module keeps that lifecycle readable: normalize inputs, create runtime state,
 * simulate until termination, then fold the result into a public episode report.
 *
 * That lifecycle matters because the trainer depends on rollouts being both
 * repeatable and interpretable. A rollout is not only "did the bird crash?"
 * It is the bridge between one seeded control problem and one scored episode
 * that can be compared fairly with other genomes.
 *
 * Rollout pipeline:
 * ```mermaid
 * flowchart LR
 *     Options["network + rollout options"] --> Context["normalize context"]
 *     Context --> Runtime["create runtime state"]
 *     Runtime --> Loop["observe -> act -> step -> shape"]
 *     Loop --> EarlyStop{"done or\nbudget exhausted?"}
 *     EarlyStop -->|No| Loop
 *     EarlyStop -->|Yes| Finalize["finalize timeout state"]
 *     Finalize --> Result["compose FlappyEpisodeResult"]
 * ```
 */
import {
  createRolloutEpisodeRuntimeState,
  finalizeRolloutEpisodeState,
  resolveRolloutEpisodeContext,
  runRolloutEpisodeLoop,
  runRolloutEpisodeLoopWithPredictor,
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
 * @example
 * ```ts
 * const result = rolloutEpisode(network, {
 *   seed: 123,
 *   normalizeFitness: true,
 *   maxFrames: 2_000,
 * });
 *
 * console.log(result.fitness, result.doneReason);
 * ```
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

/**
 * Roll out an episode against one async predictor callback.
 *
 * This browser-worker-oriented variant preserves the same seeded rollout and
 * shaping semantics as `rolloutEpisode(...)` while sourcing control decisions
 * from an async inference boundary such as `InferenceChannel.predict(...)`.
 *
 * @param options - Predictor callback plus optional rollout controls.
 * @returns Episode result details.
 */
export async function rolloutEpisodeWithPredictor(options: {
  predict: (observationVector: number[]) => Promise<unknown>;
  rolloutOptions?: FlappyRolloutOptions;
  networkId?: number;
}): Promise<FlappyEpisodeResult> {
  const rolloutOptions = options.rolloutOptions ?? {};

  // Step 1: Resolve rollout configuration and initialize mutable runtime state.
  const rolloutEpisodeContext = resolveRolloutEpisodeContext(
    {
      _id: options.networkId,
    },
    rolloutOptions,
  );
  const rolloutEpisodeRuntimeState = createRolloutEpisodeRuntimeState(
    rolloutEpisodeContext,
  );

  // Step 2: Simulate frames until the episode terminates or the frame budget is exhausted.
  await runRolloutEpisodeLoopWithPredictor(
    options.predict,
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
