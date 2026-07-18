/**
 * Activity/bias-aware plasticity for the NGE juvenile module.
 *
 * This boundary provides a reward-gated, activity-aware weight and bias
 * adjustment that combines three signals:
 *
 * 1. **Activity signal** — per-connection activity (0..1) that scales the
 *    nudge magnitude.  Connections that fired more strongly get larger
 *    adjustments.
 * 2. **Reward-gated nudge** — a scalar reward signal (−1..+1) that gates
 *    whether the activity-driven nudge is positive (reinforce) or negative
 *    (decay).
 * 3. **Small random noise** — a stochastic perturbation drawn from the
 *    caller-supplied RNG so that exploration is still possible when
 *    activity and reward are both near zero.
 *
 * When `biasMutationRate > 0` the function also perturbs node biases using
 * the same three-signal model but scoped to per-node activity.
 *
 * Configuration defaults are imported from `neat.nge-juvenile.config.ts`, the
 * resolver that broke the `grow-stabilize.ts` ↔ `plasticity.ts` import cycle.
 *
 * @module neat/nge-juvenile/plasticity
 */

import type Network from '../../architecture/network';
import { resolveGrowStabilizeConfig } from './neat.nge-juvenile.config';
import type { NgeGrowStabilizeConfig } from './neat.nge-juvenile.types';

/**
 * Inputs driving one plasticity pass.
 *
 * @property activity - Per-connection activity keyed by connection
 *   innovation ID (0..1).  Missing entries are treated as zero activity.
 * @property rewardSignal - Scalar reward signal (−1..+1) gating the
 *   direction of the activity-driven nudge.  Positive reinforces, negative
 *   decays.
 * @property stabilizationIntensity - Scale factor for the activity-driven
 *   nudge, derived from the resolved lifecycle stage. Baby networks use a
 *   lower intensity (more exploration), adult networks use a higher intensity
 *   (more exploitation). Default: 1.0 (no scaling).
 * @property mutationMagnitude - Scale factor for the random noise component,
 *   derived from the resolved lifecycle stage. Baby networks use a higher
 *   magnitude (broader exploration), adult networks use a lower magnitude
 *   (fine-tuning). Default: 1.0 (no scaling).
 */
export interface NgePlasticityInput {
  activity: Map<number, number>;
  rewardSignal: number;
  /** Scale factor for the activity-driven nudge from lifecycle stage. Default: 1.0. */
  stabilizationIntensity?: number;
  /** Scale factor for the random noise component from lifecycle stage. Default: 1.0. */
  mutationMagnitude?: number;
}

/**
 * Apply activity/bias-aware plasticity to a network.
 *
 * Combines three signals for each connection: per-connection activity
 * scales the nudge, a scalar reward signal gates whether the nudge is
 * positive (reinforce) or negative (decay), and small random noise
 * enables exploration when activity and reward are both near zero.
 * When `biasMutationRate > 0`, also perturbs non-input node biases
 * with random noise.
 *
 * @param network - The network whose weights and biases will be adjusted.
 * @param random - Caller-supplied RNG function returning [0, 1).
 * @param input - Activity map and reward signal driving the plasticity pass.
 * @param config - Optional config overrides for mutation rates and magnitudes.
 * @returns The number of connections and nodes that were adjusted.
 *
 * @example
 * ```ts
 * const adjusted = applyPlasticity(network, Math.random, {
 *   activity: new Map([[conn.innovation, 0.8]]),
 *   rewardSignal: 1.0,
 * });
 * console.log(adjusted); // e.g. 5
 * ```
 */
export function applyPlasticity(
  network: Network,
  random: () => number,
  input: NgePlasticityInput,
  config?: Partial<NgeGrowStabilizeConfig>,
): number {
  const resolved = resolveGrowStabilizeConfig(config);
  const intensity = input.stabilizationIntensity ?? 1.0;
  const noiseScale = input.mutationMagnitude ?? 1.0;
  let adjustedCount = 0;

  // Step 1: Adjust connection weights using activity + reward + noise
  for (const connection of network.connections) {
    if (random() < resolved.weightMutationRate) {
      const noise =
        (random() * 2 - 1) * resolved.weightMutationMagnitude * noiseScale;
      const activityValue = input.activity.get(connection.innovation) ?? 0;
      const delta =
        activityValue *
          input.rewardSignal *
          resolved.weightMutationMagnitude *
          intensity +
        noise;
      connection.weight += delta;
      adjustedCount++;
    }
  }

  // Step 2: Adjust node biases using noise (skip input nodes)
  for (const node of network.nodes) {
    if (node.type === 'input') {
      continue;
    }
    if (random() < resolved.biasMutationRate) {
      const noise =
        (random() * 2 - 1) * resolved.biasMutationMagnitude * noiseScale;
      node.bias += noise;
      adjustedCount++;
    }
  }

  return adjustedCount;
}
