/**
 * Resolved configuration for the NGE juvenile grow-stabilize cycle.
 *
 * This module owns `resolveGrowStabilizeConfig`, a pure helper that fills in
 * every optional field of `NgeGrowStabilizeConfig` with canonical defaults.
 * Keeping the resolver in its own file breaks the import cycle between the
 * grow-stabilize orchestrator (which calls `applyPlasticity`) and the
 * plasticity pass (which needs the resolved config defaults). Both
 * `grow-stabilize.ts` and `plasticity.ts` now import the resolver from here,
 * so neither depends on the other for config defaults.
 *
 * ```mermaid
 * flowchart TD
 *   GS["neat.nge-juvenile.grow-stabilize.ts"] -->|imports| Config["neat.nge-juvenile.config.ts"]
 *   Plasticity["neat.nge-juvenile.plasticity.ts"] -->|imports| Config
 *   GS -->|calls applyPlasticity| Plasticity
 * ```
 *
 * @module neat/nge-juvenile/config
 */

import { resolveBufferPoolMaxPooledBytes } from '../../acceleration/acceleration.constants';
import {
  NGE_GROW_STABILIZE_BIAS_MUTATION_MAGNITUDE,
  NGE_GROW_STABILIZE_BIAS_MUTATION_RATE,
  NGE_GROW_STABILIZE_DEFAULT_MAX_STRUCTURAL_EDITS_PER_STEP,
  NGE_GROW_STABILIZE_DEFAULT_MODULE_ID,
  NGE_GROW_STABILIZE_GROWTH_THROTTLE_BASE_INTERVAL_TICKS,
  NGE_GROW_STABILIZE_LARGE_NETWORK_NODE_THRESHOLD,
  NGE_GROW_STABILIZE_LIFECYCLE_COOLDOWN_WINDOW_COUNT,
  NGE_GROW_STABILIZE_MAX_EPISODIC_SLOTS,
  NGE_GROW_STABILIZE_MAX_FORWARD_PASS_SAMPLES,
  NGE_GROW_STABILIZE_MAX_STABILIZATION_TICKS,
  NGE_GROW_STABILIZE_MIN_STABILIZATION_TICKS,
  NGE_GROW_STABILIZE_MUTATION_COOLDOWN_TICKS,
  NGE_GROW_STABILIZE_PLATEAU_VARIANCE_THRESHOLD,
  NGE_GROW_STABILIZE_PLATEAU_WINDOW_SIZE,
  NGE_GROW_STABILIZE_ROLLBACK_COOLDOWN_TICKS,
  NGE_GROW_STABILIZE_WEIGHT_MUTATION_MAGNITUDE,
  NGE_GROW_STABILIZE_WEIGHT_MUTATION_RATE,
  NGE_LIFECYCLE_DEFAULT_ADULT_GROWTH_CADENCE,
  NGE_LIFECYCLE_DEFAULT_ADULT_MUTATION_MAGNITUDE,
  NGE_LIFECYCLE_DEFAULT_ADULT_STABILIZATION_INTENSITY,
  NGE_LIFECYCLE_DEFAULT_ADULT_VARIANT_COUNT,
  NGE_LIFECYCLE_DEFAULT_BABY_GROWTH_CADENCE,
  NGE_LIFECYCLE_DEFAULT_BABY_MUTATION_MAGNITUDE,
  NGE_LIFECYCLE_DEFAULT_BABY_NODE_THRESHOLD,
  NGE_LIFECYCLE_DEFAULT_BABY_STABILIZATION_INTENSITY,
  NGE_LIFECYCLE_DEFAULT_BABY_VARIANT_COUNT,
  NGE_LIFECYCLE_DEFAULT_JUVENILE_NODE_THRESHOLD,
  NGE_LIFECYCLE_DEFAULT_JUVENILE_VARIANT_COUNT,
  NGE_MAX_EDGE_CAPACITY,
  NGE_MAX_NODE_CAPACITY,
} from './neat.nge-juvenile.constants';
import type { NgeGrowStabilizeConfig } from './neat.nge-juvenile.types';

/**
 * Static default values for every non-computed field of
 * {@link NgeGrowStabilizeConfig}.
 *
 * Fields whose defaults are computed at call time (for example
 * `bufferPoolMaxPooledBytes`, which depends on the device cap) are intentionally
 * omitted and filled in by {@link createDefaultGrowStabilizeConfig}.
 */
const DEFAULT_GROW_STABILIZE_CONFIG: Omit<
  NgeGrowStabilizeConfig,
  'bufferPoolMaxPooledBytes' | 'accelerationConfig'
> = {
  maxStructuralEditsPerStep:
    NGE_GROW_STABILIZE_DEFAULT_MAX_STRUCTURAL_EDITS_PER_STEP,
  maxNodes: NGE_MAX_NODE_CAPACITY,
  maxConnections: NGE_MAX_EDGE_CAPACITY,
  maxEpisodicSlots: NGE_GROW_STABILIZE_MAX_EPISODIC_SLOTS,
  moduleId: NGE_GROW_STABILIZE_DEFAULT_MODULE_ID,
  plateauWindowSize: NGE_GROW_STABILIZE_PLATEAU_WINDOW_SIZE,
  plateauVarianceThreshold: NGE_GROW_STABILIZE_PLATEAU_VARIANCE_THRESHOLD,
  minStabilizationTicks: NGE_GROW_STABILIZE_MIN_STABILIZATION_TICKS,
  maxStabilizationTicks: NGE_GROW_STABILIZE_MAX_STABILIZATION_TICKS,
  weightMutationRate: NGE_GROW_STABILIZE_WEIGHT_MUTATION_RATE,
  weightMutationMagnitude: NGE_GROW_STABILIZE_WEIGHT_MUTATION_MAGNITUDE,
  largeNetworkNodeThreshold: NGE_GROW_STABILIZE_LARGE_NETWORK_NODE_THRESHOLD,
  growthThrottleBaseIntervalTicks:
    NGE_GROW_STABILIZE_GROWTH_THROTTLE_BASE_INTERVAL_TICKS,
  maxForwardPassSamples: NGE_GROW_STABILIZE_MAX_FORWARD_PASS_SAMPLES,
  improvementThreshold: 0.01,
  mutationCooldownTicks: NGE_GROW_STABILIZE_MUTATION_COOLDOWN_TICKS,
  rollbackCooldownTicks: NGE_GROW_STABILIZE_ROLLBACK_COOLDOWN_TICKS,
  lifecycleCooldownWindowCount:
    NGE_GROW_STABILIZE_LIFECYCLE_COOLDOWN_WINDOW_COUNT,
  biasMutationRate: NGE_GROW_STABILIZE_BIAS_MUTATION_RATE,
  biasMutationMagnitude: NGE_GROW_STABILIZE_BIAS_MUTATION_MAGNITUDE,
  babyNodeThreshold: NGE_LIFECYCLE_DEFAULT_BABY_NODE_THRESHOLD,
  juvenileNodeThreshold: NGE_LIFECYCLE_DEFAULT_JUVENILE_NODE_THRESHOLD,
  babyVariantCount: NGE_LIFECYCLE_DEFAULT_BABY_VARIANT_COUNT,
  juvenileVariantCount: NGE_LIFECYCLE_DEFAULT_JUVENILE_VARIANT_COUNT,
  adultVariantCount: NGE_LIFECYCLE_DEFAULT_ADULT_VARIANT_COUNT,
  babyGrowthCadence: NGE_LIFECYCLE_DEFAULT_BABY_GROWTH_CADENCE,
  adultGrowthCadence: NGE_LIFECYCLE_DEFAULT_ADULT_GROWTH_CADENCE,
  babyStabilizationIntensity:
    NGE_LIFECYCLE_DEFAULT_BABY_STABILIZATION_INTENSITY,
  adultStabilizationIntensity:
    NGE_LIFECYCLE_DEFAULT_ADULT_STABILIZATION_INTENSITY,
  babyMutationMagnitude: NGE_LIFECYCLE_DEFAULT_BABY_MUTATION_MAGNITUDE,
  adultMutationMagnitude: NGE_LIFECYCLE_DEFAULT_ADULT_MUTATION_MAGNITUDE,
  disableGPU: false,
  disableWorkers: false,
};

/**
 * Build a fully defaulted grow-stabilize config, including the computed
 * `bufferPoolMaxPooledBytes` field.
 *
 * @returns A config with every field set to its canonical default.
 */
function createDefaultGrowStabilizeConfig(): NgeGrowStabilizeConfig {
  return {
    ...DEFAULT_GROW_STABILIZE_CONFIG,
    bufferPoolMaxPooledBytes: resolveBufferPoolMaxPooledBytes({
      nodeCount: NGE_MAX_NODE_CAPACITY,
      variantCount: 1,
    }),
  };
}

/**
 * Resolve a partial grow-stabilize config with sensible canonical default values.
 *
 * @param partial - Caller-supplied config overrides.
 * @returns Fully resolved config.
 *
 * @example
 * ```ts
 * const config = resolveGrowStabilizeConfig({ maxNodes: 500, weightMutationRate: 0.5 });
 * console.log(config.maxNodes); // 500
 * console.log(config.plateauWindowSize); // 5 (default)
 * ```
 */
export function resolveGrowStabilizeConfig(
  partial?: Partial<NgeGrowStabilizeConfig>,
): NgeGrowStabilizeConfig {
  return { ...createDefaultGrowStabilizeConfig(), ...partial };
}
