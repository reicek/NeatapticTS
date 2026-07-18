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
 * Resolve a partial grow-stabilize config with sensible defaults.
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
  return {
    maxStructuralEditsPerStep:
      partial?.maxStructuralEditsPerStep ??
      NGE_GROW_STABILIZE_DEFAULT_MAX_STRUCTURAL_EDITS_PER_STEP,
    maxNodes: partial?.maxNodes ?? NGE_MAX_NODE_CAPACITY,
    maxConnections: partial?.maxConnections ?? NGE_MAX_EDGE_CAPACITY,
    maxEpisodicSlots:
      partial?.maxEpisodicSlots ?? NGE_GROW_STABILIZE_MAX_EPISODIC_SLOTS,
    moduleId: partial?.moduleId ?? NGE_GROW_STABILIZE_DEFAULT_MODULE_ID,
    ...(partial?.accelerationConfig
      ? { accelerationConfig: partial.accelerationConfig }
      : {}),
    plateauWindowSize:
      partial?.plateauWindowSize ?? NGE_GROW_STABILIZE_PLATEAU_WINDOW_SIZE,
    plateauVarianceThreshold:
      partial?.plateauVarianceThreshold ??
      NGE_GROW_STABILIZE_PLATEAU_VARIANCE_THRESHOLD,
    minStabilizationTicks:
      partial?.minStabilizationTicks ??
      NGE_GROW_STABILIZE_MIN_STABILIZATION_TICKS,
    maxStabilizationTicks:
      partial?.maxStabilizationTicks ??
      NGE_GROW_STABILIZE_MAX_STABILIZATION_TICKS,
    weightMutationRate:
      partial?.weightMutationRate ?? NGE_GROW_STABILIZE_WEIGHT_MUTATION_RATE,
    weightMutationMagnitude:
      partial?.weightMutationMagnitude ??
      NGE_GROW_STABILIZE_WEIGHT_MUTATION_MAGNITUDE,
    largeNetworkNodeThreshold:
      partial?.largeNetworkNodeThreshold ??
      NGE_GROW_STABILIZE_LARGE_NETWORK_NODE_THRESHOLD,
    growthThrottleBaseIntervalTicks:
      partial?.growthThrottleBaseIntervalTicks ??
      NGE_GROW_STABILIZE_GROWTH_THROTTLE_BASE_INTERVAL_TICKS,
    maxForwardPassSamples:
      partial?.maxForwardPassSamples ??
      NGE_GROW_STABILIZE_MAX_FORWARD_PASS_SAMPLES,
    improvementThreshold: partial?.improvementThreshold ?? 0.01,
    mutationCooldownTicks:
      partial?.mutationCooldownTicks ??
      NGE_GROW_STABILIZE_MUTATION_COOLDOWN_TICKS,
    rollbackCooldownTicks:
      partial?.rollbackCooldownTicks ??
      NGE_GROW_STABILIZE_ROLLBACK_COOLDOWN_TICKS,
    lifecycleCooldownWindowCount:
      partial?.lifecycleCooldownWindowCount ??
      NGE_GROW_STABILIZE_LIFECYCLE_COOLDOWN_WINDOW_COUNT,
    biasMutationRate:
      partial?.biasMutationRate ?? NGE_GROW_STABILIZE_BIAS_MUTATION_RATE,
    biasMutationMagnitude:
      partial?.biasMutationMagnitude ??
      NGE_GROW_STABILIZE_BIAS_MUTATION_MAGNITUDE,
    bufferPoolMaxPooledBytes:
      partial?.bufferPoolMaxPooledBytes ??
      resolveBufferPoolMaxPooledBytes({
        nodeCount: NGE_MAX_NODE_CAPACITY,
        variantCount: 1,
      }),
    // Lifecycle stage bands
    babyNodeThreshold:
      partial?.babyNodeThreshold ?? NGE_LIFECYCLE_DEFAULT_BABY_NODE_THRESHOLD,
    juvenileNodeThreshold:
      partial?.juvenileNodeThreshold ??
      NGE_LIFECYCLE_DEFAULT_JUVENILE_NODE_THRESHOLD,
    babyVariantCount:
      partial?.babyVariantCount ?? NGE_LIFECYCLE_DEFAULT_BABY_VARIANT_COUNT,
    juvenileVariantCount:
      partial?.juvenileVariantCount ??
      NGE_LIFECYCLE_DEFAULT_JUVENILE_VARIANT_COUNT,
    adultVariantCount:
      partial?.adultVariantCount ?? NGE_LIFECYCLE_DEFAULT_ADULT_VARIANT_COUNT,
    babyGrowthCadence:
      partial?.babyGrowthCadence ?? NGE_LIFECYCLE_DEFAULT_BABY_GROWTH_CADENCE,
    adultGrowthCadence:
      partial?.adultGrowthCadence ?? NGE_LIFECYCLE_DEFAULT_ADULT_GROWTH_CADENCE,
    babyStabilizationIntensity:
      partial?.babyStabilizationIntensity ??
      NGE_LIFECYCLE_DEFAULT_BABY_STABILIZATION_INTENSITY,
    adultStabilizationIntensity:
      partial?.adultStabilizationIntensity ??
      NGE_LIFECYCLE_DEFAULT_ADULT_STABILIZATION_INTENSITY,
    babyMutationMagnitude:
      partial?.babyMutationMagnitude ??
      NGE_LIFECYCLE_DEFAULT_BABY_MUTATION_MAGNITUDE,
    adultMutationMagnitude:
      partial?.adultMutationMagnitude ??
      NGE_LIFECYCLE_DEFAULT_ADULT_MUTATION_MAGNITUDE,
    // Acceleration defaults
    disableGPU: partial?.disableGPU ?? false,
    disableWorkers: partial?.disableWorkers ?? false,
  };
}
