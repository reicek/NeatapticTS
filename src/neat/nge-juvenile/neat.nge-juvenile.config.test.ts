/**
 * Sibling unit tests for `neat.nge-juvenile.config.ts`.
 *
 * These tests focus on `resolveGrowStabilizeConfig`, the pure resolver that
 * fills a partial `NgeGrowStabilizeConfig` with canonical defaults. They live
 * next to the config module so `npm run quality:folder` can validate the new
 * file without pulling in unrelated `src/architecture/` imports.
 *
 * Single-expect rule enforced throughout; each `it()` block contains exactly
 * one top-level `expect(...)` call.
 */

import { resolveBufferPoolMaxPooledBytes } from '../../acceleration/acceleration.constants';
import { resolveGrowStabilizeConfig } from './neat.nge-juvenile.config';
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
 * Build the canonical default resolved config for direct comparison.
 * Mirrors the fallback logic in `resolveGrowStabilizeConfig`.
 */
function buildExpectedDefaultConfig(): NgeGrowStabilizeConfig {
  return {
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
    bufferPoolMaxPooledBytes: resolveBufferPoolMaxPooledBytes({
      nodeCount: NGE_MAX_NODE_CAPACITY,
      variantCount: 1,
    }),
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
}

describe('resolveGrowStabilizeConfig', () => {
  describe('default resolution', () => {
    it('returns canonical defaults when called with no arguments', () => {
      // Act
      const config = resolveGrowStabilizeConfig();

      // Assert
      expect(config).toEqual(buildExpectedDefaultConfig());
    });

    it('returns canonical defaults when called with an empty partial object', () => {
      // Act
      const config = resolveGrowStabilizeConfig({});

      // Assert
      expect(config).toEqual(buildExpectedDefaultConfig());
    });
  });

  describe('field overrides', () => {
    it('applies grow-stabilize numeric overrides', () => {
      // Arrange
      const partial: Partial<NgeGrowStabilizeConfig> = {
        maxStructuralEditsPerStep: 3,
        maxNodes: 500,
        maxConnections: 2_000,
        maxEpisodicSlots: 7,
        plateauWindowSize: 7,
        plateauVarianceThreshold: 0.05,
        minStabilizationTicks: 3,
        maxStabilizationTicks: 10,
        weightMutationRate: 0.5,
        weightMutationMagnitude: 0.2,
        largeNetworkNodeThreshold: 500,
        growthThrottleBaseIntervalTicks: 6,
        maxForwardPassSamples: 8,
        improvementThreshold: 0.05,
        mutationCooldownTicks: 7,
        rollbackCooldownTicks: 8,
        lifecycleCooldownWindowCount: 9,
        biasMutationRate: 0.2,
        biasMutationMagnitude: 0.3,
      };

      // Act
      const config = resolveGrowStabilizeConfig(partial);

      // Assert
      expect(config).toMatchObject(partial);
    });

    it('applies a custom moduleId override', () => {
      // Arrange
      const moduleId = 'test:custom-module';

      // Act
      const config = resolveGrowStabilizeConfig({ moduleId });

      // Assert
      expect(config.moduleId).toBe(moduleId);
    });

    it('applies lifecycle stage band overrides', () => {
      // Arrange
      const partial: Partial<NgeGrowStabilizeConfig> = {
        babyNodeThreshold: 200,
        juvenileNodeThreshold: 800,
        babyVariantCount: 4,
        juvenileVariantCount: 6,
        adultVariantCount: 3,
        babyGrowthCadence: 0.9,
        adultGrowthCadence: 0.3,
        babyStabilizationIntensity: 0.4,
        adultStabilizationIntensity: 0.6,
        babyMutationMagnitude: 0.25,
        adultMutationMagnitude: 0.15,
      };

      // Act
      const config = resolveGrowStabilizeConfig(partial);

      // Assert
      expect(config).toMatchObject(partial);
    });

    it('applies acceleration disable flags', () => {
      // Arrange
      const partial: Partial<NgeGrowStabilizeConfig> = {
        disableGPU: true,
        disableWorkers: true,
      };

      // Act
      const config = resolveGrowStabilizeConfig(partial);

      // Assert
      expect(config).toMatchObject(partial);
    });

    it('preserves a provided accelerationConfig object in the resolved config', () => {
      // Arrange
      const accelerationConfig = { backend: 'cpu' as const };

      // Act
      const config = resolveGrowStabilizeConfig({ accelerationConfig });

      // Assert
      expect(config.accelerationConfig).toBe(accelerationConfig);
    });
  });

  describe('buffer pool budget', () => {
    it('defaults bufferPoolMaxPooledBytes from the acceleration resolver', () => {
      // Act
      const config = resolveGrowStabilizeConfig();

      // Assert
      expect(config.bufferPoolMaxPooledBytes).toBe(
        resolveBufferPoolMaxPooledBytes({
          nodeCount: NGE_MAX_NODE_CAPACITY,
          variantCount: 1,
        }),
      );
    });

    it('applies a custom bufferPoolMaxPooledBytes override', () => {
      // Arrange
      const bufferPoolMaxPooledBytes = 8_388_608;

      // Act
      const config = resolveGrowStabilizeConfig({ bufferPoolMaxPooledBytes });

      // Assert
      expect(config.bufferPoolMaxPooledBytes).toBe(bufferPoolMaxPooledBytes);
    });
  });

  describe('edge cases', () => {
    it('falls back to defaults when partial fields are explicitly undefined', () => {
      // Arrange — every field is supplied but set to undefined
      const partial: Partial<NgeGrowStabilizeConfig> = {
        maxNodes: undefined,
        weightMutationRate: undefined,
        moduleId: undefined,
        disableGPU: undefined,
        bufferPoolMaxPooledBytes: undefined,
      };

      // Act
      const config = resolveGrowStabilizeConfig(partial);

      // Assert — undefined values are ignored and defaults win
      expect(config).toEqual(buildExpectedDefaultConfig());
    });
  });
});
