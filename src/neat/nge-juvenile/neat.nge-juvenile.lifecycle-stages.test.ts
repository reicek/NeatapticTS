/**
 * Red-phase test contracts for Phase 3 slice A3 — baby/juvenile/adult lifecycle.
 *
 * These tests define the expected behavior for the lifecycle-stages module
 * (`src/neat/nge-juvenile/neat.nge-juvenile.lifecycle-stages.ts`) before the
 * implementation exists. They cover:
 *
 * 1. `resolveVariantCount` returns 16 variants ≤1k nodes, ramps through 1k-4k,
 *    returns 2 >4k.
 * 2. Growth cadence and stabilization intensity differ by stage
 *    (baby = aggressive, adult = stability-focused).
 * 3. Baby phase uses higher mutation magnitude, adult uses lower.
 * 4. Lifecycle thresholds (node count bands) are config fields, not hardcoded.
 * 5. All values are defaults overridable via config.
 *
 * All tests fail because `./neat.nge-juvenile.lifecycle-stages` does not exist
 * yet. The failure reason is "missing implementation" (TS2307 Cannot find
 * module), not a syntax error or bad fixture.
 *
 * Single-expect rule enforced throughout. AAA structure in every test.
 */

import {
  resolveVariantCount,
  resolveLifecycleStage,
  resolveGrowthCadence,
  resolveStabilizationIntensity,
  resolveMutationMagnitude,
  type NgeLifecycleStage,
  type NgeLifecycleStageConfig,
} from './neat.nge-juvenile.lifecycle-stages';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Default config used when no overrides are supplied. */
const defaultConfig: NgeLifecycleStageConfig = {};

/** Baby-phase node count (well below the 1k threshold). */
const BABY_NODE_COUNT = 500;

/** Juvenile-phase node count (between 1k and 4k). */
const JUVENILE_NODE_COUNT = 2_000;

/** Adult-phase node count (above 4k). */
const ADULT_NODE_COUNT = 5_000;

describe('NGE lifecycle stages', () => {
  // -------------------------------------------------------------------------
  // 1. resolveVariantCount
  // -------------------------------------------------------------------------
  describe('resolveVariantCount', () => {
    it('returns 16 variants for networks with ≤1k nodes', () => {
      // Arrange — node count at the baby-phase boundary
      const nodeCount = 1_000;

      // Act
      const result = resolveVariantCount(nodeCount, defaultConfig);

      // Assert
      expect(result).toBe(16);
    });

    it('returns 2 variants for networks with >4k nodes', () => {
      // Arrange — node count above the juvenile threshold
      const nodeCount = ADULT_NODE_COUNT;

      // Act
      const result = resolveVariantCount(nodeCount, defaultConfig);

      // Assert
      expect(result).toBe(2);
    });

    it('ramps between 16 and 2 for node counts between 1k and 4k', () => {
      // Arrange — node count in the ramp zone
      const nodeCount = 2_500;

      // Act
      const result = resolveVariantCount(nodeCount, defaultConfig);

      // Assert — ramp value must be strictly between the two extremes
      expect(result).toBeGreaterThan(2);
      expect(result).toBeLessThan(16);
    });

    it('returns a value at or below 16 for node counts below 1k', () => {
      // Arrange — small network
      const nodeCount = BABY_NODE_COUNT;

      // Act
      const result = resolveVariantCount(nodeCount, defaultConfig);

      // Assert
      expect(result).toBe(16);
    });

    it('ramps through the second-half piecewise-linear segment for node counts above the ramp midpoint', () => {
      // Arrange — node count strictly between the ramp midpoint (2,500) and the juvenile threshold (4,000)
      const nodeCount = 3_000;

      // Act
      const result = resolveVariantCount(nodeCount, defaultConfig);

      // Assert — second-half ramp from juvenile variants (8) toward adult variants (2)
      // ratio = (3000 - 2500) / (4000 - 2500) = 1/3; lerp(8, 2, 1/3) = round(6) = 6
      expect(result).toBe(6);
    });
  });

  // -------------------------------------------------------------------------
  // 2. Growth cadence and stabilization intensity differ by stage
  // -------------------------------------------------------------------------
  describe('growth cadence by stage', () => {
    it('baby stage has higher (more aggressive) growth cadence than adult', () => {
      // Arrange — baby and adult stages resolved from node counts
      const babyStage = resolveLifecycleStage(BABY_NODE_COUNT, defaultConfig);
      const adultStage = resolveLifecycleStage(ADULT_NODE_COUNT, defaultConfig);

      // Act
      const babyCadence = resolveGrowthCadence(babyStage, defaultConfig);
      const adultCadence = resolveGrowthCadence(adultStage, defaultConfig);

      // Assert — baby should grow more aggressively
      expect(babyCadence).toBeGreaterThan(adultCadence);
    });

    it('juvenile stage growth cadence is between baby and adult', () => {
      // Arrange — all three stages
      const babyStage = resolveLifecycleStage(BABY_NODE_COUNT, defaultConfig);
      const juvenileStage = resolveLifecycleStage(
        JUVENILE_NODE_COUNT,
        defaultConfig,
      );
      const adultStage = resolveLifecycleStage(ADULT_NODE_COUNT, defaultConfig);

      // Act
      const babyCadence = resolveGrowthCadence(babyStage, defaultConfig);
      const juvenileCadence = resolveGrowthCadence(
        juvenileStage,
        defaultConfig,
      );
      const adultCadence = resolveGrowthCadence(adultStage, defaultConfig);

      // Assert — juvenile cadence falls between baby and adult
      expect(juvenileCadence).toBeLessThan(babyCadence);
      expect(juvenileCadence).toBeGreaterThan(adultCadence);
    });
  });

  describe('stabilization intensity by stage', () => {
    it('adult stage has higher (more stability-focused) stabilization intensity than baby', () => {
      // Arrange — baby and adult stages
      const babyStage = resolveLifecycleStage(BABY_NODE_COUNT, defaultConfig);
      const adultStage = resolveLifecycleStage(ADULT_NODE_COUNT, defaultConfig);

      // Act
      const babyIntensity = resolveStabilizationIntensity(
        babyStage,
        defaultConfig,
      );
      const adultIntensity = resolveStabilizationIntensity(
        adultStage,
        defaultConfig,
      );

      // Assert — adult focuses more on stability
      expect(adultIntensity).toBeGreaterThan(babyIntensity);
    });

    it('juvenile stage stabilization intensity is the midpoint of baby and adult', () => {
      // Arrange — juvenile stage
      const juvenileStage: NgeLifecycleStage = 'juvenile';

      // Act
      const result = resolveStabilizationIntensity(
        juvenileStage,
        defaultConfig,
      );

      // Assert — midpoint of baby intensity (0.3) and adult intensity (0.7) is 0.5
      expect(result).toBe(0.5);
    });
  });

  // -------------------------------------------------------------------------
  // 3. Baby phase uses higher mutation magnitude, adult uses lower
  // -------------------------------------------------------------------------
  describe('mutation magnitude by stage', () => {
    it('baby stage has higher mutation magnitude than adult', () => {
      // Arrange — baby and adult stages
      const babyStage = resolveLifecycleStage(BABY_NODE_COUNT, defaultConfig);
      const adultStage = resolveLifecycleStage(ADULT_NODE_COUNT, defaultConfig);

      // Act
      const babyMagnitude = resolveMutationMagnitude(babyStage, defaultConfig);
      const adultMagnitude = resolveMutationMagnitude(
        adultStage,
        defaultConfig,
      );

      // Assert — baby uses higher magnitude for exploration
      expect(babyMagnitude).toBeGreaterThan(adultMagnitude);
    });

    it('juvenile stage mutation magnitude is between baby and adult', () => {
      // Arrange — all three stages
      const babyStage = resolveLifecycleStage(BABY_NODE_COUNT, defaultConfig);
      const juvenileStage = resolveLifecycleStage(
        JUVENILE_NODE_COUNT,
        defaultConfig,
      );
      const adultStage = resolveLifecycleStage(ADULT_NODE_COUNT, defaultConfig);

      // Act
      const babyMagnitude = resolveMutationMagnitude(babyStage, defaultConfig);
      const juvenileMagnitude = resolveMutationMagnitude(
        juvenileStage,
        defaultConfig,
      );
      const adultMagnitude = resolveMutationMagnitude(
        adultStage,
        defaultConfig,
      );

      // Assert — juvenile magnitude is between baby and adult
      expect(juvenileMagnitude).toBeLessThan(babyMagnitude);
      expect(juvenileMagnitude).toBeGreaterThan(adultMagnitude);
    });
  });

  // -------------------------------------------------------------------------
  // 4. Lifecycle thresholds are config fields, not hardcoded
  // -------------------------------------------------------------------------
  describe('lifecycle stage resolution', () => {
    it('returns baby for node count at or below babyNodeThreshold', () => {
      // Arrange — node count at the default baby threshold
      const nodeCount = 1_000;

      // Act
      const stage = resolveLifecycleStage(nodeCount, defaultConfig);

      // Assert
      expect(stage).toBe('baby');
    });

    it('returns juvenile for node count between baby and juvenile thresholds', () => {
      // Arrange — node count in the juvenile band
      const nodeCount = JUVENILE_NODE_COUNT;

      // Act
      const stage = resolveLifecycleStage(nodeCount, defaultConfig);

      // Assert
      expect(stage).toBe('juvenile');
    });

    it('returns adult for node count above juvenileNodeThreshold', () => {
      // Arrange — node count above the juvenile threshold
      const nodeCount = ADULT_NODE_COUNT;

      // Act
      const stage = resolveLifecycleStage(nodeCount, defaultConfig);

      // Assert
      expect(stage).toBe('adult');
    });

    it('uses custom babyNodeThreshold from config to shift the band boundary', () => {
      // Arrange — custom threshold of 500; a 750-node network should be juvenile
      const customConfig: NgeLifecycleStageConfig = {
        babyNodeThreshold: 500,
        juvenileNodeThreshold: 4_000,
      };
      const nodeCount = 750;

      // Act
      const stage = resolveLifecycleStage(nodeCount, customConfig);

      // Assert — with a lowered baby threshold, 750 nodes is juvenile
      expect(stage).toBe('juvenile');
    });

    it('uses custom juvenileNodeThreshold from config to shift the adult boundary', () => {
      // Arrange — custom juvenile threshold of 2_000; a 3_000-node network is adult
      const customConfig: NgeLifecycleStageConfig = {
        babyNodeThreshold: 1_000,
        juvenileNodeThreshold: 2_000,
      };
      const nodeCount = 3_000;

      // Act
      const stage = resolveLifecycleStage(nodeCount, customConfig);

      // Assert — with a lowered juvenile threshold, 3k nodes is adult
      expect(stage).toBe('adult');
    });
  });

  // -------------------------------------------------------------------------
  // 5. All values are defaults overridable via config
  // -------------------------------------------------------------------------
  describe('config overrides', () => {
    it('resolveVariantCount honors custom babyVariantCount', () => {
      // Arrange — override baby variant count to 24
      const customConfig: NgeLifecycleStageConfig = {
        babyVariantCount: 24,
      };
      const nodeCount = BABY_NODE_COUNT;

      // Act
      const result = resolveVariantCount(nodeCount, customConfig);

      // Assert
      expect(result).toBe(24);
    });

    it('resolveVariantCount honors custom adultVariantCount', () => {
      // Arrange — override adult variant count to 1
      const customConfig: NgeLifecycleStageConfig = {
        adultVariantCount: 1,
      };
      const nodeCount = ADULT_NODE_COUNT;

      // Act
      const result = resolveVariantCount(nodeCount, customConfig);

      // Assert
      expect(result).toBe(1);
    });

    it('resolveGrowthCadence honors custom baby growth cadence override', () => {
      // Arrange — custom baby cadence lower than default adult cadence
      const customConfig: NgeLifecycleStageConfig = {
        babyGrowthCadence: 0.1,
      };
      const babyStage: NgeLifecycleStage = 'baby';

      // Act
      const result = resolveGrowthCadence(babyStage, customConfig);

      // Assert — overridden value is used
      expect(result).toBe(0.1);
    });

    it('resolveStabilizationIntensity honors custom adult stabilization intensity', () => {
      // Arrange — custom adult stabilization intensity
      const customConfig: NgeLifecycleStageConfig = {
        adultStabilizationIntensity: 0.95,
      };
      const adultStage: NgeLifecycleStage = 'adult';

      // Act
      const result = resolveStabilizationIntensity(adultStage, customConfig);

      // Assert
      expect(result).toBe(0.95);
    });

    it('resolveMutationMagnitude honors custom baby mutation magnitude', () => {
      // Arrange — custom baby mutation magnitude
      const customConfig: NgeLifecycleStageConfig = {
        babyMutationMagnitude: 0.3,
      };
      const babyStage: NgeLifecycleStage = 'baby';

      // Act
      const result = resolveMutationMagnitude(babyStage, customConfig);

      // Assert
      expect(result).toBe(0.3);
    });

    it('resolveMutationMagnitude honors custom adult mutation magnitude', () => {
      // Arrange — custom adult mutation magnitude
      const customConfig: NgeLifecycleStageConfig = {
        adultMutationMagnitude: 0.01,
      };
      const adultStage: NgeLifecycleStage = 'adult';

      // Act
      const result = resolveMutationMagnitude(adultStage, customConfig);

      // Assert
      expect(result).toBe(0.01);
    });
  });
});
