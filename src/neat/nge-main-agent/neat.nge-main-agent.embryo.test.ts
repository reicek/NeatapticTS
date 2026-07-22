/**
 * Green-phase test contracts for slice 04-main-embryo-build.
 *
 * Covers AC-406: the main-agent embryo builder integrates the deterministic
 * substrate coordinate allocator and the reproduction-mode policy while
 * keeping topology tier-capped and deterministic.
 *
 * Single-expect rule enforced. AAA structure in every test.
 */

import type { NgeMainAgentLifecycleConfig } from './neat.nge-main-agent.types';

import { buildMainAgentEmbryo } from './neat.nge-main-agent.embryo';
import { reproductionModeHysteresis } from '../nge-evolution/neat.nge-evolution.reproduction-mode';
import type { NgeReproductionPolicy } from '../nge-dna/neat.nge-dna.types';

const defaultConfig: NgeMainAgentLifecycleConfig = {
  seed: 42,
  maxNodes: 1024,
  maxEdges: 4096,
};

function createBasePolicy(
  overrides?: Partial<NgeReproductionPolicy>,
): NgeReproductionPolicy {
  return {
    mode: 'parthenogenesis',
    parthenogenesisMutationRate: 0.05,
    polyandricDroneCount: 2,
    polyandricDroneContributionFraction: 1,
    queenBias: 1,
    assignedRegionStrategy: 'roundRobin',
    modeIsEvolvable: true,
    seedPolicy: { siblingsDifferBySeed: true, twinsAllowed: false },
    ...overrides,
  };
}

describe('buildMainAgentEmbryo', () => {
  describe('coordinate allocator integration', () => {
    it('assigns a substrate coordinate to every archetype', () => {
      // Act
      const embryo = buildMainAgentEmbryo(defaultConfig);

      // Assert
      expect(embryo.archetypes.every((archetype) => archetype.coordinate)).toBe(
        true,
      );
    });

    it('assigns a zone id to every archetype', () => {
      // Act
      const embryo = buildMainAgentEmbryo(defaultConfig);

      // Assert
      expect(
        embryo.archetypes.every(
          (archetype) => archetype.zoneId && archetype.zoneId.length > 0,
        ),
      ).toBe(true);
    });

    it('keeps every allocated coordinate inside the unit cube', () => {
      // Act
      const embryo = buildMainAgentEmbryo(defaultConfig);

      // Assert
      expect(
        embryo.archetypes.every((archetype) =>
          archetype.coordinate.every(
            (axisValue: number) => axisValue >= 0 && axisValue <= 1,
          ),
        ),
      ).toBe(true);
    });

    it('produces three-dimensional coordinates for every archetype', () => {
      // Act
      const embryo = buildMainAgentEmbryo(defaultConfig);

      // Assert
      expect(
        embryo.archetypes.every(
          (archetype) => archetype.coordinate.length === 3,
        ),
      ).toBe(true);
    });

    it('rebuilds identical coordinates from the same config', () => {
      // Act
      const embryoA = buildMainAgentEmbryo(defaultConfig);
      const embryoB = buildMainAgentEmbryo(defaultConfig);

      // Assert
      expect(embryoA.archetypes.map((a) => a.coordinate)).toEqual(
        embryoB.archetypes.map((a) => a.coordinate),
      );
    });

    it('produces different coordinates for different seeds', () => {
      // Arrange
      const configA: NgeMainAgentLifecycleConfig = {
        ...defaultConfig,
        seed: 42,
      };
      const configB: NgeMainAgentLifecycleConfig = {
        ...defaultConfig,
        seed: 43,
      };

      // Act
      const embryoA = buildMainAgentEmbryo(configA);
      const embryoB = buildMainAgentEmbryo(configB);

      // Assert
      expect(embryoA.archetypes.map((a) => a.coordinate)).not.toEqual(
        embryoB.archetypes.map((a) => a.coordinate),
      );
    });

    it('assigns stable integer-formatted zone ids', () => {
      // Act
      const embryo = buildMainAgentEmbryo(defaultConfig);

      // Assert
      expect(
        embryo.archetypes.every((archetype) =>
          /^z:\d+:\d+:\d+$/.test(archetype.zoneId),
        ),
      ).toBe(true);
    });
  });

  describe('reproduction-mode policy integration', () => {
    it('starts with parthenogenesis as the default mode', () => {
      // Act
      const embryo = buildMainAgentEmbryo(defaultConfig);

      // Assert
      expect(embryo.reproductionMode).toBe('parthenogenesis');
    });

    it('declares the reproduction mode evolvable by default', () => {
      // Act
      const embryo = buildMainAgentEmbryo(defaultConfig);

      // Assert
      expect(embryo.modeIsEvolvable).toBe(true);
    });

    it('allows the hysteresis policy to update the evolvable mode', () => {
      // Arrange
      const embryo = buildMainAgentEmbryo(defaultConfig);
      const policy = createBasePolicy({
        mode: embryo.reproductionMode,
        modeIsEvolvable: embryo.modeIsEvolvable,
      });

      // Act
      const result = reproductionModeHysteresis({
        policy,
        generationPressure: [
          {
            generation: 1,
            isDominating: false,
            isStruggling: true,
            isStalemate: false,
          },
          {
            generation: 2,
            isDominating: false,
            isStruggling: true,
            isStalemate: false,
          },
          {
            generation: 3,
            isDominating: false,
            isStruggling: true,
            isStalemate: false,
          },
        ],
      });

      // Assert
      expect(result.policy.mode).toBe('polyandric');
    });
  });

  describe('topology budget', () => {
    it('caps node count at the embryo tier budget', () => {
      // Act
      const embryo = buildMainAgentEmbryo(defaultConfig);

      // Assert
      expect(embryo.nodeCount).toBeLessThanOrEqual(64);
    });

    it('caps edge count at the embryo tier budget', () => {
      // Act
      const embryo = buildMainAgentEmbryo(defaultConfig);

      // Assert
      expect(embryo.edgeCount).toBeLessThanOrEqual(256);
    });

    it('remains deterministic for the same config', () => {
      // Act
      const embryoA = buildMainAgentEmbryo(defaultConfig);
      const embryoB = buildMainAgentEmbryo(defaultConfig);

      // Assert
      expect(embryoA).toEqual(embryoB);
    });
  });
});
