/**
 * Red-phase test contracts for Phase 4 Step 01 — combat-pressure reproduction
 * mode hysteresis policy.
 *
 * Covers AC-406: reproductionModeHysteresis computes a 3-generation majority-vote
 * mode selection and writes NgeReproductionPolicy.mode only when modeIsEvolvable
 * is true.
 *
 * All tests fail because the imported source modules do not exist yet. The
 * expected failure reason is TS2307 "Cannot find module".
 *
 * Single-expect rule enforced. AAA structure in every test.
 */

import type { NgeReproductionPolicy } from '../nge-dna/neat.nge-dna.types';

import {
  reproductionModeHysteresis,
  type ReproductionModeHysteresisInput,
} from './neat.nge-evolution.reproduction-mode';

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

describe('reproductionModeHysteresis', () => {
  describe('3-generation majority vote', () => {
    it('selects parthenogenesis when the last three generations are dominating', () => {
      // Arrange — three consecutive dominant signals
      const input: ReproductionModeHysteresisInput = {
        policy: createBasePolicy(),
        generationPressure: [
          {
            generation: 1,
            isDominating: true,
            isStruggling: false,
            isStalemate: false,
          },
          {
            generation: 2,
            isDominating: true,
            isStruggling: false,
            isStalemate: false,
          },
          {
            generation: 3,
            isDominating: true,
            isStruggling: false,
            isStalemate: false,
          },
        ],
      };

      // Act
      const result = reproductionModeHysteresis(input);

      // Assert
      expect(result.mode).toBe('parthenogenesis');
    });

    it('selects polyandric when the last three generations are struggling', () => {
      // Arrange — three consecutive struggling signals
      const input: ReproductionModeHysteresisInput = {
        policy: createBasePolicy(),
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
      };

      // Act
      const result = reproductionModeHysteresis(input);

      // Assert
      expect(result.mode).toBe('polyandric');
    });

    it('selects sexual when the last three generations are stalemate', () => {
      // Arrange — three consecutive stalemate signals
      const input: ReproductionModeHysteresisInput = {
        policy: createBasePolicy(),
        generationPressure: [
          {
            generation: 1,
            isDominating: false,
            isStruggling: false,
            isStalemate: true,
          },
          {
            generation: 2,
            isDominating: false,
            isStruggling: false,
            isStalemate: true,
          },
          {
            generation: 3,
            isDominating: false,
            isStruggling: false,
            isStalemate: true,
          },
        ],
      };

      // Act
      const result = reproductionModeHysteresis(input);

      // Assert
      expect(result.mode).toBe('sexual');
    });

    it('applies majority vote over exactly three generations', () => {
      // Arrange — two dominating, one struggling: dominating wins by majority
      const input: ReproductionModeHysteresisInput = {
        policy: createBasePolicy(),
        generationPressure: [
          {
            generation: 1,
            isDominating: true,
            isStruggling: false,
            isStalemate: false,
          },
          {
            generation: 2,
            isDominating: true,
            isStruggling: false,
            isStalemate: false,
          },
          {
            generation: 3,
            isDominating: false,
            isStruggling: true,
            isStalemate: false,
          },
        ],
      };

      // Act
      const result = reproductionModeHysteresis(input);

      // Assert
      expect(result.mode).toBe('parthenogenesis');
    });

    it('falls back to parthenogenesis when no generation pressure is supplied', () => {
      // Arrange
      const input: ReproductionModeHysteresisInput = {
        policy: createBasePolicy(),
        generationPressure: [],
      };

      // Act
      const result = reproductionModeHysteresis(input);

      // Assert
      expect(result.mode).toBe('parthenogenesis');
    });
  });

  describe('modeIsEvolvable guard', () => {
    it('writes the selected mode when modeIsEvolvable is true', () => {
      // Arrange
      const input: ReproductionModeHysteresisInput = {
        policy: createBasePolicy({ modeIsEvolvable: true }),
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
      };

      // Act
      const result = reproductionModeHysteresis(input);

      // Assert
      expect(result.policy.mode).toBe('polyandric');
    });

    it('does not change the mode when modeIsEvolvable is false', () => {
      // Arrange
      const input: ReproductionModeHysteresisInput = {
        policy: createBasePolicy({
          mode: 'parthenogenesis',
          modeIsEvolvable: false,
        }),
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
      };

      // Act
      const result = reproductionModeHysteresis(input);

      // Assert
      expect(result.policy.mode).toBe('parthenogenesis');
    });
  });

  describe('result contract', () => {
    it('returns a result with the original policy updated', () => {
      // Arrange
      const input: ReproductionModeHysteresisInput = {
        policy: createBasePolicy(),
        generationPressure: [
          {
            generation: 1,
            isDominating: true,
            isStruggling: false,
            isStalemate: false,
          },
          {
            generation: 2,
            isDominating: true,
            isStruggling: false,
            isStalemate: false,
          },
          {
            generation: 3,
            isDominating: true,
            isStruggling: false,
            isStalemate: false,
          },
        ],
      };

      // Act
      const result = reproductionModeHysteresis(input);

      // Assert
      expect(result.mode).toBeDefined();
    });
  });
});
