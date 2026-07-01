import { NGE_DNA } from './neat.nge-dna';

describe('NGE_DNA schema alignment (P5)', () => {
  describe('seed policy shorthand', () => {
    it('accepts queen-weighted seed-policy string and normalizes to canonical object', () => {
      // Arrange
      const dna = new NGE_DNA({
        reproductionPolicy: {
          seedPolicy: 'queen-weighted' as const,
        },
      });

      // Act
      const canonicalSeedPolicy = dna.toCanonical().reproductionPolicy.seedPolicy;

      // Assert
      expect(canonicalSeedPolicy).toEqual({
        siblingsDifferBySeed: true,
        twinsAllowed: false,
      });
    });

    it('keeps canonical seed policy stable after serialization roundtrip', () => {
      // Arrange
      const originalDna = new NGE_DNA({
        reproductionPolicy: {
          seedPolicy: 'queen-weighted' as const,
        },
      });

      // Act
      const roundtrippedDna = NGE_DNA.deserialize(originalDna.serialize());
      const canonicalSeedPolicy =
        roundtrippedDna.toCanonical().reproductionPolicy.seedPolicy;

      // Assert
      expect(canonicalSeedPolicy).toEqual({
        siblingsDifferBySeed: true,
        twinsAllowed: false,
      });
    });
  });

  describe('assigned region strategy', () => {
    it('accepts non-overlapping region strategy through the constructor', () => {
      // Arrange
      const dna = new NGE_DNA({
        reproductionPolicy: {
          assignedRegionStrategy: 'non-overlapping' as const,
        },
      });

      // Act
      const strategy = dna.toCanonical().reproductionPolicy.assignedRegionStrategy;

      // Assert
      expect(strategy).toBe('non-overlapping');
    });
  });
});
