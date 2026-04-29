import {
  calculateMaxSamplePairs,
  collectAncestorIds,
  computeAverageDistance,
  computePairDistances,
  createInitialQueue,
  hasMinimumPopulation,
  normalizeParentIds,
  sampleGenomePairs,
} from './lineage.core';
import type { GenomeIndexPair, GenomeLike } from './lineage.types';

describe('lineage core chapter', () => {
  describe('normalizeParentIds', () => {
    describe('given the genome has no recorded parents', () => {
      it('returns an empty parent list', () => {
        // Arrange
        const genome = createGenome(7);

        // Act
        const parentIds = normalizeParentIds(genome);

        // Assert
        expect(parentIds).toEqual([]);
      });
    });
  });

  describe('createInitialQueue', () => {
    describe('given the direct parent ids include a missing ancestor', () => {
      it('resolves known genome refs and leaves unknown ones undefined', () => {
        // Arrange
        const population = [createGenome(2, [1])];

        // Act
        const queueEntries = createInitialQueue([2, 99], population);

        // Assert
        expect(queueEntries).toEqual([
          { ancestorId: 2, depth: 1, genomeRef: population[0] },
          { ancestorId: 99, depth: 1, genomeRef: undefined },
        ]);
      });
    });
  });

  describe('collectAncestorIds', () => {
    describe('given the ancestry chain extends beyond the recent-depth window', () => {
      it('collects only the recent ancestors up to depth four', () => {
        // Arrange
        const population = [
          createGenome(1),
          createGenome(2, [1]),
          createGenome(3, [2]),
          createGenome(4, [3]),
          createGenome(5, [4]),
        ];
        const queueEntries = createInitialQueue([5], population);

        // Act
        const ancestorIds = collectAncestorIds(queueEntries, population);

        // Assert
        expect(Array.from(ancestorIds)).toEqual([5, 4, 3, 2]);
      });
    });

    describe('given a queued ancestor cannot be resolved from the population', () => {
      it('keeps traversal stable and includes the unresolved ancestor id', () => {
        // Arrange
        const population = [createGenome(2, [99])];
        const queueEntries = createInitialQueue([2], population);

        // Act
        const ancestorIds = collectAncestorIds(queueEntries, population);

        // Assert
        expect(Array.from(ancestorIds)).toEqual([2, 99]);
      });
    });
  });

  describe('population guards', () => {
    describe('given the caller checks the pair budget for small and large populations', () => {
      it('enforces the minimum size and the sample cap correctly', () => {
        // Act
        const result = {
          largePopulationCap: calculateMaxSamplePairs(20),
          smallPopulationHasPairs: hasMinimumPopulation(1),
          smallPopulationPairCap: calculateMaxSamplePairs(3),
        };

        // Assert
        expect(result).toEqual({
          largePopulationCap: 30,
          smallPopulationHasPairs: false,
          smallPopulationPairCap: 3,
        });
      });
    });
  });

  describe('sampleGenomePairs', () => {
    describe('given the RNG keeps choosing the same first index', () => {
      it('nudges the second index so each pair remains distinct', () => {
        // Arrange
        const rngFactory = () => () => 0;

        // Act
        const pairs = sampleGenomePairs(2, 3, rngFactory);

        // Assert
        expect(pairs).toEqual([
          { firstIndex: 0, secondIndex: 1 },
          { firstIndex: 0, secondIndex: 1 },
        ]);
      });
    });

    describe('given the RNG returns different values for each sampled index', () => {
      it('keeps the generated pair unchanged when the indices are already distinct', () => {
        // Arrange
        const randomValues = [0.1, 0.8];
        const rngFactory = () => {
          let readIndex = 0;
          return () => {
            const nextValue =
              randomValues[Math.min(readIndex, randomValues.length - 1)];
            readIndex += 1;
            return nextValue;
          };
        };

        // Act
        const pairs = sampleGenomePairs(1, 3, rngFactory);

        // Assert
        expect(pairs).toEqual([{ firstIndex: 0, secondIndex: 2 }]);
      });
    });
  });

  describe('computePairDistances', () => {
    describe('given one sampled pair has no ancestor data and another has no overlap', () => {
      it('skips the empty pair and returns the non-overlap distance', () => {
        // Arrange
        const population = [createGenome(1), createGenome(2), createGenome(3)];
        const pairs: GenomeIndexPair[] = [
          { firstIndex: 0, secondIndex: 1 },
          { firstIndex: 1, secondIndex: 2 },
        ];
        const ancestorSets = new Map<number, Set<number>>([
          [1, new Set()],
          [2, new Set()],
          [3, new Set([7, 8])],
        ]);

        // Act
        const distances = computePairDistances(
          pairs,
          population,
          (genome) => ancestorSets.get(genome._id) ?? new Set(),
        );

        // Assert
        expect(distances).toEqual([1]);
      });
    });

    describe('given the first ancestor set is larger and only partially overlaps', () => {
      it('returns the expected jaccard distance from the overlap count', () => {
        // Arrange
        const population = [createGenome(1), createGenome(2)];
        const pairs: GenomeIndexPair[] = [{ firstIndex: 0, secondIndex: 1 }];
        const ancestorSets = new Map<number, Set<number>>([
          [1, new Set([1, 2, 3])],
          [2, new Set([2, 4])],
        ]);

        // Act
        const distances = computePairDistances(
          pairs,
          population,
          (genome) => ancestorSets.get(genome._id) ?? new Set(),
        );

        // Assert
        expect(distances).toEqual([0.75]);
      });
    });
  });

  describe('computeAverageDistance', () => {
    describe('given the sampled pair distances include a rounded mean and an empty fallback', () => {
      it('returns the configured rounded mean and the zero fallback', () => {
        // Act
        const result = {
          emptyAverage: computeAverageDistance([]),
          roundedAverage: computeAverageDistance([0.1234, 0.9876]),
        };

        // Assert
        expect(result).toEqual({ emptyAverage: 0, roundedAverage: 0.555 });
      });
    });
  });
});

function createGenome(genomeId: number, parents?: number[]): GenomeLike {
  if (!parents || parents.length === 0) {
    return { _id: genomeId };
  }

  return { _id: genomeId, _parents: parents };
}
