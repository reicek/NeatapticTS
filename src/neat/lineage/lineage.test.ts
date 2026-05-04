import { buildAnc, computeAncestorUniqueness } from './lineage';
import type { GenomeLike, NeatLineageContext } from './lineage';

function createLineagePopulationChain(): GenomeLike[] {
  return Array.from({ length: 6 }, (_unusedValue, genomeId) => ({
    _id: genomeId,
    _parents: genomeId === 0 ? [] : [genomeId - 1],
  }));
}

describe('neat lineage chapter', () => {
  describe('buildAnc', () => {
    describe('given a genome with no recorded parents', () => {
      it('returns an empty ancestor set', () => {
        // Arrange
        const population = createLineagePopulationChain();
        const lineageContext: NeatLineageContext = {
          population,
          _getRNG: () => () => 0.5,
        };
        const rootGenome = population[0] as GenomeLike;

        // Act
        const ancestorIds = buildAnc.call(lineageContext, rootGenome);

        // Assert
        expect([...ancestorIds]).toEqual([]);
      });
    });

    describe('given a parent chain that extends beyond the bounded ancestry window', () => {
      it('returns only the ancestor ids that fall within that depth window', () => {
        // Arrange
        const population = createLineagePopulationChain();
        const lineageContext: NeatLineageContext = {
          population,
          _getRNG: () => () => 0.5,
        };
        const deepestGenome = population.at(-1) as GenomeLike;

        // Act
        const ancestorIds = buildAnc.call(lineageContext, deepestGenome);

        // Assert
        expect(
          [...ancestorIds].toSorted((leftId, rightId) => leftId - rightId),
        ).toEqual([1, 2, 3, 4]);
      });
    });
  });

  describe('computeAncestorUniqueness', () => {
    describe('given a population that cannot form a genome pair', () => {
      it('returns zero', () => {
        // Arrange
        const lineageContext: NeatLineageContext = {
          population: [{ _id: 1 }],
          _getRNG: () => () => 0.5,
        };

        // Act
        const ancestorUniqueness =
          computeAncestorUniqueness.call(lineageContext);

        // Assert
        expect(ancestorUniqueness).toBe(0);
      });
    });

    describe('given a deterministic three-genome lineage sample', () => {
      it('returns the mean sampled ancestor distance', () => {
        // Arrange
        const lineageContext: NeatLineageContext = {
          population: [
            { _id: 0 },
            { _id: 1, _parents: [0] },
            { _id: 2, _parents: [1] },
          ],
          _getRNG: () => () => 0,
        };

        // Act
        const ancestorUniqueness =
          computeAncestorUniqueness.call(lineageContext);

        // Assert
        expect(ancestorUniqueness).toBe(1);
      });
    });
  });
});
