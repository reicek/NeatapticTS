import { buildAnc } from './lineage';
import type { GenomeLike, NeatLineageContext } from './lineage';

function createLineagePopulationChain(): GenomeLike[] {
  return Array.from({ length: 6 }, (_unusedValue, genomeId) => ({
    _id: genomeId,
    _parents: genomeId === 0 ? [] : [genomeId - 1],
  }));
}

describe('neat lineage chapter', () => {
  describe('buildAnc', () => {
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
});
