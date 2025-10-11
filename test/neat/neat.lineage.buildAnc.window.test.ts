import { buildAnc } from '../../src/neat/neat.lineage';
import type {
  GenomeLike,
  NeatLineageContext,
} from '../../src/neat/neat.lineage';

/**
 * Direct unit test for buildAnc depth-bounded BFS ancestor collection.
 */
describe('Lineage buildAnc helper', () => {
  describe('bounded ancestor window depth', () => {
    /** Mock genome graph with depth > window to test truncation. */
    const population: GenomeLike[] = [];
    // Build chain 0<-1<-2<-3<-4 (parents point toward earlier ids)
    for (let index = 0; index < 6; index += 1) {
      population.push({ _id: index, _parents: index ? [index - 1] : [] });
    }
    /** Context providing population and trivial RNG. */
    const context: NeatLineageContext = {
      population,
      _getRNG: () => () => 0.5,
    };
    test('ancestor set excludes beyond depth window', () => {
      // Arrange: select deep genome (id 5)
      const genome = population.at(-1) as GenomeLike;
      // Act: compute ancestor set limited by internal window (default 4)
      const ancestors = buildAnc.call(context, genome);
      // Assert: oldest ancestor id 0 excluded when window < full chain (expect size <5)
      expect(ancestors.size).toBeLessThan(5);
    });
  });
});
