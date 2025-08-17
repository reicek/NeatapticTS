import { buildAnc } from '../../src/neat/neat.lineage';

/**
 * Direct unit test for buildAnc depth-bounded BFS ancestor collection.
 */
describe('Lineage buildAnc helper', () => {
  describe('bounded ancestor window depth', () => {
    /** Mock genome graph with depth > window to test truncation. */
    const population: any[] = [];
    // Build chain 0<-1<-2<-3<-4 (parents point toward earlier ids)
    for (let i = 0; i < 6; i++)
      population.push({ _id: i, _parents: i ? [i - 1] : [] });
    /** Context providing population and trivial RNG. */
    const ctx = { population, _getRNG: () => () => 0.5 } as any;
    test('ancestor set excludes beyond depth window', () => {
      // Arrange: select deep genome (id 5)
      const genome = population[5];
      // Act: compute ancestor set limited by internal window (default 4)
      const ancs = buildAnc.call(ctx, genome);
      // Assert: oldest ancestor id 0 excluded when window < full chain (expect size <5)
      expect(ancs.size).toBeLessThan(5);
    });
  });
});
