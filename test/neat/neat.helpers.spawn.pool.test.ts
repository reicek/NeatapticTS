import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

/**
 * Tests for helper utilities in `neat.helpers.ts` (spawnFromParent, createPool, addGenome).
 */
describe('NEAT Helper Utilities', () => {
  describe('spawnFromParent lineage metadata', () => {
    /** Deterministic fitness returning node count. */
    const fitness = (n: Network) => n.nodes.length;
    /** Neat instance with minimal configuration. */
    const neat = new Neat(2, 1, fitness, { popsize: 4, seed: 222 });
    let parent: any;
    beforeAll(async () => {
      await neat.evaluate();
      parent = neat.population[0];
    });
    test('child references single parent id', () => {
      // Arrange: spawn child from parent
      const child = (neat as any).spawnFromParent(parent, 1);
      // Act: inspect lineage metadata
      const parentIds = child._parents;
      // Assert: exactly one parent id recorded
      expect(parentIds.length).toBe(1);
    });
  });
  describe('createPool seeded cloning', () => {
    /** Fitness returns connection length to avoid score ties influencing logic. */
    const fitness = (n: Network) => n.connections.length;
    /** Seed network used for cloning across pool. */
    const seedNet = new Network(2, 1);
    /** Neat instance built with popsize 5 for pool creation. */
    const neat = new Neat(2, 1, fitness, { popsize: 5, seed: 333 });
    beforeAll(() => {
      // Arrange: create new pool from seed network
      (neat as any).createPool(seedNet);
    });
    test('all genomes cloned from seed have identical IO counts', () => {
      // Act: collect distinct (input,output) signatures
      const sigs = new Set(
        neat.population.map((g) => `${g.input}-${g.output}`)
      );
      // Assert: only one signature means consistent cloning
      expect(sigs.size).toBe(1);
    });
  });
});
