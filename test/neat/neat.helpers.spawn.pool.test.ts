import Neat from '../../src/neat';
import Network from '../../src/architecture/network';
import type {
  LineageTrackedNetwork,
  NeatLineageHarness,
} from '../../src/neat/neat.harness.types';

/**
 * Tests for helper utilities in `neat.helpers.ts` (spawnFromParent, createPool, addGenome).
 */
describe('NEAT Helper Utilities', () => {
  describe('spawnFromParent lineage metadata', () => {
    /** Deterministic fitness returning node count. */
    const fitness = (network: Network) => network.nodes.length;
    /** Neat instance with minimal configuration. */
    const neat = new Neat(2, 1, fitness, { popsize: 4, seed: 222 });
    let parent: LineageTrackedNetwork;
    beforeAll(async () => {
      await neat.evaluate();
      parent = neat.population[0] as LineageTrackedNetwork;
    });
    test('child references single parent id', async () => {
      // Arrange: spawn child from parent
      const helper = neat as NeatLineageHarness;
      const child = (await helper.spawnFromParent(
        parent,
        1,
      )) as LineageTrackedNetwork;
      // Act & Assert: lineage metadata captures single parent id
      expect(child._parents).toEqual([parent._id]);
    });
  });
  describe('createPool seeded cloning', () => {
    /** Fitness returns connection length to avoid score ties influencing logic. */
    const fitness = (network: Network) => network.connections.length;
    /** Seed network used for cloning across pool. */
    const seedNet = new Network(2, 1);
    /** Neat instance built with popsize 5 for pool creation. */
    const neat = new Neat(2, 1, fitness, { popsize: 5, seed: 333 });
    beforeAll(() => {
      // Arrange: create new pool from seed network
      neat.createPool(seedNet);
    });
    test('all genomes cloned from seed have identical IO counts', () => {
      // Act: collect distinct (input,output) signatures
      const sigs = new Set(
        neat.population.map((g) => `${g.input}-${g.output}`),
      );
      // Assert: only one signature means consistent cloning
      expect(sigs.size).toBe(1);
    });
  });
});
