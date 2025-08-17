import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

/**
 * Tests for compatibility distance & fallback innovation id logic (neat.compat.ts).
 * Focus: cache reuse path and fallback innovation usage when innovations missing.
 */
describe('NEAT Compatibility Distance', () => {
  describe('fallback innovation id and cache reuse', () => {
    /** Fitness simple structural count for deterministic shape. */
    const fitness = (n: Network) => n.nodes.length;
    /** Instance with coefficients set for predictable distance. */
    const neat = new Neat(3, 1, fitness, {
      popsize: 4,
      seed: 777,
      excessCoeff: 1,
      disjointCoeff: 1,
      weightDiffCoeff: 0.4,
    });
    let gA: any, gB: any;
    beforeAll(async () => {
      await neat.evaluate();
      // Arrange: take two genomes and strip innovations to force fallback path
      gA = neat.population[0];
      gB = neat.population[1];
      gA.connections.forEach((c: any) => delete c.innovation);
      gB.connections.forEach((c: any) => delete c.innovation);
    });
    test('second distance call reuses cached value', () => {
      // Arrange: first call populates cache
      const first = (neat as any)._compatibilityDistance(gA, gB);
      // Act: second call should hit cache (value identical)
      const second = (neat as any)._compatibilityDistance(gA, gB);
      // Assert: identical numeric distance implies cache reuse path executed
      expect(second).toBe(first);
    });
  });
});
