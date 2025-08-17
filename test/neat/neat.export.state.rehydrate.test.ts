import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

/**
 * Tests covering serialization & rehydration helpers in `src/neat/neat.export.ts`.
 * Each test follows AAA (Arrange, Act, Assert) and has a single expectation.
 */
describe('NEAT Export / Import State', () => {
  describe('invalid bundle handling', () => {
    /** Fitness function returning constant for deterministic behavior. */
    const fitness = (n: Network) => n.nodes.length;
    test('throws on invalid state bundle', () => {
      // Arrange: capture callable that will invoke static import with bad input
      const act = () => (Neat as any).importState(undefined, fitness);
      // Act & Assert: expect error thrown (single expectation)
      expect(act).toThrow();
    });
  });
  describe('round-trip exportState / importState', () => {
    /** Base fitness using connection count to introduce structural variance. */
    const fitness = (n: Network) => n.connections.length;
    /** Original instance used to produce state snapshot. */
    const neat = new Neat(3, 1, fitness, { popsize: 6, seed: 11 });
    beforeAll(async () => {
      // Arrange: evolve a couple generations to change generation counter & innovations
      await neat.evolve();
      await neat.evolve();
    });
    test('rehydrates identical population size', () => {
      // Arrange: export full state bundle
      const bundle = neat.exportState();
      // Act: rehydrate via static helper
      const restored = (Neat as any).importState(bundle, fitness);
      // Assert: population length preserved across round-trip
      expect(restored.population.length).toBe(neat.population.length);
    });
  });
});
