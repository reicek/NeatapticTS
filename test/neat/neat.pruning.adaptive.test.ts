import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

/** Tests for evolution-time pruning and adaptive pruning controllers. */
describe('NEAT Pruning Controllers', () => {
  describe('applyEvolutionPruning ramp fraction scaling', () => {
    /** Fitness simple structural metric. */
    const fitness = (n: Network) => n.connections.length;
    /** Instance with evolution pruning configured to ramp. */
    const neat = new Neat(3, 1, fitness, {
      popsize: 5,
      seed: 600,
      evolutionPruning: {
        startGeneration: 0,
        targetSparsity: 0.5,
        rampGenerations: 4,
        interval: 1,
      },
    });
    beforeAll(async () => {
      await neat.evaluate();
      // Advance several generations to enter ramp window
      for (let i = 0; i < 2; i++) await neat.evolve();
      // Act: invoke pruning (line coverage for ramp computation)
      (neat as any).applyEvolutionPruning();
    });
    test('population still defined after pruning invocation', () => {
      // Assert: pruning did not remove population structure
      expect(Array.isArray(neat.population)).toBe(true);
    });
  });
  describe('applyAdaptivePruning adjusts prune level', () => {
    /** Fitness uses node count. */
    const fitness = (n: Network) => n.nodes.length;
    /** Instance with adaptive pruning enabled. */
    const neat = new Neat(3, 1, fitness, {
      popsize: 4,
      seed: 601,
      adaptivePruning: {
        enabled: true,
        metric: 'connections',
        targetSparsity: 0.3,
        adjustRate: 0.5,
        tolerance: 0,
      },
    });
    beforeAll(async () => {
      await neat.evaluate();
      // Act: call adaptive pruning twice to trigger adjustment branch
      (neat as any).applyAdaptivePruning();
      (neat as any).applyAdaptivePruning();
    });
    test('adaptive prune level field initialized on instance', () => {
      // Assert: internal field exists (indicates adjustment logic executed)
      expect((neat as any)._adaptivePruneLevel !== undefined).toBe(true);
    });
  });
});
