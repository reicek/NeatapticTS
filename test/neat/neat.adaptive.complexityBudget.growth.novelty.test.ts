import Neat from '../../src/neat';

/**
 * Tests adaptive complexity budget growth path with positive improvement & noveltyFactor=1.
 */
describe('Adaptive Complexity Budget Growth', () => {
  describe('increases node/conn budget on improvement with sufficient novelty', () => {
    /** Fitness grows each call by mutating internal counter via closure. */
    let score = 0;
    const fitness = () => ++score; // strictly increasing
    /** Instance configured for adaptive growth including connection budget. */
    const neat = new Neat(2, 1, fitness, {
      popsize: 3,
      seed: 960,
      complexityBudget: {
        enabled: true,
        mode: 'adaptive',
        improvementWindow: 3,
        maxNodesStart: 8,
        maxNodesEnd: 20,
        maxConnsStart: 10,
        maxConnsEnd: 30,
        increaseFactor: 1.1,
        stagnationFactor: 0.95,
      },
    });
    test('node budget increases after improvements', () => {
      // Arrange: fill novelty archive to disable novelty dampening
      (neat as any)._noveltyArchive = [1, 2, 3, 4, 5, 6];
      // Simulate improving best score each generation (history drives growth)
      for (let i = 0; i < 4; i++) {
        // Emulate evaluation step: assign an increasing score to best genome
        (neat.population[0] as any).score = i + 1;
        require('../../src/neat/neat.adaptive').applyComplexityBudget.call(
          neat as any
        );
      }
      const after = neat.options.maxNodes;
      // Assert: increased relative to start
      expect(after).toBeGreaterThan(8);
    });
  });
});
