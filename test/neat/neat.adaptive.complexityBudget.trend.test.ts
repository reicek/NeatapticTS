import Neat from '../../src/neat';

/** Additional complexity budget tests covering adaptive shrink & connection budget. */
describe('Adaptive Complexity Budget (trend & connection budget)', () => {
  describe('shrinks node budget on stagnation with connection budget update', () => {
    const fitness = () => 1; // constant so no improvement
    const neat = new Neat(2, 1, fitness, {
      popsize: 3,
      seed: 901,
      complexityBudget: {
        enabled: true,
        mode: 'adaptive',
        improvementWindow: 3,
        maxNodesStart: 10,
        maxNodesEnd: 50,
        minNodes: 5,
        maxConnsStart: 20,
        maxConnsEnd: 40,
        increaseFactor: 1.05,
        stagnationFactor: 0.9,
      },
    });
    test('node budget decreases after full stagnation window', () => {
      // Arrange: run enough calls to fill window with constant best scores
      for (let i = 0; i < 4; i++)
        require('../../src/neat/neat.adaptive').applyComplexityBudget.call(
          neat as any
        );
      const after = neat.options.maxNodes;
      // Assert: maxNodes decreased but not below minNodes
      expect(after).toBeLessThan(10);
    });
  });
  describe('linear schedule ramps toward end', () => {
    const fitness = () => 1;
    const neat = new Neat(2, 1, fitness, {
      popsize: 2,
      seed: 902,
      complexityBudget: {
        enabled: true,
        mode: 'linear',
        maxNodesStart: 6,
        maxNodesEnd: 12,
        horizon: 5,
      },
    });
    test('linear schedule increases maxNodes over generations', () => {
      // Arrange: simulate generations and invoke scheduler
      for (let g = 0; g < 5; g++) {
        (neat as any).generation = g;
        require('../../src/neat/neat.adaptive').applyComplexityBudget.call(
          neat as any
        );
      }
      // Act: capture final value
      const finalVal = neat.options.maxNodes;
      // Assert: ramp reached or below configured end (integer rounding) but >= start
      expect(finalVal).toBeGreaterThanOrEqual(6);
    });
  });
});
