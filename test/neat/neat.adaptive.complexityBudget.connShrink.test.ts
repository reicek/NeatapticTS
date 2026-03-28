import Neat from '../../src/neat';
import type { NeatLikeWithAdaptive } from '../../src/neat/adaptive/adaptive';

/** Covers connection budget shrink path in adaptive complexity budget. */
describe('Adaptive Complexity Budget connection shrink', () => {
  describe('connection budget decreases on stagnation', () => {
    const fitness = () => 1; // stagnation
    const neat = new Neat(2, 1, fitness, {
      popsize: 3,
      seed: 1140,
      complexityBudget: {
        enabled: true,
        mode: 'adaptive',
        improvementWindow: 3,
        maxNodesStart: 6,
        maxConnsStart: 18,
        maxConnsEnd: 30,
        stagnationFactor: 0.5,
      },
    });
    test('maxConns decreases after stagnation window', async () => {
      // Need at least one improvement cycle: record initial
      const start = neat.options.maxConns || 18;
      const { applyComplexityBudget } =
        await import('../../src/neat/adaptive/adaptive');
      for (let i = 0; i < 4; i++) {
        applyComplexityBudget.call(neat as unknown as NeatLikeWithAdaptive);
      }
      const after = neat.options.maxConns;
      expect(after).toBeLessThan(start);
    });
  });
});
