import Neat from '../../src/neat';
import type { NeatLikeWithAdaptive } from '../../src/neat/adaptive/adaptive';
import { applyComplexityBudget } from '../../src/neat/adaptive/adaptive';

/** Covers explicit minNodes clamp path in adaptive complexity budget when shrink exceeds min. */
describe('Adaptive Complexity Budget minNodes clamp', () => {
  describe('does not shrink below configured minNodes', () => {
    const fitness = () => 1; // stagnation -> shrink attempts
    const neat = new Neat(2, 1, fitness, {
      popsize: 2,
      seed: 1130,
      complexityBudget: {
        enabled: true,
        mode: 'adaptive',
        improvementWindow: 2,
        maxNodesStart: 9,
        minNodes: 8,
        stagnationFactor: 0.1,
      },
    });
    test('maxNodes stays >= minNodes after shrink cycles', () => {
      for (let i = 0; i < 5; i++)
        applyComplexityBudget.call(neat as unknown as NeatLikeWithAdaptive);
      expect(neat.options.maxNodes).toBeGreaterThanOrEqual(8);
    });
  });
});
