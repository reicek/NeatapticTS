import Neat from '../../src/neat';

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
        require('../../src/neat/neat.adaptive').applyComplexityBudget.call(
          neat as any
        );
      expect(neat.options.maxNodes).toBeGreaterThanOrEqual(8);
    });
  });
});
