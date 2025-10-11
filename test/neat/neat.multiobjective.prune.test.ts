import Neat from '../../src/neat';

// Test that a constant objective is pruned after specified stale window
// while protected variable objective remains.

describe('multi-objective inactive pruning', () => {
  test('removes stagnant objective after window', async () => {
    const neat = new Neat(10, 2, () => Math.random(), {
      mutationRate: 0.3,
      popsize: 30,
      elitism: 2,
      seed: 123,
      multiObjective: {
        enabled: true,
        objectives: [
          { key: 'objConst', direction: 'max', accessor: () => 1 }, // constant -> zero range
          {
            key: 'objVar',
            direction: 'max',
            accessor: () => Math.random(),
          },
        ],
        pruneInactive: {
          enabled: true,
          window: 3,
          rangeEps: 1e-9,
          protect: ['objVar'],
        },
      },
    });

    for (let generationIndex = 0; generationIndex < 5; generationIndex += 1) {
      await neat.evolve();
    }
    const objectivePresence = neat.getObjectives().reduce(
      (accumulator, objective) => {
        if (objective.key === 'objConst') accumulator.hasConst = true;
        if (objective.key === 'objVar') accumulator.hasVar = true;
        return accumulator;
      },
      { hasConst: false, hasVar: false }
    );
    expect(objectivePresence).toEqual({ hasConst: false, hasVar: true });
  });
});
