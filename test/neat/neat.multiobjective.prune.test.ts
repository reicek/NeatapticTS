import Neat from '../../src/neat';

// Test that a constant objective is pruned after specified stale window
// while protected variable objective remains.

describe('multi-objective inactive pruning', () => {
  test('removes stagnant objective after window', async () => {
    const neat = new Neat(10, 2, (g: any) => Math.random(), {
      mutationRate: 0.3,
      popsize: 30,
      elitism: 2,
      seed: 123,
      multiObjective: {
        enabled: true,
        objectives: [
          { key: 'objConst', direction: 'max', accessor: (g: any) => 1 }, // constant -> zero range
          {
            key: 'objVar',
            direction: 'max',
            accessor: (g: any) => Math.random(),
          },
        ],
        pruneInactive: {
          enabled: true,
          window: 3,
          rangeEps: 1e-9,
          protect: ['objVar'],
        },
      },
    } as any);

    for (let i = 0; i < 5; i++) {
      await neat.evolve();
    }
    const keys = neat.getObjectives().map((o) => o.key);
    expect(keys).not.toContain('objConst');
    expect(keys).toContain('objVar');
  });
});
