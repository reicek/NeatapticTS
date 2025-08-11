import Neat from '../../src/neat';

describe('inactive objective pruning behavior', () => {
  test('does not prune when disabled', async () => {
    const neat = new Neat(6, 2, (g: any) => Math.random(), {
      popsize: 15,
      seed: 77,
      multiObjective: {
        enabled: true,
        objectives: [
          { key: 'constA', direction: 'max', accessor: () => 1 },
          { key: 'varB', direction: 'max', accessor: () => Math.random() },
        ],
        pruneInactive: { enabled: false, window: 2, rangeEps: 1e-9 },
      },
    } as any);
    for (let i = 0; i < 4; i++) await neat.evolve();
    const keys = neat
      .getObjectives()
      .map((o) => o.key)
      .sort();
    expect(keys).toEqual(['constA', 'varB'].sort());
  });

  test('prunes only after required consecutive stagnant window', async () => {
    const neat = new Neat(6, 2, (g: any) => Math.random(), {
      popsize: 18,
      seed: 88,
      multiObjective: {
        enabled: true,
        objectives: [
          { key: 'constA', direction: 'max', accessor: () => 1 },
          { key: 'constB', direction: 'max', accessor: () => 2 },
          { key: 'varB', direction: 'max', accessor: () => Math.random() },
        ],
        pruneInactive: {
          enabled: true,
          window: 3,
          rangeEps: 1e-9,
          protect: ['varB'],
        },
      },
    } as any);
    for (let i = 0; i < 2; i++) await neat.evolve();
    let keys = neat.getObjectives().map((o) => o.key);
    expect(keys).toEqual(expect.arrayContaining(['constA', 'constB', 'varB']));
    await neat.evolve();
    await neat.evolve();
    keys = neat.getObjectives().map((o) => o.key);
    expect(keys).not.toContain('constA');
    expect(keys).not.toContain('constB');
    expect(keys).toContain('varB');
  });
});
