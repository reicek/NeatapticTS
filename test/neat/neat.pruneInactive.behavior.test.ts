import Neat from '../../src/neat';

describe('inactive objective pruning behavior', () => {
  test('does not prune when disabled', async () => {
    const neat = new Neat(6, 2, () => Math.random(), {
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
    });
    for (let generationIndex = 0; generationIndex < 4; generationIndex += 1) {
      await neat.evolve();
    }
    const keys = neat
      .getObjectives()
      .map((o) => o.key)
      .sort();
    expect(keys).toEqual(['constA', 'varB'].sort());
  });

  test('prunes only after required consecutive stagnant window', async () => {
    const neat = new Neat(6, 2, () => Math.random(), {
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
    });
    for (let generationIndex = 0; generationIndex < 2; generationIndex += 1) {
      await neat.evolve();
    }
    let keys = neat.getObjectives().map((objective) => objective.key);
    expect(keys).toEqual(expect.arrayContaining(['constA', 'constB', 'varB']));
    await neat.evolve();
    await neat.evolve();
    keys = neat.getObjectives().map((objective) => objective.key);
    const objectivePresence = keys.reduce(
      (accumulator, key) => {
        if (key === 'constA') accumulator.constA = true;
        if (key === 'constB') accumulator.constB = true;
        if (key === 'varB') accumulator.varB = true;
        return accumulator;
      },
      { constA: false, constB: false, varB: false },
    );
    expect(objectivePresence).toEqual({
      constA: false,
      constB: false,
      varB: true,
    });
  });
});
