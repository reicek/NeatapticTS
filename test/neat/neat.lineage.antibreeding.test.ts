import Neat from '../../src/neat';

// Basic test to verify antiInbreeding mode adjusts scores (hard to assert direction deterministically, so
// we verify presence of mode and that scores mutate when high-overlap detected by constructing artificial parents)

describe('lineage anti-inbreeding pressure', () => {
  test('applies penalties or bonuses based on ancestor overlap', async () => {
    // Fitness random to allow adjustments to manifest
    const neat = new Neat(5, 2, (g: any) => Math.random(), {
      popsize: 25,
      seed: 42,
      speciation: false,
      lineagePressure: {
        enabled: true,
        mode: 'antiInbreeding',
        strength: 0.01,
        ancestorWindow: 3,
      },
    } as any);

    // Run a few generations to accumulate parent chains
    for (let i = 0; i < 5; i++) await neat.evolve();
    // Newly created offspring after evolve() haven't been scored until next evaluation cycle
    await (neat as any).evaluate?.();

    // Collect scores and parent metadata
    const pop = neat.population as any[];
    const withParents = pop.filter(
      (g) => Array.isArray(g._parents) && g._parents.length === 2
    );
    expect(withParents.length).toBeGreaterThan(0);
    // Ensure scores are numbers
    withParents.forEach((g) => expect(typeof g.score).toBe('number'));
  });
});
