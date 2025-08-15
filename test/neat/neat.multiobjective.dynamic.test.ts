import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

// Simple helper to coerce evaluation quickly
const fitness = (n: Network) =>
  (n as any).nodes.length - (n as any).connections.length * 0.01;

describe('Dynamic objective registration', () => {
  test('can register custom entropy objective and see it reflected in fronts', async () => {
    const neat = new Neat(3, 2, fitness, {
      popsize: 30,
      seed: 42,
      speciation: false,
      multiObjective: { enabled: true, complexityMetric: 'nodes' },
    });
    // Register a custom objective using structural entropy proxy (private method via cast)
    neat.registerObjective('entropy', 'max', (g: Network) =>
      (neat as any)._structuralEntropy(g)
    );
    await neat.evaluate();
    await neat.evolve();
    const objs = neat.getObjectives();
    expect(objs.find((o) => o.key === 'entropy')).toBeTruthy();
    const fronts = neat.getParetoFronts(2);
    expect(fronts.length).toBeGreaterThan(0);
    // Ensure at least one genome has rank 0 and entropy value
    const metrics = neat.getMultiObjectiveMetrics();
    const rank0 = metrics.filter((m) => m.rank === 0);
    expect(rank0.length).toBeGreaterThan(0);
  });
});
