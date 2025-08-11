import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('Objective lifetime tracking (objAges telemetry)', () => {
  const fitness = (net: Network) => net.nodes.length;
  it('tracks increasing ages and later-added objectives have lower age', async () => {
    const neat = new Neat(2, 1, fitness, {
      popsize: 15,
      seed: 42,
      multiObjective: {
        enabled: true,
        autoEntropy: true,
        dynamic: { enabled: true, addComplexityAt: 2, addEntropyAt: 4 },
      },
      telemetry: { enabled: true },
    });
    // Run several generations
    for (let g = 0; g < 8; g++) {
      await neat.evaluate();
      await neat.evolve();
    }
    const telem = neat.getTelemetry();
    const last = telem[telem.length - 1];
    expect(last.objAges).toBeDefined();
    const ages = last.objAges as Record<string, number>;
    // Fitness always present, complexity added at gen>=2, entropy at gen>=4 so ordering should reflect <= relationship
    expect(ages.fitness).toBeGreaterThan(0);
    if (ages.complexity != null)
      expect(ages.fitness).toBeGreaterThanOrEqual(ages.complexity);
    if (ages.entropy != null && ages.complexity != null)
      expect(ages.complexity).toBeGreaterThanOrEqual(ages.entropy);
  });
});
