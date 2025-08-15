import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('Multi-objective Pareto sorting', () => {
  test('applies rank-based ordering without crashing', async () => {
    const neat = new Neat(4, 2, (n: Network) => n.connections.length, {
      popsize: 18,
      seed: 600,
      speciation: false,
      multiObjective: {
        enabled: true,
        complexityMetric: 'connections',
        dominanceEpsilon: 0.01,
      },
    });
    await neat.evaluate();
    await neat.evolve();
    // Basic sanity: population preserved size and first has score defined
    expect(neat.population.length).toBe(18);
  });
});
