import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('enhanced adaptive mutation strategies', () => {
  const fitness = (net: Network) => {
    // Simple score: number of enabled connections
    const conns = (net as any).connections.filter(
      (c: any) => c.enabled !== false
    ).length;
    return conns;
  };
  test('twoTier strategy adjusts rates differently for halves', async () => {
    const neat = new Neat(3, 1, fitness, {
      popsize: 40,
      adaptiveMutation: {
        enabled: true,
        initialRate: 0.5,
        sigma: 0.1,
        strategy: 'twoTier',
        adaptEvery: 1,
      },
    });
    await neat.evolve();
    await neat.evaluate();
    // Capture initial adapted rates
    const firstRates = neat.population.map((g) => (g as any)._mutRate).slice();
    // Run several generations
    for (let i = 0; i < 4; i++) {
      await neat.evolve();
      await neat.evaluate();
    }
    const finalRates = neat.population.map((g) => (g as any)._mutRate);
    // Expect variance present and at least one rate decreased and one increased relative to 0.5 baseline
    expect(finalRates.some((r) => r < 0.5)).toBe(true);
    expect(finalRates.some((r) => r > 0.5)).toBe(true);
  });
});
