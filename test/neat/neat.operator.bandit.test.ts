import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('operator bandit selection', () => {
  const fitness = (net: Network) => (net as any).connections.length;
  test('bandit populates operator stats', async () => {
    const neat = new Neat(4, 2, fitness, {
      popsize: 25,
      operatorBandit: { enabled: true, c: 1.0, minAttempts: 3 },
      adaptiveMutation: { enabled: true, initialRate: 0.9 },
    });
    for (let i = 0; i < 3; i++) await neat.evolve();
    await neat.evaluate();
    const stats = neat.getOperatorStats();
    expect(stats.length).toBeGreaterThan(0);
  });
});
