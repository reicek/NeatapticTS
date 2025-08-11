import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('Diversity metrics', () => {
  test('computes diversity snapshot with compatibility and entropy stats', async () => {
    const neat = new Neat(4, 2, (n: Network) => n.connections.length, {
      popsize: 25,
      seed: 77,
      speciation: true,
      diversityMetrics: { enabled: true, pairSample: 30, graphletSample: 40 },
    });
    await neat.evaluate();
    await neat.evolve();
    const stats = neat.getDiversityStats();
    expect(stats).toBeTruthy();
    expect(typeof stats.meanCompat).toBe('number');
    expect(typeof stats.graphletEntropy).toBe('number');
    const tel = neat.getTelemetry().slice(-1)[0];
    expect(tel.diversity).toBeTruthy();
  });
});
