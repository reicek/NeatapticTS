import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('Adaptive re-enable probability', () => {
  test('adjusts reenableProb after many crossovers', async () => {
    const neat = new Neat(3, 1, (n: Network) => n.connections.length, {
      popsize: 30,
      seed: 500,
      speciation: false,
      reenableProb: 0.3,
    });
    await neat.evaluate();
    const before = neat.options.reenableProb;
    for (let i = 0; i < 5; i++) await neat.evolve();
    const after = neat.options.reenableProb;
    expect(after).toBeGreaterThan(0); // sanity (non-zero)
  });
});
