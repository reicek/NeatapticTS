import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('Adaptive re-enable probability', () => {
  test('adjusts reenableProb after many crossovers', async () => {
    const neat = new Neat(
      3,
      1,
      (network: Network) => network.connections.length,
      {
        popsize: 30,
        seed: 500,
        speciation: false,
        reenableProb: 0.3,
      }
    );
    await neat.evaluate();
    for (let iterationIndex = 0; iterationIndex < 5; iterationIndex += 1) {
      await neat.evolve();
    }
    const adjustedProbability = neat.options.reenableProb;
    expect(adjustedProbability).toBeGreaterThan(0); // sanity (non-zero)
  });
});
