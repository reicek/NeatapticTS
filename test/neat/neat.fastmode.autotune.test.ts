import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('fastMode auto-tuning', () => {
  test('auto lowers sampling defaults when unspecified', async () => {
    const neat = new Neat(4, 2, () => Math.random(), {
      popsize: 20,
      seed: 123,
      fastMode: true,
      diversityMetrics: { enabled: true }, // pairSample / graphletSample undefined => should be tuned
      novelty: {
        enabled: true,
        descriptor: (network: Network) => [
          network.nodes.length,
          network.connections.length,
        ],
      },
    });

    await neat.evolve(); // triggers diversity stats computation and fastMode tuning once

    expect(neat.options.diversityMetrics?.pairSample).toBe(20);
    expect(neat.options.diversityMetrics?.graphletSample).toBe(30);
    expect(neat.options.novelty?.k).toBe(5);
  });

  test('does not override user supplied sampling values', async () => {
    const neat = new Neat(4, 2, () => Math.random(), {
      popsize: 20,
      seed: 321,
      fastMode: true,
      diversityMetrics: { enabled: true, pairSample: 50, graphletSample: 70 },
      novelty: {
        enabled: true,
        k: 11,
        descriptor: (network: Network) => [network.nodes.length],
      },
    });

    await neat.evolve();

    expect(neat.options.diversityMetrics?.pairSample).toBe(50);
    expect(neat.options.diversityMetrics?.graphletSample).toBe(70);
    expect(neat.options.novelty?.k).toBe(11);
  });
});
