import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('Dynamic multi-objective scheduling', () => {
  const fitness = (network: Network) =>
    network.connections.filter((connection) => connection.enabled !== false)
      .length;

  test('delayed complexity objective and entropy drop/readd', async () => {
    const neat = new Neat(2, 1, fitness, {
      popsize: 25,
      seed: 42,
      multiObjective: {
        enabled: true,
        autoEntropy: true,
        complexityMetric: 'nodes',
        dynamic: {
          enabled: true,
          addComplexityAt: 4,
          addEntropyAt: 3,
          dropEntropyOnStagnation: 6,
          readdEntropyAfter: 2,
        },
      },
      telemetry: { enabled: true },
      lineageTracking: false,
    });
    const keysByGen: string[][] = [];
    for (let generationIndex = 0; generationIndex < 10; generationIndex += 1) {
      await neat.evaluate();
      await neat.evolve();
      keysByGen.push(neat.getObjectiveKeys());
    }
    // keysByGen[i] corresponds to generation i+1 (since generation increments after evolve)
    // Complexity absent for generations <4 => indices 0..2, present at generation 4 => index 3
    for (let index = 0; index < 3; index += 1)
      expect(keysByGen[index]).not.toContain('complexity');
    expect(keysByGen[3]).toContain('complexity');
    // Entropy absent for generations <3 => indices 0..1, present at generation 3 => index 2
    for (let index = 0; index < 2; index += 1)
      expect(keysByGen[index]).not.toContain('entropy');
    expect(keysByGen[2]).toContain('entropy');
    for (const keys of keysByGen) expect(keys).toContain('fitness');
  }, 45000);
});
