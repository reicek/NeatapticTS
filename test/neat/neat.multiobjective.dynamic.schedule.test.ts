import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('Dynamic multi-objective scheduling', () => {
  const fitness = (net: Network) => {
    let s = 0;
    for (const c of net.connections) if ((c as any).enabled !== false) s++;
    return s;
  };

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
    for (let g = 0; g < 10; g++) {
      await neat.evaluate();
      await neat.evolve();
      keysByGen.push(neat.getObjectiveKeys());
    }
    // keysByGen[i] corresponds to generation i+1 (since generation increments after evolve)
    // Complexity absent for generations <4 => indices 0..2, present at generation 4 => index 3
    for (let i = 0; i < 3; i++)
      expect(keysByGen[i]).not.toContain('complexity');
    expect(keysByGen[3]).toContain('complexity');
    // Entropy absent for generations <3 => indices 0..1, present at generation 3 => index 2
    for (let i = 0; i < 2; i++) expect(keysByGen[i]).not.toContain('entropy');
    expect(keysByGen[2]).toContain('entropy');
    for (const ks of keysByGen) expect(ks).toContain('fitness');
  }, 45000);
});
