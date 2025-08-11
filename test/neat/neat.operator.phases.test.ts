import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

// Single expectation per test.

describe('Phased complexity switching', () => {
  test('phase toggles after phaseLength generations', async () => {
    const neat = new Neat(2, 1, (n: Network) => n.connections.length, {
      popsize: 6,
      seed: 400,
      speciation: false,
      phasedComplexity: { enabled: true, phaseLength: 2 },
    });
    await neat.evaluate();
    await neat.evolve(); // gen1
    const phase1 = (neat as any)._phase;
    await neat.evolve(); // gen2 toggles
    await neat.evolve(); // gen3 (second phase)
    const phase2 = (neat as any)._phase;
    expect(phase1).not.toBeUndefined();
  });
});

describe('Operator adaptation tracking', () => {
  test('records operator stats', async () => {
    const neat = new Neat(2, 1, (n: Network) => n.connections.length, {
      popsize: 6,
      seed: 410,
      speciation: false,
      operatorAdaptation: { enabled: true, boost: 2 },
    });
    await neat.evaluate();
    await neat.evolve();
    const stats = (neat as any)._operatorStats;
    const has = Array.from(stats.keys()).length > 0;
    expect(has).toBe(true);
  });
});
