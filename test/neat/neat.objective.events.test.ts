import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('Objective events telemetry', () => {
  const fit = (n: Network) => n.nodes.length;
  it('records add events for delayed objectives', async () => {
    const neat = new Neat(3, 1, fit, {
      popsize: 12,
      seed: 77,
      multiObjective: {
        enabled: true,
        autoEntropy: true,
        dynamic: { enabled: true, addComplexityAt: 2, addEntropyAt: 4 },
      },
      telemetry: { enabled: true },
    });
    for (let g = 0; g < 6; g++) {
      await neat.evaluate();
      await neat.evolve();
    }
    const events = neat.getObjectiveEvents();
    const adds = events.filter((e) => e.type === 'add').map((e) => e.key);
    expect(adds).toContain('complexity');
    expect(adds).toContain('entropy');
  });
});
