import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('Objective importance telemetry', () => {
  const fit = (n: Network) => n.nodes.length;
  it('emits objImportance with range/var for objectives', async () => {
    const neat = new Neat(3, 1, fit, {
      popsize: 18,
      seed: 311,
      multiObjective: { enabled: true, autoEntropy: true },
      telemetry: { enabled: true },
    });
    for (let g = 0; g < 4; g++) {
      await neat.evaluate();
      await neat.evolve();
    }
    const last = neat.getTelemetry().slice(-1)[0];
    expect(last.objImportance).toBeDefined();
    expect(last.objImportance.fitness).toBeDefined();
  });
});
