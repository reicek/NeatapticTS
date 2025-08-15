import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

describe('Adaptive pruning', () => {
  const fit = (n: Network) => n.nodes.length;
  it('increases pruning level toward target sparsity', async () => {
    const neat = new Neat(5, 2, fit, {
      popsize: 30,
      seed: 202,
      adaptivePruning: {
        enabled: true,
        targetSparsity: 0.4,
        adjustRate: 0.2,
        metric: 'connections',
      },
    });
    for (let g = 0; g < 6; g++) {
      await neat.evaluate();
      await neat.evolve();
    }
    // Internal level should move above zero
    const lvl = (neat as any)._adaptivePruneLevel || 0;
    expect(lvl).toBeGreaterThan(0);
  });
});
