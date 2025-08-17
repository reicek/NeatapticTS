import Neat from '../../src/neat';
import Network from '../../src/architecture/network';
import { mutation } from '../../src/methods/mutation';

/** Tests anneal adaptive mutation strategy (deltas shrink over progress). */
describe('Adaptive Mutation anneal strategy', () => {
  describe('rates remain within bounds after annealing adaptation', () => {
    const fitness = (n: Network) => n.nodes.length;
    const neat = new Neat(2, 1, fitness, {
      popsize: 6,
      seed: 990,
      mutation: [mutation.MOD_WEIGHT],
      adaptiveMutation: {
        enabled: true,
        strategy: 'anneal',
        initialRate: 0.6,
        sigma: 0.3,
        minRate: 0.01,
        maxRate: 1,
      },
    });
    test('all rates bounded within [min,max] after adaptation', async () => {
      await neat.evaluate();
      // increase generation to trigger progress scaling
      (neat as any).generation = 40;
      neat.mutate();
      require('../../src/neat/neat.adaptive').applyAdaptiveMutation.call(
        neat as any
      );
      const within = neat.population.every(
        (g: any) => g._mutRate >= 0.01 && g._mutRate <= 1
      );
      expect(within).toBe(true);
    });
  });
});
