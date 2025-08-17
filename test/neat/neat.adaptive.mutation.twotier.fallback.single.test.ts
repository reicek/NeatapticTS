import Neat from '../../src/neat';
import Network from '../../src/architecture/network';
import { mutation } from '../../src/methods/mutation';

/** Covers twoTier fallback balancing path when only one genome present (only anyUp set). */
describe('Adaptive Mutation twoTier fallback (single genome)', () => {
  describe('fallback balancing adjusts _mutRate', () => {
    const fitness = (n: Network) => n.nodes.length;
    const neat = new Neat(2, 1, fitness, {
      popsize: 1,
      seed: 1100,
      mutation: [mutation.MOD_WEIGHT],
      adaptiveMutation: {
        enabled: true,
        strategy: 'twoTier',
        initialRate: 0.4,
        sigma: 0.2,
        minRate: 0.01,
        maxRate: 1,
      },
    });
    test('mutRate differs from baseline after fallback', async () => {
      await neat.evaluate();
      neat.mutate();
      require('../../src/neat/neat.adaptive').applyAdaptiveMutation.call(
        neat as any
      );
      const rate = (neat.population[0] as any)._mutRate;
      // Expect rate not equal baseline => changed via fallback balancing
      expect(rate).not.toBe(0.4);
    });
  });
});
