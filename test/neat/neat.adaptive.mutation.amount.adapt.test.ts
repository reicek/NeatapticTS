import Neat from '../../src/neat';
import Network from '../../src/architecture/network';
import { mutation } from '../../src/methods/mutation';

/** Covers adaptAmount branch in adaptive mutation (twoTier provides opposite deltas). */
describe('Adaptive Mutation amount adaptation', () => {
  describe('adapts _mutAmount field within bounds', () => {
    const fitness = (n: Network) => n.connections.length;
    const neat = new Neat(3, 1, fitness, {
      popsize: 6,
      seed: 1110,
      mutation: [mutation.MOD_WEIGHT],
      adaptiveMutation: {
        enabled: true,
        strategy: 'twoTier',
        initialRate: 0.5,
        sigma: 0.15,
        adaptAmount: true,
        amountSigma: 0.6,
        minAmount: 1,
        maxAmount: 6,
      },
    });
    test('mutAmount values within configured bounds', async () => {
      await neat.evaluate();
      neat.mutate();
      require('../../src/neat/neat.adaptive').applyAdaptiveMutation.call(
        neat as any
      );
      const ok = neat.population.every(
        (g: any) => g._mutAmount >= 1 && g._mutAmount <= 6
      );
      expect(ok).toBe(true);
    });
  });
});
