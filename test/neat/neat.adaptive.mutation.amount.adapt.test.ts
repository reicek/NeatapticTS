import Neat from '../../src/neat';
import Network from '../../src/architecture/network';
import { mutation } from '../../src/methods/mutation';
import type { NeatLikeWithAdaptive } from '../../src/neat/neat.adaptive';

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
      await neat.mutate();
      // use ES import to access adaptive helpers
      const { applyAdaptiveMutation } = await import(
        '../../src/neat/neat.adaptive'
      );
      applyAdaptiveMutation.call((neat as unknown) as NeatLikeWithAdaptive);
      type GenomeWithAdaptiveAmount = Network & { _mutAmount: number };
      const populationWithAdaptiveAmount = neat.population as Array<GenomeWithAdaptiveAmount>;
      const mutationAmountWithinBounds = populationWithAdaptiveAmount.every(
        (genome) => genome._mutAmount >= 1 && genome._mutAmount <= 6
      );
      expect(mutationAmountWithinBounds).toBe(true);
    });
  });
});
