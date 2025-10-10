import Neat from '../../src/neat';
import type { NeatLikeWithAdaptive } from '../../src/neat/neat.adaptive';
import Network from '../../src/architecture/network';
import { mutation } from '../../src/methods/mutation';

/** Tests twoTier adaptive mutation ensuring both up and down adjustments occur. */
describe('Adaptive Mutation twoTier balancing', () => {
  describe('ensures rates diverge up and down', () => {
    const fitness = (n: Network) => n.connections.length;
    const neat = new Neat(3, 1, fitness, {
      popsize: 8,
      seed: 940,
      mutation: [mutation.MOD_WEIGHT],
      adaptiveMutation: {
        enabled: true,
        initialRate: 0.5,
        strategy: 'twoTier',
        sigma: 0.1,
        minRate: 0.01,
        maxRate: 1,
      },
    });
    test('population contains at least two distinct _mutRate values after adaptation', async () => {
      // Arrange: evaluate to assign scores then seed per-genome _mutRate via mutate
      await neat.evaluate();
      neat.mutate();
      const { applyAdaptiveMutation } = await import(
        '../../src/neat/neat.adaptive'
      );
      applyAdaptiveMutation.call(neat as unknown as NeatLikeWithAdaptive);
      // Act: collect distinct rates
      type GenomeLike = { _mutRate: number };
      const distinct = new Set(
        (neat.population as unknown as GenomeLike[]).map((g) => g._mutRate),
      );
      // Assert: at least two different rates (divergence achieved)
      expect(distinct.size).toBeGreaterThan(1);
    });
  });
});

