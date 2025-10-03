import Neat from '../../src/neat';
import Network from '../../src/architecture/network';
import { mutation } from '../../src/methods/mutation';

/** Tests exploreLow adaptive mutation strategy (bottom half gets upward deltas). */
describe('Adaptive Mutation exploreLow strategy', () => {
  describe('bottom half mutation rates increase relative to top', () => {
    const fitness = (n: Network) => n.connections.length;
    const neat = new Neat(3, 1, fitness, {
      popsize: 10,
      seed: 980,
      mutation: [mutation.MOD_WEIGHT],
      adaptiveMutation: {
        enabled: true,
        strategy: 'exploreLow',
        initialRate: 0.4,
        sigma: 0.2,
        minRate: 0.01,
        maxRate: 1,
      },
    });
    test('max rate difference positive after adaptation', async () => {
      await neat.evaluate();
      // Seed per-genome rates
      neat.mutate();
      const { applyAdaptiveMutation } = await import('../../src/neat/neat.adaptive');
      applyAdaptiveMutation.call(neat as unknown as Record<string, unknown>);
      type GenomeLike = { _mutRate: number };
      const rates = (neat.population as unknown as GenomeLike[])
        .map((g) => g._mutRate)
        .sort((a, b) => a - b);
      // Assert: spread across rates (top - bottom > 0)
      expect(rates[rates.length - 1] - rates[0]).toBeGreaterThan(0);
    });
  });
});
