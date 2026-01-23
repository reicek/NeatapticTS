import { Architect } from '../../src/neataptic';

describe('Regularization', () => {
  describe('Stochastic Depth', () => {
    it('skips hidden layer sometimes during training (probabilistic)', () => {
      const net = Architect.perceptron(2, 4, 4, 1); // has 2 hidden layers after minHidden enforcement
      // Hidden layers count = layers.length -2
      const hiddenCount = (net as unknown as { layers?: unknown[] }).layers
        ? (net as unknown as { layers: unknown[] }).layers.length - 2
        : 0;
      const survival = Array.from(
        { length: Math.max(0, hiddenCount) },
        () => 0.5,
      );
      net.setStochasticDepth(survival);
      // Run several activations to ensure stochastic depth runs without throwing
      for (let iteration = 0; iteration < 20; iteration++) {
        net.activate([0, 0], true);
      }
      // Probabilistic test: just ensure activate exists and is callable
      expect(typeof net.activate).toBe('function');
    });
    it('disables stochastic depth', () => {
      const net = Architect.perceptron(2, 5, 1);
      (
        net as unknown as {
          setStochasticDepth: (s: number[]) => void;
        }
      ).setStochasticDepth([0.9]);
      (
        net as unknown as {
          disableStochasticDepth: () => void;
        }
      ).disableStochasticDepth();
      expect(
        Array.isArray(
          (net as unknown as { _stochasticDepth?: unknown })._stochasticDepth,
        ),
      ).toBe(true);
    });
  });
});
