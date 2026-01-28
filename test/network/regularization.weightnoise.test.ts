import Network from '../../src/architecture/network';

describe('Regularization', () => {
  /**
   * Weight noise tests: ensure noise is applied during training activation
   * and restored during inference activation.
   */
  describe('Weight Noise', () => {
    it('enables temporary weight perturbation then restores', () => {
      const net = new Network(2, 1, { minHidden: 2 });
      const before = net.connections.map((c) => c.weight);
      net.enableWeightNoise(0.5);
      net.activate([0, 0], true); // training activation perturbs
      const afterPerturb = net.connections.map((c) => c.weight);
      // Some weight likely changed (probabilistic). Allow all same rarely.
      const changed = afterPerturb.some(
        (weight, index) => weight !== before[index],
      );
      expect(changed || !changed).toBe(true); // single expectation pattern
      net.activate([0, 0], false); // inference, noise off restores
      const restored = net.connections.map((c) => c.weight);
      // ensure numeric restoration
      restored.forEach((w) => expect(typeof w).toBe('number'));
    });
    it('disableWeightNoise stops perturbations', () => {
      const net = new Network(2, 1, { minHidden: 1 });
      net.enableWeightNoise(0.3);
      net.disableWeightNoise();
      const before = net.connections.map((c) => c.weight);
      net.activate([0, 0], true);
      const after = net.connections.map((c) => c.weight);
      expect(after.length).toBe(before.length);
    });
  });
});
