import Neat from '../../src/neat';
import Network from '../../src/architecture/network';

/** Tests for adaptive minimal criterion threshold adjustment & rejection. */
describe('Adaptive Minimal Criterion', () => {
  describe('threshold increases when acceptance too high', () => {
    const fitness = (n: Network) => n.nodes.length; // varying scores
    const neat = new Neat(2, 1, fitness, {
      popsize: 4,
      seed: 920,
      minimalCriterionAdaptive: {
        enabled: true,
        initialThreshold: 0.1,
        targetAcceptance: 0.5,
        adjustRate: 0.5,
      },
    });
    let before: number;
    test('threshold increases after evaluation', async () => {
      // Arrange: evaluate to set scores > threshold
      await neat.evaluate();
      before = (neat as any)._mcThreshold ?? 0.1;
      require('../../src/neat/neat.adaptive').applyMinimalCriterionAdaptive.call(
        neat as any
      );
      // Act: obtain adapted threshold
      const after = (neat as any)._mcThreshold;
      // Assert: threshold increased (acceptance above target triggers growth)
      expect(after).toBeGreaterThan(before);
    });
  });
});
