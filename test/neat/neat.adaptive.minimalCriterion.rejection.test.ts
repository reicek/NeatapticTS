import Neat from '../../src/neat';
import Network from '../../src/architecture/network';
import { applyMinimalCriterionAdaptive } from '../../src/neat/neat.adaptive';

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
      before =
        (Reflect.get(neat as object, '_mcThreshold') as number | undefined) ??
        0.1;
      applyMinimalCriterionAdaptive.call(neat);
      // Act: obtain adapted threshold
      const after = Reflect.get(neat as object, '_mcThreshold') as number;
      // Assert: threshold increased (acceptance above target triggers growth)
      expect(after).toBeGreaterThan(before);
    });
  });
});
