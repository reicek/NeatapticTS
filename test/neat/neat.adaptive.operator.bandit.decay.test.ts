import Neat from '../../src/neat';
import type { NeatLikeWithAdaptive } from '../../src/neat/neat.adaptive';
import Network from '../../src/architecture/network';
import { applyOperatorAdaptation } from '../../src/neat/neat.adaptive';
import { mutation } from '../../src/methods/mutation';

/** Tests operator adaptation decay & bandit exploration bonus path. */
describe('Operator Adaptation & Bandit', () => {
  describe('decays stats and selects operator with exploration bonus', () => {
    const fitness = (n: Network) => n.nodes.length;
    const neat = new Neat(3, 1, fitness, {
      popsize: 4,
      seed: 950,
      mutation: [mutation.ADD_NODE, mutation.SUB_CONN],
      operatorAdaptation: { enabled: true, decay: 0.5, boost: 2 },
      operatorBandit: { enabled: true, c: 1.4, minAttempts: 1 },
    });
    test('operator stats map populated after mutation & decay', async () => {
      // Arrange: evaluate, mutate twice to populate stats
      await neat.evaluate();
      await neat.mutate();
      await neat.mutate();
      applyOperatorAdaptation.call((neat as unknown) as NeatLikeWithAdaptive);
      // Act: extract stats entries count
      const count = neat.getOperatorStats().length;
      // Assert: at least one operator stat tracked
      expect(count).toBeGreaterThan(0);
    });
  });
});
