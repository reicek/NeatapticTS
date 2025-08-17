import Neat from '../../src/neat';
import Network from '../../src/architecture/network';
import { mutation } from '../../src/methods/mutation';

/**
 * Extended operator adaptation coverage: ensure decay reduces attempts and bandit explores.
 */
describe('Operator Adaptation Decay & Bandit Exploration', () => {
  describe('decay reduces attempts counts over successive applications', () => {
    const fitness = (n: Network) => n.connections.length;
    const neat = new Neat(3, 1, fitness, {
      popsize: 5,
      seed: 1000,
      mutation: [mutation.ADD_NODE, mutation.SUB_CONN],
      operatorAdaptation: { enabled: true, decay: 0.5, boost: 3 },
      operatorBandit: { enabled: true, c: 1.2, minAttempts: 1 },
    });
    test('attempts count decreases after decay application', async () => {
      await neat.evaluate();
      neat.mutate();
      neat.mutate();
      require('../../src/neat/neat.adaptive').applyOperatorAdaptation.call(
        neat as any
      );
      const before = neat
        .getOperatorStats()
        .reduce((s, r) => s + r.attempts, 0);
      require('../../src/neat/neat.adaptive').applyOperatorAdaptation.call(
        neat as any
      );
      const after = neat.getOperatorStats().reduce((s, r) => s + r.attempts, 0);
      expect(after).toBeLessThan(before);
    });
  });
});
