import Neat from '../../src/neat';
import type { NeatLikeWithAdaptive } from '../../src/neat/neat.adaptive';
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
      mutation: [mutation.ADD_NODE, mutation.SUB_CONN, mutation.MOD_WEIGHT],
      mutationRate: 1, // Ensure mutations always happen
      mutationAmount: 1,
      operatorAdaptation: { enabled: true, decay: 0.5, boost: 3 },
      operatorBandit: { enabled: true, c: 1.2, minAttempts: 1 },
    });
    test('attempts count decreases after decay application', async () => {
      await neat.evaluate();
      await neat.mutate();
      await neat.mutate();
      const { applyOperatorAdaptation } =
        await import('../../src/neat/neat.adaptive');
      applyOperatorAdaptation.call(neat as unknown as NeatLikeWithAdaptive);
      const before = neat
        .getOperatorStats()
        .reduce((sum, record) => sum + record.attempts, 0);
      applyOperatorAdaptation.call(neat as unknown as NeatLikeWithAdaptive);
      const after = neat
        .getOperatorStats()
        .reduce((sum, record) => sum + record.attempts, 0);
      expect(after).toBeLessThan(before);
    });
  });
});
