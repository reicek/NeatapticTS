import Neat from '../../src/neat';
import * as methods from '../../src/methods/methods';

/**
 * Operator adaptation decay should reduce (or maintain) success/attempt counts after decay step.
 */

describe('Operator Adaptation Decay', () => {
  it('decays operator stats', async () => {
    const neat = new Neat(3, 1, (n: any) => (n as any).connections.length, {
      popsize: 12,
      mutation: [methods.mutation.ADD_CONN, methods.mutation.ADD_NODE],
      mutationRate: 1,
      mutationAmount: 2,
      operatorAdaptation: { enabled: true, decay: 0.5 },
    });
    // Warmup evolutions to accumulate operator stats
    await neat.evolve();
    await neat.evolve();
    const before = neat.getOperatorStats().map((s) => ({ ...s }));
    // Disable new mutations so subsequent evolve only applies decay
    (neat.options as any).mutationRate = 0;
    await neat.evolve();
    const after = neat.getOperatorStats();
    for (const b of before) {
      const a = after.find((x) => x.name === b.name);
      if (!a) continue;
      expect(a.attempts).toBeLessThanOrEqual(b.attempts + 0.001);
    }
  });
});
