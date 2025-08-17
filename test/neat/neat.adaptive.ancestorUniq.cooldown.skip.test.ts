import Neat from '../../src/neat';

/** Covers cooldown early-return branch in applyAncestorUniqAdaptive. */
describe('Ancestor Uniqueness Adaptive cooldown skip', () => {
  describe('no change when within cooldown window', () => {
    const fitness = () => 1;
    const neat = new Neat(2, 1, fitness, {
      popsize: 3,
      seed: 1120,
      ancestorUniqAdaptive: {
        enabled: true,
        mode: 'epsilon',
        lowThreshold: 0.2,
        highThreshold: 0.8,
        adjust: 0.05,
        cooldown: 5,
      },
      multiObjective: {
        adaptiveEpsilon: { enabled: true },
        dominanceEpsilon: 0.1,
      },
    });
    test('epsilon unchanged due to cooldown', () => {
      // Arrange: set last adjust generation to near current
      (neat as any)._lastAncestorUniqAdjustGen = 9;
      (neat as any).generation = 12; // difference 3 < cooldown 5
      (neat as any)._telemetry.push({ lineage: { ancestorUniq: 0.0 } });
      require('../../src/neat/neat.adaptive').applyAncestorUniqAdaptive.call(
        neat as any
      );
      const eps = neat.options.multiObjective.dominanceEpsilon;
      // Assert: unchanged at baseline 0.1
      expect(eps).toBe(0.1);
    });
  });
});
