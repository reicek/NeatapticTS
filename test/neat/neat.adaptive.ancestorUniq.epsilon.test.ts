import Neat from '../../src/neat';

/** Tests ancestor uniqueness adaptive epsilon adjustments (both directions). */
describe('Ancestor Uniqueness Adaptive (epsilon mode)', () => {
  describe('adjusts epsilon when uniqueness below low threshold', () => {
    const fitness = () => 1;
    const neat = new Neat(2, 1, fitness, {
      popsize: 3,
      seed: 930,
      ancestorUniqAdaptive: {
        enabled: true,
        mode: 'epsilon',
        lowThreshold: 0.4,
        highThreshold: 0.6,
        adjust: 0.05,
        cooldown: 0,
      },
      multiObjective: {
        adaptiveEpsilon: { enabled: true },
        dominanceEpsilon: 0.1,
      },
    });
    test('epsilon increases when below low threshold', () => {
      // Arrange: fake telemetry with low ancestor uniqueness
      (neat as any)._telemetry.push({ lineage: { ancestorUniq: 0.1 } });
      require('../../src/neat/neat.adaptive').applyAncestorUniqAdaptive.call(
        neat as any
      );
      // Act: capture epsilon
      const eps = neat.options.multiObjective.dominanceEpsilon;
      // Assert: epsilon moved upwards
      expect(eps).toBeGreaterThan(0.1);
    });
  });
  describe('decreases epsilon when above high threshold', () => {
    const fitness = () => 1;
    const neat = new Neat(2, 1, fitness, {
      popsize: 3,
      seed: 931,
      ancestorUniqAdaptive: {
        enabled: true,
        mode: 'epsilon',
        lowThreshold: 0.2,
        highThreshold: 0.3,
        adjust: 0.05,
        cooldown: 0,
      },
      multiObjective: {
        adaptiveEpsilon: { enabled: true },
        dominanceEpsilon: 0.2,
      },
    });
    test('epsilon decreases when above high threshold', () => {
      // Arrange: telemetry with high ancestor uniqueness
      (neat as any)._telemetry.push({ lineage: { ancestorUniq: 0.9 } });
      require('../../src/neat/neat.adaptive').applyAncestorUniqAdaptive.call(
        neat as any
      );
      // Act: capture epsilon
      const eps = neat.options.multiObjective.dominanceEpsilon;
      // Assert: epsilon decreased (clamped to non-negative)
      expect(eps).toBeLessThan(0.2);
    });
  });
});
