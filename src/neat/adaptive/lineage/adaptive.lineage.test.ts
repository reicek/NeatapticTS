import {
  applyAncestorUniqAdaptive,
  applyUniquenessAdjustment,
  extractAncestorUniqueness,
  isCooldownSatisfied,
  resolveUniquenessThresholds,
} from './adaptive.lineage';
import type {
  AncestorUniqAdaptiveConfig,
  NeatLikeWithAdaptive,
} from '../core/adaptive.core.types';

function createEpsilonController(input: {
  ancestorUniqAdaptive: AncestorUniqAdaptiveConfig;
  ancestorUniq?: number;
  generation?: number;
  lastAdjustGeneration?: number;
  dominanceEpsilon: number;
}): NeatLikeWithAdaptive {
  const telemetry =
    input.ancestorUniq === undefined
      ? []
      : [{ lineage: { ancestorUniq: input.ancestorUniq } }];

  return {
    options: {
      ancestorUniqAdaptive: input.ancestorUniqAdaptive,
      multiObjective: {
        adaptiveEpsilon: { enabled: true },
        dominanceEpsilon: input.dominanceEpsilon,
      },
    },
    population: [],
    input: 2,
    output: 1,
    generation: input.generation ?? 0,
    _lastAncestorUniqAdjustGen: input.lastAdjustGeneration,
    _telemetry: telemetry,
  };
}

function createLineagePressureController(input: {
  ancestorUniqAdaptive: AncestorUniqAdaptiveConfig;
  ancestorUniq: number;
}): NeatLikeWithAdaptive {
  return {
    options: {
      ancestorUniqAdaptive: input.ancestorUniqAdaptive,
    },
    population: [],
    input: 2,
    output: 1,
    generation: 0,
    _telemetry: [{ lineage: { ancestorUniq: input.ancestorUniq } }],
  };
}

describe('neat adaptive lineage chapter', () => {
  describe('adaptive.lineage facade re-exports', () => {
    it('exports applyAncestorUniqAdaptive as a function', () => {
      expect(typeof applyAncestorUniqAdaptive).toBe('function');
    });

    it('exports extractAncestorUniqueness as a function', () => {
      expect(typeof extractAncestorUniqueness).toBe('function');
    });

    it('exports isCooldownSatisfied as a function', () => {
      expect(typeof isCooldownSatisfied).toBe('function');
    });

    it('exports resolveUniquenessThresholds as a function', () => {
      expect(typeof resolveUniquenessThresholds).toBe('function');
    });

    it('exports applyUniquenessAdjustment as a function', () => {
      expect(typeof applyUniquenessAdjustment).toBe('function');
    });
  });

  describe('applyAncestorUniqAdaptive', () => {
    describe('given epsilon mode', () => {
      describe('when the cooldown window has not elapsed yet', () => {
        it('keeps the dominance epsilon and last-adjust generation unchanged', () => {
          // Arrange
          const adaptiveController = createEpsilonController({
            ancestorUniqAdaptive: {
              enabled: true,
              mode: 'epsilon',
              lowThreshold: 0.2,
              highThreshold: 0.8,
              adjust: 0.05,
              cooldown: 5,
            },
            ancestorUniq: 0,
            generation: 12,
            lastAdjustGeneration: 9,
            dominanceEpsilon: 0.1,
          });

          // Act
          applyAncestorUniqAdaptive.call(adaptiveController);

          // Assert
          expect({
            dominanceEpsilon:
              adaptiveController.options.multiObjective?.dominanceEpsilon,
            lastAdjustGeneration: adaptiveController._lastAncestorUniqAdjustGen,
          }).toEqual({
            dominanceEpsilon: 0.1,
            lastAdjustGeneration: 9,
          });
        });
      });

      describe('when ancestor uniqueness falls below the low threshold', () => {
        it('raises the dominance epsilon by the configured adjustment amount', () => {
          // Arrange
          const adaptiveController = createEpsilonController({
            ancestorUniqAdaptive: {
              enabled: true,
              mode: 'epsilon',
              lowThreshold: 0.4,
              highThreshold: 0.6,
              adjust: 0.05,
              cooldown: 0,
            },
            ancestorUniq: 0.1,
            dominanceEpsilon: 0.1,
          });

          // Act
          applyAncestorUniqAdaptive.call(adaptiveController);

          // Assert
          expect({
            dominanceEpsilon: Number(
              adaptiveController.options.multiObjective?.dominanceEpsilon?.toFixed(
                2,
              ),
            ),
            lastAdjustGeneration: adaptiveController._lastAncestorUniqAdjustGen,
          }).toEqual({
            dominanceEpsilon: 0.15,
            lastAdjustGeneration: 0,
          });
        });
      });

      describe('when ancestor uniqueness rises above the high threshold', () => {
        it('lowers the dominance epsilon by the configured adjustment amount', () => {
          // Arrange
          const adaptiveController = createEpsilonController({
            ancestorUniqAdaptive: {
              enabled: true,
              mode: 'epsilon',
              lowThreshold: 0.2,
              highThreshold: 0.3,
              adjust: 0.05,
              cooldown: 0,
            },
            ancestorUniq: 0.9,
            dominanceEpsilon: 0.2,
          });

          // Act
          applyAncestorUniqAdaptive.call(adaptiveController);

          // Assert
          expect({
            dominanceEpsilon: Number(
              adaptiveController.options.multiObjective?.dominanceEpsilon?.toFixed(
                2,
              ),
            ),
            lastAdjustGeneration: adaptiveController._lastAncestorUniqAdjustGen,
          }).toEqual({
            dominanceEpsilon: 0.15,
            lastAdjustGeneration: 0,
          });
        });
      });
    });

    describe('given lineage pressure mode', () => {
      describe('when ancestor uniqueness falls below the low threshold', () => {
        it('creates spread-mode lineage pressure and increases its strength', () => {
          // Arrange
          const adaptiveController = createLineagePressureController({
            ancestorUniqAdaptive: {
              enabled: true,
              mode: 'lineagePressure',
              lowThreshold: 0.5,
              highThreshold: 0.9,
              adjust: 0.05,
              cooldown: 0,
            },
            ancestorUniq: 0.2,
          });

          // Act
          applyAncestorUniqAdaptive.call(adaptiveController);

          // Assert
          expect(adaptiveController.options.lineagePressure).toEqual({
            enabled: true,
            mode: 'spread',
            strength: 0.0115,
          });
        });
      });

      describe('when ancestor uniqueness rises above the high threshold', () => {
        it('creates spread-mode lineage pressure and relaxes its default strength', () => {
          // Arrange
          const adaptiveController = createLineagePressureController({
            ancestorUniqAdaptive: {
              enabled: true,
              mode: 'lineagePressure',
              lowThreshold: 0.1,
              highThreshold: 0.2,
              adjust: 0.05,
              cooldown: 0,
            },
            ancestorUniq: 0.95,
          });

          // Act
          applyAncestorUniqAdaptive.call(adaptiveController);

          // Assert
          expect(adaptiveController.options.lineagePressure).toEqual({
            enabled: true,
            mode: 'spread',
            strength: 0.009000000000000001,
          });
        });
      });
    });
  });
});
