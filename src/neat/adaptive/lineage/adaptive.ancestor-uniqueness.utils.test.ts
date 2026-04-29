import {
  applyEpsilonAdjustment,
  applyLineagePressureAdjustment,
  applyUniquenessAdjustment,
  extractAncestorUniqueness,
  isCooldownSatisfied,
  resolveAdjustmentMagnitude,
  resolveUniquenessThresholds,
} from './adaptive.ancestor-uniqueness.utils';
import {
  DEFAULT_ANCESTOR_UNIQ_ADJUST,
  DEFAULT_ANCESTOR_UNIQ_HIGH_THRESHOLD,
  DEFAULT_ANCESTOR_UNIQ_LOW_THRESHOLD,
  LINEAGE_PRESSURE_INCREASE_MULTIPLIER,
  DEFAULT_LINEAGE_PRESSURE_STRENGTH,
} from '../core/adaptive.core.constants';
import type { NeatLikeWithAdaptive } from '../core/adaptive.core.types';

jest.retryTimes(2, { logErrorsBeforeRetry: true });

function createMinimalEngine(
  overrides: Partial<NeatLikeWithAdaptive> = {},
): NeatLikeWithAdaptive {
  return {
    options: {},
    population: [],
    input: 1,
    output: 1,
    generation: 0,
    ...overrides,
  };
}

describe('adaptive ancestor-uniqueness utils chapter', () => {
  describe('isCooldownSatisfied()', () => {
    describe('given config with no cooldown property', () => {
      it('uses the default cooldown and returns true when the generation gap meets it', () => {
        // Arrange: default cooldown = 5; generation=5, lastAdjust=0, gap=5 >= 5
        const engine = createMinimalEngine({ generation: 5 });

        // Act
        const result = isCooldownSatisfied(engine, {});

        // Assert
        expect(result).toBe(true);
      });
    });
  });

  describe('extractAncestorUniqueness()', () => {
    describe('given empty telemetry', () => {
      it('returns undefined when no telemetry snapshot is available', () => {
        // Arrange
        const engine = createMinimalEngine({ _telemetry: [] });

        // Act
        const result = extractAncestorUniqueness(engine);

        // Assert
        expect(result).toBeUndefined();
      });
    });
  });

  describe('resolveUniquenessThresholds()', () => {
    describe('given config with no threshold properties', () => {
      it('returns the default low and high thresholds', () => {
        // Act
        const result = resolveUniquenessThresholds({});

        // Assert
        expect(result).toEqual({
          lowThreshold: DEFAULT_ANCESTOR_UNIQ_LOW_THRESHOLD,
          highThreshold: DEFAULT_ANCESTOR_UNIQ_HIGH_THRESHOLD,
        });
      });
    });
  });

  describe('resolveAdjustmentMagnitude()', () => {
    describe('given config with no adjust property', () => {
      it('returns the default adjustment magnitude', () => {
        // Act
        const result = resolveAdjustmentMagnitude({});

        // Assert
        expect(result).toBe(DEFAULT_ANCESTOR_UNIQ_ADJUST);
      });
    });
  });

  describe('applyUniquenessAdjustment()', () => {
    describe('given an unrecognized mode', () => {
      it('leaves the engine state unchanged', () => {
        // Arrange
        const engine = createMinimalEngine();

        // Act
        applyUniquenessAdjustment(
          engine,
          { mode: 'unrecognized' },
          0.5,
          { lowThreshold: 0.3, highThreshold: 0.7 },
          0.05,
        );

        // Assert: no adjustment recorded
        expect(engine._lastAncestorUniqAdjustGen).toBeUndefined();
      });
    });
  });

  describe('applyEpsilonAdjustment()', () => {
    describe('given engine with adaptive epsilon disabled', () => {
      it('returns early without modifying the dominance epsilon', () => {
        // Arrange
        const engine = createMinimalEngine({
          options: { multiObjective: { adaptiveEpsilon: { enabled: false } } },
        });

        // Act
        applyEpsilonAdjustment(
          engine,
          0.1,
          { lowThreshold: 0.3, highThreshold: 0.7 },
          0.05,
        );

        // Assert
        expect(engine.options.multiObjective?.dominanceEpsilon).toBeUndefined();
      });
    });

    describe('given ancestor uniqueness within the acceptable range', () => {
      it('returns early without adjusting the epsilon', () => {
        // Arrange
        const engine = createMinimalEngine({
          options: {
            multiObjective: {
              adaptiveEpsilon: { enabled: true },
              dominanceEpsilon: 0.1,
            },
          },
        });

        // Act
        applyEpsilonAdjustment(
          engine,
          0.5,
          { lowThreshold: 0.3, highThreshold: 0.7 },
          0.05,
        );

        // Assert: epsilon is in range, so no adjustment
        expect(engine.options.multiObjective!.dominanceEpsilon).toBe(0.1);
      });
    });

    describe('given engine with no dominanceEpsilon property', () => {
      it('treats the missing epsilon as zero before applying the increase', () => {
        // Arrange: adaptiveEpsilon enabled, no dominanceEpsilon → ?? 0
        const engine = createMinimalEngine({
          options: { multiObjective: { adaptiveEpsilon: { enabled: true } } },
        });

        // Act: ancestorUniq=0.1 < lowThreshold=0.3 → increase by adjustMagnitude=0.05
        applyEpsilonAdjustment(
          engine,
          0.1,
          { lowThreshold: 0.3, highThreshold: 0.7 },
          0.05,
        );

        // Assert: 0 + 0.05 = 0.05
        expect(engine.options.multiObjective!.dominanceEpsilon).toBe(0.05);
      });
    });
  });

  describe('applyLineagePressureAdjustment()', () => {
    describe('given ancestor uniqueness within the acceptable range', () => {
      it('returns early without modifying lineage pressure strength', () => {
        // Arrange
        const engine = createMinimalEngine({
          options: {
            lineagePressure: { enabled: true, mode: 'spread', strength: 0.01 },
          },
        });

        // Act: ancestorUniq=0.5 is between low=0.3 and high=0.7 → no adjustment
        applyLineagePressureAdjustment(engine, 0.5, {
          lowThreshold: 0.3,
          highThreshold: 0.7,
        });

        // Assert
        expect(engine.options.lineagePressure!.strength).toBe(0.01);
      });
    });

    describe('given pre-existing lineage pressure state without a strength property', () => {
      it('defaults the strength and applies the increase multiplier', () => {
        // Arrange: lineagePressure exists (skips init) but has no strength
        const engine = createMinimalEngine({
          options: {
            lineagePressure: { enabled: true, mode: 'spread' },
          },
        });

        // Act: ancestorUniq=0.1 < lowThreshold=0.3 → increase using default strength
        applyLineagePressureAdjustment(engine, 0.1, {
          lowThreshold: 0.3,
          highThreshold: 0.7,
        });

        // Assert: DEFAULT_LINEAGE_PRESSURE_STRENGTH * LINEAGE_PRESSURE_INCREASE_MULTIPLIER
        expect(engine.options.lineagePressure!.strength).toBe(
          DEFAULT_LINEAGE_PRESSURE_STRENGTH *
            LINEAGE_PRESSURE_INCREASE_MULTIPLIER,
        );
      });
    });
  });
});
