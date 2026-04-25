/**
 * Owner-local test shelf for the `evaluate.ts` root facade.
 *
 * Coverage targets:
 * - All 21 re-exported constants from `./shared/evaluate.constants` (each getter
 *   in the compiled output must be accessed through the facade module to be counted).
 * - The `evaluate` function's `this.options || {}` fallback branch, triggered by
 *   passing a controller whose `options` field is undefined at runtime.
 */
import {
  evaluate,
  NOVELTY_DEFAULT_NEIGHBORS,
  NOVELTY_DEFAULT_BLEND,
  NOVELTY_ARCHIVE_CAP,
  ENTROPY_VAR_TARGET_DEFAULT,
  ENTROPY_VAR_ADJUST_DEFAULT,
  ENTROPY_VAR_MIN_SIGMA_DEFAULT,
  ENTROPY_VAR_MAX_SIGMA_DEFAULT,
  ENTROPY_VAR_LOW_BAND,
  ENTROPY_VAR_HIGH_BAND,
  ENTROPY_TARGET_DEFAULT,
  ENTROPY_DEADBAND_DEFAULT,
  ENTROPY_ADJUST_DEFAULT,
  COMPAT_THRESHOLD_DEFAULT,
  COMPAT_MIN_THRESHOLD_DEFAULT,
  COMPAT_MAX_THRESHOLD_DEFAULT,
  AUTO_COEFF_ADJUST_DEFAULT,
  AUTO_COEFF_MIN_DEFAULT,
  AUTO_COEFF_MAX_DEFAULT,
  DISTANCE_COEFF_DEFAULT,
  VARIANCE_DECREASE_THRESHOLD,
  VARIANCE_INCREASE_THRESHOLD,
} from './evaluate';
import type { NeatControllerForEval } from './shared/evaluate.types';

// ---------------------------------------------------------------------------
// Re-exported constant accessor coverage
// Each test imports the constant through the facade module so Istanbul counts
// the re-export getter function as hit.
// ---------------------------------------------------------------------------

describe('evaluate.ts root facade', () => {
  describe('re-exported novelty constants', () => {
    it('exports NOVELTY_DEFAULT_NEIGHBORS as a number', () => {
      expect(typeof NOVELTY_DEFAULT_NEIGHBORS).toBe('number');
    });

    it('exports NOVELTY_DEFAULT_BLEND as a number', () => {
      expect(typeof NOVELTY_DEFAULT_BLEND).toBe('number');
    });

    it('exports NOVELTY_ARCHIVE_CAP as a number', () => {
      expect(typeof NOVELTY_ARCHIVE_CAP).toBe('number');
    });
  });

  describe('re-exported entropy-variance constants', () => {
    it('exports ENTROPY_VAR_TARGET_DEFAULT as a number', () => {
      expect(typeof ENTROPY_VAR_TARGET_DEFAULT).toBe('number');
    });

    it('exports ENTROPY_VAR_ADJUST_DEFAULT as a number', () => {
      expect(typeof ENTROPY_VAR_ADJUST_DEFAULT).toBe('number');
    });

    it('exports ENTROPY_VAR_MIN_SIGMA_DEFAULT as a number', () => {
      expect(typeof ENTROPY_VAR_MIN_SIGMA_DEFAULT).toBe('number');
    });

    it('exports ENTROPY_VAR_MAX_SIGMA_DEFAULT as a number', () => {
      expect(typeof ENTROPY_VAR_MAX_SIGMA_DEFAULT).toBe('number');
    });

    it('exports ENTROPY_VAR_LOW_BAND as a number', () => {
      expect(typeof ENTROPY_VAR_LOW_BAND).toBe('number');
    });

    it('exports ENTROPY_VAR_HIGH_BAND as a number', () => {
      expect(typeof ENTROPY_VAR_HIGH_BAND).toBe('number');
    });
  });

  describe('re-exported entropy-compat constants', () => {
    it('exports ENTROPY_TARGET_DEFAULT as a number', () => {
      expect(typeof ENTROPY_TARGET_DEFAULT).toBe('number');
    });

    it('exports ENTROPY_DEADBAND_DEFAULT as a number', () => {
      expect(typeof ENTROPY_DEADBAND_DEFAULT).toBe('number');
    });

    it('exports ENTROPY_ADJUST_DEFAULT as a number', () => {
      expect(typeof ENTROPY_ADJUST_DEFAULT).toBe('number');
    });
  });

  describe('re-exported compat-threshold constants', () => {
    it('exports COMPAT_THRESHOLD_DEFAULT as a number', () => {
      expect(typeof COMPAT_THRESHOLD_DEFAULT).toBe('number');
    });

    it('exports COMPAT_MIN_THRESHOLD_DEFAULT as a number', () => {
      expect(typeof COMPAT_MIN_THRESHOLD_DEFAULT).toBe('number');
    });

    it('exports COMPAT_MAX_THRESHOLD_DEFAULT as a number', () => {
      expect(typeof COMPAT_MAX_THRESHOLD_DEFAULT).toBe('number');
    });
  });

  describe('re-exported auto-distance constants', () => {
    it('exports AUTO_COEFF_ADJUST_DEFAULT as a number', () => {
      expect(typeof AUTO_COEFF_ADJUST_DEFAULT).toBe('number');
    });

    it('exports AUTO_COEFF_MIN_DEFAULT as a number', () => {
      expect(typeof AUTO_COEFF_MIN_DEFAULT).toBe('number');
    });

    it('exports AUTO_COEFF_MAX_DEFAULT as a number', () => {
      expect(typeof AUTO_COEFF_MAX_DEFAULT).toBe('number');
    });

    it('exports DISTANCE_COEFF_DEFAULT as a number', () => {
      expect(typeof DISTANCE_COEFF_DEFAULT).toBe('number');
    });
  });

  describe('re-exported variance-threshold constants', () => {
    it('exports VARIANCE_DECREASE_THRESHOLD as a number', () => {
      expect(typeof VARIANCE_DECREASE_THRESHOLD).toBe('number');
    });

    it('exports VARIANCE_INCREASE_THRESHOLD as a number', () => {
      expect(typeof VARIANCE_INCREASE_THRESHOLD).toBe('number');
    });
  });

  // ---------------------------------------------------------------------------
  // evaluate() branch coverage
  // ---------------------------------------------------------------------------

  describe('evaluate()', () => {
    it('falls back to empty options when this.options is undefined', async () => {
      // Arrange: controller with options=undefined to trigger the `|| {}` fallback
      // branch. An empty population means the fitness delegate is never called.
      const controller = {
        // eslint-disable-next-line @typescript-eslint/no-explicit-any -- deliberate undefined to test fallback branch
        options: undefined as any,
        population: [],
        fitness: async () => {},
      } as unknown as NeatControllerForEval;

      // Act + Assert: the function resolves without throwing.
      await expect(evaluate.call(controller)).resolves.toBeUndefined();
    });

    it('resolves normally when options is provided', async () => {
      // Arrange: minimal controller with all features disabled.
      const controller = {
        options: {},
        population: [],
        fitness: async () => {},
      } as unknown as NeatControllerForEval;

      // Act + Assert
      await expect(evaluate.call(controller)).resolves.toBeUndefined();
    });
  });
});
