import {
  createCosineAnnealingRateSchedule,
  createCosineAnnealingWarmRestartsSchedule,
  createExponentialRateSchedule,
  createInverseRateSchedule,
  createLinearWarmupDecaySchedule,
  createReduceOnPlateauSchedule,
  createStepRateSchedule,
  DEFAULT_DECAY_STEP_SIZE,
  DEFAULT_EXPONENTIAL_DECAY_FACTOR,
  DEFAULT_INVERSE_DECAY_FACTOR,
  DEFAULT_INVERSE_POWER,
  DEFAULT_STEP_DECAY_FACTOR,
} from './rate.utils';

jest.retryTimes(2, { logErrorsBeforeRetry: true });

describe('rate utilities chapter (direct API)', () => {
  describe('createStepRateSchedule()', () => {
    describe('given no arguments', () => {
      it('applies the default decay factor and step size after one decay interval', () => {
        // Act
        const schedule = createStepRateSchedule();

        // Assert: after DEFAULT_DECAY_STEP_SIZE iterations, one decay step has elapsed
        expect(schedule(0.1, DEFAULT_DECAY_STEP_SIZE)).toBeCloseTo(
          0.1 * DEFAULT_STEP_DECAY_FACTOR,
        );
      });
    });
  });

  describe('createExponentialRateSchedule()', () => {
    describe('given no arguments', () => {
      it('applies the default exponential decay factor at iteration 10', () => {
        // Act
        const schedule = createExponentialRateSchedule();

        // Assert
        expect(schedule(0.1, 10)).toBeCloseTo(
          0.1 * DEFAULT_EXPONENTIAL_DECAY_FACTOR ** 10,
        );
      });
    });
  });

  describe('createInverseRateSchedule()', () => {
    describe('given no arguments', () => {
      it('applies the default inverse decay factor and power at iteration 10', () => {
        // Act
        const schedule = createInverseRateSchedule();

        // Assert
        expect(schedule(0.1, 10)).toBeCloseTo(
          0.1 /
            (1 + DEFAULT_INVERSE_DECAY_FACTOR * 10 ** DEFAULT_INVERSE_POWER),
        );
      });
    });
  });

  describe('createCosineAnnealingRateSchedule()', () => {
    describe('given no arguments', () => {
      it('returns the base rate at the start of the default cycle', () => {
        // Act
        const schedule = createCosineAnnealingRateSchedule();

        // Assert: cosine at iteration 0 peaks at baseRate
        expect(schedule(0.1, 0)).toBeCloseTo(0.1);
      });
    });
  });

  describe('createCosineAnnealingWarmRestartsSchedule()', () => {
    describe('given no arguments', () => {
      it('returns the base rate at the start of the default initial period', () => {
        // Act
        const schedule = createCosineAnnealingWarmRestartsSchedule();

        // Assert: first iteration starts at baseRate
        expect(schedule(0.1, 0)).toBeCloseTo(0.1, 10);
      });
    });
  });

  describe('createLinearWarmupDecaySchedule()', () => {
    describe('given only totalStepCount with warmupStepCount and endRate omitted', () => {
      it('reaches zero at totalStepCount using the default zero end rate', () => {
        // Act: endRate defaults to 0; warmupStepCount defaults to 10% of totalStepCount
        const schedule = createLinearWarmupDecaySchedule(10);

        // Assert: at totalStepCount the schedule clamps to endRate=0
        expect(schedule(0.1, 10)).toBe(0);
      });
    });
  });

  describe('createReduceOnPlateauSchedule()', () => {
    describe('given no options', () => {
      it('returns the base rate when called without a monitored error signal', () => {
        // Act: options defaults to {}; lastError omitted triggers the undefined path
        const schedule = createReduceOnPlateauSchedule();

        // Assert: no error monitoring → schedule stays at base rate
        expect(schedule(0.1, 0)).toBe(0.1);
      });
    });
  });
});
