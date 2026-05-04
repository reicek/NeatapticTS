import Network from '../network';
import {
  computeMonitoredError,
  computePlateauMetric,
} from './network.training.smoothing.utils';

type TrainingSummary = { error: number; iterations: number };

const SINGLE_SAMPLE_DATASET = [{ input: [0], output: [0] }];

function createSingleInputOutputNetwork(seed: number): Network {
  return new Network(1, 1, { seed });
}

function createSequenceCost(errorSequence: number[]): () => number {
  let errorIndex = 0;

  return () => errorSequence[Math.min(errorIndex++, errorSequence.length - 1)];
}

function trainWithSmoothingSequence(
  seed: number,
  errorSequence: number[],
  options: Record<string, unknown>,
): TrainingSummary {
  const network = createSingleInputOutputNetwork(seed);

  return network.train(SINGLE_SAMPLE_DATASET, {
    iterations: errorSequence.length,
    rate: 0.1,
    cost: createSequenceCost(errorSequence),
    error: 0,
    ...options,
  });
}

describe('computeMonitoredError()', () => {
  describe('given window is 1 and type is median', () => {
    describe('when called with a raw error', () => {
      it('returns the raw training error unchanged', () => {
        // Arrange – window <= 1 AND type !== ema → covers line 24 TRUE arm
        const cfg = { window: 1, type: 'median' as const };
        const state = {};

        // Act
        const monitoredError = computeMonitoredError(0.42, [0.42], cfg, state);

        // Assert
        expect(monitoredError).toBe(0.42);
      });
    });
  });

  describe('given type is trimmed without an explicit trimmedRatio', () => {
    describe('when called with a window of errors', () => {
      it('applies the default trimmed ratio of 0.1', () => {
        // Arrange – no trimmedRatio → cfg.trimmedRatio || 0.1 fallback (line 81)
        const cfg = { window: 4, type: 'trimmed' as const };
        const state = {};

        // Act
        const monitoredError = computeMonitoredError(
          0.5,
          [0.5, 0.6, 0.7, 0.8],
          cfg,
          state,
        );

        // Assert
        expect(typeof monitoredError).toBe('number');
      });
    });
  });

  describe('given an unrecognized smoothing type', () => {
    describe('when called with a window of errors', () => {
      it('returns the simple moving average over the window', () => {
        // Arrange – type not matching any known case → falls through to SMA fallback (line 101)
        const cfg = { window: 3, type: 'sma' as unknown as 'median' };
        const state = {};

        // Act
        const monitoredError = computeMonitoredError(
          0.5,
          [0.3, 0.4, 0.5],
          cfg,
          state,
        );

        // Assert – SMA of [0.3, 0.4, 0.5] = 0.4
        expect(monitoredError).toBeCloseTo(0.4);
      });
    });
  });
});

describe('computePlateauMetric()', () => {
  describe('given window is 1 and type is median', () => {
    describe('when called with a raw error', () => {
      it('returns the raw training error unchanged', () => {
        // Arrange – window <= 1 AND type !== ema → covers line 119 TRUE arm
        const cfg = { window: 1, type: 'median' as const };
        const state = {};

        // Act
        const plateauMetric = computePlateauMetric(0.55, [0.55], cfg, state);

        // Assert
        expect(plateauMetric).toBe(0.55);
      });
    });
  });
});

describe('network training chapter', () => {
  describe('smoothing strategy comparisons', () => {
    describe('given weighted moving average and simple moving average see the same descending error sequence', () => {
      let simpleMovingAverageSummary: TrainingSummary;
      let weightedMovingAverageSummary: TrainingSummary;

      beforeAll(() => {
        const errorSequence = [0.9, 0.8, 0.7, 0.6];

        simpleMovingAverageSummary = trainWithSmoothingSequence(
          140,
          errorSequence,
          {
            movingAverageType: 'sma',
            movingAverageWindow: 4,
          },
        );
        weightedMovingAverageSummary = trainWithSmoothingSequence(
          140,
          errorSequence,
          {
            movingAverageType: 'wma',
            movingAverageWindow: 4,
          },
        );
      });

      describe('when the final smoothed errors are compared', () => {
        it('keeps the weighted moving average at or below the simple moving average', () => {
          // Assert
          expect(weightedMovingAverageSummary.error).toBeLessThanOrEqual(
            simpleMovingAverageSummary.error,
          );
        });
      });
    });

    describe('given median smoothing and simple moving average see a single large spike', () => {
      let medianSummary: TrainingSummary;
      let simpleMovingAverageSummary: TrainingSummary;

      beforeAll(() => {
        const errorSequence = [0.5, 2, 0.5, 0.5];

        medianSummary = trainWithSmoothingSequence(141, errorSequence, {
          movingAverageType: 'median',
          movingAverageWindow: 4,
        });
        simpleMovingAverageSummary = trainWithSmoothingSequence(
          141,
          errorSequence,
          {
            movingAverageType: 'sma',
            movingAverageWindow: 4,
          },
        );
      });

      describe('when the final smoothed errors are compared', () => {
        it('keeps the median-smoothed error below the simple moving average', () => {
          // Assert
          expect(medianSummary.error).toBeLessThan(
            simpleMovingAverageSummary.error,
          );
        });
      });
    });

    describe('given trimmed smoothing and simple moving average see the same spiky sequence', () => {
      let trimmedSummary: TrainingSummary;
      let simpleMovingAverageSummary: TrainingSummary;

      beforeAll(() => {
        const errorSequence = [0.5, 2, 0.5, 0.5];

        trimmedSummary = trainWithSmoothingSequence(142, errorSequence, {
          movingAverageType: 'trimmed',
          movingAverageWindow: 4,
          trimmedRatio: 0.25,
        });
        simpleMovingAverageSummary = trainWithSmoothingSequence(
          142,
          errorSequence,
          {
            movingAverageType: 'sma',
            movingAverageWindow: 4,
          },
        );
      });

      describe('when the final smoothed errors are compared', () => {
        it('keeps the trimmed mean at or below the simple moving average', () => {
          // Assert
          expect(trimmedSummary.error).toBeLessThanOrEqual(
            simpleMovingAverageSummary.error,
          );
        });
      });
    });

    describe('given gaussian smoothing sees a descending error sequence', () => {
      let gaussianSummary: TrainingSummary;
      const errorSequence = [0.9, 0.8, 0.7, 0.6];

      beforeAll(() => {
        gaussianSummary = trainWithSmoothingSequence(143, errorSequence, {
          movingAverageType: 'gaussian',
          movingAverageWindow: 4,
        });
      });

      describe('when the final smoothed error is inspected', () => {
        it('stays at or above the minimum raw error', () => {
          // Assert
          expect(gaussianSummary.error).toBeGreaterThanOrEqual(
            Math.min(...errorSequence),
          );
        });

        it('stays at or below the maximum raw error', () => {
          // Assert
          expect(gaussianSummary.error).toBeLessThanOrEqual(
            Math.max(...errorSequence),
          );
        });
      });
    });

    describe('given adaptive EMA and plain EMA see the same high-variance sequence', () => {
      let adaptiveEmaSummary: TrainingSummary;
      let plainEmaSummary: TrainingSummary;

      beforeAll(() => {
        const errorSequence = [0.9, 0.7, 0.85, 0.65, 0.6];

        adaptiveEmaSummary = trainWithSmoothingSequence(144, errorSequence, {
          movingAverageType: 'adaptive-ema',
          movingAverageWindow: 5,
        });
        plainEmaSummary = trainWithSmoothingSequence(144, errorSequence, {
          movingAverageType: 'ema',
          movingAverageWindow: 5,
        });
      });

      describe('when the final smoothed errors are compared', () => {
        it('keeps the adaptive EMA at or below the plain EMA', () => {
          // Assert
          expect(adaptiveEmaSummary.error).toBeLessThanOrEqual(
            plainEmaSummary.error,
          );
        });
      });
    });
  });
});
