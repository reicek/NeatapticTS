import Network from '../network';

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
