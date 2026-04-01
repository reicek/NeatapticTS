import Network from '../network';

type MixedPrecisionStats = {
  lossScale: number;
  mp: {
    lastOverflowStep: number;
    overflowCount: number;
  };
};

describe('network training chapter', () => {
  describe('mixed precision overflow handling', () => {
    describe('given a forced overflow occurs after an initial mixed-precision step', () => {
      let trainingStatsAfterOverflow: MixedPrecisionStats;
      let trainingStatsBeforeOverflow: MixedPrecisionStats;

      beforeAll(() => {
        const network = new Network(2, 1, { seed: 42 });
        const trainingDataset = [{ input: [0, 0], output: [0] }];

        network.train(trainingDataset, {
          iterations: 1,
          rate: 0.01,
          optimizer: 'adam',
          mixedPrecision: {
            lossScale: 1024,
            dynamic: { minScale: 1, maxScale: 2048, increaseEvery: 5 },
          },
        });
        trainingStatsBeforeOverflow =
          network.getTrainingStats() as MixedPrecisionStats;

        network.testForceOverflow();
        network.train(trainingDataset, {
          iterations: 1,
          rate: 0.01,
          optimizer: 'adam',
          mixedPrecision: {
            lossScale: trainingStatsBeforeOverflow.lossScale,
            dynamic: { minScale: 1, maxScale: 2048, increaseEvery: 5 },
          },
        });
        trainingStatsAfterOverflow =
          network.getTrainingStats() as MixedPrecisionStats;
      });

      describe('when the overflow counters are compared', () => {
        it('increments the overflow count', () => {
          // Arrange
          const overflowCountBefore =
            trainingStatsBeforeOverflow.mp.overflowCount;

          // Act
          const overflowCountAfter =
            trainingStatsAfterOverflow.mp.overflowCount;

          // Assert
          expect(overflowCountAfter).toBeGreaterThan(overflowCountBefore);
        });
      });

      describe('when the recorded overflow step is inspected', () => {
        it('stores a non-negative overflow step index', () => {
          // Arrange
          const lastOverflowStep =
            trainingStatsAfterOverflow.mp.lastOverflowStep;

          // Act
          const overflowStepWasRecorded = lastOverflowStep >= 0;

          // Assert
          expect(overflowStepWasRecorded).toBe(true);
        });
      });

      describe('when the loss scales are compared before and after overflow', () => {
        it('prevents the loss scale from increasing', () => {
          // Arrange
          const lossScaleBefore = trainingStatsBeforeOverflow.lossScale;

          // Act
          const lossScaleAfter = trainingStatsAfterOverflow.lossScale;

          // Assert
          expect(lossScaleAfter).toBeLessThanOrEqual(lossScaleBefore);
        });
      });
    });

    describe('given the lightweight dynamic mixed-precision path is used', () => {
      describe('when a forced overflow is injected between two steps', () => {
        it('still exposes the overflow counter field', () => {
          // Arrange
          const network = new Network(1, 1, { seed: 9 });
          const trainingDataset = [{ input: [1], output: [2] }];

          network.train(trainingDataset, {
            iterations: 1,
            batchSize: 1,
            mixedPrecision: {
              dynamic: true,
              lossScale: 128,
              scaleFactor: 2,
              scaleWindow: 1,
            },
          });
          network.testForceOverflow();
          network.train(trainingDataset, {
            iterations: 1,
            batchSize: 1,
            mixedPrecision: {
              dynamic: true,
              lossScale: 128,
              scaleFactor: 2,
              scaleWindow: 1,
            },
          });

          const trainingStats =
            network.getTrainingStats() as MixedPrecisionStats;

          // Act
          const overflowCount = trainingStats.mp.overflowCount;

          // Assert
          expect(overflowCount).toBeGreaterThanOrEqual(0);
        });
      });
    });
  });
});
