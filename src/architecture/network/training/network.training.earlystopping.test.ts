import Network from '../network';

type TrainingResult = { error?: number; iterations?: number };

function createSingleInputOutputNetwork(seed: number): Network {
  return new Network(1, 1, { seed });
}

function createSingleZeroDataset() {
  return [{ input: [0], output: [0] }];
}

describe('network training chapter', () => {
  describe('early stopping behavior', () => {
    describe('given a moving-average window smooths an oscillating error sequence', () => {
      describe('when train() runs with a stricter target than the smoothed series reaches', () => {
        it('uses the full requested iteration budget', () => {
          // Arrange
          const network = createSingleInputOutputNetwork(130);
          const trainingDataset = createSingleZeroDataset();
          const rawErrors = Array.from({ length: 20 }, (_, errorIndex) =>
            errorIndex % 2 === 0 ? 0.5 : 0.7,
          );
          let errorIndex = 0;
          const cost = () =>
            rawErrors[Math.min(errorIndex++, rawErrors.length - 1)];

          // Act
          const trainingSummary: TrainingResult = network.train(
            trainingDataset,
            {
              iterations: 20,
              rate: 0.1,
              cost,
              movingAverageWindow: 4,
              error: 0.4,
              optimizer: 'sgd',
            },
          );

          // Assert
          expect(trainingSummary.iterations ?? 0).toBe(20);
        });
      });
    });

    describe('given EMA smoothing sees a descending raw-error sequence', () => {
      describe('when train() completes', () => {
        it('reports a final error below the initial raw value', () => {
          // Arrange
          const network = createSingleInputOutputNetwork(131);
          const trainingDataset = createSingleZeroDataset();
          const rawErrors = [0.9, 0.8, 0.7, 0.6, 0.5];
          let errorIndex = 0;
          const cost = () =>
            rawErrors[Math.min(errorIndex++, rawErrors.length - 1)];

          // Act
          const trainingSummary: TrainingResult = network.train(
            trainingDataset,
            {
              iterations: 5,
              rate: 0.1,
              cost,
              movingAverageWindow: 5,
              movingAverageType: 'ema',
              error: 0.4,
              optimizer: 'sgd',
            },
          );

          // Assert
          expect(trainingSummary.error ?? Infinity).toBeLessThan(0.9);
        });
      });
    });

    describe('given earlyStopPatience sees no improvement at all', () => {
      describe('when train() runs', () => {
        it('stops before the max iteration count', () => {
          // Arrange
          const network = createSingleInputOutputNetwork(132);
          const trainingDataset = createSingleZeroDataset();
          const cost = () => 0.4;

          // Act
          const trainingSummary: TrainingResult = network.train(
            trainingDataset,
            {
              iterations: 50,
              rate: 0.1,
              cost,
              earlyStopPatience: 3,
              error: 0.0,
              optimizer: 'sgd',
            },
          );

          // Assert
          expect(trainingSummary.iterations ?? 0).toBeLessThan(50);
        });
      });
    });

    describe('given intermittent improvements exceed the minimum delta', () => {
      describe('when train() runs with earlyStopPatience enabled', () => {
        it('continues past the first patience window', () => {
          // Arrange
          const network = createSingleInputOutputNetwork(133);
          const trainingDataset = createSingleZeroDataset();
          const rawErrors = [0.5, 0.49, 0.49, 0.48, 0.48, 0.47];
          let errorIndex = 0;
          const cost = () =>
            rawErrors[Math.min(errorIndex++, rawErrors.length - 1)];

          // Act
          const trainingSummary: TrainingResult = network.train(
            trainingDataset,
            {
              iterations: 30,
              rate: 0.1,
              cost,
              earlyStopPatience: 2,
              earlyStopMinDelta: 0.0005,
              error: 0.0,
              optimizer: 'sgd',
            },
          );

          // Assert
          expect(trainingSummary.iterations ?? 0).toBeGreaterThan(4);
        });
      });
    });
  });
});
