import Network from '../network';
import { applyGradientClippingImpl, trainImpl } from './network.training.utils';

type TrainingDataset = Parameters<typeof trainImpl>[1];

interface NetworkInternals {
  _lastGradClipGroupCount: number;
  _mixedPrecision: { enabled: boolean; lossScale: number };
}

function createSingleSampleDataset(outputValue: number): TrainingDataset {
  return [{ input: [0.1], output: [outputValue] }];
}

function createSingleInputOutputNetwork(seed: number): Network {
  return new Network(1, 1, { seed });
}

function getNetworkInternal<Key extends keyof NetworkInternals>(
  network: Network,
  key: Key,
): NetworkInternals[Key] {
  return Reflect.get(network, key) as NetworkInternals[Key];
}

describe('network training chapter', () => {
  describe('advanced training orchestration', () => {
    describe('moving-average monitoring strategies', () => {
      describe('given EMA smoothing is configured', () => {
        describe('when trainImpl runs for two iterations', () => {
          it('completes the requested iteration count', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(120);
            const trainingDataset = createSingleSampleDataset(0.2);

            // Act
            const trainingSummary = trainImpl(network, trainingDataset, {
              iterations: 2,
              rate: 0.1,
              movingAverageType: 'ema',
              movingAverageWindow: 3,
            });

            // Assert
            expect(trainingSummary.iterations).toBe(2);
          });
        });
      });

      describe('given adaptive EMA smoothing is configured', () => {
        describe('when trainImpl runs for two iterations', () => {
          it('completes the requested iteration count', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(121);
            const trainingDataset = createSingleSampleDataset(0.3);

            // Act
            const trainingSummary = trainImpl(network, trainingDataset, {
              iterations: 2,
              rate: 0.1,
              movingAverageType: 'adaptive-ema',
              movingAverageWindow: 3,
            });

            // Assert
            expect(trainingSummary.iterations).toBe(2);
          });
        });
      });

      describe('given gaussian smoothing is configured', () => {
        describe('when trainImpl runs for two iterations', () => {
          it('completes the requested iteration count', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(122);
            const trainingDataset = createSingleSampleDataset(0.25);

            // Act
            const trainingSummary = trainImpl(network, trainingDataset, {
              iterations: 2,
              rate: 0.1,
              movingAverageType: 'gaussian',
              movingAverageWindow: 4,
            });

            // Assert
            expect(trainingSummary.iterations).toBe(2);
          });
        });
      });

      describe('given trimmed smoothing is configured', () => {
        describe('when trainImpl runs for three iterations', () => {
          it('completes the requested iteration count', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(123);
            const trainingDataset = createSingleSampleDataset(0.25);

            // Act
            const trainingSummary = trainImpl(network, trainingDataset, {
              iterations: 3,
              rate: 0.1,
              movingAverageType: 'trimmed',
              movingAverageWindow: 5,
              trimmedRatio: 0.2,
            });

            // Assert
            expect(trainingSummary.iterations).toBe(3);
          });
        });
      });

      describe('given weighted moving average smoothing is configured', () => {
        describe('when trainImpl runs for two iterations', () => {
          it('completes the requested iteration count', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(124);
            const trainingDataset = createSingleSampleDataset(0.25);

            // Act
            const trainingSummary = trainImpl(network, trainingDataset, {
              iterations: 2,
              rate: 0.1,
              movingAverageType: 'wma',
              movingAverageWindow: 4,
            });

            // Assert
            expect(trainingSummary.iterations).toBe(2);
          });
        });
      });
    });

    describe('gradient clipping', () => {
      describe('given layerwise norm clipping runs without explicit layer metadata', () => {
        describe('when clipping is applied', () => {
          it('records at least one clipping group', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(125);
            const inputNode = network.nodes[0];
            const outputNode = network.nodes.find(
              (node) => node.type === 'output',
            )!;
            const [connection] = network.connect(inputNode, outputNode);
            connection.totalDeltaWeight = 5;

            // Act
            applyGradientClippingImpl(network, {
              mode: 'layerwiseNorm',
              maxNorm: 1,
            });
            const groupCount = getNetworkInternal(
              network,
              '_lastGradClipGroupCount',
            );

            // Assert
            expect(groupCount).toBeGreaterThanOrEqual(1);
          });
        });
      });
    });

    describe('mixed precision', () => {
      describe('given dynamic mixed precision can scale up after good steps', () => {
        describe('when trainImpl runs with an adaptive optimizer', () => {
          it('increases the loss scale above its starting value', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(126);
            const trainingDataset = [{ input: [0.2], output: [0.3] }];

            // Act
            trainImpl(network, trainingDataset, {
              iterations: 2,
              rate: 0.1,
              optimizer: 'adam',
              mixedPrecision: { lossScale: 2, dynamic: { increaseEvery: 1 } },
            });
            const lossScale = getNetworkInternal(
              network,
              '_mixedPrecision',
            ).lossScale;

            // Assert
            expect(lossScale).toBeGreaterThan(2);
          });
        });
      });
    });

    describe('stopping conditions', () => {
      describe('given the target error threshold is loose', () => {
        describe('when trainImpl reaches the target before max iterations', () => {
          it('returns early', () => {
            // Arrange
            const network = createSingleInputOutputNetwork(127);
            const trainingDataset = [{ input: [0.2], output: [0.2] }];

            // Act
            const trainingSummary = trainImpl(network, trainingDataset, {
              iterations: 50,
              rate: 0.1,
              error: 0.9,
            });

            // Assert
            expect(trainingSummary.iterations).toBeLessThan(50);
          });
        });
      });
    });
  });
});
