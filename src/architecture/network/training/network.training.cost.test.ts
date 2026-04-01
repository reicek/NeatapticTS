import type Network from '../network';
import { Architect, methods } from '../../../neataptic';

type TrainingSample = { input: number[]; output: number[] };

const XOR_DATASET: TrainingSample[] = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

function applyDeterministicWeights(network: Network): void {
  network.connections.forEach((connection, connectionIndex) => {
    connection.weight = ((((connectionIndex + 5) * 11) % 19) - 9) / 10;
  });
}

describe('network training chapter', () => {
  describe('cost functions', () => {
    describe('given train() uses MAE cost', () => {
      describe('when the returned error is inspected', () => {
        it('stays below one on XOR training', () => {
          // Arrange
          const network = Architect.perceptron(2, 4, 1);

          // Act
          const trainingSummary = network.train(XOR_DATASET, {
            iterations: 100,
            error: 0.1,
            cost: methods.Cost.mae,
          });

          // Assert
          expect(trainingSummary.error).toBeLessThan(1);
        });
      });
    });

    describe('given test() uses MAE cost after a short MAE training run', () => {
      describe('when the returned error is inspected', () => {
        it('stays below one on XOR evaluation', () => {
          // Arrange
          const network = Architect.perceptron(2, 4, 1);
          network.train(XOR_DATASET, {
            iterations: 10,
            cost: methods.Cost.mae,
          });

          // Act
          const testSummary = network.test(XOR_DATASET, methods.Cost.mae);

          // Assert
          expect(testSummary.error).toBeLessThan(1);
        });
      });
    });

    describe('given MAE and MSE training runs start from the same weights', () => {
      describe('when both final evaluation errors are compared', () => {
        it('produces measurably different results', () => {
          // Arrange
          const baseNetwork = Architect.perceptron(2, 4, 1);
          applyDeterministicWeights(baseNetwork);
          const maeNetwork = baseNetwork.clone();
          const mseNetwork = baseNetwork.clone();

          maeNetwork.train(XOR_DATASET, {
            iterations: 20,
            cost: methods.Cost.mae,
          });
          mseNetwork.train(XOR_DATASET, {
            iterations: 20,
            cost: methods.Cost.mse,
          });
          const maeError = maeNetwork.test(XOR_DATASET, methods.Cost.mae).error;

          // Act
          const mseError = mseNetwork.test(XOR_DATASET, methods.Cost.mse).error;

          // Assert
          expect(Math.abs(maeError - mseError)).toBeGreaterThan(1e-3);
        });
      });
    });

    describe('given an invalid cost selector is passed to train()', () => {
      describe('when training starts', () => {
        it('throws the invalid-cost error', () => {
          // Arrange
          const network = Architect.perceptron(2, 4, 1);

          // Act
          const trainWithInvalidCost = () => {
            network.train(XOR_DATASET, {
              iterations: 100,
              error: 0.1,
              cost: 'notARealCostFn',
            });
          };

          // Assert
          expect(trainWithInvalidCost).toThrow(
            'Invalid cost function provided to Network.train.',
          );
        });
      });
    });

    describe('given a custom calculate-and-derivative cost object is passed to train()', () => {
      describe('when the returned error is inspected', () => {
        it('trains successfully and returns a finite error', () => {
          // Arrange
          const network = Architect.perceptron(2, 4, 1);
          const customCost = {
            calculate: (target: number[], output: number[]) => {
              return (
                target.reduce((sum, targetValue, index) => {
                  return sum + Math.abs(targetValue - output[index]);
                }, 0) / target.length
              );
            },
            derivative: (target: number, output: number) => {
              return output > target ? 1 : -1;
            },
          };

          // Act
          const trainingSummary = network.train(XOR_DATASET, {
            iterations: 5,
            cost: customCost,
          });

          // Assert
          expect(Number.isFinite(trainingSummary.error)).toBe(true);
        });
      });
    });
  });
});
