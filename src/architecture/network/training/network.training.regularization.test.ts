import Network from '../network';
import { Architect, methods } from '../../../neataptic';

type TrainingSample = { input: number[]; output: number[] };

const XOR_DATASET: TrainingSample[] = [
  { input: [0, 0], output: [0] },
  { input: [0, 1], output: [1] },
  { input: [1, 0], output: [1] },
  { input: [1, 1], output: [0] },
];

function applyDeterministicWeights(network: Network, scale = 1): void {
  network.connections.forEach((connection, connectionIndex) => {
    connection.weight = (((((connectionIndex + 3) * 7) % 17) - 8) / 10) * scale;
  });
}

function averageAbsoluteWeight(network: Network): number {
  return (
    network.connections.reduce((sum, connection) => {
      return sum + Math.abs(connection.weight);
    }, 0) / Math.max(network.connections.length, 1)
  );
}

function countNearZeroWeights(network: Network, zeroThreshold: number): number {
  return network.connections.filter((connection) => {
    return Math.abs(connection.weight) < zeroThreshold;
  }).length;
}

function createNoisyXorData(
  sampleCount: number,
  noiseLevel: number,
  randomFunction: () => number,
): TrainingSample[] {
  return Array.from({ length: sampleCount }, () => {
    const firstInput = randomFunction() > 0.5 ? 1 : 0;
    const secondInput = randomFunction() > 0.5 ? 1 : 0;
    const noiseSign = randomFunction() > 0.5 ? 1 : -1;
    const noisyTarget =
      (firstInput !== secondInput ? 1 : 0) +
      randomFunction() * noiseLevel * noiseSign;

    return {
      input: [firstInput, secondInput],
      output: [noisyTarget],
    };
  });
}

function createMulberry32(seed: number): () => number {
  let state = seed;

  return () => {
    let mixedState = (state += 0x6d2b79f5);
    mixedState = Math.imul(mixedState ^ (mixedState >>> 15), mixedState | 1);
    mixedState ^=
      mixedState + Math.imul(mixedState ^ (mixedState >>> 7), mixedState | 61);
    return ((mixedState ^ (mixedState >>> 14)) >>> 0) / 4294967296;
  };
}

describe('network training chapter', () => {
  describe('regularization', () => {
    describe('given identical starting weights with and without scalar L2 regularization', () => {
      describe('when both runs finish', () => {
        it('returns finite training errors for both runs', () => {
          // Arrange
          const baseNetwork = Architect.perceptron(2, 4, 1);
          applyDeterministicWeights(baseNetwork);
          const unregularizedNetwork = baseNetwork.clone();
          const regularizedNetwork = baseNetwork.clone();

          const unregularizedSummary = unregularizedNetwork.train(XOR_DATASET, {
            iterations: 100,
            error: 0.01,
            regularization: 0,
          });

          // Act
          const regularizedSummary = regularizedNetwork.train(XOR_DATASET, {
            iterations: 100,
            error: 0.01,
            regularization: 10,
          });

          // Assert
          expect(
            Number.isFinite(unregularizedSummary.error) &&
              Number.isFinite(regularizedSummary.error),
          ).toBe(true);
        });
      });
    });

    describe('given large identical starting weights with and without scalar L2 regularization', () => {
      describe('when both runs finish', () => {
        it('shrinks average absolute weight magnitude more under regularization', () => {
          // Arrange
          const baseNetwork = Architect.perceptron(2, 4, 1);
          baseNetwork.connections.forEach((connection) => {
            connection.weight = 3;
          });
          const unregularizedNetwork = baseNetwork.clone();
          const regularizedNetwork = baseNetwork.clone();

          unregularizedNetwork.train(XOR_DATASET, {
            iterations: 50,
            error: 0.01,
            regularization: 0,
          });

          regularizedNetwork.train(XOR_DATASET, {
            iterations: 50,
            error: 0.01,
            regularization: 10,
          });

          // Act
          const unregularizedAverage =
            averageAbsoluteWeight(unregularizedNetwork);
          const regularizedAverage = averageAbsoluteWeight(regularizedNetwork);

          // Assert
          expect(regularizedAverage).toBeLessThanOrEqual(unregularizedAverage);
        });
      });
    });

    describe('given explicit zero regularization and omitted regularization start from the same weights', () => {
      describe('when both runs finish', () => {
        it('produces nearly identical training errors', () => {
          // Arrange
          const baseNetwork = Architect.perceptron(2, 4, 1);
          applyDeterministicWeights(baseNetwork);
          const explicitZeroNetwork = baseNetwork.clone();
          const implicitZeroNetwork = baseNetwork.clone();

          const explicitZeroSummary = explicitZeroNetwork.train(XOR_DATASET, {
            iterations: 10,
            error: 0.01,
            regularization: 0,
          });

          // Act
          const implicitZeroSummary = implicitZeroNetwork.train(XOR_DATASET, {
            iterations: 10,
            error: 0.01,
          });

          // Assert
          expect(explicitZeroSummary.error).toBeCloseTo(
            implicitZeroSummary.error,
            6,
          );
        });
      });

      describe('when both final weight vectors are inspected', () => {
        it('updates the weights the same way', () => {
          // Arrange
          const baseNetwork = Architect.perceptron(2, 4, 1);
          applyDeterministicWeights(baseNetwork);
          const explicitZeroNetwork = baseNetwork.clone();
          const implicitZeroNetwork = baseNetwork.clone();

          explicitZeroNetwork.train(XOR_DATASET, {
            iterations: 10,
            error: 0.01,
            regularization: 0,
          });

          implicitZeroNetwork.train(XOR_DATASET, {
            iterations: 10,
            error: 0.01,
          });

          // Act
          const weightsMatch = explicitZeroNetwork.connections.every(
            (connection, connectionIndex) => {
              return (
                Math.abs(
                  connection.weight -
                    implicitZeroNetwork.connections[connectionIndex].weight,
                ) < 1e-9
              );
            },
          );

          // Assert
          expect(weightsMatch).toBe(true);
        });
      });
    });

    describe('given identical small starting weights under L1 and L2 regularization', () => {
      describe('when both runs finish', () => {
        it('leaves at least as many near-zero weights under L1 as under L2', () => {
          // Arrange
          const baseNetwork = Architect.perceptron(2, 10, 1);
          applyDeterministicWeights(baseNetwork, 0.05);
          const l1Network = baseNetwork.clone();
          const l2Network = baseNetwork.clone();

          l1Network.train(XOR_DATASET, {
            iterations: 50,
            error: 0.01,
            regularization: { type: 'L1', lambda: 0.01 },
          });

          l2Network.train(XOR_DATASET, {
            iterations: 50,
            error: 0.01,
            regularization: { type: 'L2', lambda: 0.01 },
          });

          // Act
          const l1ZeroWeights = countNearZeroWeights(l1Network, 1e-4);
          const l2ZeroWeights = countNearZeroWeights(l2Network, 1e-4);

          // Assert
          expect(l1ZeroWeights).toBeGreaterThanOrEqual(l2ZeroWeights);
        });
      });
    });

    describe('given noisy XOR training data and clean evaluation data', () => {
      describe('when L2 regularization is compared with no regularization', () => {
        it('does not perform materially worse on the clean evaluation set', () => {
          // Arrange
          const trainingRandom = createMulberry32(123456789);
          const evaluationRandom = createMulberry32(987654321);
          const trainingData = createNoisyXorData(60, 0.2, trainingRandom);
          const evaluationData = createNoisyXorData(80, 0, evaluationRandom);
          const baseNetwork = new Network(2, 1, { seed: 61 });

          baseNetwork.mutate(methods.mutation.ADD_NODE);
          baseNetwork.mutate(methods.mutation.ADD_NODE);
          applyDeterministicWeights(baseNetwork, 0.25);

          const unregularizedNetwork = baseNetwork.clone();
          const regularizedNetwork = baseNetwork.clone();

          unregularizedNetwork.train(trainingData, {
            iterations: 5000,
            error: 0.01,
            rate: 0.05,
          });

          regularizedNetwork.train(trainingData, {
            iterations: 5000,
            error: 0.01,
            rate: 0.05,
            regularization: 0.0005,
            regularizationType: 'L2',
          });

          const unregularizedError =
            unregularizedNetwork.test(evaluationData).error;

          // Act
          const regularizedError =
            regularizedNetwork.test(evaluationData).error;

          // Assert
          expect(regularizedError).toBeLessThanOrEqual(
            unregularizedError + 0.02,
          );
        });
      });
    });
  });
});
