import Network from '../network';
import {
  NetworkStatsTestSampleInputSizeMismatchError,
  NetworkStatsTestSampleOutputSizeMismatchError,
  NetworkStatsTestSetValidationError,
} from './network.stats.errors';

interface RegularizationStatsSnapshot {
  l1Penalty?: number;
  dropped?: number;
  custom?: { depth: number };
  [key: string]: unknown;
}

function createStatsNetwork(seed: number): Network {
  return new Network(2, 1, { seed });
}

function setRegularizationStatsSnapshot(
  network: Network,
  statsSnapshot: RegularizationStatsSnapshot | null,
): void {
  Reflect.set(network, '_lastStats', statsSnapshot);
}

function getRegularizationStatsSnapshot(
  network: Network,
): RegularizationStatsSnapshot | null {
  return network.getRegularizationStats() as RegularizationStatsSnapshot | null;
}

describe('network stats chapter', () => {
  describe('Network.test()', () => {
    describe('given the evaluation set is not an array', () => {
      describe('when test() is called', () => {
        it('throws the test-set validation error', () => {
          // Arrange
          const network = createStatsNetwork(299);

          // Act
          const testWithNonArraySet = () =>
            network.test(undefined as unknown as Array<{ input: number[]; output: number[] }>);

          // Assert
          expect(testWithNonArraySet).toThrow(NetworkStatsTestSetValidationError);
        });
      });
    });

    describe('given the evaluation set input width is too small', () => {
      describe('when test() is called', () => {
        it('throws the input-size mismatch error', () => {
          // Arrange
          const network = createStatsNetwork(300);

          // Act
          const testWithTooFewInputs = () =>
            network.test([{ input: [1], output: [1] }]);

          // Assert
          expect(testWithTooFewInputs).toThrow(
            NetworkStatsTestSampleInputSizeMismatchError,
          );
        });
      });
    });

    describe('given one sample input vector is undefined', () => {
      describe('when test() is called', () => {
        it('throws an input-size mismatch that reports undefined width', () => {
          // Arrange
          const network = createStatsNetwork(3001);

          // Act
          const testWithUndefinedInput = () =>
            network.test([
              { input: undefined as unknown as number[], output: [1] },
            ]);

          // Assert
          expect(testWithUndefinedInput).toThrow(
            'Test sample input size mismatch: expected 2, got undefined',
          );
        });
      });
    });

    describe('given the evaluation set output width is too large', () => {
      describe('when test() is called', () => {
        it('throws the output-size mismatch error', () => {
          // Arrange
          const network = createStatsNetwork(301);

          // Act
          const testWithTooManyOutputs = () =>
            network.test([{ input: [1, 2], output: [1, 2] }]);

          // Assert
          expect(testWithTooManyOutputs).toThrow(
            NetworkStatsTestSampleOutputSizeMismatchError,
          );
        });
      });
    });

    describe('given one sample output vector is undefined', () => {
      describe('when test() is called', () => {
        it('throws an output-size mismatch that reports undefined width', () => {
          // Arrange
          const network = createStatsNetwork(3002);

          // Act
          const testWithUndefinedOutput = () =>
            network.test([
              { input: [1, 2], output: undefined as unknown as number[] },
            ]);

          // Assert
          expect(testWithUndefinedOutput).toThrow(
            'Test sample output size mismatch: expected 1, got undefined',
          );
        });
      });
    });

    describe('given the evaluation set is empty', () => {
      describe('when test() is called', () => {
        it('throws the test-set validation error', () => {
          // Arrange
          const network = createStatsNetwork(302);

          // Act
          const testWithEmptySet = () => network.test([]);

          // Assert
          expect(testWithEmptySet).toThrow(NetworkStatsTestSetValidationError);
        });
      });
    });

    describe('given the evaluation set matches the network dimensions', () => {
      describe('when test() is called', () => {
        it('does not throw', () => {
          // Arrange
          const network = createStatsNetwork(303);

          // Act
          const testWithValidSet = () =>
            network.test([{ input: [1, 2], output: [1] }]);

          // Assert
          expect(testWithValidSet).not.toThrow();
        });
      });

      describe('when a custom cost function is provided', () => {
        it('uses the provided cost function during evaluation', () => {
          // Arrange
          const network = createStatsNetwork(3031);
          const costFunction = jest.fn(() => 0);

          // Act
          network.test([{ input: [1, 2], output: [1] }], costFunction);

          // Assert
          expect(costFunction).toHaveBeenCalledTimes(1);
        });
      });

      describe('when dropout is enabled before the run', () => {
        it('restores the original dropout value after evaluation', () => {
          // Arrange
          const network = createStatsNetwork(3032);
          network.dropout = 0.35;

          // Act
          network.test([{ input: [1, 2], output: [1] }]);

          // Assert
          expect(network.dropout).toBe(0.35);
        });
      });

      describe('when hidden-node masks are inactive before the run', () => {
        it('reactivates hidden-node masks during evaluation setup', () => {
          // Arrange
          const network = Network.createMLP(2, [2], 1);
          const hiddenNodes = network.nodes.filter((node) => node.type === 'hidden');
          hiddenNodes.forEach((node) => {
            node.mask = 0;
          });

          // Act
          network.test([{ input: [1, 2], output: [1] }]);

          // Assert
          expect(hiddenNodes.every((node) => node.mask === 1)).toBe(true);
        });
      });
    });
  });

  describe('Network.getRegularizationStats()', () => {
    describe('given regularization stats were never recorded', () => {
      describe('when getRegularizationStats() is called', () => {
        it('returns null', () => {
          // Arrange
          const network = new Network(1, 1, {
            seed: 304,
            enforceAcyclic: true,
          });

          // Act
          const regularizationStats = network.getRegularizationStats();

          // Assert
          expect(regularizationStats).toBeNull();
        });
      });
    });

    describe('given regularization stats are present', () => {
      describe('when the returned snapshot reference is compared to internal state', () => {
        it('returns a cloned object', () => {
          // Arrange
          const network = new Network(1, 1, {
            seed: 305,
            enforceAcyclic: true,
          });
          const internalStatsSnapshot: RegularizationStatsSnapshot = {
            l1Penalty: 0.1,
            dropped: 0.25,
            custom: { depth: 3 },
          };
          setRegularizationStatsSnapshot(network, internalStatsSnapshot);

          // Act
          const regularizationStats = network.getRegularizationStats();

          // Assert
          expect(regularizationStats === internalStatsSnapshot).toBe(false);
        });
      });

      describe('when the returned clone is deep-mutated externally', () => {
        it('does not change the stored internal stats', () => {
          // Arrange
          const network = new Network(1, 1, {
            seed: 306,
            enforceAcyclic: true,
          });
          setRegularizationStatsSnapshot(network, { custom: { depth: 3 } });
          const regularizationStats = getRegularizationStatsSnapshot(network);
          if (!regularizationStats || !regularizationStats.custom) {
            throw new Error('Regularization stats should include custom depth');
          }

          // Act
          regularizationStats.custom.depth = 99;
          const rereadStatsSnapshot = getRegularizationStatsSnapshot(network);
          if (!rereadStatsSnapshot || !rereadStatsSnapshot.custom) {
            throw new Error('Regularization stats should include custom depth');
          }

          // Assert
          expect(rereadStatsSnapshot.custom.depth).toBe(3);
        });
      });
    });
  });
});
