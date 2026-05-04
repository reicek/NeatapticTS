import Network from '../network';
import { RNG_WEYL_INCREMENT } from './network.deterministic.utils.types';

type InternalRand = () => number;

function invokeInternalRand(network: Network): number {
  const randomFunction = Reflect.get(network, '_rand') as InternalRand;
  return randomFunction.call(network);
}

function arraysMatchWithinTolerance(
  firstValues: number[],
  secondValues: number[],
  tolerance: number,
): boolean {
  return (
    firstValues.length === secondValues.length &&
    firstValues.every((value, valueIndex) => {
      return Math.abs(value - secondValues[valueIndex]) < tolerance;
    })
  );
}

function arraysMatchExactly(
  firstValues: Array<number | undefined>,
  secondValues: Array<number | undefined>,
): boolean {
  return (
    firstValues.length === secondValues.length &&
    firstValues.every((value, valueIndex) => value === secondValues[valueIndex])
  );
}

describe('network deterministic chapter', () => {
  describe('setSeed()', () => {
    describe('given two networks share the same seed', () => {
      describe('when the first deterministic sample is read', () => {
        it('produces the same sampled value', () => {
          // Arrange
          const firstNetwork = new Network(1, 1, {
            seed: 123,
            enforceAcyclic: true,
          });
          const secondNetwork = new Network(1, 1, {
            seed: 123,
            enforceAcyclic: true,
          });

          // Act
          const firstSample = invokeInternalRand(firstNetwork);
          const secondSample = invokeInternalRand(secondNetwork);

          // Assert
          expect(firstSample).toBe(secondSample);
        });
      });

      describe('when both networks activate the same initial input', () => {
        it('produces identical initial activations within tolerance', () => {
          // Arrange
          const firstNetwork = new Network(3, 2, { seed: 123 });
          const secondNetwork = new Network(3, 2, { seed: 123 });
          const inputVector = [0.1, -0.2, 0.3];

          // Act
          const firstOutput = firstNetwork.activate(inputVector);
          const secondOutput = secondNetwork.activate(inputVector);

          // Assert
          expect(
            arraysMatchWithinTolerance(firstOutput, secondOutput, 1e-12),
          ).toBe(true);
        });
      });

      describe('when the initial connection weights are compared', () => {
        it('creates the same weight vector', () => {
          // Arrange
          const firstNetwork = new Network(1, 1, { seed: 123 });
          const secondNetwork = new Network(1, 1, { seed: 123 });

          // Act
          const firstWeights = firstNetwork.connections.map(
            (connection) => connection.weight,
          );
          const secondWeights = secondNetwork.connections.map(
            (connection) => connection.weight,
          );

          // Assert
          expect(arraysMatchExactly(firstWeights, secondWeights)).toBe(true);
        });
      });

      describe('when the initial node biases are compared', () => {
        it('creates the same bias vector', () => {
          // Arrange
          const firstNetwork = new Network(1, 1, { seed: 123 });
          const secondNetwork = new Network(1, 1, { seed: 123 });

          // Act
          const firstBiases = firstNetwork.nodes.map((node) => node.bias);
          const secondBiases = secondNetwork.nodes.map((node) => node.bias);

          // Assert
          expect(arraysMatchExactly(firstBiases, secondBiases)).toBe(true);
        });
      });
    });

    describe('given two networks use different seeds', () => {
      describe('when their initial connection weights are compared', () => {
        it('produces different weight vectors', () => {
          // Arrange
          const firstNetwork = new Network(1, 1, { seed: 111 });
          const secondNetwork = new Network(1, 1, { seed: 222 });

          // Act
          const firstWeights = firstNetwork.connections.map(
            (connection) => connection.weight,
          );
          const secondWeights = secondNetwork.connections.map(
            (connection) => connection.weight,
          );

          // Assert
          expect(arraysMatchExactly(firstWeights, secondWeights)).toBe(false);
        });
      });
    });

    describe('given deterministic random state is reset to undefined', () => {
      describe('when the internal random function advances once', () => {
        it('falls back to zero before applying the Weyl increment', () => {
          // Arrange
          const network = new Network(1, 1, { seed: 123 });
          Reflect.set(network, '_rngState', undefined);

          // Act
          invokeInternalRand(network);
          const stateAfterAdvance = network.getRNGState();

          // Assert
          expect(stateAfterAdvance).toBe(RNG_WEYL_INCREMENT);
        });
      });
    });
  });

  describe('snapshotRNG()', () => {
    describe('given a seeded network', () => {
      describe('when a snapshot is captured', () => {
        it('includes a numeric state word', () => {
          // Arrange
          const network = new Network(1, 1, { seed: 9, enforceAcyclic: true });

          // Act
          const snapshot = network.snapshotRNG();

          // Assert
          expect(typeof snapshot.state).toBe('number');
        });
      });
    });
  });

  describe('setRNGState()', () => {
    describe('given a snapshot state word was captured earlier', () => {
      describe('when that state word is restored after one random advance', () => {
        it('round-trips the numeric RNG state', () => {
          // Arrange
          const network = new Network(1, 1, { seed: 9, enforceAcyclic: true });
          const snapshot = network.snapshotRNG();
          const snapshotState = snapshot.state;

          if (typeof snapshotState !== 'number') {
            throw new Error(
              'Snapshot state should be numeric before restoration',
            );
          }

          invokeInternalRand(network);

          // Act
          network.setRNGState(snapshotState);
          const restoredState = network.getRNGState();

          // Assert
          expect(restoredState).toBe(snapshotState);
        });
      });
    });

    describe('given a seeded network with an existing RNG state', () => {
      describe('when a non-number state value is applied at runtime', () => {
        it('preserves the previous numeric RNG state', () => {
          // Arrange
          const network = new Network(1, 1, { seed: 13, enforceAcyclic: true });
          const initialState = network.getRNGState();

          if (typeof initialState !== 'number') {
            throw new Error(
              'Initial RNG state should be numeric before update',
            );
          }

          // Act
          network.setRNGState('invalid-state' as unknown as number);
          const preservedState = network.getRNGState();

          // Assert
          expect(preservedState).toBe(initialState);
        });
      });
    });
  });

  describe('getRandomFn()', () => {
    describe('given a custom RNG implementation is installed', () => {
      describe('when the active random function is requested', () => {
        it('returns the installed RNG reference', () => {
          // Arrange
          const network = new Network(1, 1, { seed: 42, enforceAcyclic: true });
          const expectedRandomFunction = () => 0.25;
          network.restoreRNG(expectedRandomFunction);

          // Act
          const randomFunction = network.getRandomFn();

          // Assert
          expect(randomFunction).toBe(expectedRandomFunction);
        });
      });
    });
  });

  describe('restoreRNG()', () => {
    describe('given a custom RNG implementation is installed', () => {
      describe('when the internal random function is invoked', () => {
        it('uses the injected RNG function', () => {
          // Arrange
          const network = new Network(1, 1, { seed: 42, enforceAcyclic: true });
          const expectedValue = 0.5;
          network.restoreRNG(() => expectedValue);

          // Act
          const value = invokeInternalRand(network);

          // Assert
          expect(value).toBe(expectedValue);
        });
      });
    });

    describe('given a custom RNG implementation is installed', () => {
      describe('when the numeric RNG state is queried afterward', () => {
        it('clears the stored numeric state', () => {
          // Arrange
          const network = new Network(1, 1, { seed: 42, enforceAcyclic: true });
          network.restoreRNG(() => 0.5);

          // Act
          const state = network.getRNGState();

          // Assert
          expect(state).toBeUndefined();
        });
      });
    });
  });
});
