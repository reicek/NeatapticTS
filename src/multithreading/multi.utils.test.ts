import {
  ACTIVATION_FUNCTIONS,
  absoluteActivation,
  activateSerializedNetwork,
  bentIdentityActivation,
  bipolarActivation,
  bipolarSigmoidActivation,
  deserializeDataSet,
  gaussianActivation,
  hardTanhActivation,
  identityActivation,
  inverseActivation,
  logisticActivation,
  reluActivation,
  seluActivation,
  serializeDataSet,
  sinusoidActivation,
  softplusActivation,
  softsignActivation,
  stepActivation,
  tanhActivation,
  testSerializedSet,
} from './multi.utils';
import type { ActivationFn, SerializedSample } from './types';

const SAMPLE_GRID = [-2, -1, 0, 1, 2];
const SERIALIZED_NETWORK = [1, 1, 0, 0, 2, 0, -1, 0, 0.5, -1, -2];
const ROUND_TRIP_DATA_SET: SerializedSample[] = [
  { input: [1, 2], output: [3] },
  { input: [4, 5], output: [9] },
];
const PERFECT_SCORE_DATA_SET: SerializedSample[] = [
  { input: [2], output: [1] },
  { input: [4], output: [2] },
];
const ACTIVATION_CASES: Array<[string, ActivationFn]> = [
  ['logisticActivation', logisticActivation],
  ['tanhActivation', tanhActivation],
  ['identityActivation', identityActivation],
  ['stepActivation', stepActivation],
  ['reluActivation', reluActivation],
  ['softsignActivation', softsignActivation],
  ['sinusoidActivation', sinusoidActivation],
  ['gaussianActivation', gaussianActivation],
  ['bentIdentityActivation', bentIdentityActivation],
  ['bipolarActivation', bipolarActivation],
  ['bipolarSigmoidActivation', bipolarSigmoidActivation],
  ['hardTanhActivation', hardTanhActivation],
  ['absoluteActivation', absoluteActivation],
  ['inverseActivation', inverseActivation],
  ['seluActivation', seluActivation],
  ['softplusActivation', softplusActivation],
];

describe('multithreading utility chapter', () => {
  describe('activation helpers', () => {
    describe('given the compiled activation shelf is enumerated directly', () => {
      it('keeps the public activation registry aligned with the exported helpers', () => {
        // Assert
        expect(ACTIVATION_FUNCTIONS).toEqual(
          ACTIVATION_CASES.map(([, activationFunction]) => activationFunction),
        );
      });
    });

    describe.each(ACTIVATION_CASES)(
      'given %s is evaluated across the sample grid',
      (activationName, activationFunction) => {
        it('returns only finite values', () => {
          // Act
          const allOutputsAreFinite = SAMPLE_GRID.every((sampleValue) =>
            Number.isFinite(activationFunction(sampleValue)),
          );

          // Assert
          expect({ activationName, allOutputsAreFinite }).toEqual({
            activationName,
            allOutputsAreFinite: true,
          });
        });
      },
    );
  });

  describe('serializeDataSet', () => {
    describe('given the caller passes two samples', () => {
      it('flattens the input and output counts plus sample values in order', () => {
        // Act
        const serializedDataSet = serializeDataSet(ROUND_TRIP_DATA_SET);

        // Assert
        expect(serializedDataSet).toEqual([2, 1, 1, 2, 3, 4, 5, 9]);
      });
    });

    describe('given the caller passes an empty dataset', () => {
      it('throws because no header can be derived', () => {
        // Assert
        expect(() => serializeDataSet([])).toThrow();
      });
    });
  });

  describe('deserializeDataSet', () => {
    describe('given the serialized payload is well formed', () => {
      it('reconstructs the original input and output samples', () => {
        // Arrange
        const serializedDataSet = serializeDataSet(ROUND_TRIP_DATA_SET);

        // Act
        const deserializedDataSet = deserializeDataSet(serializedDataSet);

        // Assert
        expect(deserializedDataSet).toEqual(ROUND_TRIP_DATA_SET);
      });
    });

    describe('given the serialized payload is empty', () => {
      it('returns an empty dataset', () => {
        // Act
        const deserializedDataSet = deserializeDataSet([]);

        // Assert
        expect(deserializedDataSet).toEqual([]);
      });
    });

    describe('given the serialized payload is truncated before any sample data', () => {
      it('returns an empty dataset instead of inventing malformed samples', () => {
        // Act
        const deserializedDataSet = deserializeDataSet([2]);

        // Assert
        expect(deserializedDataSet).toEqual([]);
      });
    });
  });

  describe('activateSerializedNetwork', () => {
    describe('given a single-node identity network with one incoming connection', () => {
      it('returns the activated output value', () => {
        // Arrange
        const inputValues = [2];
        const activationValues = [0];
        const stateValues = [0];

        // Act
        const outputValues = activateSerializedNetwork(
          inputValues,
          activationValues,
          stateValues,
          SERIALIZED_NETWORK,
          ACTIVATION_FUNCTIONS,
        );

        // Assert
        expect(outputValues).toEqual([1]);
      });
    });

    describe('given a network with a gated self-connection and gated incoming connection', () => {
      describe('when the output node has gaterIndex !== NO_SELF_GATER', () => {
        it('returns a finite output value using the gater activation', () => {
          // Arrange – header [1 input, 1 output], then output node (index 1):
          // [nodeIndex=1, nodeBias=0, activationIndex=0, selfWeight=0.5, selfGaterIndex=0]
          // connection: [sourceIndex=0, weight=0.5, gaterIndex=0], terminator=-2
          // selfGaterIndex=0 and gaterIndex=0 hit the FALSE arms at lines 100 and 113.
          const gatedNetwork = [1, 1, 1, 0, 0, 0.5, 0, 0, 0.5, 0, -2];
          const inputValues = [0.7];
          const activationValues = [0, 0];
          const stateValues = [0, 0];

          // Act
          const outputValues = activateSerializedNetwork(
            inputValues,
            activationValues,
            stateValues,
            gatedNetwork,
            ACTIVATION_FUNCTIONS,
          );

          // Assert
          expect(Number.isFinite(outputValues[0])).toBe(true);
        });
      });
    });
  });

  describe('testSerializedSet', () => {
    describe('given the serialized network matches the expected outputs exactly', () => {
      it('returns a zero average cost', () => {
        // Arrange
        const costFunction = (
          expectedOutputs: number[],
          actualOutputs: number[],
        ) => Math.abs(expectedOutputs[0] - actualOutputs[0]);

        // Act
        const averageCost = testSerializedSet(
          PERFECT_SCORE_DATA_SET,
          costFunction,
          [0],
          [0],
          SERIALIZED_NETWORK,
          ACTIVATION_FUNCTIONS,
        );

        // Assert
        expect(averageCost).toBe(0);
      });
    });

    describe('given the sample set is empty', () => {
      it('returns NaN because there is no average to compute', () => {
        // Act
        const averageCost = testSerializedSet(
          [],
          () => 0,
          [0],
          [0],
          SERIALIZED_NETWORK,
          ACTIVATION_FUNCTIONS,
        );

        // Assert
        expect(Number.isNaN(averageCost)).toBe(true);
      });
    });

    describe('given the cost function returns NaN', () => {
      it('propagates the invalid average as NaN', () => {
        // Act
        const averageCost = testSerializedSet(
          PERFECT_SCORE_DATA_SET,
          () => Number.NaN,
          [0],
          [0],
          SERIALIZED_NETWORK,
          ACTIVATION_FUNCTIONS,
        );

        // Assert
        expect(Number.isNaN(averageCost)).toBe(true);
      });
    });
  });
});
