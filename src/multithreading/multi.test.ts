import Multi from './multi';
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
import { TestWorker as BrowserTestWorker } from './workers/browser/testworker';
import { TestWorker as NodeTestWorker } from './workers/node/testworker';
import { Workers } from './workers/workers';

const ACTIVATION_INPUT = 0.75;
const SERIALIZED_NETWORK = [1, 1, 0, 0, 2, 0, -1, 0, 0.5, -1, -2];
const PERFECT_SCORE_DATA_SET = [
  { input: [2], output: [1] },
  { input: [4], output: [2] },
];

describe('multithreading facade chapter', () => {
  describe('static shelves', () => {
    describe('given the root facade exposes the worker and activation contracts', () => {
      it('keeps those public references aligned with the lower-level shelves', () => {
        // Assert
        expect({
          workers: Multi.workers,
          activations: Multi.activations,
        }).toEqual({
          workers: Workers,
          activations: ACTIVATION_FUNCTIONS,
        });
      });
    });
  });

  describe('dataset facade helpers', () => {
    describe('given the caller round-trips a serialized dataset through the root facade', () => {
      it('matches the lower-level utility behavior exactly', () => {
        // Arrange
        const dataSet = [{ input: [1, 2], output: [3] }];

        // Act
        const facadeRoundTrip = Multi.deserializeDataSet(
          Multi.serializeDataSet(dataSet),
        );

        // Assert
        expect(facadeRoundTrip).toEqual(
          deserializeDataSet(serializeDataSet(dataSet)),
        );
      });
    });
  });

  describe('activation facade helpers', () => {
    describe('given the root facade exposes the compiled activation shelf', () => {
      it('keeps each wrapper aligned with the lower-level activation helper', () => {
        // Act
        const facadeOutputs = {
          absolute: Multi.absolute(ACTIVATION_INPUT),
          bentIdentity: Multi.bentIdentity(ACTIVATION_INPUT),
          bipolar: Multi.bipolar(ACTIVATION_INPUT),
          bipolarSigmoid: Multi.bipolarSigmoid(ACTIVATION_INPUT),
          gaussian: Multi.gaussian(ACTIVATION_INPUT),
          hardTanh: Multi.hardTanh(ACTIVATION_INPUT),
          identity: Multi.identity(ACTIVATION_INPUT),
          inverse: Multi.inverse(ACTIVATION_INPUT),
          logistic: Multi.logistic(ACTIVATION_INPUT),
          relu: Multi.relu(ACTIVATION_INPUT),
          selu: Multi.selu(ACTIVATION_INPUT),
          sinusoid: Multi.sinusoid(ACTIVATION_INPUT),
          softplus: Multi.softplus(ACTIVATION_INPUT),
          softsign: Multi.softsign(ACTIVATION_INPUT),
          step: Multi.step(ACTIVATION_INPUT),
          tanh: Multi.tanh(ACTIVATION_INPUT),
        };

        // Assert
        expect(facadeOutputs).toEqual({
          absolute: absoluteActivation(ACTIVATION_INPUT),
          bentIdentity: bentIdentityActivation(ACTIVATION_INPUT),
          bipolar: bipolarActivation(ACTIVATION_INPUT),
          bipolarSigmoid: bipolarSigmoidActivation(ACTIVATION_INPUT),
          gaussian: gaussianActivation(ACTIVATION_INPUT),
          hardTanh: hardTanhActivation(ACTIVATION_INPUT),
          identity: identityActivation(ACTIVATION_INPUT),
          inverse: inverseActivation(ACTIVATION_INPUT),
          logistic: logisticActivation(ACTIVATION_INPUT),
          relu: reluActivation(ACTIVATION_INPUT),
          selu: seluActivation(ACTIVATION_INPUT),
          sinusoid: sinusoidActivation(ACTIVATION_INPUT),
          softplus: softplusActivation(ACTIVATION_INPUT),
          softsign: softsignActivation(ACTIVATION_INPUT),
          step: stepActivation(ACTIVATION_INPUT),
          tanh: tanhActivation(ACTIVATION_INPUT),
        });
      });
    });
  });

  describe('flat execution facade helpers', () => {
    describe('given the root facade proxies the worker-compatible execution path', () => {
      it('matches the lower-level activation and dataset testing helpers', () => {
        // Arrange
        const costFunction = (
          expectedOutputs: number[],
          actualOutputs: number[],
        ) => Math.abs(expectedOutputs[0] - actualOutputs[0]);

        // Act
        const facadeResult = {
          averageCost: Multi.testSerializedSet(
            PERFECT_SCORE_DATA_SET,
            costFunction,
            [0],
            [0],
            SERIALIZED_NETWORK,
            ACTIVATION_FUNCTIONS,
          ),
          outputValues: Multi.activateSerializedNetwork(
            [2],
            [0],
            [0],
            SERIALIZED_NETWORK,
            ACTIVATION_FUNCTIONS,
          ),
        };

        // Assert
        expect(facadeResult).toEqual({
          averageCost: testSerializedSet(
            PERFECT_SCORE_DATA_SET,
            costFunction,
            [0],
            [0],
            SERIALIZED_NETWORK,
            ACTIVATION_FUNCTIONS,
          ),
          outputValues: activateSerializedNetwork(
            [2],
            [0],
            [0],
            SERIALIZED_NETWORK,
            ACTIVATION_FUNCTIONS,
          ),
        });
      });
    });
  });

  describe('getBrowserTestWorker', () => {
    describe('given the browser wrapper module is available', () => {
      it('resolves to the browser worker class', async () => {
        // Assert
        await expect(Multi.getBrowserTestWorker()).resolves.toBe(
          BrowserTestWorker,
        );
      });
    });
  });

  describe('getNodeTestWorker', () => {
    describe('given the node wrapper module is available', () => {
      it('resolves to the node worker class', async () => {
        // Assert
        await expect(Multi.getNodeTestWorker()).resolves.toBe(NodeTestWorker);
      });
    });
  });
});
