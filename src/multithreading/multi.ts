import { Workers } from './workers/workers';
import type {
  ActivationFn,
  SerializedSample,
  TestWorkerConstructor,
} from './types';
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

/**
 * Multi-threading utilities for neural network operations.
 *
 * This class provides methods for serializing datasets, activating serialized networks,
 * and testing serialized datasets. These utilities align with the Instinct algorithm's
 * emphasis on efficient evaluation and mutation of neural networks in parallel environments.
 *
 * @see Instinct Algorithm - Section 4 Constraints
 * @see {@link https://medium.com/data-science/neuro-evolution-on-steroids-82bd14ddc2f6}
 */
export default class Multi {
  /** Workers for multi-threading */
  static workers = Workers;

  /**
   * A list of compiled activation functions in a specific order.
   */
  static activations: ActivationFn[] = ACTIVATION_FUNCTIONS;

  /**
   * Serializes a dataset into a flat array.
   * @param {Array<{ input: number[]; output: number[] }>} dataSet - The dataset to serialize.
   * @returns {number[]} The serialized dataset.
   */
  static serializeDataSet(
    dataSet: Array<{ input: number[]; output: number[] }>,
  ): number[] {
    return serializeDataSet(dataSet);
  }

  /**
   * Activates a serialized network.
   * @param {number[]} inputValues - The input values.
   * @param {number[]} activationValues - The activations array.
   * @param {number[]} stateValues - The states array.
   * @param {number[]} serializedNetwork - The serialized network data.
   * @param {Function[]} activationFunctions - The activation functions.
   * @returns {number[]} The output values.
   */
  static activateSerializedNetwork(
    inputValues: number[],
    activationValues: number[],
    stateValues: number[],
    serializedNetwork: number[],
    activationFunctions: ActivationFn[],
  ): number[] {
    return activateSerializedNetwork(
      inputValues,
      activationValues,
      stateValues,
      serializedNetwork,
      activationFunctions,
    );
  }

  /**
   * Deserializes a dataset from a flat array.
   * @param {number[]} serializedSet - The serialized dataset.
   * @returns {Array<{ input: number[]; output: number[] }>} The deserialized dataset as an array of input-output pairs.
   */
  static deserializeDataSet(serializedSet: number[]): SerializedSample[] {
    return deserializeDataSet(serializedSet);
  }

  /**
   * Logistic activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static logistic(inputValue: number): number {
    return logisticActivation(inputValue);
  }

  /**
   * Hyperbolic tangent activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static tanh(inputValue: number): number {
    return tanhActivation(inputValue);
  }

  /**
   * Identity activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static identity(inputValue: number): number {
    return identityActivation(inputValue);
  }

  /**
   * Step activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static step(inputValue: number): number {
    return stepActivation(inputValue);
  }

  /**
   * Rectified Linear Unit (ReLU) activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static relu(inputValue: number): number {
    return reluActivation(inputValue);
  }

  /**
   * Softsign activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static softsign(inputValue: number): number {
    return softsignActivation(inputValue);
  }

  /**
   * Sinusoid activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static sinusoid(inputValue: number): number {
    return sinusoidActivation(inputValue);
  }

  /**
   * Gaussian activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static gaussian(inputValue: number): number {
    return gaussianActivation(inputValue);
  }

  /**
   * Bent Identity activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static bentIdentity(inputValue: number): number {
    return bentIdentityActivation(inputValue);
  }

  /**
   * Bipolar activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static bipolar(inputValue: number): number {
    return bipolarActivation(inputValue);
  }

  /**
   * Bipolar Sigmoid activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static bipolarSigmoid(inputValue: number): number {
    return bipolarSigmoidActivation(inputValue);
  }

  /**
   * Hard Tanh activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static hardTanh(inputValue: number): number {
    return hardTanhActivation(inputValue);
  }

  /**
   * Absolute activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static absolute(inputValue: number): number {
    return absoluteActivation(inputValue);
  }

  /**
   * Inverse activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static inverse(inputValue: number): number {
    return inverseActivation(inputValue);
  }

  /**
   * Scaled Exponential Linear Unit (SELU) activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static selu(inputValue: number): number {
    return seluActivation(inputValue);
  }

  /**
   * Softplus activation function. - Added
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static softplus(inputValue: number): number {
    return softplusActivation(inputValue);
  }

  /**
   * Tests a serialized dataset using a cost function.
   * @param {Array<{ input: number[]; output: number[] }>} serializedSampleSet - The serialized dataset as an array of input-output pairs.
   * @param {Function} cost - The cost function.
   * @param {number[]} activationValues - The activations array.
   * @param {number[]} stateValues - The states array.
   * @param {number[]} serializedNetwork - The serialized network data.
   * @param {Function[]} activationFunctions - The activation functions.
   * @returns {number} The average error.
   */
  static testSerializedSet(
    serializedSampleSet: SerializedSample[],
    cost: (expected: number[], actual: number[]) => number,
    activationValues: number[],
    stateValues: number[],
    serializedNetwork: number[],
    activationFunctions: ActivationFn[],
  ): number {
    return testSerializedSet(
      serializedSampleSet,
      cost,
      activationValues,
      stateValues,
      serializedNetwork,
      activationFunctions,
    );
  }

  /**
   * Gets the browser test worker.
   * @returns {Promise<TestWorkerConstructor>} The browser test worker.
   */
  static async getBrowserTestWorker(): Promise<TestWorkerConstructor> {
    const { TestWorker } = await import('./workers/browser/testworker');
    return TestWorker;
  }

  /**
   * Gets the node test worker.
   * @returns {Promise<TestWorkerConstructor>} The node test worker.
   */
  static async getNodeTestWorker(): Promise<TestWorkerConstructor> {
    const { TestWorker } = await import('./workers/node/testworker');
    return TestWorker;
  }
}
