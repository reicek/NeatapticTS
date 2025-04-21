import { Workers } from './workers/workers';
import Network from '../architecture/network';

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
  static activations: Array<(x: number) => number> = [
    (x) => 1 / (1 + Math.exp(-x)), // Logistic (0)
    (x) => Math.tanh(x), // Tanh (1)
    (x) => x, // Identity (2)
    (x) => (x > 0 ? 1 : 0), // Step (3)
    (x) => (x > 0 ? x : 0), // ReLU (4)
    (x) => x / (1 + Math.abs(x)), // Softsign (5)
    (x) => Math.sin(x), // Sinusoid (6)
    (x) => Math.exp(-Math.pow(x, 2)), // Gaussian (7)
    (x) => (Math.sqrt(Math.pow(x, 2) + 1) - 1) / 2 + x, // Bent Identity (8)
    (x) => (x > 0 ? 1 : -1), // Bipolar (9)
    (x) => 2 / (1 + Math.exp(-x)) - 1, // Bipolar Sigmoid (10)
    (x) => Math.max(-1, Math.min(1, x)), // Hard Tanh (11)
    (x) => Math.abs(x), // Absolute (12)
    (x) => 1 - x, // Inverse (13)
    (x) => {
      // SELU (14)
      const alpha = 1.6732632423543772848170429916717;
      const scale = 1.0507009873554804934193349852946;
      const fx = x > 0 ? x : alpha * Math.exp(x) - alpha;
      return fx * scale;
    },
    (x) => Math.log(1 + Math.exp(x)), // Softplus (15) - Added
  ];

  /**
   * Serializes a dataset into a flat array.
   * @param {Array<{ input: number[]; output: number[] }>} dataSet - The dataset to serialize.
   * @returns {number[]} The serialized dataset.
   */
  static serializeDataSet(
    dataSet: Array<{ input: number[]; output: number[] }>
  ): number[] {
    const serialized = [dataSet[0].input.length, dataSet[0].output.length];

    for (let i = 0; i < dataSet.length; i++) {
      for (let j = 0; j < serialized[0]; j++) {
        serialized.push(dataSet[i].input[j]);
      }
      for (let j = 0; j < serialized[1]; j++) {
        serialized.push(dataSet[i].output[j]);
      }
    }

    return serialized;
  }

  /**
   * Activates a serialized network.
   * @param {number[]} input - The input values.
   * @param {number[]} A - The activations array.
   * @param {number[]} S - The states array.
   * @param {number[]} data - The serialized network data.
   * @param {Function[]} F - The activation functions.
   * @returns {number[]} The output values.
   */
  static activateSerializedNetwork(
    input: number[],
    A: number[],
    S: number[],
    data: number[],
    F: Function[]
  ): number[] {
    for (let i = 0; i < data[0]; i++) A[i] = input[i];
    for (let i = 2; i < data.length; i++) {
      const index = data[i++];
      const bias = data[i++];
      const squash = data[i++];
      const selfweight = data[i++];
      const selfgater = data[i++];

      S[index] =
        (selfgater === -1 ? 1 : A[selfgater]) * selfweight * S[index] + bias;

      while (data[i] !== -2) {
        S[index] +=
          A[data[i++]] * data[i++] * (data[i++] === -1 ? 1 : A[data[i - 1]]);
      }
      A[index] = F[squash](S[index]);
    }

    const output = [];
    for (let i = A.length - data[1]; i < A.length; i++) output.push(A[i]);
    return output;
  }

  /**
   * Deserializes a dataset from a flat array.
   * @param {number[]} serializedSet - The serialized dataset.
   * @returns {Array<{ input: number[]; output: number[] }>} The deserialized dataset as an array of input-output pairs.
   */
  static deserializeDataSet(
    serializedSet: number[]
  ): Array<{ input: number[]; output: number[] }> {
    const set: Array<{ input: number[]; output: number[] }> = [];
    const sampleSize = serializedSet[0] + serializedSet[1];

    for (let i = 0; i < (serializedSet.length - 2) / sampleSize; i++) {
      const input: number[] = [];
      for (
        let j = 2 + i * sampleSize;
        j < 2 + i * sampleSize + serializedSet[0];
        j++
      ) {
        input.push(serializedSet[j]);
      }
      const output: number[] = [];
      for (
        let j = 2 + i * sampleSize + serializedSet[0];
        j < 2 + i * sampleSize + sampleSize;
        j++
      ) {
        output.push(serializedSet[j]);
      }
      set.push({ input, output });
    }

    return set;
  }

  /**
   * Logistic activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static logistic(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }

  /**
   * Hyperbolic tangent activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static tanh(x: number): number {
    return Math.tanh(x);
  }

  /**
   * Identity activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static identity(x: number): number {
    return x;
  }

  /**
   * Step activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static step(x: number): number {
    return x > 0 ? 1 : 0;
  }

  /**
   * Rectified Linear Unit (ReLU) activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static relu(x: number): number {
    return x > 0 ? x : 0;
  }

  /**
   * Softsign activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static softsign(x: number): number {
    return x / (1 + Math.abs(x));
  }

  /**
   * Sinusoid activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static sinusoid(x: number): number {
    return Math.sin(x);
  }

  /**
   * Gaussian activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static gaussian(x: number): number {
    return Math.exp(-Math.pow(x, 2));
  }

  /**
   * Bent Identity activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static bentIdentity(x: number): number {
    return (Math.sqrt(Math.pow(x, 2) + 1) - 1) / 2 + x;
  }

  /**
   * Bipolar activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static bipolar(x: number): number {
    return x > 0 ? 1 : -1;
  }

  /**
   * Bipolar Sigmoid activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static bipolarSigmoid(x: number): number {
    return 2 / (1 + Math.exp(-x)) - 1;
  }

  /**
   * Hard Tanh activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static hardTanh(x: number): number {
    return Math.max(-1, Math.min(1, x));
  }

  /**
   * Absolute activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static absolute(x: number): number {
    return Math.abs(x);
  }

  /**
   * Inverse activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static inverse(x: number): number {
    return 1 - x;
  }

  /**
   * Scaled Exponential Linear Unit (SELU) activation function.
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static selu(x: number): number {
    const alpha = 1.6732632423543772848170429916717;
    const scale = 1.0507009873554804934193349852946;
    const fx = x > 0 ? x : alpha * Math.exp(x) - alpha; // Corrected definition
    return fx * scale;
  }

  /**
   * Softplus activation function. - Added
   * @param {number} x - The input value.
   * @returns {number} The activated value.
   */
  static softplus(x: number): number {
    return Math.log(1 + Math.exp(x));
  }

  /**
   * Tests a serialized dataset using a cost function.
   * @param {Array<{ input: number[]; output: number[] }>} set - The serialized dataset as an array of input-output pairs.
   * @param {Function} cost - The cost function.
   * @param {number[]} A - The activations array.
   * @param {number[]} S - The states array.
   * @param {number[]} data - The serialized network data.
   * @param {Function[]} F - The activation functions.
   * @returns {number} The average error.
   */
  static testSerializedSet(
    set: Array<{ input: number[]; output: number[] }>,
    cost: (expected: number[], actual: number[]) => number,
    A: number[],
    S: number[],
    data: number[],
    F: Function[]
  ): number {
    let error = 0;

    for (let i = 0; i < set.length; i++) {
      const output = Multi.activateSerializedNetwork(
        set[i].input,
        A,
        S,
        data,
        F
      );
      error += cost(set[i].output, output);
    }

    return error / set.length;
  }

  /**
   * Gets the browser test worker.
   * @returns {Promise<any>} The browser test worker.
   */
  static async getBrowserTestWorker() {
    const { TestWorker } = await import('./workers/browser/testworker');
    return TestWorker;
  }

  /**
   * Gets the node test worker.
   * @returns {Promise<any>} The node test worker.
   */
  static async getNodeTestWorker() {
    const { TestWorker } = await import('./workers/node/testworker'); // Corrected path
    return TestWorker;
  }
}
