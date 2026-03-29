/**
 * Worker-oriented evaluation helpers and serialization compatibility shelf.
 *
 * This chapter exists for a practical scaling problem: evolutionary evaluation
 * is usually embarrassingly parallel, but live `Network` instances, activation
 * closures, and environment-specific worker APIs do not cross thread boundaries
 * cleanly. The multithreading root turns that mismatch into a teachable
 * contract: flatten the network and dataset into portable numeric arrays, keep
 * activation functions in a stable index order, and let browser or Node workers
 * evaluate the same payload shape without needing the whole runtime object
 * graph.
 *
 * The most important idea here is not "threads are faster." It is boundary
 * control. A worker can only do useful NEAT work if the training host and the
 * worker agree on three things: how a network is serialized, how activations
 * are decoded, and how the result comes back as a scalar score. This root file
 * keeps that contract explicit so the rest of the library can talk about
 * parallel evaluation without hand-waving away the serialization boundary.
 *
 * The chapter is intentionally narrow. It does not implement a generic
 * scheduler or a broad actor framework. It exposes a small compatibility shelf
 * around two tasks that matter for evaluation: ship datasets and networks
 * across a worker boundary, and run the same ordered activation logic on the
 * other side. That makes the boundary useful both for actual worker-backed
 * evaluation and for teaching how data-parallel neural evaluation is shaped.
 *
 * Read the root in three passes:
 *
 * 1. `serializeDataSet()` and `deserializeDataSet()` for the portable sample
 *    format.
 * 2. `activations` and `activateSerializedNetwork()` for the flat execution
 *    contract.
 * 3. `getBrowserTestWorker()` and `getNodeTestWorker()` for the runtime-
 *    specific loader boundary.
 *
 * `browser/` and `node/` own the environment-specific worker wrappers, while
 * `multi.utils.ts` owns the flat-array execution mechanics. The root stays
 * orchestration-first for the same reason as the rest of the repo: readers
 * should see the contract before they see the inner loops.
 *
 * The background idea here is close to what distributed-systems and HPC writing
 * often call an embarrassingly parallel workload: each genome can be evaluated
 * independently once its inputs and scoring context are serialized. See
 * Wikipedia contributors,
 * [Embarrassingly parallel](https://en.wikipedia.org/wiki/Embarrassingly_parallel),
 * for compact background on why evolutionary evaluation is such a natural fit
 * for worker-style execution.
 *
 * ```mermaid
 * flowchart LR
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   Host[Training or test host]:::base --> Serialize[Flatten dataset and network]:::accent
 *   Serialize --> Worker[Browser or Node worker]:::base
 *   Worker --> Activate[Run ordered activation logic]:::base
 *   Activate --> Score[Return scalar evaluation]:::accent
 * ```
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#08131f,stroke:#1ea7ff,color:#dff6ff,stroke-width:1px;
 *   classDef accent fill:#0f2233,stroke:#ffd166,color:#fff4cc,stroke-width:1.5px;
 *
 *   Multi[Multi root facade]:::accent --> Dataset[serializeDataSet / deserializeDataSet]:::base
 *   Multi --> Runtime[Browser and Node worker loaders]:::base
 *   Multi --> Activations[Stable activation index registry]:::base
 *   Activations --> FlatExecution[activateSerializedNetwork]:::base
 *   Runtime --> Workers[workers/]:::base
 * ```
 *
 * Example: serialize one dataset once and evaluate a network in a Node worker.
 *
 * ```ts
 * const serializedSet = Multi.serializeDataSet([
 *   { input: [0, 0], output: [0] },
 *   { input: [1, 1], output: [1] },
 * ]);
 *
 * const NodeWorker = await Multi.getNodeTestWorker();
 * const worker = new NodeWorker(serializedSet, { name: 'mse' });
 * const score = await worker.evaluate(network);
 * worker.terminate();
 * ```
 *
 * Example: run the worker-compatible flat activation path locally.
 *
 * ```ts
 * const [activationValues, stateValues, serializedNetwork] = network.serialize();
 * const outputValues = Multi.activateSerializedNetwork(
 *   [0, 1],
 *   activationValues.slice(),
 *   stateValues.slice(),
 *   serializedNetwork,
 *   Multi.activations,
 * );
 * ```
 */
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
 * Stable compatibility facade for worker-oriented evaluation helpers.
 *
 * Read `Multi` as the small public shelf around three related contracts:
 * portable dataset serialization, flat-array network activation, and runtime-
 * specific worker loading. The heavier mechanics live in `multi.utils.ts` and
 * `workers/`, but the class keeps the outside-facing API compact and familiar.
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
