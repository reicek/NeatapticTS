/**
 * Shared contracts for the multithreading boundary.
 *
 * These types keep the worker-facing evaluation surface small: ordered
 * activation functions, serialized input/output samples, a serializable network
 * shape, and the worker constructor protocol used by the browser and Node test
 * worker wrappers.
 */
export type ActivationFn = (x: number) => number;

/**
 * A single input/output training sample for worker-based batch evaluation.
 *
 * Both arrays must have lengths consistent with the network's input and
 * output dimensions. The serialized dataset format encodes these lengths
 * once in a shared header so worker threads can decode samples without
 * out-of-band metadata.
 */
export type SerializedSample = { input: number[]; output: number[] };

/**
 * Minimal interface required of a network to participate in worker evaluation.
 *
 * Only `serialize()` is needed: workers receive the flat numeric triple
 * produced by this method and reconstruct activation state locally without
 * holding a reference to the full `Network` object graph.
 */
export interface SerializableNetwork {
  serialize(): [number[], number[], number[]];
}

/**
 * Contract for a running worker instance used in parallel genome evaluation.
 *
 * `evaluate` scores a single genome and returns a fitness value. `terminate`
 * shuts the worker down cleanly. The optional `test` hook exists for
 * diagnostic harnesses that need to probe internal worker state.
 */
export interface TestWorkerInstance {
  evaluate(network: SerializableNetwork): Promise<number> | number;
  terminate(): void;
  test?: () => unknown;
}

/**
 * Constructor signature for worker classes used in parallel genome evaluation.
 *
 * Implementations receive the flat-serialized dataset and the cost function
 * descriptor at construction time so the worker can score genomes without
 * receiving per-call dataset transfers.
 */
export interface TestWorkerConstructor {
  new (dataSet: number[], cost: { name: string }): TestWorkerInstance;
}
