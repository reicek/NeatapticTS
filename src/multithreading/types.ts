/**
 * Shared contracts for the multithreading boundary.
 *
 * These types keep the worker-facing evaluation surface small: ordered
 * activation functions, serialized input/output samples, a serializable network
 * shape, and the worker constructor protocol used by the browser and Node test
 * worker wrappers.
 */
export type ActivationFn = (x: number) => number;

export type SerializedSample = { input: number[]; output: number[] };

export interface SerializableNetwork {
  serialize(): [number[], number[], number[]];
}

export interface TestWorkerInstance {
  evaluate(network: SerializableNetwork): Promise<number> | number;
  terminate(): void;
  test?: () => unknown;
}

export interface TestWorkerConstructor {
  new (dataSet: number[], cost: { name: string }): TestWorkerInstance;
}
