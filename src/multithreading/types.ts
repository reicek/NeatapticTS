/**
 * Shared types for multithreading helpers and test workers.
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
