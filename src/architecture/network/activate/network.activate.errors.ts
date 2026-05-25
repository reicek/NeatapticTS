/**
 * Raised when the input vector supplied to `activate()` does not match the
 * network's expected input size.
 *
 * This is the most common activation error. It fires when `input.length`
 * differs from `network.input` — for example, passing 3 values to a network
 * that expects 2.
 *
 * @example
 * ```ts
 * const network = new Network(2, 1);
 * network.activate([0, 1, 0.5]); // throws NetworkActivateInputSizeMismatchError
 * ```
 */
export class NetworkActivateInputSizeMismatchError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkActivateInputSizeMismatchError';
  }
}

/**
 * Raised when activation is attempted on a network whose node structure is
 * inconsistent or internally corrupted — for example, a node list that
 * contains `null` entries, or a network reconstructed from a malformed
 * serialized snapshot.
 *
 * If you encounter this error, inspect the network's `nodes` array before
 * activation and verify the deserialization path.
 */
export class NetworkActivateCorruptedStructureError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkActivateCorruptedStructureError';
  }
}

/**
 * Raised when `activateBatch()` receives a value that is not an array as its
 * top-level `inputs` argument.
 *
 * Each element of `inputs` must itself be a `number[]` input row. Passing a
 * single flat array of numbers (instead of an array of rows) is the most
 * common trigger.
 *
 * @example
 * ```ts
 * network.activateBatch([0, 1]); // throws — should be [[0, 1]]
 * ```
 */
export class NetworkActivateBatchInputsCollectionError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkActivateBatchInputsCollectionError';
  }
}
