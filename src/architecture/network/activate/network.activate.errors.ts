/**
 * Raised when activation input dimensionality does not match network expectations.
 */
export class NetworkActivateInputSizeMismatchError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkActivateInputSizeMismatchError';
  }
}

/**
 * Raised when activation is attempted on a network with invalid node structure.
 */
export class NetworkActivateCorruptedStructureError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkActivateCorruptedStructureError';
  }
}

/**
 * Raised when batch activation receives a non-array collection.
 */
export class NetworkActivateBatchInputsCollectionError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkActivateBatchInputsCollectionError';
  }
}
