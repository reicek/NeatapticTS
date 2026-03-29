/**
 * Raised when mutation is requested without a concrete mutation method.
 */
export class NetworkMutateMethodRequiredError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkMutateMethodRequiredError';
  }
}

/**
 * Raised when recurrent mutation helpers cannot access the created layer output nodes.
 */
export class NetworkMutateRecurrentLayerOutputInitializationError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkMutateRecurrentLayerOutputInitializationError';
  }
}
