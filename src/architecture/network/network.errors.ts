/**
 * Raised when a network is constructed without the required input or output sizes.
 */
export class NetworkConstructorDimensionRequiredError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkConstructorDimensionRequiredError';
  }
}
