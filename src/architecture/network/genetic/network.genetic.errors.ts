/**
 * Raised when crossover is requested for parents with incompatible IO dimensions.
 */
export class NetworkGeneticParentCompatibilityError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkGeneticParentCompatibilityError';
  }
}
