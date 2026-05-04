/**
 * Raised when crossover is requested for parents with incompatible IO dimensions.
 *
 * Crossover assumes both parents expose the same input/output interface.
 * If they do not, there is no unambiguous way to align node slots and produce
 * one runnable offspring network.
 */
export class NetworkGeneticParentCompatibilityError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkGeneticParentCompatibilityError';
  }
}
