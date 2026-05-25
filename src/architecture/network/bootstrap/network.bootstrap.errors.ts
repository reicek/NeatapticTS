/**
 * Raised when constructor topology intent conflicts with legacy acyclic flags.
 */
export class NetworkBootstrapTopologyIntentConflictError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkBootstrapTopologyIntentConflictError';
  }
}
