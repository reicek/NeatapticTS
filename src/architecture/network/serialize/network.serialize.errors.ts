/**
 * Raised when network JSON serialization helpers receive an invalid root payload.
 */
export class NetworkSerializeInvalidJsonError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkSerializeInvalidJsonError';
  }
}
