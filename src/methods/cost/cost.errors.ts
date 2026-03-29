/**
 * Raised when cost helpers receive target and output arrays of different lengths.
 */
export class CostTargetOutputLengthMismatchError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'CostTargetOutputLengthMismatchError';
  }
}
