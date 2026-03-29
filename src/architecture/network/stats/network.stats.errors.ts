/**
 * Raised when network test helpers receive a missing or empty evaluation set.
 */
export class NetworkStatsTestSetValidationError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkStatsTestSetValidationError';
  }
}

/**
 * Raised when a test sample input vector does not match network input width.
 */
export class NetworkStatsTestSampleInputSizeMismatchError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkStatsTestSampleInputSizeMismatchError';
  }
}

/**
 * Raised when a test sample output vector does not match network output width.
 */
export class NetworkStatsTestSampleOutputSizeMismatchError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkStatsTestSampleOutputSizeMismatchError';
  }
}
