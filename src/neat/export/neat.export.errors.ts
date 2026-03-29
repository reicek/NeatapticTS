/**
 * Raised when a persisted NEAT state bundle is missing or malformed.
 */
export class NeatExportStateBundleValidationError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NeatExportStateBundleValidationError';
  }
}

/**
 * Raised when a NEAT controller cannot be rehydrated from serialized state.
 */
export class NeatExportStateControllerRestoreError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NeatExportStateControllerRestoreError';
  }
}
