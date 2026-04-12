/**
 * Raised when a persisted NEAT state bundle is missing or malformed.
 *
 * This error is generally thrown while validating the raw checkpoint payload
 * (JSON shape + required keys) before attempting to restore any runtime state.
 */
export class NeatExportStateBundleValidationError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NeatExportStateBundleValidationError';
  }
}

/**
 * Raised when a NEAT controller cannot be rehydrated from serialized state.
 *
 * This error indicates the payload may be well-formed JSON, but it cannot be
 * safely mapped onto the current controller instance (for example: missing
 * referenced genomes, duplicated stable ids, or incompatible replay contract
 * expectations).
 */
export class NeatExportStateControllerRestoreError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NeatExportStateControllerRestoreError';
  }
}

/**
 * Raised when a serialized population snapshot cannot be restored safely.
 *
 * Callers should treat this as a hard stop: continuing with a partially
 * validated population snapshot can silently corrupt deterministic replay.
 */
export class NeatExportPopulationValidationError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NeatExportPopulationValidationError';
  }
}
