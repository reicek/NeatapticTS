import type { NeatGenomeValidationIssue } from './genome.types';

/**
 * Raised when one boundary tries to project malformed state into the strict
 * genome contract.
 */
export class NeatGenomeConversionError extends Error {
  /**
   * @param message Human-readable conversion failure.
   * @param cause Optional structured cause.
   */
  constructor(message: string, cause?: unknown) {
    super(message, cause === undefined ? undefined : { cause });
    this.name = 'NeatGenomeConversionError';
  }
}

/**
 * Raised when a strict genome contract fails validation.
 */
export class NeatGenomeValidationError extends Error {
  /** Structured validator findings attached to the thrown error. */
  readonly issues: NeatGenomeValidationIssue[];

  /**
   * @param message Human-readable validation failure.
   * @param issues Structured validator findings.
   */
  constructor(message: string, issues: NeatGenomeValidationIssue[]) {
    super(message);
    this.name = 'NeatGenomeValidationError';
    this.issues = issues;
  }
}