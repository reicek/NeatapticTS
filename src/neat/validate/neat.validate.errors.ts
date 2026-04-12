import type { NativeGenomeValidationIssue } from './neat.validate.types';

/**
 * Raised when a native NEAT genome violates proper-NEAT identity invariants.
 *
 * The validator itself returns a rich report for tests and tooling, while this
 * error class powers fail-fast assertion helpers that want exception semantics.
 */
export class NeatNativeGenomeValidationError extends Error {
  /** Structured validator findings attached to the thrown error. */
  readonly issues: NativeGenomeValidationIssue[];

  constructor(
    message: string,
    issues: NativeGenomeValidationIssue[],
    options?: ErrorOptions,
  ) {
    super(message, { cause: options?.cause });
    this.name = 'NeatNativeGenomeValidationError';
    this.issues = issues;
  }
}