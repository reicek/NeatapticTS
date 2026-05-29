/**
 * Error classes for the NGE collective multi-agent system (Phase G).
 *
 * These errors are raised when shared-field operations receive invalid inputs or
 * when a collective evaluation tick cannot proceed due to an evaluator mismatch.
 *
 * Both classes forward an optional `{ cause }` to the base `Error` constructor so
 * that callers can chain underlying exceptions for full stack attribution.
 *
 * @example
 * ```ts
 * import { NgeCollective_FieldDimensionError, NgeCollective_EvaluationError } from './neat.nge-collective.errors';
 *
 * // Raised when width or height is non-positive or non-integer.
 * throw new NgeCollective_FieldDimensionError('width must be a positive integer');
 *
 * // Raised when evaluator array length does not match registered agent count.
 * throw new NgeCollective_EvaluationError('no evaluator registered for agent 2');
 * ```
 */

/**
 * Error raised when a shared-field operation receives invalid grid dimensions.
 *
 * @example
 * ```ts
 * throw new NgeCollective_FieldDimensionError('width must be a positive integer');
 * ```
 */
export class NgeCollective_FieldDimensionError extends Error {
  /**
   * @param message - Human-readable description of the dimension violation.
   * @param options - Optional error cause forwarded to the base `Error`.
   */
  public constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = 'NgeCollective_FieldDimensionError';
  }
}

/**
 * Error raised when a collective evaluation tick cannot proceed due to a
 * mismatched evaluator count or a missing evaluator for a registered agent.
 *
 * @example
 * ```ts
 * throw new NgeCollective_EvaluationError('no evaluator registered for agent 2');
 * ```
 */
export class NgeCollective_EvaluationError extends Error {
  /**
   * @param message - Human-readable description of the evaluation failure.
   * @param options - Optional error cause forwarded to the base `Error`.
   */
  public constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = 'NgeCollective_EvaluationError';
  }
}
