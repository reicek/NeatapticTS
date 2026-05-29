/**
 * Error thrown when one assimilation candidate would exceed the configured structural budget.
 */
export class AssimilationBudgetError extends Error {
  /**
   * @param message - Human-readable description of the budget failure.
   * @param options - Optional native error options containing the underlying cause.
   */
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = 'AssimilationBudgetError';
  }
}

/**
 * Error thrown when one assimilation candidate fails schema or envelope validation.
 */
export class AssimilationSchemaError extends Error {
  /**
   * @param message - Human-readable description of the schema validation failure.
   * @param options - Optional native error options containing the underlying cause.
   */
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = 'AssimilationSchemaError';
  }
}
