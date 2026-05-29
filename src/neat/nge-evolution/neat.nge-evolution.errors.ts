/**
 * Error thrown when one requested reproduction mode is unavailable for the current evolution context.
 */
export class NgeEvolution_ModeError extends Error {
  /**
   * @param message - Human-readable reproduction-mode failure message.
   * @param options - Optional native error options containing the underlying cause.
   */
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NgeEvolution_ModeError';
  }
}

/**
 * Error thrown when one polyandric or region-based assignment request is invalid.
 */
export class NgeEvolution_RegionError extends Error {
  /**
   * @param message - Human-readable region-assignment failure message.
   * @param options - Optional native error options containing the underlying cause.
   */
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NgeEvolution_RegionError';
  }
}

/**
 * Error thrown when one evolution operator exceeds the configured structural or patch budget.
 */
export class NgeEvolution_BudgetError extends Error {
  /**
   * @param message - Human-readable budget failure message.
   * @param options - Optional native error options containing the underlying cause.
   */
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NgeEvolution_BudgetError';
  }
}
