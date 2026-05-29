/**
 * Error raised when one juvenile morph delta would exceed a DNA-configured budget.
 */
export class NgeJuvenile_BudgetError extends Error {
  /**
   * @param message - Human-readable budget failure description.
   * @param options - Optional error cause forwarded to the base `Error`.
   */
  public constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = 'NgeJuvenile_BudgetError';
  }
}

/**
 * Error raised when one dry-run juvenile morph validation fails locally.
 */
export class NgeJuvenile_MorphError extends Error {
  /**
   * @param message - Human-readable morph validation failure description.
   * @param options - Optional error cause forwarded to the base `Error`.
   */
  public constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = 'NgeJuvenile_MorphError';
  }
}

/**
 * Error raised when probe scheduling or ledger deserialization fails validation.
 */
export class NgeJuvenile_ProbeError extends Error {
  /**
   * @param message - Human-readable scheduler or ledger validation failure description.
   * @param options - Optional error cause forwarded to the base `Error`.
   */
  public constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = 'NgeJuvenile_ProbeError';
  }
}
