/**
 * Error raised when adult plateau evaluation cannot produce a valid local decision.
 */
export class NgeAdult_PlateauError extends Error {
  /**
   * @param message - Human-readable plateau evaluation failure description.
   * @param options - Optional error cause forwarded to the base `Error`.
   */
  public constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = 'NgeAdult_PlateauError';
  }
}

/**
 * Error raised when adult equilibrium evaluation cannot produce a valid local decision.
 */
export class NgeAdult_EquilibriumError extends Error {
  /**
   * @param message - Human-readable equilibrium evaluation failure description.
   * @param options - Optional error cause forwarded to the base `Error`.
   */
  public constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = 'NgeAdult_EquilibriumError';
  }
}
