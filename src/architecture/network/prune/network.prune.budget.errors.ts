/**
 * Error raised when a sparsity-budget max-connection cap is invalid.
 */
export class NetworkPruneBudgetMaxConnectionsError extends RangeError {
  /**
   * Create a max-connections validation error.
   *
   * @param message - Human-readable validation detail.
   */
  constructor(message: string) {
    super(message);
    this.name = 'NetworkPruneBudgetMaxConnectionsError';
  }
}

/**
 * Error raised when sparsity-budget grace configuration is invalid.
 */
export class NetworkPruneBudgetGrowthGraceFractionError extends RangeError {
  /**
   * Create a growth-grace validation error.
   *
   * @param message - Human-readable validation detail.
   */
  constructor(message: string) {
    super(message);
    this.name = 'NetworkPruneBudgetGrowthGraceFractionError';
  }
}
