/**
 * Error thrown when one DNA payload is missing required identity fields or uses an incompatible schema version.
 */
export class NGE_DNA_SchemaError extends Error {
  /**
   * @param message - Human-readable validation failure message.
   * @param options - Optional native error options containing the underlying cause.
   */
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NGE_DNA_SchemaError';
  }
}

/**
 * Error thrown when one resolved substrate budget exceeds the module's conservative guardrails.
 */
export class NGE_DNA_BudgetError extends Error {
  /**
   * @param message - Human-readable budget validation failure message.
   * @param options - Optional native error options containing the underlying cause.
   */
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NGE_DNA_BudgetError';
  }
}

/**
 * Error thrown when one substrate coordinate or zone-partition input is invalid.
 */
export class NGE_DNA_SubstrateError extends Error {
  /**
   * @param message - Human-readable substrate validation failure message.
   * @param options - Optional native error options containing the underlying cause.
   */
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NGE_DNA_SubstrateError';
  }
}

/**
 * Error thrown when one Phase A CPPN descriptor or realization dispatch contract is invalid.
 */
export class NGE_DNA_CppnError extends Error {
  /**
   * @param message - Human-readable CPPN or dispatch validation failure message.
   * @param options - Optional native error options containing the underlying cause.
   */
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NGE_DNA_CppnError';
  }
}
