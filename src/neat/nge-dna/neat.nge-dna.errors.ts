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
 * Error thrown when one canonical CPPN descriptor or realization dispatch contract is invalid.
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

/**
 * Error thrown when the phenotype → Network bridge cannot materialize or extract
 * a canonical envelope, such as when a descriptor has zero modules or a Network
 * carries no NGE extension bag.
 *
 * This error is the bridge's only failure surface. It is thrown by
 * {@link materializeNetworkFromPhenotype} when the descriptor is empty and by
 * {@link extractCanonicalEnvelopeFromNetwork} when the Network's serialized
 * extension bag contains no `ngeEnvelope` carrier. Callers that need to
 * distinguish between the two failure modes should inspect the error message.
 *
 * @param message - Human-readable bridge validation failure message.
 * @param options - Optional native error options containing the underlying cause.
 *
 * @example
 * ```ts
 * try {
 *   materializeNetworkFromPhenotype(envelope, plan, emptyDescriptor);
 * } catch (err) {
 *   if (err instanceof NGE_DNA_BridgeError) {
 *     console.log(err.message); // "Cannot materialize a Network from a phenotype descriptor with zero modules."
 *   }
 * }
 * ```
 */
export class NGE_DNA_BridgeError extends Error {
  /**
   * @param message - Human-readable bridge validation failure message.
   * @param options - Optional native error options containing the underlying cause.
   */
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NGE_DNA_BridgeError';
  }
}
