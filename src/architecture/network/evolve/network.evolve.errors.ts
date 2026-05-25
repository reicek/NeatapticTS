/**
 * Raised when the evolve dataset is missing or does not match network IO.
 *
 * @example
 * ```ts
 * throw new NetworkEvolveDatasetCompatibilityError(
 *   'Dataset should have at least one sample and matching input/output sizes.',
 * );
 * ```
 */
/**
 * Contract for NetworkEvolveDatasetCompatibilityError.
 */
export class NetworkEvolveDatasetCompatibilityError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkEvolveDatasetCompatibilityError';
  }
}

/**
 * Raised when evolve options do not declare any stopping condition.
 *
 * @example
 * ```ts
 * throw new NetworkEvolveStoppingConditionRequiredError(
 *   'Evolution requires either iterations or error to be set.',
 * );
 * ```
 */
/**
 * Contract for NetworkEvolveStoppingConditionRequiredError.
 */
export class NetworkEvolveStoppingConditionRequiredError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkEvolveStoppingConditionRequiredError';
  }
}
