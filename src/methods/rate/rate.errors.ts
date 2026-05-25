/**
 * Raised when a linear warmup-decay schedule receives a non-positive step count.
 */
export class RateLinearWarmupTotalStepsError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'RateLinearWarmupTotalStepsError';
  }
}
