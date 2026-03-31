/**
 * Raised when the training dataset is missing or does not match network IO dimensions.
 */
export class NetworkTrainingDatasetCompatibilityError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkTrainingDatasetCompatibilityError';
  }
}

/**
 * Raised when no stopping condition is provided to training.
 */
export class NetworkTrainingStoppingConditionRequiredError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkTrainingStoppingConditionRequiredError';
  }
}

/**
 * Raised when the provided cost function is not callable or recognized.
 */
export class NetworkTrainingInvalidCostFunctionError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkTrainingInvalidCostFunctionError';
  }
}

/**
 * Raised when dropout is outside the expected range [0, 1).
 */
export class NetworkTrainingDropoutRangeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkTrainingDropoutRangeError';
  }
}

/**
 * Raised when configured batch size exceeds dataset size.
 */
export class NetworkTrainingBatchSizeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkTrainingBatchSizeError';
  }
}

/**
 * Raised when accumulation steps is invalid.
 */
export class NetworkTrainingAccumulationStepsError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkTrainingAccumulationStepsError';
  }
}

/**
 * Raised when optimizer option type is not supported.
 */
export class NetworkTrainingInvalidOptimizerOptionError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkTrainingInvalidOptimizerOptionError';
  }
}

/**
 * Raised when optimizer type is unknown.
 */
export class NetworkTrainingUnknownOptimizerTypeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkTrainingUnknownOptimizerTypeError';
  }
}

/**
 * Raised when lookahead is configured with a nested lookahead base type.
 */
export class NetworkTrainingNestedLookaheadError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkTrainingNestedLookaheadError';
  }
}

/**
 * Raised when lookahead base optimizer type is unknown.
 */
export class NetworkTrainingUnknownLookaheadBaseTypeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkTrainingUnknownLookaheadBaseTypeError';
  }
}

/**
 * Raised when output target length does not match the network output width.
 */
export class NetworkTrainingOutputTargetLengthError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkTrainingOutputTargetLengthError';
  }
}
