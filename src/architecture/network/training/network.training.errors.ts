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
 * Raised when training is started without a stopping condition such as a maximum error target or iteration limit.
 */
export class NetworkTrainingStoppingConditionRequiredError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkTrainingStoppingConditionRequiredError';
  }
}

/**
 * Raised when the provided cost function is not callable or does not match any recognized cost-function identifier in the registry.
 */
export class NetworkTrainingInvalidCostFunctionError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkTrainingInvalidCostFunctionError';
  }
}

/**
 * Raised when the dropout probability falls outside the required half-open interval [0, 1) accepted by the training configuration validator.
 */
export class NetworkTrainingDropoutRangeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkTrainingDropoutRangeError';
  }
}

/**
 * Raised when the configured batch size exceeds the total dataset size, making mini-batch gradient accumulation impossible.
 */
export class NetworkTrainingBatchSizeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkTrainingBatchSizeError';
  }
}

/**
 * Raised when accumulation step count is zero, negative, or not a whole number as required for valid gradient accumulation.
 */
export class NetworkTrainingAccumulationStepsError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkTrainingAccumulationStepsError';
  }
}

/**
 * Raised when an optimizer configuration option carries a type that the selected optimizer does not recognize or accept.
 */
export class NetworkTrainingInvalidOptimizerOptionError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkTrainingInvalidOptimizerOptionError';
  }
}

/**
 * Raised when the optimizer type string does not match any registered optimizer in the network training configuration registry.
 */
export class NetworkTrainingUnknownOptimizerTypeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkTrainingUnknownOptimizerTypeError';
  }
}

/**
 * Raised when lookahead is configured with another lookahead optimizer as its base, which is not a supported inner optimizer combination.
 */
export class NetworkTrainingNestedLookaheadError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkTrainingNestedLookaheadError';
  }
}

/**
 * Raised when the lookahead base optimizer type does not match any supported inner optimizer in the current training stack configuration.
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
