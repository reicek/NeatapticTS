/**
 * Raised when a pruning schedule window size is zero, negative, or otherwise falls outside the valid range accepted by the activation-ordering runtime.
 */
export class NetworkRuntimePruningScheduleWindowError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRuntimePruningScheduleWindowError';
  }
}

/**
 * Raised when pruning target sparsity is outside the open interval (0, 1).
 */
export class NetworkRuntimeTargetSparsityRangeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRuntimeTargetSparsityRangeError';
  }
}

/**
 * Raised when the weight-noise standard deviation is negative; only non-negative values produce a well-defined Gaussian noise distribution.
 */
export class NetworkRuntimeWeightNoiseStdDevRangeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRuntimeWeightNoiseStdDevRangeError';
  }
}

/**
 * Raised when the weight-noise configuration contains an unexpected shape, is missing required fields, or carries incompatible type combinations.
 */
export class NetworkRuntimeWeightNoiseConfigurationError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRuntimeWeightNoiseConfigurationError';
  }
}

/**
 * Raised when per-hidden-layer weight noise is requested but the target network was not constructed with an explicit layered topology.
 */
export class NetworkRuntimeLayeredWeightNoiseRequiredError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRuntimeLayeredWeightNoiseRequiredError';
  }
}

/**
 * Raised when the number of hidden-layer weight-noise entries does not match the hidden-layer count in the network topology.
 */
export class NetworkRuntimeWeightNoiseEntryCountError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRuntimeWeightNoiseEntryCountError';
  }
}

/**
 * Raised when a per-hidden-layer weight-noise standard deviation is negative; each layer entry must be zero or a positive value.
 */
export class NetworkRuntimeWeightNoisePerLayerRangeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRuntimeWeightNoisePerLayerRangeError';
  }
}

/**
 * Raised when the stochastic-depth survival probability input is not an array of per-layer probability values as required by the runtime.
 */
export class NetworkRuntimeStochasticDepthSurvivalArrayError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRuntimeStochasticDepthSurvivalArrayError';
  }
}

/**
 * Raised when a stochastic-depth survival probability value falls outside the open-closed interval (0, 1] required for valid layer retention.
 */
export class NetworkRuntimeStochasticDepthSurvivalRangeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRuntimeStochasticDepthSurvivalRangeError';
  }
}

/**
 * Raised when stochastic depth is requested but the target network was not constructed with an explicit layered topology as required.
 */
export class NetworkRuntimeStochasticDepthLayeredNetworkRequiredError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRuntimeStochasticDepthLayeredNetworkRequiredError';
  }
}

/**
 * Raised when the count of stochastic-depth survival entries does not match the number of hidden layers in the network topology.
 */
export class NetworkRuntimeStochasticDepthEntryCountError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRuntimeStochasticDepthEntryCountError';
  }
}

/**
 * Raised when the DropConnect drop probability falls outside the required half-open interval [0, 1) for valid stochastic connection masking.
 */
export class NetworkRuntimeDropConnectProbabilityRangeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRuntimeDropConnectProbabilityRangeError';
  }
}
