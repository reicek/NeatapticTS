/**
 * Raised when a pruning schedule window is invalid.
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
 * Raised when weight-noise standard deviation is negative.
 */
export class NetworkRuntimeWeightNoiseStdDevRangeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRuntimeWeightNoiseStdDevRangeError';
  }
}

/**
 * Raised when weight-noise configuration shape is invalid.
 */
export class NetworkRuntimeWeightNoiseConfigurationError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRuntimeWeightNoiseConfigurationError';
  }
}

/**
 * Raised when per-hidden-layer weight noise is requested on a non-layered network.
 */
export class NetworkRuntimeLayeredWeightNoiseRequiredError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRuntimeLayeredWeightNoiseRequiredError';
  }
}

/**
 * Raised when hidden-layer weight-noise entries do not match hidden-layer count.
 */
export class NetworkRuntimeWeightNoiseEntryCountError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRuntimeWeightNoiseEntryCountError';
  }
}

/**
 * Raised when a per-hidden-layer weight-noise value is negative.
 */
export class NetworkRuntimeWeightNoisePerLayerRangeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRuntimeWeightNoisePerLayerRangeError';
  }
}

/**
 * Raised when stochastic-depth survival input is not an array.
 */
export class NetworkRuntimeStochasticDepthSurvivalArrayError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRuntimeStochasticDepthSurvivalArrayError';
  }
}

/**
 * Raised when a stochastic-depth survival probability falls outside (0, 1].
 */
export class NetworkRuntimeStochasticDepthSurvivalRangeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRuntimeStochasticDepthSurvivalRangeError';
  }
}

/**
 * Raised when stochastic depth is requested on a non-layered network.
 */
export class NetworkRuntimeStochasticDepthLayeredNetworkRequiredError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRuntimeStochasticDepthLayeredNetworkRequiredError';
  }
}

/**
 * Raised when stochastic-depth survival entries do not match hidden-layer count.
 */
export class NetworkRuntimeStochasticDepthEntryCountError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRuntimeStochasticDepthEntryCountError';
  }
}

/**
 * Raised when DropConnect probability is outside [0, 1).
 */
export class NetworkRuntimeDropConnectProbabilityRangeError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkRuntimeDropConnectProbabilityRangeError';
  }
}
