/**
 * Raised when ONNX export cannot resolve a valid layered ordering.
 */
export class NetworkOnnxLayerOrderingUnresolvableError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkOnnxLayerOrderingUnresolvableError';
  }
}

/**
 * Raised when ONNX export encounters mixed activations without mixed-activation support enabled.
 */
export class NetworkOnnxMixedActivationsUnsupportedError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkOnnxMixedActivationsUnsupportedError';
  }
}

/**
 * Raised when ONNX export requires a connection that is missing.
 */
export class NetworkOnnxPartialConnectivityUnsupportedError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkOnnxPartialConnectivityUnsupportedError';
  }
}

/**
 * Raised when ONNX import perceptron metadata omits required input/output sizes.
 */
export class NetworkOnnxPerceptronSizeValidationError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkOnnxPerceptronSizeValidationError';
  }
}

/**
 * Raised when recurrent ONNX export encounters unsupported mixed activations.
 */
export class NetworkOnnxRecurrentMixedActivationsUnsupportedError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, { cause: options?.cause });
    this.name = 'NetworkOnnxRecurrentMixedActivationsUnsupportedError';
  }
}
