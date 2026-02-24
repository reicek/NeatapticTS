/**
 * Activation function implementation type.
 * @param inputValue - Input to the activation function.
 * @param shouldComputeDerivative - Whether to compute the derivative instead of the value.
 * @returns Activation output or derivative at the input.
 */
export type ActivationFunction = (
  inputValue: number,
  shouldComputeDerivative?: boolean,
) => number;

/** Exponent used for tanh derivative: d/dx tanh(x) = 1 - tanh(x)^2. */
const tanhDerivativeExponent = 2;
/** Gaussian derivative factor: d/dx exp(-x^2) = -2x * exp(-x^2). */
const gaussianDerivativeCoefficient = -2;
/** Offset under the square root for bent identity: sqrt(x^2 + 1). */
const bentIdentitySqrtOffset = 1;
/** Divisor in bent identity slope term: x / (2 * sqrt(x^2 + 1)). */
const bentIdentityDerivativeDivisor = 2;
/** Linear offset added in bent identity derivative to keep slope near 1. */
const bentIdentityUnitOffset = 1;
/** Sigmoid scaling to map output from [0,1] to [-1,1] in bipolar sigmoid. */
const bipolarSigmoidScale = 2;
/** Derivative scale for bipolar sigmoid: 0.5 * (1 + y) * (1 - y). */
const bipolarSigmoidDerivativeScale = 0.5;
/** Lower clamp for hard tanh piecewise linear region. */
const hardTanhLowerBound = -1;
/** Upper clamp for hard tanh piecewise linear region. */
const hardTanhUpperBound = 1;
/** SELU alpha constant from Klambauer et al. (2017) for negative slope. */
// eslint-disable-next-line no-loss-of-precision
const seluAlpha = 1.6732632423543772848170429916717;
/** SELU scale constant ensuring self-normalizing variance. */
// eslint-disable-next-line no-loss-of-precision
const seluScale = 1.0507009873554804934193349852946;
/** Softplus upper threshold where log(1 + exp(x)) ~= x for stability. */
const softplusPositiveApproximationThreshold = 30;
/** Softplus lower threshold where log(1 + exp(x)) ~= exp(x) to avoid underflow. */
const softplusNegativeApproximationThreshold = -30;
/** CDF scaling factor for GELU approximation: 0.5 * (1 + tanh(...)). */
const geluCdfScale = 0.5;
/** Cubic term coefficient (0.044715) from Hendrycks & Gimpel GELU approximation. */
const geluTanhCoefficient = 0.044715;
/** Quadratic factor inside the GELU derivative approximation. */
const geluIntermediateCoefficient = 0.134145;
/** Shared exponent for squared terms in several activations. */
const squaredExponent = 2;
/** Precomputed sqrt(2/pi) factor used in GELU tanh argument. */
const geluSqrtTwoDividedByPi = Math.sqrt(2 / Math.PI);
/** Cubic exponent used in the GELU tanh approximation argument. */
const geluCubicExponent = 3;

/**
 * Logistic (sigmoid) activation implementation.
 * @param inputValue - Input to evaluate.
 * @param shouldComputeDerivative - Whether to compute the derivative.
 * @returns Logistic output or derivative.
 */
export function logisticActivation(
  inputValue: number,
  shouldComputeDerivative: boolean = false,
): number {
  const logisticValue = 1 / (1 + Math.exp(-inputValue));
  if (shouldComputeDerivative) {
    return logisticValue * (1 - logisticValue);
  }
  return logisticValue;
}

/**
 * Hyperbolic tangent activation implementation.
 * @param inputValue - Input to evaluate.
 * @param shouldComputeDerivative - Whether to compute the derivative.
 * @returns Tanh output or derivative.
 */
export function tanhActivation(
  inputValue: number,
  shouldComputeDerivative: boolean = false,
): number {
  if (!shouldComputeDerivative) {
    return Math.tanh(inputValue);
  }
  const tanhValue = Math.tanh(inputValue);
  return 1 - Math.pow(tanhValue, tanhDerivativeExponent);
}

/**
 * Identity activation implementation.
 * @param inputValue - Input to evaluate.
 * @param shouldComputeDerivative - Whether to compute the derivative.
 * @returns Identity output or derivative.
 */
export function identityActivation(
  inputValue: number,
  shouldComputeDerivative: boolean = false,
): number {
  return shouldComputeDerivative ? 1 : inputValue;
}

/**
 * Step activation implementation.
 * @param inputValue - Input to evaluate.
 * @param shouldComputeDerivative - Whether to compute the derivative.
 * @returns Step output or derivative.
 */
export function stepActivation(
  inputValue: number,
  shouldComputeDerivative: boolean = false,
): number {
  if (shouldComputeDerivative) {
    return 0;
  }
  return inputValue > 0 ? 1 : 0;
}

/**
 * Rectified Linear Unit (ReLU) activation implementation.
 * @param inputValue - Input to evaluate.
 * @param shouldComputeDerivative - Whether to compute the derivative.
 * @returns ReLU output or derivative.
 */
export function reluActivation(
  inputValue: number,
  shouldComputeDerivative: boolean = false,
): number {
  if (shouldComputeDerivative) {
    return inputValue > 0 ? 1 : 0;
  }
  return inputValue > 0 ? inputValue : 0;
}

/**
 * Softsign activation implementation.
 * @param inputValue - Input to evaluate.
 * @param shouldComputeDerivative - Whether to compute the derivative.
 * @returns Softsign output or derivative.
 */
export function softsignActivation(
  inputValue: number,
  shouldComputeDerivative: boolean = false,
): number {
  const denominator = 1 + Math.abs(inputValue);
  if (shouldComputeDerivative) {
    return 1 / Math.pow(denominator, squaredExponent);
  }
  return inputValue / denominator;
}

/**
 * Sinusoid activation implementation.
 * @param inputValue - Input to evaluate.
 * @param shouldComputeDerivative - Whether to compute the derivative.
 * @returns Sinusoid output or derivative.
 */
export function sinusoidActivation(
  inputValue: number,
  shouldComputeDerivative: boolean = false,
): number {
  return shouldComputeDerivative ? Math.cos(inputValue) : Math.sin(inputValue);
}

/**
 * Gaussian activation implementation.
 * @param inputValue - Input to evaluate.
 * @param shouldComputeDerivative - Whether to compute the derivative.
 * @returns Gaussian output or derivative.
 */
export function gaussianActivation(
  inputValue: number,
  shouldComputeDerivative: boolean = false,
): number {
  const gaussianValue = Math.exp(-Math.pow(inputValue, squaredExponent));
  if (shouldComputeDerivative) {
    return gaussianDerivativeCoefficient * inputValue * gaussianValue;
  }
  return gaussianValue;
}

/**
 * Bent identity activation implementation.
 * @param inputValue - Input to evaluate.
 * @param shouldComputeDerivative - Whether to compute the derivative.
 * @returns Bent identity output or derivative.
 */
export function bentIdentityActivation(
  inputValue: number,
  shouldComputeDerivative: boolean = false,
): number {
  const bentIdentityBase = Math.sqrt(
    Math.pow(inputValue, squaredExponent) + bentIdentitySqrtOffset,
  );
  if (shouldComputeDerivative) {
    return (
      inputValue / (bentIdentityDerivativeDivisor * bentIdentityBase) +
      bentIdentityUnitOffset
    );
  }
  return (
    (bentIdentityBase - bentIdentityUnitOffset) /
      bentIdentityDerivativeDivisor +
    inputValue
  );
}

/**
 * Bipolar activation implementation.
 * @param inputValue - Input to evaluate.
 * @param shouldComputeDerivative - Whether to compute the derivative.
 * @returns Bipolar output or derivative.
 */
export function bipolarActivation(
  inputValue: number,
  shouldComputeDerivative: boolean = false,
): number {
  if (shouldComputeDerivative) {
    return 0;
  }
  return inputValue > 0 ? 1 : -1;
}

/**
 * Bipolar sigmoid activation implementation.
 * @param inputValue - Input to evaluate.
 * @param shouldComputeDerivative - Whether to compute the derivative.
 * @returns Bipolar sigmoid output or derivative.
 */
export function bipolarSigmoidActivation(
  inputValue: number,
  shouldComputeDerivative: boolean = false,
): number {
  const bipolarSigmoidValue =
    bipolarSigmoidScale / (1 + Math.exp(-inputValue)) - 1;
  if (shouldComputeDerivative) {
    return (
      bipolarSigmoidDerivativeScale *
      (1 + bipolarSigmoidValue) *
      (1 - bipolarSigmoidValue)
    );
  }
  return bipolarSigmoidValue;
}

/**
 * Hard tanh activation implementation.
 * @param inputValue - Input to evaluate.
 * @param shouldComputeDerivative - Whether to compute the derivative.
 * @returns Hard tanh output or derivative.
 */
export function hardTanhActivation(
  inputValue: number,
  shouldComputeDerivative: boolean = false,
): number {
  if (shouldComputeDerivative) {
    return inputValue > hardTanhLowerBound && inputValue < hardTanhUpperBound
      ? 1
      : 0;
  }
  return Math.max(hardTanhLowerBound, Math.min(hardTanhUpperBound, inputValue));
}

/**
 * Absolute activation implementation.
 * @param inputValue - Input to evaluate.
 * @param shouldComputeDerivative - Whether to compute the derivative.
 * @returns Absolute output or derivative.
 */
export function absoluteActivation(
  inputValue: number,
  shouldComputeDerivative: boolean = false,
): number {
  if (shouldComputeDerivative) {
    return inputValue < 0 ? -1 : 1;
  }
  return Math.abs(inputValue);
}

/**
 * Inverse activation implementation.
 * @param inputValue - Input to evaluate.
 * @param shouldComputeDerivative - Whether to compute the derivative.
 * @returns Inverse output or derivative.
 */
export function inverseActivation(
  inputValue: number,
  shouldComputeDerivative: boolean = false,
): number {
  return shouldComputeDerivative ? -1 : 1 - inputValue;
}

/**
 * Scaled Exponential Linear Unit (SELU) activation implementation.
 * @param inputValue - Input to evaluate.
 * @param shouldComputeDerivative - Whether to compute the derivative.
 * @returns SELU output or derivative.
 */
export function seluActivation(
  inputValue: number,
  shouldComputeDerivative: boolean = false,
): number {
  const seluBase =
    inputValue > 0 ? inputValue : seluAlpha * Math.exp(inputValue) - seluAlpha;
  if (shouldComputeDerivative) {
    return inputValue > 0 ? seluScale : (seluBase + seluAlpha) * seluScale;
  }
  return seluBase * seluScale;
}

/**
 * Softplus activation implementation with stability guards.
 * @param inputValue - Input to evaluate.
 * @param shouldComputeDerivative - Whether to compute the derivative.
 * @returns Softplus output or derivative.
 */
export function softplusActivation(
  inputValue: number,
  shouldComputeDerivative: boolean = false,
): number {
  const logisticValue = logisticActivation(inputValue, false);
  if (shouldComputeDerivative) {
    return logisticValue;
  }
  if (inputValue > softplusPositiveApproximationThreshold) {
    return inputValue;
  }
  if (inputValue < softplusNegativeApproximationThreshold) {
    return Math.exp(inputValue);
  }
  return (
    Math.max(0, inputValue) + Math.log(1 + Math.exp(-Math.abs(inputValue)))
  );
}

/**
 * Swish activation implementation.
 * @param inputValue - Input to evaluate.
 * @param shouldComputeDerivative - Whether to compute the derivative.
 * @returns Swish output or derivative.
 */
export function swishActivation(
  inputValue: number,
  shouldComputeDerivative: boolean = false,
): number {
  const sigmoidValue = logisticActivation(inputValue, false);
  const swishValue = inputValue * sigmoidValue;
  if (shouldComputeDerivative) {
    return swishValue + sigmoidValue * (1 - swishValue);
  }
  return swishValue;
}

/**
 * Gaussian Error Linear Unit (GELU) activation implementation.
 * @param inputValue - Input to evaluate.
 * @param shouldComputeDerivative - Whether to compute the derivative.
 * @returns GELU output or derivative.
 */
export function geluActivation(
  inputValue: number,
  shouldComputeDerivative: boolean = false,
): number {
  const tanhArgument =
    geluSqrtTwoDividedByPi *
    (inputValue +
      geluTanhCoefficient * Math.pow(inputValue, geluCubicExponent));
  const cdfValue = geluCdfScale * (1 + Math.tanh(tanhArgument));
  if (shouldComputeDerivative) {
    const intermediateFactor =
      geluSqrtTwoDividedByPi *
      (1 + geluIntermediateCoefficient * inputValue * inputValue);
    const hyperbolicCosineValue = Math.cosh(tanhArgument);
    const hyperbolicSecant = 1 / hyperbolicCosineValue;
    const hyperbolicSecantSquared = hyperbolicSecant * hyperbolicSecant;
    return (
      cdfValue +
      inputValue * geluCdfScale * intermediateFactor * hyperbolicSecantSquared
    );
  }
  return inputValue * cdfValue;
}

/**
 * Mish activation implementation.
 * @param inputValue - Input to evaluate.
 * @param shouldComputeDerivative - Whether to compute the derivative.
 * @returns Mish output or derivative.
 */
export function mishActivation(
  inputValue: number,
  shouldComputeDerivative: boolean = false,
): number {
  const softplusValue = (() => {
    if (inputValue > softplusPositiveApproximationThreshold) {
      return inputValue;
    }
    if (inputValue < softplusNegativeApproximationThreshold) {
      return Math.exp(inputValue);
    }
    return (
      Math.max(0, inputValue) + Math.log(1 + Math.exp(-Math.abs(inputValue)))
    );
  })();

  const tanhSoftplusValue = Math.tanh(softplusValue);
  if (shouldComputeDerivative) {
    const sigmoidValue = logisticActivation(inputValue, false);
    const hyperbolicSecant = 1 / Math.cosh(softplusValue);
    const hyperbolicSecantSquared = hyperbolicSecant * hyperbolicSecant;
    return (
      tanhSoftplusValue + inputValue * hyperbolicSecantSquared * sigmoidValue
    );
  }
  return inputValue * tanhSoftplusValue;
}

/**
 * Sigmoid alias activation implementation.
 * @param inputValue - Input to evaluate.
 * @param shouldComputeDerivative - Whether to compute the derivative.
 * @returns Sigmoid output or derivative.
 */
export function sigmoidActivation(
  inputValue: number,
  shouldComputeDerivative: boolean = false,
): number {
  return logisticActivation(inputValue, shouldComputeDerivative);
}
