/**
 * Shared contract for activation implementations used by the runtime registry.
 *
 * This utility layer is where the math-facing side of the activation chapter
 * lives. Each exported function keeps the same two-mode signature so the
 * higher-level {@link Activation} registry can switch between forward
 * evaluation and derivative lookup without wrapping every implementation in a
 * different adapter.
 *
 * Read the implementations in four practical groups:
 *
 * - bounded classics such as {@link logisticActivation},
 *   {@link sigmoidActivation}, and {@link tanhActivation},
 * - sparse or piecewise-linear gates such as {@link reluActivation},
 *   {@link stepActivation}, and {@link hardTanhActivation},
 * - shape-specialized functions such as {@link gaussianActivation},
 *   {@link sinusoidActivation}, and {@link bentIdentityActivation},
 * - smoother modern hidden-layer candidates such as {@link softplusActivation},
 *   {@link swishActivation}, {@link geluActivation}, and
 *   {@link mishActivation}.
 *
 * A useful reading order is:
 *
 * 1. compare {@link logisticActivation}, {@link tanhActivation}, and
 *    {@link reluActivation} to anchor the classic bounded-versus-sparse trade,
 * 2. scan {@link softplusActivation}, {@link swishActivation},
 *    {@link geluActivation}, and {@link mishActivation} when you want smoother
 *    hidden-layer behavior,
 * 3. finish with the shape-specialized helpers when your experiment needs a
 *    periodic, radial, or sign-like response rather than a general default.
 *
 * Keep the derivative flag in mind while reading: every helper answers both the
 * forward-value question and the local-slope question, which is why the file is
 * organized around reusable transfer-curve families instead of separate forward
 * and derivative tables.
 *
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
 *
 * This is the bounded baseline most readers already know: useful when a node
 * should behave like a probability-like squashing unit, but also the easiest
 * activation to saturate if pre-activation values become too large in
 * magnitude.
 *
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
 *
 * Compared with {@link logisticActivation}, tanh stays bounded while remaining
 * zero-centered, which often makes hidden activations easier to interpret when
 * positive and negative evidence should balance around zero.
 *
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
 *
 * Use this when you explicitly want a linear pass-through unit. It is most
 * common in regression-style output layers or in experiments where the upstream
 * topology already provides the non-linearity and you only need a value relay.
 *
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
 *
 * This is the hard-threshold ancestor of smoother gates: it cleanly separates
 * negative from positive evidence, but its zero derivative almost everywhere
 * makes it a poor default for gradient-based training.
 *
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
 *
 * ReLU is the practical default for many hidden layers because it is cheap,
 * sparse, and easy to optimize. This implementation follows the common library
 * convention of returning `0` for the derivative at exactly `0`, even though
 * the mathematical derivative is not uniquely defined there.
 *
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
 *
 * Softsign behaves like a gentler, cheaper-to-reason-about cousin of tanh. It
 * still compresses values into a bounded range, but the tails decay more
 * gradually, which can make it a useful comparison point when tanh feels too
 * eager to saturate.
 *
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
 *
 * Reach for this when periodic structure matters more than monotonicity. A
 * sinusoidal response can encode cycles and phase relationships that ordinary
 * squashing activations tend to smooth away.
 *
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
 *
 * Gaussian responses peak at the origin and shrink toward zero on both sides,
 * which makes them useful when you want a node to behave more like a localized
 * detector than a broad monotonic amplifier.
 *
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
 *
 * Bent identity stays close to a linear pass-through while adding a gentle
 * non-linearity near the origin. It is useful when pure identity feels too weak
 * but a strongly saturating activation would distort the signal too early.
 *
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
 *
 * This is the sign-function version of a hard classifier: values collapse to
 * `-1` or `1` with no middle ground. It is mostly useful as a historical or
 * diagnostic contrast against smoother bounded activations.
 *
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
 *
 * This is the `[-1, 1]` counterpart to {@link logisticActivation}. In practice
 * it is another route to a tanh-like curve, but the explicit bipolar naming is
 * helpful when comparing older NEAT-era literature or porting legacy settings.
 *
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
 *
 * Hard tanh keeps tanh's familiar `[-1, 1]` output range while replacing the
 * curved middle section with a cheap piecewise-linear clamp. That makes it a
 * practical compromise when you want bounded outputs without paying for a full
 * smooth tanh evaluation.
 *
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
 *
 * Absolute value discards sign and keeps only magnitude. It is unusual as a
 * default hidden-layer choice, but it can be useful in experiments where the
 * intensity of a signal matters more than whether it was positive or negative.
 *
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
 *
 * This helper mirrors a value around `1`, which makes it more of a niche
 * transformation than a general-purpose hidden activation. It is best read as
 * part of the method vocabulary's legacy and experimentation shelf rather than
 * as a recommended first-choice default.
 *
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
 *
 * SELU couples its nonlinearity with fixed scale constants so layers tend to
 * drift back toward a stable mean and variance under the assumptions of the
 * self-normalizing network paper. In practice, it is most useful when the
 * whole hidden stack is designed around SELU rather than mixed casually with
 * unrelated activation families.
 *
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
 *
 * Softplus is the smooth sibling of {@link reluActivation}: for large positive
 * values it behaves almost linearly, for large negative values it fades toward
 * zero, and around the origin it avoids the hard corner that makes ReLU
 * piecewise. The threshold checks keep the implementation numerically stable in
 * the extreme tails.
 *
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
 *
 * Swish multiplies the input by a sigmoid gate, so the unit can dampen itself
 * smoothly instead of snapping to zero as ReLU does. That makes it a useful
 * comparison point when experimenting with smoother hidden-layer behavior.
 *
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
 *
 * GELU keeps the soft gating idea of Swish but uses a Gaussian-CDF-shaped
 * weighting curve. This implementation uses the common tanh-based
 * approximation, which is fast enough for ordinary training code while staying
 * close to the exact GELU shape used in many transformer-era models.
 *
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
 *
 * Mish is another smooth self-gated option, built from `x * tanh(softplus(x))`.
 * It keeps more negative-side nuance than ReLU while still rewarding large
 * positive evidence. The implementation reuses the same softplus stability
 * strategy so the helper behaves sensibly in the far positive and negative
 * tails.
 *
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
 *
 * This intentionally delegates to {@link logisticActivation} so callers can use
 * either the mathematically explicit `logistic` name or the more common
 * deep-learning alias `sigmoid` without creating two separate implementations.
 *
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
