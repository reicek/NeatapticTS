import type { ActivationFn, SerializedSample } from './types';

const SERIALIZED_INPUT_LENGTH_INDEX = 0;
const SERIALIZED_OUTPUT_LENGTH_INDEX = 1;
const SERIALIZED_HEADER_LENGTH = 2;
const NO_SELF_GATER = -1;
const CONNECTION_LIST_TERMINATOR = -2;
const ZERO_VALUE = 0;
const UNIT_VALUE = 1;
const NEGATIVE_UNIT_VALUE = -1;
const BENT_IDENTITY_DIVISOR = 2;
const BIPOLAR_SIGMOID_SCALE = 2;
const STEP_THRESHOLD = 0;
const HARD_TANH_MIN = -1;
const HARD_TANH_MAX = 1;
const DEFAULT_GATE_VALUE = 1;
const NODE_HEADER_LENGTH = 5;
const CONNECTION_FIELD_COUNT = 3;
const SQUARED_EXPONENT = 2;
const GELU_CDF_SCALE = 0.5;
const GELU_TANH_COEFFICIENT = 0.044715;
const GELU_CUBIC_EXPONENT = 3;
const GELU_SQRT_TWO_DIVIDED_BY_PI = Math.sqrt(2 / Math.PI);
const SOFTPLUS_POSITIVE_APPROXIMATION_THRESHOLD = 20;
const SOFTPLUS_NEGATIVE_APPROXIMATION_THRESHOLD = -20;

// SELU constants from Klambauer et al., 2017
// eslint-disable-next-line no-loss-of-precision
const SELU_ALPHA = 1.6732632423543772848170429916717;
// eslint-disable-next-line no-loss-of-precision
const SELU_SCALE = 1.0507009873554804934193349852946;

/**
 * Ordered registry of all built-in activation functions for serialization compatibility.
 *
 * Worker threads decode the compact network format using the numeric activation
 * index stored per node. The index is a direct position into this array, so
 * **order must never change**. New activations must be appended at the end to
 * preserve backward compatibility with previously serialized networks.
 */
export const ACTIVATION_FUNCTIONS: ActivationFn[] = [
  logisticActivation,
  tanhActivation,
  identityActivation,
  stepActivation,
  reluActivation,
  softsignActivation,
  sinusoidActivation,
  gaussianActivation,
  bentIdentityActivation,
  bipolarActivation,
  bipolarSigmoidActivation,
  hardTanhActivation,
  absoluteActivation,
  inverseActivation,
  seluActivation,
  softplusActivation,
  swishActivation,
  geluActivation,
  mishActivation,
];

/**
 * Serializes a dataset into a flat numeric array.
 * The flattened layout minimizes worker message overhead by encoding one header followed by contiguous input and output rows for each sample.
 * @param dataSet - Collection of samples with input and output arrays.
 * @returns Flat serialized representation [inputCount, outputCount, ...samples].
 */
export function serializeDataSet(
  dataSet: Array<{ input: number[]; output: number[] }>,
): number[] {
  const inputCount = dataSet[0].input.length;
  const outputCount = dataSet[0].output.length;
  const serializedDataSet = [inputCount, outputCount];

  for (const sample of dataSet) {
    serializedDataSet.push(...sample.input);
    serializedDataSet.push(...sample.output);
  }

  return serializedDataSet;
}

/**
 * Activates a serialized network and produces outputs.
 * This interpreter executes the compact numeric encoding used by worker threads, including self-gated recurrent state updates and per-edge gating, so predictions can run without rehydrating full object graphs.
 * @param inputValues - Inputs to feed into the network.
 * @param activationValues - Mutable activation register shared across runs.
 * @param stateValues - Mutable state register shared across runs.
 * @param serializedNetwork - Flat encoded network data.
 * @param activationFunctions - Ordered activation functions.
 * @returns Activated outputs.
 */
export function activateSerializedNetwork(
  inputValues: number[],
  activationValues: number[],
  stateValues: number[],
  serializedNetwork: number[],
  activationFunctions: ActivationFn[],
): number[] {
  const inputCount = serializedNetwork[SERIALIZED_INPUT_LENGTH_INDEX];
  const outputCount = serializedNetwork[SERIALIZED_OUTPUT_LENGTH_INDEX];

  for (let inputIndex = 0; inputIndex < inputCount; inputIndex += 1) {
    activationValues[inputIndex] = inputValues[inputIndex];
  }

  let serializedIndex = SERIALIZED_HEADER_LENGTH;
  while (serializedIndex < serializedNetwork.length) {
    const nodeIndex = serializedNetwork[serializedIndex];
    const nodeBias = serializedNetwork[serializedIndex + 1];
    const activationIndex = serializedNetwork[serializedIndex + 2];
    const selfWeight = serializedNetwork[serializedIndex + 3];
    const selfGaterIndex = serializedNetwork[serializedIndex + 4];
    serializedIndex += NODE_HEADER_LENGTH;

    const selfGateValue =
      selfGaterIndex === NO_SELF_GATER
        ? DEFAULT_GATE_VALUE
        : activationValues[selfGaterIndex];
    stateValues[nodeIndex] =
      selfGateValue * selfWeight * stateValues[nodeIndex] + nodeBias;

    while (serializedNetwork[serializedIndex] !== CONNECTION_LIST_TERMINATOR) {
      const sourceIndex = serializedNetwork[serializedIndex];
      const connectionWeight = serializedNetwork[serializedIndex + 1];
      const gaterIndex = serializedNetwork[serializedIndex + 2];
      serializedIndex += CONNECTION_FIELD_COUNT;

      const gateValue =
        gaterIndex === NO_SELF_GATER
          ? DEFAULT_GATE_VALUE
          : activationValues[gaterIndex];
      stateValues[nodeIndex] +=
        activationValues[sourceIndex] * connectionWeight * gateValue;
    }

    serializedIndex += 1; // Skip CONNECTION_LIST_TERMINATOR.
    activationValues[nodeIndex] = activationFunctions[activationIndex](
      stateValues[nodeIndex],
    );
  }

  const outputStartIndex = activationValues.length - outputCount;
  return activationValues.slice(outputStartIndex);
}

/**
 * Deserializes a dataset from its flat representation.
 * The deserializer reverses `serializeDataSet` by reconstructing fixed-width sample rows from the shared header, preserving deterministic sample order for batch evaluation.
 * @param serializedSet - Flat serialized dataset array.
 * @returns Array of input/output sample pairs.
 */
export function deserializeDataSet(
  serializedSet: number[],
): SerializedSample[] {
  const deserializedSet: SerializedSample[] = [];
  const inputCount = serializedSet[SERIALIZED_INPUT_LENGTH_INDEX];
  const outputCount = serializedSet[SERIALIZED_OUTPUT_LENGTH_INDEX];
  const sampleSize = inputCount + outputCount;
  const sampleCount =
    (serializedSet.length - SERIALIZED_HEADER_LENGTH) / sampleSize;

  for (let sampleIndex = 0; sampleIndex < sampleCount; sampleIndex += 1) {
    const sampleOffset = SERIALIZED_HEADER_LENGTH + sampleIndex * sampleSize;
    const inputValues = serializedSet.slice(
      sampleOffset,
      sampleOffset + inputCount,
    );
    const outputValues = serializedSet.slice(
      sampleOffset + inputCount,
      sampleOffset + sampleSize,
    );
    deserializedSet.push({ input: inputValues, output: outputValues });
  }

  return deserializedSet;
}

/**
 * Logistic (sigmoid) activation — maps any real input to the open interval (0, 1).
 *
 * Commonly used in output layers for binary classification or as a smooth
 * squashing function. See Wikipedia contributors,
 * [Sigmoid function](https://en.wikipedia.org/wiki/Sigmoid_function).
 *
 * @param value - Pre-activation input value.
 * @returns Activation output in the range (0, 1).
 */
export function logisticActivation(value: number): number {
  return UNIT_VALUE / (UNIT_VALUE + Math.exp(-value));
}

/**
 * Hyperbolic tangent activation — maps any real input to the open interval (−1, 1).
 *
 * Zero-centered and saturating; a common choice for hidden layers. See Wikipedia
 * contributors, [Hyperbolic functions](https://en.wikipedia.org/wiki/Hyperbolic_functions).
 *
 * @param value - Pre-activation input value.
 * @returns Activation output in the range (−1, 1).
 */
export function tanhActivation(value: number): number {
  return Math.tanh(value);
}

/**
 * Identity (linear) activation — passes the input through unchanged.
 *
 * Useful for output nodes in regression networks where no squashing is desired.
 *
 * @param value - Pre-activation input value.
 * @returns The same value, unmodified.
 */
export function identityActivation(value: number): number {
  return value;
}

/**
 * Step (Heaviside) activation — outputs 1 for positive inputs, 0 otherwise.
 *
 * A hard threshold function with zero gradient almost everywhere. Rarely used
 * in gradient-based training but useful for binary thresholding in evaluation.
 *
 * @param value - Pre-activation input value.
 * @returns 1 if value > 0, otherwise 0.
 */
export function stepActivation(value: number): number {
  return value > STEP_THRESHOLD ? UNIT_VALUE : ZERO_VALUE;
}

/**
 * Rectified Linear Unit (ReLU) activation — passes positive values, zeros negatives.
 *
 * The most widely used hidden-layer activation in deep learning due to its
 * computational simplicity and resistance to vanishing gradients. See Wikipedia
 * contributors, [Rectifier](https://en.wikipedia.org/wiki/Rectifier_(neural_networks)).
 *
 * @param value - Pre-activation input value.
 * @returns value if value > 0, otherwise 0.
 */
export function reluActivation(value: number): number {
  return value > STEP_THRESHOLD ? value : ZERO_VALUE;
}

/**
 * Softsign activation — a smooth, non-saturating alternative to tanh.
 *
 * Outputs range in (−1, 1) but with gentler saturation than tanh, preserving
 * gradient flow further from zero. Formula: x / (1 + |x|).
 *
 * @param value - Pre-activation input value.
 * @returns Activation output in the range (−1, 1).
 */
export function softsignActivation(value: number): number {
  return value / (UNIT_VALUE + Math.abs(value));
}

/**
 * Sinusoidal activation — applies the sine function to the pre-activation value.
 *
 * Produces periodic, bounded output in [−1, 1]. Useful for networks that
 * need to learn cyclic or frequency-based patterns.
 *
 * @param value - Pre-activation input value.
 * @returns sin(value).
 */
export function sinusoidActivation(value: number): number {
  return Math.sin(value);
}

/**
 * Gaussian activation — bell-curve response centered at zero.
 *
 * Outputs the normal probability density shape exp(−x²), which peaks at 1
 * when x = 0 and decays to 0 for large |x|. Useful in radial basis function
 * style networks.
 *
 * @param value - Pre-activation input value.
 * @returns exp(−value²).
 */
export function gaussianActivation(value: number): number {
  return Math.exp(-Math.pow(value, SQUARED_EXPONENT));
}

/**
 * Bent identity activation — smooth, near-linear with gentle curvature.
 *
 * Outputs approximately x for large |x| but introduces a small nonlinear
 * bend near zero. Formula: (sqrt(x² + 1) − 1) / 2 + x.
 *
 * @param value - Pre-activation input value.
 * @returns Bent identity output.
 */
export function bentIdentityActivation(value: number): number {
  return (
    (Math.sqrt(Math.pow(value, SQUARED_EXPONENT) + UNIT_VALUE) - UNIT_VALUE) /
      BENT_IDENTITY_DIVISOR +
    value
  );
}

/**
 * Bipolar step activation — outputs +1 for positive inputs, −1 otherwise.
 *
 * A hard threshold centered at zero. Useful as a binary decision unit where
 * outputs must be exactly ±1 rather than 0/1.
 *
 * @param value - Pre-activation input value.
 * @returns 1 if value > 0, otherwise −1.
 */
export function bipolarActivation(value: number): number {
  return value > STEP_THRESHOLD ? UNIT_VALUE : NEGATIVE_UNIT_VALUE;
}

/**
 * Bipolar sigmoid activation — a logistic function rescaled to the range (−1, 1).
 *
 * Formula: 2 / (1 + exp(−x)) − 1. Equivalent to tanh in range but computed
 * differently; retains the zero-crossing property of bipolar functions.
 *
 * @param value - Pre-activation input value.
 * @returns Activation output in the range (−1, 1).
 */
export function bipolarSigmoidActivation(value: number): number {
  return BIPOLAR_SIGMOID_SCALE / (UNIT_VALUE + Math.exp(-value)) - UNIT_VALUE;
}

/**
 * Hard tanh activation — clamps the input to the range [−1, 1].
 *
 * A piecewise linear approximation of tanh that is free of exponential
 * operations. Output is exactly −1, identity, or +1 depending on the input.
 *
 * @param value - Pre-activation input value.
 * @returns Clamped value in [−1, 1].
 */
export function hardTanhActivation(value: number): number {
  return Math.max(HARD_TANH_MIN, Math.min(HARD_TANH_MAX, value));
}

/**
 * Absolute value activation — maps the input to its non-negative magnitude.
 *
 * Introduces a V-shaped nonlinearity: zero gradient for positive inputs,
 * negated gradient for negative inputs. Useful where magnitude matters but
 * sign does not.
 *
 * @param value - Pre-activation input value.
 * @returns |value|.
 */
export function absoluteActivation(value: number): number {
  return Math.abs(value);
}

/**
 * Inverse (complement) activation — reflects the input around 0.5.
 *
 * Formula: 1 − x. Useful when a node's output should represent the
 * complementary probability or the negated contribution of its input.
 *
 * @param value - Pre-activation input value.
 * @returns 1 − value.
 */
export function inverseActivation(value: number): number {
  return UNIT_VALUE - value;
}

/**
 * Scaled Exponential Linear Unit (SELU) activation — self-normalizing variant of ELU.
 *
 * Designed to push activations toward zero mean and unit variance when used
 * throughout a fully connected network, without explicit batch normalization.
 * Constants α and λ from Klambauer et al., 2017. See Wikipedia contributors,
 * [SELU](https://en.wikipedia.org/wiki/Activation_function#Scaled_exponential_linear_unit).
 *
 * @param value - Pre-activation input value.
 * @returns Scaled activation output.
 */
export function seluActivation(value: number): number {
  const seluBase =
    value > STEP_THRESHOLD ? value : SELU_ALPHA * Math.exp(value) - SELU_ALPHA;
  return seluBase * SELU_SCALE;
}

/**
 * Softplus activation — a smooth approximation of ReLU.
 *
 * Formula: log(1 + exp(x)). Always positive; approaches x for large x and 0
 * for large negative x. Uses numerical approximations at the tails to avoid
 * overflow.
 *
 * @param value - Pre-activation input value.
 * @returns log(1 + exp(value)), with tail approximations for stability.
 */
export function softplusActivation(value: number): number {
  if (value > SOFTPLUS_POSITIVE_APPROXIMATION_THRESHOLD) {
    return value;
  }

  if (value < SOFTPLUS_NEGATIVE_APPROXIMATION_THRESHOLD) {
    return Math.exp(value);
  }

  return Math.max(0, value) + Math.log(UNIT_VALUE + Math.exp(-Math.abs(value)));
}

/**
 * Swish activation — gated variant of the identity function.
 *
 * Formula: x · σ(x), where σ is the logistic sigmoid. Self-gated,
 * non-monotonic, and smooth. Empirically outperforms ReLU on deeper
 * architectures. Proposed by Ramachandran et al., 2017.
 *
 * @param value - Pre-activation input value.
 * @returns value · sigmoid(value).
 */
export function swishActivation(value: number): number {
  return value * logisticActivation(value);
}

/**
 * Gaussian Error Linear Unit (GELU) activation — smooth stochastic regularizer.
 *
 * Formula: x · Φ(x), where Φ is the Gaussian CDF approximated via tanh.
 * GELU weighs inputs by their probability under a standard normal, producing
 * smooth, non-monotonic behavior. Widely used in transformer architectures.
 * See Hendrycks and Gimpel, 2016.
 *
 * @param value - Pre-activation input value.
 * @returns GELU-activated output.
 */
export function geluActivation(value: number): number {
  const tanhArgument =
    GELU_SQRT_TWO_DIVIDED_BY_PI *
    (value + GELU_TANH_COEFFICIENT * Math.pow(value, GELU_CUBIC_EXPONENT));
  const cdfValue = GELU_CDF_SCALE * (UNIT_VALUE + Math.tanh(tanhArgument));
  return value * cdfValue;
}

/**
 * Mish smooth non-monotonic activation function.
 *
 * Computes `x * tanh(softplus(x))` where `softplus(x) = ln(1 + e^x)`.
 * Mish avoids hard zero-saturation and provides better gradient flow than ReLU
 * in many deep architectures. See Misra, 2019, "Mish: A Self Regularized Non-Monotonic Activation Function".
 *
 * @param value - Pre-activation input value.
 * @returns Mish-activated output.
 */
export function mishActivation(value: number): number {
  return value * Math.tanh(softplusActivation(value));
}

/**
 * Tests a serialized dataset using a cost function.
 * Each sample is evaluated through the serialized-network interpreter and accumulated into an average finite cost, returning `NaN` when any sample or score violates numeric validity.
 * @param serializedSampleSet - Serialized dataset samples.
 * @param costFunction - Cost function comparing expected and actual outputs.
 * @param activationValues - Mutable activation register.
 * @param stateValues - Mutable state register.
 * @param serializedNetwork - Serialized network data.
 * @param activationFunctions - Activation functions to apply.
 * @returns Average cost or NaN when invalid input.
 */
export function testSerializedSet(
  serializedSampleSet: SerializedSample[],
  costFunction: (expected: number[], actual: number[]) => number,
  activationValues: number[],
  stateValues: number[],
  serializedNetwork: number[],
  activationFunctions: ActivationFn[],
): number {
  if (serializedSampleSet.length === 0) return Number.NaN;

  let errorSum = 0;
  for (const sample of serializedSampleSet) {
    const outputValues = activateSerializedNetwork(
      sample.input,
      activationValues,
      stateValues,
      serializedNetwork,
      activationFunctions,
    );
    const costValue = costFunction(sample.output, outputValues);
    if (!Number.isFinite(costValue)) return Number.NaN;
    errorSum += costValue;
  }

  return errorSum / serializedSampleSet.length;
}
