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

// SELU constants from Klambauer et al., 2017
// eslint-disable-next-line no-loss-of-precision
const SELU_ALPHA = 1.6732632423543772848170429916717;
// eslint-disable-next-line no-loss-of-precision
const SELU_SCALE = 1.0507009873554804934193349852946;

/** @returns Activation functions ordered for serialization compatibility. */
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
];

/**
 * Serializes a dataset into a flat numeric array.
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

/** @param value - Input value. @returns Logistic activation. */
export function logisticActivation(value: number): number {
  return UNIT_VALUE / (UNIT_VALUE + Math.exp(-value));
}

/** @param value - Input value. @returns Hyperbolic tangent activation. */
export function tanhActivation(value: number): number {
  return Math.tanh(value);
}

/** @param value - Input value. @returns Identity activation. */
export function identityActivation(value: number): number {
  return value;
}

/** @param value - Input value. @returns Step activation. */
export function stepActivation(value: number): number {
  return value > STEP_THRESHOLD ? UNIT_VALUE : ZERO_VALUE;
}

/** @param value - Input value. @returns ReLU activation. */
export function reluActivation(value: number): number {
  return value > STEP_THRESHOLD ? value : ZERO_VALUE;
}

/** @param value - Input value. @returns Softsign activation. */
export function softsignActivation(value: number): number {
  return value / (UNIT_VALUE + Math.abs(value));
}

/** @param value - Input value. @returns Sinusoid activation. */
export function sinusoidActivation(value: number): number {
  return Math.sin(value);
}

/** @param value - Input value. @returns Gaussian activation. */
export function gaussianActivation(value: number): number {
  return Math.exp(-Math.pow(value, SQUARED_EXPONENT));
}

/** @param value - Input value. @returns Bent identity activation. */
export function bentIdentityActivation(value: number): number {
  return (
    (Math.sqrt(Math.pow(value, SQUARED_EXPONENT) + UNIT_VALUE) - UNIT_VALUE) /
      BENT_IDENTITY_DIVISOR +
    value
  );
}

/** @param value - Input value. @returns Bipolar activation. */
export function bipolarActivation(value: number): number {
  return value > STEP_THRESHOLD ? UNIT_VALUE : NEGATIVE_UNIT_VALUE;
}

/** @param value - Input value. @returns Bipolar sigmoid activation. */
export function bipolarSigmoidActivation(value: number): number {
  return BIPOLAR_SIGMOID_SCALE / (UNIT_VALUE + Math.exp(-value)) - UNIT_VALUE;
}

/** @param value - Input value. @returns Hard tanh activation. */
export function hardTanhActivation(value: number): number {
  return Math.max(HARD_TANH_MIN, Math.min(HARD_TANH_MAX, value));
}

/** @param value - Input value. @returns Absolute activation. */
export function absoluteActivation(value: number): number {
  return Math.abs(value);
}

/** @param value - Input value. @returns Inverse activation. */
export function inverseActivation(value: number): number {
  return UNIT_VALUE - value;
}

/** @param value - Input value. @returns SELU activation. */
export function seluActivation(value: number): number {
  const seluBase =
    value > STEP_THRESHOLD ? value : SELU_ALPHA * Math.exp(value) - SELU_ALPHA;
  return seluBase * SELU_SCALE;
}

/** @param value - Input value. @returns Softplus activation. */
export function softplusActivation(value: number): number {
  return Math.log(UNIT_VALUE + Math.exp(value));
}

/**
 * Tests a serialized dataset using a cost function.
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
