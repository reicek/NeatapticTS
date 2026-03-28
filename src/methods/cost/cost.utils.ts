import { PROB_EPSILON } from '../../neat/neat.constants';

/** Error message thrown when target and output arrays differ in length. */
export const LENGTH_MISMATCH_MESSAGE =
  'Target and output arrays must have the same length.';

/** Canonical positive label used by binary-oriented helpers. */
export const POSITIVE_CLASS_LABEL = 1;

/** Canonical negative label used by binary-oriented helpers. */
export const NEGATIVE_CLASS_LABEL = 0;

/** Threshold for binarizing probabilities into class predictions. */
export const BINARY_CLASSIFICATION_THRESHOLD = 0.5;

/** Margin enforced by hinge loss. */
export const HINGE_MARGIN = 1;

/** Default focusing parameter for focal loss. */
export const DEFAULT_FOCAL_GAMMA = 2;

/** Default class balancing parameter for focal loss. */
export const DEFAULT_FOCAL_ALPHA = 0.25;

/** Default smoothing factor for label smoothing. */
export const DEFAULT_LABEL_SMOOTHING = 0.1;

/** Baseline probability used when smoothing targets. */
export const LABEL_SMOOTHING_BASELINE = 0.5;

/** Lower bound for softmax denominator to avoid division by zero. */
export const SOFTMAX_SUM_GUARD = 1;

const PROBABILITY_LOWER_BOUND = PROB_EPSILON;
const PROBABILITY_UPPER_BOUND = 1 - PROB_EPSILON;
const NON_NEGATIVE_FLOOR = 0;

/**
 * Computes the Cross Entropy error over the provided targets and outputs.
 *
 * @param targets - Desired target probabilities (may be soft labels between 0 and 1).
 * @param outputs - Model output probabilities.
 * @returns Mean cross-entropy error across all samples.
 */
export function computeCrossEntropy(
  targets: number[],
  outputs: number[],
): number {
  validateMatchingLength(targets, outputs);

  const totalError = outputs.reduce(
    (accumulatedError, outputProbability, outputIndex) => {
      const targetProbability = targets[outputIndex];
      const clampedProbability = clampProbability(outputProbability);
      const sampleError = crossEntropyTerm(
        targetProbability,
        clampedProbability,
      );
      return accumulatedError + sampleError;
    },
    0,
  );

  return totalError / outputs.length;
}

/**
 * Computes the softmax cross entropy given targets and raw score outputs.
 *
 * @param targets - Desired target probabilities that should sum to 1 (will be normalized if not).
 * @param outputs - Raw logits or scores for each class.
 * @returns Total (non-averaged) softmax cross-entropy loss.
 */
export function computeSoftmaxCrossEntropy(
  targets: number[],
  outputs: number[],
): number {
  validateMatchingLength(targets, outputs);

  const normalizedTargets = normalizeTargets(targets);
  const softmaxProbabilities = stableSoftmax(outputs);

  const totalLoss = softmaxProbabilities.reduce(
    (accumulatedLoss, softmaxProbability, outputIndex) => {
      const targetProbability = normalizedTargets[outputIndex];
      const boundedProbability = clampProbability(softmaxProbability);
      return accumulatedLoss - targetProbability * Math.log(boundedProbability);
    },
    0,
  );

  return totalLoss;
}

/**
 * Computes mean squared error between targets and outputs.
 *
 * @param targets - Desired target values.
 * @param outputs - Model outputs.
 * @returns Mean squared error.
 */
export function computeMeanSquaredError(
  targets: number[],
  outputs: number[],
): number {
  validateMatchingLength(targets, outputs);

  const totalSquaredError = outputs.reduce(
    (accumulatedError, outputValue, outputIndex) => {
      const targetValue = targets[outputIndex];
      const squaredDifference = (targetValue - outputValue) ** 2;
      return accumulatedError + squaredDifference;
    },
    0,
  );

  return totalSquaredError / outputs.length;
}

/**
 * Computes binary classification error rate.
 *
 * @param targets - Target labels (0 or 1).
 * @param outputs - Predicted probabilities.
 * @returns Proportion of misclassified samples.
 */
export function computeBinaryError(
  targets: number[],
  outputs: number[],
): number {
  validateMatchingLength(targets, outputs);

  const totalMisses = outputs.reduce(
    (missCount, outputProbability, outputIndex) => {
      const targetProbability = targets[outputIndex];
      const targetLabel = classifyBinary(targetProbability);
      const predictedLabel = classifyBinary(outputProbability);
      return targetLabel === predictedLabel ? missCount : missCount + 1;
    },
    0,
  );

  return totalMisses / outputs.length;
}

/**
 * Computes mean absolute error between targets and outputs.
 *
 * @param targets - Desired target values.
 * @param outputs - Model outputs.
 * @returns Mean absolute error.
 */
export function computeMeanAbsoluteError(
  targets: number[],
  outputs: number[],
): number {
  validateMatchingLength(targets, outputs);

  const totalAbsoluteError = outputs.reduce(
    (accumulatedError, outputValue, outputIndex) => {
      const targetValue = targets[outputIndex];
      const absoluteDifference = Math.abs(targetValue - outputValue);
      return accumulatedError + absoluteDifference;
    },
    0,
  );

  return totalAbsoluteError / outputs.length;
}

/**
 * Computes mean absolute percentage error between targets and outputs.
 *
 * @param targets - Desired target values.
 * @param outputs - Model outputs.
 * @returns Mean absolute percentage error (fractional form).
 */
export function computeMeanAbsolutePercentageError(
  targets: number[],
  outputs: number[],
): number {
  validateMatchingLength(targets, outputs);

  const totalPercentageError = outputs.reduce(
    (accumulatedError, outputValue, outputIndex) => {
      const targetValue = targets[outputIndex];
      const safeDenominator = Math.max(
        Math.abs(targetValue),
        PROBABILITY_LOWER_BOUND,
      );
      const absolutePercentageError = Math.abs(
        (targetValue - outputValue) / safeDenominator,
      );
      return accumulatedError + absolutePercentageError;
    },
    0,
  );

  return totalPercentageError / outputs.length;
}

/**
 * Computes mean squared logarithmic error between targets and outputs.
 *
 * @param targets - Desired non-negative target values.
 * @param outputs - Model outputs (expected non-negative).
 * @returns Mean squared logarithmic error.
 */
export function computeMeanSquaredLogarithmicError(
  targets: number[],
  outputs: number[],
): number {
  validateMatchingLength(targets, outputs);

  const totalLogarithmicError = outputs.reduce(
    (accumulatedError, outputValue, outputIndex) => {
      const targetValue = targets[outputIndex];
      const logTarget = Math.log1p(Math.max(targetValue, NON_NEGATIVE_FLOOR));
      const logOutput = Math.log1p(Math.max(outputValue, NON_NEGATIVE_FLOOR));
      const squaredDifference = (logTarget - logOutput) ** 2;
      return accumulatedError + squaredDifference;
    },
    0,
  );

  return totalLogarithmicError / outputs.length;
}

/**
 * Computes hinge loss for margin-based classification.
 *
 * @param targets - Target labels encoded as -1 or 1.
 * @param outputs - Model outputs (raw scores).
 * @returns Mean hinge loss.
 */
export function computeHingeLoss(targets: number[], outputs: number[]): number {
  validateMatchingLength(targets, outputs);

  const totalHingeLoss = outputs.reduce(
    (accumulatedLoss, outputValue, outputIndex) => {
      const targetValue = targets[outputIndex];
      const marginViolation = Math.max(
        NON_NEGATIVE_FLOOR,
        HINGE_MARGIN - targetValue * outputValue,
      );
      return accumulatedLoss + marginViolation;
    },
    0,
  );

  return totalHingeLoss / outputs.length;
}

/**
 * Computes focal loss for imbalanced classification tasks.
 *
 * @param targets - Target labels (0 or 1) or soft labels.
 * @param outputs - Predicted probabilities.
 * @param gamma - Focusing parameter controlling hard example emphasis.
 * @param alpha - Balancing parameter for class weighting.
 * @returns Mean focal loss.
 */
export function computeFocalLoss(
  targets: number[],
  outputs: number[],
  gamma: number,
  alpha: number,
): number {
  validateMatchingLength(targets, outputs);

  const totalFocalLoss = outputs.reduce(
    (accumulatedLoss, outputProbability, outputIndex) => {
      const targetProbability = targets[outputIndex];
      const boundedProbability = clampProbability(outputProbability);
      const probabilityForTarget =
        targetProbability === POSITIVE_CLASS_LABEL
          ? boundedProbability
          : 1 - boundedProbability;
      const alphaForTarget =
        targetProbability === POSITIVE_CLASS_LABEL ? alpha : 1 - alpha;
      const modulationFactor = (1 - probabilityForTarget) ** gamma;
      const sampleLoss =
        -alphaForTarget * modulationFactor * Math.log(probabilityForTarget);
      return accumulatedLoss + sampleLoss;
    },
    0,
  );

  return totalFocalLoss / outputs.length;
}

/**
 * Computes cross entropy with label smoothing applied to targets.
 *
 * @param targets - Target labels (0 or 1) or soft labels.
 * @param outputs - Predicted probabilities.
 * @param smoothing - Smoothing factor between 0 and 1.
 * @returns Mean cross-entropy loss with smoothed targets.
 */
export function computeLabelSmoothingLoss(
  targets: number[],
  outputs: number[],
  smoothing: number,
): number {
  validateMatchingLength(targets, outputs);

  const totalLoss = outputs.reduce(
    (accumulatedLoss, outputProbability, outputIndex) => {
      const targetProbability = targets[outputIndex];
      const smoothedTarget = smoothTarget(targetProbability, smoothing);
      const boundedProbability = clampProbability(outputProbability);
      const complementaryProbability = clampProbability(1 - boundedProbability);
      const sampleLoss =
        -smoothedTarget * Math.log(boundedProbability) -
        (1 - smoothedTarget) * Math.log(complementaryProbability);
      return accumulatedLoss + sampleLoss;
    },
    0,
  );

  return totalLoss / outputs.length;
}

function validateMatchingLength(targets: number[], outputs: number[]): void {
  if (targets.length !== outputs.length) {
    throw new Error(LENGTH_MISMATCH_MESSAGE);
  }
}

/**
 * Clamps a probability into the inclusive bounds defined by PROBABILITY_LOWER_BOUND and PROBABILITY_UPPER_BOUND.
 *
 * @param probability - Raw probability value to bound.
 * @returns Probability constrained to the numeric stability range.
 */
function clampProbability(probability: number): number {
  return Math.max(
    PROBABILITY_LOWER_BOUND,
    Math.min(PROBABILITY_UPPER_BOUND, probability),
  );
}

/**
 * Computes the cross-entropy contribution for a single target/output pair.
 *
 * @param targetProbability - Target probability for the sample (may be soft).
 * @param clampedProbability - Output probability already clamped for stability.
 * @returns Cross-entropy term for the sample.
 */
function crossEntropyTerm(
  targetProbability: number,
  clampedProbability: number,
): number {
  if (targetProbability === POSITIVE_CLASS_LABEL) {
    return -Math.log(clampedProbability);
  }
  if (targetProbability === NEGATIVE_CLASS_LABEL) {
    return -Math.log(1 - clampedProbability);
  }
  return -(
    targetProbability * Math.log(clampedProbability) +
    (1 - targetProbability) * Math.log(1 - clampedProbability)
  );
}

/**
 * Normalizes target probabilities so they sum to 1 when possible.
 *
 * @param targets - Raw target probabilities.
 * @returns Normalized target probabilities; returns a shallow copy when the sum is zero.
 */
function normalizeTargets(targets: number[]): number[] {
  const targetSum = targets.reduce(
    (runningSum, targetValue) => runningSum + targetValue,
    0,
  );
  if (targetSum > 0) {
    return targets.map((targetValue) => targetValue / targetSum);
  }
  return [...targets];
}

/**
 * Computes a numerically stable softmax from raw output scores.
 *
 * @param outputs - Raw logits or scores.
 * @returns Softmax probabilities corresponding to the inputs.
 */
function stableSoftmax(outputs: number[]): number[] {
  const maximumOutput = Math.max(...outputs);
  const shiftedExponentials = outputs.map((outputValue) =>
    Math.exp(outputValue - maximumOutput),
  );
  const exponentialSum = shiftedExponentials.reduce(
    (runningSum, exponentialValue) => runningSum + exponentialValue,
    0,
  );
  const safeDenominator = Math.max(exponentialSum, SOFTMAX_SUM_GUARD);
  return shiftedExponentials.map(
    (exponentialValue) => exponentialValue / safeDenominator,
  );
}

/**
 * Converts a probability into a binary class label using the configured threshold.
 *
 * @param probability - Probability to classify.
 * @returns POSITIVE_CLASS_LABEL when above or equal to threshold; otherwise NEGATIVE_CLASS_LABEL.
 */
function classifyBinary(probability: number): number {
  return probability >= BINARY_CLASSIFICATION_THRESHOLD
    ? POSITIVE_CLASS_LABEL
    : NEGATIVE_CLASS_LABEL;
}

/**
 * Applies label smoothing to a target probability.
 *
 * @param targetProbability - Original target probability.
 * @param smoothing - Smoothing factor between 0 and 1.
 * @returns Smoothed target probability.
 */
function smoothTarget(targetProbability: number, smoothing: number): number {
  const smoothingScale = 1 - smoothing;
  return (
    targetProbability * smoothingScale + LABEL_SMOOTHING_BASELINE * smoothing
  );
}
