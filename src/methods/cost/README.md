# methods/cost

Provides a collection of standard cost functions (also known as loss functions)
used for evaluating the performance of neural networks during training.

Cost functions quantify the difference between the network's predictions
and the actual target values. The goal of training is typically to minimize
the value of the cost function. The choice of cost function is crucial and
depends on the specific task (e.g., regression, classification) and the
desired behavior of the model.

Read this chapter as an answer to one practical modeling question: what kind
of mistake do you want the network to care about most?

The methods cluster into three teaching-friendly families:

- classification losses such as `crossEntropy()`, `softmaxCrossEntropy()`,
  `binary()`, and `hinge()` care about confidence, separability, or error
  rate,
- regression losses such as `mse()`, `mae()`, `mape()`, and `msle()` care
  about scale, outliers, or percentage error,
- calibration helpers such as `focalLoss()` and `labelSmoothing()` change how
  harshly easy examples or overconfident predictions should be treated.

## methods/cost/cost.ts

### default

#### binary

```ts
binary(
  targets: number[],
  outputs: number[],
): number
```

Calculates the Binary Error rate, often used as a simple accuracy metric for classification.

This function calculates the proportion of misclassifications by comparing the
rounded network outputs (thresholded at 0.5) against the target labels.
It assumes target values are 0 or 1, and outputs are probabilities between 0 and 1.
Note: This is equivalent to `1 - accuracy` for binary classification.

Returns: The proportion of misclassified samples (error rate, between 0 and 1).

#### crossEntropy

```ts
crossEntropy(
  targets: number[],
  outputs: number[],
): number
```

Calculates the Cross Entropy error, commonly used for classification tasks.

This function measures the performance of a classification model whose output is
a probability value between 0 and 1. Cross-entropy loss increases as the
predicted probability diverges from the actual label.

It uses a small epsilon (PROB_EPSILON = 1e-15) to prevent `log(0)` which would result in `NaN`.
Output values are clamped to the range `[epsilon, 1 - epsilon]` for numerical stability.

Returns: The mean cross-entropy error over all samples.

#### focalLoss

```ts
focalLoss(
  targets: number[],
  outputs: number[],
  focalGamma: number,
  focalAlpha: number,
): number
```

Calculates the Focal Loss, which is useful for addressing class imbalance in classification tasks.
Focal loss down-weights easy examples and focuses training on hard negatives.

Returns: The mean focal loss.

#### hinge

```ts
hinge(
  targets: number[],
  outputs: number[],
): number
```

Calculates the Mean Hinge loss, primarily used for "maximum-margin" classification,
most notably for Support Vector Machines (SVMs).

Hinge loss is used for training classifiers. It penalizes predictions that are
not only incorrect but also those that are correct but not confident (i.e., close to the decision boundary).
Assumes target values are encoded as -1 or 1.

Returns: The mean hinge loss.

#### labelSmoothing

```ts
labelSmoothing(
  targets: number[],
  outputs: number[],
  smoothingFactor: number,
): number
```

Calculates the Cross Entropy with Label Smoothing.
Label smoothing prevents the model from becoming overconfident by softening the targets.

Returns: The mean cross-entropy loss with label smoothing.

#### mae

```ts
mae(
  targets: number[],
  outputs: number[],
): number
```

Calculates the Mean Absolute Error (MAE), another common loss function for regression tasks.

MAE measures the average of the absolute differences between predictions and actual values.
Compared to MSE, it is less sensitive to outliers because errors are not squared.

Returns: The mean absolute error.

#### mape

```ts
mape(
  targets: number[],
  outputs: number[],
): number
```

Calculates the Mean Absolute Percentage Error (MAPE).

MAPE expresses the error as a percentage of the actual value. It can be useful
for understanding the error relative to the magnitude of the target values.
However, it has limitations: it's undefined when the target value is zero and
can be skewed by target values close to zero.

Returns: The mean absolute percentage error, expressed as a proportion (e.g., 0.1 for 10%).

#### mse

```ts
mse(
  targets: number[],
  outputs: number[],
): number
```

Calculates the Mean Squared Error (MSE), a common loss function for regression tasks.

MSE measures the average of the squares of the errors—that is, the average
squared difference between the estimated values and the actual value.
It is sensitive to outliers due to the squaring of the error terms.

Returns: The mean squared error.

#### msle

```ts
msle(
  targets: number[],
  outputs: number[],
): number
```

Calculates the Mean Squared Logarithmic Error (MSLE).

MSLE is often used in regression tasks where the target values span a large range
or when penalizing under-predictions more than over-predictions is desired.
It measures the squared difference between the logarithms of the predicted and actual values.
Uses `log(1 + x)` instead of `log(x)` for numerical stability and to handle inputs of 0.
Assumes both targets and outputs are non-negative.

Returns: The mean squared logarithmic error.

#### softmaxCrossEntropy

```ts
softmaxCrossEntropy(
  targets: number[],
  outputs: number[],
): number
```

Softmax Cross Entropy for mutually exclusive multi-class outputs given raw (pre-softmax or arbitrary) scores.
Applies a numerically stable softmax to the outputs internally then computes -sum(target * log(prob)).
Targets may be soft labels and are expected to sum to 1 (will be re-normalized if not).

## methods/cost/cost.utils.ts

### BINARY_CLASSIFICATION_THRESHOLD

Threshold for binarizing probabilities into class predictions.

### clampProbability

```ts
clampProbability(
  probability: number,
): number
```

Clamps a probability into the inclusive bounds defined by PROBABILITY_LOWER_BOUND and PROBABILITY_UPPER_BOUND.

Parameters:
- `probability` - - Raw probability value to bound.

Returns: Probability constrained to the numeric stability range.

### classifyBinary

```ts
classifyBinary(
  probability: number,
): number
```

Converts a probability into a binary class label using the configured threshold.

Parameters:
- `probability` - - Probability to classify.

Returns: POSITIVE_CLASS_LABEL when above or equal to threshold; otherwise NEGATIVE_CLASS_LABEL.

### computeBinaryError

```ts
computeBinaryError(
  targets: number[],
  outputs: number[],
): number
```

Computes binary classification error rate.

Parameters:
- `targets` - - Target labels (0 or 1).
- `outputs` - - Predicted probabilities.

Returns: Proportion of misclassified samples.

### computeCrossEntropy

```ts
computeCrossEntropy(
  targets: number[],
  outputs: number[],
): number
```

Computes the Cross Entropy error over the provided targets and outputs.

Parameters:
- `targets` - - Desired target probabilities (may be soft labels between 0 and 1).
- `outputs` - - Model output probabilities.

Returns: Mean cross-entropy error across all samples.

### computeFocalLoss

```ts
computeFocalLoss(
  targets: number[],
  outputs: number[],
  gamma: number,
  alpha: number,
): number
```

Computes focal loss for imbalanced classification tasks.

Parameters:
- `targets` - - Target labels (0 or 1) or soft labels.
- `outputs` - - Predicted probabilities.
- `gamma` - - Focusing parameter controlling hard example emphasis.
- `alpha` - - Balancing parameter for class weighting.

Returns: Mean focal loss.

### computeHingeLoss

```ts
computeHingeLoss(
  targets: number[],
  outputs: number[],
): number
```

Computes hinge loss for margin-based classification.

Parameters:
- `targets` - - Target labels encoded as -1 or 1.
- `outputs` - - Model outputs (raw scores).

Returns: Mean hinge loss.

### computeLabelSmoothingLoss

```ts
computeLabelSmoothingLoss(
  targets: number[],
  outputs: number[],
  smoothing: number,
): number
```

Computes cross entropy with label smoothing applied to targets.

Parameters:
- `targets` - - Target labels (0 or 1) or soft labels.
- `outputs` - - Predicted probabilities.
- `smoothing` - - Smoothing factor between 0 and 1.

Returns: Mean cross-entropy loss with smoothed targets.

### computeMeanAbsoluteError

```ts
computeMeanAbsoluteError(
  targets: number[],
  outputs: number[],
): number
```

Computes mean absolute error between targets and outputs.

Parameters:
- `targets` - - Desired target values.
- `outputs` - - Model outputs.

Returns: Mean absolute error.

### computeMeanAbsolutePercentageError

```ts
computeMeanAbsolutePercentageError(
  targets: number[],
  outputs: number[],
): number
```

Computes mean absolute percentage error between targets and outputs.

Parameters:
- `targets` - - Desired target values.
- `outputs` - - Model outputs.

Returns: Mean absolute percentage error (fractional form).

### computeMeanSquaredError

```ts
computeMeanSquaredError(
  targets: number[],
  outputs: number[],
): number
```

Computes mean squared error between targets and outputs.

Parameters:
- `targets` - - Desired target values.
- `outputs` - - Model outputs.

Returns: Mean squared error.

### computeMeanSquaredLogarithmicError

```ts
computeMeanSquaredLogarithmicError(
  targets: number[],
  outputs: number[],
): number
```

Computes mean squared logarithmic error between targets and outputs.

Parameters:
- `targets` - - Desired non-negative target values.
- `outputs` - - Model outputs (expected non-negative).

Returns: Mean squared logarithmic error.

### computeSoftmaxCrossEntropy

```ts
computeSoftmaxCrossEntropy(
  targets: number[],
  outputs: number[],
): number
```

Computes the softmax cross entropy given targets and raw score outputs.

Parameters:
- `targets` - - Desired target probabilities that should sum to 1 (will be normalized if not).
- `outputs` - - Raw logits or scores for each class.

Returns: Total (non-averaged) softmax cross-entropy loss.

### crossEntropyTerm

```ts
crossEntropyTerm(
  targetProbability: number,
  clampedProbability: number,
): number
```

Computes the cross-entropy contribution for a single target/output pair.

Parameters:
- `targetProbability` - - Target probability for the sample (may be soft).
- `clampedProbability` - - Output probability already clamped for stability.

Returns: Cross-entropy term for the sample.

### DEFAULT_FOCAL_ALPHA

Default class balancing parameter for focal loss.

### DEFAULT_FOCAL_GAMMA

Default focusing parameter for focal loss.

### DEFAULT_LABEL_SMOOTHING

Default smoothing factor for label smoothing.

### HINGE_MARGIN

Margin enforced by hinge loss.

### LABEL_SMOOTHING_BASELINE

Baseline probability used when smoothing targets.

### LENGTH_MISMATCH_MESSAGE

Error message thrown when target and output arrays differ in length.

### NEGATIVE_CLASS_LABEL

Canonical negative label used by binary-oriented helpers.

### normalizeTargets

```ts
normalizeTargets(
  targets: number[],
): number[]
```

Normalizes target probabilities so they sum to 1 when possible.

Parameters:
- `targets` - - Raw target probabilities.

Returns: Normalized target probabilities; returns a shallow copy when the sum is zero.

### POSITIVE_CLASS_LABEL

Canonical positive label used by binary-oriented helpers.

### smoothTarget

```ts
smoothTarget(
  targetProbability: number,
  smoothing: number,
): number
```

Applies label smoothing to a target probability.

Parameters:
- `targetProbability` - - Original target probability.
- `smoothing` - - Smoothing factor between 0 and 1.

Returns: Smoothed target probability.

### SOFTMAX_SUM_GUARD

Lower bound for softmax denominator to avoid division by zero.

### stableSoftmax

```ts
stableSoftmax(
  outputs: number[],
): number[]
```

Computes a numerically stable softmax from raw output scores.

Parameters:
- `outputs` - - Raw logits or scores.

Returns: Softmax probabilities corresponding to the inputs.

## methods/cost/cost.errors.ts

Raised when cost helpers receive target and output arrays of different lengths.

### CostTargetOutputLengthMismatchError

Raised when cost helpers receive target and output arrays of different lengths.
