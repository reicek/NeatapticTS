# methods

## methods/activation.ts

### Activation

### registerCustomActivation

`(activationName: string, activationFunction: import("C:/NeatapticTS/src/methods/activation.utils").ActivationFunction) => void`

## methods/activation.utils.ts

### absoluteActivation

`(inputValue: number, shouldComputeDerivative: boolean) => number`

Absolute activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Absolute output or derivative.

### ActivationFunction

`(inputValue: number, shouldComputeDerivative: boolean | undefined) => number`

Activation function implementation type.

Parameters:
- `inputValue` - - Input to the activation function.
- `shouldComputeDerivative` - - Whether to compute the derivative instead of the value.

Returns: Activation output or derivative at the input.

### bentIdentityActivation

`(inputValue: number, shouldComputeDerivative: boolean) => number`

Bent identity activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Bent identity output or derivative.

### bipolarActivation

`(inputValue: number, shouldComputeDerivative: boolean) => number`

Bipolar activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Bipolar output or derivative.

### bipolarSigmoidActivation

`(inputValue: number, shouldComputeDerivative: boolean) => number`

Bipolar sigmoid activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Bipolar sigmoid output or derivative.

### gaussianActivation

`(inputValue: number, shouldComputeDerivative: boolean) => number`

Gaussian activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Gaussian output or derivative.

### geluActivation

`(inputValue: number, shouldComputeDerivative: boolean) => number`

Gaussian Error Linear Unit (GELU) activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: GELU output or derivative.

### hardTanhActivation

`(inputValue: number, shouldComputeDerivative: boolean) => number`

Hard tanh activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Hard tanh output or derivative.

### identityActivation

`(inputValue: number, shouldComputeDerivative: boolean) => number`

Identity activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Identity output or derivative.

### inverseActivation

`(inputValue: number, shouldComputeDerivative: boolean) => number`

Inverse activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Inverse output or derivative.

### logisticActivation

`(inputValue: number, shouldComputeDerivative: boolean) => number`

Logistic (sigmoid) activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Logistic output or derivative.

### mishActivation

`(inputValue: number, shouldComputeDerivative: boolean) => number`

Mish activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Mish output or derivative.

### reluActivation

`(inputValue: number, shouldComputeDerivative: boolean) => number`

Rectified Linear Unit (ReLU) activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: ReLU output or derivative.

### seluActivation

`(inputValue: number, shouldComputeDerivative: boolean) => number`

Scaled Exponential Linear Unit (SELU) activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: SELU output or derivative.

### sigmoidActivation

`(inputValue: number, shouldComputeDerivative: boolean) => number`

Sigmoid alias activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Sigmoid output or derivative.

### sinusoidActivation

`(inputValue: number, shouldComputeDerivative: boolean) => number`

Sinusoid activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Sinusoid output or derivative.

### softplusActivation

`(inputValue: number, shouldComputeDerivative: boolean) => number`

Softplus activation implementation with stability guards.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Softplus output or derivative.

### softsignActivation

`(inputValue: number, shouldComputeDerivative: boolean) => number`

Softsign activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Softsign output or derivative.

### stepActivation

`(inputValue: number, shouldComputeDerivative: boolean) => number`

Step activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Step output or derivative.

### swishActivation

`(inputValue: number, shouldComputeDerivative: boolean) => number`

Swish activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Swish output or derivative.

### tanhActivation

`(inputValue: number, shouldComputeDerivative: boolean) => number`

Hyperbolic tangent activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Tanh output or derivative.

## methods/connection.ts

### connection

Export the connection object as the default export.

### groupConnection

## methods/cost.ts

### cost

### default

#### binary

`(targets: number[], outputs: number[]) => number`

Calculates the Binary Error rate, often used as a simple accuracy metric for classification.

This function calculates the proportion of misclassifications by comparing the
rounded network outputs (thresholded at 0.5) against the target labels.
It assumes target values are 0 or 1, and outputs are probabilities between 0 and 1.
Note: This is equivalent to `1 - accuracy` for binary classification.

Parameters:
- `` - - An array of target values, expected to be 0 or 1.
- `` - - An array of output values from the network, typically probabilities between 0 and 1.

Returns: The proportion of misclassified samples (error rate, between 0 and 1).

#### crossEntropy

`(targets: number[], outputs: number[]) => number`

Calculates the Cross Entropy error, commonly used for classification tasks.

This function measures the performance of a classification model whose output is
a probability value between 0 and 1. Cross-entropy loss increases as the
predicted probability diverges from the actual label.

It uses a small epsilon (PROB_EPSILON = 1e-15) to prevent `log(0)` which would result in `NaN`.
Output values are clamped to the range `[epsilon, 1 - epsilon]` for numerical stability.

Parameters:
- `` - - An array of target values, typically 0 or 1 for binary classification, or probabilities for soft labels.
- `` - - An array of output values from the network, representing probabilities (expected to be between 0 and 1).

Returns: The mean cross-entropy error over all samples.

#### focalLoss

`(targets: number[], outputs: number[], focalGamma: number, focalAlpha: number) => number`

Calculates the Focal Loss, which is useful for addressing class imbalance in classification tasks.
Focal loss down-weights easy examples and focuses training on hard negatives.

Parameters:
- `` - - Array of target values (0 or 1 for binary, or probabilities for soft labels).
- `` - - Array of predicted probabilities (between 0 and 1).
- `` - - Focusing parameter (default 2).
- `` - - Balancing parameter (default 0.25).

Returns: The mean focal loss.

#### hinge

`(targets: number[], outputs: number[]) => number`

Calculates the Mean Hinge loss, primarily used for "maximum-margin" classification,
most notably for Support Vector Machines (SVMs).

Hinge loss is used for training classifiers. It penalizes predictions that are
not only incorrect but also those that are correct but not confident (i.e., close to the decision boundary).
Assumes target values are encoded as -1 or 1.

Parameters:
- `` - - An array of target values, expected to be -1 or 1.
- `` - - An array of output values from the network (raw scores, not necessarily probabilities).

Returns: The mean hinge loss.

#### labelSmoothing

`(targets: number[], outputs: number[], smoothingFactor: number) => number`

Calculates the Cross Entropy with Label Smoothing.
Label smoothing prevents the model from becoming overconfident by softening the targets.

Parameters:
- `` - - Array of target values (0 or 1 for binary, or probabilities for soft labels).
- `` - - Array of predicted probabilities (between 0 and 1).
- `` - - Smoothing factor (between 0 and 1, e.g., 0.1).

Returns: The mean cross-entropy loss with label smoothing.

#### mae

`(targets: number[], outputs: number[]) => number`

Calculates the Mean Absolute Error (MAE), another common loss function for regression tasks.

MAE measures the average of the absolute differences between predictions and actual values.
Compared to MSE, it is less sensitive to outliers because errors are not squared.

Parameters:
- `` - - An array of target numerical values.
- `` - - An array of output values from the network.

Returns: The mean absolute error.

#### mape

`(targets: number[], outputs: number[]) => number`

Calculates the Mean Absolute Percentage Error (MAPE).

MAPE expresses the error as a percentage of the actual value. It can be useful
for understanding the error relative to the magnitude of the target values.
However, it has limitations: it's undefined when the target value is zero and
can be skewed by target values close to zero.

Parameters:
- `` - - An array of target numerical values. Should not contain zeros for standard MAPE.
- `` - - An array of output values from the network.

Returns: The mean absolute percentage error, expressed as a proportion (e.g., 0.1 for 10%).

#### mse

`(targets: number[], outputs: number[]) => number`

Calculates the Mean Squared Error (MSE), a common loss function for regression tasks.

MSE measures the average of the squares of the errors—that is, the average
squared difference between the estimated values and the actual value.
It is sensitive to outliers due to the squaring of the error terms.

Parameters:
- `` - - An array of target numerical values.
- `` - - An array of output values from the network.

Returns: The mean squared error.

#### msle

`(targets: number[], outputs: number[]) => number`

Calculates the Mean Squared Logarithmic Error (MSLE).

MSLE is often used in regression tasks where the target values span a large range
or when penalizing under-predictions more than over-predictions is desired.
It measures the squared difference between the logarithms of the predicted and actual values.
Uses `log(1 + x)` instead of `log(x)` for numerical stability and to handle inputs of 0.
Assumes both targets and outputs are non-negative.

Parameters:
- `` - - An array of target numerical values (assumed >= 0).
- `` - - An array of output values from the network (assumed >= 0).

Returns: The mean squared logarithmic error.

#### softmaxCrossEntropy

`(targets: number[], outputs: number[]) => number`

Softmax Cross Entropy for mutually exclusive multi-class outputs given raw (pre-softmax or arbitrary) scores.
Applies a numerically stable softmax to the outputs internally then computes -sum(target * log(prob)).
Targets may be soft labels and are expected to sum to 1 (will be re-normalized if not).

## methods/cost.utils.ts

### BINARY_CLASSIFICATION_THRESHOLD

### clampProbability

`(probability: number) => number`

Clamps a probability into the inclusive bounds defined by PROBABILITY_LOWER_BOUND and PROBABILITY_UPPER_BOUND.

Parameters:
- `probability` - - Raw probability value to bound.

Returns: Probability constrained to the numeric stability range.

### classifyBinary

`(probability: number) => number`

Converts a probability into a binary class label using the configured threshold.

Parameters:
- `probability` - - Probability to classify.

Returns: POSITIVE_CLASS_LABEL when above or equal to threshold; otherwise NEGATIVE_CLASS_LABEL.

### computeBinaryError

`(targets: number[], outputs: number[]) => number`

Computes binary classification error rate.

Parameters:
- `targets` - - Target labels (0 or 1).
- `outputs` - - Predicted probabilities.

Returns: Proportion of misclassified samples.

### computeCrossEntropy

`(targets: number[], outputs: number[]) => number`

Computes the Cross Entropy error over the provided targets and outputs.

Parameters:
- `targets` - - Desired target probabilities (may be soft labels between 0 and 1).
- `outputs` - - Model output probabilities.

Returns: Mean cross-entropy error across all samples.

### computeFocalLoss

`(targets: number[], outputs: number[], gamma: number, alpha: number) => number`

Computes focal loss for imbalanced classification tasks.

Parameters:
- `targets` - - Target labels (0 or 1) or soft labels.
- `outputs` - - Predicted probabilities.
- `gamma` - - Focusing parameter controlling hard example emphasis.
- `alpha` - - Balancing parameter for class weighting.

Returns: Mean focal loss.

### computeHingeLoss

`(targets: number[], outputs: number[]) => number`

Computes hinge loss for margin-based classification.

Parameters:
- `targets` - - Target labels encoded as -1 or 1.
- `outputs` - - Model outputs (raw scores).

Returns: Mean hinge loss.

### computeLabelSmoothingLoss

`(targets: number[], outputs: number[], smoothing: number) => number`

Computes cross entropy with label smoothing applied to targets.

Parameters:
- `targets` - - Target labels (0 or 1) or soft labels.
- `outputs` - - Predicted probabilities.
- `smoothing` - - Smoothing factor between 0 and 1.

Returns: Mean cross-entropy loss with smoothed targets.

### computeMeanAbsoluteError

`(targets: number[], outputs: number[]) => number`

Computes mean absolute error between targets and outputs.

Parameters:
- `targets` - - Desired target values.
- `outputs` - - Model outputs.

Returns: Mean absolute error.

### computeMeanAbsolutePercentageError

`(targets: number[], outputs: number[]) => number`

Computes mean absolute percentage error between targets and outputs.

Parameters:
- `targets` - - Desired target values.
- `outputs` - - Model outputs.

Returns: Mean absolute percentage error (fractional form).

### computeMeanSquaredError

`(targets: number[], outputs: number[]) => number`

Computes mean squared error between targets and outputs.

Parameters:
- `targets` - - Desired target values.
- `outputs` - - Model outputs.

Returns: Mean squared error.

### computeMeanSquaredLogarithmicError

`(targets: number[], outputs: number[]) => number`

Computes mean squared logarithmic error between targets and outputs.

Parameters:
- `targets` - - Desired non-negative target values.
- `outputs` - - Model outputs (expected non-negative).

Returns: Mean squared logarithmic error.

### computeSoftmaxCrossEntropy

`(targets: number[], outputs: number[]) => number`

Computes the softmax cross entropy given targets and raw score outputs.

Parameters:
- `targets` - - Desired target probabilities that should sum to 1 (will be normalized if not).
- `outputs` - - Raw logits or scores for each class.

Returns: Total (non-averaged) softmax cross-entropy loss.

### crossEntropyTerm

`(targetProbability: number, clampedProbability: number) => number`

Computes the cross-entropy contribution for a single target/output pair.

Parameters:
- `targetProbability` - - Target probability for the sample (may be soft).
- `clampedProbability` - - Output probability already clamped for stability.

Returns: Cross-entropy term for the sample.

### DEFAULT_FOCAL_ALPHA

### DEFAULT_FOCAL_GAMMA

### DEFAULT_LABEL_SMOOTHING

### HINGE_MARGIN

### LABEL_SMOOTHING_BASELINE

### LENGTH_MISMATCH_MESSAGE

### NEGATIVE_CLASS_LABEL

### normalizeTargets

`(targets: number[]) => number[]`

Normalizes target probabilities so they sum to 1 when possible.

Parameters:
- `targets` - - Raw target probabilities.

Returns: Normalized target probabilities; returns a shallow copy when the sum is zero.

### POSITIVE_CLASS_LABEL

### smoothTarget

`(targetProbability: number, smoothing: number) => number`

Applies label smoothing to a target probability.

Parameters:
- `targetProbability` - - Original target probability.
- `smoothing` - - Smoothing factor between 0 and 1.

Returns: Smoothed target probability.

### SOFTMAX_SUM_GUARD

### stableSoftmax

`(outputs: number[]) => number[]`

Computes a numerically stable softmax from raw output scores.

Parameters:
- `outputs` - - Raw logits or scores.

Returns: Softmax probabilities corresponding to the inputs.

## methods/crossover.ts

### crossover

## methods/gating.ts

### gating

## methods/methods.ts

### Activation

### crossover

### gating

### groupConnection

### methods

Provides various methods for implementing learning rate schedules.

Learning rate schedules dynamically adjust the learning rate during the training
process of machine learning models, particularly neural networks. Adjusting the
learning rate can significantly impact training speed and performance. A high
rate might lead to overshooting the optimal solution, while a very low rate
can result in slow convergence or getting stuck in local minima. These methods
offer different strategies to balance exploration and exploitation during training.

### mutation

### selection

### default

#### binary

`(targets: number[], outputs: number[]) => number`

Calculates the Binary Error rate, often used as a simple accuracy metric for classification.

This function calculates the proportion of misclassifications by comparing the
rounded network outputs (thresholded at 0.5) against the target labels.
It assumes target values are 0 or 1, and outputs are probabilities between 0 and 1.
Note: This is equivalent to `1 - accuracy` for binary classification.

Parameters:
- `` - - An array of target values, expected to be 0 or 1.
- `` - - An array of output values from the network, typically probabilities between 0 and 1.

Returns: The proportion of misclassified samples (error rate, between 0 and 1).

#### cosineAnnealing

`(period: number, minRate: number) => (baseRate: number, iteration: number) => number`

Implements a Cosine Annealing learning rate schedule.

This schedule varies the learning rate cyclically according to a cosine function.
It starts at the `baseRate` and smoothly anneals down to `minRate` over a
specified `period` of iterations, then potentially repeats. This can help
the model escape local minima and explore the loss landscape more effectively.
Often used with "warm restarts" where the cycle repeats.

Formula: `learning_rate = minRate + 0.5 * (baseRate - minRate) * (1 + cos(pi * current_cycle_iteration / period))`

Parameters:
- `period` - The number of iterations over which the learning rate anneals from `baseRate` to `minRate` in one cycle. Defaults to 1000.
- `minRate` - The minimum learning rate value at the end of a cycle. Defaults to 0.
- `baseRate` - The initial (maximum) learning rate for the cycle.
- `iteration` - The current training iteration.

Returns: A function that calculates the learning rate for a given iteration based on the cosine annealing schedule.

#### cosineAnnealingWarmRestarts

`(initialPeriod: number, minRate: number, tMult: number) => (baseRate: number, iteration: number) => number`

Cosine Annealing with Warm Restarts (SGDR style) where the cycle length can grow by a multiplier (tMult) after each restart.

Parameters:
- `initialPeriod` - Length of the first cycle in iterations.
- `minRate` - Minimum learning rate at valley.
- `tMult` - Factor to multiply the period after each restart (>=1).

#### crossEntropy

`(targets: number[], outputs: number[]) => number`

Calculates the Cross Entropy error, commonly used for classification tasks.

This function measures the performance of a classification model whose output is
a probability value between 0 and 1. Cross-entropy loss increases as the
predicted probability diverges from the actual label.

It uses a small epsilon (PROB_EPSILON = 1e-15) to prevent `log(0)` which would result in `NaN`.
Output values are clamped to the range `[epsilon, 1 - epsilon]` for numerical stability.

Parameters:
- `` - - An array of target values, typically 0 or 1 for binary classification, or probabilities for soft labels.
- `` - - An array of output values from the network, representing probabilities (expected to be between 0 and 1).

Returns: The mean cross-entropy error over all samples.

#### exp

`(gamma: number) => (baseRate: number, iteration: number) => number`

Implements an exponential decay learning rate schedule.

The learning rate decreases exponentially after each iteration, multiplying
by the decay factor `gamma`. This provides a smooth, continuous reduction
in the learning rate over time.

Formula: `learning_rate = baseRate * gamma ^ iteration`

Parameters:
- `gamma` - The decay factor applied at each iteration. Should be less than 1. Defaults to 0.999.
- `baseRate` - The initial learning rate.
- `iteration` - The current training iteration.

Returns: A function that calculates the exponentially decayed learning rate for a given iteration.

#### fixed

`() => (baseRate: number, iteration: number) => number`

Implements a fixed learning rate schedule.

The learning rate remains constant throughout the entire training process.
This is the simplest schedule and serves as a baseline, but may not be
optimal for complex problems.

Parameters:
- `baseRate` - The initial learning rate, which will remain constant.
- `iteration` - The current training iteration (unused in this method, but included for consistency).

Returns: A function that takes the base learning rate and the current iteration number, and always returns the base learning rate.

#### focalLoss

`(targets: number[], outputs: number[], focalGamma: number, focalAlpha: number) => number`

Calculates the Focal Loss, which is useful for addressing class imbalance in classification tasks.
Focal loss down-weights easy examples and focuses training on hard negatives.

Parameters:
- `` - - Array of target values (0 or 1 for binary, or probabilities for soft labels).
- `` - - Array of predicted probabilities (between 0 and 1).
- `` - - Focusing parameter (default 2).
- `` - - Balancing parameter (default 0.25).

Returns: The mean focal loss.

#### hinge

`(targets: number[], outputs: number[]) => number`

Calculates the Mean Hinge loss, primarily used for "maximum-margin" classification,
most notably for Support Vector Machines (SVMs).

Hinge loss is used for training classifiers. It penalizes predictions that are
not only incorrect but also those that are correct but not confident (i.e., close to the decision boundary).
Assumes target values are encoded as -1 or 1.

Parameters:
- `` - - An array of target values, expected to be -1 or 1.
- `` - - An array of output values from the network (raw scores, not necessarily probabilities).

Returns: The mean hinge loss.

#### inv

`(gamma: number, power: number) => (baseRate: number, iteration: number) => number`

Implements an inverse decay learning rate schedule.

The learning rate decreases as the inverse of the iteration number,
controlled by the decay factor `gamma` and exponent `power`. The rate
decreases more slowly over time compared to exponential decay.

Formula: `learning_rate = baseRate / (1 + gamma * Math.pow(iteration, power))`

Parameters:
- `gamma` - Controls the rate of decay. Higher values lead to faster decay. Defaults to 0.001.
- `power` - The exponent controlling the shape of the decay curve. Defaults to 2.
- `baseRate` - The initial learning rate.
- `iteration` - The current training iteration.

Returns: A function that calculates the inversely decayed learning rate for a given iteration.

#### labelSmoothing

`(targets: number[], outputs: number[], smoothingFactor: number) => number`

Calculates the Cross Entropy with Label Smoothing.
Label smoothing prevents the model from becoming overconfident by softening the targets.

Parameters:
- `` - - Array of target values (0 or 1 for binary, or probabilities for soft labels).
- `` - - Array of predicted probabilities (between 0 and 1).
- `` - - Smoothing factor (between 0 and 1, e.g., 0.1).

Returns: The mean cross-entropy loss with label smoothing.

#### linearWarmupDecay

`(totalSteps: number, warmupSteps: number | undefined, endRate: number) => (baseRate: number, iteration: number) => number`

Linear Warmup followed by Linear Decay to an end rate.
Warmup linearly increases LR from near 0 up to baseRate over warmupSteps, then linearly decays to endRate at totalSteps.
Iterations beyond totalSteps clamp to endRate.

Parameters:
- `totalSteps` - Total steps for full schedule (must be > 0).
- `warmupSteps` - Steps for warmup (< totalSteps). Defaults to 10% of totalSteps.
- `endRate` - Final rate at totalSteps.

#### mae

`(targets: number[], outputs: number[]) => number`

Calculates the Mean Absolute Error (MAE), another common loss function for regression tasks.

MAE measures the average of the absolute differences between predictions and actual values.
Compared to MSE, it is less sensitive to outliers because errors are not squared.

Parameters:
- `` - - An array of target numerical values.
- `` - - An array of output values from the network.

Returns: The mean absolute error.

#### mape

`(targets: number[], outputs: number[]) => number`

Calculates the Mean Absolute Percentage Error (MAPE).

MAPE expresses the error as a percentage of the actual value. It can be useful
for understanding the error relative to the magnitude of the target values.
However, it has limitations: it's undefined when the target value is zero and
can be skewed by target values close to zero.

Parameters:
- `` - - An array of target numerical values. Should not contain zeros for standard MAPE.
- `` - - An array of output values from the network.

Returns: The mean absolute percentage error, expressed as a proportion (e.g., 0.1 for 10%).

#### mse

`(targets: number[], outputs: number[]) => number`

Calculates the Mean Squared Error (MSE), a common loss function for regression tasks.

MSE measures the average of the squares of the errors—that is, the average
squared difference between the estimated values and the actual value.
It is sensitive to outliers due to the squaring of the error terms.

Parameters:
- `` - - An array of target numerical values.
- `` - - An array of output values from the network.

Returns: The mean squared error.

#### msle

`(targets: number[], outputs: number[]) => number`

Calculates the Mean Squared Logarithmic Error (MSLE).

MSLE is often used in regression tasks where the target values span a large range
or when penalizing under-predictions more than over-predictions is desired.
It measures the squared difference between the logarithms of the predicted and actual values.
Uses `log(1 + x)` instead of `log(x)` for numerical stability and to handle inputs of 0.
Assumes both targets and outputs are non-negative.

Parameters:
- `` - - An array of target numerical values (assumed >= 0).
- `` - - An array of output values from the network (assumed >= 0).

Returns: The mean squared logarithmic error.

#### reduceOnPlateau

`(options: { factor?: number | undefined; patience?: number | undefined; minDelta?: number | undefined; cooldown?: number | undefined; minRate?: number | undefined; verbose?: boolean | undefined; } | undefined) => (baseRate: number, iteration: number, lastError?: number | undefined) => number`

ReduceLROnPlateau style scheduler (stateful closure) that monitors error signal (third argument if provided)
and reduces rate by 'factor' if no improvement beyond 'minDelta' for 'patience' iterations.
Cooldown prevents immediate successive reductions.
NOTE: Requires the training loop to call with signature (baseRate, iteration, lastError).

#### softmaxCrossEntropy

`(targets: number[], outputs: number[]) => number`

Softmax Cross Entropy for mutually exclusive multi-class outputs given raw (pre-softmax or arbitrary) scores.
Applies a numerically stable softmax to the outputs internally then computes -sum(target * log(prob)).
Targets may be soft labels and are expected to sum to 1 (will be re-normalized if not).

#### step

`(gamma: number, stepSize: number) => (baseRate: number, iteration: number) => number`

Implements a step decay learning rate schedule.

The learning rate is reduced by a multiplicative factor (`gamma`)
at predefined intervals (`stepSize` iterations). This allows for
faster initial learning, followed by finer adjustments as training progresses.

Formula: `learning_rate = baseRate * gamma ^ floor(iteration / stepSize)`

Parameters:
- `gamma` - The factor by which the learning rate is multiplied at each step. Should be less than 1. Defaults to 0.9.
- `stepSize` - The number of iterations after which the learning rate decays. Defaults to 100.
- `baseRate` - The initial learning rate.
- `iteration` - The current training iteration.

Returns: A function that calculates the decayed learning rate for a given iteration.

## methods/mutation.ts

### mutation

### MutationConfig

Configuration object for a single mutation operation.

## methods/rate.ts

### rate

Provides various methods for implementing learning rate schedules.

Learning rate schedules dynamically adjust the learning rate during the training
process of machine learning models, particularly neural networks. Adjusting the
learning rate can significantly impact training speed and performance. A high
rate might lead to overshooting the optimal solution, while a very low rate
can result in slow convergence or getting stuck in local minima. These methods
offer different strategies to balance exploration and exploitation during training.

### Rate

Provides various methods for implementing learning rate schedules.

Learning rate schedules dynamically adjust the learning rate during the training
process of machine learning models, particularly neural networks. Adjusting the
learning rate can significantly impact training speed and performance. A high
rate might lead to overshooting the optimal solution, while a very low rate
can result in slow convergence or getting stuck in local minima. These methods
offer different strategies to balance exploration and exploitation during training.

### default

#### cosineAnnealing

`(period: number, minRate: number) => (baseRate: number, iteration: number) => number`

Implements a Cosine Annealing learning rate schedule.

This schedule varies the learning rate cyclically according to a cosine function.
It starts at the `baseRate` and smoothly anneals down to `minRate` over a
specified `period` of iterations, then potentially repeats. This can help
the model escape local minima and explore the loss landscape more effectively.
Often used with "warm restarts" where the cycle repeats.

Formula: `learning_rate = minRate + 0.5 * (baseRate - minRate) * (1 + cos(pi * current_cycle_iteration / period))`

Parameters:
- `period` - The number of iterations over which the learning rate anneals from `baseRate` to `minRate` in one cycle. Defaults to 1000.
- `minRate` - The minimum learning rate value at the end of a cycle. Defaults to 0.
- `baseRate` - The initial (maximum) learning rate for the cycle.
- `iteration` - The current training iteration.

Returns: A function that calculates the learning rate for a given iteration based on the cosine annealing schedule.

#### cosineAnnealingWarmRestarts

`(initialPeriod: number, minRate: number, tMult: number) => (baseRate: number, iteration: number) => number`

Cosine Annealing with Warm Restarts (SGDR style) where the cycle length can grow by a multiplier (tMult) after each restart.

Parameters:
- `initialPeriod` - Length of the first cycle in iterations.
- `minRate` - Minimum learning rate at valley.
- `tMult` - Factor to multiply the period after each restart (>=1).

#### exp

`(gamma: number) => (baseRate: number, iteration: number) => number`

Implements an exponential decay learning rate schedule.

The learning rate decreases exponentially after each iteration, multiplying
by the decay factor `gamma`. This provides a smooth, continuous reduction
in the learning rate over time.

Formula: `learning_rate = baseRate * gamma ^ iteration`

Parameters:
- `gamma` - The decay factor applied at each iteration. Should be less than 1. Defaults to 0.999.
- `baseRate` - The initial learning rate.
- `iteration` - The current training iteration.

Returns: A function that calculates the exponentially decayed learning rate for a given iteration.

#### fixed

`() => (baseRate: number, iteration: number) => number`

Implements a fixed learning rate schedule.

The learning rate remains constant throughout the entire training process.
This is the simplest schedule and serves as a baseline, but may not be
optimal for complex problems.

Parameters:
- `baseRate` - The initial learning rate, which will remain constant.
- `iteration` - The current training iteration (unused in this method, but included for consistency).

Returns: A function that takes the base learning rate and the current iteration number, and always returns the base learning rate.

#### inv

`(gamma: number, power: number) => (baseRate: number, iteration: number) => number`

Implements an inverse decay learning rate schedule.

The learning rate decreases as the inverse of the iteration number,
controlled by the decay factor `gamma` and exponent `power`. The rate
decreases more slowly over time compared to exponential decay.

Formula: `learning_rate = baseRate / (1 + gamma * Math.pow(iteration, power))`

Parameters:
- `gamma` - Controls the rate of decay. Higher values lead to faster decay. Defaults to 0.001.
- `power` - The exponent controlling the shape of the decay curve. Defaults to 2.
- `baseRate` - The initial learning rate.
- `iteration` - The current training iteration.

Returns: A function that calculates the inversely decayed learning rate for a given iteration.

#### linearWarmupDecay

`(totalSteps: number, warmupSteps: number | undefined, endRate: number) => (baseRate: number, iteration: number) => number`

Linear Warmup followed by Linear Decay to an end rate.
Warmup linearly increases LR from near 0 up to baseRate over warmupSteps, then linearly decays to endRate at totalSteps.
Iterations beyond totalSteps clamp to endRate.

Parameters:
- `totalSteps` - Total steps for full schedule (must be > 0).
- `warmupSteps` - Steps for warmup (< totalSteps). Defaults to 10% of totalSteps.
- `endRate` - Final rate at totalSteps.

#### reduceOnPlateau

`(options: { factor?: number | undefined; patience?: number | undefined; minDelta?: number | undefined; cooldown?: number | undefined; minRate?: number | undefined; verbose?: boolean | undefined; } | undefined) => (baseRate: number, iteration: number, lastError?: number | undefined) => number`

ReduceLROnPlateau style scheduler (stateful closure) that monitors error signal (third argument if provided)
and reduces rate by 'factor' if no improvement beyond 'minDelta' for 'patience' iterations.
Cooldown prevents immediate successive reductions.
NOTE: Requires the training loop to call with signature (baseRate, iteration, lastError).

#### step

`(gamma: number, stepSize: number) => (baseRate: number, iteration: number) => number`

Implements a step decay learning rate schedule.

The learning rate is reduced by a multiplicative factor (`gamma`)
at predefined intervals (`stepSize` iterations). This allows for
faster initial learning, followed by finer adjustments as training progresses.

Formula: `learning_rate = baseRate * gamma ^ floor(iteration / stepSize)`

Parameters:
- `gamma` - The factor by which the learning rate is multiplied at each step. Should be less than 1. Defaults to 0.9.
- `stepSize` - The number of iterations after which the learning rate decays. Defaults to 100.
- `baseRate` - The initial learning rate.
- `iteration` - The current training iteration.

Returns: A function that calculates the decayed learning rate for a given iteration.

## methods/selection.ts

### selection
