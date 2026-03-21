# methods

Provides a collection of standard cost functions (also known as loss functions)
used for evaluating the performance of neural networks during training.

Cost functions quantify the difference between the network's predictions
and the actual target values. The goal of training is typically to minimize
the value of the cost function. The choice of cost function is crucial and
depends on the specific task (e.g., regression, classification) and the
desired behavior of the model.

## methods/cost.ts

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

```ts
cosineAnnealing(
  period: number,
  minimumRate: number,
): (baseRate: number, iteration: number) => number
```

Implements a Cosine Annealing learning rate schedule.

This schedule varies the learning rate cyclically according to a cosine function.
It starts at the `baseRate` and smoothly anneals down to `minimumRate` over a
specified `period` of iterations, then potentially repeats. This can help
the model escape local minima and explore the loss landscape more effectively.
Often used with "warm restarts" where the cycle repeats.

Formula: `learning_rate = minimumRate + 0.5 * (baseRate - minimumRate) * (1 + cos(pi * current_cycle_iteration / period))`

Parameters:
- `period` - The number of iterations over which the learning rate anneals from `baseRate` to `minimumRate` in one cycle. Defaults to 1000.
- `minimumRate` - The minimum learning rate value at the end of a cycle. Defaults to 0.
- `baseRate` - The initial (maximum) learning rate for the cycle.
- `iteration` - The current training iteration.

Returns: A function that calculates the learning rate for a given iteration based on the cosine annealing schedule.

#### cosineAnnealingWarmRestarts

```ts
cosineAnnealingWarmRestarts(
  initialPeriod: number,
  minimumRate: number,
  periodGrowthMultiplier: number,
): (baseRate: number, iteration: number) => number
```

Cosine Annealing with Warm Restarts (SGDR style) where the cycle length can grow by a multiplier after each restart.

Parameters:
- `initialPeriod` - Length of the first cycle in iterations.
- `minimumRate` - Minimum learning rate at valley.
- `periodGrowthMultiplier` - Factor to multiply the period after each restart (>=1).

#### exp

```ts
exp(
  decayFactor: number,
): (baseRate: number, iteration: number) => number
```

Implements an exponential decay learning rate schedule.

The learning rate decreases exponentially after each iteration, multiplying
by the decay factor `decayFactor`. This provides a smooth, continuous reduction
in the learning rate over time.

Formula: `learning_rate = baseRate * decayFactor ^ iteration`

Parameters:
- `decayFactor` - The decay factor applied at each iteration. Should be less than 1. Defaults to 0.999.
- `baseRate` - The initial learning rate.
- `iteration` - The current training iteration.

Returns: A function that calculates the exponentially decayed learning rate for a given iteration.

#### fixed

```ts
fixed(): (baseRate: number, iteration: number) => number
```

Implements a fixed learning rate schedule.

The learning rate remains constant throughout the entire training process.
This is the simplest schedule and serves as a baseline, but may not be
optimal for complex problems.

Parameters:
- `baseRate` - The initial learning rate, which will remain constant.
- `iteration` - The current training iteration (unused in this method, but included for consistency).

Returns: A function that takes the base learning rate and the current iteration number, and always returns the base learning rate.

#### inv

```ts
inv(
  decayFactor: number,
  decayPower: number,
): (baseRate: number, iteration: number) => number
```

Implements an inverse decay learning rate schedule.

The learning rate decreases as the inverse of the iteration number,
controlled by the decay factor `decayFactor` and exponent `decayPower`. The rate
decreases more slowly over time compared to exponential decay.

Formula: `learning_rate = baseRate / (1 + decayFactor * iteration ** decayPower)`

Parameters:
- `decayFactor` - Controls the rate of decay. Higher values lead to faster decay. Defaults to 0.001.
- `decayPower` - The exponent controlling the shape of the decay curve. Defaults to 2.
- `baseRate` - The initial learning rate.
- `iteration` - The current training iteration.

Returns: A function that calculates the inversely decayed learning rate for a given iteration.

#### linearWarmupDecay

```ts
linearWarmupDecay(
  totalStepCount: number,
  warmupStepCount: number | undefined,
  endRate: number,
): (baseRate: number, iteration: number) => number
```

Linear Warmup followed by Linear Decay to an end rate.
Warmup linearly increases LR from near 0 up to baseRate over warmupStepCount, then linearly decays to endRate at totalStepCount.
Iterations beyond totalStepCount clamp to endRate.

Parameters:
- `totalStepCount` - Total steps for full schedule (must be > 0).
- `warmupStepCount` - Steps for warmup (< totalStepCount). Defaults to 10% of totalStepCount.
- `endRate` - Final rate at totalStepCount.

#### reduceOnPlateau

```ts
reduceOnPlateau(
  options: { factor?: number | undefined; patience?: number | undefined; minDelta?: number | undefined; cooldown?: number | undefined; minRate?: number | undefined; verbose?: boolean | undefined; } | undefined,
): (baseRate: number, iteration: number, lastError?: number | undefined) => number
```

ReduceLROnPlateau style scheduler (stateful closure) that monitors error signal (third argument if provided)
and reduces rate by 'factor' if no improvement beyond 'minDelta' for 'patience' iterations.
Cooldown prevents immediate successive reductions.
NOTE: Requires the training loop to call with signature (baseRate, iteration, lastError).

#### step

```ts
step(
  decayFactor: number,
  decayStepSize: number,
): (baseRate: number, iteration: number) => number
```

Implements a step decay learning rate schedule.

The learning rate is reduced by a multiplicative factor (`decayFactor`)
at predefined intervals (`decayStepSize` iterations). This allows for
faster initial learning, followed by finer adjustments as training progresses.

Formula: `learning_rate = baseRate * decayFactor ^ floor(iteration / decayStepSize)`

Parameters:
- `decayFactor` - The factor by which the learning rate is multiplied at each step. Should be less than 1. Defaults to 0.9.
- `decayStepSize` - The number of iterations after which the learning rate decays. Defaults to 100.
- `baseRate` - The initial learning rate.
- `iteration` - The current training iteration.

Returns: A function that calculates the decayed learning rate for a given iteration.

## methods/gating.ts

Defines different methods for gating connections between neurons or groups of neurons.

Gating mechanisms dynamically control the flow of information through connections
in a neural network. This allows the network to selectively route information,
enabling more complex computations, memory functions, and adaptive behaviors.
These mechanisms are inspired by biological neural processes where certain neurons
can modulate the activity of others. Gating is particularly crucial in recurrent
neural networks (RNNs) for managing information persistence over time.

## methods/methods.ts

### methods

Provides various methods for implementing learning rate schedules.

Learning rate schedules dynamically adjust the learning rate during the training
process of machine learning models, particularly neural networks. Adjusting the
learning rate can significantly impact training speed and performance. A high
rate might lead to overshooting the optimal solution, while a very low rate
can result in slow convergence or getting stuck in local minima. These methods
offer different strategies to balance exploration and exploitation during training.

### Activation

Provides a collection of common activation functions used in neural networks.

Activation functions introduce non-linearity into the network, allowing it to
learn complex patterns. They determine the output of a node based on its
weighted inputs and bias. The choice of activation function can significantly
impact the network's performance and training dynamics.

All methods in this class are static and can be called directly, e.g., `Activation.relu(x)`.
Each method accepts an input value `x` and an optional boolean `derivate`.
If `derivate` is true, the method returns the derivative of the activation function
with respect to `x`; otherwise, it returns the activation function's output.

### gating

Defines different methods for gating connections between neurons or groups of neurons.

Gating mechanisms dynamically control the flow of information through connections
in a neural network. This allows the network to selectively route information,
enabling more complex computations, memory functions, and adaptive behaviors.
These mechanisms are inspired by biological neural processes where certain neurons
can modulate the activity of others. Gating is particularly crucial in recurrent
neural networks (RNNs) for managing information persistence over time.

### mutation

Defines various mutation methods used in neuroevolution algorithms.

Mutation introduces genetic diversity into the population by randomly
altering parts of an individual's genome (the neural network structure or parameters).
This is crucial for exploring the search space and escaping local optima.

Common mutation strategies include adding or removing nodes and connections,
modifying connection weights and node biases, and changing node activation functions.
These operations allow the network topology and parameters to adapt over generations.

The methods listed here are inspired by techniques used in algorithms like NEAT
and particularly the Instinct algorithm, providing a comprehensive set of tools
for evolving network architectures.

## Supported Mutation Methods

- `ADD_NODE`: Adds a new node by splitting an existing connection.
- `SUB_NODE`: Removes a hidden node and its connections.
- `ADD_CONN`: Adds a new connection between two unconnected nodes.
- `SUB_CONN`: Removes an existing connection.
- `MOD_WEIGHT`: Modifies the weight of an existing connection.
- `MOD_BIAS`: Modifies the bias of a node.
- `MOD_ACTIVATION`: Changes the activation function of a node.
- `ADD_SELF_CONN`: Adds a self-connection (recurrent loop) to a node.
- `SUB_SELF_CONN`: Removes a self-connection from a node.
- `ADD_GATE`: Adds a gating mechanism to a connection.
- `SUB_GATE`: Removes a gating mechanism from a connection.
- `ADD_BACK_CONN`: Adds a recurrent (backward) connection between nodes.
- `SUB_BACK_CONN`: Removes a recurrent (backward) connection.
- `SWAP_NODES`: Swaps the roles (bias and activation) of two nodes.
- `REINIT_WEIGHT`: Reinitializes all weights for a node.
- `BATCH_NORM`: Marks a node for batch normalization (stub).
- `ADD_LSTM_NODE`: Adds a new LSTM node (memory cell with gates).
- `ADD_GRU_NODE`: Adds a new GRU node (gated recurrent unit).

Also includes:
- `ALL`: Array of all mutation methods.
- `FFW`: Array of mutation methods suitable for feedforward networks.

### selection

Defines various selection methods used in genetic algorithms to choose individuals
for reproduction based on their fitness scores.

Selection is a crucial step that determines which genetic traits are passed on
to the next generation. Different methods offer varying balances between
exploration (maintaining diversity) and exploitation (favoring high-fitness individuals).
The choice of selection method significantly impacts the algorithm's convergence
speed and the diversity of the population. High selection pressure (strongly
favoring the fittest) can lead to faster convergence but may result in premature
stagnation at suboptimal solutions. Conversely, lower pressure maintains diversity
but can slow down the search process.

### crossover

Crossover methods for genetic algorithms.

These methods implement the crossover strategies described in the Instinct algorithm,
enabling the creation of offspring with unique combinations of parent traits.

### groupConnection

Specifies the manner in which two groups of nodes are connected.

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

#### cosineAnnealing

```ts
cosineAnnealing(
  period: number,
  minimumRate: number,
): (baseRate: number, iteration: number) => number
```

Implements a Cosine Annealing learning rate schedule.

This schedule varies the learning rate cyclically according to a cosine function.
It starts at the `baseRate` and smoothly anneals down to `minimumRate` over a
specified `period` of iterations, then potentially repeats. This can help
the model escape local minima and explore the loss landscape more effectively.
Often used with "warm restarts" where the cycle repeats.

Formula: `learning_rate = minimumRate + 0.5 * (baseRate - minimumRate) * (1 + cos(pi * current_cycle_iteration / period))`

Parameters:
- `period` - The number of iterations over which the learning rate anneals from `baseRate` to `minimumRate` in one cycle. Defaults to 1000.
- `minimumRate` - The minimum learning rate value at the end of a cycle. Defaults to 0.
- `baseRate` - The initial (maximum) learning rate for the cycle.
- `iteration` - The current training iteration.

Returns: A function that calculates the learning rate for a given iteration based on the cosine annealing schedule.

#### cosineAnnealingWarmRestarts

```ts
cosineAnnealingWarmRestarts(
  initialPeriod: number,
  minimumRate: number,
  periodGrowthMultiplier: number,
): (baseRate: number, iteration: number) => number
```

Cosine Annealing with Warm Restarts (SGDR style) where the cycle length can grow by a multiplier after each restart.

Parameters:
- `initialPeriod` - Length of the first cycle in iterations.
- `minimumRate` - Minimum learning rate at valley.
- `periodGrowthMultiplier` - Factor to multiply the period after each restart (>=1).

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

#### exp

```ts
exp(
  decayFactor: number,
): (baseRate: number, iteration: number) => number
```

Implements an exponential decay learning rate schedule.

The learning rate decreases exponentially after each iteration, multiplying
by the decay factor `decayFactor`. This provides a smooth, continuous reduction
in the learning rate over time.

Formula: `learning_rate = baseRate * decayFactor ^ iteration`

Parameters:
- `decayFactor` - The decay factor applied at each iteration. Should be less than 1. Defaults to 0.999.
- `baseRate` - The initial learning rate.
- `iteration` - The current training iteration.

Returns: A function that calculates the exponentially decayed learning rate for a given iteration.

#### fixed

```ts
fixed(): (baseRate: number, iteration: number) => number
```

Implements a fixed learning rate schedule.

The learning rate remains constant throughout the entire training process.
This is the simplest schedule and serves as a baseline, but may not be
optimal for complex problems.

Parameters:
- `baseRate` - The initial learning rate, which will remain constant.
- `iteration` - The current training iteration (unused in this method, but included for consistency).

Returns: A function that takes the base learning rate and the current iteration number, and always returns the base learning rate.

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

#### inv

```ts
inv(
  decayFactor: number,
  decayPower: number,
): (baseRate: number, iteration: number) => number
```

Implements an inverse decay learning rate schedule.

The learning rate decreases as the inverse of the iteration number,
controlled by the decay factor `decayFactor` and exponent `decayPower`. The rate
decreases more slowly over time compared to exponential decay.

Formula: `learning_rate = baseRate / (1 + decayFactor * iteration ** decayPower)`

Parameters:
- `decayFactor` - Controls the rate of decay. Higher values lead to faster decay. Defaults to 0.001.
- `decayPower` - The exponent controlling the shape of the decay curve. Defaults to 2.
- `baseRate` - The initial learning rate.
- `iteration` - The current training iteration.

Returns: A function that calculates the inversely decayed learning rate for a given iteration.

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

#### linearWarmupDecay

```ts
linearWarmupDecay(
  totalStepCount: number,
  warmupStepCount: number | undefined,
  endRate: number,
): (baseRate: number, iteration: number) => number
```

Linear Warmup followed by Linear Decay to an end rate.
Warmup linearly increases LR from near 0 up to baseRate over warmupStepCount, then linearly decays to endRate at totalStepCount.
Iterations beyond totalStepCount clamp to endRate.

Parameters:
- `totalStepCount` - Total steps for full schedule (must be > 0).
- `warmupStepCount` - Steps for warmup (< totalStepCount). Defaults to 10% of totalStepCount.
- `endRate` - Final rate at totalStepCount.

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

#### reduceOnPlateau

```ts
reduceOnPlateau(
  options: { factor?: number | undefined; patience?: number | undefined; minDelta?: number | undefined; cooldown?: number | undefined; minRate?: number | undefined; verbose?: boolean | undefined; } | undefined,
): (baseRate: number, iteration: number, lastError?: number | undefined) => number
```

ReduceLROnPlateau style scheduler (stateful closure) that monitors error signal (third argument if provided)
and reduces rate by 'factor' if no improvement beyond 'minDelta' for 'patience' iterations.
Cooldown prevents immediate successive reductions.
NOTE: Requires the training loop to call with signature (baseRate, iteration, lastError).

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

#### step

```ts
step(
  decayFactor: number,
  decayStepSize: number,
): (baseRate: number, iteration: number) => number
```

Implements a step decay learning rate schedule.

The learning rate is reduced by a multiplicative factor (`decayFactor`)
at predefined intervals (`decayStepSize` iterations). This allows for
faster initial learning, followed by finer adjustments as training progresses.

Formula: `learning_rate = baseRate * decayFactor ^ floor(iteration / decayStepSize)`

Parameters:
- `decayFactor` - The factor by which the learning rate is multiplied at each step. Should be less than 1. Defaults to 0.9.
- `decayStepSize` - The number of iterations after which the learning rate decays. Defaults to 100.
- `baseRate` - The initial learning rate.
- `iteration` - The current training iteration.

Returns: A function that calculates the decayed learning rate for a given iteration.

## methods/mutation.ts

### mutation

Defines various mutation methods used in neuroevolution algorithms.

Mutation introduces genetic diversity into the population by randomly
altering parts of an individual's genome (the neural network structure or parameters).
This is crucial for exploring the search space and escaping local optima.

Common mutation strategies include adding or removing nodes and connections,
modifying connection weights and node biases, and changing node activation functions.
These operations allow the network topology and parameters to adapt over generations.

The methods listed here are inspired by techniques used in algorithms like NEAT
and particularly the Instinct algorithm, providing a comprehensive set of tools
for evolving network architectures.

## Supported Mutation Methods

- `ADD_NODE`: Adds a new node by splitting an existing connection.
- `SUB_NODE`: Removes a hidden node and its connections.
- `ADD_CONN`: Adds a new connection between two unconnected nodes.
- `SUB_CONN`: Removes an existing connection.
- `MOD_WEIGHT`: Modifies the weight of an existing connection.
- `MOD_BIAS`: Modifies the bias of a node.
- `MOD_ACTIVATION`: Changes the activation function of a node.
- `ADD_SELF_CONN`: Adds a self-connection (recurrent loop) to a node.
- `SUB_SELF_CONN`: Removes a self-connection from a node.
- `ADD_GATE`: Adds a gating mechanism to a connection.
- `SUB_GATE`: Removes a gating mechanism from a connection.
- `ADD_BACK_CONN`: Adds a recurrent (backward) connection between nodes.
- `SUB_BACK_CONN`: Removes a recurrent (backward) connection.
- `SWAP_NODES`: Swaps the roles (bias and activation) of two nodes.
- `REINIT_WEIGHT`: Reinitializes all weights for a node.
- `BATCH_NORM`: Marks a node for batch normalization (stub).
- `ADD_LSTM_NODE`: Adds a new LSTM node (memory cell with gates).
- `ADD_GRU_NODE`: Adds a new GRU node (gated recurrent unit).

Also includes:
- `ALL`: Array of all mutation methods.
- `FFW`: Array of mutation methods suitable for feedforward networks.

### MutationConfig

Configuration object for a single mutation operation.

## methods/crossover.ts

Crossover methods for genetic algorithms.

These methods implement the crossover strategies described in the Instinct algorithm,
enabling the creation of offspring with unique combinations of parent traits.

## methods/selection.ts

Defines various selection methods used in genetic algorithms to choose individuals
for reproduction based on their fitness scores.

Selection is a crucial step that determines which genetic traits are passed on
to the next generation. Different methods offer varying balances between
exploration (maintaining diversity) and exploitation (favoring high-fitness individuals).
The choice of selection method significantly impacts the algorithm's convergence
speed and the diversity of the population. High selection pressure (strongly
favoring the fittest) can lead to faster convergence but may result in premature
stagnation at suboptimal solutions. Conversely, lower pressure maintains diversity
but can slow down the search process.

## methods/activation.ts

### Activation

Provides a collection of common activation functions used in neural networks.

Activation functions introduce non-linearity into the network, allowing it to
learn complex patterns. They determine the output of a node based on its
weighted inputs and bias. The choice of activation function can significantly
impact the network's performance and training dynamics.

All methods in this class are static and can be called directly, e.g., `Activation.relu(x)`.
Each method accepts an input value `x` and an optional boolean `derivate`.
If `derivate` is true, the method returns the derivative of the activation function
with respect to `x`; otherwise, it returns the activation function's output.

### registerCustomActivation

```ts
registerCustomActivation(
  activationName: string,
  activationFunction: ActivationFunction,
): void
```

Register a custom activation function at runtime.

## methods/connection.ts

Specifies the manner in which two groups of nodes are connected.

### groupConnection

Specifies the manner in which two groups of nodes are connected.

## methods/cost.utils.ts

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

### LENGTH_MISMATCH_MESSAGE

Error message thrown when target and output arrays differ in length.

### POSITIVE_CLASS_LABEL

Canonical positive label used by binary-oriented helpers.

### NEGATIVE_CLASS_LABEL

Canonical negative label used by binary-oriented helpers.

### BINARY_CLASSIFICATION_THRESHOLD

Threshold for binarizing probabilities into class predictions.

### HINGE_MARGIN

Margin enforced by hinge loss.

### DEFAULT_FOCAL_GAMMA

Default focusing parameter for focal loss.

### DEFAULT_FOCAL_ALPHA

Default class balancing parameter for focal loss.

### DEFAULT_LABEL_SMOOTHING

Default smoothing factor for label smoothing.

### LABEL_SMOOTHING_BASELINE

Baseline probability used when smoothing targets.

### SOFTMAX_SUM_GUARD

Lower bound for softmax denominator to avoid division by zero.

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

## methods/rate.utils.ts

Learning rate schedule signature that maps a base rate and iteration index to a rate value.
Useful for any stateless schedule strategy.

### createFixedRateSchedule

```ts
createFixedRateSchedule(): RateSchedule
```

Returns a schedule that always yields the base learning rate.

Returns: A learning rate schedule that ignores iteration and returns baseRate.

### createStepRateSchedule

```ts
createStepRateSchedule(
  decayFactor: number,
  decayStepSize: number,
): RateSchedule
```

Returns a step decay learning rate schedule.

Parameters:
- `decayFactor` - Multiplicative decay applied at each decay step.
- `decayStepSize` - Number of iterations before applying another decay step.

Returns: A learning rate schedule implementing step decay.

### createExponentialRateSchedule

```ts
createExponentialRateSchedule(
  decayFactor: number,
): RateSchedule
```

Returns an exponential decay learning rate schedule.

Parameters:
- `decayFactor` - Multiplicative decay applied every iteration.

Returns: A learning rate schedule implementing exponential decay.

### createInverseRateSchedule

```ts
createInverseRateSchedule(
  decayFactor: number,
  decayPower: number,
): RateSchedule
```

Returns an inverse decay learning rate schedule.

Parameters:
- `decayFactor` - Decay factor controlling the decay rate.
- `decayPower` - Exponent that shapes the decay curve.

Returns: A learning rate schedule implementing inverse decay.

### createCosineAnnealingRateSchedule

```ts
createCosineAnnealingRateSchedule(
  period: number,
  minimumRate: number,
): RateSchedule
```

Returns a cosine annealing learning rate schedule.

Parameters:
- `period` - Length of a full cosine cycle.
- `minimumRate` - Minimum rate reached at the end of a cycle.

Returns: A learning rate schedule implementing cosine annealing.

### createCosineAnnealingWarmRestartsSchedule

```ts
createCosineAnnealingWarmRestartsSchedule(
  initialPeriod: number,
  minimumRate: number,
  periodGrowthMultiplier: number,
): RateSchedule
```

Returns a cosine annealing schedule with warm restarts and growing cycles.

Parameters:
- `initialPeriod` - Length of the initial cycle.
- `minimumRate` - Minimum learning rate reached at the end of each cycle.
- `periodGrowthMultiplier` - Multiplier applied to the period after each restart.

Returns: A learning rate schedule implementing SGDR-style warm restarts.

### createLinearWarmupDecaySchedule

```ts
createLinearWarmupDecaySchedule(
  totalStepCount: number,
  warmupStepCount: number | undefined,
  endRate: number,
): RateSchedule
```

Returns a linear warmup followed by linear decay schedule.

Parameters:
- `totalStepCount` - Total number of steps in the schedule (must be positive).
- `warmupStepCount` - Optional number of warmup steps; defaults to 10% of total steps.
- `endRate` - Final rate once decay completes.

Returns: A learning rate schedule implementing warmup then decay.

### createReduceOnPlateauSchedule

```ts
createReduceOnPlateauSchedule(
  options: { factor?: number | undefined; patience?: number | undefined; minDelta?: number | undefined; cooldown?: number | undefined; minRate?: number | undefined; verbose?: boolean | undefined; } | undefined,
): ReduceOnPlateauSchedule
```

Returns a ReduceLROnPlateau-style schedule that lowers the rate when no improvement is seen.

Parameters:
- `options` - Optional configuration for factor, patience, minDelta, cooldown, and minimum rate.

Returns: A stateful schedule that reacts to lack of improvement.

### RateSchedule

```ts
RateSchedule(
  baseRate: number,
  iteration: number,
): number
```

Learning rate schedule signature that maps a base rate and iteration index to a rate value.
Useful for any stateless schedule strategy.

### ReduceOnPlateauSchedule

```ts
ReduceOnPlateauSchedule(
  baseRate: number,
  iteration: number,
  lastError: number | undefined,
): number
```

Stateful ReduceLROnPlateau schedule signature that can react to a loss signal.
The third argument is optional and only needed when monitoring validation error.

### DEFAULT_STEP_DECAY_FACTOR

Step decay multiplier (close to 1 slows decay; smaller drops faster).

### DEFAULT_DECAY_STEP_SIZE

Step decay interval in iterations; larger values mean fewer decay events.

### DEFAULT_EXPONENTIAL_DECAY_FACTOR

Per-iteration exponential decay factor; values just below 1 create gentle decay.

### DEFAULT_INVERSE_DECAY_FACTOR

Inverse decay multiplier; higher values push the denominator up faster and shrink the rate sooner.

### DEFAULT_INVERSE_POWER

Inverse decay exponent; 1 makes decay linear in iteration, 2 makes it quadratic.

### DEFAULT_COSINE_PERIOD

Length of one cosine annealing cycle in iterations.

### DEFAULT_MINIMUM_RATE

Floor learning rate for cosine schedules; keeps the rate from reaching zero.

### DEFAULT_INITIAL_PERIOD

Initial period length for cosine-with-restarts before growth is applied.

### DEFAULT_PERIOD_GROWTH_MULTIPLIER

Multiplier applied to the cosine cycle length after each restart (>= 1).

### DEFAULT_LINEAR_END_RATE

Target rate after warmup-decay finishes; often zero or a small floor.

### DEFAULT_WARMUP_RATIO

Default warmup share of the schedule; 0.1 means 10% of total steps.

### DEFAULT_REDUCE_ON_PLATEAU_FACTOR

Reduce-on-plateau shrink factor; halving (0.5) is a common conservative step.

### DEFAULT_REDUCE_ON_PLATEAU_PATIENCE

Patience for reduce-on-plateau in iterations before triggering a cut.

### DEFAULT_REDUCE_ON_PLATEAU_MIN_DELTA

Minimum required improvement to count as progress when monitoring error.

### DEFAULT_REDUCE_ON_PLATEAU_COOLDOWN

Cooldown iterations after a reduction to avoid rapid successive cuts.

### DEFAULT_REDUCE_ON_PLATEAU_MIN_RATE

Minimum rate allowed during reduce-on-plateau adjustments.

## methods/activation.utils.ts

Activation function implementation type.

### logisticActivation

```ts
logisticActivation(
  inputValue: number,
  shouldComputeDerivative: boolean,
): number
```

Logistic (sigmoid) activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Logistic output or derivative.

### tanhActivation

```ts
tanhActivation(
  inputValue: number,
  shouldComputeDerivative: boolean,
): number
```

Hyperbolic tangent activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Tanh output or derivative.

### identityActivation

```ts
identityActivation(
  inputValue: number,
  shouldComputeDerivative: boolean,
): number
```

Identity activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Identity output or derivative.

### stepActivation

```ts
stepActivation(
  inputValue: number,
  shouldComputeDerivative: boolean,
): number
```

Step activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Step output or derivative.

### reluActivation

```ts
reluActivation(
  inputValue: number,
  shouldComputeDerivative: boolean,
): number
```

Rectified Linear Unit (ReLU) activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: ReLU output or derivative.

### softsignActivation

```ts
softsignActivation(
  inputValue: number,
  shouldComputeDerivative: boolean,
): number
```

Softsign activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Softsign output or derivative.

### sinusoidActivation

```ts
sinusoidActivation(
  inputValue: number,
  shouldComputeDerivative: boolean,
): number
```

Sinusoid activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Sinusoid output or derivative.

### gaussianActivation

```ts
gaussianActivation(
  inputValue: number,
  shouldComputeDerivative: boolean,
): number
```

Gaussian activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Gaussian output or derivative.

### bentIdentityActivation

```ts
bentIdentityActivation(
  inputValue: number,
  shouldComputeDerivative: boolean,
): number
```

Bent identity activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Bent identity output or derivative.

### bipolarActivation

```ts
bipolarActivation(
  inputValue: number,
  shouldComputeDerivative: boolean,
): number
```

Bipolar activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Bipolar output or derivative.

### bipolarSigmoidActivation

```ts
bipolarSigmoidActivation(
  inputValue: number,
  shouldComputeDerivative: boolean,
): number
```

Bipolar sigmoid activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Bipolar sigmoid output or derivative.

### hardTanhActivation

```ts
hardTanhActivation(
  inputValue: number,
  shouldComputeDerivative: boolean,
): number
```

Hard tanh activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Hard tanh output or derivative.

### absoluteActivation

```ts
absoluteActivation(
  inputValue: number,
  shouldComputeDerivative: boolean,
): number
```

Absolute activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Absolute output or derivative.

### inverseActivation

```ts
inverseActivation(
  inputValue: number,
  shouldComputeDerivative: boolean,
): number
```

Inverse activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Inverse output or derivative.

### seluActivation

```ts
seluActivation(
  inputValue: number,
  shouldComputeDerivative: boolean,
): number
```

Scaled Exponential Linear Unit (SELU) activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: SELU output or derivative.

### softplusActivation

```ts
softplusActivation(
  inputValue: number,
  shouldComputeDerivative: boolean,
): number
```

Softplus activation implementation with stability guards.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Softplus output or derivative.

### swishActivation

```ts
swishActivation(
  inputValue: number,
  shouldComputeDerivative: boolean,
): number
```

Swish activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Swish output or derivative.

### geluActivation

```ts
geluActivation(
  inputValue: number,
  shouldComputeDerivative: boolean,
): number
```

Gaussian Error Linear Unit (GELU) activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: GELU output or derivative.

### mishActivation

```ts
mishActivation(
  inputValue: number,
  shouldComputeDerivative: boolean,
): number
```

Mish activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Mish output or derivative.

### sigmoidActivation

```ts
sigmoidActivation(
  inputValue: number,
  shouldComputeDerivative: boolean,
): number
```

Sigmoid alias activation implementation.

Parameters:
- `inputValue` - - Input to evaluate.
- `shouldComputeDerivative` - - Whether to compute the derivative.

Returns: Sigmoid output or derivative.

### ActivationFunction

```ts
ActivationFunction(
  inputValue: number,
  shouldComputeDerivative: boolean | undefined,
): number
```

Activation function implementation type.

Parameters:
- `inputValue` - - Input to the activation function.
- `shouldComputeDerivative` - - Whether to compute the derivative instead of the value.

Returns: Activation output or derivative at the input.
