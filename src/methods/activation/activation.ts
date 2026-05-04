/**
 * Runtime registry of built-in and custom activation functions.
 *
 * ## Why Activation Functions Matter
 *
 * Without a non-linear activation at each neuron, a network of any depth
 * collapses to a single affine transformation — it could be replaced by one
 * layer. Activation functions are the source of *representational power*: they
 * let stacked layers compose non-linear features that no linear model can
 * capture.
 *
 * The theoretical guarantee behind this is the **Universal Approximation
 * Theorem**, which establishes that a network with at least one hidden layer
 * using a non-polynomial activation function can approximate any continuous
 * function on a compact domain to arbitrary precision, given enough hidden
 * units. See Wikipedia contributors,
 * [Universal approximation theorem](https://en.wikipedia.org/wiki/Universal_approximation_theorem),
 * for the formal statement and its practical implications.
 *
 * ## The Function Interface
 *
 * Every activation in this registry shares the same calling convention:
 *
 * ```
 * f(x)          → forward pass value
 * f(x, true)    → local derivative  f'(x)
 * ```
 *
 * The derivative mode supports gradient-based training (backpropagation).
 * In NEAT evolutionary runs, the derivative is not required for the forward
 * activation pass, but it is necessary when the network is trained with
 * gradient descent rather than evolved.
 *
 * ## Function Families
 *
 * The built-in functions cluster into four families:
 *
 * - **Saturating classics** — `logistic`, `sigmoid`, `tanh`: outputs bounded,
 *   easy to reason about; historically dominant but prone to vanishing
 *   gradients in deep networks,
 * - **Piecewise linear** — `relu`, `hardTanh`, `step`: cheap to evaluate,
 *   sparse activations, strong gating behavior; `relu` is the default hidden-
 *   layer choice for most modern work,
 * - **Shape-specialized** — `gaussian`, `sinusoid`, `bentIdentity`: useful
 *   when periodic, radial, or near-linear responses are beneficial,
 * - **Smooth modern** — `softplus`, `swish`, `gelu`, `mish`: differentiable
 *   everywhere and empirically strong across many architectures.
 *
 * See Wikipedia contributors,
 * [Activation function](https://en.wikipedia.org/wiki/Activation_function),
 * for a broader survey of the design space and historical progression.
 */
import {
  absoluteActivation,
  bentIdentityActivation,
  bipolarActivation,
  bipolarSigmoidActivation,
  geluActivation,
  gaussianActivation,
  hardTanhActivation,
  identityActivation,
  inverseActivation,
  logisticActivation,
  mishActivation,
  reluActivation,
  seluActivation,
  sigmoidActivation,
  sinusoidActivation,
  softplusActivation,
  softsignActivation,
  stepActivation,
  swishActivation,
  tanhActivation,
  type ActivationFunction,
} from './activation.utils';

/**
 * Runtime registry of built-in and custom activation functions.
 *
 * The chosen activation function determines what each neuron in the network
 * can *represent* — whether it can learn smooth boundaries, sparse features,
 * periodic patterns, or gated on/off signals. In NEAT, the evolutionary
 * controller can assign different activations to different nodes, so this
 * registry is the complete vocabulary of expressible neuron behaviors.
 *
 * ## Key Formulas
 *
 * A few formulas are worth memorizing because they define the most-used choices:
 *
 * ```
 * logistic(x)  = 1 / (1 + e^-x)           range: (0, 1)
 * tanh(x)      = (e^x - e^-x) / (e^x + e^-x)  range: (-1, 1)
 * relu(x)      = max(0, x)                 range: [0, ∞)
 * softplus(x)  = ln(1 + e^x)              smooth ReLU approximation
 * swish(x)     = x · logistic(x)           self-gated, non-monotone
 * gelu(x)      ≈ x · Φ(x)                 Gaussian CDF gating
 * ```
 *
 * The derivative of each activation function determines how gradient
 * information flows backward through the network during training. Saturating
 * functions (`logistic`, `tanh`) have vanishingly small derivatives far from
 * the origin — this is the *vanishing gradient problem* that motivated
 * ReLU-family activations. See Wikipedia contributors,
 * [Vanishing gradient problem](https://en.wikipedia.org/wiki/Vanishing_gradient_problem),
 * for the historical context.
 *
 * ## Function Map
 *
 * ```mermaid
 * flowchart TD
 *   classDef base fill:#001522,stroke:#0fb5ff,color:#9fdcff,stroke-width:1.5px;
 *   classDef accent fill:#0f1f33,stroke:#00e5ff,color:#d8f6ff,stroke-width:2px;
 *
 *   Shelf["Activation registry"]:::accent --> Bounded["Saturating · bounded output"]:::base
 *   Shelf --> Piecewise["Piecewise linear · sparse"]:::base
 *   Shelf --> Specialized["Shape-specialized · periodic/radial"]:::base
 *   Shelf --> Smooth["Smooth modern · differentiable everywhere"]:::base
 *   Bounded --> B2["logistic · sigmoid · tanh\nbipolars · softsign"]:::base
 *   Piecewise --> P2["relu · hardTanh · step\nidentity · absolute · inverse"]:::base
 *   Specialized --> S2["gaussian · sinusoid\nbentIdentity · selu"]:::base
 *   Smooth --> M2["softplus · swish · gelu · mish"]:::base
 * ```
 *
 * Every activation shares the same calling convention: pass the input value
 * and optionally `true` for the local derivative.
 *
 * Minimal workflow:
 *
 * ```ts
 * const hiddenValue = Activation.relu(weightedSum);
 * const outputSlope = Activation.logistic(weightedSum, true); // derivative
 *
 * registerCustomActivation(
 *   'cube',
 *   (inputValue, shouldComputeDerivative = false) =>
 *     shouldComputeDerivative ? 3 * inputValue * inputValue : inputValue ** 3,
 * );
 *
 * const customValue = Activation.cube(0.5);
 * ```
 *
 * Practical first-experiment chooser:
 *
 * - `relu` — simplest sparse hidden-layer default; fast and effective.
 * - `tanh` — zero-centered bounded alternative; useful for recurrent setups.
 * - `softplus`, `swish`, `gelu`, or `mish` — smoother ReLU alternatives.
 * - `logistic` / `sigmoid` — bounded probability-like outputs.
 * - `registerCustomActivation()` — when the built-ins don't fit the transfer
 *   curve your experiment needs.
 *
 * @see {@link https://en.wikipedia.org/wiki/Activation_function}
 * @see {@link https://en.wikipedia.org/wiki/Universal_approximation_theorem}
 * @see {@link https://en.wikipedia.org/wiki/Rectifier_(neural_networks)}
 * @see {@link https://en.wikipedia.org/wiki/Vanishing_gradient_problem}
 */
export const Activation: Record<string, ActivationFunction> = {
  /**
   * Logistic (Sigmoid) activation function.
   * Outputs values between 0 and 1. Commonly used in older network architectures
   * and for output layers in binary classification tasks.
   * @param {number} inputValue - The input value.
   * @param {boolean} [shouldComputeDerivative=false] - Whether to compute the derivative.
   * @returns {number} The result of the logistic function or its derivative.
   */
  logistic: logisticActivation,

  /**
   * Alias for Logistic (Sigmoid) activation function.
   * Outputs values between 0 and 1. Commonly used in older network architectures
   * and for output layers in binary classification tasks.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the logistic function or its derivative.
   */
  sigmoid: sigmoidActivation,

  /**
   * Hyperbolic tangent (tanh) activation function.
   * Outputs values between -1 and 1. Often preferred over logistic sigmoid in hidden layers
   * due to its zero-centered output, which can help with training convergence.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the tanh function or its derivative.
   */
  tanh: tanhActivation,

  /**
   * Identity activation function (Linear).
   * Outputs the input value directly: f(x) = x.
   * Used when no non-linearity is desired, e.g., in output layers for regression tasks.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the identity function (x) or its derivative (1).
   */
  identity: identityActivation,

  /**
   * Step activation function (Binary Step).
   * Outputs 0 if the input is negative or zero, and 1 if the input is positive.
   * Rarely used in modern deep learning due to its zero derivative almost everywhere,
   * hindering gradient-based learning.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the step function (0 or 1) or its derivative (0).
   */
  step: stepActivation,

  /**
   * Rectified Linear Unit (ReLU) activation function.
   * Outputs the input if it's positive, and 0 otherwise: f(x) = max(0, x).
   * Widely used in deep learning due to its simplicity, computational efficiency,
   * and ability to mitigate the vanishing gradient problem.
   *
   * Note: The derivative at x=0 is ambiguous (theoretically undefined). Here, we return 0,
   * which is a common practical choice. If you need a different behavior, consider using a custom activation.
   *
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the ReLU function or its derivative (0 or 1).
   */
  relu: reluActivation,

  /**
   * Softsign activation function.
   * A smooth approximation of the sign function: f(x) = x / (1 + |x|).
   * Outputs values between -1 and 1.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the softsign function or its derivative.
   */
  softsign: softsignActivation,

  /**
   * Sinusoid activation function.
   * Uses the standard sine function: f(x) = sin(x).
   * Can be useful for tasks involving periodic patterns.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the sinusoid function or its derivative (cos(x)).
   */
  sinusoid: sinusoidActivation,

  /**
   * Gaussian activation function.
   * Uses the Gaussian (bell curve) function: f(x) = exp(-x^2).
   * Outputs values between 0 and 1. Sometimes used in radial basis function (RBF) networks.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the Gaussian function or its derivative.
   */
  gaussian: gaussianActivation,

  /**
   * Bent Identity activation function.
   * A function that behaves linearly for large positive inputs but non-linearly near zero:
   * f(x) = (sqrt(x^2 + 1) - 1) / 2 + x.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the bent identity function or its derivative.
   */
  bentIdentity: bentIdentityActivation,

  /**
   * Bipolar activation function (Sign function).
   * Outputs -1 if the input is negative or zero, and 1 if the input is positive.
   * Similar to the Step function but with outputs -1 and 1.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the bipolar function (-1 or 1) or its derivative (0).
   */
  bipolar: bipolarActivation,

  /**
   * Bipolar Sigmoid activation function.
   * A scaled and shifted version of the logistic sigmoid, outputting values between -1 and 1:
   * f(x) = 2 * logistic(x) - 1 = (1 - exp(-x)) / (1 + exp(-x)).
   * This is equivalent to the hyperbolic tangent (tanh) function.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the bipolar sigmoid function or its derivative.
   * @see {@link Activation.tanh}
   */
  bipolarSigmoid: bipolarSigmoidActivation,

  /**
   * Hard Tanh activation function.
   * A computationally cheaper, piecewise linear approximation of the tanh function:
   * f(x) = max(-1, min(1, x)). Outputs values clamped between -1 and 1.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the hard tanh function or its derivative (0 or 1).
   */
  hardTanh: hardTanhActivation,

  /**
   * Absolute activation function.
   * Outputs the absolute value of the input: f(x) = |x|.
   *
   * Note: The derivative at x=0 is ambiguous (theoretically undefined). Here, we return 1.
   * If you need a different behavior, consider using a custom activation.
   *
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the absolute function or its derivative (sign of x).
   */
  absolute: absoluteActivation,

  /**
   * Inverse activation function.
   * Outputs 1 minus the input: f(x) = 1 - x.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the inverse function or its derivative (-1).
   */
  inverse: inverseActivation,

  /**
   * Scaled Exponential Linear Unit (SELU) activation function.
   *
   * SELU aims to induce self-normalizing properties, meaning the outputs of SELU units
   * automatically converge towards zero mean and unit variance.
   * f(x) = scale * (max(0, x) + min(0, alpha * (exp(x) - 1)))
   * Recommended for deep networks composed primarily of SELU units.
   *
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the SELU function or its derivative.
   * @see {@link https://arxiv.org/abs/1706.02515} - Self-Normalizing Neural Networks paper
   * @see {@link https://github.com/wagenaartje/neataptic/wiki/Activation#selu} - Neataptic context
   */
  selu: seluActivation,

  /**
   * Softplus activation function.
   * A smooth approximation of the ReLU function: f(x) = log(1 + exp(x)).
   * Always positive. Its derivative is the logistic sigmoid function.
   * This implementation includes checks for numerical stability to avoid overflow/underflow.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the softplus function or its derivative (logistic sigmoid).
   * @see {@link https://en.wikipedia.org/wiki/Rectifier_(neural_networks)#Softplus}
   */
  softplus: softplusActivation,

  /**
   * Swish activation function (SiLU - Sigmoid Linear Unit).
   * A self-gated activation function: f(x) = x * logistic(x).
   * Often performs better than ReLU in deeper models.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the swish function or its derivative.
   * @see {@link https://arxiv.org/abs/1710.05941} - Swish paper
   */
  swish: swishActivation,

  /**
   * Gaussian Error Linear Unit (GELU) activation function.
   * Smooth approximation of ReLU, often used in Transformer models.
   * f(x) = x * Φ(x), where Φ(x) is the standard Gaussian cumulative distribution function (CDF).
   * This implementation uses a common fast approximation of GELU.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the GELU function or its derivative.
   * @see {@link https://arxiv.org/abs/1606.08415}
   */
  gelu: geluActivation,

  /**
   * Mish activation function.
   * A self-gated activation function similar to Swish: f(x) = x * tanh(softplus(x)).
   * Aims to provide better performance than ReLU and Swish in some cases.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the Mish function or its derivative.
   * @see {@link https://arxiv.org/abs/1908.08681}
   */
  mish: mishActivation,
};

/**
 * Register a custom activation function at runtime.
 *
 * Use this escape hatch when the built-in shelf is close to what you need but
 * a specific experiment wants a different transfer curve. Registration mutates
 * the shared {@link Activation} registry, so later lookups can call the custom
 * function through the same surface as the built-ins.
 *
 * ```ts
 * registerCustomActivation(
 *   'leakySquare',
 *   (inputValue, shouldComputeDerivative = false) => {
 *     if (shouldComputeDerivative) {
 *       return inputValue >= 0 ? 2 * inputValue : 0.1;
 *     }
 *
 *     return inputValue >= 0 ? inputValue ** 2 : 0.1 * inputValue;
 *   },
 * );
 * ```
 *
 * @param {string} activationName - Name used as the registry key.
 * @param {ActivationFunction} activationFunction - Forward-and-derivative implementation for the custom transfer curve.
 * @returns {void} Does not return a value; it mutates the shared activation registry.
 */
export const registerCustomActivation = (
  activationName: string,
  activationFunction: ActivationFunction,
): void => {
  Activation[activationName] = activationFunction;
};

export default Activation;
