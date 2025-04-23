/**
 * Provides a collection of common activation functions used in neural networks.
 *
 * Activation functions introduce non-linearity into the network, allowing it to
 * learn complex patterns. They determine the output of a node based on its
 * weighted inputs and bias. The choice of activation function can significantly
 * impact the network's performance and training dynamics.
 *
 * All methods in this class are static and can be called directly, e.g., `Activation.relu(x)`.
 * Each method accepts an input value `x` and an optional boolean `derivate`.
 * If `derivate` is true, the method returns the derivative of the activation function
 * with respect to `x`; otherwise, it returns the activation function's output.
 *
 * @see {@link https://en.wikipedia.org/wiki/Activation_function}
 * @see {@link https://en.wikipedia.org/wiki/Universal_approximation_theorem}
 * @see {@link https://en.wikipedia.org/wiki/Rectifier_(neural_networks)}
 */
export class Activation {
  /**
   * Logistic (Sigmoid) activation function.
   * Outputs values between 0 and 1. Commonly used in older network architectures
   * and for output layers in binary classification tasks.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the logistic function or its derivative.
   */
  static logistic(x: number, derivate: boolean = false): number {
    const fx = 1 / (1 + Math.exp(-x));
    return !derivate ? fx : fx * (1 - fx);
  }

  /**
   * Hyperbolic tangent (tanh) activation function.
   * Outputs values between -1 and 1. Often preferred over logistic sigmoid in hidden layers
   * due to its zero-centered output, which can help with training convergence.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the tanh function or its derivative.
   */
  static tanh(x: number, derivate: boolean = false): number {
    return derivate ? 1 - Math.pow(Math.tanh(x), 2) : Math.tanh(x);
  }

  /**
   * Identity activation function (Linear).
   * Outputs the input value directly: f(x) = x.
   * Used when no non-linearity is desired, e.g., in output layers for regression tasks.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the identity function (x) or its derivative (1).
   */
  static identity(x: number, derivate: boolean = false): number {
    return derivate ? 1 : x;
  }

  /**
   * Step activation function (Binary Step).
   * Outputs 0 if the input is negative or zero, and 1 if the input is positive.
   * Rarely used in modern deep learning due to its zero derivative almost everywhere,
   * hindering gradient-based learning.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the step function (0 or 1) or its derivative (0).
   */
  static step(x: number, derivate: boolean = false): number {
    return derivate ? 0 : x > 0 ? 1 : 0;
  }

  /**
   * Rectified Linear Unit (ReLU) activation function.
   * Outputs the input if it's positive, and 0 otherwise: f(x) = max(0, x).
   * Widely used in deep learning due to its simplicity, computational efficiency,
   * and ability to mitigate the vanishing gradient problem.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the ReLU function or its derivative (0 or 1).
   */
  static relu(x: number, derivate: boolean = false): number {
    return derivate ? (x > 0 ? 1 : 0) : x > 0 ? x : 0;
  }

  /**
   * Softsign activation function.
   * A smooth approximation of the sign function: f(x) = x / (1 + |x|).
   * Outputs values between -1 and 1.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the softsign function or its derivative.
   */
  static softsign(x: number, derivate: boolean = false): number {
    const d = 1 + Math.abs(x);
    // Derivative: 1 / (1 + |x|)^2
    return derivate ? 1 / Math.pow(d, 2) : x / d;
  }

  /**
   * Sinusoid activation function.
   * Uses the standard sine function: f(x) = sin(x).
   * Can be useful for tasks involving periodic patterns.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the sinusoid function or its derivative (cos(x)).
   */
  static sinusoid(x: number, derivate: boolean = false): number {
    return derivate ? Math.cos(x) : Math.sin(x);
  }

  /**
   * Gaussian activation function.
   * Uses the Gaussian (bell curve) function: f(x) = exp(-x^2).
   * Outputs values between 0 and 1. Sometimes used in radial basis function (RBF) networks.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the Gaussian function or its derivative.
   */
  static gaussian(x: number, derivate: boolean = false): number {
    const d = Math.exp(-Math.pow(x, 2));
    // Derivative: -2x * exp(-x^2)
    return derivate ? -2 * x * d : d;
  }

  /**
   * Bent Identity activation function.
   * A function that behaves linearly for large positive inputs but non-linearly near zero:
   * f(x) = (sqrt(x^2 + 1) - 1) / 2 + x.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the bent identity function or its derivative.
   */
  static bentIdentity(x: number, derivate: boolean = false): number {
    const d = Math.sqrt(Math.pow(x, 2) + 1);
    // Derivative: x / (2 * sqrt(x^2 + 1)) + 1
    return derivate ? x / (2 * d) + 1 : (d - 1) / 2 + x;
  }

  /**
   * Bipolar activation function (Sign function).
   * Outputs -1 if the input is negative or zero, and 1 if the input is positive.
   * Similar to the Step function but with outputs -1 and 1.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the bipolar function (-1 or 1) or its derivative (0).
   */
  static bipolar(x: number, derivate: boolean = false): number {
    return derivate ? 0 : x > 0 ? 1 : -1;
  }

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
  static bipolarSigmoid(x: number, derivate: boolean = false): number {
    const d = 2 / (1 + Math.exp(-x)) - 1;
    // Derivative: 0.5 * (1 + f(x)) * (1 - f(x))
    return derivate ? (1 / 2) * (1 + d) * (1 - d) : d;
  }

  /**
   * Hard Tanh activation function.
   * A computationally cheaper, piecewise linear approximation of the tanh function:
   * f(x) = max(-1, min(1, x)). Outputs values clamped between -1 and 1.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the hard tanh function or its derivative (0 or 1).
   */
  static hardTanh(x: number, derivate: boolean = false): number {
    // Derivative is 1 between -1 and 1, and 0 otherwise.
    return derivate ? (x > -1 && x < 1 ? 1 : 0) : Math.max(-1, Math.min(1, x));
  }

  /**
   * Absolute activation function.
   * Outputs the absolute value of the input: f(x) = |x|.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the absolute function or its derivative (sign of x).
   */
  static absolute(x: number, derivate: boolean = false): number {
    // Derivative is -1 for x < 0, 1 for x > 0. (Derivative at x=0 is undefined, commonly set to 1 or 0).
    return derivate ? (x < 0 ? -1 : 1) : Math.abs(x);
  }

  /**
   * Inverse activation function.
   * Outputs 1 minus the input: f(x) = 1 - x.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the inverse function or its derivative (-1).
   */
  static inverse(x: number, derivate: boolean = false): number {
    return derivate ? -1 : 1 - x;
  }

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
  static selu(x: number, derivate: boolean = false): number {
    const alpha = 1.6732632423543772848170429916717;
    const scale = 1.0507009873554804934193349852946;
    const fx = x > 0 ? x : alpha * Math.exp(x) - alpha;
    // Derivative: scale * (x > 0 ? 1 : alpha * exp(x))
    // Simplified derivative using fx: scale * (x > 0 ? 1 : fx + alpha)
    return derivate ? (x > 0 ? scale : (fx + alpha) * scale) : fx * scale;
  }

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
  static softplus(x: number, derivate: boolean = false): number {
    const fx = 1 / (1 + Math.exp(-x)); // Logistic sigmoid
    if (derivate) {
      return fx; // Derivative of softplus is logistic sigmoid
    } else {
      // Numerically stable softplus calculation:
      // log(1 + exp(x)) = log(exp(x)*(exp(-x) + 1)) = x + log(1 + exp(-x))
      // Choose calculation based on x to avoid large positive exponents causing overflow.
      if (x > 30) {
        return x; // For large positive x, softplus(x) ≈ x
      } else if (x < -30) {
        return Math.exp(x); // For large negative x, softplus(x) ≈ exp(x)
      }
      // Use the alternative stable formula for intermediate values:
      // max(0, x) + log(1 + exp(-abs(x)))
      return Math.max(0, x) + Math.log(1 + Math.exp(-Math.abs(x)));
    }
  }

  /**
   * Swish activation function (SiLU - Sigmoid Linear Unit).
   * A self-gated activation function: f(x) = x * logistic(x).
   * Often performs better than ReLU in deeper models.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the swish function or its derivative.
   * @see {@link https://arxiv.org/abs/1710.05941} - Swish paper
   */
  static swish(x: number, derivate: boolean = false): number {
    const sigmoid_x = 1 / (1 + Math.exp(-x));
    if (derivate) {
      // Derivative: sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
      // Can be rewritten using swish(x) = x * sigmoid(x):
      // swish'(x) = swish(x) + sigmoid(x) * (1 - swish(x))
      const swish_x = x * sigmoid_x;
      return swish_x + sigmoid_x * (1 - swish_x);
    } else {
      return x * sigmoid_x;
    }
  }

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
  static gelu(x: number, derivate: boolean = false): number {
    const cdf =
      0.5 *
      (1.0 +
        Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * Math.pow(x, 3))));
    if (derivate) {
      // Derivative of the GELU approximation:
      const intermediate = Math.sqrt(2.0 / Math.PI) * (1.0 + 0.134145 * x * x);
      const sech_arg =
        Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * Math.pow(x, 3));
      const sech_val = 1.0 / Math.cosh(sech_arg);
      const sech_sq = sech_val * sech_val;
      return cdf + x * 0.5 * intermediate * sech_sq;
    } else {
      return x * cdf;
    }
  }

  /**
   * Mish activation function.
   * A self-gated activation function similar to Swish: f(x) = x * tanh(softplus(x)).
   * Aims to provide better performance than ReLU and Swish in some cases.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the Mish function or its derivative.
   * @see {@link https://arxiv.org/abs/1908.08681}
   */
  static mish(x: number, derivate: boolean = false): number {
    // Use stable softplus calculation
    // softplus(x) = log(1 + exp(x))
    let sp_x: number;
    if (x > 30) {
      sp_x = x;
    } else if (x < -30) {
      sp_x = Math.exp(x);
    } else {
      sp_x = Math.max(0, x) + Math.log(1 + Math.exp(-Math.abs(x)));
    }

    const tanh_sp_x = Math.tanh(sp_x);

    if (derivate) {
      // Derivative of Mish: tanh(softplus(x)) + x * sech^2(softplus(x)) * sigmoid(x)
      const sigmoid_x = 1 / (1 + Math.exp(-x)); // Derivative of softplus
      const sech_sp_x = 1.0 / Math.cosh(sp_x); // sech(x) = 1 / cosh(x)
      const sech_sq_sp_x = sech_sp_x * sech_sp_x;
      return tanh_sp_x + x * sech_sq_sp_x * sigmoid_x;
    } else {
      return x * tanh_sp_x;
    }
  }
}

export default Activation;
