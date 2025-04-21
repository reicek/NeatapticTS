/**
 * Activation functions used in neural networks.
 *
 * Activation functions determine the output of a node in a neural network
 * based on its inputs and weights. Nonlinear activation functions enable
 * neural networks to approximate complex functions, as stated in the
 * Universal Approximation Theorem.
 *
 * Common activation functions include sigmoid, ReLU, and tanh, each with
 * unique properties such as range, differentiability, and computational
 * efficiency. These properties influence the network's training stability
 * and performance.
 *
 * @see {@link https://en.wikipedia.org/wiki/Activation_function}
 * @see {@link https://en.wikipedia.org/wiki/Universal_approximation_theorem}
 * @see {@link https://en.wikipedia.org/wiki/Rectifier_(neural_networks)}
 */
export class Activation {
  /**
   * Logistic (Sigmoid) activation function.
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
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the tanh function or its derivative.
   */
  static tanh(x: number, derivate: boolean = false): number {
    return derivate ? 1 - Math.pow(Math.tanh(x), 2) : Math.tanh(x);
  }

  /**
   * Identity activation function.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the identity function or its derivative.
   */
  static identity(x: number, derivate: boolean = false): number {
    return derivate ? 1 : x;
  }

  /**
   * Step activation function.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the step function or its derivative.
   */
  static step(x: number, derivate: boolean = false): number {
    return derivate ? 0 : x > 0 ? 1 : 0;
  }

  /**
   * Rectified Linear Unit (ReLU) activation function.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the ReLU function or its derivative.
   */
  static relu(x: number, derivate: boolean = false): number {
    return derivate ? (x > 0 ? 1 : 0) : x > 0 ? x : 0;
  }

  /**
   * Softsign activation function.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the softsign function or its derivative.
   */
  static softsign(x: number, derivate: boolean = false): number {
    const d = 1 + Math.abs(x);
    return derivate ? 1 / Math.pow(d, 2) : x / d; // Corrected derivative
  }

  /**
   * Sinusoid activation function.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the sinusoid function or its derivative.
   */
  static sinusoid(x: number, derivate: boolean = false): number {
    return derivate ? Math.cos(x) : Math.sin(x);
  }

  /**
   * Gaussian activation function.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the Gaussian function or its derivative.
   */
  static gaussian(x: number, derivate: boolean = false): number {
    const d = Math.exp(-Math.pow(x, 2));
    return derivate ? -2 * x * d : d;
  }

  /**
   * Bent Identity activation function.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the bent identity function or its derivative.
   */
  static bentIdentity(x: number, derivate: boolean = false): number {
    const d = Math.sqrt(Math.pow(x, 2) + 1);
    return derivate ? x / (2 * d) + 1 : (d - 1) / 2 + x;
  }

  /**
   * Bipolar activation function.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the bipolar function or its derivative.
   */
  static bipolar(x: number, derivate: boolean = false): number {
    return derivate ? 0 : x > 0 ? 1 : -1;
  }

  /**
   * Bipolar Sigmoid activation function.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the bipolar sigmoid function or its derivative.
   */
  static bipolarSigmoid(x: number, derivate: boolean = false): number {
    const d = 2 / (1 + Math.exp(-x)) - 1;
    return derivate ? (1 / 2) * (1 + d) * (1 - d) : d;
  }

  /**
   * Hard Tanh activation function.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the hard tanh function or its derivative.
   */
  static hardTanh(x: number, derivate: boolean = false): number {
    return derivate ? (x > -1 && x < 1 ? 1 : 0) : Math.max(-1, Math.min(1, x));
  }

  /**
   * Absolute activation function.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the absolute function or its derivative.
   */
  static absolute(x: number, derivate: boolean = false): number {
    return derivate ? (x < 0 ? -1 : 1) : Math.abs(x);
  }

  /**
   * Inverse activation function.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the inverse function or its derivative.
   */
  static inverse(x: number, derivate: boolean = false): number {
    return derivate ? -1 : 1 - x;
  }

  /**
   * Scaled Exponential Linear Unit (SELU) activation function.
   *
   * This activation function is one of the recommended options for nodes in the
   * Instinct algorithm, providing self-normalizing properties.
   *
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the SELU function or its derivative.
   * @see Instinct Algorithm - Section 3.6 Modify Squash Mutation
   */
  static selu(x: number, derivate: boolean = false): number {
    const alpha = 1.6732632423543772848170429916717;
    const scale = 1.0507009873554804934193349852946;
    const fx = x > 0 ? x : alpha * Math.exp(x) - alpha;
    return derivate ? (x > 0 ? scale : (fx + alpha) * scale) : fx * scale;
  }

  /**
   * Softplus activation function.
   * @param {number} x - The input value.
   * @param {boolean} [derivate=false] - Whether to compute the derivative.
   * @returns {number} The result of the softplus function or its derivative.
   */
  static softplus(x: number, derivate: boolean = false): number {
    const fx = Math.log(1 + Math.exp(x));
    return derivate ? 1 / (1 + Math.exp(-x)) : fx;
  }
}

export default Activation;
