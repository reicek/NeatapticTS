/**
 * Activation function
 * @see {@link https://en.wikipedia.org/wiki/Activation_function}
 * @see {@link https://stats.stackexchange.com/questions/115258/comprehensive-list-of-activation-functions-in-neural-networks-with-pros-cons}
 */
export default class Activation {
  static logistic(x, derivate) {
    const fx = 1 / (1 + Math.exp(-x));

    return !derivate ? fx : fx * (1 - fx);
  }

  static tanh(x, derivate) {
    return derivate ? 1 - Math.pow(Math.tanh(x), 2) : Math.tanh(x);
  }

  static identity(x, derivate) {
    return derivate ? 1 : x;
  }

  static step(x, derivate) {
    return derivate ? 0 : x > 0 ? 1 : 0;
  }

  static relu(x, derivate) {
    return derivate ? (x > 0 ? 1 : 0) : x > 0 ? x : 0;
  }

  static softsign(x, derivate) {
    const d = 1 + Math.abs(x);

    return derivate ? x / Math.pow(d, 2) : x / d;
  }

  static sinusoid(x, derivate) {
    return derivate ? Math.cos(x) : Math.sin(x);
  }

  static gaussian(x, derivate) {
    const d = Math.exp(-Math.pow(x, 2));

    return derivate ? -2 * x * d : d;
  }

  static bentIdentity(x, derivate) {
    const d = Math.sqrt(Math.pow(x, 2) + 1);

    return derivate ? x / (2 * d) + 1 : (d - 1) / 2 + x;
  }

  static bipolar(x, derivate) {
    return derivate ? 0 : x > 0 ? 1 : -1;
  }

  static bipolarSigmoid(x, derivate) {
    const d = 2 / (1 + Math.exp(-x)) - 1;

    return derivate ? (1 / 2) * (1 + d) * (1 - d) : d;
  }

  static hardTanh(x, derivate) {
    return derivate ? (x > -1 && x < 1 ? 1 : 0) : Math.max(-1, Math.min(1, x));
  }

  static absolute(x, derivate) {
    return derivate ? (x < 0 ? -1 : 1) : Math.abs(x);
  }

  static inverse(x, derivate) {
    return derivate ? -1 : 1 - x;
  }

  /** @see {@link https://arxiv.org/pdf/1706.02515.pdf} */
  static selu(x, derivate) {
    const alpha = 1.6732632423543772848170429916717;
    const scale = 1.0507009873554804934193349852946;
    const fx = x > 0 ? x : alpha * Math.exp(x) - alpha;

    return derivate ? (x > 0 ? scale : (fx + alpha) * scale) : fx * scale;
  }
}
