/**
 * Cost functions
 * @see {@link https://en.wikipedia.org/wiki/Loss_function}
 */
export default class Cost {
  /**
   * Cross entropy error
   * @param {*} targets
   * @param {*} outputs
   * @returns
   */
  static crossEntropy(targets, outputs) {
    let error = 0;

    outputs.forEach((output, outputIndex) => {
      // Avoid negative and zero numbers, use 1e-15 http://bit.ly/2p5W29A
      error -=
        targets[outputIndex] * Math.log(Math.max(output, 1e-15)) +
        (1 - targets[outputIndex]) * Math.log(1 - Math.max(output, 1e-15));
    });

    return error / outputs.length;
  }

  /**
   * Mean Squared Error
   * @param {*} targets
   * @param {*} outputs
   * @returns
   */
  static mse(targets, outputs) {
    let error = 0;

    outputs.forEach((output, outputIndex) => {
      error += Math.pow(targets[outputIndex] - output, 2);
    });

    return error / outputs.length;
  }

  /**
   * Binary error
   * @param {*} targets
   * @param {*} outputs
   * @returns
   */
  static binary(targets, outputs) {
    let misses = 0;

    outputs.forEach((output, outputIndex) => {
      misses += Math.round(targets[outputIndex] * 2) !== Math.round(output * 2);
    });

    return misses;
  }

  /**
   * Mean Absolute Error
   * @param {*} targets
   * @param {*} outputs
   * @returns
   */
  static mae(targets, outputs) {
    let error = 0;

    outputs.forEach((output, outputIndex) => {
      error += Math.abs(targets[outputIndex] - output);
    });

    return error / outputs.length;
  }

  /**
   * Mean Absolute Percentage Error
   * @param {*} targets
   * @param {*} outputs
   * @returns
   */
  static mape(targets, outputs) {
    let error = 0;

    outputs.forEach((output, outputIndex) => {
      error += Math.abs(
        (output - targets[outputIndex]) / Math.max(targets[outputIndex], 1e-15)
      );
    });

    return error / outputs.length;
  }

  /**
   * Mean Squared Logarithmic Error
   * @param {*} targets
   * @param {*} outputs
   * @returns
   */
  static smle(targets, outputs) {
    let error = 0;

    outputs.forEach((output, outputIndex) => {
      error +=
        Math.log(Math.max(targets[outputIndex], 1e-15)) -
        Math.log(Math.max(output, 1e-15));
    });

    return error;
  }

  /**
   * Hinge loss, for classifiers
   * @param {*} targets
   * @param {*} outputs
   * @returns
   */
  static hinge(targets, outputs) {
    let error = 0;

    outputs.forEach((output, outputIndex) => {
      error += Math.max(0, 1 - targets[outputIndex] * output);
    });

    return error;
  }
}
