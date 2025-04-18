/**
 * Cost functions used to evaluate the performance of neural networks.
 * @see {@link https://en.wikipedia.org/wiki/Loss_function}
 */
export default class Cost {
  /**
   * Cross entropy error.
   * Avoid negative and zero numbers, use 1e-15 to prevent log(0).
   * @see {@link http://bit.ly/2p5W29A}
   * @param {number[]} targets - The target values.
   * @param {number[]} outputs - The output values from the network.
   * @returns {number} The cross-entropy error.
   */
  static crossEntropy(targets: number[], outputs: number[]): number {
    let error = 0;

    outputs.forEach((output, outputIndex) => {
      // Avoid negative and zero numbers, use 1e-15 to prevent log(0)
      error -=
        targets[outputIndex] * Math.log(Math.max(output, 1e-15)) +
        (1 - targets[outputIndex]) * Math.log(1 - Math.max(output, 1e-15));
    });

    return error / outputs.length;
  }

  /**
   * Mean Squared Error (MSE).
   * @param {number[]} targets - The target values.
   * @param {number[]} outputs - The output values from the network.
   * @returns {number} The mean squared error.
   */
  static mse(targets: number[], outputs: number[]): number {
    let error = 0;

    outputs.forEach((output, outputIndex) => {
      error += Math.pow(targets[outputIndex] - output, 2);
    });

    return error / outputs.length;
  }

  /**
   * Binary error.
   * @param {number[]} targets - The target values.
   * @param {number[]} outputs - The output values from the network.
   * @returns {number} The binary error (number of misclassifications).
   */
  static binary(targets: number[], outputs: number[]): number {
    let misses = 0;

    outputs.forEach((output, outputIndex) => {
      misses +=
        Math.round(targets[outputIndex] * 2) !== Math.round(output * 2) ? 1 : 0;
    });

    return misses;
  }

  /**
   * Mean Absolute Error (MAE).
   * @param {number[]} targets - The target values.
   * @param {number[]} outputs - The output values from the network.
   * @returns {number} The mean absolute error.
   */
  static mae(targets: number[], outputs: number[]): number {
    let error = 0;

    outputs.forEach((output, outputIndex) => {
      error += Math.abs(targets[outputIndex] - output);
    });

    return error / outputs.length;
  }

  /**
   * Mean Absolute Percentage Error (MAPE).
   * @param {number[]} targets - The target values.
   * @param {number[]} outputs - The output values from the network.
   * @returns {number} The mean absolute percentage error.
   */
  static mape(targets: number[], outputs: number[]): number {
    let error = 0;

    outputs.forEach((output, outputIndex) => {
      error += Math.abs(
        (output - targets[outputIndex]) / Math.max(targets[outputIndex], 1e-15)
      );
    });

    return error / outputs.length;
  }

  /**
   * Mean Squared Logarithmic Error (MSLE).
   * Note: The original JavaScript implementation does not divide by the length of outputs.
   * @param {number[]} targets - The target values.
   * @param {number[]} outputs - The output values from the network.
   * @returns {number} The mean squared logarithmic error.
   */
  static msle(targets: number[], outputs: number[]): number {
    let error = 0;

    outputs.forEach((output, outputIndex) => {
      error += Math.pow(
        Math.log(Math.max(targets[outputIndex], 1e-15)) -
          Math.log(Math.max(output, 1e-15)),
        2
      );
    });

    // Uncomment the following line if division by outputs.length is required:
    // return error / outputs.length;
    return error;
  }

  /**
   * Hinge loss, used for classifiers.
   * @param {number[]} targets - The target values.
   * @param {number[]} outputs - The output values from the network.
   * @returns {number} The hinge loss.
   */
  static hinge(targets: number[], outputs: number[]): number {
    let error = 0;

    outputs.forEach((output, outputIndex) => {
      error += Math.max(0, 1 - targets[outputIndex] * output);
    });

    return error;
  }
}
