/**
 * Provides a collection of standard cost functions (also known as loss functions)
 * used for evaluating the performance of neural networks during training.
 *
 * Cost functions quantify the difference between the network's predictions
 * and the actual target values. The goal of training is typically to minimize
 * the value of the cost function. The choice of cost function is crucial and
 * depends on the specific task (e.g., regression, classification) and the
 * desired behavior of the model.
 *
 * @see {@link https://en.wikipedia.org/wiki/Loss_function}
 */
export default class Cost {
  /**
   * Calculates the Cross Entropy error, commonly used for classification tasks.
   *
   * This function measures the performance of a classification model whose output is
   * a probability value between 0 and 1. Cross-entropy loss increases as the
   * predicted probability diverges from the actual label.
   *
   * It uses a small epsilon (1e-15) to prevent `log(0)` which would result in `NaN`.
   * Output values are clamped to the range `[epsilon, 1 - epsilon]` for numerical stability.
   *
   * @see {@link https://en.wikipedia.org/wiki/Cross_entropy}
   * @param {number[]} targets - An array of target values, typically 0 or 1 for binary classification, or probabilities for soft labels.
   * @param {number[]} outputs - An array of output values from the network, representing probabilities (expected to be between 0 and 1).
   * @returns {number} The mean cross-entropy error over all samples.
   * @throws {Error} If the target and output arrays have different lengths.
   */
  static crossEntropy(targets: number[], outputs: number[]): number {
    let error = 0;
    const epsilon = 1e-15; // Small constant to avoid log(0)

    if (targets.length !== outputs.length) {
      throw new Error('Target and output arrays must have the same length.');
    }

    for (let i = 0; i < outputs.length; i++) {
      const target = targets[i];
      const output = outputs[i];

      // Clamp output to prevent log(0) or log(<0) issues.
      const clampedOutput = Math.max(epsilon, Math.min(1 - epsilon, output));

      // Note: Assumes target is 0 or 1 for standard binary cross-entropy.
      // The formula handles soft labels (targets between 0 and 1) correctly.
      if (target === 1) {
        error -= Math.log(clampedOutput); // Cost when target is 1
      } else if (target === 0) {
        error -= Math.log(1 - clampedOutput); // Cost when target is 0
      } else {
        // General case for targets between 0 and 1 (soft labels)
        error -=
          target * Math.log(clampedOutput) +
          (1 - target) * Math.log(1 - clampedOutput);
      }
    }

    // Return the average error over the batch/dataset.
    return error / outputs.length;
  }

  /**
   * Calculates the Mean Squared Error (MSE), a common loss function for regression tasks.
   *
   * MSE measures the average of the squares of the errors—that is, the average
   * squared difference between the estimated values and the actual value.
   * It is sensitive to outliers due to the squaring of the error terms.
   *
   * @see {@link https://en.wikipedia.org/wiki/Mean_squared_error}
   * @param {number[]} targets - An array of target numerical values.
   * @param {number[]} outputs - An array of output values from the network.
   * @returns {number} The mean squared error.
   * @throws {Error} If the target and output arrays have different lengths (implicitly via forEach).
   */
  static mse(targets: number[], outputs: number[]): number {
    let error = 0;

    // Assumes targets and outputs have the same length.
    outputs.forEach((output, outputIndex) => {
      // Calculate the squared difference for each sample.
      error += Math.pow(targets[outputIndex] - output, 2);
    });

    // Return the average squared error.
    return error / outputs.length;
  }

  /**
   * Calculates the Binary Error rate, often used as a simple accuracy metric for classification.
   *
   * This function calculates the proportion of misclassifications by comparing the
   * rounded network outputs (thresholded at 0.5) against the target labels.
   * It assumes target values are 0 or 1, and outputs are probabilities between 0 and 1.
   * Note: This is equivalent to `1 - accuracy` for binary classification.
   *
   * @param {number[]} targets - An array of target values, expected to be 0 or 1.
   * @param {number[]} outputs - An array of output values from the network, typically probabilities between 0 and 1.
   * @returns {number} The proportion of misclassified samples (error rate, between 0 and 1).
   * @throws {Error} If the target and output arrays have different lengths (implicitly via forEach).
   */
  static binary(targets: number[], outputs: number[]): number {
    let misses = 0;

    // Assumes targets and outputs have the same length.
    outputs.forEach((output, outputIndex) => {
      // Round output to nearest integer (0 or 1) using a 0.5 threshold.
      // Compare rounded output to the target label.
      misses += Math.round(targets[outputIndex]) !== Math.round(output) ? 1 : 0;
    });

    // Return the error rate (proportion of misses).
    return misses / outputs.length;
    // Alternative: return `misses` to get the raw count of misclassifications.
  }

  /**
   * Calculates the Mean Absolute Error (MAE), another common loss function for regression tasks.
   *
   * MAE measures the average of the absolute differences between predictions and actual values.
   * Compared to MSE, it is less sensitive to outliers because errors are not squared.
   *
   * @see {@link https://en.wikipedia.org/wiki/Mean_absolute_error}
   * @param {number[]} targets - An array of target numerical values.
   * @param {number[]} outputs - An array of output values from the network.
   * @returns {number} The mean absolute error.
   * @throws {Error} If the target and output arrays have different lengths (implicitly via forEach).
   */
  static mae(targets: number[], outputs: number[]): number {
    let error = 0;

    // Assumes targets and outputs have the same length.
    outputs.forEach((output, outputIndex) => {
      // Calculate the absolute difference for each sample.
      error += Math.abs(targets[outputIndex] - output);
    });

    // Return the average absolute error.
    return error / outputs.length;
  }

  /**
   * Calculates the Mean Absolute Percentage Error (MAPE).
   *
   * MAPE expresses the error as a percentage of the actual value. It can be useful
   * for understanding the error relative to the magnitude of the target values.
   * However, it has limitations: it's undefined when the target value is zero and
   * can be skewed by target values close to zero.
   *
   * @see {@link https://en.wikipedia.org/wiki/Mean_absolute_percentage_error}
   * @param {number[]} targets - An array of target numerical values. Should not contain zeros for standard MAPE.
   * @param {number[]} outputs - An array of output values from the network.
   * @returns {number} The mean absolute percentage error, expressed as a proportion (e.g., 0.1 for 10%).
   * @throws {Error} If the target and output arrays have different lengths (implicitly via forEach).
   */
  static mape(targets: number[], outputs: number[]): number {
    let error = 0;
    const epsilon = 1e-15; // Small constant to avoid division by zero or near-zero target values.

    // Assumes targets and outputs have the same length.
    outputs.forEach((output, outputIndex) => {
      const target = targets[outputIndex];
      // Calculate the absolute percentage error for each sample.
      // Use Math.max with epsilon to prevent division by zero.
      error += Math.abs(
        (target - output) / Math.max(Math.abs(target), epsilon)
      );
    });

    // Return the average absolute percentage error (as a proportion).
    // Multiply by 100 if a percentage value is desired.
    return error / outputs.length;
  }

  /**
   * Calculates the Mean Squared Logarithmic Error (MSLE).
   *
   * MSLE is often used in regression tasks where the target values span a large range
   * or when penalizing under-predictions more than over-predictions is desired.
   * It measures the squared difference between the logarithms of the predicted and actual values.
   * Uses `log(1 + x)` instead of `log(x)` for numerical stability and to handle inputs of 0.
   * Assumes both targets and outputs are non-negative.
   *
   * @see {@link https://peltarion.com/knowledge-center/documentation/modeling-view/build-an-ai-model/loss-functions/mean-squared-logarithmic-error}
   * @param {number[]} targets - An array of target numerical values (assumed >= 0).
   * @param {number[]} outputs - An array of output values from the network (assumed >= 0).
   * @returns {number} The mean squared logarithmic error.
   * @throws {Error} If the target and output arrays have different lengths (implicitly via forEach).
   */
  static msle(targets: number[], outputs: number[]): number {
    let error = 0;

    // Assumes targets and outputs have the same length.
    outputs.forEach((output, outputIndex) => {
      const target = targets[outputIndex];
      // Ensure inputs are non-negative before adding 1 for the logarithm.
      // Using log(1 + x) avoids issues with log(0) and handles values >= 0.
      const logTarget = Math.log(Math.max(target, 0) + 1);
      const logOutput = Math.log(Math.max(output, 0) + 1);
      // Calculate the squared difference of the logarithms.
      error += Math.pow(logTarget - logOutput, 2);
    });

    // Return the average squared logarithmic error.
    return error / outputs.length;
  }

  /**
   * Calculates the Mean Hinge loss, primarily used for "maximum-margin" classification,
   * most notably for Support Vector Machines (SVMs).
   *
   * Hinge loss is used for training classifiers. It penalizes predictions that are
   * not only incorrect but also those that are correct but not confident (i.e., close to the decision boundary).
   * Assumes target values are encoded as -1 or 1.
   *
   * @see {@link https://en.wikipedia.org/wiki/Hinge_loss}
   * @param {number[]} targets - An array of target values, expected to be -1 or 1.
   * @param {number[]} outputs - An array of output values from the network (raw scores, not necessarily probabilities).
   * @returns {number} The mean hinge loss.
   * @throws {Error} If the target and output arrays have different lengths (implicitly via forEach).
   */
  static hinge(targets: number[], outputs: number[]): number {
    let error = 0;

    // Assumes targets and outputs have the same length.
    outputs.forEach((output, outputIndex) => {
      const target = targets[outputIndex]; // Should be -1 or 1 for standard hinge loss.
      // The term `target * output` should be >= 1 for a correct and confident prediction.
      // Loss is incurred if `target * output < 1`.
      error += Math.max(0, 1 - target * output);
    });

    // Return the average hinge loss.
    return error / outputs.length;
  }
}
