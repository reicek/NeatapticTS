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
import {
  computeBinaryError,
  computeCrossEntropy,
  computeFocalLoss,
  computeHingeLoss,
  computeLabelSmoothingLoss,
  computeMeanAbsoluteError,
  computeMeanAbsolutePercentageError,
  computeMeanSquaredError,
  computeMeanSquaredLogarithmicError,
  computeSoftmaxCrossEntropy,
  DEFAULT_FOCAL_ALPHA,
  DEFAULT_FOCAL_GAMMA,
  DEFAULT_LABEL_SMOOTHING,
} from './cost.utils';

export default class Cost {
  /**
   * Calculates the Cross Entropy error, commonly used for classification tasks.
   *
   * This function measures the performance of a classification model whose output is
   * a probability value between 0 and 1. Cross-entropy loss increases as the
   * predicted probability diverges from the actual label.
   *
   * It uses a small epsilon (PROB_EPSILON = 1e-15) to prevent `log(0)` which would result in `NaN`.
   * Output values are clamped to the range `[epsilon, 1 - epsilon]` for numerical stability.
   *
   * @see {@link https://en.wikipedia.org/wiki/Cross_entropy}
   * @param {number[]} targets - An array of target values, typically 0 or 1 for binary classification, or probabilities for soft labels.
   * @param {number[]} outputs - An array of output values from the network, representing probabilities (expected to be between 0 and 1).
   * @returns {number} The mean cross-entropy error over all samples.
   * @throws {Error} If the target and output arrays have different lengths.
   */
  static crossEntropy(targets: number[], outputs: number[]): number {
    return computeCrossEntropy(targets, outputs);
  }

  /**
   * Softmax Cross Entropy for mutually exclusive multi-class outputs given raw (pre-softmax or arbitrary) scores.
   * Applies a numerically stable softmax to the outputs internally then computes -sum(target * log(prob)).
   * Targets may be soft labels and are expected to sum to 1 (will be re-normalized if not).
   */
  static softmaxCrossEntropy(targets: number[], outputs: number[]): number {
    return computeSoftmaxCrossEntropy(targets, outputs);
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
    return computeMeanSquaredError(targets, outputs);
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
    return computeBinaryError(targets, outputs);
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
    return computeMeanAbsoluteError(targets, outputs);
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
    return computeMeanAbsolutePercentageError(targets, outputs);
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
    return computeMeanSquaredLogarithmicError(targets, outputs);
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
    return computeHingeLoss(targets, outputs);
  }

  /**
   * Calculates the Focal Loss, which is useful for addressing class imbalance in classification tasks.
   * Focal loss down-weights easy examples and focuses training on hard negatives.
   *
   * @see https://arxiv.org/abs/1708.02002
   * @param {number[]} targets - Array of target values (0 or 1 for binary, or probabilities for soft labels).
   * @param {number[]} outputs - Array of predicted probabilities (between 0 and 1).
   * @param {number} focalGamma - Focusing parameter (default 2).
   * @param {number} focalAlpha - Balancing parameter (default 0.25).
   * @returns {number} The mean focal loss.
   */
  static focalLoss(
    targets: number[],
    outputs: number[],
    focalGamma: number = DEFAULT_FOCAL_GAMMA,
    focalAlpha: number = DEFAULT_FOCAL_ALPHA,
  ): number {
    return computeFocalLoss(targets, outputs, focalGamma, focalAlpha);
  }

  /**
   * Calculates the Cross Entropy with Label Smoothing.
   * Label smoothing prevents the model from becoming overconfident by softening the targets.
   *
   * @see https://arxiv.org/abs/1512.00567
   * @param {number[]} targets - Array of target values (0 or 1 for binary, or probabilities for soft labels).
   * @param {number[]} outputs - Array of predicted probabilities (between 0 and 1).
   * @param {number} smoothingFactor - Smoothing factor (between 0 and 1, e.g., 0.1).
   * @returns {number} The mean cross-entropy loss with label smoothing.
   */
  static labelSmoothing(
    targets: number[],
    outputs: number[],
    smoothingFactor: number = DEFAULT_LABEL_SMOOTHING,
  ): number {
    return computeLabelSmoothingLoss(targets, outputs, smoothingFactor);
  }
}
