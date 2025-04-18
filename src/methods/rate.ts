/**
 * Learning rate methods for adjusting the learning rate during training.
 * This is a TypeScript translation of the original JavaScript implementation.
 * Original JavaScript method names: FIXED, STEP, EXP, INV.
 * @see {@link https://stackoverflow.com/questions/30033096/what-is-lr-policy-in-caffe/30045244}
 */
export default class Rate {
  /**
   * Fixed learning rate.
   * The learning rate remains constant throughout training.
   * @returns {(baseRate: number, iteration: number) => number} A function that returns the base rate.
   */
  static fixed(): (baseRate: number, iteration: number) => number {
    const func = (baseRate: number, iteration: number): number => {
      return baseRate;
    };

    return func;
  }

  /**
   * Step decay learning rate.
   * The learning rate decreases by a factor of `gamma` every `stepSize` iterations.
   * @param {number} [gamma=0.9] - The decay factor. Defaults to 0.9.
   * @param {number} [stepSize=100] - The number of iterations after which the rate decays. Defaults to 100.
   * @returns {(baseRate: number, iteration: number) => number} A function that calculates the learning rate.
   */
  static step(
    gamma: number = 0.9,
    stepSize: number = 100
  ): (baseRate: number, iteration: number) => number {
    const func = (baseRate: number, iteration: number): number => {
      return baseRate * Math.pow(gamma, Math.floor(iteration / stepSize));
    };

    return func;
  }

  /**
   * Exponential decay learning rate.
   * The learning rate decreases exponentially with each iteration.
   * @param {number} [gamma=0.999] - The decay factor. Defaults to 0.999.
   * @returns {(baseRate: number, iteration: number) => number} A function that calculates the learning rate.
   */
  static exp(
    gamma: number = 0.999
  ): (baseRate: number, iteration: number) => number {
    const func = (baseRate: number, iteration: number): number => {
      return baseRate * Math.pow(gamma, iteration);
    };

    return func;
  }

  /**
   * Inverse decay learning rate.
   * The learning rate decreases inversely proportional to the iteration count.
   * @param {number} [gamma=0.001] - The decay factor. Defaults to 0.001.
   * @param {number} [power=2] - The power to which the iteration count is raised. Defaults to 2.
   * @returns {(baseRate: number, iteration: number) => number} A function that calculates the learning rate.
   */
  static inv(
    gamma: number = 0.001,
    power: number = 2
  ): (baseRate: number, iteration: number) => number {
    const func = (baseRate: number, iteration: number): number => {
      return baseRate * Math.pow(1 + gamma * iteration, -power);
    };

    return func;
  }
}
