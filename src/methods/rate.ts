/**
 * Learning rate methods for adjusting the learning rate during training.
 *
 * Learning rate schedules like fixed, step decay, and exponential decay
 * control the speed of convergence in optimization algorithms. Proper
 * adjustment prevents overshooting and ensures efficient training.
 *
 * @see {@link https://en.wikipedia.org/wiki/Learning_rate}
 * @see {@link https://en.wikipedia.org/wiki/Stochastic_gradient_descent}
 */
export default class Rate {
  /**
   * Fixed learning rate.
   * The learning rate remains constant throughout training.
   * @returns {(baseRate: number, iteration: number) => number} A function returning the base rate.
   */
  static fixed(): (baseRate: number, iteration: number) => number {
    const func = (baseRate: number, iteration: number): number => {
      return baseRate;
    };

    return func;
  }

  /**
   * Step decay learning rate.
   * Decreases the learning rate by a factor of `gamma` every `stepSize` iterations.
   * @param {number} [gamma=0.9] - Decay factor. Defaults to 0.9.
   * @param {number} [stepSize=100] - Iterations after which the rate decays. Defaults to 100.
   * @returns {(baseRate: number, iteration: number) => number} A function calculating the learning rate.
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
   * Decreases the learning rate exponentially with each iteration.
   * @param {number} [gamma=0.999] - Decay factor. Defaults to 0.999.
   * @returns {(baseRate: number, iteration: number) => number} A function calculating the learning rate.
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
   * Decreases the learning rate inversely proportional to the iteration count.
   * @param {number} [gamma=0.001] - Decay factor. Defaults to 0.001.
   * @param {number} [power=2] - Power to which the iteration count is raised. Defaults to 2.
   * @returns {(baseRate: number, iteration: number) => number} A function calculating the learning rate.
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
