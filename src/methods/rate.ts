/**
 * Provides various methods for implementing learning rate schedules.
 *
 * Learning rate schedules dynamically adjust the learning rate during the training
 * process of machine learning models, particularly neural networks. Adjusting the
 * learning rate can significantly impact training speed and performance. A high
 * rate might lead to overshooting the optimal solution, while a very low rate
 * can result in slow convergence or getting stuck in local minima. These methods
 * offer different strategies to balance exploration and exploitation during training.
 *
 * @see {@link https://en.wikipedia.org/wiki/Learning_rate Learning Rate on Wikipedia}
 * @see {@link https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10 Understanding Learning Rates}
 */
export default class Rate {
  /**
   * Implements a fixed learning rate schedule.
   *
   * The learning rate remains constant throughout the entire training process.
   * This is the simplest schedule and serves as a baseline, but may not be
   * optimal for complex problems.
   *
   * @returns A function that takes the base learning rate and the current iteration number, and always returns the base learning rate.
   * @param baseRate The initial learning rate, which will remain constant.
   * @param iteration The current training iteration (unused in this method, but included for consistency).
   */
  static fixed(): (baseRate: number, iteration: number) => number {
    const func = (baseRate: number, iteration: number): number => {
      return baseRate;
    };

    return func;
  }

  /**
   * Implements a step decay learning rate schedule.
   *
   * The learning rate is reduced by a multiplicative factor (`gamma`)
   * at predefined intervals (`stepSize` iterations). This allows for
   * faster initial learning, followed by finer adjustments as training progresses.
   *
   * Formula: `learning_rate = baseRate * gamma ^ floor(iteration / stepSize)`
   *
   * @param gamma The factor by which the learning rate is multiplied at each step. Should be less than 1. Defaults to 0.9.
   * @param stepSize The number of iterations after which the learning rate decays. Defaults to 100.
   * @returns A function that calculates the decayed learning rate for a given iteration.
   * @param baseRate The initial learning rate.
   * @param iteration The current training iteration.
   */
  static step(
    gamma: number = 0.9,
    stepSize: number = 100
  ): (baseRate: number, iteration: number) => number {
    const func = (baseRate: number, iteration: number): number => {
      return Math.max(0, baseRate * Math.pow(gamma, Math.floor(iteration / stepSize)));
    };

    return func;
  }

  /**
   * Implements an exponential decay learning rate schedule.
   *
   * The learning rate decreases exponentially after each iteration, multiplying
   * by the decay factor `gamma`. This provides a smooth, continuous reduction
   * in the learning rate over time.
   *
   * Formula: `learning_rate = baseRate * gamma ^ iteration`
   *
   * @param gamma The decay factor applied at each iteration. Should be less than 1. Defaults to 0.999.
   * @returns A function that calculates the exponentially decayed learning rate for a given iteration.
   * @param baseRate The initial learning rate.
   * @param iteration The current training iteration.
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
   * Implements an inverse decay learning rate schedule.
   *
   * The learning rate decreases as the inverse of the iteration number,
   * controlled by the decay factor `gamma` and exponent `power`. The rate
   * decreases more slowly over time compared to exponential decay.
   *
   * Formula: `learning_rate = baseRate / (1 + gamma * Math.pow(iteration, power))`
   *
   * @param gamma Controls the rate of decay. Higher values lead to faster decay. Defaults to 0.001.
   * @param power The exponent controlling the shape of the decay curve. Defaults to 2.
   * @returns A function that calculates the inversely decayed learning rate for a given iteration.
   * @param baseRate The initial learning rate.
   * @param iteration The current training iteration.
   */
  static inv(
    gamma: number = 0.001,
    power: number = 2
  ): (baseRate: number, iteration: number) => number {
    const func = (baseRate: number, iteration: number): number => {
      // Use formula expected by tests: baseRate / (1 + gamma * Math.pow(iteration, power))
      return baseRate / (1 + gamma * Math.pow(iteration, power));
    };

    return func;
  }

  /**
   * Implements a Cosine Annealing learning rate schedule.
   *
   * This schedule varies the learning rate cyclically according to a cosine function.
   * It starts at the `baseRate` and smoothly anneals down to `minRate` over a
   * specified `period` of iterations, then potentially repeats. This can help
   * the model escape local minima and explore the loss landscape more effectively.
   * Often used with "warm restarts" where the cycle repeats.
   *
   * Formula: `learning_rate = minRate + 0.5 * (baseRate - minRate) * (1 + cos(pi * current_cycle_iteration / period))`
   *
   * @param period The number of iterations over which the learning rate anneals from `baseRate` to `minRate` in one cycle. Defaults to 1000.
   * @param minRate The minimum learning rate value at the end of a cycle. Defaults to 0.
   * @returns A function that calculates the learning rate for a given iteration based on the cosine annealing schedule.
   * @param baseRate The initial (maximum) learning rate for the cycle.
   * @param iteration The current training iteration.
   * @see {@link https://arxiv.org/abs/1608.03983 SGDR: Stochastic Gradient Descent with Warm Restarts} - The paper introducing this technique.
   */
  static cosineAnnealing(
    period: number = 1000,
    minRate: number = 0
  ): (baseRate: number, iteration: number) => number {
    const func = (baseRate: number, iteration: number): number => {
      // Calculate the current position within the cycle
      const currentCycleIteration = iteration % period;
      // Calculate the cosine decay factor (ranges from 1 down to 0)
      const cosineDecay =
        0.5 * (1 + Math.cos((currentCycleIteration / period) * Math.PI));
      // Apply the decay to the range between baseRate and minRate
      return minRate + (baseRate - minRate) * cosineDecay;
    };
    return func;
  }

  /**
   * Cosine Annealing with Warm Restarts (SGDR style) where the cycle length can grow by a multiplier (tMult) after each restart.
   *
   * @param initialPeriod Length of the first cycle in iterations.
   * @param minRate Minimum learning rate at valley.
   * @param tMult Factor to multiply the period after each restart (>=1).
   */
  static cosineAnnealingWarmRestarts(
    initialPeriod: number = 1000,
    minRate: number = 0,
    tMult: number = 1
  ): (baseRate: number, iteration: number) => number {
    let period = initialPeriod;
    let cycleStart = 0;
    let cycleEnd = period;
    return (baseRate: number, iteration: number): number => {
      // Advance cycles if iteration beyond current
      while (iteration >= cycleEnd) {
        cycleStart = cycleEnd;
        period = Math.max(1, Math.round(period * tMult));
        cycleEnd = cycleStart + period;
      }
      const cyclePos = iteration - cycleStart;
      const cosineDecay = 0.5 * (1 + Math.cos((cyclePos / period) * Math.PI));
      return minRate + (baseRate - minRate) * cosineDecay;
    };
  }

  /**
   * Linear Warmup followed by Linear Decay to an end rate.
   * Warmup linearly increases LR from near 0 up to baseRate over warmupSteps, then linearly decays to endRate at totalSteps.
   * Iterations beyond totalSteps clamp to endRate.
   *
   * @param totalSteps Total steps for full schedule (must be > 0).
   * @param warmupSteps Steps for warmup (< totalSteps). Defaults to 10% of totalSteps.
   * @param endRate Final rate at totalSteps.
   */
  static linearWarmupDecay(
    totalSteps: number,
    warmupSteps?: number,
    endRate: number = 0
  ): (baseRate: number, iteration: number) => number {
    if (totalSteps <= 0) throw new Error('totalSteps must be > 0');
    const warm = Math.min(warmupSteps ?? Math.max(1, Math.floor(totalSteps * 0.1)), totalSteps - 1);
    return (baseRate: number, iteration: number): number => {
      if (iteration <= warm) {
        return baseRate * (iteration / Math.max(1, warm));
      }
      if (iteration >= totalSteps) return endRate;
      const decaySteps = totalSteps - warm;
      const progress = (iteration - warm) / decaySteps; // 0..1
      return endRate + (baseRate - endRate) * (1 - progress);
    };
  }

  /**
   * ReduceLROnPlateau style scheduler (stateful closure) that monitors error signal (third argument if provided)
   * and reduces rate by 'factor' if no improvement beyond 'minDelta' for 'patience' iterations.
   * Cooldown prevents immediate successive reductions.
   * NOTE: Requires the training loop to call with signature (baseRate, iteration, lastError).
   */
  static reduceOnPlateau(options?: {
    factor?: number; // multiplicative decrease (0<f<1)
    patience?: number; // iterations to wait for improvement
    minDelta?: number; // significant improvement threshold
    cooldown?: number; // iterations to wait after a reduction
    minRate?: number; // floor rate
    verbose?: boolean;
  }): (baseRate: number, iteration: number, lastError?: number) => number {
    const {
      factor = 0.5,
      patience = 10,
      minDelta = 1e-4,
      cooldown = 0,
      minRate = 0,
      verbose = false
    } = options || {};
    let currentRate: number | undefined; // lazily initialize to baseRate first call
    let bestError: number | undefined;
    let lastImprovementIter = 0;
    let cooldownUntil = -1;
    return (baseRate: number, iteration: number, lastError?: number): number => {
      if (currentRate === undefined) currentRate = baseRate;
      if (lastError !== undefined) {
        if (bestError === undefined || lastError < bestError - minDelta) {
          bestError = lastError;
          lastImprovementIter = iteration;
        } else if (iteration - lastImprovementIter >= patience && iteration >= cooldownUntil) {
          const newRate = Math.max(minRate, currentRate * factor);
          if (newRate < currentRate) {
            if (verbose) console.log(`[reduceOnPlateau] Reducing LR from ${currentRate} to ${newRate} at iteration ${iteration}`);
            currentRate = newRate;
            cooldownUntil = iteration + cooldown;
            lastImprovementIter = iteration; // reset wait after reduction
          }
        }
      }
      return currentRate;
    };
  }
}
