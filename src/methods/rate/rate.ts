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
 * Read this chapter as a tempo-control guide for training. The base learning
 * rate says how large a step feels reasonable at the start; the schedule says
 * how that step should change once the run has momentum, noise, or stagnation.
 *
 * The schedules fall into four practical families:
 *
 * - fixed and smooth decay schedules (`fixed()`, `exp()`, `inv()`) for simple
 *   long-run tempo control,
 * - piecewise schedules (`step()`, `linearWarmupDecay()`) for explicit phase
 *   changes,
 * - cyclic schedules (`cosineAnnealing()`, `cosineAnnealingWarmRestarts()`) for
 *   repeated exploration and settling,
 * - reactive schedules (`reduceOnPlateau()`) for runs that should respond to a
 *   monitored error signal.
 */
import {
  DEFAULT_COSINE_PERIOD,
  DEFAULT_DECAY_STEP_SIZE,
  DEFAULT_EXPONENTIAL_DECAY_FACTOR,
  DEFAULT_INITIAL_PERIOD,
  DEFAULT_INVERSE_DECAY_FACTOR,
  DEFAULT_INVERSE_POWER,
  DEFAULT_LINEAR_END_RATE,
  DEFAULT_MINIMUM_RATE,
  DEFAULT_PERIOD_GROWTH_MULTIPLIER,
  DEFAULT_STEP_DECAY_FACTOR,
  createCosineAnnealingRateSchedule,
  createCosineAnnealingWarmRestartsSchedule,
  createExponentialRateSchedule,
  createFixedRateSchedule,
  createInverseRateSchedule,
  createLinearWarmupDecaySchedule,
  createReduceOnPlateauSchedule,
  createStepRateSchedule,
} from './rate.utils';

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
 * Read this chapter as a tempo-control guide for training. The base learning
 * rate says how large a step feels reasonable at the start; the schedule says
 * how that step should change once the run has momentum, noise, or stagnation.
 *
 * The schedules fall into four practical families:
 *
 * - fixed and smooth decay schedules (`fixed()`, `exp()`, `inv()`) for simple
 *   long-run tempo control,
 * - piecewise schedules (`step()`, `linearWarmupDecay()`) for explicit phase
 *   changes,
 * - cyclic schedules (`cosineAnnealing()`, `cosineAnnealingWarmRestarts()`) for
 *   repeated exploration and settling,
 * - reactive schedules (`reduceOnPlateau()`) for runs that should respond to a
 *   monitored error signal.
 *
 * Read the chapter in that same order: smooth baselines first, planned phase
 * changes next, cyclic schedules after that, and stateful reactive control
 * last.
 *
 * ```mermaid
 * flowchart TD
 *   Base[Base learning rate] --> Smooth[Smooth decay family]
 *   Base --> Piecewise[Piecewise phase family]
 *   Base --> Cyclic[Cyclic family]
 *   Base --> Reactive[Reactive family]
 *   Smooth --> SmoothItems[fixed exp inv]
 *   Piecewise --> PieceItems[step linearWarmupDecay]
 *   Cyclic --> CyclicItems[cosineAnnealing warmRestarts]
 *   Reactive --> ReactiveItems[reduceOnPlateau]
 * ```
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
   * optimal for complex problems. Use it when you want the rest of the system,
   * not the schedule, to carry the full burden of training stability.
   *
   * @returns A function that takes the base learning rate and the current iteration number, and always returns the base learning rate.
   * @param baseRate The initial learning rate, which will remain constant.
   * @param iteration The current training iteration (unused in this method, but included for consistency).
   */
  static fixed(): (baseRate: number, iteration: number) => number {
    return createFixedRateSchedule();
  }

  /**
   * Implements a step decay learning rate schedule.
   *
   * The learning rate is reduced by a multiplicative factor (`decayFactor`)
   * at predefined intervals (`decayStepSize` iterations). This allows for
   * faster initial learning, followed by finer adjustments as training progresses.
   * It is a good fit when you want training to move through a few deliberate
   * phases rather than one perfectly smooth curve.
   *
   * Formula: `learning_rate = baseRate * decayFactor ^ floor(iteration / decayStepSize)`
   *
   * @param decayFactor The factor by which the learning rate is multiplied at each step. Should be less than 1. Defaults to 0.9.
   * @param decayStepSize The number of iterations after which the learning rate decays. Defaults to 100.
   * @returns A function that calculates the decayed learning rate for a given iteration.
   * @param baseRate The initial learning rate.
   * @param iteration The current training iteration.
   */
  static step(
    decayFactor: number = DEFAULT_STEP_DECAY_FACTOR,
    decayStepSize: number = DEFAULT_DECAY_STEP_SIZE,
  ): (baseRate: number, iteration: number) => number {
    return createStepRateSchedule(decayFactor, decayStepSize);
  }

  /**
   * Implements an exponential decay learning rate schedule.
   *
   * The learning rate decreases exponentially after each iteration, multiplying
   * by the decay factor `decayFactor`. This provides a smooth, continuous reduction
   * in the learning rate over time. Compared with step decay, the policy is less
   * about distinct phases and more about a steady fade in aggressiveness.
   *
   * Formula: `learning_rate = baseRate * decayFactor ^ iteration`
   *
   * @param decayFactor The decay factor applied at each iteration. Should be less than 1. Defaults to 0.999.
   * @returns A function that calculates the exponentially decayed learning rate for a given iteration.
   * @param baseRate The initial learning rate.
   * @param iteration The current training iteration.
   */
  static exp(
    decayFactor: number = DEFAULT_EXPONENTIAL_DECAY_FACTOR,
  ): (baseRate: number, iteration: number) => number {
    return createExponentialRateSchedule(decayFactor);
  }

  /**
   * Implements an inverse decay learning rate schedule.
   *
   * The learning rate decreases as the inverse of the iteration number,
   * controlled by the decay factor `decayFactor` and exponent `decayPower`. The rate
   * decreases more slowly over time compared to exponential decay. Use it when
   * you want long training runs to keep some learning energy instead of cooling
   * too quickly.
   *
   * Formula: `learning_rate = baseRate / (1 + decayFactor * iteration ** decayPower)`
   *
   * @param decayFactor Controls the rate of decay. Higher values lead to faster decay. Defaults to 0.001.
   * @param decayPower The exponent controlling the shape of the decay curve. Defaults to 2.
   * @returns A function that calculates the inversely decayed learning rate for a given iteration.
   * @param baseRate The initial learning rate.
   * @param iteration The current training iteration.
   */
  static inv(
    decayFactor: number = DEFAULT_INVERSE_DECAY_FACTOR,
    decayPower: number = DEFAULT_INVERSE_POWER,
  ): (baseRate: number, iteration: number) => number {
    return createInverseRateSchedule(decayFactor, decayPower);
  }

  /**
   * Implements a Cosine Annealing learning rate schedule.
   *
   * This schedule varies the learning rate cyclically according to a cosine function.
   * It starts at the `baseRate` and smoothly anneals down to `minimumRate` over a
   * specified `period` of iterations, then potentially repeats. This can help
   * the model escape local minima and explore the loss landscape more effectively.
   * Often used with "warm restarts" where the cycle repeats. The mental model is
   * deliberate breathing: ramp down to settle, then restart high enough to
   * explore again.
   *
   * Formula: `learning_rate = minimumRate + 0.5 * (baseRate - minimumRate) * (1 + cos(pi * current_cycle_iteration / period))`
   *
   * @param period The number of iterations over which the learning rate anneals from `baseRate` to `minimumRate` in one cycle. Defaults to 1000.
   * @param minimumRate The minimum learning rate value at the end of a cycle. Defaults to 0.
   * @returns A function that calculates the learning rate for a given iteration based on the cosine annealing schedule.
   * @param baseRate The initial (maximum) learning rate for the cycle.
   * @param iteration The current training iteration.
   * @see {@link https://arxiv.org/abs/1608.03983 SGDR: Stochastic Gradient Descent with Warm Restarts} - The paper introducing this technique.
   */
  static cosineAnnealing(
    period: number = DEFAULT_COSINE_PERIOD,
    minimumRate: number = DEFAULT_MINIMUM_RATE,
  ): (baseRate: number, iteration: number) => number {
    return createCosineAnnealingRateSchedule(period, minimumRate);
  }

  /**
   * Cosine Annealing with Warm Restarts (SGDR style) where the cycle length can grow by a multiplier after each restart.
   *
   * This variant keeps the exploratory reset behavior of cosine annealing while
   * allowing later cycles to last longer. That makes it useful when early
   * exploration should be frequent but later training should settle for longer
   * stretches between restarts.
   *
   * @param initialPeriod Length of the first cycle in iterations.
   * @param minimumRate Minimum learning rate at valley.
   * @param periodGrowthMultiplier Factor to multiply the period after each restart (>=1).
   * @returns A function that replays cosine cycles whose length can grow after each restart.
   */
  static cosineAnnealingWarmRestarts(
    initialPeriod: number = DEFAULT_INITIAL_PERIOD,
    minimumRate: number = DEFAULT_MINIMUM_RATE,
    periodGrowthMultiplier: number = DEFAULT_PERIOD_GROWTH_MULTIPLIER,
  ): (baseRate: number, iteration: number) => number {
    return createCosineAnnealingWarmRestartsSchedule(
      initialPeriod,
      minimumRate,
      periodGrowthMultiplier,
    );
  }

  /**
   * Linear Warmup followed by Linear Decay to an end rate.
   * Warmup linearly increases LR from near 0 up to baseRate over warmupStepCount, then linearly decays to endRate at totalStepCount.
   * Iterations beyond totalStepCount clamp to endRate.
   *
   * This schedule is common when the earliest steps are the most unstable: start
   * gentle, reach full speed, then taper predictably.
   *
   * @param totalStepCount Total steps for full schedule (must be > 0).
   * @param warmupStepCount Steps for warmup (< totalStepCount). Defaults to 10% of totalStepCount.
   * @param endRate Final rate at totalStepCount.
   * @returns A function that warms the learning rate up, then decays it toward a fixed floor.
   */
  static linearWarmupDecay(
    totalStepCount: number,
    warmupStepCount?: number,
    endRate: number = DEFAULT_LINEAR_END_RATE,
  ): (baseRate: number, iteration: number) => number {
    return createLinearWarmupDecaySchedule(
      totalStepCount,
      warmupStepCount,
      endRate,
    );
  }

  /**
   * ReduceLROnPlateau style scheduler (stateful closure) that monitors error signal (third argument if provided)
   * and reduces rate by 'factor' if no improvement beyond 'minDelta' for 'patience' iterations.
   * Cooldown prevents immediate successive reductions.
   * NOTE: Requires the training loop to call with signature (baseRate, iteration, lastError).
   *
   * This is the chapter's reactive option. Instead of following a pre-planned
   * calendar, the schedule listens for stalled improvement and responds only when
   * the run appears to flatten out.
   *
   * @param options Optional reactive-control settings such as patience, cooldown, and minimum rate floor.
   * @returns A stateful schedule function that may lower the learning rate when the monitored error stops improving.
   */
  static reduceOnPlateau(options?: {
    factor?: number; // multiplicative decrease (0<f<1)
    patience?: number; // iterations to wait for improvement
    minDelta?: number; // significant improvement threshold
    cooldown?: number; // iterations to wait after a reduction
    minRate?: number; // floor rate
    verbose?: boolean;
  }): (baseRate: number, iteration: number, lastError?: number) => number {
    return createReduceOnPlateauSchedule(options);
  }
}
