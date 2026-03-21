/**
 * Learning rate schedule signature that maps a base rate and iteration index to a rate value.
 * Useful for any stateless schedule strategy.
 */
export type RateSchedule = (baseRate: number, iteration: number) => number;

/**
 * Stateful ReduceLROnPlateau schedule signature that can react to a loss signal.
 * The third argument is optional and only needed when monitoring validation error.
 */
export type ReduceOnPlateauSchedule = (
  baseRate: number,
  iteration: number,
  lastError?: number,
) => number;

/**
 * Step decay multiplier (close to 1 slows decay; smaller drops faster).
 */
export const DEFAULT_STEP_DECAY_FACTOR = 0.9;

/**
 * Step decay interval in iterations; larger values mean fewer decay events.
 */
export const DEFAULT_DECAY_STEP_SIZE = 100;

/**
 * Per-iteration exponential decay factor; values just below 1 create gentle decay.
 */
export const DEFAULT_EXPONENTIAL_DECAY_FACTOR = 0.999;

/**
 * Inverse decay multiplier; higher values push the denominator up faster and shrink the rate sooner.
 */
export const DEFAULT_INVERSE_DECAY_FACTOR = 0.001;

/**
 * Inverse decay exponent; 1 makes decay linear in iteration, 2 makes it quadratic.
 */
export const DEFAULT_INVERSE_POWER = 2;

/**
 * Length of one cosine annealing cycle in iterations.
 */
export const DEFAULT_COSINE_PERIOD = 1000;

/**
 * Floor learning rate for cosine schedules; keeps the rate from reaching zero.
 */
export const DEFAULT_MINIMUM_RATE = 0;

/**
 * Initial period length for cosine-with-restarts before growth is applied.
 */
export const DEFAULT_INITIAL_PERIOD = 1000;

/**
 * Multiplier applied to the cosine cycle length after each restart (>= 1).
 */
export const DEFAULT_PERIOD_GROWTH_MULTIPLIER = 1;

/**
 * Target rate after warmup-decay finishes; often zero or a small floor.
 */
export const DEFAULT_LINEAR_END_RATE = 0;

/**
 * Default warmup share of the schedule; 0.1 means 10% of total steps.
 */
export const DEFAULT_WARMUP_RATIO = 0.1;

/**
 * Reduce-on-plateau shrink factor; halving (0.5) is a common conservative step.
 */
export const DEFAULT_REDUCE_ON_PLATEAU_FACTOR = 0.5;

/**
 * Patience for reduce-on-plateau in iterations before triggering a cut.
 */
export const DEFAULT_REDUCE_ON_PLATEAU_PATIENCE = 10;

/**
 * Minimum required improvement to count as progress when monitoring error.
 */
export const DEFAULT_REDUCE_ON_PLATEAU_MIN_DELTA = 0.0001;

/**
 * Cooldown iterations after a reduction to avoid rapid successive cuts.
 */
export const DEFAULT_REDUCE_ON_PLATEAU_COOLDOWN = 0;

/**
 * Minimum rate allowed during reduce-on-plateau adjustments.
 */
export const DEFAULT_REDUCE_ON_PLATEAU_MIN_RATE = 0;

/**
 * Safety floor to prevent negative learning rates when applying decay.
 */
const ZERO_RATE_FLOOR = 0;

/**
 * Cosine amplitude scaling so the decay oscillates between 0 and 1.
 */
const COSINE_DECAY_SCALE = 0.5;

/**
 * Phase offset that shifts cosine from [-1, 1] to [0, 2] before scaling.
 */
const COSINE_PHASE_OFFSET = 1;

/**
 * Denominator offset in inverse decay to keep the initial rate unchanged at iteration 0.
 */
const INVERSE_DENOMINATOR_OFFSET = 1;

/**
 * Smallest allowed cosine period when warm restarts shrink aggressively.
 */
const MINIMUM_PERIOD_LENGTH = 1;

/**
 * Lower bound for warmup steps to avoid division by zero during ramp-up.
 */
const MINIMUM_WARMUP_STEPS = 1;

/**
 * Minimum remaining steps for decay so schedules always have a decay phase.
 */
const MINIMUM_DECAY_STEPS = 1;

/**
 * Sentinel for cooldown tracking before any reductions occur.
 */
const INITIAL_COOLDOWN_SENTINEL = -1;

/**
 * Multiplier representing full weight (100%) in linear decay math.
 */
const FULL_WEIGHT = 1;

/**
 * Initial iteration index used to seed improvement tracking in plateau reduction.
 */
const INITIAL_IMPROVEMENT_ITERATION = 0;

/**
 * Error message used when schedule setup receives a non-positive total step count.
 */
const TOTAL_STEPS_ERROR_MESSAGE = 'totalSteps must be > 0';

/**
 * Returns a schedule that always yields the base learning rate.
 *
 * @returns A learning rate schedule that ignores iteration and returns baseRate.
 */
export function createFixedRateSchedule(): RateSchedule {
  return (baseRate: number): number => {
    return baseRate;
  };
}

/**
 * Returns a step decay learning rate schedule.
 *
 * @param decayFactor Multiplicative decay applied at each decay step.
 * @param decayStepSize Number of iterations before applying another decay step.
 * @returns A learning rate schedule implementing step decay.
 */
export function createStepRateSchedule(
  decayFactor: number = DEFAULT_STEP_DECAY_FACTOR,
  decayStepSize: number = DEFAULT_DECAY_STEP_SIZE,
): RateSchedule {
  return (baseRate: number, iteration: number): number => {
    const decayStepsElapsed = Math.floor(iteration / decayStepSize);
    const decayedRate = baseRate * decayFactor ** decayStepsElapsed;
    return Math.max(ZERO_RATE_FLOOR, decayedRate);
  };
}

/**
 * Returns an exponential decay learning rate schedule.
 *
 * @param decayFactor Multiplicative decay applied every iteration.
 * @returns A learning rate schedule implementing exponential decay.
 */
export function createExponentialRateSchedule(
  decayFactor: number = DEFAULT_EXPONENTIAL_DECAY_FACTOR,
): RateSchedule {
  return (baseRate: number, iteration: number): number => {
    return baseRate * decayFactor ** iteration;
  };
}

/**
 * Returns an inverse decay learning rate schedule.
 *
 * @param decayFactor Decay factor controlling the decay rate.
 * @param decayPower Exponent that shapes the decay curve.
 * @returns A learning rate schedule implementing inverse decay.
 */
export function createInverseRateSchedule(
  decayFactor: number = DEFAULT_INVERSE_DECAY_FACTOR,
  decayPower: number = DEFAULT_INVERSE_POWER,
): RateSchedule {
  return (baseRate: number, iteration: number): number => {
    const denominator =
      INVERSE_DENOMINATOR_OFFSET + decayFactor * iteration ** decayPower;
    return baseRate / denominator;
  };
}

/**
 * Returns a cosine annealing learning rate schedule.
 *
 * @param period Length of a full cosine cycle.
 * @param minimumRate Minimum rate reached at the end of a cycle.
 * @returns A learning rate schedule implementing cosine annealing.
 */
export function createCosineAnnealingRateSchedule(
  period: number = DEFAULT_COSINE_PERIOD,
  minimumRate: number = DEFAULT_MINIMUM_RATE,
): RateSchedule {
  return (baseRate: number, iteration: number): number => {
    const currentCycleIteration = iteration % period;
    const cosineDecay =
      COSINE_DECAY_SCALE *
      (COSINE_PHASE_OFFSET +
        Math.cos((currentCycleIteration / period) * Math.PI));
    return minimumRate + (baseRate - minimumRate) * cosineDecay;
  };
}

/**
 * Returns a cosine annealing schedule with warm restarts and growing cycles.
 *
 * @param initialPeriod Length of the initial cycle.
 * @param minimumRate Minimum learning rate reached at the end of each cycle.
 * @param periodGrowthMultiplier Multiplier applied to the period after each restart.
 * @returns A learning rate schedule implementing SGDR-style warm restarts.
 */
export function createCosineAnnealingWarmRestartsSchedule(
  initialPeriod: number = DEFAULT_INITIAL_PERIOD,
  minimumRate: number = DEFAULT_MINIMUM_RATE,
  periodGrowthMultiplier: number = DEFAULT_PERIOD_GROWTH_MULTIPLIER,
): RateSchedule {
  let currentPeriod = initialPeriod;
  let cycleStartIteration = 0;
  let cycleEndIteration = currentPeriod;

  return (baseRate: number, iteration: number): number => {
    while (iteration >= cycleEndIteration) {
      cycleStartIteration = cycleEndIteration;
      currentPeriod = Math.max(
        MINIMUM_PERIOD_LENGTH,
        Math.round(currentPeriod * periodGrowthMultiplier),
      );
      cycleEndIteration = cycleStartIteration + currentPeriod;
    }

    const cycleRelativeIteration = iteration - cycleStartIteration;
    const cosineDecay =
      COSINE_DECAY_SCALE *
      (COSINE_PHASE_OFFSET +
        Math.cos((cycleRelativeIteration / currentPeriod) * Math.PI));

    return minimumRate + (baseRate - minimumRate) * cosineDecay;
  };
}

/**
 * Returns a linear warmup followed by linear decay schedule.
 *
 * @param totalStepCount Total number of steps in the schedule (must be positive).
 * @param warmupStepCount Optional number of warmup steps; defaults to 10% of total steps.
 * @param endRate Final rate once decay completes.
 * @returns A learning rate schedule implementing warmup then decay.
 */
export function createLinearWarmupDecaySchedule(
  totalStepCount: number,
  warmupStepCount?: number,
  endRate: number = DEFAULT_LINEAR_END_RATE,
): RateSchedule {
  if (totalStepCount <= ZERO_RATE_FLOOR)
    throw new Error(TOTAL_STEPS_ERROR_MESSAGE);

  const maximumWarmupSteps = totalStepCount - MINIMUM_DECAY_STEPS;
  const resolvedWarmupSteps = Math.min(
    warmupStepCount ??
      Math.max(
        MINIMUM_WARMUP_STEPS,
        Math.floor(totalStepCount * DEFAULT_WARMUP_RATIO),
      ),
    maximumWarmupSteps,
  );

  return (baseRate: number, iteration: number): number => {
    if (iteration <= resolvedWarmupSteps) {
      const warmupDenominator = Math.max(
        MINIMUM_WARMUP_STEPS,
        resolvedWarmupSteps,
      );
      return baseRate * (iteration / warmupDenominator);
    }

    if (iteration >= totalStepCount) return endRate;

    const decayStepCount = totalStepCount - resolvedWarmupSteps;
    const decayProgress = (iteration - resolvedWarmupSteps) / decayStepCount;
    return endRate + (baseRate - endRate) * (FULL_WEIGHT - decayProgress);
  };
}

/**
 * Returns a ReduceLROnPlateau-style schedule that lowers the rate when no improvement is seen.
 *
 * @param options Optional configuration for factor, patience, minDelta, cooldown, and minimum rate.
 * @returns A stateful schedule that reacts to lack of improvement.
 */
export function createReduceOnPlateauSchedule(options?: {
  factor?: number;
  patience?: number;
  minDelta?: number;
  cooldown?: number;
  minRate?: number;
  verbose?: boolean;
}): ReduceOnPlateauSchedule {
  const {
    factor = DEFAULT_REDUCE_ON_PLATEAU_FACTOR,
    patience = DEFAULT_REDUCE_ON_PLATEAU_PATIENCE,
    minDelta = DEFAULT_REDUCE_ON_PLATEAU_MIN_DELTA,
    cooldown = DEFAULT_REDUCE_ON_PLATEAU_COOLDOWN,
    minRate = DEFAULT_REDUCE_ON_PLATEAU_MIN_RATE,
    verbose: verboseLoggingEnabled = false,
  } = options ?? {};

  let currentLearningRate: number | undefined;
  let bestErrorValue: number | undefined;
  let lastImprovementIteration = INITIAL_IMPROVEMENT_ITERATION;
  let cooldownReleaseIteration = INITIAL_COOLDOWN_SENTINEL;

  return (baseRate: number, iteration: number, lastError?: number): number => {
    if (currentLearningRate === undefined) currentLearningRate = baseRate;

    if (lastError !== undefined) {
      if (
        bestErrorValue === undefined ||
        lastError < bestErrorValue - minDelta
      ) {
        bestErrorValue = lastError;
        lastImprovementIteration = iteration;
      } else if (
        iteration - lastImprovementIteration >= patience &&
        iteration >= cooldownReleaseIteration
      ) {
        const candidateRate = Math.max(minRate, currentLearningRate * factor);
        if (candidateRate < currentLearningRate) {
          currentLearningRate = candidateRate;
          cooldownReleaseIteration = iteration + cooldown;
          lastImprovementIteration = iteration;
        }
      }
    }

    if (verboseLoggingEnabled) {
      // Placeholder to acknowledge the verbose flag without altering behavior.
    }

    return currentLearningRate;
  };
}
