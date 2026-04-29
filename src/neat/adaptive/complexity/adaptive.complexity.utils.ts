import { EPSILON } from '../../neat.constants';
import {
  BUDGET_GROWTH_MULTIPLIER,
  COMPLEXITY_MODE_ADAPTIVE,
  DEFAULT_CB_INCREASE_FACTOR,
  DEFAULT_CB_STAGNATION_FACTOR,
  DEFAULT_IMPROVEMENT_WINDOW,
  HISTORY_MIN_IMPROVEMENT_COUNT,
  HISTORY_MIN_SLOPE_COUNT,
  LINEAR_HORIZON_DEFAULT,
  MINIMAL_TOPOLOGY_OFFSET,
  NEGATIVE_ONE,
  NOVELTY_ARCHIVE_MIN_SIZE,
  NOVELTY_FACTOR_DEFAULT,
  NOVELTY_FACTOR_SMALL,
  PROGRESS_RATIO_MAX,
  SLOPE_BOOST_MULTIPLIER,
  SLOPE_NORMALIZE_CLAMP,
  SLOPE_PENALTY_MULTIPLIER,
  ZERO,
} from '../core/adaptive.core.constants';
import type {
  ComplexityBudgetConfig,
  NeatLikeWithAdaptive,
} from '../core/adaptive.core.types';

/**
 * Schedule helpers for adaptive complexity budgets.
 *
 * This file owns the trend-driven and linear rules that convert recent run
 * evidence into controller-level node and connection caps. It stays separate
 * from the phase helpers because schedule logic answers "how large may the
 * topology grow?" while phase logic answers "which structural mood is active?"
 *
 * The schedule pipeline is intentionally compact:
 *
 * 1. record recent best-score history,
 * 2. derive trend and novelty signals,
 * 3. resolve adaptive or linear budget updates,
 * 4. clamp the result back onto controller state.
 */

/* Module introduction boundary for generated README output. */

/**
 * Apply the complexity budget schedule for the configured mode.
 *
 * This is the main dispatcher for the complexity subtree. It keeps the public
 * entrypoint easy to scan by delegating immediately into either the adaptive
 * schedule or the linear schedule, depending on the configured policy.
 *
 * @param engine - NEAT engine instance.
 * @param config - Complexity budget configuration.
 * @returns Nothing.
 */
export function applyComplexityBudgetSchedule(
  engine: NeatLikeWithAdaptive,
  config: ComplexityBudgetConfig,
): void {
  if (config.mode === COMPLEXITY_MODE_ADAPTIVE) {
    applyAdaptiveSchedule(engine, config);
    return;
  }

  applyLinearSchedule(engine, config);
}

/**
 * Apply adaptive complexity budget scheduling.
 *
 * The adaptive schedule is the evidence-driven branch. It watches recent best
 * scores, slope, and novelty pressure, then expands or contracts structural
 * budgets so later mutation passes can respond to genuine search progress
 * rather than following a fixed calendar.
 *
 * @param engine - NEAT engine instance with adaptive state.
 * @param config - Complexity budget configuration.
 * @returns Nothing.
 */
export function applyAdaptiveSchedule(
  engine: NeatLikeWithAdaptive,
  config: ComplexityBudgetConfig,
): void {
  const history = updateScoreHistory(engine, config);
  const trends = computeTrends(history);
  const factors = computeAdjustmentFactors(config, trends, history);
  const noveltyFactor = computeNoveltyFactor(engine);

  initializeNodeBudget(engine, config);
  adjustNodeBudget(engine, config, trends, factors, noveltyFactor, history);
  clampNodeBudget(engine, config);
  engine.options.maxNodes = engine._cbMaxNodes;

  if (config.maxConnsStart) {
    initializeConnectionBudget(engine, config);
    adjustConnectionBudget(
      engine,
      config,
      trends,
      factors,
      noveltyFactor,
      history,
    );
    engine.options.maxConns = engine._cbMaxConns;
  }
}

/**
 * Apply linear complexity budget scheduling.
 *
 * The linear schedule is the deterministic counterpart to the adaptive mode.
 * It ignores run-time improvement signals and simply interpolates between a
 * start and end budget across a configured horizon.
 *
 * @param engine - NEAT engine instance.
 * @param config - Complexity budget configuration.
 * @returns Nothing.
 */
export function applyLinearSchedule(
  engine: NeatLikeWithAdaptive,
  config: ComplexityBudgetConfig,
): void {
  const minimalTopology =
    engine.input + engine.output + MINIMAL_TOPOLOGY_OFFSET;
  const startBudget = config.maxNodesStart ?? minimalTopology;
  const endBudget =
    config.maxNodesEnd ?? startBudget * BUDGET_GROWTH_MULTIPLIER;
  const horizonGens = config.horizon ?? LINEAR_HORIZON_DEFAULT;
  const progress = Math.min(
    PROGRESS_RATIO_MAX,
    engine.generation / horizonGens,
  );

  engine.options.maxNodes = Math.floor(
    startBudget + (endBudget - startBudget) * progress,
  );
}

/**
 * Update rolling score history with current best score.
 *
 * Score history is the minimum evidence the adaptive scheduler needs in order
 * to talk about improvement or stagnation. The helper keeps that history bounded
 * to the configured window so later slope and delta calculations stay local to
 * recent generations.
 *
 * @param engine - NEAT engine instance.
 * @param config - Complexity budget configuration.
 * @returns Rolling history array after update.
 */
export function updateScoreHistory(
  engine: NeatLikeWithAdaptive,
  config: ComplexityBudgetConfig,
): number[] {
  if (!engine._cbHistory) engine._cbHistory = [];

  const currentBestScore = engine.population[ZERO]?.score ?? ZERO;
  engine._cbHistory.push(currentBestScore);

  const windowSize = config.improvementWindow ?? DEFAULT_IMPROVEMENT_WINDOW;
  if (engine._cbHistory.length > windowSize) {
    engine._cbHistory.shift();
  }

  return engine._cbHistory;
}

/**
 * Compute improvement and slope trends from score history.
 *
 * Improvement answers whether the window ended above where it started; slope
 * answers how strongly the trajectory points upward or downward across the
 * window as a whole. The adaptive scheduler uses both so it can distinguish a
 * noisy plateau from sustained progress.
 *
 * @param history - Rolling history of best scores.
 * @returns Trend metrics (improvement and slope).
 */
export function computeTrends(history: number[]): {
  improvement: number;
  slope: number;
} {
  const improvement =
    history.length >= HISTORY_MIN_IMPROVEMENT_COUNT
      ? history.at(NEGATIVE_ONE)! - history[ZERO]
      : ZERO;
  const slope =
    history.length >= HISTORY_MIN_SLOPE_COUNT ? computeSlope(history) : ZERO;
  return { improvement, slope };
}

/**
 * Compute linear regression slope using ordinary least squares.
 *
 * @param history - Rolling history of best scores.
 * @returns OLS slope estimate.
 */
export function computeSlope(history: number[]): number {
  const count = history.length;
  let sumIndices = ZERO;
  let sumScores = ZERO;
  let sumIndexScore = ZERO;
  let sumIndexSquared = ZERO;

  for (let index = ZERO; index < count; index++) {
    sumIndices += index;
    sumScores += history[index];
    sumIndexScore += index * history[index];
    sumIndexSquared += index * index;
  }

  const denominator = count * sumIndexSquared - sumIndices * sumIndices;
  return (count * sumIndexScore - sumIndices * sumScores) / denominator;
}

/**
 * Compute adjustment factors for budget growth and decay.
 *
 * These factors translate raw trend signals into multiplicative budget updates.
 * Positive normalized slope boosts growth pressure, while negative slope makes
 * stagnation shrinkage more aggressive.
 *
 * @param config - Complexity budget configuration.
 * @param trends - Improvement and slope metrics.
 * @param history - Rolling history of best scores.
 * @returns Adjustment factors (increase and stagnation multipliers).
 */
export function computeAdjustmentFactors(
  config: ComplexityBudgetConfig,
  trends: { improvement: number; slope: number },
  history: number[],
): { increaseFactor: number; stagnationFactor: number } {
  const baseIncrease = config.increaseFactor ?? DEFAULT_CB_INCREASE_FACTOR;
  const baseStagnation =
    config.stagnationFactor ?? DEFAULT_CB_STAGNATION_FACTOR;
  const normalizedSlope = normalizeSlope(trends.slope, history[ZERO]);

  const increaseFactor =
    baseIncrease + SLOPE_BOOST_MULTIPLIER * Math.max(ZERO, normalizedSlope);
  const stagnationFactor =
    baseStagnation -
    SLOPE_PENALTY_MULTIPLIER * Math.max(ZERO, -normalizedSlope);

  return { increaseFactor, stagnationFactor };
}

/**
 * Normalize slope magnitude relative to initial score.
 *
 * @param slope - Raw OLS slope.
 * @param initialScore - First score in history window.
 * @returns Normalized slope clamped to [-2, 2].
 */
export function normalizeSlope(slope: number, initialScore: number): number {
  return Math.min(
    SLOPE_NORMALIZE_CLAMP,
    Math.max(
      -SLOPE_NORMALIZE_CLAMP,
      slope / (Math.abs(initialScore) + EPSILON),
    ),
  );
}

/**
 * Compute novelty factor based on archive size.
 *
 * Novelty acts as a small confidence signal for structural growth. A larger
 * novelty archive implies the search is still exploring enough distinct
 * behavior to justify the default growth multiplier.
 *
 * @param engine - NEAT engine instance.
 * @returns Novelty multiplier (0.9 if archive small, 1.0 otherwise).
 */
export function computeNoveltyFactor(engine: NeatLikeWithAdaptive): number {
  return (engine._noveltyArchive?.length ?? ZERO) > NOVELTY_ARCHIVE_MIN_SIZE
    ? NOVELTY_FACTOR_DEFAULT
    : NOVELTY_FACTOR_SMALL;
}

/**
 * Initialize node budget if undefined.
 *
 * @param engine - NEAT engine instance.
 * @param config - Complexity budget configuration.
 * @returns {void}
 */
export function initializeNodeBudget(
  engine: NeatLikeWithAdaptive,
  config: ComplexityBudgetConfig,
): void {
  if (engine._cbMaxNodes === undefined) {
    const minimalTopology =
      engine.input + engine.output + MINIMAL_TOPOLOGY_OFFSET;
    engine._cbMaxNodes = config.maxNodesStart ?? minimalTopology;
  }
}

/**
 * Adjust node budget based on trends and factors.
 *
 * Node-budget adjustment is where the adaptive policy becomes concrete. When
 * improvement or positive slope is present, the helper expands the budget up to
 * the configured ceiling; when the observation window is full and the search is
 * flat, it contracts back toward the configured minimum.
 *
 * @param engine - NEAT engine instance.
 * @param config - Complexity budget configuration.
 * @param trends - Improvement and slope metrics.
 * @param factors - Adjustment factors.
 * @param noveltyFactor - Novelty multiplier.
 * @param history - Rolling history for window checks.
 * @returns {void}
 */
export function adjustNodeBudget(
  engine: NeatLikeWithAdaptive,
  config: ComplexityBudgetConfig,
  trends: { improvement: number; slope: number },
  factors: { increaseFactor: number; stagnationFactor: number },
  noveltyFactor: number,
  history: number[],
): void {
  const windowSize = config.improvementWindow ?? DEFAULT_IMPROVEMENT_WINDOW;
  const isImproving = trends.improvement > ZERO || trends.slope > ZERO;
  const isWindowFull = history.length === windowSize;

  if (isImproving) {
    const maxCap =
      config.maxNodesEnd ?? engine._cbMaxNodes! * BUDGET_GROWTH_MULTIPLIER;
    const proposed = Math.floor(
      engine._cbMaxNodes! * factors.increaseFactor * noveltyFactor,
    );
    engine._cbMaxNodes = Math.min(maxCap, proposed);
  } else if (isWindowFull) {
    const minimalTopology =
      engine.input + engine.output + MINIMAL_TOPOLOGY_OFFSET;
    const minCap = config.minNodes ?? minimalTopology;
    const proposed = Math.floor(engine._cbMaxNodes! * factors.stagnationFactor);
    engine._cbMaxNodes = Math.max(minCap, proposed);
  }
}

/**
 * Clamp node budget to configured minimum.
 *
 * This final guard keeps the adaptive loop from shrinking below the smallest
 * topology the controller can reasonably support.
 *
 * @param engine - NEAT engine instance.
 * @param config - Complexity budget configuration.
 * @returns Nothing.
 */
export function clampNodeBudget(
  engine: NeatLikeWithAdaptive,
  config: ComplexityBudgetConfig,
): void {
  const minimalTopology =
    engine.input + engine.output + MINIMAL_TOPOLOGY_OFFSET;
  const minAllowed = config.minNodes ?? minimalTopology;
  engine._cbMaxNodes = Math.max(minAllowed, engine._cbMaxNodes!);
}

/**
 * Initialize connection budget if undefined.
 *
 * Connection budgets are optional, so the helper only seeds this state when the
 * configuration explicitly opts into a connection-cap schedule.
 *
 * @param engine - NEAT engine instance.
 * @param config - Complexity budget configuration.
 * @returns Nothing.
 */
export function initializeConnectionBudget(
  engine: NeatLikeWithAdaptive,
  config: ComplexityBudgetConfig,
): void {
  if (engine._cbMaxConns === undefined) {
    engine._cbMaxConns = config.maxConnsStart!;
  }
}

/**
 * Adjust connection budget based on trends and factors.
 *
 * Connection-budget adjustment mirrors the node-budget path so both structural
 * ceilings respond coherently to the same improvement and stagnation signals.
 *
 * @param engine - NEAT engine instance.
 * @param config - Complexity budget configuration.
 * @param trends - Improvement and slope metrics.
 * @param factors - Adjustment factors.
 * @param noveltyFactor - Novelty multiplier.
 * @param history - Rolling history for window checks.
 * @returns Nothing.
 */
export function adjustConnectionBudget(
  engine: NeatLikeWithAdaptive,
  config: ComplexityBudgetConfig,
  trends: { improvement: number; slope: number },
  factors: { increaseFactor: number; stagnationFactor: number },
  noveltyFactor: number,
  history: number[],
): void {
  const windowSize = config.improvementWindow ?? DEFAULT_IMPROVEMENT_WINDOW;
  const isImproving = trends.improvement > ZERO || trends.slope > ZERO;
  const isWindowFull = history.length === windowSize;

  if (isImproving) {
    const maxCap =
      config.maxConnsEnd ?? engine._cbMaxConns! * BUDGET_GROWTH_MULTIPLIER;
    const proposed = Math.floor(
      engine._cbMaxConns! * factors.increaseFactor * noveltyFactor,
    );
    engine._cbMaxConns = Math.min(maxCap, proposed);
  } else if (isWindowFull) {
    const minCap = config.maxConnsStart!;
    const proposed = Math.floor(engine._cbMaxConns! * factors.stagnationFactor);
    engine._cbMaxConns = Math.max(minCap, proposed);
  }
}
