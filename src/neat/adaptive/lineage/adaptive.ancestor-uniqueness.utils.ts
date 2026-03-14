import {
  ANCESTOR_UNIQ_MODE_EPSILON,
  ANCESTOR_UNIQ_MODE_LINEAGE_PRESSURE,
  DEFAULT_ANCESTOR_UNIQ_ADJUST,
  DEFAULT_ANCESTOR_UNIQ_COOLDOWN,
  DEFAULT_ANCESTOR_UNIQ_HIGH_THRESHOLD,
  DEFAULT_ANCESTOR_UNIQ_LOW_THRESHOLD,
  DEFAULT_LINEAGE_PRESSURE_STRENGTH,
  LINEAGE_PRESSURE_DECREASE_MULTIPLIER,
  LINEAGE_PRESSURE_INCREASE_MULTIPLIER,
  LINEAGE_PRESSURE_MODE_SPREAD,
  NEGATIVE_ONE,
  ZERO,
} from '../core/adaptive.core.constants';
import type {
  AncestorUniqAdaptiveConfig,
  NeatLikeWithAdaptive,
} from '../core/adaptive.core.types';

/**
 * Determine whether the cooldown window has elapsed.
 *
 * @param engine - NEAT engine instance.
 * @param config - Ancestor uniqueness adaptive configuration.
 * @returns True when adjustment is allowed.
 */
export function isCooldownSatisfied(
  engine: NeatLikeWithAdaptive,
  config: AncestorUniqAdaptiveConfig,
): boolean {
  const cooldown = config.cooldown ?? DEFAULT_ANCESTOR_UNIQ_COOLDOWN;
  const lastAdjustGeneration = engine._lastAncestorUniqAdjustGen ?? ZERO;

  return engine.generation - lastAdjustGeneration >= cooldown;
}

/**
 * Extract the latest ancestor-uniqueness metric from telemetry.
 *
 * @param engine - NEAT engine instance.
 * @returns Ancestor uniqueness value or undefined when missing.
 */
export function extractAncestorUniqueness(
  engine: NeatLikeWithAdaptive,
): number | undefined {
  const lineageTelemetry = engine._telemetry?.at(NEGATIVE_ONE)?.lineage;
  const ancestorUniqueness = lineageTelemetry?.ancestorUniq;

  return typeof ancestorUniqueness === 'number'
    ? ancestorUniqueness
    : undefined;
}

/**
 * Resolve thresholds for ancestor-uniqueness decisions.
 *
 * @param config - Ancestor uniqueness adaptive configuration.
 * @returns Threshold bounds.
 */
export function resolveUniquenessThresholds(
  config: AncestorUniqAdaptiveConfig,
): { lowThreshold: number; highThreshold: number } {
  const lowThreshold =
    config.lowThreshold ?? DEFAULT_ANCESTOR_UNIQ_LOW_THRESHOLD;
  const highThreshold =
    config.highThreshold ?? DEFAULT_ANCESTOR_UNIQ_HIGH_THRESHOLD;

  return { lowThreshold, highThreshold };
}

/**
 * Resolve adjustment magnitude for nudging controlled parameters.
 *
 * @param config - Ancestor uniqueness adaptive configuration.
 * @returns Adjustment magnitude.
 */
export function resolveAdjustmentMagnitude(
  config: AncestorUniqAdaptiveConfig,
): number {
  return config.adjust ?? DEFAULT_ANCESTOR_UNIQ_ADJUST;
}

/**
 * Apply an adjustment for the configured mode.
 *
 * @param engine - NEAT engine instance.
 * @param config - Ancestor uniqueness adaptive configuration.
 * @param ancestorUniq - Current ancestor uniqueness metric.
 * @param thresholds - Threshold bounds for decisions.
 * @param adjustMagnitude - Adjustment magnitude.
 * @returns {void}
 */
export function applyUniquenessAdjustment(
  engine: NeatLikeWithAdaptive,
  config: AncestorUniqAdaptiveConfig,
  ancestorUniq: number,
  thresholds: { lowThreshold: number; highThreshold: number },
  adjustMagnitude: number,
): void {
  if (config.mode === ANCESTOR_UNIQ_MODE_EPSILON) {
    applyEpsilonAdjustment(engine, ancestorUniq, thresholds, adjustMagnitude);

    return;
  }

  if (config.mode === ANCESTOR_UNIQ_MODE_LINEAGE_PRESSURE) {
    applyLineagePressureAdjustment(engine, ancestorUniq, thresholds);
  }
}

/**
 * Apply dominance-epsilon adjustments when configured.
 *
 * @param engine - NEAT engine instance.
 * @param ancestorUniq - Current ancestor uniqueness metric.
 * @param thresholds - Threshold bounds for decisions.
 * @param adjustMagnitude - Adjustment magnitude.
 * @returns {void}
 */
export function applyEpsilonAdjustment(
  engine: NeatLikeWithAdaptive,
  ancestorUniq: number,
  thresholds: { lowThreshold: number; highThreshold: number },
  adjustMagnitude: number,
): void {
  if (!engine.options.multiObjective?.adaptiveEpsilon?.enabled) return;

  const shouldIncrease = ancestorUniq < thresholds.lowThreshold;
  const shouldDecrease = ancestorUniq > thresholds.highThreshold;
  if (!shouldIncrease && !shouldDecrease) return;

  const currentEpsilon = engine.options.multiObjective.dominanceEpsilon ?? ZERO;
  engine.options.multiObjective.dominanceEpsilon = shouldIncrease
    ? currentEpsilon + adjustMagnitude
    : Math.max(ZERO, currentEpsilon - adjustMagnitude);

  recordAdjustment(engine);
}

/**
 * Apply lineage pressure strength adjustments.
 *
 * @param engine - NEAT engine instance.
 * @param ancestorUniq - Current ancestor uniqueness metric.
 * @param thresholds - Threshold bounds for decisions.
 * @returns {void}
 */
export function applyLineagePressureAdjustment(
  engine: NeatLikeWithAdaptive,
  ancestorUniq: number,
  thresholds: { lowThreshold: number; highThreshold: number },
): void {
  const lineagePressureState = ensureLineagePressureState(engine);

  const shouldIncrease = ancestorUniq < thresholds.lowThreshold;
  const shouldDecrease = ancestorUniq > thresholds.highThreshold;
  if (!shouldIncrease && !shouldDecrease) return;

  const currentStrength =
    lineagePressureState.strength ?? DEFAULT_LINEAGE_PRESSURE_STRENGTH;
  lineagePressureState.strength = shouldIncrease
    ? currentStrength * LINEAGE_PRESSURE_INCREASE_MULTIPLIER
    : currentStrength * LINEAGE_PRESSURE_DECREASE_MULTIPLIER;

  if (shouldIncrease) lineagePressureState.mode = LINEAGE_PRESSURE_MODE_SPREAD;

  recordAdjustment(engine);
}

/**
 * Ensure lineage pressure state is available.
 *
 * @param engine - NEAT engine instance.
 * @returns Lineage pressure configuration object.
 */
export function ensureLineagePressureState(
  engine: NeatLikeWithAdaptive,
): NonNullable<NeatLikeWithAdaptive['options']['lineagePressure']> {
  if (!engine.options.lineagePressure) {
    engine.options.lineagePressure = {
      enabled: true,
      mode: LINEAGE_PRESSURE_MODE_SPREAD,
      strength: DEFAULT_LINEAGE_PRESSURE_STRENGTH,
    };
  }

  return engine.options.lineagePressure;
}

/**
 * Record the generation when an adjustment is applied.
 *
 * @param engine - NEAT engine instance.
 * @returns {void}
 */
export function recordAdjustment(engine: NeatLikeWithAdaptive): void {
  engine._lastAncestorUniqAdjustGen = engine.generation;
}