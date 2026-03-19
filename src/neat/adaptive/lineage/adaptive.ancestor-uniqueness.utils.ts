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
 * Ancestor-uniqueness helpers for adaptive lineage feedback.
 *
 * This file owns the small telemetry-to-policy loop behind lineage adaptation.
 * It stays separate from the root controller entrypoint so the generated docs
 * can explain how evidence is extracted, gated, interpreted, and translated
 * into one of two policy nudges.
 *
 * The helper flow is intentionally compact:
 *
 * 1. verify the cooldown window has elapsed,
 * 2. extract the most recent ancestor-uniqueness metric,
 * 3. resolve thresholds and adjustment magnitude,
 * 4. route the adjustment into epsilon or lineage-pressure mode.
 */

/* Module introduction boundary for generated README output. */

/**
 * Determine whether the cooldown window has elapsed.
 *
 * Cooldowns prevent the controller from thrashing lineage policy on every
 * generation. Once an adjustment has been recorded, later generations must wait
 * for the configured gap before another nudge is allowed.
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
 * Lineage adaptation only trusts the most recent telemetry snapshot because it
 * represents the latest scored generation. Missing or non-numeric lineage
 * evidence simply disables the adjustment for that cycle.
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
 * These bounds define the acceptable ancestry-diversity band. Values below the
 * lower threshold suggest the population is converging onto similar family
 * trees, while values above the upper threshold suggest diversity pressure may
 * already be stronger than needed.
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
 * Magnitude resolution keeps defaulting logic away from the mode-specific
 * adjusters so those helpers can focus on policy semantics.
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
 * This dispatcher is the decision fork for lineage adaptation. The thresholds
 * and telemetry signal have already been resolved by the time this helper runs,
 * so its only job is to send the adjustment into the correct policy surface.
 *
 * @param engine - NEAT engine instance.
 * @param config - Ancestor uniqueness adaptive configuration.
 * @param ancestorUniq - Current ancestor uniqueness metric.
 * @param thresholds - Threshold bounds for decisions.
 * @param adjustMagnitude - Adjustment magnitude.
 * @returns Nothing.
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
 * Epsilon mode nudges the multi-objective dominance tolerance when ancestry is
 * too uniform or too diffuse. That lets later Pareto comparisons become slightly
 * more or less permissive without changing the current generation directly.
 *
 * @param engine - NEAT engine instance.
 * @param ancestorUniq - Current ancestor uniqueness metric.
 * @param thresholds - Threshold bounds for decisions.
 * @param adjustMagnitude - Adjustment magnitude.
 * @returns Nothing.
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
 * Lineage-pressure mode keeps the feedback inside the ancestry-based selection
 * settings themselves. Low uniqueness increases spread pressure, while high
 * uniqueness relaxes it so the search does not over-penalize related genomes.
 *
 * @param engine - NEAT engine instance.
 * @param ancestorUniq - Current ancestor uniqueness metric.
 * @param thresholds - Threshold bounds for decisions.
 * @returns Nothing.
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
 * Some runs do not seed lineage-pressure options up front. This helper creates a
 * minimal spread-oriented state only when lineage-feedback mode actually needs
 * one.
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
 * Recording the adjustment generation is what makes the cooldown guard work on
 * later cycles.
 *
 * @param engine - NEAT engine instance.
 * @returns Nothing.
 */
export function recordAdjustment(engine: NeatLikeWithAdaptive): void {
  engine._lastAncestorUniqAdjustGen = engine.generation;
}
