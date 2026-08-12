/**
 * Combat-pressure to reproduction-mode mapping for the NGE
 * (Neuro-evolutionary Genesis Engine) extension.
 *
 * Raw combat metrics are folded into a single deterministic
 * {@link ReproductionModePressureSignal}. The signal tells downstream evolution
 * operators whether the current generation is dominating, struggling, or in a
 * stalemate, which in turn influences how aggressively the population should
 * explore or exploit.
 */

import type { ReproductionModePressureSignal } from './neat.nge-evolution.reproduction-mode';

/**
 * Raw combat performance numbers captured for one generation or evaluation window.
 */
export interface CombatMetrics {
  /** Damage dealt by the subject(s) to opponents. */
  readonly damageDealt: number;
  /** Damage taken by the subject(s) from opponents. */
  readonly damageTaken: number;
  /** Number of opponents killed. */
  readonly kills: number;
  /** Number of deaths suffered. */
  readonly deaths: number;
  /** Generation number the metrics belong to. */
  readonly generation: number;
}

/** Weight applied to each kill when computing the combat advantage score. */
const KILL_ADVANTAGE_WEIGHT = 100;

/** Weight applied to each death when computing the combat advantage score. */
const DEATH_DISADVANTAGE_WEIGHT = 100;

/** Absolute advantage threshold above which the signal is considered dominating. */
const DOMINANCE_PRESSURE_THRESHOLD = 200;

/** Absolute disadvantage threshold below which the signal is considered struggling. */
const STRUGGLE_PRESSURE_THRESHOLD = 200;

/**
 * Map raw combat metrics to a deterministic reproduction-mode pressure signal.
 *
 * Exactly one of `isDominating`, `isStruggling`, or `isStalemate` is true on the
 * returned signal. The input `generation` is preserved unchanged so callers can
 * correlate the signal with the population generation it describes.
 *
 * @param metrics - Raw combat performance numbers.
 * @returns A pressure signal with a single active flag and the same generation number.
 */
export function evaluateCombatPressure(
  metrics: CombatMetrics,
): ReproductionModePressureSignal {
  const rawAdvantage =
    metrics.damageDealt +
    metrics.kills * KILL_ADVANTAGE_WEIGHT -
    metrics.damageTaken -
    metrics.deaths * DEATH_DISADVANTAGE_WEIGHT;

  if (rawAdvantage > DOMINANCE_PRESSURE_THRESHOLD) {
    return {
      generation: metrics.generation,
      isDominating: true,
      isStruggling: false,
      isStalemate: false,
    };
  }

  if (rawAdvantage < -STRUGGLE_PRESSURE_THRESHOLD) {
    return {
      generation: metrics.generation,
      isDominating: false,
      isStruggling: true,
      isStalemate: false,
    };
  }

  return {
    generation: metrics.generation,
    isDominating: false,
    isStruggling: false,
    isStalemate: true,
  };
}
