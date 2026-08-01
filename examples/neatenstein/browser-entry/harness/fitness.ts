/**
 * Combat-quality fitness composite for the Neatenstein asymmetric co-evolution
 * harness.
 *
 * The composite turns a raw {@link CombatQualitySignal} episode summary into a
 * scalar fitness score that the selection step can rank. Positive metrics
 * (survival, damage dealt, kills, complexity that helped) are rewarded;
 * negative metrics (damage taken, missed shots, excessive wiring density) are
 * penalized. A parsimony band keeps the main agent's graph from collapsing
 * (too few parameters to express interesting behavior) or bloating (too much
 * wiring for the task).
 *
 * @module
 */

import type {
  CombatQualitySignal,
  EnemyTeamFitnessConfig,
  FitnessScore,
} from './types';
import {
  NEATENSTEIN_WEIGHT_SURVIVAL_TICKS,
  NEATENSTEIN_WEIGHT_DAMAGE_DEALT,
  NEATENSTEIN_WEIGHT_KILLS,
  NEATENSTEIN_WEIGHT_DAMAGE_TAKEN,
  NEATENSTEIN_WEIGHT_AIM_MISS_RATE,
  NEATENSTEIN_WEIGHT_COMPLEXITY_BONUS,
  NEATENSTEIN_WEIGHT_PARSIMONY_DENSITY_PENALTY,
  NEATENSTEIN_ENEMY_TEAM_DAMAGE_WEIGHT,
  NEATENSTEIN_ENEMY_TEAM_SURVIVAL_WEIGHT,
} from './constants';

/**
 * Lower bound of the desired neuron/synapse complexity band.
 *
 * Complexity counts below this value are penalized because the network is too
 * small to sustain interesting behavior for the shooter task.
 */
export const NEATENSTEIN_PARSIMONY_LOWER_BOUND = 800;

/**
 * Upper bound of the desired neuron/synapse complexity band.
 *
 * Complexity counts above this value are penalized to discourage wiring bloat.
 */
export const NEATENSTEIN_PARSIMONY_UPPER_BOUND = 3000;

/**
 * Compute a scalar fitness score from a combat-quality signal.
 *
 * The formula uses the weights defined in {@link ./constants.ts}:
 *
 * ```text
 * fitness = survivalTicks   * wSurvival
 *         + damageDealt     * wDamageDealt
 *         + kills           * wKills
 *         - damageTaken     * wDamageTaken
 *         - aimMissRate     * wAimMissRate
 *         + complexityBonus * wComplexityBonus
 *         - parsimonyPenalty * wParsimonyPenalty
 * ```
 *
 * When an optional `complexity` count (neurons + synapses) is supplied, an
 * additional parsimony penalty is applied for counts outside the
 * `[NEATENSTEIN_PARSIMONY_LOWER_BOUND, NEATENSTEIN_PARSIMONY_UPPER_BOUND]`
 * band. This keeps the evolved networks in a healthy size range without
 * hard-coding a topology.
 *
 * @param signal - Raw episode telemetry for the main agent.
 * @param complexity - Optional neuron/synapse count used for the parsimony band.
 * @returns A scalar fitness score; higher is better.
 *
 * @example
 * ```ts
 * const score = computeCombatQualitySignal({
 *   survivalTicks: 120,
 *   damageDealt: 45,
 *   kills: 2,
 *   damageTaken: 10,
 *   aimMissRate: 0.2,
 *   complexityBonus: 5,
 *   parsimonyDensityPenalty: 0,
 * }, 1500);
 * ```
 */
export function computeCombatQualitySignal(
  signal: CombatQualitySignal,
  complexity?: number,
): FitnessScore {
  // Step 1: Build the weighted base score from the raw signal.
  const baseScore =
    signal.survivalTicks * NEATENSTEIN_WEIGHT_SURVIVAL_TICKS +
    signal.damageDealt * NEATENSTEIN_WEIGHT_DAMAGE_DEALT +
    signal.kills * NEATENSTEIN_WEIGHT_KILLS -
    signal.damageTaken * NEATENSTEIN_WEIGHT_DAMAGE_TAKEN -
    signal.aimMissRate * NEATENSTEIN_WEIGHT_AIM_MISS_RATE +
    signal.complexityBonus * NEATENSTEIN_WEIGHT_COMPLEXITY_BONUS -
    signal.parsimonyDensityPenalty *
      NEATENSTEIN_WEIGHT_PARSIMONY_DENSITY_PENALTY;

  // Step 2: Apply an optional parsimony band penalty if complexity is known.
  if (complexity === undefined) {
    return baseScore;
  }

  if (complexity < NEATENSTEIN_PARSIMONY_LOWER_BOUND) {
    const distance = NEATENSTEIN_PARSIMONY_LOWER_BOUND - complexity;
    return baseScore - distance * NEATENSTEIN_WEIGHT_PARSIMONY_DENSITY_PENALTY;
  }

  if (complexity > NEATENSTEIN_PARSIMONY_UPPER_BOUND) {
    const distance = complexity - NEATENSTEIN_PARSIMONY_UPPER_BOUND;
    return baseScore - distance * NEATENSTEIN_WEIGHT_PARSIMONY_DENSITY_PENALTY;
  }

  return baseScore;
}

/**
 * Compute a team-level enemy fitness scalar from collective damage and survival.
 *
 * The enemy population is evaluated as a team: one scalar represents how much
 * pressure the entire enemy swarm applied to the main agent. Higher values
 * mean a more threatening swarm.
 *
 * ```text
 * fitness = damageDealt * damageWeight + enemiesSurvived * survivalWeight
 * ```
 *
 * The default weights reward both damage dealt and survival equally, but
 * callers can override either weight through the optional `config` object to
 * experiment with different selection pressures.
 *
 * @param damageDealt - Total damage the enemy team dealt to the main agent.
 * @param enemiesSurvived - Number of enemy variants still alive at episode end.
 * @param config - Optional weights overriding the defaults.
 * @returns A scalar fitness score; higher is better.
 *
 * @example
 * ```ts
 * const score = computeEnemyTeamFitness(120, 4, { damageWeight: 2 });
 * console.log(score); // 244 when using default survivalWeight of 1
 * ```
 */
export function computeEnemyTeamFitness(
  damageDealt: number,
  enemiesSurvived: number,
  config?: EnemyTeamFitnessConfig,
): FitnessScore {
  const damageWeight =
    config?.damageWeight ?? NEATENSTEIN_ENEMY_TEAM_DAMAGE_WEIGHT;
  const survivalWeight =
    config?.survivalWeight ?? NEATENSTEIN_ENEMY_TEAM_SURVIVAL_WEIGHT;
  return damageDealt * damageWeight + enemiesSurvived * survivalWeight;
}
