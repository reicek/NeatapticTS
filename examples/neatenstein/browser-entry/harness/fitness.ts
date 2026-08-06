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
  EnemyEpisodeTelemetry,
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
  NEATENSTEIN_ENEMY_NAV_WEIGHT,
  NEATENSTEIN_ENEMY_COMBAT_WEIGHT,
  NEATENSTEIN_ENEMY_EXPLORATION_BONUS,
  NEATENSTEIN_ENEMY_STAGNATION_THRESHOLD,
  NEATENSTEIN_ENEMY_STAGNATION_PENALTY,
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
 * Compute a navigation fitness scalar from enemy episode telemetry
 * (AC-10.5e-001).
 *
 * The navigation fitness rewards progress toward the player goal, rewards
 * exploration of unique cells, and penalizes stagnation above a threshold:
 *
 * ```text
 * progress    = Σ (prevDist − curDist) per step  (telescoping to initial − final)
 * exploration = cellsVisited × EXPLORATION_BONUS
 * antiStall   = max(0, stagnationTicks − THRESHOLD) × STAGNATION_PENALTY
 * navFitness  = progress + exploration − antiStall
 * ```
 *
 * @param telemetry - Per-step enemy episode telemetry.
 * @returns A scalar navigation fitness score; higher is better.
 *
 * @example
 * ```ts
 * const score = computeEnemyNavigationFitness({
 *   position: { x: 60, y: 60 },
 *   bfsDistances: [20, 18, 16, 14, 12],
 *   damageDealt: 0,
 *   enemiesSurvived: 1,
 *   cellsVisited: 5,
 *   stagnationTicks: 0,
 *   finalDistance: 10,
 * });
 * ```
 */
export function computeEnemyNavigationFitness(
  telemetry: EnemyEpisodeTelemetry,
): FitnessScore {
  // Progress reward: Σ(prevDist - curDist) per step.
  const steps = telemetry.bfsDistances;
  let progress = 0;
  for (let i = 0; i < steps.length; i++) {
    const prevDist = steps[i];
    const curDist =
      i + 1 < steps.length ? steps[i + 1] : telemetry.finalDistance;
    progress += prevDist - curDist;
  }

  // Exploration bonus: +0.5 per unique cell visited.
  const exploration =
    telemetry.cellsVisited * NEATENSTEIN_ENEMY_EXPLORATION_BONUS;

  // Anti-stall penalty: −1 per stagnation tick above threshold.
  const excessStagnation = Math.max(
    0,
    telemetry.stagnationTicks - NEATENSTEIN_ENEMY_STAGNATION_THRESHOLD,
  );
  const antiStall = excessStagnation * NEATENSTEIN_ENEMY_STAGNATION_PENALTY;

  return progress + exploration - antiStall;
}

/**
 * Compute a composite team-level enemy fitness scalar from episode telemetry
 * (AC-10.5e-003).
 *
 * The composite blends navigation fitness (progress, exploration, anti-stall)
 * with combat fitness (damage dealt, survival):
 *
 * ```text
 * navigationFitness = computeEnemyNavigationFitness(telemetry)
 * combatFitness     = damageDealt * damageWeight + enemiesSurvived * survivalWeight
 * fitness           = navigationFitness * navWeight + combatFitness * combatWeight
 * ```
 *
 * The old `(damageDealt, enemiesSurvived, config?)` signature has been replaced
 * with `(telemetry, config?)`; no backward-compatibility wrapper is provided.
 *
 * @param telemetry - Per-step enemy episode telemetry.
 * @param config - Optional weights overriding the defaults.
 * @returns A scalar fitness score; higher is better.
 *
 * @example
 * ```ts
 * const score = computeEnemyTeamFitness(telemetry, { combatWeight: 2 });
 * ```
 */
export function computeEnemyTeamFitness(
  telemetry: EnemyEpisodeTelemetry,
  config?: EnemyTeamFitnessConfig,
): FitnessScore {
  const navigationWeight =
    config?.navigationWeight ?? NEATENSTEIN_ENEMY_NAV_WEIGHT;
  const combatWeight = config?.combatWeight ?? NEATENSTEIN_ENEMY_COMBAT_WEIGHT;
  const damageWeight =
    config?.damageWeight ?? NEATENSTEIN_ENEMY_TEAM_DAMAGE_WEIGHT;
  const survivalWeight =
    config?.survivalWeight ?? NEATENSTEIN_ENEMY_TEAM_SURVIVAL_WEIGHT;

  const navigationFitness = computeEnemyNavigationFitness(telemetry);
  const combatFitness =
    telemetry.damageDealt * damageWeight +
    telemetry.enemiesSurvived * survivalWeight;

  return navigationFitness * navigationWeight + combatFitness * combatWeight;
}
