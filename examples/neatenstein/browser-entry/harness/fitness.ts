/**
 * Combat-quality fitness composite for the Neatenstein asymmetric co-evolution
 * harness.
 *
 * The composite turns a raw {@link CombatQualitySignal} episode summary into a
 * scalar fitness score that the selection step can rank. Positive metrics
 * (survival, damage dealt, kills, kill efficiency, hit/kill/fire rates,
 * complexity that helped) are rewarded; negative metrics (damage taken,
 * missed shots, blind fire, wall hits, excessive wiring density) are penalized.
 * A parsimony band keeps the main agent's graph from collapsing (too few
 * parameters to express interesting behavior) or bloating (too much wiring for
 * the task).
 *
 * P4S1 additions (AC-P4S1b-001 through 006):
 * - **killEfficiency** multiplier replacing the old ammoEfficiency concept
 * - **Scaled aimMissRate** weight (1 → 20) for stronger accuracy pressure
 * - **Blind-fire and wall-hit** shot penalties
 * - **Rate metrics** (hitRate, killRate, fireRate) with NaN guards
 *
 * @module
 */

import type { CombatQualitySignal, FitnessScore } from './types';
import type { EpisodeTelemetry, GameState } from '../host/game/types';
import {
  NEATENSTEIN_WEIGHT_SURVIVAL_TICKS,
  NEATENSTEIN_WEIGHT_DAMAGE_DEALT,
  NEATENSTEIN_WEIGHT_KILLS,
  NEATENSTEIN_WEIGHT_DAMAGE_TAKEN,
  NEATENSTEIN_WEIGHT_AIM_MISS_RATE,
  NEATENSTEIN_WEIGHT_COMPLEXITY_BONUS,
  NEATENSTEIN_WEIGHT_PARSIMONY_DENSITY_PENALTY,
  NEATENSTEIN_WEIGHT_KILL_EFFICIENCY,
  NEATENSTEIN_WEIGHT_BLIND_FIRE_PENALTY,
  NEATENSTEIN_WEIGHT_WALL_HIT_PENALTY,
  NEATENSTEIN_WEIGHT_HIT_RATE,
  NEATENSTEIN_WEIGHT_KILL_RATE,
  NEATENSTEIN_WEIGHT_FIRE_RATE,
  NEATENSTEIN_FIXED_TIMESTEP_MS,
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
 *         + killEfficiency  * wKillEfficiency      // P4S1b-001
 *         + hitRate         * wHitRate              // P4S1b-005
 *         + killRate        * wKillRate             // P4S1b-005
 *         + fireRate        * wFireRate             // P4S1b-005
 *         - damageTaken     * wDamageTaken
 *         - aimMissRate     * wAimMissRate
 *         - shotsBlindFire  * wBlindFirePenalty     // P4S1b-003
 *         - shotsWallHit    * wWallHitPenalty       // P4S1b-003
 *         + complexityBonus * wComplexityBonus
 *         - parsimonyPenalty * wParsimonyPenalty
 * ```
 *
 * Rate metrics use `max(shotsFired, 1)` as the denominator to prevent NaN
 * from division by zero (AC-P4S1b-006). When the optional signal fields
 * (`shotsFired`, `shotsHit`, etc.) are absent, the rate metrics default to
 * zero and only the base formula applies, preserving backward compatibility.
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
 *   shotsFired: 30,
 *   shotsHit: 15,
 *   shotsBlindFire: 2,
 *   shotsWallHit: 5,
 *   ticksElapsed: 120,
 * }, 1500);
 * ```
 */
export function computeCombatQualitySignal(
  signal: CombatQualitySignal,
  complexity?: number,
): FitnessScore {
  // --- P4S1: Rate metrics with NaN guards (AC-P4S1b-005, AC-P4S1b-006) ---
  const shotsFired = signal.shotsFired ?? 0;
  const safeShots = Math.max(shotsFired, 1); // AC-P4S1b-006: no NaN
  const shotsHit = signal.shotsHit ?? 0;
  const hitRate = shotsHit / safeShots;
  const killRate = signal.kills / safeShots;
  const ticksElapsed = signal.ticksElapsed ?? signal.survivalTicks;
  const fireRate = shotsFired / Math.max(ticksElapsed, 1);

  // --- P4S1: Kill efficiency multiplier (AC-P4S1b-001) ---
  // killEfficiency = kills / max(shotsFired, 1) — replaces old ammoEfficiency.
  const killEfficiency = signal.kills / safeShots;

  // Step 1: Build the weighted base score from the raw signal.
  const baseScore =
    signal.survivalTicks * NEATENSTEIN_WEIGHT_SURVIVAL_TICKS +
    signal.damageDealt * NEATENSTEIN_WEIGHT_DAMAGE_DEALT +
    signal.kills * NEATENSTEIN_WEIGHT_KILLS +
    killEfficiency * NEATENSTEIN_WEIGHT_KILL_EFFICIENCY +
    hitRate * NEATENSTEIN_WEIGHT_HIT_RATE +
    killRate * NEATENSTEIN_WEIGHT_KILL_RATE +
    fireRate * NEATENSTEIN_WEIGHT_FIRE_RATE -
    signal.damageTaken * NEATENSTEIN_WEIGHT_DAMAGE_TAKEN -
    signal.aimMissRate * NEATENSTEIN_WEIGHT_AIM_MISS_RATE -
    (signal.shotsBlindFire ?? 0) * NEATENSTEIN_WEIGHT_BLIND_FIRE_PENALTY -
    (signal.shotsWallHit ?? 0) * NEATENSTEIN_WEIGHT_WALL_HIT_PENALTY +
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
 * Extract a {@link CombatQualitySignal} from real game telemetry and final
 * game state (AC-088).
 *
 * Maps the raw per-episode counters accumulated by the combat simulation
 * (`EpisodeTelemetry`) and the final `GameState` snapshot into the structured
 * combat-quality signal consumed by {@link computeCombatQualitySignal}.
 *
 * ```text
 * survivalTicks = gameState.episodeTimeMs / NEATENSTEIN_FIXED_TIMESTEP_MS
 * damageDealt   = telemetry.damageDealt
 * kills         = gameState.kills
 * damageTaken   = (deaths * maxHealth) + (maxHealth − player.health)
 * aimMissRate   = telemetry.aimMissRate
 * shotsFired    = telemetry.shotsFired            // P4S1b
 * shotsHit      = telemetry.shotsHit              // P4S1b
 * shotsBlindFire = telemetry.shotsBlindFire       // P4S1b
 * shotsWallHit  = telemetry.shotsWallHit           // P4S1b
 * ticksElapsed  = survivalTicks                    // P4S1b
 * ```
 *
 * `complexityBonus` and `parsimonyDensityPenalty` default to 0; the selection
 * step may enrich the signal afterwards.
 *
 * @param gameState - Final game state snapshot after the episode completes.
 * @param telemetry - Per-episode combat telemetry accumulated during the run.
 * @returns A {@link CombatQualitySignal} derived from real gameplay metrics.
 *
 * @example
 * ```ts
 * const signal = extractCombatQualitySignal(finalState, telemetry);
 * const fitness = computeCombatQualitySignal(signal, complexity);
 * ```
 */
export function extractCombatQualitySignal(
  gameState: GameState,
  telemetry: EpisodeTelemetry,
): CombatQualitySignal {
  const maxHealth = gameState.player.maxHealth;
  const deaths = gameState.deaths ?? 0;
  const damageTaken =
    deaths * maxHealth + Math.max(0, maxHealth - gameState.player.health);

  const survivalTicks = Math.round(
    gameState.episodeTimeMs / NEATENSTEIN_FIXED_TIMESTEP_MS,
  );

  const signal: CombatQualitySignal = {
    survivalTicks,
    damageDealt: telemetry.damageDealt,
    kills: gameState.kills,
    damageTaken,
    aimMissRate: telemetry.aimMissRate,
    complexityBonus: 0,
    parsimonyDensityPenalty: 0,
    // P4S1b: Populate shot-quality and rate-metric fields from telemetry.
    shotsFired: telemetry.shotsFired,
    shotsHit: telemetry.shotsHit,
    shotsBlindFire: telemetry.shotsBlindFire,
    shotsWallHit: telemetry.shotsWallHit,
    ticksElapsed: survivalTicks,
  };

  return signal;
}
