/**
 * Curriculum-based respawn difficulty scaling for the Neatenstein enemy
 * population.
 *
 * Scales enemy capability based on real player performance telemetry,
 * replacing the previous RNG-based difficulty metrics.
 *
 * Stub module — implementation lands in Step B4 (04-implementing).
 * Tests in `curriculum-difficulty.test.ts` define the RED contracts.
 *
 * @module
 */

/**
 * Curriculum-based respawn difficulty scaling.
 *
 * Scales enemy capability based on real player performance telemetry,
 * replacing RNG-based difficulty metrics. Uses survival, damage, and kill
 * statistics to compute a difficulty score in [0, 1], which is then mapped
 * to enemy mutation sigma and wave difficulty progression.
 *
 * @module
 */

import { clamp01 } from '../shared/math-guards.utils';

/** Reference survival ticks for normalization (strong player baseline). */
const REFERENCE_SURVIVAL_TICKS = 500;

/** Reference damage dealt for normalization. */
const REFERENCE_DAMAGE_DEALT = 100;

/** Reference damage taken for normalization. */
const REFERENCE_DAMAGE_TAKEN = 100;

/** Reference kills for normalization. */
const REFERENCE_KILLS = 10;

/** Weight for survival component in difficulty calculation. */
const W_SURVIVAL = 0.3;

/** Weight for damage dealt component. */
const W_DAMAGE_DEALT = 0.3;

/** Weight for damage taken (inverse) component. */
const W_DAMAGE_TAKEN = 0.2;

/** Weight for kills component. */
const W_KILLS = 0.2;

/** Scaling factor for enemy capability at maximum difficulty. */
const CAPABILITY_SCALE = 1.0;

/** Base wave difficulty increment per wave. */
const WAVE_INCREMENT = 0.1;

/**
 * Player performance telemetry used for difficulty computation.
 */
export interface PlayerPerformanceTelemetry {
  /** Number of ticks the player survived. */
  survivalTicks: number;
  /** Damage dealt by the player. */
  damageDealt: number;
  /** Damage taken by the player. */
  damageTaken: number;
  /** Number of enemy kills. */
  kills: number;
  /** Number of player deaths. */
  deaths: number;
}

/**
 * Computes a curriculum difficulty score in [0, 1] from player performance
 * telemetry. Higher values indicate stronger player performance, warranting
 * tougher enemies.
 *
 * @param telemetry Player performance telemetry.
 * @returns Difficulty score in [0, 1].
 */
export function computeCurriculumDifficulty(
  telemetry: PlayerPerformanceTelemetry,
): number {
  const survivalScore = clamp01(
    telemetry.survivalTicks / REFERENCE_SURVIVAL_TICKS,
  );
  const damageDealtScore = clamp01(
    telemetry.damageDealt / REFERENCE_DAMAGE_DEALT,
  );
  const damageTakenScore = clamp01(
    1 - telemetry.damageTaken / REFERENCE_DAMAGE_TAKEN,
  );
  const killsScore = clamp01(telemetry.kills / REFERENCE_KILLS);

  const difficulty =
    W_SURVIVAL * survivalScore +
    W_DAMAGE_DEALT * damageDealtScore +
    W_DAMAGE_TAKEN * damageTakenScore +
    W_KILLS * killsScore;

  return clamp01(difficulty);
}

/**
 * Scales enemy mutation sigma proportional to the curriculum difficulty.
 * At difficulty=0, returns the baseline sigma. At difficulty=1, returns
 * double the baseline.
 *
 * @param baseSigma The base mutation sigma.
 * @param difficulty The curriculum difficulty in [0, 1].
 * @returns Scaled mutation sigma.
 */
export function scaleEnemyCapability(
  baseSigma: number,
  difficulty: number,
): number {
  return baseSigma * (1 + CAPABILITY_SCALE * clamp01(difficulty));
}

/**
 * Configuration for computeWaveDifficulty.
 */
export interface WaveDifficultyConfig {
  /** The current wave number (1-indexed). */
  wave: number;
  /** The player's survival rate in [0, 1]. */
  playerSurvivalRate: number;
}

/**
 * Computes wave difficulty that increases across waves when the player
 * survives consistently.
 *
 * @param config Configuration with wave number and player survival rate.
 * @returns Wave difficulty in [0, 1].
 */
export function computeWaveDifficulty(config: WaveDifficultyConfig): number {
  const waveComponent = config.wave * WAVE_INCREMENT;
  return clamp01(waveComponent * clamp01(config.playerSurvivalRate));
}
