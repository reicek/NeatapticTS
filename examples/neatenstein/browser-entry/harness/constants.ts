/**
 * Shared numeric constants for the Neatenstein asymmetric co-evolution harness.
 *
 * These values control refresh cadence, enemy backend sizing, and the default
 * weighting of the combat-quality fitness composite.
 *
 * @module
 */

/**
 * Re-export the fixed simulation timestep so harness modules can import all
 * cadence constants from a single location.
 *
 * The authoritative value lives in `host/game/constants.ts`; this re-export
 * avoids redefining it and prevents drift between the game-simulation and
 * harness evaluation paths.
 */
export { NEATENSTEIN_FIXED_TIMESTEP_MS } from '../host/game/constants';

/**
 * Number of main-agent variants evaluated in a single generation.
 *
 * Reduced from the original 8 to 4 so each variant plays a full episode while
 * keeping total generation time practical for headless batch evaluation.
 */
export const NEATENSTEIN_MAIN_VARIANT_COUNT = 4;

/**
 * Duration of one fitness evaluation episode in milliseconds of simulated
 * time (5 seconds).
 */
export const NEATENSTEIN_FITNESS_EPISODE_DURATION_MS = 5000;

/**
 * Maximum number of ticks in one fitness evaluation episode.
 *
 * Derived as `Math.floor(NEATENSTEIN_FITNESS_EPISODE_DURATION_MS /
 * NEATENSTEIN_FIXED_TIMESTEP_MS)` = 312.
 */
export const NEATENSTEIN_FITNESS_MAX_EPISODE_TICKS = Math.floor(
  NEATENSTEIN_FITNESS_EPISODE_DURATION_MS / 16,
); // 312

/**
 * Number of ticks per evaluation chunk for cooperative yielding in workers.
 *
 * The evaluation loop yields (via `setTimeout(0)`) after every chunk of this
 * many ticks so `onmessage` can fire between chunks, preventing the worker
 * from blocking.
 */
export const NEATENSTEIN_EVAL_CHUNK_TICKS = 32;

/**
 * Number of generations between MLP enemy population refreshes.
 *
 * The MLP backend is slower-moving than the SWARM backend because its
 * weight-only evolution needs more main-agent exposure to produce meaningful
 * pressure.
 */
export const NEATENSTEIN_MLP_REFRESH_INTERVAL_GENERATIONS = 5;

/**
 * Number of generations between SWARM enemy population refreshes.
 */
export const NEATENSTEIN_SWARM_REFRESH_INTERVAL_GENERATIONS = 3;

/**
 * Fixed layer topology for the MLP enemy backend.
 *
 * The MLP receives six world inputs (the vision vector from the BFS distance
 * map), compresses through two hidden layers, and produces four movement
 * outputs: move, strafe, turn, and fire.
 */
export const NEATENSTEIN_MLP_TOPOLOGY: readonly number[] = [6, 6, 4, 4];

/**
 * Number of weight-only MLP enemy variants maintained by the population.
 */
export const NEATENSTEIN_MLP_VARIANT_COUNT = 32;

/**
 * Maximum number of agents in the SWARM enemy backend.
 */
export const NEATENSTEIN_SWARM_MAX_SIZE = 8;

/**
 * Default fitness weight for survival time.
 *
 * Longer survival is the primary reward signal.
 */
export const NEATENSTEIN_WEIGHT_SURVIVAL_TICKS = 1;

/**
 * Default fitness weight for damage dealt.
 */
export const NEATENSTEIN_WEIGHT_DAMAGE_DEALT = 2;

/**
 * Default fitness weight for confirmed kills.
 */
export const NEATENSTEIN_WEIGHT_KILLS = 5;

/**
 * Default penalty weight for damage taken.
 *
 * The fitness composite subtracts `damageTaken * weight`.
 */
export const NEATENSTEIN_WEIGHT_DAMAGE_TAKEN = 1;

/**
 * Default penalty weight for aim miss rate.
 *
 * Scaled from 1 to 20 (AC-P4S1b-002) so that poor accuracy is strongly
 * penalized relative to the other combat-quality components.
 *
 * The fitness composite subtracts `aimMissRate * weight`.
 */
export const NEATENSTEIN_WEIGHT_AIM_MISS_RATE = 20;

/**
 * Default bonus weight for network complexity that improved performance.
 */
export const NEATENSTEIN_WEIGHT_COMPLEXITY_BONUS = 0.1;

/**
 * Default penalty weight for excessive wiring density (parsimony pressure).
 */
export const NEATENSTEIN_WEIGHT_PARSIMONY_DENSITY_PENALTY = 0.01;

// ---------------------------------------------------------------------------
// P4S1-fitness-weights: kill efficiency, shot-quality penalties, and rate
// metrics (AC-P4S1b-001, AC-P4S1b-003, AC-P4S1b-005, AC-P4S1b-006).
// ---------------------------------------------------------------------------

/**
 * Reward weight for the kill-efficiency multiplier (AC-P4S1b-001).
 *
 * `killEfficiency = kills / max(shotsFired, 1)`. This rewards agents that
 * convert shots into kills efficiently, replacing the old ammoEfficiency
 * concept.
 */
export const NEATENSTEIN_WEIGHT_KILL_EFFICIENCY = 10;

/**
 * Penalty weight per blind-fire shot (AC-P4S1b-003).
 *
 * Blind-fire shots are fired when no active enemy exists in the world.
 * Each such shot incurs this penalty to discourage wasting ammo.
 */
export const NEATENSTEIN_WEIGHT_BLIND_FIRE_PENALTY = 3;

/**
 * Penalty weight per wall-hit shot (AC-P4S1b-003).
 *
 * Wall-hit shots terminate on a wall with no enemy near the bolt path,
 * indicating poor aim. Each such shot incurs this penalty.
 */
export const NEATENSTEIN_WEIGHT_WALL_HIT_PENALTY = 2;

/**
 * Reward weight for the hit-rate metric (AC-P4S1b-005).
 *
 * `hitRate = shotsHit / max(shotsFired, 1)`. Contributes positively to
 * fitness to reward accuracy.
 */
export const NEATENSTEIN_WEIGHT_HIT_RATE = 5;

/**
 * Reward weight for the kill-rate metric (AC-P4S1b-005).
 *
 * `killRate = kills / max(shotsFired, 1)`. Contributes positively to
 * fitness to reward lethality per shot.
 */
export const NEATENSTEIN_WEIGHT_KILL_RATE = 5;

/**
 * Reward weight for the fire-rate metric (AC-P4S1b-005).
 *
 * `fireRate = shotsFired / max(ticksElapsed, 1)`. Contributes positively
 * to fitness to encourage active engagement.
 */
export const NEATENSTEIN_WEIGHT_FIRE_RATE = 1;
