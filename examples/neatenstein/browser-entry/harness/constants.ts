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
 * NEATENSTEIN_FIXED_TIMESTEP_MS)` = 312. This is a NEW constant for the fitness
 * evaluation path; the existing {@link NEATENSTEIN_MAX_EPISODE_TICKS} (240) is
 * used by `enemy-runner.ts` and is NOT changed.
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
 * Total number of enemy variants maintained across the enemy population.
 *
 * This is the full MLP population size; each generation selects up to
 * {@link NEATENSTEIN_MAX_ACTIVE_ENEMIES} to spawn on screen at once.
 */
export const NEATENSTEIN_ENEMY_POPULATION_SIZE = 32;

/**
 * Maximum number of enemies that may be active on screen at the same time.
 */
export const NEATENSTEIN_MAX_ACTIVE_ENEMIES = 8;

/**
 * Deterministic episode duration used for one enemy evaluation, in milliseconds.
 *
 * A 10-second episode is long enough for movement and combat differences to
 * surface without making headless batch evaluation prohibitively expensive.
 */
export const NEATENSTEIN_ENEMY_EVALUATION_DURATION_MS = 10_000;

/**
 * Maximum number of ticks (frames) in one enemy episode rollout.
 *
 * Each tick advances the simulation by one fixed timestep (~16 ms), so 240
 * ticks correspond to roughly 3.84 seconds of simulated time. The bound keeps
 * headless batch evaluation finite while giving the MLP enough steps to
 * navigate toward the static player goal.
 */
export const NEATENSTEIN_MAX_EPISODE_TICKS = 240;

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
 * The fitness composite subtracts `aimMissRate * weight`.
 */
export const NEATENSTEIN_WEIGHT_AIM_MISS_RATE = 1;

/**
 * Default bonus weight for network complexity that improved performance.
 */
export const NEATENSTEIN_WEIGHT_COMPLEXITY_BONUS = 0.1;

/**
 * Default penalty weight for excessive wiring density (parsimony pressure).
 */
export const NEATENSTEIN_WEIGHT_PARSIMONY_DENSITY_PENALTY = 0.01;

/**
 * Default weight for collective damage dealt by the enemy team.
 *
 * Damage dealt is the primary reward signal for the enemy population because
 * it directly measures pressure applied to the main agent.
 */
export const NEATENSTEIN_ENEMY_TEAM_DAMAGE_WEIGHT = 1;

/**
 * Default weight for enemy survival count.
 *
 * Surviving enemies receive a smaller reward than damage dealt so that
 * aggressive behavior is preferred over passive longevity.
 */
export const NEATENSTEIN_ENEMY_TEAM_SURVIVAL_WEIGHT = 1;

/**
 * Default weight for the enemy navigation fitness component.
 *
 * Navigation fitness rewards progress toward the player goal, rewards
 * exploration of unique cells, and penalizes stagnation above a threshold.
 */
export const NEATENSTEIN_ENEMY_NAV_WEIGHT = 1;

/**
 * Default weight for the enemy combat fitness component.
 *
 * Combat fitness rewards damage dealt and survival.
 */
export const NEATENSTEIN_ENEMY_COMBAT_WEIGHT = 1;

/**
 * Bonus per unique cell visited by the enemy.
 *
 * Encourages exploration of the maze rather than camping in one spot.
 */
export const NEATENSTEIN_ENEMY_EXPLORATION_BONUS = 0.5;

/**
 * Stagnation tick threshold above which the anti-stall penalty applies.
 *
 * For 240-tick episodes, ticks above this threshold incur a per-tick penalty
 * to discourage the enemy from getting stuck against walls.
 */
export const NEATENSTEIN_ENEMY_STAGNATION_THRESHOLD = 80;

/**
 * Per-tick penalty for each stagnation tick above the threshold.
 */
export const NEATENSTEIN_ENEMY_STAGNATION_PENALTY = 1;
