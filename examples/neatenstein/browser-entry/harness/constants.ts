/**
 * Shared numeric constants for the Neatenstein asymmetric co-evolution harness.
 *
 * These values control refresh cadence, enemy backend sizing, and the default
 * weighting of the combat-quality fitness composite.
 *
 * @module
 */

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
 * The MLP receives eight world inputs (similar to the main agent's raycast
 * buffer), compresses through two hidden layers, and produces two movement
 * outputs.
 */
export const NEATENSTEIN_MLP_TOPOLOGY: readonly number[] = [8, 6, 4, 2];

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
