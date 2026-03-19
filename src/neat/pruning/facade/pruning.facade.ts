import type { NeatLikeForPruning } from '../core/pruning.types';

/**
 * Public pruning facade helpers for the stable `Neat` entrypoint.
 *
 * This chapter is the public pruning-policy bridge for the stable `Neat`
 * entrypoint. The actual pruning orchestration still lives in
 * `pruning/pruning.ts`, but the top-level class preserves two lazy async
 * wrappers so optional pruning behavior does not crowd the main facade or force
 * every caller to load pruning mechanics up front.
 *
 * That separation matters for both teaching and behavior. The root pruning
 * chapter answers controller-facing questions about schedule-driven versus
 * adaptive pruning, while this facade answers the public-API question of how a
 * caller safely reaches those controllers from `Neat` itself.
 *
 * Read this file when you want the stable wrapper story:
 *
 * - `applyEvolutionPruning()` forwards scheduled generation-time pruning,
 * - `applyAdaptivePruning()` forwards metric-driven pruning,
 * - both helpers preserve best-effort lazy loading instead of turning pruning
 *   into a mandatory runtime dependency.
 */

/**
 * Narrow `Neat` host surface required by the public pruning facade.
 *
 * This stays intentionally small because the facade only forwards scheduled and
 * adaptive pruning calls into the extracted pruning implementation.
 *
 * The facade does not own the pruning math itself. It only needs the host
 * state that the root pruning chapter already consumes, which keeps this layer
 * from becoming a second public pruning implementation.
 */
export interface NeatPruningFacadeHost extends NeatLikeForPruning {}

/**
 * Apply evolution-time pruning through the stable public `Neat` facade.
 *
 * This is the public wrapper for schedule-driven pruning. Use it when the
 * caller wants the root pruning chapter to inspect generation timing and apply
 * pruning only when the configured evolution schedule says it is active.
 *
 * The underlying pruning module is loaded lazily so pruning remains optional,
 * and the facade preserves the same best-effort behavior as before the pruning
 * logic was extracted out of the main `Neat` class.
 *
 * @param host - `Neat` instance exposing pruning options, generation state, and population.
 * @returns Promise that resolves after the best-effort pruning attempt finishes.
 *
 * @example
 * ```ts
 * await neat.applyEvolutionPruning();
 * ```
 */
export async function applyEvolutionPruning(
  host: NeatPruningFacadeHost,
): Promise<void> {
  try {
    const pruningModule = await import('../pruning');
    pruningModule.applyEvolutionPruning.call(host as never);
  } catch {
    // Evolution-time pruning is optional; ignore failures.
  }
}

/**
 * Run adaptive pruning through the stable public `Neat` facade.
 *
 * This is the public wrapper for metric-driven pruning. Use it when pruning
 * should react to live population complexity signals rather than a fixed
 * generation schedule.
 *
 * The facade keeps the same lazy optional-loading behavior so callers can
 * continue to treat adaptive pruning as an additive maintenance feature rather
 * than a required runtime dependency.
 *
 * @param host - `Neat` instance exposing adaptive pruning state and population metrics.
 * @returns Promise that resolves after the best-effort adaptive pruning attempt finishes.
 *
 * @example
 * ```ts
 * await neat.applyAdaptivePruning();
 * ```
 */
export async function applyAdaptivePruning(
  host: NeatPruningFacadeHost,
): Promise<void> {
  try {
    const pruningModule = await import('../pruning');
    pruningModule.applyAdaptivePruning.call(host as never);
  } catch {
    // Adaptive pruning is optional; ignore failures.
  }
}
