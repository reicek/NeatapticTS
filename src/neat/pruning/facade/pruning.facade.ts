import type { NeatLikeForPruning } from '../core/pruning.types';

/**
 * Public pruning facade helpers for the stable `Neat` entrypoint.
 *
 * The pruning algorithms live in `pruning/pruning.ts`, while this facade keeps
 * the public class wrappers small and lazy-loaded so optional pruning behavior
 * does not clutter the main `Neat` surface.
 */

/**
 * Narrow `Neat` host surface required by the public pruning facade.
 *
 * This stays intentionally small because the facade only forwards scheduled and
 * adaptive pruning calls into the extracted pruning implementation.
 */
export interface NeatPruningFacadeHost extends NeatLikeForPruning {}

/**
 * Apply evolution-time pruning through the stable public `Neat` facade.
 *
 * The underlying pruning module is loaded lazily so pruning remains optional
 * and the public facade preserves the same best-effort behavior as before this
 * extraction.
 *
 * @param host - `Neat` instance exposing pruning options, generation state, and population.
 * @returns Promise that resolves after the best-effort pruning attempt finishes.
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
 * The facade keeps the lazy optional-loading behavior so callers can continue
 * to treat adaptive pruning as an additive maintenance feature rather than a
 * required runtime dependency.
 *
 * @param host - `Neat` instance exposing adaptive pruning state and population metrics.
 * @returns Promise that resolves after the best-effort adaptive pruning attempt finishes.
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