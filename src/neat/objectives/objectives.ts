import type {
  GenomeLike,
  ObjectiveDescriptor,
} from '../shared/neat.shared.types';
import {
  collectDefaultObjectives,
  collectUserObjectives,
  ensureMultiObjectiveOptions,
  ensureObjectivesList,
  replaceObjectiveByKey,
} from './core/objectives.core';
import type { NeatLikeWithObjectives } from './core/objectives.types';

/**
 * Objective-management helpers for the NEAT controller.
 *
 * Objective management is the layer that decides what "better" means during a
 * run. In the simplest case that means the default fitness score. In a richer
 * multi-objective run it can also mean novelty, diversity, complexity, energy
 * use, or another measurable trait a caller wants to optimize or constrain.
 *
 * The root chapter keeps the public API small so those responsibilities stay
 * easy to understand:
 *
 * 1. `_getObjectives()` resolves the final ordered objective list the
 *    controller should use,
 * 2. `registerObjective()` adds or replaces user-defined objectives by key,
 * 3. `clearObjectives()` removes the user-defined layer and invalidates the
 *    cached resolution surface.
 *
 * The detailed mechanics stay in `core/`, where the controller decides when to
 * keep the default fitness objective, how to validate user-provided objective
 * descriptors, and how to replace an existing objective without mutating the
 * old list in place.
 *
 * Ownership boundary: objective registration is a controller-policy
 * surface. It tells later ranking or selection passes how to read progress,
 * but it does not mutate genome identity, compatibility distance, or replay
 * semantics by itself.
 *
 * Read this root chapter when you want the public controller contract first.
 * Drop into `core/` when you need the exact validation rules, default-objective
 * behavior, or list replacement semantics.
 *
 * ```mermaid
 * flowchart TD
 *   Defaults[Default fitness objective]
 *   UserObjectives[User-registered objectives]
 *   Resolver[Final ordered list]
 *   Register[Add or replace by key]
 *   Clear[Reset user layer]
 *   Core[core/<br/>validation and list rules]
 *
 *   Defaults --> Resolver
 *   UserObjectives --> Resolver
 *   Register --> UserObjectives
 *   Clear --> UserObjectives
 *   Resolver --> Core
 *   Register --> Core
 * ```
 *
 * Example:
 *
 * ```ts
 * neat.registerObjective('complexity', 'min', (genome) => genome.nodes.length);
 * const objectives = neat._getObjectives();
 * neat.clearObjectives();
 * ```
 */

/**
 * Build and return the list of registered objectives for this NEAT instance.
 *
 * This is the resolution step that turns controller configuration into the
 * final ordered objective list used by the rest of the multi-objective flow.
 * It reuses a cached list when available, otherwise it composes the active
 * default objective layer with the valid user-registered objective layer and
 * stores the resulting sequence for later reads.
 *
 * Use this when you need to inspect or debug the exact objective set a run will
 * evaluate rather than the raw options that happened to be configured earlier.
 *
 * @param this - NEAT host exposing multi-objective options and the cached objective list.
 * @returns Objective descriptors in the order they should be applied.
 *
 * @example
 * ```ts
 * const objectives = neat._getObjectives();
 * console.log(objectives.map((objective) => objective.key));
 * ```
 */
export function _getObjectives(
  this: NeatLikeWithObjectives,
): ObjectiveDescriptor[] {
  // Step 1: Reuse the cached objective list when available.
  if (this._objectivesList) {
    return this._objectivesList;
  }

  // Step 2: Collect default and user-registered objectives.
  const defaultObjectives = collectDefaultObjectives(this);
  const userObjectives = collectUserObjectives(this);

  // Step 3: Build, cache, and return the deterministic objective list.
  const objectivesList = [...defaultObjectives, ...userObjectives];
  this._objectivesList = objectivesList;
  return objectivesList;
}

/**
 * Register a new objective descriptor.
 *
 * Use this to extend or refine the controller's definition of progress. If an
 * objective with the same key already exists, the new descriptor replaces it so
 * callers can update objective behavior without manually clearing the full list
 * first.
 *
 * Registering an objective only updates configuration and invalidates the
 * cached resolved list. The final ordered objective set is rebuilt lazily the
 * next time {@link _getObjectives} runs.
 *
 * @param this - NEAT host exposing multi-objective options and the cached objective list.
 * @param key - Unique name for the objective.
 * @param direction - Whether the objective should be minimized or maximized.
 * @param accessor - Function that extracts a numeric value from a genome.
 * @returns Nothing. The host objective configuration is updated in place.
 *
 * @example
 * ```ts
 * neat.registerObjective('energy', 'min', (genome) => genome.cost ?? 0);
 * ```
 */
export function registerObjective(
  this: NeatLikeWithObjectives,
  key: string,
  direction: 'min' | 'max',
  accessor: (genome: GenomeLike) => number,
): void {
  // Step 1: Ensure the multi-objective container exists.
  const multiObjectiveOptions = ensureMultiObjectiveOptions(this);

  // Step 2: Ensure the registered objectives list exists.
  const objectivesList = ensureObjectivesList(multiObjectiveOptions);

  // Step 3: Replace the existing objective with the same key, when present.
  multiObjectiveOptions.objectives = replaceObjectiveByKey(
    objectivesList,
    key,
    direction,
    accessor,
  );

  // Step 4: Invalidate the cached objective list.
  this._objectivesList = undefined;
}

/**
 * Clear all registered multi-objectives.
 *
 * Use this when a run should fall back to its default objective behavior or
 * when a test needs a clean objective slate before registering a new set. This
 * clears the user-defined objective layer and invalidates the cached resolved
 * list so the next read rebuilds it from the remaining controller defaults.
 *
 * @param this - NEAT host exposing multi-objective options and the cached objective list.
 * @returns Nothing. Registered user objectives and the cached objective list are cleared.
 *
 * @example
 * ```ts
 * neat.clearObjectives();
 * const objectives = neat._getObjectives();
 * ```
 */
export function clearObjectives(this: NeatLikeWithObjectives): void {
  if (this.options.multiObjective?.objectives) {
    this.options.multiObjective.objectives = [];
  }

  this._objectivesList = undefined;
}
