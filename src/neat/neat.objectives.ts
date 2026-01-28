import type { ObjectiveDescriptor, GenomeLike } from './neat.types';
import {
  collectDefaultObjectives,
  collectUserObjectives,
  ensureMultiObjectiveOptions,
  ensureObjectivesList,
  replaceObjectiveByKey,
  type NeatLikeWithObjectives,
} from './neat.objectives.utils';

/**
 * Build and return the list of registered objectives for this NEAT instance.
 *
 * This function lazily builds `this._objectivesList` from the built-in
 * fitness objective (unless suppressed) and any user-registered multi-
 * objective descriptors found on `this.options.multiObjective.objectives`.
 *
 * Typical use: the evolution loop calls this to know which objectives to
 * evaluate and whether each objective should be maximized or minimized.
 *
 * Example:
 * ```ts
 * const objectives = neatInstance._getObjectives();
 * // objectives: Array<ObjectiveDescriptor>
 * ```
 *
 * @returns {ObjectiveDescriptor[]} Array of objective descriptors in the
 *   order they should be applied. If multi-objective support is disabled or
 *   no objectives are registered, this will contain only the built-in
 *   fitness objective (unless suppressed).
 */
export function _getObjectives(
  this: NeatLikeWithObjectives,
): ObjectiveDescriptor[] {
  // Step 1: Reuse cached objectives when available.
  if (this._objectivesList) return this._objectivesList;

  /**
   * The working list of objectives we will populate and cache on `this`.
   *
   * @example
   * ```ts
   * const objectivesList: ObjectiveDescriptor[] = [];
   * ```
   */
  const objectivesList: ObjectiveDescriptor[] = [];

  // Step 2: Collect default and user-registered objectives.
  const defaultObjectives = collectDefaultObjectives(this);
  const userObjectives = collectUserObjectives(this);

  // Step 3: Fold into the working list in deterministic order.
  objectivesList.push(...defaultObjectives, ...userObjectives);

  // Step 4: Cache and return the computed objectives list.
  this._objectivesList = objectivesList;
  return objectivesList;
}

/**
 * Register a new objective descriptor.
 *
 * This adds or replaces an objective with the given `key`. The objective is a
 * lightweight descriptor with a `key`, `direction` ('min' | 'max'), and an
 * `accessor` function that maps a genome to a numeric objective value.
 *
 * Example:
 * ```ts
 * // register an objective that measures model sparsity (lower is better)
 * neat.registerObjective('sparsity', 'min', genome => computeSparsity(genome));
 * ```
 *
 * Notes:
 * - If `this.options.multiObjective` doesn't exist it will be created and
 *   enabled.
 * - Registering an objective replaces any previous objective with the same
 *   `key`.
 *
 * @param {string} key Unique name for the objective (used for sorting/lookup)
 * @param {'min'|'max'} direction Whether the objective should be minimized or maximized
 * @param {(g: GenomeLike) => number} accessor Function to extract a numeric value from a genome
 */
export function registerObjective(
  this: NeatLikeWithObjectives,
  key: string,
  direction: 'min' | 'max',
  accessor: (genome: GenomeLike) => number,
) {
  // Step 1: Ensure the multi-objective container is initialized.
  const multiObjectiveOptions = ensureMultiObjectiveOptions(this);

  // Step 2: Ensure the objectives list exists.
  const objectivesList = ensureObjectivesList(multiObjectiveOptions);

  // Step 3: Replace any existing objective with the same key.
  const updatedObjectives = replaceObjectiveByKey(
    objectivesList,
    key,
    direction,
    accessor,
  );

  // Step 4: Persist the updated list.
  multiObjectiveOptions.objectives = updatedObjectives;

  // Step 5: Invalidate cached list so callers will pick up the change.
  this._objectivesList = undefined;
}

/**
 * Clear all registered multi-objectives.
 *
 * This resets `this.options.multiObjective.objectives` to an empty array and
 * clears the cached objectives list so that subsequent calls will reflect the
 * cleared state.
 *
 * Example:
 * ```ts
 * neat.clearObjectives();
 * // now only the default fitness objective (unless suppressed) will remain
 * ```
 */
export function clearObjectives(this: NeatLikeWithObjectives) {
  // Reset the registered objectives array when present
  if (this.options.multiObjective?.objectives)
    this.options.multiObjective.objectives = [];

  // Invalidate the cached objectives list
  this._objectivesList = undefined;
}
