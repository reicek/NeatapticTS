import { ObjectiveDescriptor, GenomeLike } from './neat.types';

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
export function _getObjectives(this: any): ObjectiveDescriptor[] {
  // Return cached objectives list if already computed
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

  // Step 1: Add the default single-objective 'fitness' unless explicitly suppressed
  if (!this._suppressFitnessObjective) {
    objectivesList.push({
      key: 'fitness',
      direction: 'max',
      /**
       * Default accessor extracts the `score` property from a genome.
       *
       * @example
       * ```ts
       * // genome.score is used as the fitness metric by default
       * const value = defaultAccessor(genome);
       * ```
       */
      accessor: (genome: GenomeLike) => (genome as any).score || 0,
    });
  }

  // Step 2: If multi-objective is enabled and objectives array exists, append them
  if (
    this.options.multiObjective?.enabled &&
    Array.isArray(this.options.multiObjective.objectives)
  ) {
    for (const candidateObjective of this.options.multiObjective
      .objectives as ObjectiveDescriptor[]) {
      // Validate shape before accepting
      if (
        !candidateObjective ||
        !candidateObjective.key ||
        typeof candidateObjective.accessor !== 'function'
      )
        continue;
      objectivesList.push(candidateObjective as ObjectiveDescriptor);
    }
  }

  // Cache the computed objectives list for subsequent calls
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
  this: any,
  key: string,
  direction: 'min' | 'max',
  accessor: (genome: GenomeLike) => number
) {
  // Ensure multi-objective container exists and is enabled
  if (!this.options.multiObjective)
    this.options.multiObjective = { enabled: true } as any;

  /**
   * Convenience reference to multi-objective related options on `this`.
   *
   * @example
   * ```ts
   * const multiObjectiveOptions = this.options.multiObjective as any;
   * ```
   */
  const multiObjectiveOptions: any = this.options.multiObjective;

  // Ensure the objectives array exists
  if (!multiObjectiveOptions.objectives) multiObjectiveOptions.objectives = [];

  // Step: remove any existing objective with the same key (replace semantics)
  multiObjectiveOptions.objectives = (multiObjectiveOptions.objectives as ObjectiveDescriptor[]).filter(
    (existingObjective) => existingObjective.key !== key
  );

  // Step: push new objective descriptor
  multiObjectiveOptions.objectives.push({ key, direction, accessor });

  // Invalidate cached list so callers will pick up the change
  this._objectivesList = undefined as any;
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
export function clearObjectives(this: any) {
  // Reset the registered objectives array when present
  if (this.options.multiObjective?.objectives)
    this.options.multiObjective.objectives = [];

  // Invalidate the cached objectives list
  this._objectivesList = undefined as any;
}
