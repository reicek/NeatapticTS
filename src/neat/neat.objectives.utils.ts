import type { ObjectiveDescriptor, GenomeLike } from './neat.types';

/**
 * Minimal interface for NEAT instances using objective management.
 *
 * This shape is intentionally small and only includes the pieces needed by
 * `_getObjectives`, `registerObjective`, and `clearObjectives`.
 *
 * @example
 * ```ts
 * const neatLike: NeatLikeWithObjectives = {
 *   options: { multiObjective: { enabled: true, objectives: [] } },
 * };
 * ```
 */
export interface NeatLikeWithObjectives {
  /**
   * Runtime configuration for multi-objective behavior.
   */
  options: {
    /**
     * Multi-objective settings (optional).
     */
    multiObjective?: {
      /**
       * Whether multi-objective mode is enabled.
       */
      enabled?: boolean;
      /**
       * Registered objective descriptors.
       */
      objectives?: ObjectiveDescriptor[];
    };
  };
  /**
   * Cached objective list built by `_getObjectives`.
   */
  _objectivesList?: ObjectiveDescriptor[];
  /**
   * When true, omits the default single-objective fitness descriptor.
   */
  _suppressFitnessObjective?: boolean;
}

/**
 * @param neatInstance - Instance providing objective settings.
 * @returns Default objectives when fitness is not suppressed.
 */
export function collectDefaultObjectives(
  neatInstance: NeatLikeWithObjectives,
): ObjectiveDescriptor[] {
  // Step 1: Skip when the default fitness objective is suppressed.
  if (neatInstance._suppressFitnessObjective) return [];

  // Step 2: Build the default fitness descriptor.
  return [buildDefaultFitnessObjective()];
}

/**
 * @param neatInstance - Instance providing objective settings.
 * @returns Valid user-registered objectives when multi-objective is enabled.
 */
export function collectUserObjectives(
  neatInstance: NeatLikeWithObjectives,
): ObjectiveDescriptor[] {
  // Step 1: Guard against disabled or missing multi-objective settings.
  if (!isMultiObjectiveEnabled(neatInstance)) return [];

  // Step 2: Filter only valid objective descriptors.
  return getObjectiveCandidates(neatInstance).filter(isValidObjective);
}

/**
 * @returns Default fitness objective descriptor.
 */
export function buildDefaultFitnessObjective(): ObjectiveDescriptor {
  return {
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
    accessor: (genome: GenomeLike) => {
      interface GenomeWithScore {
        score?: number;
      }
      return (genome as GenomeWithScore).score ?? 0;
    },
  };
}

/**
 * @param neatInstance - Instance providing objective settings.
 * @returns Whether multi-objective mode is enabled with a candidate list.
 */
export function isMultiObjectiveEnabled(
  neatInstance: NeatLikeWithObjectives,
): boolean {
  // Step 1: Confirm the enabled flag is set.
  if (!neatInstance.options.multiObjective?.enabled) return false;

  // Step 2: Confirm the objectives list is an array.
  return Array.isArray(neatInstance.options.multiObjective.objectives);
}

/**
 * @param neatInstance - Instance providing objective settings.
 * @returns Candidate objectives from configuration.
 */
export function getObjectiveCandidates(
  neatInstance: NeatLikeWithObjectives,
): ObjectiveDescriptor[] {
  return neatInstance.options.multiObjective?.objectives ?? [];
}

/**
 * @param candidateObjective - Candidate descriptor to validate.
 * @returns True when the descriptor has the required shape.
 */
export function isValidObjective(
  candidateObjective: ObjectiveDescriptor | undefined,
): candidateObjective is ObjectiveDescriptor {
  // Step 1: Ensure the candidate exists.
  if (!candidateObjective) return false;

  // Step 2: Ensure required fields are present and valid.
  if (!candidateObjective.key) return false;
  return typeof candidateObjective.accessor === 'function';
}

/**
 * @param neatInstance - Instance receiving the multi-objective container.
 * @returns Initialized multi-objective options.
 */
export function ensureMultiObjectiveOptions(
  neatInstance: NeatLikeWithObjectives,
): NonNullable<NeatLikeWithObjectives['options']['multiObjective']> {
  // Step 1: Create and enable the container when missing.
  if (!neatInstance.options.multiObjective)
    neatInstance.options.multiObjective = { enabled: true };

  // Step 2: Return the initialized container.
  return neatInstance.options.multiObjective;
}

/**
 * @param multiObjectiveOptions - Multi-objective container to hydrate.
 * @returns Objectives list for mutation-free operations.
 */
export function ensureObjectivesList(
  multiObjectiveOptions: NonNullable<
    NeatLikeWithObjectives['options']['multiObjective']
  >,
): ObjectiveDescriptor[] {
  // Step 1: Create an empty objectives list when missing.
  if (!multiObjectiveOptions.objectives) multiObjectiveOptions.objectives = [];

  // Step 2: Return the list.
  return multiObjectiveOptions.objectives;
}

/**
 * @param objectivesList - Existing objectives to update.
 * @param objectiveKey - Key to replace.
 * @param objectiveDirection - Direction for the new objective.
 * @param objectiveAccessor - Accessor for the new objective.
 * @returns Updated objectives list with the new descriptor appended.
 */
export function replaceObjectiveByKey(
  objectivesList: ObjectiveDescriptor[],
  objectiveKey: string,
  objectiveDirection: 'min' | 'max',
  objectiveAccessor: (genome: GenomeLike) => number,
): ObjectiveDescriptor[] {
  // Step 1: Remove any existing objective with the same key.
  const filteredObjectives = objectivesList.filter(
    (existingObjective) => existingObjective.key !== objectiveKey,
  );

  // Step 2: Append the new objective descriptor.
  filteredObjectives.push({
    key: objectiveKey,
    direction: objectiveDirection,
    accessor: objectiveAccessor,
  });

  // Step 3: Return the updated list.
  return filteredObjectives;
}
