import type { GenomeLike, ObjectiveDescriptor } from '../../neat.types';
import type { NeatLikeWithObjectives } from './objectives.types';

/**
 * Objective-list mechanics used by the NEAT controller.
 *
 * This chapter holds the default fitness objective, validation helpers, and
 * the small list-management helpers behind objective registration.
 */

/**
 * Collect the default objectives when fitness is not suppressed.
 *
 * @param neatInstance - NEAT host exposing objective settings.
 * @returns Default objective descriptors.
 */
export function collectDefaultObjectives(
  neatInstance: NeatLikeWithObjectives,
): ObjectiveDescriptor[] {
  if (neatInstance._suppressFitnessObjective) {
    return [];
  }

  return [buildDefaultFitnessObjective()];
}

/**
 * Collect valid user-registered objectives when multi-objective mode is enabled.
 *
 * @param neatInstance - NEAT host exposing objective settings.
 * @returns Valid user objective descriptors.
 */
export function collectUserObjectives(
  neatInstance: NeatLikeWithObjectives,
): ObjectiveDescriptor[] {
  if (!isMultiObjectiveEnabled(neatInstance)) {
    return [];
  }

  return getObjectiveCandidates(neatInstance).filter(isValidObjective);
}

/**
 * Build the default fitness objective descriptor.
 *
 * @returns Default fitness objective descriptor.
 */
export function buildDefaultFitnessObjective(): ObjectiveDescriptor {
  return {
    key: 'fitness',
    direction: 'max',
    accessor: (genome: GenomeLike) => {
      interface GenomeWithScore {
        score?: number;
      }

      return (genome as GenomeWithScore).score ?? 0;
    },
  };
}

/**
 * Check whether multi-objective mode is enabled with a candidate list.
 *
 * @param neatInstance - NEAT host exposing objective settings.
 * @returns `true` when multi-objective mode is enabled and an objective list exists.
 */
export function isMultiObjectiveEnabled(
  neatInstance: NeatLikeWithObjectives,
): boolean {
  if (!neatInstance.options.multiObjective?.enabled) {
    return false;
  }

  return Array.isArray(neatInstance.options.multiObjective.objectives);
}

/**
 * Get the configured objective candidates.
 *
 * @param neatInstance - NEAT host exposing objective settings.
 * @returns Objective candidates from configuration.
 */
export function getObjectiveCandidates(
  neatInstance: NeatLikeWithObjectives,
): ObjectiveDescriptor[] {
  return neatInstance.options.multiObjective?.objectives ?? [];
}

/**
 * Validate that an objective descriptor has the required shape.
 *
 * @param candidateObjective - Candidate descriptor to validate.
 * @returns `true` when the descriptor can be used safely.
 */
export function isValidObjective(
  candidateObjective: ObjectiveDescriptor | undefined,
): candidateObjective is ObjectiveDescriptor {
  if (!candidateObjective) {
    return false;
  }

  if (!candidateObjective.key) {
    return false;
  }

  return typeof candidateObjective.accessor === 'function';
}

/**
 * Ensure the multi-objective options container exists.
 *
 * @param neatInstance - NEAT host receiving the multi-objective container.
 * @returns Initialized multi-objective options.
 */
export function ensureMultiObjectiveOptions(
  neatInstance: NeatLikeWithObjectives,
): NonNullable<NeatLikeWithObjectives['options']['multiObjective']> {
  if (!neatInstance.options.multiObjective) {
    neatInstance.options.multiObjective = { enabled: true };
  }

  return neatInstance.options.multiObjective;
}

/**
 * Ensure the objectives list exists on the multi-objective container.
 *
 * @param multiObjectiveOptions - Multi-objective container to hydrate.
 * @returns Objectives list ready for non-destructive operations.
 */
export function ensureObjectivesList(
  multiObjectiveOptions: NonNullable<
    NeatLikeWithObjectives['options']['multiObjective']
  >,
): ObjectiveDescriptor[] {
  if (!multiObjectiveOptions.objectives) {
    multiObjectiveOptions.objectives = [];
  }

  return multiObjectiveOptions.objectives;
}

/**
 * Replace any existing objective with the same key and append the new descriptor.
 *
 * @param objectivesList - Existing objectives to update.
 * @param objectiveKey - Key to replace.
 * @param objectiveDirection - Direction for the new objective.
 * @param objectiveAccessor - Accessor for the new objective.
 * @returns Updated objectives list.
 */
export function replaceObjectiveByKey(
  objectivesList: ObjectiveDescriptor[],
  objectiveKey: string,
  objectiveDirection: 'min' | 'max',
  objectiveAccessor: (genome: GenomeLike) => number,
): ObjectiveDescriptor[] {
  const filteredObjectives = objectivesList.filter(
    (existingObjective) => existingObjective.key !== objectiveKey,
  );

  filteredObjectives.push({
    key: objectiveKey,
    direction: objectiveDirection,
    accessor: objectiveAccessor,
  });

  return filteredObjectives;
}