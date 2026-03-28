import type {
  GenomeLike,
  ObjectiveDescriptor,
} from '../../shared/neat.shared.types';
import type { NeatLikeWithObjectives } from './objectives.types';

/**
 * Objective-list mechanics used by the NEAT controller.
 *
 * This chapter holds the default fitness objective, validation helpers, and
 * the small list-management helpers behind objective registration.
 *
 * The root objectives chapter explains the public API. This core chapter
 * explains the mechanics underneath that public surface: when the default
 * fitness objective appears, how user-provided candidates are filtered,
 * how the multi-objective container is hydrated lazily, and how replacement by
 * key stays non-destructive.
 *
 * Read this chapter when you want to answer questions such as:
 * - Why does the controller still expose a fitness objective even before the
 *   user registers anything custom?
 * - What makes a user objective descriptor safe enough to include?
 * - Why are objective arrays replaced immutably instead of edited in place?
 * - Which helpers are responsible for hydrating missing multi-objective state?
 *
 * The mental model is a four-step resolution flow:
 * 1. decide whether the default fitness objective should exist,
 * 2. collect candidate user objectives when multi-objective mode is active,
 * 3. validate and normalize the candidate list,
 * 4. replace or append objectives by key without mutating the previous list.
 *
 * ```mermaid
 * flowchart TD
 *   Default[Default fitness objective decision] --> Merge[Merge with user candidates]
 *   Candidates[Configured user objectives] --> Merge
 *   Merge --> Validate[Validate safe descriptors]
 *   Validate --> Replace[Replace or append by key]
 *   Replace --> Cached[Later _getObjectives reads resolved list]
 * ```
 *
 * The helper order in this file mirrors that flow. Early helpers decide what
 * must exist by default, middle helpers gather and validate user intent, and
 * later helpers hydrate missing containers or replace one objective without
 * mutating an older list in place.
 */

/**
 * Collect the default objectives when fitness is not suppressed.
 *
 * This helper keeps the root objective story predictable: unless the host has
 * explicitly suppressed fitness, the controller still has one baseline notion
 * of progress even before user-defined objectives are registered.
 *
 * In other words, this helper keeps ordinary single-objective NEAT behavior as
 * the default case and makes suppression an explicit opt-out rather than an
 * accidental side effect of enabling richer objective configuration later.
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
 * User objectives are intentionally gated twice: multi-objective mode must be
 * active, and each candidate must still pass structural validation before it is
 * allowed into the resolved list.
 *
 * That two-stage gate is what keeps the core chapter honest: configuration may
 * contain raw candidates, but the resolved objective list only receives entries
 * that are both contextually enabled and structurally safe.
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
 * The default fitness objective is the anchor that keeps ordinary NEAT runs
 * meaningful even when no richer objective policy has been configured yet.
 * Core resolution builds it on demand so callers do not need a special
 * bootstrap path elsewhere.
 *
 * Keeping this builder separate from the collection logic also makes the core
 * flow easier to teach: one helper decides *whether* the default should exist,
 * this helper decides *what* that default descriptor looks like.
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
 * This guard is intentionally strict: enabling multi-objective mode without an
 * objective array is treated as incomplete configuration, so downstream helpers
 * can avoid reasoning about half-hydrated state.
 *
 * That strictness keeps later helpers simpler because they can assume that a
 * `true` result means both the mode and the candidate container are present.
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
 * This helper exposes the raw configured candidates before validation so the
 * rest of the mechanics layer can separate retrieval from filtering.
 * That separation makes the later validation helpers easier to test and easier
 * to explain in the generated docs.
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
 * Core resolution only requires a stable key and an accessor function. Other
 * fields can remain lightweight because this layer's job is to reject clearly
 * unsafe descriptors, not to impose a heavier policy than the public chapter
 * promises.
 *
 * This is intentionally a safety check, not a semantic ranking policy. It tells
 * the controller whether a descriptor is usable, not whether it is a good idea.
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
 * Objective registration and resolution are allowed to hydrate this container
 * lazily so controller construction does not need to eagerly materialize every
 * optional configuration branch.
 *
 * This is one of the small helpers that keeps the core chapter mutation-safe:
 * later logic can assume the container exists without scattering defensive
 * object creation throughout the file.
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
 * This helper complements container hydration by guaranteeing a writable list
 * surface for later non-destructive replacement logic.
 *
 * It exists so registration code can talk about replacing objectives by key
 * instead of repeatedly guarding against `undefined` lists.
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
 * Replace the existing objective with the same key and append the new descriptor.
 *
 * Replacement is immutable on purpose. Callers get a fresh list with prior
 * matching key removed, which keeps the mechanics predictable and avoids hidden
 * in-place mutations of a list that another part of the controller might still
 * be holding.
 *
 * This is the final fold step in the core flow: once a descriptor is accepted,
 * replacement by key turns that decision into a new resolved list.
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
