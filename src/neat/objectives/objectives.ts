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
 * The root objectives chapter keeps the public registration and resolution
 * methods small, while `core/` holds the validation and list-management logic.
 *
 * - `core/` explains default fitness objectives, user objective filtering, and list replacement rules.
 */

/**
 * Build and return the list of registered objectives for this NEAT instance.
 *
 * @param this - NEAT host exposing multi-objective options and the cached objective list.
 * @returns Objective descriptors in the order they should be applied.
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
 * @param this - NEAT host exposing multi-objective options and the cached objective list.
 * @param key - Unique name for the objective.
 * @param direction - Whether the objective should be minimized or maximized.
 * @param accessor - Function that extracts a numeric value from a genome.
 * @returns Nothing. The host objective configuration is updated in place.
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

  // Step 3: Replace any existing objective with the same key.
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
 * @param this - NEAT host exposing multi-objective options and the cached objective list.
 * @returns Nothing. Registered user objectives and the cached objective list are cleared.
 */
export function clearObjectives(this: NeatLikeWithObjectives): void {
  if (this.options.multiObjective?.objectives) {
    this.options.multiObjective.objectives = [];
  }

  this._objectivesList = undefined;
}
