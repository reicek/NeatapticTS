import type { ObjectiveDescriptor } from '../../shared/neat.shared.types';

/**
 * Shared contracts for the objectives core chapter.
 *
 * The root `objectives/` chapter explains the public controller API for reading,
 * registering, and clearing objectives. This narrower file defines the minimal
 * host shape that the core resolution helpers actually need in order to build
 * defaults, validate candidates, and manage the cached objective list.
 *
 * Read this chapter when you want to answer questions such as:
 * - Which part of the controller state does objective resolution depend on?
 * - Why does the core mechanics layer use a narrow host contract instead of the
 *   full `Neat` surface?
 * - Which settings influence whether the default fitness objective is kept or
 *   suppressed?
 *
 * This boundary stays intentionally small so the list-resolution helpers remain
 * focused on objective mechanics rather than widening into another public
 * facade.
 *
 * There is only one exported contract here, but it bundles three important
 * state families that the core chapter needs to coordinate:
 *
 * 1. configuration for multi-objective mode,
 * 2. the cached resolved objective list,
 * 3. the explicit switch that suppresses the default fitness objective.
 *
 * Read this file first when you want to understand why the core helpers can
 * stay small and deterministic: they depend on one narrow host seam instead of
 * the whole `Neat` controller.
 */

// Objective-core contracts begin here.

/**
 * Minimal NEAT host contract required by objective-management helpers.
 *
 * Objective resolution only needs multi-objective configuration, a cached
 * objective list, and the flag that suppresses the default fitness objective.
 * Everything else about the controller is intentionally out of scope for this
 * mechanics layer.
 *
 * This host seam is intentionally tiny because the objectives core only answers
 * one controller question: what should the current resolved objective list be?
 * Any state unrelated to that question stays outside the contract.
 */
export interface NeatLikeWithObjectives {
  /** Runtime configuration for multi-objective behavior. */
  options: {
    /** Multi-objective settings. */
    multiObjective?: {
      /** Whether multi-objective mode is enabled. */
      enabled?: boolean;
      /** Registered objective descriptors. */
      objectives?: ObjectiveDescriptor[];
    };
  };
  /** Cached objective list built by `_getObjectives()`. */
  _objectivesList?: ObjectiveDescriptor[];
  /** When true, omits the default fitness objective. */
  _suppressFitnessObjective?: boolean;
}
