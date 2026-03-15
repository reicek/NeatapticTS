import type { ObjectiveDescriptor } from '../../shared/neat.shared.types';

/**
 * Minimal NEAT host contract required by objective-management helpers.
 *
 * Objective registration only needs multi-objective configuration, a cached
 * objective list, and the flag that suppresses the default fitness objective.
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
