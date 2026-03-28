/**
 * Index and reading map for the adaptive shared-vocabulary chapter.
 *
 * This file is intentionally thin. It does not define one more adaptive
 * control loop; it points the rest of the subtree at the shared language that
 * complexity, acceptance, mutation, and lineage helpers all reuse.
 *
 * Read this chapter when you want the shortest route into `adaptive/core/`:
 *
 * 1. start with `adaptive.core.types.ts` to see the host contract, option
 *    slices, and normalized working shapes,
 * 2. use the re-exported aliases in this file as the public index into that
 *    contract map,
 * 3. finish with `adaptive.core.constants.ts` for the defaults, labels, and
 *    guard rails that keep those adaptive loops speaking one vocabulary.
 *
 * Think of this file as the chapter's table of contents. The deeper semantics
 * live in the types and constants files; this surface exists so downstream
 * helpers can import one stable adaptive vocabulary from a single place.
 */
export type {
  AdaptiveMutationConfig,
  AncestorUniqAdaptiveConfig,
  ComplexityBudgetConfig,
  Genome,
  MinimalCriterionAdaptiveConfig,
  MutationOutcome,
  MutationPartitions,
  MutationSettings,
  NeatLikeWithAdaptive,
  OperatorAdaptationConfig,
  PhasedComplexityConfig,
} from './adaptive.core.types';
