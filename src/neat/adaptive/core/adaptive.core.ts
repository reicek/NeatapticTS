/**
 * Shared vocabulary for the adaptive controllers.
 *
 * Read this folder when you need the common constants, config shapes, and
 * runtime contracts that the other adaptive categories build on.
 *
 * The adaptive subtree is easier to understand when each local chapter can stay
 * focused on one control loop. This root file exists so complexity, acceptance,
 * mutation, and lineage helpers can all share one stable language for host
 * fields, option slices, runtime scratch state, and common labels.
 *
 * Read this chapter when you want to understand:
 *
 * - which controller fields adaptive helpers are allowed to inspect or rewrite,
 * - how the major adaptive configuration families are grouped,
 * - why the subtree reuses one shared pool of constants and type aliases.
 *
 * The reading order is easiest to retain in three layers:
 *
 * 1. start with `adaptive.core.types.ts` for the host contract and typed config
 *    slices,
 * 2. read the exported aliases that package those slices for helper files,
 * 3. scan `adaptive.core.constants.ts` for the shared defaults and mode labels.
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
