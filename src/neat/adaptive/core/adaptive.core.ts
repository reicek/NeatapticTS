/**
 * Reading map and stable index for the adaptive shared-vocabulary chapter.
 *
 * The other adaptive folders explain concrete feedback loops: acceptance
 * rewrites scores, complexity rewrites structural budgets, mutation rewrites
 * per-genome pressure, and lineage rewrites future diversity pressure. This
 * folder exists for the question underneath all of them: what shared language
 * lets those controllers stay narrow without each one inventing its own host
 * contract, option slice, scratch-field names, and fallback constants?
 *
 * That is why `adaptive/core/` should be read as a glossary with ownership,
 * not as a bag of types. `adaptive.core.types.ts` defines what adaptive
 * helpers are allowed to read and rewrite. `adaptive.core.constants.ts`
 * defines the shared defaults, mode names, and numeric guard rails that keep
 * those helpers speaking the same policy dialect. This file is the stable
 * import surface that points later helpers at both shelves without making them
 * depend on file-by-file internals.
 *
 * A useful mental model is to treat the chapter as the protocol layer between
 * adaptive controllers. Complexity, acceptance, mutation, and lineage each own
 * a different control loop, but they still have to agree on the shape of the
 * host, the option slices they may inspect, and the small amount of adaptive
 * state they may persist across generations.
 *
 * ```mermaid
 * flowchart TD
 *   Core[adaptive core index] --> Types[types<br/>host contract and working shapes]
 *   Core --> Constants[constants<br/>defaults modes and guard rails]
 *   Types --> Acceptance[acceptance helpers]
 *   Types --> Complexity[complexity helpers]
 *   Types --> Mutation[mutation helpers]
 *   Types --> Lineage[lineage helpers]
 *   Constants --> Acceptance
 *   Constants --> Complexity
 *   Constants --> Mutation
 *   Constants --> Lineage
 * ```
 *
 * Read the chapter in three passes:
 *
 * 1. start with `adaptive.core.types.ts` when the missing question is what an
 *    adaptive helper may legally read or rewrite,
 * 2. continue through the re-exported aliases in this file when you want the
 *    shortest stable import path for that vocabulary,
 * 3. finish with `adaptive.core.constants.ts` when the missing question is
 *    which defaults, phase labels, thresholds, and clamps keep the adaptive
 *    loops aligned.
 *
 * Example: type a new adaptive helper against the shared host contract instead
 * of a broader `Neat` runtime surface.
 *
 * ```ts
 * import type {
 *   NeatLikeWithAdaptive,
 *   ComplexityBudgetConfig,
 * } from './adaptive.core';
 *
 * function previewBudget(
 *   engine: NeatLikeWithAdaptive,
 *   config: ComplexityBudgetConfig,
 * ) {
 *   return { generation: engine.generation, enabled: config.enabled ?? false };
 * }
 * ```
 *
 * Example: keep constants and contracts on one import path when an extracted
 * helper needs both a typed option slice and a shared mode label.
 *
 * ```ts
 * import type { PhasedComplexityConfig } from './adaptive.core';
 * import { PHASE_COMPLEXIFY } from './adaptive.core.constants';
 *
 * function isComplexifyPhase(config: PhasedComplexityConfig) {
 *   return (config.initialPhase ?? PHASE_COMPLEXIFY) === PHASE_COMPLEXIFY;
 * }
 * ```
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
