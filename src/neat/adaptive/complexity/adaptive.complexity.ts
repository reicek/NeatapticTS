/**
 * Complexity-control heuristics for adaptive NEAT.
 *
 * This category explains how the engine decides when to grow, pause, or shrink
 * network structure so learners can study budget scheduling separately from
 * mutation or lineage pressure.
 *
 * The folder combines two related but distinct ideas:
 *
 * - budget schedules answer how large the allowed topology should be,
 * - phase toggles answer whether the current mood favors growth or simplification.
 *
 * Read this chapter when you want to understand:
 *
 * - why adaptive complexity keeps a separate structural-budget vocabulary,
 * - how linear and trend-driven schedules differ,
 * - when novelty and score trends are allowed to expand the budget,
 * - how phase flips stay smaller than the broader schedule logic.
 *
 * The reading order is easiest to retain in two steps:
 *
 * 1. start with `adaptive.complexity.utils.ts` for budget growth and shrink rules,
 * 2. then read `adaptive.phases.utils.ts` for phase-state initialization and flips.
 *
 * ```mermaid
 * flowchart TD
 *   Signals[score history, novelty, generation] --> Schedule[Resolve linear or adaptive schedule]
 *   Schedule --> Budgets[Update node and connection budgets]
 *   Signals --> Phase[Initialize or toggle phase]
 *   Budgets --> Later[Later mutation and evolve stages]
 *   Phase --> Later
 * ```
 */
export {
  applyAdaptiveSchedule,
  applyComplexityBudgetSchedule,
  applyLinearSchedule,
} from './adaptive.complexity.utils';
export {
  initializePhaseState,
  togglePhaseIfNeeded,
} from './adaptive.phases.utils';
