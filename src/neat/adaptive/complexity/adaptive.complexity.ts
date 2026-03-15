/**
 * Complexity-control heuristics for adaptive NEAT.
 *
 * This category explains how the engine decides when to grow, pause, or shrink
 * network structure so learners can study budget scheduling separately from
 * mutation or lineage pressure.
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
