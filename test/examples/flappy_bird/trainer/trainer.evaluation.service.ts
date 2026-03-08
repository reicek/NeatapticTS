/**
 * Trainer evaluation compatibility facade.
 *
 * The staged population-evaluation implementation now lives in the dedicated
 * `trainer/evaluation/` submodule so orchestration, scoring helpers, internal
 * contracts, and sub-services can evolve behind a focused boundary.
 */
export {
  commitPopulationScores,
  evaluatePopulationFullStage,
  evaluatePopulationQuickStage,
  evaluatePopulationReevaluationStage,
} from './evaluation/trainer.evaluation.service';
