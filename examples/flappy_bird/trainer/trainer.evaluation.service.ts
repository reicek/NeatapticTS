/**
 * Trainer evaluation compatibility facade.
 *
 * The staged population-evaluation implementation now lives in the dedicated
 * `trainer/evaluation/` submodule so orchestration, scoring helpers, internal
 * contracts, and sub-services can evolve behind a focused boundary.
 *
 * Use this file when you want the public trainer-level shelf for staged
 * evaluation without learning the internal subfolder layout first.
 *
 * Staged evaluation ladder:
 * ```mermaid
 * flowchart LR
 *     Population["population"] --> Quick["quick stage\ncheap shared-seed screen"]
 *     Quick --> Full["full stage\nlonger comparison for survivors"]
 *     Full --> Reeval["reevaluation stage\nlarger anti-luck batch"]
 *     Reeval --> Commit["commitPopulationScores()\nwrite final provisional scores"]
 * ```
 */
export {
  commitPopulationScores,
  evaluatePopulationFullStage,
  evaluatePopulationQuickStage,
  evaluatePopulationReevaluationStage,
} from './evaluation/trainer.evaluation.service';
