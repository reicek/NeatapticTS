/**
 * Acceptance-gating heuristics for adaptive NEAT.
 *
 * This category focuses on the minimal-criterion threshold that decides when a
 * genome is good enough to stay in play, making it easier to study selection
 * pressure separately from topology growth or mutation schedules.
 */
export { applyMinimalCriterionAdaptive } from '../adaptive';
export {
  applyRejection,
  collectScores,
  computeAcceptance,
  initializeThreshold,
  resolveTargetSettings,
  updateThreshold,
} from './adaptive.minimal-criterion.utils';
