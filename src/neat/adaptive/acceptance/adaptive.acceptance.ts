/**
 * Acceptance-gating heuristics for adaptive NEAT.
 *
 * This category focuses on the minimal-criterion threshold that decides when a
 * genome is good enough to stay in play, making it easier to study selection
 * pressure separately from topology growth or mutation schedules.
 *
 * The acceptance controller answers a narrower question than the rest of the
 * adaptive subtree: not how much structure to allow or how strongly to mutate,
 * but how selective the current generation should be before selection starts
 * rewarding it.
 *
 * Read this chapter when you want to understand:
 *
 * - why adaptive acceptance rewrites the current generation instead of only
 *   adjusting future controller policy,
 * - how threshold initialization, acceptance measurement, threshold tuning, and
 *   rejection fit into one same-generation flow,
 * - where the public controller-facing entrypoint stops and the smaller
 *   minimal-criterion bookkeeping helpers begin.
 *
 * The reading order is easiest to retain as one feedback loop:
 *
 * 1. seed the threshold if the run has not used acceptance pressure before,
 * 2. snapshot the current generation's scores,
 * 3. compare observed acceptance against the configured target band,
 * 4. update the threshold and reject genomes that still fall short.
 *
 * ```mermaid
 * flowchart TD
 *   Scores[Current generation scores] --> Init[Initialize threshold if needed]
 *   Init --> Measure[Measure acceptance at current threshold]
 *   Measure --> Tune[Adjust threshold toward target acceptance]
 *   Tune --> Reject[Zero scores below final threshold]
 *   Reject --> Selection[Same-generation selection sees filtered scores]
 * ```
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
