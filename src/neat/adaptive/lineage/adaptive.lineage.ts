/**
 * Lineage-diversity adaptive controllers.
 *
 * This category explains how ancestor uniqueness telemetry feeds back into the
 * search so the population can recover when family trees become too uniform.
 *
 * The lineage branch of the adaptive subtree is the feedback bridge between
 * recorded telemetry and future search policy. It does not mutate genomes in
 * place for the current generation. Instead it reads ancestry evidence from the
 * latest telemetry snapshot, decides whether lineage diversity has drifted too
 * low or too high, and nudges controller settings that later generations will
 * feel.
 *
 * Read this chapter when you want to understand:
 *
 * - why lineage adaptation depends on telemetry rather than direct population
 *   scans,
 * - how cooldown checks, uniqueness thresholds, and mode-specific nudges fit
 *   together,
 * - where the public controller-facing entrypoint stops and the smaller
 *   telemetry extraction plus adjustment helpers begin.
 *
 * The reading order is easiest to retain as one feedback loop:
 *
 * 1. read the latest ancestor-uniqueness signal from telemetry,
 * 2. check whether the controller is allowed to adjust again yet,
 * 3. compare the signal with the configured low and high thresholds,
 * 4. nudge either dominance epsilon or lineage-pressure strength.
 *
 * ```mermaid
 * flowchart TD
 *   Telemetry[Latest telemetry lineage block] --> Extract[Extract ancestor uniqueness]
 *   Extract --> Cooldown{Cooldown satisfied?}
 *   Cooldown -- No --> Wait[Keep current policy]
 *   Cooldown -- Yes --> Thresholds[Resolve low and high thresholds]
 *   Thresholds --> Mode{Configured adjustment mode}
 *   Mode -- Epsilon --> Epsilon[Nudge dominance epsilon]
 *   Mode -- Lineage --> Pressure[Nudge lineage pressure]
 *   Epsilon --> Future[Future generations see updated policy]
 *   Pressure --> Future
 * ```
 */
export { applyAncestorUniqAdaptive } from '../adaptive';
export {
  applyUniquenessAdjustment,
  extractAncestorUniqueness,
  isCooldownSatisfied,
  resolveUniquenessThresholds,
} from './adaptive.ancestor-uniqueness.utils';
