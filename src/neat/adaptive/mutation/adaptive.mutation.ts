/**
 * Mutation-side adaptive controllers.
 *
 * This category covers how NEAT changes per-genome mutation pressure and how
 * operator success statistics are decayed so exploration stays responsive over
 * long training runs.
 *
 * The mutation branch of the adaptive subtree answers two related questions:
 * how strongly each genome should be perturbed right now, and how much the
 * controller should still trust older operator-success evidence when choosing
 * future mutations.
 *
 * Read this chapter when you want to understand:
 *
 * - why adaptive mutation rewrites per-genome mutation fields instead of
 *   mutating topology directly,
 * - how cadence checks, score partitions, strategy-specific deltas, and
 *   fallback balancing fit together,
 * - where per-genome mutation-rate tuning stops and operator-stat decay begins.
 *
 * The reading order is easiest to retain as one pressure-maintenance loop:
 *
 * 1. decide whether this generation should adapt at all,
 * 2. resolve settings and partition the scored population,
 * 3. apply strategy-specific rate and amount deltas to genomes,
 * 4. decay operator statistics so later mutation choices weight recent evidence.
 *
 * ```mermaid
 * flowchart TD
 *   Generation[Current generation] --> Cadence[Check adaptation cadence]
 *   Cadence --> Settings[Resolve mutation settings]
 *   Settings --> Partition[Partition scored genomes]
 *   Partition --> Deltas[Apply strategy-specific deltas]
 *   Deltas --> Fallback[Optionally rebalance with two-tier fallback]
 *   Fallback --> Ready[Later mutation stage reads updated per-genome pressure]
 *   Settings --> Decay[Decay operator statistics]
 *   Decay --> Ready
 * ```
 */
export { applyAdaptiveMutation } from '../adaptive';
export {
  applyMutationsToPopulation,
  applyTwoTierFallback,
  resolveMutationSettings,
  shouldAdaptThisGeneration,
  shouldApplyTwoTierFallback,
} from './adaptive.mutation.utils';
export {
  applyOperatorDecay,
  resolveOperatorDecay,
} from './adaptive.operator.utils';
