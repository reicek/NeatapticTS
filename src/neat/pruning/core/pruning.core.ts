import type {
  AdaptivePruningOptions,
  EvolutionPruningOptions,
  NeatLikeForPruning,
  PopulationMetrics,
} from './pruning.types';

/**
 * Pruning mechanics used by scheduled and adaptive NEAT pruning.
 *
 * This chapter holds the policy resolution and metric math that sit underneath
 * the public pruning entrypoints.
 *
 * The public pruning chapter explains *when* pruning is invoked. This file
 * explains how both pruning modes reduce to one shared pipeline:
 *
 * 1. resolve the active policy block,
 * 2. measure the population if adaptive control needs live evidence,
 * 3. compute the target sparsity or shared prune level,
 * 4. apply the resulting pruning instruction across compatible genomes.
 *
 * Scheduled pruning and adaptive pruning differ mainly in where the target
 * comes from. Scheduled pruning derives it from generation timing and a ramp.
 * Adaptive pruning derives it from current population metrics relative to a
 * remembered baseline.
 *
 * ```mermaid
 * flowchart TD
 *   Schedule[Resolve scheduled options] --> Ramp[Compute current target sparsity]
 *   Ramp --> ApplyScheduled[Prune compatible genomes in place]
 *   Adaptive[Resolve adaptive options] --> Metrics[Measure population complexity]
 *   Metrics --> Baseline[Resolve baseline and target remaining metric]
 *   Baseline --> Drift{Drift exceeds tolerance?}
 *   Drift -->|yes| Level[Update shared prune level]
 *   Drift -->|no| Hold[Keep current prune level]
 *   Level --> ApplyAdaptive[Prune compatible genomes in place]
 *   Hold --> ApplyAdaptive
 * ```
 */

/**
 * Resolve scheduled pruning options when they are active for the current generation.
 *
 * This helper is the gatekeeper for the calendar-driven pruning path. It keeps
 * the public pruning wrapper simple by answering one precise question: does the
 * current generation actually belong to the configured pruning schedule?
 *
 * @param host - NEAT host exposing generation and pruning options.
 * @returns Evolution pruning options when active, otherwise `null`.
 */
export function resolveActiveEvolutionPruningOptions(
  host: NeatLikeForPruning,
): EvolutionPruningOptions | null {
  // Step 1: Skip when scheduled pruning is not configured.
  const options = host.options.evolutionPruning ?? null;
  if (!options) {
    return null;
  }

  // Step 2: Skip until the configured start generation is reached.
  const startGeneration = options.startGeneration ?? 0;
  if (host.generation < startGeneration) {
    return null;
  }

  // Step 3: Skip generations outside the configured interval.
  const interval = options.interval ?? 1;
  if ((host.generation - startGeneration) % interval !== 0) {
    return null;
  }

  return options;
}

/**
 * Compute the target sparsity for the current generation.
 *
 * Scheduled pruning ramps toward its configured sparsity target instead of
 * snapping there immediately. This helper converts the current ramp progress
 * into the exact target sparsity the active generation should use.
 *
 * @param host - NEAT host exposing generation state.
 * @param options - Active scheduled pruning options.
 * @returns Target sparsity for the current generation.
 */
export function computeTargetSparsityNow(
  host: NeatLikeForPruning,
  options: EvolutionPruningOptions,
): number {
  // Step 1: Resolve ramp progress through the schedule.
  const rampFraction = computeRampFraction(host, options);

  // Step 2: Scale the configured target by the ramp fraction.
  const configuredTarget = options.targetSparsity ?? 0;
  return configuredTarget * rampFraction;
}

/**
 * Compute the ramp completion fraction for scheduled pruning.
 *
 * The ramp fraction is the soft-start mechanism for scheduled pruning. It lets
 * the controller phase sparsity in gradually over several generations so the
 * population does not experience one abrupt structural shock.
 *
 * @param host - NEAT host exposing generation state.
 * @param options - Active scheduled pruning options.
 * @returns Fraction in `[0, 1]` indicating ramp completion.
 */
export function computeRampFraction(
  host: NeatLikeForPruning,
  options: EvolutionPruningOptions,
): number {
  // Step 1: Return full effect when ramping is disabled.
  const rampGenerations = options.rampGenerations ?? 0;
  if (rampGenerations <= 0) {
    return 1;
  }

  // Step 2: Normalize progress through the ramp window.
  const startGeneration = options.startGeneration ?? 0;
  const rawProgress = (host.generation - startGeneration) / rampGenerations;
  return Math.min(1, Math.max(0, rawProgress));
}

/**
 * Apply scheduled pruning to each genome in the population.
 *
 * By the time this helper runs, policy resolution is already finished. Its job
 * is simply to fan the computed scheduled sparsity target out across genomes
 * that actually implement pruning support.
 *
 * @param host - NEAT host exposing the population.
 * @param options - Active scheduled pruning options.
 * @param targetSparsity - Target sparsity to apply.
 * @returns Nothing. Genomes are pruned in place when supported.
 */
export function applyPruningToPopulation(
  host: NeatLikeForPruning,
  options: EvolutionPruningOptions,
  targetSparsity: number,
): void {
  // Step 1: Resolve the pruning method.
  const method = options.method ?? 'magnitude';

  // Step 2: Apply pruning to each compatible genome.
  for (const genome of host.population) {
    if (!genome || typeof genome.pruneToSparsity !== 'function') {
      continue;
    }
    genome.pruneToSparsity(targetSparsity, method);
  }
}

/**
 * Resolve adaptive pruning options when enabled.
 *
 * This is the narrow on-ramp to the feedback-controller branch. It makes the
 * rest of the adaptive helpers read linearly by collapsing disabled or missing
 * configuration into one `null` check.
 *
 * @param host - NEAT host exposing adaptive pruning options.
 * @returns Adaptive pruning options when enabled, otherwise `null`.
 */
export function resolveActiveAdaptivePruningOptions(
  host: NeatLikeForPruning,
): AdaptivePruningOptions | null {
  // Step 1: Exit when adaptive pruning is not configured or disabled.
  const options = host.options.adaptivePruning ?? null;
  if (!options?.enabled) {
    return null;
  }

  return options;
}

/**
 * Ensure the adaptive pruning state exists on the host.
 *
 * Adaptive pruning keeps one shared prune level on the host so the whole
 * population can react coherently across generations. This helper bootstraps
 * that state once, rather than making every downstream helper repeat the same
 * initialization guard.
 *
 * @param host - NEAT host exposing adaptive pruning state.
 * @returns Nothing. The shared prune level is initialized when missing.
 */
export function initializeAdaptivePruningState(host: NeatLikeForPruning): void {
  // Step 1: Initialize the shared prune level when needed.
  if (host._adaptivePruneLevel === undefined) {
    host._adaptivePruneLevel = 0;
  }
}

/**
 * Compute the population metrics used by adaptive pruning.
 *
 * Adaptive pruning reacts to the population as a whole, not to one genome at a
 * time. This helper produces the small aggregate evidence packet that later
 * helpers use to decide whether complexity is drifting away from the desired
 * sparsity target.
 *
 * @param host - NEAT host exposing the population.
 * @returns Summary of mean node and connection counts.
 */
export function computePopulationMetrics(
  host: NeatLikeForPruning,
): PopulationMetrics {
  // Step 1: Compute mean node and connection counts across the population.
  const meanNodeCount = computeMeanNodeCount(host);
  const meanConnectionCount = computeMeanConnectionCount(host);

  return { meanNodeCount, meanConnectionCount };
}

/**
 * Compute the average node count per genome.
 *
 * Node count is one of the two complexity signals the adaptive controller can
 * watch. It is intentionally averaged so population size changes do not by
 * themselves distort the pruning signal.
 *
 * @param host - NEAT host exposing the population.
 * @returns Average number of nodes per genome.
 */
export function computeMeanNodeCount(host: NeatLikeForPruning): number {
  // Step 1: Sum node counts across genomes.
  const totalNodes = host.population.reduce(
    (accumulator, genome) => accumulator + genome.nodes.length,
    0,
  );

  // Step 2: Normalize by population size.
  return totalNodes / (host.population.length || 1);
}

/**
 * Compute the average connection count per genome.
 *
 * Connection count is the denser complexity signal commonly used for sparsity
 * control. Like node count, it is reduced to a population mean so the adaptive
 * controller reacts to trend rather than to one outlier genome.
 *
 * @param host - NEAT host exposing the population.
 * @returns Average number of connections per genome.
 */
export function computeMeanConnectionCount(host: NeatLikeForPruning): number {
  // Step 1: Sum connection counts across genomes.
  const totalConnections = host.population.reduce(
    (accumulator, genome) => accumulator + genome.connections.length,
    0,
  );

  // Step 2: Normalize by population size.
  return totalConnections / (host.population.length || 1);
}

/**
 * Resolve the currently observed population metric for adaptive pruning.
 *
 * Adaptive pruning can watch either node count or connection count. This helper
 * turns the configured metric name into the actual observed value that the rest
 * of the controller math will compare with the target remaining complexity.
 *
 * @param options - Adaptive pruning options.
 * @param metrics - Population metric summary.
 * @returns Current observed metric value used for adaptation.
 */
export function resolveObservedMetricValue(
  options: AdaptivePruningOptions,
  metrics: PopulationMetrics,
): number {
  // Step 1: Resolve which metric drives adaptation.
  const metricName = options.metric ?? 'connections';
  return metricName === 'nodes'
    ? metrics.meanNodeCount
    : metrics.meanConnectionCount;
}

/**
 * Resolve and persist the adaptive pruning baseline.
 *
 * The baseline is adaptive pruning's memory of where the population started
 * when the controller first engaged. Later drift calculations are measured
 * against that remembered baseline rather than against a moving target.
 *
 * @param host - NEAT host exposing adaptive baseline state.
 * @param currentMetricValue - Currently observed metric value.
 * @returns Baseline metric value used for adaptation.
 */
export function resolveAdaptivePruneBaseline(
  host: NeatLikeForPruning,
  currentMetricValue: number,
): number {
  // Step 1: Initialize the baseline on the first run.
  if (host._adaptivePruneBaseline === undefined) {
    host._adaptivePruneBaseline = currentMetricValue;
  }

  return host._adaptivePruneBaseline;
}

/**
 * Compute the target remaining metric implied by the desired sparsity.
 *
 * Adaptive pruning expresses its goal as desired sparsity, but the feedback loop
 * compares live complexity metrics. This helper bridges those two views by
 * translating the baseline metric into the remaining amount of structure the
 * controller wants to keep.
 *
 * @param options - Adaptive pruning options.
 * @param adaptivePruneBaseline - Baseline metric value.
 * @returns Target remaining metric value.
 */
export function computeTargetRemainingMetric(
  options: AdaptivePruningOptions,
  adaptivePruneBaseline: number,
): number {
  // Step 1: Resolve the desired sparsity target.
  const desiredSparsity = options.targetSparsity ?? 0.5;

  // Step 2: Convert sparsity into a remaining metric target.
  return adaptivePruneBaseline * (1 - desiredSparsity);
}

/**
 * Decide whether adaptive pruning should adjust the prune level.
 *
 * This is the dead-band check for the adaptive controller. Small fluctuations
 * around the target are ignored so the prune level does not chatter on every
 * minor metric wobble.
 *
 * @param options - Adaptive pruning options.
 * @param currentMetricValue - Current observed metric value.
 * @param targetRemainingMetric - Target remaining metric value.
 * @param adaptivePruneBaseline - Baseline metric value.
 * @returns `true` when the normalized drift exceeds the configured tolerance.
 */
export function shouldAdjustAdaptivePruning(
  options: AdaptivePruningOptions,
  currentMetricValue: number,
  targetRemainingMetric: number,
  adaptivePruneBaseline: number,
): boolean {
  // Step 1: Resolve the tolerance band.
  const tolerance = options.tolerance ?? 0.05;

  // Step 2: Measure normalized drift from the target remaining metric.
  const normalizedDifference =
    (currentMetricValue - targetRemainingMetric) / (adaptivePruneBaseline || 1);

  return Math.abs(normalizedDifference) > tolerance;
}

/**
 * Compute the next adaptive prune level.
 *
 * Once the controller decides that drift is large enough, this helper turns the
 * direction of that drift into a bounded update of the shared prune level.
 * Higher-than-target complexity tightens pruning; lower-than-target complexity
 * relaxes it.
 *
 * @param options - Adaptive pruning options.
 * @param currentPruneLevel - Current shared prune level.
 * @param currentMetricValue - Current observed metric value.
 * @param targetRemainingMetric - Target remaining metric value.
 * @returns Updated prune level clamped into the valid sparsity range.
 */
export function computeNextAdaptivePruneLevel(
  options: AdaptivePruningOptions,
  currentPruneLevel: number,
  currentMetricValue: number,
  targetRemainingMetric: number,
): number {
  // Step 1: Resolve the adjustment rate and desired sparsity.
  const adjustRate = options.adjustRate ?? 0.02;
  const desiredSparsity = options.targetSparsity ?? 0.5;

  // Step 2: Determine whether pruning should increase or relax.
  const adjustmentDirection =
    currentMetricValue > targetRemainingMetric ? 1 : -1;

  // Step 3: Apply the adjustment and clamp to the valid range.
  const proposedPruneLevel =
    currentPruneLevel + adjustRate * adjustmentDirection;
  return Math.max(0, Math.min(desiredSparsity, proposedPruneLevel));
}

/**
 * Apply the shared adaptive prune level to every compatible genome.
 *
 * This is the final fan-out step for adaptive pruning. The host maintains one
 * shared prune level, and this helper applies that single controller decision
 * uniformly across genomes that support sparsity pruning.
 *
 * @param host - NEAT host exposing the population.
 * @param pruneLevel - Prune level to apply.
 * @returns Nothing. Compatible genomes are pruned in place.
 */
export function applyAdaptivePruneLevelToPopulation(
  host: NeatLikeForPruning,
  pruneLevel: number,
): void {
  // Step 1: Apply pruning to each compatible genome.
  for (const genome of host.population) {
    if (typeof genome.pruneToSparsity !== 'function') {
      continue;
    }
    genome.pruneToSparsity(pruneLevel, 'magnitude');
  }
}
