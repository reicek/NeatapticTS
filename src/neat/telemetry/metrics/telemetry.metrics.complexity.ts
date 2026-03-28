import type {
  GenomeDetailed,
  NeatOptions,
} from '../../shared/neat.shared.types';
import type {
  TelemetryDiversityOptions,
  TelemetryEntryRecord,
} from '../types/telemetry.types';

/**
 * Collect node and connection counts for the population.
 *
 * @param populationSnapshot - Population snapshot.
 * @returns Node and connection counts arrays.
 */
export function collectPopulationCounts(populationSnapshot: GenomeDetailed[]): {
  nodeCounts: number[];
  connectionCounts: number[];
} {
  // Step 1: Map genomes to counts.
  return {
    nodeCounts: populationSnapshot.map((genome) => genome.nodes.length),
    connectionCounts: populationSnapshot.map(
      (genome) => genome.connections.length,
    ),
  };
}

/**
 * Compute mean node and connection counts.
 *
 * @param counts - Node and connection counts arrays.
 * @returns Mean node and connection counts.
 */
export function computeMeanCounts(counts: {
  nodeCounts: number[];
  connectionCounts: number[];
}): { meanNodes: number; meanConns: number } {
  // Step 1: Fold counts into mean values.
  const meanNodes =
    counts.nodeCounts.reduce((sum, value) => sum + value, 0) /
    (counts.nodeCounts.length || 1);
  const meanConns =
    counts.connectionCounts.reduce((sum, value) => sum + value, 0) /
    (counts.connectionCounts.length || 1);
  return { meanNodes, meanConns };
}

/**
 * Compute max node and connection counts.
 *
 * @param counts - Node and connection counts arrays.
 * @returns Max node and connection counts.
 */
export function computeMaxCounts(counts: {
  nodeCounts: number[];
  connectionCounts: number[];
}): { maxNodes: number; maxConns: number } {
  // Step 1: Derive maxima with empty safeguards.
  const maxNodes = counts.nodeCounts.length
    ? Math.max(...counts.nodeCounts)
    : 0;
  const maxConns = counts.connectionCounts.length
    ? Math.max(...counts.connectionCounts)
    : 0;
  return { maxNodes, maxConns };
}

/**
 * Compute enabled ratios per genome.
 *
 * @param populationSnapshot - Population snapshot.
 * @returns Array of enabled ratios.
 */
export function computeEnabledRatios(
  populationSnapshot: GenomeDetailed[],
): number[] {
  // Step 1: Count enabled vs disabled connections per genome.
  return populationSnapshot.map((genome) => {
    let enabledCount = 0;
    let disabledCount = 0;
    for (const connection of genome.connections) {
      if (connection.enabled === false) disabledCount++;
      else enabledCount++;
    }
    const totalCount = enabledCount + disabledCount;
    return totalCount ? enabledCount / totalCount : 0;
  });
}

/**
 * Compute mean of enabled ratios.
 *
 * @param enabledRatios - Enabled ratios per genome.
 * @returns Mean enabled ratio.
 */
export function computeMeanEnabledRatio(enabledRatios: number[]): number {
  // Step 1: Average the ratios.
  return (
    enabledRatios.reduce((sum, value) => sum + value, 0) /
    (enabledRatios.length || 1)
  );
}

/**
 * Compute growth values and store the latest means on the context.
 *
 * @param context - Neat-like context with previous mean values.
 * @param meanCounts - Current mean node/connection counts.
 * @returns Growth values for nodes and connections.
 */
export function computeAndStoreGrowthValues(
  context: {
    _lastMeanNodes?: number;
    _lastMeanConns?: number;
  },
  meanCounts: { meanNodes: number; meanConns: number },
): { growthNodes: number; growthConns: number } {
  // Step 1: Resolve previous mean values.
  const lastMeanNodes = context._lastMeanNodes;
  const lastMeanConns = context._lastMeanConns;

  // Step 2: Compute growth deltas.
  const growthNodes =
    lastMeanNodes !== undefined ? meanCounts.meanNodes - lastMeanNodes : 0;
  const growthConns =
    lastMeanConns !== undefined ? meanCounts.meanConns - lastMeanConns : 0;

  // Step 3: Store current means on the context.
  context._lastMeanNodes = meanCounts.meanNodes;
  context._lastMeanConns = meanCounts.meanConns;

  // Step 4: Return growth deltas.
  return { growthNodes, growthConns };
}

/**
 * Build the complexity entry payload for multi-objective mode.
 *
 * @param telemetryOptions - Options controlling complexity telemetry.
 * @param meanCounts - Mean node/connection counts.
 * @param maxCounts - Max node/connection counts.
 * @param meanEnabledRatio - Mean enabled ratio.
 * @param growthValues - Growth deltas.
 * @returns Complexity entry payload.
 */
export function buildComplexityEntry(
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  meanCounts: { meanNodes: number; meanConns: number },
  maxCounts: { maxNodes: number; maxConns: number },
  meanEnabledRatio: number,
  growthValues: { growthNodes: number; growthConns: number },
): {
  meanNodes: number;
  meanConns: number;
  maxNodes: number;
  maxConns: number;
  meanEnabledRatio: number;
  growthNodes: number;
  growthConns: number;
  budgetMaxNodes: number;
  budgetMaxConns: number;
} {
  // Step 1: Assemble the payload with rounded values.
  return {
    meanNodes: +meanCounts.meanNodes.toFixed(2),
    meanConns: +meanCounts.meanConns.toFixed(2),
    maxNodes: maxCounts.maxNodes,
    maxConns: maxCounts.maxConns,
    meanEnabledRatio: +meanEnabledRatio.toFixed(3),
    growthNodes: +growthValues.growthNodes.toFixed(2),
    growthConns: +growthValues.growthConns.toFixed(2),
    budgetMaxNodes: telemetryOptions.maxNodes ?? 0,
    budgetMaxConns: telemetryOptions.maxConns ?? 0,
  };
}

/**
 * Attach complexity stats for multi-objective mode.
 *
 * @param telemetryContext - Neat-like context with population state.
 * @param telemetryOptions - Options controlling complexity telemetry.
 * @param population - Population snapshot.
 * @param entry - Telemetry entry to update.
 */
export function applyComplexityStatsMultiObjective(
  telemetryContext: {
    _lastMeanNodes?: number;
    _lastMeanConns?: number;
  },
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  population: GenomeDetailed[],
  entry: TelemetryEntryRecord,
): void {
  // Step 1: Skip if complexity telemetry is disabled.
  if (!telemetryOptions.telemetry?.complexity) return;

  // Step 2: Compute complexity stats from the provided population.
  const populationCounts = collectPopulationCounts(population);
  const meanCounts = computeMeanCounts(populationCounts);
  const maxCounts = computeMaxCounts(populationCounts);
  const enabledRatios = computeEnabledRatios(population);
  const meanEnabledRatio = computeMeanEnabledRatio(enabledRatios);
  const growthValues = computeAndStoreGrowthValues(
    telemetryContext,
    meanCounts,
  );
  entry.complexity = buildComplexityEntry(
    telemetryOptions,
    meanCounts,
    maxCounts,
    meanEnabledRatio,
    growthValues,
  );
}

/**
 * Attach complexity stats for mono-objective mode.
 *
 * @param telemetryContext - Neat-like context with population state.
 * @param telemetryOptions - Options controlling complexity telemetry.
 * @param entry - Telemetry entry to update.
 */
export function applyComplexityStatsMonoObjective(
  telemetryContext: {
    _lastMeanNodes?: number;
    _lastMeanConns?: number;
  },
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  populationSnapshot: GenomeDetailed[],
  entry: TelemetryEntryRecord,
): void {
  // Step 1: Skip if complexity telemetry is disabled.
  if (!telemetryOptions.telemetry?.complexity) return;

  // Step 2: Compute complexity stats from the current population.
  const populationCounts = collectPopulationCounts(populationSnapshot);
  const meanCounts = computeMeanCounts(populationCounts);
  const maxCounts = computeMaxCounts(populationCounts);
  const enabledRatios = computeEnabledRatios(populationSnapshot);
  const meanEnabledRatio = computeMeanEnabledRatio(enabledRatios);
  const growthValues = computeAndStoreGrowthValues(
    telemetryContext,
    meanCounts,
  );
  entry.complexity = buildComplexityEntry(
    telemetryOptions,
    meanCounts,
    maxCounts,
    meanEnabledRatio,
    growthValues,
  );
}
