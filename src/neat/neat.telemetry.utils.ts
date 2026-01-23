import { EPSILON } from './neat.constants';
import type {
  ConnectionLike,
  NeatOptions,
  NodeLike,
  TelemetryEntry,
  GenomeDetailed,
  ObjImportance,
  SpeciesAlloc,
  ObjectiveEvent,
} from './neat.types';
import type { NeatLineageContext as LineageContext } from './neat.lineage';
import { buildAnc, computeAncestorUniqueness } from './neat.lineage';

/** Minimal genome shape used by telemetry helpers. */
export type TelemetryGenome = {
  nodes: NodeLike[];
  connections: ConnectionLike[];
  _depth?: number;
};

/** Diversity telemetry options for sampling and novelty defaults. */
export type TelemetryDiversityOptions = {
  diversityMetrics?: {
    enabled?: boolean;
    pairSample?: number;
    graphletSample?: number;
  };
  fastMode?: boolean;
  novelty?: { enabled?: boolean; k?: number };
};

/** Minimal telemetry stream options for streaming helpers. */
export type TelemetryStreamOptions = {
  telemetryStream?: {
    enabled?: boolean;
    onEntry?: (entry: TelemetryEntry) => void;
  };
};

/** Telemetry entry shape used for constructing snapshots. */
export type TelemetryEntryRecord = TelemetryEntry & Record<string, unknown>;

/** Operator stats map shape for telemetry extraction. */
export type OperatorStatsMap = Map<
  string,
  { success: number; attempts: number }
>;

/** Minimal telemetry buffer context shape. */
export type TelemetryBufferContext = {
  _telemetry?: TelemetryEntry[];
};

/** Minimal telemetry selection context shape. */
export type TelemetrySelectContext = {
  _telemetrySelect?: Set<string>;
};

/**
 * Core telemetry field keys used by selection helpers.
 */
export type TelemetryCoreFields = readonly string[];

/**
 * Read a cached entropy value if it exists and belongs to the current
 * generation.
 *
 * @param generation - Current generation number.
 * @param entropyGraph - Genome-like graph object.
 * @returns Cached entropy number, or undefined when not available.
 */
export function getCachedEntropy(
  generation: number | undefined,
  entropyGraph: Record<string, unknown>,
): number | undefined {
  // Step 1: Accept cache only when generation matches.
  if (entropyGraph._entropyGen !== generation) return;
  // Step 2: Accept cache only when value is numeric.
  if (typeof entropyGraph._entropyVal !== 'number') return;
  // Step 3: Return cached entropy.
  return entropyGraph._entropyVal as number;
}

/**
 * Compute per-node degree counts for enabled connections.
 *
 * @param entropyGraph - Genome-like graph object.
 * @returns Map geneId -> degree count.
 */
export function computeDegreeCounts(entropyGraph: {
  nodes: Array<{ geneId: number }>;
  connections: Array<{
    from: { geneId: number };
    to: { geneId: number };
    enabled: boolean;
  }>;
}): Record<number, number> {
  // Step 1: Seed degree counts with all node ids at 0.
  const degreeCounts: Record<number, number> = {};
  for (const node of entropyGraph.nodes) degreeCounts[node.geneId] = 0;

  // Step 2: Add degree contributions from enabled connections.
  for (const connection of entropyGraph.connections) {
    if (!connection.enabled) continue;
    const sourceGeneId = connection.from.geneId;
    const targetGeneId = connection.to.geneId;
    if (degreeCounts[sourceGeneId] !== undefined) degreeCounts[sourceGeneId]!++;
    if (degreeCounts[targetGeneId] !== undefined) degreeCounts[targetGeneId]!++;
  }

  // Step 3: Return the degree count table.
  return degreeCounts;
}

/**
 * Build a histogram of degree frequencies from a degree-count table.
 *
 * @param counts - Map geneId -> degree count.
 * @returns Map degree -> number of nodes with that degree.
 */
export function buildDegreeHistogram(
  counts: Record<number, number>,
): Record<number, number> {
  // Step 1: Convert node degrees into degree-frequency buckets.
  const degreeHistogram: Record<number, number> = {};
  for (const degree of Object.values(counts)) {
    degreeHistogram[degree] = (degreeHistogram[degree] ?? 0) + 1;
  }
  // Step 2: Return the histogram.
  return degreeHistogram;
}

/**
 * Compute entropy from a degree-frequency histogram.
 *
 * @param histogram - Map degree -> number of nodes.
 * @param totalNodes - Total node count used to normalize into probabilities.
 * @returns Entropy value (non-negative).
 */
export function computeEntropyFromHistogram(
  histogram: Record<number, number>,
  totalNodes: number,
): number {
  // Step 1: Fold histogram into entropy scalar.
  let entropy = 0;
  for (const [degree, frequency] of Object.entries(histogram)) {
    void degree;
    const probability = frequency / totalNodes;
    if (probability > 0)
      entropy -= probability * Math.log(probability + EPSILON);
  }
  // Step 2: Return the computed entropy.
  return entropy;
}

/**
 * Cache an entropy value for the current generation on the graph object.
 *
 * @param generation - Current generation number.
 * @param entropyGraph - Genome-like graph object.
 * @param entropyValue - Entropy value to cache.
 */
export function setCachedEntropy(
  generation: number | undefined,
  entropyGraph: Record<string, unknown>,
  entropyValue: number,
): void {
  // Step 1: Tag cache with generation.
  entropyGraph._entropyGen = generation;
  // Step 2: Store computed entropy.
  entropyGraph._entropyVal = entropyValue;
}

/**
 * Build a snapshot of the core telemetry fields present on the entry; does
 * not mutate the source entry.
 *
 * @param sourceEntry - Source telemetry object.
 * @param fields - Core telemetry field keys to preserve.
 * @returns Shallow snapshot of core fields that exist on the entry.
 */
export function getTelemetryCoreSnapshot(
  sourceEntry: Record<string, unknown>,
  fields: TelemetryCoreFields,
): Partial<Record<string, unknown>> {
  // Step 1: Initialize an empty snapshot for the core fields.
  const coreSnapshot: Partial<Record<string, unknown>> = {};

  // Step 2: Capture only core fields that exist on the source entry.
  for (const field of fields) {
    if (Object.hasOwn(sourceEntry, field)) {
      coreSnapshot[field] = sourceEntry[field];
    }
  }

  // Step 3: Return the shallow snapshot without mutating the source entry.
  return coreSnapshot;
}

/**
 * Remove non-core keys that are not whitelisted by the selection set.
 * Mutates the provided entry in-place for efficiency.
 *
 * @param sourceEntry - Telemetry entry being filtered.
 * @param selection - Whitelist of additional telemetry keys.
 * @param fields - Core telemetry field keys that must be preserved.
 * @returns The same entry reference after filtering.
 */
export function stripUnselectedTelemetryKeys(
  sourceEntry: Record<string, unknown>,
  selection: Set<string>,
  fields: TelemetryCoreFields,
): Record<string, unknown> {
  // Step 1: Walk current keys to decide which fields remain.
  for (const key of Object.keys(sourceEntry)) {
    // Step 2: Always keep core fields regardless of selection.
    if (fields.includes(key)) continue;
    // Step 3: Remove non-core keys not present in the selection set.
    if (!selection.has(key)) {
      delete sourceEntry[key];
    }
  }

  // Step 4: Return the same entry reference after filtering.
  return sourceEntry;
}

/**
 * Re-attach core fields to the filtered entry.
 * Mutates the entry so the caller keeps the original reference.
 *
 * @param sourceEntry - Filtered telemetry entry to update.
 * @param coreSnapshot - Snapshot of core fields to ensure presence.
 * @returns The same entry reference with core fields restored.
 */
export function mergeTelemetryCoreFields(
  sourceEntry: Record<string, unknown>,
  coreSnapshot: Partial<Record<string, unknown>>,
): Record<string, unknown> {
  // Step 1: Re-apply core fields to ensure presence and ordering.
  for (const [key, value] of Object.entries(coreSnapshot)) {
    sourceEntry[key] = value;
  }

  // Step 2: Return the same entry reference for call-site chaining.
  return sourceEntry;
}

/**
 * Apply fast-mode tuning to diversity sampling and novelty defaults.
 *
 * @param telemetryContext - Context object storing fast-mode tuning flag.
 * @param telemetryOptions - Options with diversity and novelty settings.
 */
export function applyFastModeDefaults(
  telemetryContext: { _fastModeTuned?: boolean },
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
): void {
  // Step 1: Skip when fast mode is disabled or already tuned.
  if (!telemetryOptions.fastMode) return;
  if (telemetryContext._fastModeTuned) return;

  // Step 2: Ensure default sampling values are populated.
  const diversityMetrics = telemetryOptions.diversityMetrics;
  if (diversityMetrics) {
    if (diversityMetrics.pairSample == null) diversityMetrics.pairSample = 20;
    if (diversityMetrics.graphletSample == null)
      diversityMetrics.graphletSample = 30;
  }

  // Step 3: Ensure novelty neighborhood defaults are populated.
  if (telemetryOptions.novelty?.enabled && telemetryOptions.novelty.k == null)
    telemetryOptions.novelty.k = 5;

  // Step 4: Mark tuning as applied.
  telemetryContext._fastModeTuned = true;
}

/**
 * Compute pairwise compatibility statistics via sampling.
 *
 * @param genomes - Population snapshot.
 * @param size - Population size.
 * @param pairSampleCount - Number of pairs to sample.
 * @param rngFactoryFn - RNG factory returning a uniform random function.
 * @param compatibilityDistance - Optional compatibility distance function.
 * @returns Mean and variance of sampled compatibilities.
 */
export function computeCompatibilityStats(
  genomes: TelemetryGenome[],
  size: number,
  pairSampleCount: number,
  rngFactoryFn: () => () => number,
  compatibilityDistance?: (a: TelemetryGenome, b: TelemetryGenome) => number,
): { meanCompat: number; varCompat: number } {
  // Step 1: Accumulate sampled distances.
  let compatibilitySum = 0;
  let compatibilitySumSq = 0;
  let compatibilityCount = 0;
  for (let sampleIndex = 0; sampleIndex < pairSampleCount; sampleIndex++) {
    if (size < 2) break;
    const rng = rngFactoryFn();
    const firstIndex = Math.floor(rng() * size);
    let secondIndex = Math.floor(rng() * size);
    if (secondIndex === firstIndex) secondIndex = (secondIndex + 1) % size;
    const distance = compatibilityDistance
      ? compatibilityDistance(genomes[firstIndex], genomes[secondIndex])
      : 0;
    compatibilitySum += distance;
    compatibilitySumSq += distance * distance;
    compatibilityCount++;
  }

  // Step 2: Derive mean and variance from the sampled distances.
  const meanCompat = compatibilityCount
    ? compatibilitySum / compatibilityCount
    : 0;
  const varCompat = compatibilityCount
    ? Math.max(
        0,
        compatibilitySumSq / compatibilityCount - meanCompat * meanCompat,
      )
    : 0;

  // Step 3: Return the computed stats.
  return { meanCompat, varCompat };
}

/**
 * Compute structural entropy mean and variance across the population.
 *
 * @param genomes - Population snapshot.
 * @param structuralEntropyFn - Function to compute entropy for a genome.
 * @returns Mean and variance of entropy values.
 */
export function computeEntropyStats(
  genomes: TelemetryGenome[],
  structuralEntropyFn: (genome: TelemetryGenome) => number,
): { meanEntropy: number; varEntropy: number } {
  // Step 1: Compute entropy per genome.
  const entropies = genomes.map((genome) => structuralEntropyFn(genome));

  // Step 2: Compute mean entropy.
  const meanEntropy =
    entropies.reduce((sum, value) => sum + value, 0) / (entropies.length || 1);

  // Step 3: Compute variance of entropy.
  const varEntropy = entropies.length
    ? entropies.reduce(
        (sum, value) => sum + (value - meanEntropy) * (value - meanEntropy),
        0,
      ) / entropies.length
    : 0;

  // Step 4: Return the entropy stats.
  return { meanEntropy, varEntropy };
}

/**
 * Sample graphlet motifs and compute entropy over their edge counts.
 *
 * @param genomes - Population snapshot.
 * @param size - Population size.
 * @param graphletSampleCount - Number of graphlets to sample.
 * @param rngFactoryFn - RNG factory returning a uniform random function.
 * @returns Graphlet entropy value.
 */
export function computeGraphletEntropy(
  genomes: TelemetryGenome[],
  size: number,
  graphletSampleCount: number,
  rngFactoryFn: () => () => number,
): number {
  // Step 1: Initialize motif buckets for 0..3 edges.
  const motifCounts = [0, 0, 0, 0];

  // Step 2: Sample graphlets.
  for (let sampleIndex = 0; sampleIndex < graphletSampleCount; sampleIndex++) {
    if (size === 0) break;
    const rng = rngFactoryFn();
    const genome = genomes[Math.floor(rng() * size)];
    if (!genome) break;
    if (genome.nodes.length < 3) continue;

    const selectedNodeIndices = pickDistinctIndices(
      genome.nodes.length,
      3,
      rng,
    );
    const selectedNodes = selectedNodeIndices.map(
      (nodeIndex) => genome.nodes[nodeIndex],
    );
    const edgeCount = countEnabledEdges(genome, selectedNodes);
    motifCounts[edgeCount]++;
  }

  // Step 3: Convert motif counts to entropy.
  const totalMotifs = motifCounts.reduce((sum, value) => sum + value, 0) || 1;
  let graphletEntropy = 0;
  for (let motifIndex = 0; motifIndex < motifCounts.length; motifIndex++) {
    const probability = motifCounts[motifIndex] / totalMotifs;
    if (probability > 0) graphletEntropy -= probability * Math.log(probability);
  }

  // Step 4: Return the graphlet entropy value.
  return graphletEntropy;
}

/**
 * Compute lineage depth and pairwise depth-distance statistics.
 *
 * @param lineageEnabled - Whether lineage metrics are enabled.
 * @param genomes - Population snapshot.
 * @param size - Population size.
 * @param pairSampleCount - Number of pairs to sample.
 * @param rngFactoryFn - RNG factory returning a uniform random function.
 * @returns Lineage mean depth and pairwise distance.
 */
export function computeLineageStats(
  lineageEnabled: boolean,
  genomes: TelemetryGenome[],
  size: number,
  pairSampleCount: number,
  rngFactoryFn: () => () => number,
): { lineageMeanDepth: number; lineageMeanPairDist: number } {
  // Step 1: Short-circuit when lineage is disabled or population is empty.
  if (!lineageEnabled || size === 0)
    return { lineageMeanDepth: 0, lineageMeanPairDist: 0 };

  // Step 2: Compute mean depth.
  const depths = genomes.map((genome) => genome._depth ?? 0);
  const lineageMeanDepth = depths.reduce((sum, value) => sum + value, 0) / size;

  // Step 3: Sample depth distances across pairs.
  let lineagePairSum = 0;
  let lineagePairCount = 0;
  const pairsToSample = Math.min(pairSampleCount, (size * (size - 1)) / 2);
  for (let sampleIndex = 0; sampleIndex < pairsToSample; sampleIndex++) {
    if (size < 2) break;
    const rng = rngFactoryFn();
    const firstIndex = Math.floor(rng() * size);
    let secondIndex = Math.floor(rng() * size);
    if (secondIndex === firstIndex) secondIndex = (secondIndex + 1) % size;
    lineagePairSum += Math.abs(depths[firstIndex] - depths[secondIndex]);
    lineagePairCount++;
  }

  // Step 4: Fold into mean pairwise distance.
  const lineageMeanPairDist = lineagePairCount
    ? lineagePairSum / lineagePairCount
    : 0;

  // Step 5: Return lineage stats.
  return { lineageMeanDepth, lineageMeanPairDist };
}

/**
 * Pick a fixed number of distinct random indices.
 *
 * @param upperBound - Exclusive upper bound for random indices.
 * @param count - Number of distinct indices to pick.
 * @param rng - RNG function returning values in [0,1).
 * @returns Array of distinct indices.
 */
export function pickDistinctIndices(
  upperBound: number,
  count: number,
  rng: () => number,
): number[] {
  // Step 1: Gather unique indices using a set.
  const selectedIndices = new Set<number>();
  while (selectedIndices.size < count)
    selectedIndices.add(Math.floor(rng() * upperBound));

  // Step 2: Return indices as an array.
  return Array.from(selectedIndices);
}

/**
 * Count enabled edges between the selected nodes in a genome.
 *
 * @param genome - Genome with connections to inspect.
 * @param selectedNodes - Nodes forming the graphlet sample.
 * @returns Edge count capped at 3.
 */
export function countEnabledEdges(
  genome: TelemetryGenome,
  selectedNodes: NodeLike[],
): number {
  // Step 1: Count enabled edges among selected nodes.
  let edgeCount = 0;
  for (const connection of genome.connections) {
    if (!connection.enabled) continue;
    if (
      selectedNodes.includes(connection.from) &&
      selectedNodes.includes(connection.to)
    ) {
      edgeCount++;
    }
  }

  // Step 2: Cap the edge count at 3.
  return Math.min(edgeCount, 3);
}

/**
 * Apply telemetry selection while swallowing any selection errors.
 *
 * @param telemetryContext - Neat-like context with telemetry selection.
 * @param telemetryEntry - Entry to filter in place.
 * @param applyTelemetrySelectFn - Selection helper to invoke.
 */
export function safelyApplyTelemetrySelect<
  TContext extends TelemetrySelectContext,
>(
  telemetryContext: TContext,
  telemetryEntry: TelemetryEntry,
  applyTelemetrySelectFn: (
    this: TContext,
    entry: Record<string, unknown>,
  ) => Record<string, unknown>,
): void {
  // Step 1: Attempt to apply selection; ignore failures.
  try {
    applyTelemetrySelectFn.call(
      telemetryContext,
      telemetryEntry as Record<string, unknown>,
    );
  } catch {
    // ignored: telemetry selection errors are non-fatal and safe to drop
  }
}

/**
 * Ensure the telemetry buffer is initialized.
 *
 * @param telemetryContext - Neat-like context holding telemetry buffer.
 * @returns A mutable telemetry buffer.
 */
export function ensureTelemetryBuffer(
  telemetryContext: TelemetryBufferContext,
): TelemetryEntry[] {
  // Step 1: Initialize buffer if missing.
  if (!telemetryContext._telemetry) telemetryContext._telemetry = [];
  // Step 2: Return the buffer reference.
  return telemetryContext._telemetry;
}

/**
 * Stream telemetry entry when a stream callback is configured.
 *
 * @param telemetryContext - Neat-like context with stream settings.
 * @param telemetryEntry - Entry to stream.
 */
export function safelyStreamTelemetryEntry(
  telemetryContext: { options?: TelemetryStreamOptions },
  telemetryEntry: TelemetryEntry,
): void {
  // Step 1: Check if streaming is enabled and the callback is valid.
  try {
    const telemetryStream = telemetryContext.options?.telemetryStream;
    if (
      telemetryStream?.enabled &&
      typeof telemetryStream.onEntry === 'function'
    ) {
      telemetryStream.onEntry(telemetryEntry);
    }
  } catch {
    // ignored: telemetry stream callback errors should not disrupt evolution
  }
}

/**
 * Trim the telemetry buffer to a maximum size.
 *
 * @param telemetryBufferRef - Buffer to trim in-place.
 * @param maxEntries - Maximum entries to keep.
 */
export function trimTelemetryBuffer(
  telemetryBufferRef: TelemetryEntry[],
  maxEntries: number,
): void {
  // Step 1: Drop oldest entries when over limit.
  if (telemetryBufferRef.length > maxEntries) telemetryBufferRef.shift();
}

/**
 * Snapshot operator statistics into a telemetry-friendly array.
 *
 * @param operatorStats - Operator stats map (opName -> success/attempts).
 * @returns Operator stats snapshot array.
 */
export function computeOperatorStatsSnapshot(
  operatorStats: OperatorStatsMap | undefined,
): Array<{ op: string; succ: number; att: number }> {
  // Step 1: Convert operator stats map into a list.
  return Array.from((operatorStats ?? new Map()).entries()).map(
    ([operationName, stats]) => ({
      op: operationName,
      succ: stats.success,
      att: stats.attempts,
    }),
  );
}

/**
 * Compute a hypervolume-like proxy for the Pareto front.
 *
 * @param telemetryOptions - Options controlling complexity metric.
 * @param population - Population snapshot.
 * @returns Hypervolume proxy value.
 */
export function computeHyperVolumeProxy(
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  population: GenomeDetailed[],
): number {
  // Step 1: Resolve complexity metric.
  const complexityMetric: 'nodes' | 'connections' =
    telemetryOptions.multiObjective?.complexityMetric || 'connections';

  // Step 2: Compute normalization bounds for primary objective scores.
  const primaryObjectiveScores = population.map(
    (genome) => (genome.score as number) || 0,
  );
  const minPrimaryScore = Math.min(...primaryObjectiveScores);
  const maxPrimaryScore = Math.max(...primaryObjectiveScores);

  // Step 3: Fold Pareto-front genomes into the hypervolume proxy.
  let hyperVolumeProxy = 0;
  for (const genome of population) {
    const rank = genome._moRank ?? 0;
    if (rank !== 0) continue;
    const normalizedScore =
      maxPrimaryScore > minPrimaryScore
        ? ((genome.score || 0) - minPrimaryScore) /
          (maxPrimaryScore - minPrimaryScore)
        : 0;
    const genomeComplexity =
      complexityMetric === 'nodes'
        ? genome.nodes.length
        : genome.connections.length;
    hyperVolumeProxy += normalizedScore * (1 / (genomeComplexity + 1));
  }

  // Step 4: Return the computed proxy value.
  return hyperVolumeProxy;
}

/**
 * Compute sizes of early Pareto fronts.
 *
 * @param population - Population snapshot.
 * @returns Array of front sizes (rank 0..4).
 */
export function computeParetoFrontSizes(
  population: GenomeDetailed[],
): number[] {
  // Step 1: Collect sizes for the first few ranks.
  const paretoFrontSizes: number[] = [];
  for (let rankIndex = 0; rankIndex < 5; rankIndex++) {
    const frontSize = population.filter(
      (genome) => ((genome as GenomeDetailed)._moRank ?? 0) === rankIndex,
    ).length;
    if (!frontSize) break;
    paretoFrontSizes.push(frontSize);
  }

  // Step 2: Return the collected sizes.
  return paretoFrontSizes;
}

/**
 * Apply the most recent objective importance snapshot.
 *
 * @param telemetryContext - Neat-like context with objective importance.
 * @param entry - Telemetry entry to update.
 */
export function applyObjectiveImportance(
  telemetryContext: { _lastObjImportance?: ObjImportance },
  entry: TelemetryEntryRecord,
): void {
  // Step 1: Default to an empty importance map.
  if (!entry.objImportance) entry.objImportance = {};

  // Step 2: Apply the latest importance snapshot when available.
  if (telemetryContext._lastObjImportance) {
    const lastImportance = telemetryContext._lastObjImportance;
    if (typeof lastImportance === 'object' && lastImportance !== null)
      entry.objImportance = lastImportance;
  }
}

/**
 * Apply objective age snapshots to the entry.
 *
 * @param telemetryContext - Neat-like context with objective ages.
 * @param entry - Telemetry entry to update.
 */
export function applyObjectiveAges(
  telemetryContext: { _objectiveAges?: Map<string, number> },
  entry: TelemetryEntryRecord,
): void {
  // Step 1: Attach objective ages when available.
  if (telemetryContext._objectiveAges?.size) {
    entry.objAges = Object.fromEntries(
      telemetryContext._objectiveAges.entries(),
    );
  }
}

/**
 * Apply and flush objective lifecycle events.
 *
 * @param telemetryContext - Neat-like context holding objective events.
 * @param entry - Telemetry entry to update.
 * @param generation - Generation index for event records.
 */
export function applyObjectiveEvents(
  telemetryContext: {
    _pendingObjectiveAdds?: string[];
    _pendingObjectiveRemoves?: string[];
    _objectiveEvents?: ObjectiveEvent[];
  },
  entry: TelemetryEntryRecord,
  generation: number,
): void {
  // Step 1: Skip if there are no pending events.
  if (
    !telemetryContext._pendingObjectiveAdds?.length &&
    !telemetryContext._pendingObjectiveRemoves?.length
  )
    return;

  // Step 2: Build event list from pending changes.
  entry.objEvents = [];
  for (const objectiveKey of telemetryContext._pendingObjectiveAdds || []) {
    entry.objEvents.push({ gen: generation, type: 'add', key: objectiveKey });
  }
  for (const objectiveKey of telemetryContext._pendingObjectiveRemoves || []) {
    entry.objEvents.push({
      gen: generation,
      type: 'remove',
      key: objectiveKey,
    });
  }

  // Step 3: Persist events and clear pending arrays.
  telemetryContext._objectiveEvents = telemetryContext._objectiveEvents || [];
  telemetryContext._objectiveEvents.push(...entry.objEvents);
  telemetryContext._pendingObjectiveAdds = [];
  telemetryContext._pendingObjectiveRemoves = [];
}

/**
 * Apply per-species offspring allocation snapshot.
 *
 * @param telemetryContext - Neat-like context with allocation snapshot.
 * @param entry - Telemetry entry to update.
 */
export function applySpeciesAllocation(
  telemetryContext: { _lastOffspringAlloc?: SpeciesAlloc[] },
  entry: TelemetryEntryRecord,
): void {
  // Step 1: Attach last allocation snapshot when present.
  if (telemetryContext._lastOffspringAlloc) {
    const lastAlloc = telemetryContext._lastOffspringAlloc;
    if (Array.isArray(lastAlloc)) entry.speciesAlloc = lastAlloc.slice();
  }
}

/**
 * Apply objectives list snapshot (keys only).
 *
 * @param telemetryContext - Neat-like context with objective provider.
 * @param entry - Telemetry entry to update.
 */
export function applyObjectivesSnapshot(
  telemetryContext: { _getObjectives?: () => { key: string }[] },
  entry: TelemetryEntryRecord,
): void {
  // Step 1: Attempt to read objective keys; ignore absence.
  try {
    entry.objectives =
      telemetryContext._getObjectives?.().map((objective) => objective.key) ||
      [];
  } catch {
    // ignored: objective provider not present — skip objectives snapshot
  }
}

/**
 * Attach RNG state when configured.
 *
 * @param telemetryContext - Neat-like context with RNG state.
 * @param telemetryOptions - Options controlling RNG telemetry.
 * @param entry - Telemetry entry to update.
 */
export function applyRngState(
  telemetryContext: { _rngState?: unknown },
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  entry: TelemetryEntryRecord,
): void {
  // Step 1: Attach RNG state only when enabled and present.
  if (telemetryOptions.rngState && telemetryContext._rngState !== undefined)
    entry.rng = telemetryContext._rngState as number | undefined;
}

/**
 * Apply lineage stats for multi-objective mode using ancestor uniqueness.
 *
 * @param telemetryContext - Neat-like context with lineage settings.
 * @param population - Population snapshot.
 * @param entry - Telemetry entry to update.
 */
export function applyLineageStatsMultiObjective(
  telemetryContext: {
    _lineageEnabled?: boolean;
    _getRNG?: () => () => number;
    _lastMeanDepth?: number;
    _prevInbreedingCount?: number;
  },
  population: GenomeDetailed[],
  entry: TelemetryEntryRecord,
): void {
  // Step 1: Skip when lineage is disabled or population is empty.
  if (!telemetryContext._lineageEnabled || population.length === 0) return;

  // Step 2: Compute depth metrics.
  const bestGenome = population[0] as GenomeDetailed;
  const depths = population.map((genome) => genome._depth ?? 0);
  telemetryContext._lastMeanDepth =
    depths.reduce((sum, value) => sum + value, 0) / (depths.length || 1);

  // Step 3: Compute ancestor uniqueness via lineage helper.
  const lineageContext: LineageContext = {
    population: population || [],
    _getRNG:
      typeof telemetryContext._getRNG === 'function'
        ? telemetryContext._getRNG
        : () => Math.random,
  };
  const ancestorUniqueness = computeAncestorUniqueness.call(lineageContext);

  // Step 4: Attach lineage stats to the entry.
  entry.lineage = {
    parents: Array.isArray(bestGenome._parents)
      ? bestGenome._parents.slice()
      : [],
    depthBest: bestGenome._depth ?? 0,
    meanDepth: +(telemetryContext._lastMeanDepth ?? 0).toFixed(2),
    inbreeding: telemetryContext._prevInbreedingCount ?? 0,
    ancestorUniq: ancestorUniqueness,
  };
}

/**
 * Apply lineage stats for mono-objective mode using sampled ancestors.
 *
 * @param telemetryContext - Neat-like context with lineage settings.
 * @param population - Population snapshot.
 * @param entry - Telemetry entry to update.
 */
export function applyLineageStatsMonoObjective(
  telemetryContext: {
    _lineageEnabled?: boolean;
    _getRNG?: () => () => number;
    _lastMeanDepth?: number;
    _prevInbreedingCount?: number;
  },
  populationSnapshot: GenomeDetailed[],
  entry: TelemetryEntryRecord,
): void {
  // Step 1: Resolve population snapshot and eligibility.
  if (!isLineageEligible(telemetryContext, populationSnapshot)) return;

  // Step 2: Compute depth and ancestor uniqueness metrics.
  const bestGenome = populationSnapshot[0] as GenomeDetailed;
  const depths = collectDepths(populationSnapshot);
  const meanDepth = computeMeanDepth(depths);
  telemetryContext._lastMeanDepth = meanDepth;
  const ancestorUniqueness = computeAncestorUniquenessSampled(
    telemetryContext,
    populationSnapshot,
  );

  // Step 3: Attach lineage stats to the entry.
  entry.lineage = buildLineageEntry(
    telemetryContext,
    bestGenome,
    meanDepth,
    ancestorUniqueness,
  );
}

/**
 * Check whether lineage metrics should be computed.
 *
 * @param context - Neat-like context with lineage flag.
 * @param populationSnapshot - Population snapshot to validate.
 * @returns True when lineage stats should be computed.
 */
export function isLineageEligible(
  context: { _lineageEnabled?: boolean },
  populationSnapshot: GenomeDetailed[],
): boolean {
  // Step 1: Require lineage enabled and non-empty population.
  return Boolean(context._lineageEnabled) && populationSnapshot.length > 0;
}

/**
 * Collect depth values for the current population.
 *
 * @param populationSnapshot - Population snapshot.
 * @returns Array of depth values (defaults to 0).
 */
export function collectDepths(populationSnapshot: GenomeDetailed[]): number[] {
  // Step 1: Map genomes to depth numbers.
  return populationSnapshot.map((genome) => genome._depth ?? 0);
}

/**
 * Compute the mean depth from a depth list.
 *
 * @param depthValues - Depth values to average.
 * @returns Mean depth value.
 */
export function computeMeanDepth(depthValues: number[]): number {
  // Step 1: Fold depths into a mean value.
  return (
    depthValues.reduce((sum, value) => sum + value, 0) /
    (depthValues.length || 1)
  );
}

/**
 * Compute ancestor uniqueness using sampled Jaccard distance.
 *
 * @param context - Neat-like context with RNG helpers.
 * @param populationSnapshot - Population snapshot.
 * @returns Rounded ancestor uniqueness score.
 */
export function computeAncestorUniquenessSampled(
  context: { _getRNG?: () => () => number },
  populationSnapshot: GenomeDetailed[],
): number {
  // Step 1: Resolve sampling bounds.
  const populationSize = populationSnapshot.length;
  const maxPairsToSample = Math.min(
    30,
    (populationSize * (populationSize - 1)) / 2,
  );

  // Step 2: Accumulate Jaccard distances across sampled pairs.
  let sampledPairs = 0;
  let jaccardSum = 0;
  for (let sampleIndex = 0; sampleIndex < maxPairsToSample; sampleIndex++) {
    if (populationSize < 2) break;
    const { firstIndex, secondIndex } = pickDistinctPairIndices(
      context,
      populationSize,
    );
    const jaccardDistance = computePairJaccardDistance(
      context,
      populationSnapshot,
      firstIndex,
      secondIndex,
    );
    if (jaccardDistance === undefined) continue;
    jaccardSum += jaccardDistance;
    sampledPairs++;
  }

  // Step 3: Normalize and round the uniqueness score.
  return sampledPairs ? +(jaccardSum / sampledPairs).toFixed(3) : 0;
}

/**
 * Pick two distinct indices using the context RNG.
 *
 * @param context - Neat-like context with RNG factory.
 * @param populationSize - Population size for index bounds.
 * @returns Pair of distinct indices.
 */
export function pickDistinctPairIndices(
  context: { _getRNG?: () => () => number },
  populationSize: number,
): { firstIndex: number; secondIndex: number } {
  // Step 1: Resolve RNG function.
  const rngFunction =
    typeof context._getRNG === 'function' ? context._getRNG() : Math.random;

  // Step 2: Pick first index and a different second index.
  const firstIndex = Math.floor(rngFunction() * populationSize);
  let secondIndex = Math.floor(rngFunction() * populationSize);
  if (secondIndex === firstIndex)
    secondIndex = (secondIndex + 1) % populationSize;

  // Step 3: Return the pair.
  return { firstIndex, secondIndex };
}

/**
 * Compute Jaccard distance between ancestor sets for a pair.
 *
 * @param context - Neat-like context for lineage helpers.
 * @param populationSnapshot - Population snapshot.
 * @param firstIndex - First genome index.
 * @param secondIndex - Second genome index.
 * @returns Jaccard distance or undefined when both sets are empty.
 */
export function computePairJaccardDistance(
  context: { _getRNG?: () => () => number },
  populationSnapshot: GenomeDetailed[],
  firstIndex: number,
  secondIndex: number,
): number | undefined {
  // Step 1: Build lineage context for ancestor discovery.
  const lineageContext = buildLineageContext(context, populationSnapshot);

  // Step 2: Resolve ancestor sets.
  const ancestorsA = buildAnc.call(
    lineageContext,
    populationSnapshot[firstIndex],
  );
  const ancestorsB = buildAnc.call(
    lineageContext,
    populationSnapshot[secondIndex],
  );
  if (ancestorsA.size === 0 && ancestorsB.size === 0) return;

  // Step 3: Compute intersection and union counts.
  const intersectionCount = countAncestorIntersection(ancestorsA, ancestorsB);
  const unionCount = ancestorsA.size + ancestorsB.size - intersectionCount || 1;

  // Step 4: Return the Jaccard distance.
  return 1 - intersectionCount / unionCount;
}

/**
 * Build a lineage helper context for ancestor operations.
 *
 * @param context - Neat-like context with RNG helpers.
 * @param populationSnapshot - Population snapshot.
 * @returns Lineage helper context.
 */
export function buildLineageContext(
  context: { _getRNG?: () => () => number },
  populationSnapshot: GenomeDetailed[],
): LineageContext {
  // Step 1: Provide population and RNG factory to lineage helpers.
  return {
    population: populationSnapshot,
    _getRNG:
      typeof context._getRNG === 'function'
        ? context._getRNG
        : () => Math.random,
  } as LineageContext;
}

/**
 * Count the size of an ancestor intersection.
 *
 * @param ancestorsA - First ancestor set.
 * @param ancestorsB - Second ancestor set.
 * @returns Intersection count.
 */
export function countAncestorIntersection(
  ancestorsA: Set<number>,
  ancestorsB: Set<number>,
): number {
  // Step 1: Count shared ancestor ids.
  let intersectionCount = 0;
  for (const ancestorId of ancestorsA) {
    if (ancestorsB.has(ancestorId)) intersectionCount++;
  }
  return intersectionCount;
}

/**
 * Build the lineage entry payload.
 *
 * @param context - Neat-like context with lineage info.
 * @param bestGenomeSnapshot - Best genome snapshot.
 * @param meanDepthValue - Mean lineage depth.
 * @param ancestorUniquenessScore - Ancestor uniqueness score.
 * @returns Lineage entry payload.
 */
export function buildLineageEntry(
  context: { _prevInbreedingCount?: number },
  bestGenomeSnapshot: GenomeDetailed,
  meanDepthValue: number,
  ancestorUniquenessScore: number,
): {
  parents: number[];
  depthBest: number;
  meanDepth: number;
  inbreeding: number;
  ancestorUniq: number;
} {
  // Step 1: Assemble lineage payload with rounded values.
  return {
    parents: Array.isArray(bestGenomeSnapshot._parents)
      ? bestGenomeSnapshot._parents.slice()
      : [],
    depthBest: bestGenomeSnapshot._depth ?? 0,
    meanDepth: +meanDepthValue.toFixed(2),
    inbreeding: context._prevInbreedingCount ?? 0,
    ancestorUniq: ancestorUniquenessScore,
  };
}

/**
 * Attach hypervolume scalar when requested.
 *
 * @param telemetryOptions - Options controlling telemetry fields.
 * @param hyperVolumeProxy - Hypervolume proxy value.
 * @param entry - Telemetry entry to update.
 */
export function applyHypervolumeTelemetry(
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  hyperVolumeProxy: number,
  entry: TelemetryEntryRecord,
): void {
  // Step 1: Attach rounded hypervolume proxy when enabled.
  if (
    telemetryOptions.telemetry?.hypervolume &&
    telemetryOptions.multiObjective?.enabled
  )
    entry.hv = +hyperVolumeProxy.toFixed(4);
}

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

/**
 * Attach performance stats when configured.
 *
 * @param telemetryContext - Neat-like context with performance data.
 * @param telemetryOptions - Options controlling performance telemetry.
 * @param entry - Telemetry entry to update.
 */
export function applyPerformanceStats(
  telemetryContext: {
    _lastEvalDuration?: number;
    _lastEvolveDuration?: number;
  },
  telemetryOptions: NeatOptions & TelemetryDiversityOptions,
  entry: TelemetryEntryRecord,
): void {
  // Step 1: Skip if performance telemetry is disabled.
  if (!telemetryOptions.telemetry?.performance) return;

  // Step 2: Attach duration metrics.
  entry.perf = {
    evalMs: telemetryContext._lastEvalDuration,
    evolveMs: telemetryContext._lastEvolveDuration,
  };
}
