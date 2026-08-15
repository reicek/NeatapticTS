import type { GenomeDetailed } from '../../shared/neat.shared.types';
import type { NeatLineageContext as LineageContext } from '../../lineage/lineage';
import { buildAnc, computeAncestorUniqueness } from '../../lineage/lineage';
import type {
  TelemetryEntryRecord,
  TelemetryGenome,
} from '../types/telemetry.types';

/**
 * Create a default RNG factory backed by Math.random.
 *
 * @returns RNG factory returning Math.random.
 */
function createMathRandomFactory(): () => number {
  // Step 1: Return the default Math.random RNG.
  return Math.random;
}

/**
 * Resolve the RNG factory used by lineage helpers.
 *
 * @param providedRngFactory - Optional caller-provided RNG factory.
 * @returns Caller RNG factory or the Math.random fallback.
 */
function resolveLineageRngFactory(
  providedRngFactory?: () => () => number,
): () => () => number {
  // Step 1: Preserve caller-provided RNG factories when available.
  return typeof providedRngFactory === 'function'
    ? providedRngFactory
    : createMathRandomFactory;
}

/**
 * Compute lineage depth and pairwise depth-distance statistics so telemetry can expose ancestry spread and genealogical divergence for the current generation.
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
 * Apply lineage stats for multi-objective mode using ancestor uniqueness so entries capture genealogy health alongside Pareto progress signals across generations.
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
  const meanDepth =
    depths.reduce((sum, value) => sum + value, 0) / depths.length;
  telemetryContext._lastMeanDepth = meanDepth;

  // Step 3: Compute ancestor uniqueness via lineage helper.
  const lineageContext: LineageContext = {
    population,
    _getRNG: resolveLineageRngFactory(telemetryContext._getRNG),
  };
  const ancestorUniqueness = computeAncestorUniqueness.call(lineageContext);

  // Step 4: Attach lineage stats to the entry.
  entry.lineage = {
    parents: Array.isArray(bestGenome._parents)
      ? bestGenome._parents.slice()
      : [],
    depthBest: bestGenome._depth ?? 0,
    meanDepth: +meanDepth.toFixed(2),
    inbreeding: telemetryContext._prevInbreedingCount ?? 0,
    ancestorUniq: ancestorUniqueness,
  };
}

/**
 * Apply lineage stats for mono-objective mode using sampled ancestors so single-score runs still report ancestry diversity pressure transparently across long experiments.
 *
 * @param telemetryContext - Neat-like context with lineage settings.
 * @param populationSnapshot - Snapshot of the current population used for lineage stats.
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
 * Check whether lineage metrics should be computed for this snapshot so helper calls can short-circuit before any ancestry sampling work.
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
 * Collect depth values for the current population so lineage summaries can be derived consistently from one deterministic per-genome mapping pass.
 *
 * @param populationSnapshot - Population snapshot.
 * @returns Array of depth values (defaults to 0).
 */
export function collectDepths(populationSnapshot: GenomeDetailed[]): number[] {
  // Step 1: Map genomes to depth numbers.
  return populationSnapshot.map((genome) => genome._depth ?? 0);
}

/**
 * Compute the mean depth from a depth list so telemetry can expose ancestry maturity with one stable, noise-reduced scalar for trend charts.
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
 * Compute ancestor uniqueness using sampled Jaccard distance so recorder output reflects how distinct elite ancestry remains across the population over time.
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
 * Pick two distinct indices using the context RNG so pairwise lineage sampling remains deterministic under seeded controller configurations and replay workflows.
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
 * Compute Jaccard distance between ancestor sets for one sampled pair so uniqueness estimates remain interpretable and mathematically grounded during diagnostics.
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
  const unionCount = ancestorsA.size + ancestorsB.size - intersectionCount;

  // Step 4: Return the Jaccard distance.
  return 1 - intersectionCount / unionCount;
}

/**
 * Build a lineage helper context for ancestor operations so ancestry utilities share population and RNG state through one explicit boundary object.
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
    _getRNG: resolveLineageRngFactory(context._getRNG),
  } as LineageContext;
}

/**
 * Count the size of an ancestor intersection so Jaccard distance computation can reuse a clear and testable set-overlap primitive helper.
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
 * Build the lineage entry payload so recorder output stores parent identifiers, depth metrics, and inbreeding context in one normalized shape.
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
