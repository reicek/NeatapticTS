import { NGE_EVOLUTION_DEFAULT_COMPATIBILITY_DISTANCE_WEIGHTS } from './neat.nge-evolution.constants';
import type {
  NgeEvolutionCompatibilityComparisonInput,
  NgeEvolutionCompatibilityDistanceContext,
  NgeEvolutionCompatibilityDistanceResult,
  NgeEvolutionCompatibilityDistanceTerm,
  NgeEvolutionCompatibilityDistanceTermName,
  NgeEvolutionCompatibilityDistanceWeights,
  NgeEvolutionCompatibilityGenomeInput,
  NgeEvolutionCompatibilityWiringCostWeights,
} from './neat.nge-evolution.types';

const MEMORY_CAPACITY_BIN_THRESHOLDS = [4, 8, 16] as const;

/**
 * Compute the NGE composite compatibility distance for one NGE genome pair.
 *
 * The classic NEAT topology distance stays injected rather than recomputed here.
 * The additional NGE-only terms derive from DNA archetype composition, memory
 * tier shape, and an owner-local lifecycle sidecar that carries cadence and
 * wiring-preference knobs until those fields land in the canonical DNA schema.
 *
 * @param comparison - Target pairwise comparison to score.
 * @param context - Optional normalization slice and alpha-weight overrides.
 * @returns Composite NGE compatibility-distance result for the target pair.
 */
export function computeNgeEvolutionCompatibilityDistance(
  comparison: NgeEvolutionCompatibilityComparisonInput,
  context: NgeEvolutionCompatibilityDistanceContext = {},
): NgeEvolutionCompatibilityDistanceResult {
  // Step 1: Resolve the normalization slice and alpha weights.
  const populationSlice = resolvePopulationSlice(
    comparison,
    context.populationSlice,
  );
  const resolvedWeights = resolveCompatibilityDistanceWeights(context.weights);

  // Step 2: Compute raw term distances for the full slice before normalization.
  const rawDistanceRecords = populationSlice.map((populationComparison) =>
    buildRawDistanceRecord(populationComparison),
  );
  const targetRecord = rawDistanceRecords[0];

  // Step 3: Preserve classic NEAT behavior when NGE is inactive for the pair.
  if (!targetRecord.ngeEnabled) {
    return buildClassicCompatibilityResult(targetRecord.topologyDistance);
  }

  // Step 4: Normalize each term independently across the current slice.
  const normalizationBounds = buildNormalizationBounds(rawDistanceRecords);
  const terms = {
    topology: buildCompatibilityDistanceTerm(
      'topology',
      targetRecord.topologyDistance,
      normalizeTermDistance(
        targetRecord.topologyDistance,
        normalizationBounds.topology,
      ),
      resolvedWeights.topology,
    ),
    computation: buildCompatibilityDistanceTerm(
      'computation',
      targetRecord.computationDistance,
      normalizeTermDistance(
        targetRecord.computationDistance,
        normalizationBounds.computation,
      ),
      resolvedWeights.computation,
    ),
    memory: buildCompatibilityDistanceTerm(
      'memory',
      targetRecord.memoryDistance,
      normalizeTermDistance(
        targetRecord.memoryDistance,
        normalizationBounds.memory,
      ),
      resolvedWeights.memory,
    ),
    lifecycle: buildCompatibilityDistanceTerm(
      'lifecycle',
      targetRecord.lifecycleDistance,
      normalizeTermDistance(
        targetRecord.lifecycleDistance,
        normalizationBounds.lifecycle,
      ),
      resolvedWeights.lifecycle,
    ),
  };

  // Step 5: Fold the normalized weighted terms into the bounded composite distance.
  const distance = clampUnitInterval(
    Object.values(terms).reduce(
      (currentDistance, term) => currentDistance + term.weightedDistance,
      0,
    ),
  );

  return {
    distance,
    ngeEnabled: true,
    terms,
  };
}

interface NgeEvolutionRawDistanceRecord {
  ngeEnabled: boolean;
  topologyDistance: number;
  computationDistance: number;
  memoryDistance: number;
  lifecycleDistance: number;
}

interface NgeEvolutionNormalizationBounds {
  minimum: number;
  maximum: number;
}

interface NgeEvolutionMemoryProfile {
  recurrentPresence: boolean;
  recurrentCapacityBin: number;
  episodicPresence: boolean;
  episodicCapacityBin: number;
}

/**
 * Resolve the slice used for min-max normalization while keeping the target pair first.
 *
 * @param comparison - Target comparison that must remain the first result row.
 * @param populationSlice - Optional additional comparisons from the active population slice.
 * @returns Stable slice with the target comparison in the first slot.
 */
function resolvePopulationSlice(
  comparison: NgeEvolutionCompatibilityComparisonInput,
  populationSlice: readonly NgeEvolutionCompatibilityComparisonInput[] = [],
): readonly NgeEvolutionCompatibilityComparisonInput[] {
  return [
    comparison,
    ...populationSlice.filter(
      (populationComparison) => populationComparison !== comparison,
    ),
  ];
}

/**
 * Normalize a weight bag so the enabled composite sum stays bounded by one.
 *
 * @param weights - Optional caller overrides merged onto the NGE defaults.
 * @returns Normalized alpha weights whose sum is `1` unless every entry is `0`.
 */
function resolveCompatibilityDistanceWeights(
  weights?: Partial<NgeEvolutionCompatibilityDistanceWeights>,
): NgeEvolutionCompatibilityDistanceWeights {
  const mergedWeights = {
    ...NGE_EVOLUTION_DEFAULT_COMPATIBILITY_DISTANCE_WEIGHTS,
    ...(weights ?? {}),
  };
  const totalWeight =
    mergedWeights.topology +
    mergedWeights.computation +
    mergedWeights.memory +
    mergedWeights.lifecycle;
  const safeTotalWeight = totalWeight || 1;

  return {
    topology: mergedWeights.topology / safeTotalWeight,
    computation: mergedWeights.computation / safeTotalWeight,
    memory: mergedWeights.memory / safeTotalWeight,
    lifecycle: mergedWeights.lifecycle / safeTotalWeight,
  };
}

/**
 * Build the raw term record for one pairwise comparison.
 *
 * @param comparison - Pairwise comparison to analyze.
 * @returns Raw term distances before slice-level normalization.
 */
function buildRawDistanceRecord(
  comparison: NgeEvolutionCompatibilityComparisonInput,
): NgeEvolutionRawDistanceRecord {
  const topologyDistance = Math.max(0, comparison.topologyDistance);
  const ngeEnabled =
    comparison.ngeEnabled &&
    comparison.leftGenome.dna !== null &&
    comparison.rightGenome.dna !== null;

  if (!ngeEnabled) {
    return {
      ngeEnabled: false,
      topologyDistance,
      computationDistance: 0,
      memoryDistance: 0,
      lifecycleDistance: 0,
    };
  }

  const leftGenome = comparison.leftGenome;
  const rightGenome = comparison.rightGenome;

  return {
    ngeEnabled: true,
    topologyDistance,
    computationDistance: computeRawComputationDistance(leftGenome, rightGenome),
    memoryDistance: computeRawMemoryDistance(leftGenome, rightGenome),
    lifecycleDistance: computeRawLifecycleDistance(leftGenome, rightGenome),
  };
}

/**
 * Build one weighted compatibility-distance term.
 *
 * @param name - Stable term name emitted in the result shelf.
 * @param rawDistance - Raw pre-normalization distance.
 * @param normalizedDistance - Slice-normalized distance.
 * @param weight - Alpha weight applied to the normalized distance.
 * @returns Fully expanded weighted term result.
 */
function buildCompatibilityDistanceTerm(
  name: NgeEvolutionCompatibilityDistanceTermName,
  rawDistance: number,
  normalizedDistance: number,
  weight: number,
): NgeEvolutionCompatibilityDistanceTerm {
  return {
    name,
    rawDistance,
    normalizedDistance,
    weight,
    weightedDistance: normalizedDistance * weight,
  };
}

/**
 * Build the classic passthrough result used when NGE is disabled.
 *
 * @param topologyDistance - Raw classic NEAT compatibility distance.
 * @returns Classic passthrough result with the NGE-only terms collapsed to zero.
 */
function buildClassicCompatibilityResult(
  topologyDistance: number,
): NgeEvolutionCompatibilityDistanceResult {
  return {
    distance: topologyDistance,
    ngeEnabled: false,
    terms: {
      topology: {
        name: 'topology',
        rawDistance: topologyDistance,
        normalizedDistance: topologyDistance,
        weight: 1,
        weightedDistance: topologyDistance,
      },
      computation: {
        name: 'computation',
        rawDistance: 0,
        normalizedDistance: 0,
        weight: 0,
        weightedDistance: 0,
      },
      memory: {
        name: 'memory',
        rawDistance: 0,
        normalizedDistance: 0,
        weight: 0,
        weightedDistance: 0,
      },
      lifecycle: {
        name: 'lifecycle',
        rawDistance: 0,
        normalizedDistance: 0,
        weight: 0,
        weightedDistance: 0,
      },
    },
  };
}

/**
 * Collect min-max bounds for every normalized term across the active slice.
 *
 * @param rawDistanceRecords - Raw slice records prepared ahead of normalization.
 * @returns Per-term min-max bounds.
 */
function buildNormalizationBounds(
  rawDistanceRecords: readonly NgeEvolutionRawDistanceRecord[],
): Record<
  NgeEvolutionCompatibilityDistanceTermName,
  NgeEvolutionNormalizationBounds
> {
  return {
    topology: buildTermBounds(
      rawDistanceRecords.map(({ topologyDistance }) => topologyDistance),
    ),
    computation: buildTermBounds(
      rawDistanceRecords.map(({ computationDistance }) => computationDistance),
    ),
    memory: buildTermBounds(
      rawDistanceRecords.map(({ memoryDistance }) => memoryDistance),
    ),
    lifecycle: buildTermBounds(
      rawDistanceRecords.map(({ lifecycleDistance }) => lifecycleDistance),
    ),
  };
}

/**
 * Build the min-max bounds for one normalized term column.
 *
 * @param values - Raw numeric term values for the current slice.
 * @returns Minimum and maximum values for the column.
 */
function buildTermBounds(
  values: readonly number[],
): NgeEvolutionNormalizationBounds {
  return {
    minimum: Math.min(...values),
    maximum: Math.max(...values),
  };
}

/**
 * Normalize one raw term distance into the unit interval.
 *
 * Degenerate non-zero slices resolve to `1` so a single differing pair still
 * contributes fully when no wider slice context is available.
 *
 * @param rawDistance - Raw value to normalize.
 * @param bounds - Slice min-max bounds for the term.
 * @returns Normalized unit-interval distance.
 */
function normalizeTermDistance(
  rawDistance: number,
  bounds: NgeEvolutionNormalizationBounds,
): number {
  const valueRange = bounds.maximum - bounds.minimum;

  if (valueRange === 0) {
    return bounds.maximum === 0 ? 0 : 1;
  }

  return (rawDistance - bounds.minimum) / valueRange;
}

/**
 * Compute the raw computation-motif distance for one pair.
 *
 * @param leftGenome - Left genome-side comparison input.
 * @param rightGenome - Right genome-side comparison input.
 * @returns Raw computation-motif distance before normalization.
 */
function computeRawComputationDistance(
  leftGenome: NgeEvolutionCompatibilityGenomeInput,
  rightGenome: NgeEvolutionCompatibilityGenomeInput,
): number {
  const leftCounts = collectComputationCounts(leftGenome);
  const rightCounts = collectComputationCounts(rightGenome);
  const computationTypes = [
    ...new Set([...Object.keys(leftCounts), ...Object.keys(rightCounts)]),
  ].toSorted();
  const leftTotalCount = Object.values(leftCounts).reduce(
    (currentTotal, currentCount) => currentTotal + currentCount,
    0,
  );
  const rightTotalCount = Object.values(rightCounts).reduce(
    (currentTotal, currentCount) => currentTotal + currentCount,
    0,
  );
  const perTypeDifferences = computationTypes.map((computationType) =>
    computeDifferenceRatio(
      leftCounts[computationType] ?? 0,
      rightCounts[computationType] ?? 0,
    ),
  );

  return averageValues([
    computeDifferenceRatio(leftTotalCount, rightTotalCount),
    ...perTypeDifferences,
  ]);
}

/**
 * Collect the module-archetype counts keyed by computation type.
 *
 * @param genome - Genome-side input carrying the canonical DNA envelope.
 * @returns Count map keyed by computation type.
 */
function collectComputationCounts(
  genome: NgeEvolutionCompatibilityGenomeInput,
): Record<string, number> {
  return genome.dna!.moduleArchetypes.reduce<Record<string, number>>(
    (countByComputationType, moduleArchetype) => ({
      ...countByComputationType,
      [moduleArchetype.computationType]:
        (countByComputationType[moduleArchetype.computationType] ?? 0) + 1,
    }),
    {},
  );
}

/**
 * Compute the raw memory-tier distance for one pair.
 *
 * @param leftGenome - Left genome-side comparison input.
 * @param rightGenome - Right genome-side comparison input.
 * @returns Raw memory-tier distance before normalization.
 */
function computeRawMemoryDistance(
  leftGenome: NgeEvolutionCompatibilityGenomeInput,
  rightGenome: NgeEvolutionCompatibilityGenomeInput,
): number {
  const leftProfile = collectMemoryProfile(leftGenome);
  const rightProfile = collectMemoryProfile(rightGenome);

  return averageValues([
    Number(leftProfile.recurrentPresence !== rightProfile.recurrentPresence),
    Number(leftProfile.episodicPresence !== rightProfile.episodicPresence),
    computeDifferenceRatio(
      leftProfile.recurrentCapacityBin,
      rightProfile.recurrentCapacityBin,
    ),
    computeDifferenceRatio(
      leftProfile.episodicCapacityBin,
      rightProfile.episodicCapacityBin,
    ),
  ]);
}

/**
 * Collect the memory-tier profile used by the memory-distance term.
 *
 * @param genome - Genome-side comparison input carrying the canonical DNA envelope.
 * @returns Presence and capacity-bin summary for recurrent and episodic tiers.
 */
function collectMemoryProfile(
  genome: NgeEvolutionCompatibilityGenomeInput,
): NgeEvolutionMemoryProfile {
  const moduleArchetypes = genome.dna!.moduleArchetypes;
  const recurrentModules = moduleArchetypes.filter(
    ({ computationType }) => computationType === 'GatedRecurrentCell',
  );
  const episodicModules = moduleArchetypes.filter(
    ({ computationType }) => computationType === 'EpisodicSlot',
  );
  const recurrentCapacity = recurrentModules.reduce(
    (currentCapacity, moduleArchetype) =>
      currentCapacity +
      resolveNumericParameter(moduleArchetype.parameterSchema, 'hiddenDim'),
    0,
  );
  const episodicCapacity = episodicModules.reduce(
    (currentCapacity, moduleArchetype) =>
      currentCapacity +
      resolveNumericParameter(moduleArchetype.parameterSchema, 'slotCount'),
    0,
  );

  return {
    recurrentPresence: recurrentModules.length > 0,
    recurrentCapacityBin: resolveMemoryCapacityBin(recurrentCapacity),
    episodicPresence: episodicModules.length > 0,
    episodicCapacityBin: resolveMemoryCapacityBin(episodicCapacity),
  };
}

/**
 * Resolve the bucketed memory-capacity bin for one total capacity value.
 *
 * @param capacity - Aggregate hidden-dimension or slot-count value.
 * @returns Stable ordinal capacity bin.
 */
function resolveMemoryCapacityBin(capacity: number): number {
  const matchingThresholdIndex = MEMORY_CAPACITY_BIN_THRESHOLDS.findIndex(
    (threshold) => capacity > 0 && capacity <= threshold,
  );

  if (matchingThresholdIndex === -1) {
    return capacity === 0 ? 0 : MEMORY_CAPACITY_BIN_THRESHOLDS.length + 1;
  }

  return matchingThresholdIndex + 1;
}

/**
 * Resolve one numeric module-archetype parameter or return `0` when absent.
 *
 * @param parameterSchema - Optional archetype parameter schema.
 * @param parameterName - Parameter key to resolve.
 * @returns Numeric parameter value or `0` when missing.
 */
function resolveNumericParameter(
  parameterSchema: Record<string, unknown> | undefined,
  parameterName: string,
): number {
  const parameterValue = parameterSchema?.[parameterName];

  return typeof parameterValue === 'number' ? parameterValue : 0;
}

/**
 * Compute the raw lifecycle-governance distance for one pair.
 *
 * @param leftGenome - Left genome-side comparison input.
 * @param rightGenome - Right genome-side comparison input.
 * @returns Raw lifecycle-governance distance before normalization.
 */
function computeRawLifecycleDistance(
  leftGenome: NgeEvolutionCompatibilityGenomeInput,
  rightGenome: NgeEvolutionCompatibilityGenomeInput,
): number {
  const reproductionModeDifference = Number(
    leftGenome.dna?.reproductionPolicy.mode !==
      rightGenome.dna?.reproductionPolicy.mode,
  );
  const cadenceDifference = computeDifferenceRatio(
    leftGenome.assimilationCadence ?? 0,
    rightGenome.assimilationCadence ?? 0,
  );
  const wiringCostDifference = computeWiringPreferenceDifference(
    leftGenome.wiringCostWeights,
    rightGenome.wiringCostWeights,
  );

  return averageValues([
    reproductionModeDifference,
    cadenceDifference,
    wiringCostDifference,
  ]);
}

/**
 * Compute the normalized difference across one wiring-cost preference bag.
 *
 * @param leftWeights - Left-side wiring-cost weights.
 * @param rightWeights - Right-side wiring-cost weights.
 * @returns Mean normalized difference across the shared key union.
 */
function computeWiringPreferenceDifference(
  leftWeights: NgeEvolutionCompatibilityWiringCostWeights | undefined,
  rightWeights: NgeEvolutionCompatibilityWiringCostWeights | undefined,
): number {
  const preferenceKeys = [
    ...new Set([
      ...Object.keys(leftWeights ?? {}),
      ...Object.keys(rightWeights ?? {}),
    ]),
  ].toSorted();

  return averageValues(
    preferenceKeys.map((preferenceKey) =>
      computeDifferenceRatio(
        leftWeights?.[
          preferenceKey as keyof NgeEvolutionCompatibilityWiringCostWeights
        ] ?? 0,
        rightWeights?.[
          preferenceKey as keyof NgeEvolutionCompatibilityWiringCostWeights
        ] ?? 0,
      ),
    ),
  );
}

/**
 * Compute a symmetric normalized difference ratio for two scalar values.
 *
 * @param leftValue - Left-side scalar value.
 * @param rightValue - Right-side scalar value.
 * @returns Absolute difference divided by the larger absolute magnitude.
 */
function computeDifferenceRatio(leftValue: number, rightValue: number): number {
  const comparisonScale = Math.max(Math.abs(leftValue), Math.abs(rightValue));

  if (comparisonScale === 0) {
    return 0;
  }

  return Math.abs(leftValue - rightValue) / comparisonScale;
}

/**
 * Compute the arithmetic mean for one numeric vector.
 *
 * @param values - Numeric vector to average.
 * @returns Mean of the vector or `0` when the vector is empty.
 */
function averageValues(values: readonly number[]): number {
  if (values.length === 0) {
    return 0;
  }

  const totalValue = values.reduce(
    (currentTotal, currentValue) => currentTotal + currentValue,
    0,
  );

  return totalValue / values.length;
}

/**
 * Clamp one enabled composite distance into the unit interval.
 *
 * @param value - Composite weighted distance before clamping.
 * @returns Unit-interval bounded composite distance.
 */
function clampUnitInterval(value: number): number {
  return Math.min(1, Math.max(0, value));
}
