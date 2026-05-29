# neat/nge-evolution

Error thrown when one requested reproduction mode is unavailable for the current evolution context.

## neat/nge-evolution/neat.nge-evolution.types.ts

### NgeEvolutionCompatibilityComparisonInput

One pairwise comparison input evaluated by the Phase E composite compatibility calculator.

### NgeEvolutionCompatibilityDistanceContext

Context bag controlling one Phase E compatibility-distance computation.

### NgeEvolutionCompatibilityDistanceResult

Composite compatibility-distance result returned by the Phase E speciation helper.

### NgeEvolutionCompatibilityDistanceTerm

One weighted compatibility-distance term captured during Phase E speciation scoring.

### NgeEvolutionCompatibilityDistanceTermName

Fixed term names used by the Phase E composite compatibility-distance calculator.

### NgeEvolutionCompatibilityDistanceTerms

Fully expanded term shelf returned by the Phase E compatibility-distance calculator.

### NgeEvolutionCompatibilityDistanceWeights

Alpha weights applied to the four Phase E compatibility-distance terms.

### NgeEvolutionCompatibilityGenomeInput

One genome-side input consumed by the Phase E composite compatibility calculator.

The canonical DNA envelope does not yet own lifecycle cadence or wiring-preference
knobs, so the calculator accepts those traits as an owner-local sidecar.

### NgeEvolutionCompatibilityWiringCostWeights

Wiring-cost preference knobs compared by the Phase E lifecycle-distance term.

### NgeEvolutionContributionKind

High-level contribution kinds used to describe parent input at the reproduction boundary.

### NgeEvolutionEpigeneticPriorInput

Input contract consumed by the optional Phase E epigenetic prior operator.

### NgeEvolutionEpigeneticPriorResult

Output contract returned by the optional Phase E epigenetic prior operator.

### NgeEvolutionEpigeneticReference

Two-parent weak reference captured for one birth-time epigenetic prior update.

### NgeEvolutionParentContribution

One parent contribution reported by a Phase E reproduction operator.

### NgeEvolutionParentRole

Parent-role labels used when Phase E operators report how one offspring was assembled.

### NgeEvolutionPolyandricAssignedRegion

Region-assignment record for one drone contribution in polyandric reproduction.

### NgeEvolutionPolyandricRegionAssignmentResult

Polyandric region-assignment result reported before drone patches are applied.

### NgeEvolutionReproductionOutcome

High-level offspring outcome labels surfaced by the Phase E reproduction operators.

### NgeEvolutionReproductionResult

Shared reproduction result returned by all three Phase E reproduction modes.

## neat/nge-evolution/neat.nge-evolution.ts

### applyNgeEvolutionEpigeneticPrior

```ts
applyNgeEvolutionEpigeneticPrior(
  input: NgeEvolutionEpigeneticPriorInput<ParameterVector>,
): NgeEvolutionEpigeneticPriorResult<ParameterVector>
```

Public birth-time epigenetic prior entrypoint exposed from one stable owner-local facade.

### computeNgeEvolutionCompatibilityDistance

```ts
computeNgeEvolutionCompatibilityDistance(
  comparison: NgeEvolutionCompatibilityComparisonInput,
  context: NgeEvolutionCompatibilityDistanceContext,
): NgeEvolutionCompatibilityDistanceResult
```

Public Phase E compatibility-distance entrypoint exposed from one stable owner-local facade.

### NGE_EVOLUTION_DEFAULT_ALPHA_COMPUTATION

Default alpha weight for the computation-motif distance term.

### NGE_EVOLUTION_DEFAULT_ALPHA_LIFECYCLE

Default alpha weight for the lifecycle-policy distance term.

### NGE_EVOLUTION_DEFAULT_ALPHA_MEMORY

Default alpha weight for the memory-tier distance term.

### NGE_EVOLUTION_DEFAULT_ALPHA_TOPOLOGY

Default alpha weight for the classic topology-distance term.

### NGE_EVOLUTION_DEFAULT_COMPATIBILITY_DISTANCE_WEIGHTS

Default per-term alpha bag for callers that want the Phase E compatibility defaults.

### NGE_EVOLUTION_DEFAULT_EPIGENETIC_DECAY

Default weak-reference decay used by the optional epigenetic prior operator.

### NGE_EVOLUTION_DEFAULT_POLYANDRIC_DRONE_CONTRIBUTION_FRACTION

Default fraction of DNA regions that polyandric drone donors may patch.

### NGE_EVOLUTION_DEFAULT_POLYANDRIC_QUEEN_BIAS

Default queen-bias multiplier where `1.0` means queen data wins all conflicts.

### ngeEvolution

Default bundle for the nge-evolution owner boundary.

Import this object when a caller wants the full runtime shelf for Phase E
from one stable owner-local path instead of stitching together leaf modules.

### NgeEvolution_BudgetError

Public error class thrown when one operator exceeds the configured budget.

### NgeEvolution_ModeError

Public error class thrown when one requested reproduction mode is unavailable.

### NgeEvolution_RegionError

Public error class thrown when one region-assignment request is invalid.

### NgeEvolutionCompatibilityComparisonInput

One pairwise comparison input evaluated by the Phase E composite compatibility calculator.

### NgeEvolutionCompatibilityDistanceContext

Context bag controlling one Phase E compatibility-distance computation.

### NgeEvolutionCompatibilityDistanceResult

Composite compatibility-distance result returned by the Phase E speciation helper.

### NgeEvolutionCompatibilityDistanceTerm

One weighted compatibility-distance term captured during Phase E speciation scoring.

### NgeEvolutionCompatibilityDistanceTermName

Fixed term names used by the Phase E composite compatibility-distance calculator.

### NgeEvolutionCompatibilityDistanceTerms

Fully expanded term shelf returned by the Phase E compatibility-distance calculator.

### NgeEvolutionCompatibilityDistanceWeights

Alpha weights applied to the four Phase E compatibility-distance terms.

### NgeEvolutionCompatibilityGenomeInput

One genome-side input consumed by the Phase E composite compatibility calculator.

The canonical DNA envelope does not yet own lifecycle cadence or wiring-preference
knobs, so the calculator accepts those traits as an owner-local sidecar.

### NgeEvolutionCompatibilityWiringCostWeights

Wiring-cost preference knobs compared by the Phase E lifecycle-distance term.

### NgeEvolutionContributionKind

High-level contribution kinds used to describe parent input at the reproduction boundary.

### NgeEvolutionEpigeneticPriorInput

Input contract consumed by the optional Phase E epigenetic prior operator.

### NgeEvolutionEpigeneticPriorResult

Output contract returned by the optional Phase E epigenetic prior operator.

### NgeEvolutionEpigeneticReference

Two-parent weak reference captured for one birth-time epigenetic prior update.

### NgeEvolutionParentContribution

One parent contribution reported by a Phase E reproduction operator.

### NgeEvolutionParentRole

Parent-role labels used when Phase E operators report how one offspring was assembled.

### NgeEvolutionPolyandricAssignedRegion

Region-assignment record for one drone contribution in polyandric reproduction.

### NgeEvolutionPolyandricRegionAssignmentResult

Polyandric region-assignment result reported before drone patches are applied.

### NgeEvolutionReproductionOutcome

High-level offspring outcome labels surfaced by the Phase E reproduction operators.

### NgeEvolutionReproductionResult

Shared reproduction result returned by all three Phase E reproduction modes.

### reproduceParthenogenesis

```ts
reproduceParthenogenesis(
  input: NgeParthenogenesisInput,
  mutateOffspring: NgeParthenogenesisMutationApplier,
): NgeEvolutionReproductionResult<NgeDnaCanonicalEnvelope, readonly number[]>
```

Public parthenogenesis reproduction operator exposed from one stable owner-local facade.

### reproducePolyandric

```ts
reproducePolyandric(
  input: NgePolyandricInput,
): NgeEvolutionReproductionResult<NgeDnaCanonicalEnvelope, readonly number[]>
```

Public polyandric reproduction operator exposed from one stable owner-local facade.

### reproduceSexual

```ts
reproduceSexual(
  input: NgeSexualInput,
  randomGenerator: () => number,
): NgeEvolutionReproductionResult<NgeDnaCanonicalEnvelope, readonly number[]>
```

Public sexual reproduction operator exposed from one stable owner-local facade.

## neat/nge-evolution/neat.nge-evolution.errors.ts

### NgeEvolution_BudgetError

Error thrown when one evolution operator exceeds the configured structural or patch budget.

### NgeEvolution_ModeError

Error thrown when one requested reproduction mode is unavailable for the current evolution context.

### NgeEvolution_RegionError

Error thrown when one polyandric or region-based assignment request is invalid.

## neat/nge-evolution/neat.nge-evolution.distance.ts

### averageValues

```ts
averageValues(
  values: readonly number[],
): number
```

Compute the arithmetic mean for one numeric vector.

Parameters:
- `values` - Numeric vector to average.

Returns: Mean of the vector or `0` when the vector is empty.

### buildClassicCompatibilityResult

```ts
buildClassicCompatibilityResult(
  topologyDistance: number,
): NgeEvolutionCompatibilityDistanceResult
```

Build the classic passthrough result used when NGE is disabled.

Parameters:
- `topologyDistance` - Raw classic NEAT compatibility distance.

Returns: Classic passthrough result with the NGE-only terms collapsed to zero.

### buildCompatibilityDistanceTerm

```ts
buildCompatibilityDistanceTerm(
  name: NgeEvolutionCompatibilityDistanceTermName,
  rawDistance: number,
  normalizedDistance: number,
  weight: number,
): NgeEvolutionCompatibilityDistanceTerm
```

Build one weighted compatibility-distance term.

Parameters:
- `name` - Stable term name emitted in the result shelf.
- `rawDistance` - Raw pre-normalization distance.
- `normalizedDistance` - Slice-normalized distance.
- `weight` - Alpha weight applied to the normalized distance.

Returns: Fully expanded weighted term result.

### buildNormalizationBounds

```ts
buildNormalizationBounds(
  rawDistanceRecords: readonly NgeEvolutionRawDistanceRecord[],
): Record<NgeEvolutionCompatibilityDistanceTermName, NgeEvolutionNormalizationBounds>
```

Collect min-max bounds for every normalized term across the active slice.

Parameters:
- `rawDistanceRecords` - Raw slice records prepared ahead of normalization.

Returns: Per-term min-max bounds.

### buildRawDistanceRecord

```ts
buildRawDistanceRecord(
  comparison: NgeEvolutionCompatibilityComparisonInput,
): NgeEvolutionRawDistanceRecord
```

Build the raw term record for one pairwise comparison.

Parameters:
- `comparison` - Pairwise comparison to analyze.

Returns: Raw term distances before slice-level normalization.

### buildTermBounds

```ts
buildTermBounds(
  values: readonly number[],
): NgeEvolutionNormalizationBounds
```

Build the min-max bounds for one normalized term column.

Parameters:
- `values` - Raw numeric term values for the current slice.

Returns: Minimum and maximum values for the column.

### clampUnitInterval

```ts
clampUnitInterval(
  value: number,
): number
```

Clamp one enabled composite distance into the unit interval.

Parameters:
- `value` - Composite weighted distance before clamping.

Returns: Unit-interval bounded composite distance.

### collectComputationCounts

```ts
collectComputationCounts(
  genome: NgeEvolutionCompatibilityGenomeInput,
): Record<string, number>
```

Collect the module-archetype counts keyed by computation type.

Parameters:
- `genome` - Genome-side input carrying the canonical DNA envelope.

Returns: Count map keyed by computation type.

### collectMemoryProfile

```ts
collectMemoryProfile(
  genome: NgeEvolutionCompatibilityGenomeInput,
): NgeEvolutionMemoryProfile
```

Collect the memory-tier profile used by the memory-distance term.

Parameters:
- `genome` - Genome-side comparison input carrying the canonical DNA envelope.

Returns: Presence and capacity-bin summary for recurrent and episodic tiers.

### computeDifferenceRatio

```ts
computeDifferenceRatio(
  leftValue: number,
  rightValue: number,
): number
```

Compute a symmetric normalized difference ratio for two scalar values.

Parameters:
- `leftValue` - Left-side scalar value.
- `rightValue` - Right-side scalar value.

Returns: Absolute difference divided by the larger absolute magnitude.

### computeNgeEvolutionCompatibilityDistance

```ts
computeNgeEvolutionCompatibilityDistance(
  comparison: NgeEvolutionCompatibilityComparisonInput,
  context: NgeEvolutionCompatibilityDistanceContext,
): NgeEvolutionCompatibilityDistanceResult
```

Compute the Phase E composite compatibility distance for one NGE genome pair.

The classic NEAT topology distance stays injected rather than recomputed here.
The additional NGE-only terms derive from DNA archetype composition, memory
tier shape, and an owner-local lifecycle sidecar that carries cadence and
wiring-preference knobs until those fields land in the canonical DNA schema.

Parameters:
- `comparison` - Target pairwise comparison to score.
- `context` - Optional normalization slice and alpha-weight overrides.

Returns: Composite Phase E compatibility-distance result for the target pair.

### computeRawComputationDistance

```ts
computeRawComputationDistance(
  leftGenome: NgeEvolutionCompatibilityGenomeInput,
  rightGenome: NgeEvolutionCompatibilityGenomeInput,
): number
```

Compute the raw computation-motif distance for one pair.

Parameters:
- `leftGenome` - Left genome-side comparison input.
- `rightGenome` - Right genome-side comparison input.

Returns: Raw computation-motif distance before normalization.

### computeRawLifecycleDistance

```ts
computeRawLifecycleDistance(
  leftGenome: NgeEvolutionCompatibilityGenomeInput,
  rightGenome: NgeEvolutionCompatibilityGenomeInput,
): number
```

Compute the raw lifecycle-governance distance for one pair.

Parameters:
- `leftGenome` - Left genome-side comparison input.
- `rightGenome` - Right genome-side comparison input.

Returns: Raw lifecycle-governance distance before normalization.

### computeRawMemoryDistance

```ts
computeRawMemoryDistance(
  leftGenome: NgeEvolutionCompatibilityGenomeInput,
  rightGenome: NgeEvolutionCompatibilityGenomeInput,
): number
```

Compute the raw memory-tier distance for one pair.

Parameters:
- `leftGenome` - Left genome-side comparison input.
- `rightGenome` - Right genome-side comparison input.

Returns: Raw memory-tier distance before normalization.

### computeWiringPreferenceDifference

```ts
computeWiringPreferenceDifference(
  leftWeights: NgeEvolutionCompatibilityWiringCostWeights | undefined,
  rightWeights: NgeEvolutionCompatibilityWiringCostWeights | undefined,
): number
```

Compute the normalized difference across one wiring-cost preference bag.

Parameters:
- `leftWeights` - Left-side wiring-cost weights.
- `rightWeights` - Right-side wiring-cost weights.

Returns: Mean normalized difference across the shared key union.

### normalizeTermDistance

```ts
normalizeTermDistance(
  rawDistance: number,
  bounds: NgeEvolutionNormalizationBounds,
): number
```

Normalize one raw term distance into the unit interval.

Degenerate non-zero slices resolve to `1` so a single differing pair still
contributes fully when no wider slice context is available.

Parameters:
- `rawDistance` - Raw value to normalize.
- `bounds` - Slice min-max bounds for the term.

Returns: Normalized unit-interval distance.

### resolveCompatibilityDistanceWeights

```ts
resolveCompatibilityDistanceWeights(
  weights: Partial<NgeEvolutionCompatibilityDistanceWeights> | undefined,
): NgeEvolutionCompatibilityDistanceWeights
```

Normalize a weight bag so the enabled composite sum stays bounded by one.

Parameters:
- `weights` - Optional caller overrides merged onto the Phase E defaults.

Returns: Normalized alpha weights whose sum is `1` unless every entry is `0`.

### resolveMemoryCapacityBin

```ts
resolveMemoryCapacityBin(
  capacity: number,
): number
```

Resolve the bucketed memory-capacity bin for one total capacity value.

Parameters:
- `capacity` - Aggregate hidden-dimension or slot-count value.

Returns: Stable ordinal capacity bin.

### resolveNumericParameter

```ts
resolveNumericParameter(
  parameterSchema: Record<string, unknown> | undefined,
  parameterName: string,
): number
```

Resolve one numeric module-archetype parameter or return `0` when absent.

Parameters:
- `parameterSchema` - Optional archetype parameter schema.
- `parameterName` - Parameter key to resolve.

Returns: Numeric parameter value or `0` when missing.

### resolvePopulationSlice

```ts
resolvePopulationSlice(
  comparison: NgeEvolutionCompatibilityComparisonInput,
  populationSlice: readonly NgeEvolutionCompatibilityComparisonInput[],
): readonly NgeEvolutionCompatibilityComparisonInput[]
```

Resolve the slice used for min-max normalization while keeping the target pair first.

Parameters:
- `comparison` - Target comparison that must remain the first result row.
- `populationSlice` - Optional additional comparisons from the active population slice.

Returns: Stable slice with the target comparison in the first slot.

## neat/nge-evolution/neat.nge-evolution.constants.ts

### NGE_EVOLUTION_DEFAULT_ALPHA_COMPUTATION

Default alpha weight for the computation-motif distance term.

### NGE_EVOLUTION_DEFAULT_ALPHA_LIFECYCLE

Default alpha weight for the lifecycle-policy distance term.

### NGE_EVOLUTION_DEFAULT_ALPHA_MEMORY

Default alpha weight for the memory-tier distance term.

### NGE_EVOLUTION_DEFAULT_ALPHA_TOPOLOGY

Default alpha weight for the classic topology-distance term.

### NGE_EVOLUTION_DEFAULT_COMPATIBILITY_DISTANCE_WEIGHTS

Default per-term alpha bag used when callers do not inject custom weights.

### NGE_EVOLUTION_DEFAULT_EPIGENETIC_DECAY

Default weak-reference decay used by the optional epigenetic prior operator.

### NGE_EVOLUTION_DEFAULT_POLYANDRIC_DRONE_CONTRIBUTION_FRACTION

Default fraction of DNA regions that polyandric drone donors may patch.

### NGE_EVOLUTION_DEFAULT_POLYANDRIC_QUEEN_BIAS

Default queen-bias multiplier where `1.0` means queen data wins all conflicts.

## neat/nge-evolution/neat.nge-evolution.epigenetic.ts

### applyNgeEvolutionEpigeneticPrior

```ts
applyNgeEvolutionEpigeneticPrior(
  input: NgeEvolutionEpigeneticPriorInput<ParameterVector>,
): NgeEvolutionEpigeneticPriorResult<ParameterVector>
```

Apply the optional birth-time epigenetic prior to one child parameter vector.

When no two-parent reference is configured, the operator stays a strict no-op and
returns the original child vector reference without allocating any parameter shelves.
When configured, it deterministically blends both parent references, applies the
pending mutation delta, and adds the weak decay-scaled nudge toward that blend.

Parameters:
- `input` - Child parameter state, pending mutation delta, and optional parent reference vectors.

Returns: The final child vector plus the resolved decay metadata for telemetry or tests.

Example:

```ts
const result = applyNgeEvolutionEpigeneticPrior({
  childParameters: [1, 2],
  mutationDelta: [0.5, -0.25],
  reference: {
    firstParentParameters: [2, 4],
    secondParentParameters: [0, 6],
    blendedReference: [1, 5],
  },
  decay: 0.2,
});
```

## neat/nge-evolution/neat.nge-evolution.reproduction.ts

### reproduceParthenogenesis

```ts
reproduceParthenogenesis(
  input: NgeParthenogenesisInput,
  mutateOffspring: NgeParthenogenesisMutationApplier,
): NgeEvolutionReproductionResult<NgeDnaCanonicalEnvelope, readonly number[]>
```

Build one parthenogenetic offspring from a single NGE DNA parent.

Parameters:
- `input` - Operator context containing the source parent DNA and mode flags.
- `mutateOffspring` - Optional mutation callback applied only when the configured rate is non-zero.

Returns: Canonical offspring DNA plus parent-contribution metadata.

### reproducePolyandric

```ts
reproducePolyandric(
  input: NgePolyandricInput,
): NgeEvolutionReproductionResult<NgeDnaCanonicalEnvelope, readonly number[]>
```

Build one polyandric offspring from a queen DNA template plus optional drone donors.

Parameters:
- `input` - Operator context containing the queen DNA, drone donors, and policy overrides.

Returns: Canonical offspring DNA plus the resolved region-assignment report.

### reproduceSexual

```ts
reproduceSexual(
  input: NgeSexualInput,
  randomGenerator: () => number,
): NgeEvolutionReproductionResult<NgeDnaCanonicalEnvelope, readonly number[]>
```

Build one sexual offspring using NEAT-aligned fitter-parent handling for disjoint regions.

Parameters:
- `input` - Operator context containing both parent DNAs and their relative fitness scores.
- `randomGenerator` - Deterministic selector used for matching-region crossover choices.

Returns: Canonical offspring DNA plus per-parent contribution records.

## neat/nge-evolution/neat.nge-evolution.utils.ts

### ngeEvolutionCompatibilityUtils

Compatibility-distance helpers grouped under one stable owner-local namespace.

### ngeEvolutionConstants

Default Phase E constants grouped under one stable owner-local namespace.

### ngeEvolutionEpigeneticUtils

Birth-time epigenetic helper grouped under one stable owner-local namespace.

### ngeEvolutionErrors

Error classes grouped under one stable owner-local namespace.

### ngeEvolutionReproductionUtils

Reproduction-mode helpers grouped under one stable owner-local namespace.

### ngeEvolutionUtils

Default helper bundle for the nge-evolution owner boundary.
