# neat/multiobjective/crowding

## neat/multiobjective/crowding/multiobjective.crowding.ts

### accumulateCrowdingForObjective

```ts
accumulateCrowdingForObjective(
  sortedFront: default[],
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<default, number>,
  objectiveIndex: number,
): void
```

Accumulates crowding distance contributions for a single objective.

Pre-conditions / expectations:
- `sortedFront` must be sorted ascending by the selected objective.
- {@link initializeCrowding} has already set `_moCrowd = 0` for the front.
- {@link markBoundaryCrowding} is typically called before this to set the
  boundary genomes to `Infinity`.

Edge cases:
- If the front has fewer than 2 genomes, this is a no-op.
- If the objective range is `0`, a range of `1` is used (see
  {@link resolveObjectiveRange}).

Parameters:
- `sortedFront` - - Front sorted by objective.
- `valuesMatrixInput` - - Values matrix.
- `genomeIndexByReference` - - Lookup map.
- `objectiveIndex` - - Objective column index.

### accumulateInteriorCrowding

```ts
accumulateInteriorCrowding(
  sortedFront: default[],
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<default, number>,
  objectiveIndex: number,
  valueRange: number,
): void
```

Accumulates crowding deltas for the interior genomes of a sorted front.

Interior genomes receive a normalized spacing delta:
`delta = (nextValue - previousValue) / valueRange`.

Parameters:
- `sortedFront` - - Front sorted by objective.
- `valuesMatrixInput` - - Values matrix.
- `genomeIndexByReference` - - Lookup map.
- `objectiveIndex` - - Objective column index.
- `valueRange` - - Normalized objective range.

### applyCrowdingDelta

```ts
applyCrowdingDelta(
  currentGenome: NetworkWithMOAnnotations,
  previousValue: number,
  nextValue: number,
  valueRange: number,
): void
```

Applies a normalized crowding-distance delta to a genome.

If the genome’s crowding distance is `Infinity`, it will remain `Infinity`.
This helper only updates when `_moCrowd` is initialized.

Parameters:
- `currentGenome` - - Genome to update.
- `previousValue` - - Objective value of previous genome.
- `nextValue` - - Objective value of next genome.
- `valueRange` - - Normalized objective range.

### applyCrowdingForObjective

```ts
applyCrowdingForObjective(
  front: default[],
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<default, number>,
  objectiveIndex: number,
): void
```

Applies crowding-distance accumulation for a single objective within a
single front.

Parameters:
- `front` - - Pareto front.
- `valuesMatrixInput` - - Values matrix.
- `genomeIndexByReference` - - Lookup map.
- `objectiveIndex` - - Objective column index.

### assignCrowdingDistances

```ts
assignCrowdingDistances(
  fronts: default[][],
  valuesMatrixInput: number[][],
  descriptors: ObjectiveDescriptor[],
  population: default[],
): void
```

Assigns crowding-distance annotations for each Pareto front.

This implements the crowding distance component of NSGA-II selection. Each
genome in each front receives a `_moCrowd` value representing how isolated
it is in objective space within its front.

Notes:
- This function sorts each front by each objective (ascending raw values).
  Objective direction (min vs max) does not affect the computed spacing
  magnitude; extrema are treated as boundaries either way.
- Empty fronts are skipped.

Side effects:
- Writes `_moCrowd` on each genome in each front.

Parameters:
- `fronts` - - Pareto fronts.
- `valuesMatrixInput` - - Values matrix.
- `descriptors` - - Objective descriptors (provides objective count).
- `population` - - Population to resolve indices.

### assignCrowdingForFront

```ts
assignCrowdingForFront(
  front: default[],
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<default, number>,
  objectiveIndices: number[],
): void
```

Assigns crowding distances for a single front across all objectives.

Parameters:
- `front` - - Pareto front.
- `valuesMatrixInput` - - Values matrix.
- `genomeIndexByReference` - - Lookup map.
- `objectiveIndices` - - Objective indices to process.

### buildGenomeIndexByReference

```ts
buildGenomeIndexByReference(
  population: default[],
): Map<default, number>
```

Builds a stable mapping from genome object references to their population
index.

This relies on object identity (reference equality), not structural
equality. It is used to resolve objective values from a values matrix when
working with reordered views (e.g., sorted fronts).

Parameters:
- `population` - - Genomes in population order.

Returns: Map from genome references to their index.

### buildInteriorIndexRange

```ts
buildInteriorIndexRange(
  frontLength: number,
): number[]
```

Builds the index range for interior genomes of a front.

Boundary genomes are excluded because their crowding distance is treated as
infinite.

Parameters:
- `frontLength` - - Length of the sorted front.

Returns: Interior indices excluding boundary genomes.

### buildObjectiveIndexRange

```ts
buildObjectiveIndexRange(
  objectiveCount: number,
): number[]
```

Builds a stable objective index range.

Parameters:
- `objectiveCount` - - Number of objectives.

Returns: Objective indices `0..objectiveCount-1`.

### buildSortedFrontByObjective

```ts
buildSortedFrontByObjective(
  front: default[],
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<default, number>,
  objectiveIndex: number,
): default[]
```

Builds a copy of the front sorted by the specified objective.

Sorting is ascending by the raw objective value. This ordering is used for
computing neighbor spacing in objective space.

Parameters:
- `front` - - Pareto front.
- `valuesMatrixInput` - - Values matrix.
- `genomeIndexByReference` - - Lookup map.
- `objectiveIndex` - - Objective column index.

Returns: Front sorted by objective value.

### compareObjectiveValuesForCrowding

```ts
compareObjectiveValuesForCrowding(
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<default, number>,
  objectiveIndex: number,
  leftGenome: default,
  rightGenome: default,
): number
```

Comparator used to sort genomes by a specific objective value.

Parameters:
- `valuesMatrixInput` - - Values matrix.
- `genomeIndexByReference` - - Lookup map.
- `objectiveIndex` - - Objective column index.
- `leftGenome` - - Left genome.
- `rightGenome` - - Right genome.

Returns: Numeric sort comparison value (ascending).

### initializeCrowding

```ts
initializeCrowding(
  front: default[],
): void
```

Initializes crowding-distance annotations for a front.

This sets each genome’s `_moCrowd` to `0`. Later steps accumulate per-
objective spacing deltas.

Parameters:
- `front` - - Pareto front.

### markBoundaryCrowding

```ts
markBoundaryCrowding(
  sortedFront: default[],
): void
```

Marks the boundary genomes of a sorted front as infinitely crowded.

In NSGA-II style crowding distance, boundary solutions (extremes for the
objective) are assigned an infinite crowding distance to ensure they are
always preferred when ranks tie.

Parameters:
- `sortedFront` - - Front sorted by the current objective.

### resolveBoundaryGenomes

```ts
resolveBoundaryGenomes(
  sortedFront: default[],
): { firstGenome: default; lastGenome: default; } | null
```

Resolves the boundary (first/last) genomes for a sorted front.

Parameters:
- `sortedFront` - - Front sorted by objective.

Returns: Boundary genomes, or `null` if the front is empty.

### resolveGenomeIndex

```ts
resolveGenomeIndex(
  genomeIndexByReference: Map<default, number>,
  genomeItem: default,
): number
```

Resolves a genome’s index using a reference-based map.

Parameters:
- `genomeIndexByReference` - - Lookup map created by
 *  {@link buildGenomeIndexByReference} .
- `genomeItem` - - Genome to resolve.

Returns: The population index of the genome.

### resolveNeighborPair

```ts
resolveNeighborPair(
  sortedFront: default[],
  sortedIndex: number,
): { previousGenome: default; nextGenome: default; }
```

Resolves the neighbor genomes for an interior element of a sorted front.

Parameters:
- `sortedFront` - - Front sorted by objective.
- `sortedIndex` - - Current index in sorted front.

Returns: Previous and next neighbor genomes.

### resolveObjectiveRange

```ts
resolveObjectiveRange(
  minValue: number,
  maxValue: number,
): number
```

Resolves a non-zero objective range used to normalize crowding deltas.

If all genomes have the same objective value, the raw range is `0`. This
returns `1` in that case to avoid division by zero while still producing a
well-defined crowding delta of `0`.

Parameters:
- `minValue` - - Minimum objective value.
- `maxValue` - - Maximum objective value.

Returns: Normalized range with a non-zero floor.

### resolveObjectiveRangeFromBoundaries

```ts
resolveObjectiveRangeFromBoundaries(
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<default, number>,
  boundaryGenomes: { firstGenome: default; lastGenome: default; },
  objectiveIndex: number,
): number
```

Resolves the normalized objective range for a front from its boundary
genomes.

Because `sortedFront` is sorted by objective, the first and last genomes are
the extrema used for range normalization.

Parameters:
- `valuesMatrixInput` - - Values matrix.
- `genomeIndexByReference` - - Lookup map.
- `boundaryGenomes` - - Boundary genomes for the front.
- `objectiveIndex` - - Objective column index.

Returns: Normalized value range for the objective.

### resolveObjectiveValue

```ts
resolveObjectiveValue(
  valuesMatrixInput: number[][],
  genomeIndexByReference: Map<default, number>,
  genomeItem: default,
  objectiveIndex: number,
): number
```

Resolves an objective value for a genome from a values matrix.

This is a convenience helper for working with sorted/reordered views of the
population while keeping objective values in a dense matrix.

Parameters:
- `valuesMatrixInput` - - Values matrix indexed by population index.
- `genomeIndexByReference` - - Lookup map from genome reference to index.
- `genomeItem` - - Genome to resolve.
- `objectiveIndex` - - Objective column index.

Returns: The objective value for the genome.

### shouldSkipCrowdingFront

```ts
shouldSkipCrowdingFront(
  front: default[],
): boolean
```

Determines whether crowding-distance processing should be skipped for a
front.

Parameters:
- `front` - - Pareto front.

Returns: `true` if the front should be skipped.
