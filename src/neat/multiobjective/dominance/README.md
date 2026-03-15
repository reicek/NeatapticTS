# neat/multiobjective/dominance

## neat/multiobjective/dominance/multiobjective.dominance.ts

### applyPairwiseDominance

```ts
applyPairwiseDominance(
  dominanceState: DominanceState,
  valuesMatrixInput: number[][],
  descriptors: ObjectiveDescriptor[],
  candidateIndex: number,
  opponentIndex: number,
): void
```

Applies a single pairwise dominance update between candidate and opponent.

If the candidate dominates the opponent, the opponent index is appended to
`dominatedIndicesByIndex[candidateIndex]`. If the candidate is dominated by
the opponent, `dominationCounts[candidateIndex]` is incremented.

Parameters:
- `dominanceState` - - Dominance bookkeeping.
- `valuesMatrixInput` - - Matrix of objective values.
- `descriptors` - - Objective descriptors.
- `candidateIndex` - - Candidate genome index.
- `opponentIndex` - - Opponent genome index.

### buildDominanceState

```ts
buildDominanceState(
  valuesMatrixInput: number[][],
  descriptors: ObjectiveDescriptor[],
): DominanceState
```

Builds dominance bookkeeping structures used by fast non-dominated sorting.

This computes (pairwise):
- `dominationCounts[i]`: how many genomes dominate genome `i`.
- `dominatedIndicesByIndex[i]`: which genomes are dominated by genome `i`.
- `firstFrontIndices`: genomes with `dominationCounts[i] === 0`.

Complexity:
- Time: $O(n^2 \cdot m)$ where $n$ is population size and $m$ is objective
  count.
- Space: $O(n^2)$ in the worst case for the dominated adjacency lists.

Assumptions:
- Each row in `valuesMatrixInput` is a vector aligned with `descriptors`.
- Genome ordering in later steps is expected to match the matrix ordering.

Parameters:
- `valuesMatrixInput` - - Matrix of objective values (row = genome).
- `descriptors` - - Objective descriptors (direction semantics).

Returns: Dominance bookkeeping structures for ranking.

### buildIndexRange

```ts
buildIndexRange(
  populationSize: number,
): number[]
```

Builds a stable index range for iterating the population.

Parameters:
- `populationSize` - - Number of genomes.

Returns: Array of indices `0..populationSize-1`.

### compareObjectiveValues

```ts
compareObjectiveValues(
  direction: "max" | "min",
  candidateValue: number,
  opponentValue: number,
): { isDominated: boolean; isStrictlyBetter: boolean; }
```

Compares a candidate and opponent value for a single objective.

This does not compute full Pareto dominance; it returns per-objective flags
used by the vector-level dominance check.

Parameters:
- `direction` - - Objective direction.
- `candidateValue` - - Candidate objective value.
- `opponentValue` - - Opponent objective value.

Returns: Comparison flags for this objective.

### createEmptyDominanceState

```ts
createEmptyDominanceState(
  populationSize: number,
): DominanceState
```

Creates an empty dominance state container sized to the population.

Parameters:
- `populationSize` - - Number of genomes.

Returns: An initialized dominance state with zeroed counts.

### DominanceState

Dominance bookkeeping structures for fast non-dominated sorting.

These structures are typically produced once per generation (from the values
matrix) and then consumed to build Pareto fronts.

### isCandidateDominatedByObjective

```ts
isCandidateDominatedByObjective(
  direction: "max" | "min",
  candidateValue: number,
  opponentValue: number,
): boolean
```

Checks if the candidate is worse than the opponent for a single objective.

For dominance, being worse on any objective makes the candidate unable to
dominate the opponent.

Parameters:
- `direction` - - Objective direction.
- `candidateValue` - - Candidate objective value.
- `opponentValue` - - Opponent objective value.

Returns: `true` if the candidate is dominated for this objective.

### isCandidateStrictlyBetterForObjective

```ts
isCandidateStrictlyBetterForObjective(
  direction: "max" | "min",
  candidateValue: number,
  opponentValue: number,
): boolean
```

Checks if the candidate is strictly better than the opponent for a single
objective.

Strict improvement in at least one objective is required for Pareto
dominance when the candidate is not worse in any objective.

Parameters:
- `direction` - - Objective direction.
- `candidateValue` - - Candidate objective value.
- `opponentValue` - - Opponent objective value.

Returns: `true` if the candidate is strictly better for this objective.

### isNonDominatedCandidate

```ts
isNonDominatedCandidate(
  dominanceState: DominanceState,
  candidateIndex: number,
): boolean
```

Determines whether a candidate has zero domination count.

Parameters:
- `dominanceState` - - Dominance bookkeeping.
- `candidateIndex` - - Candidate genome index.

Returns: `true` if the candidate is currently non-dominated.

### resolveDominanceOutcome

```ts
resolveDominanceOutcome(
  candidateVector: number[],
  opponentVector: number[],
  descriptors: ObjectiveDescriptor[],
): "dominates" | "dominated" | "indifferent"
```

Resolves dominance outcome between two objective vectors.

Outcome meanings:
- `'dominates'`: candidate dominates opponent.
- `'dominated'`: candidate is dominated by opponent.
- `'indifferent'`: neither dominates the other.

Parameters:
- `candidateVector` - - Candidate objective values.
- `opponentVector` - - Opponent objective values.
- `descriptors` - - Objective descriptors.

Returns: Dominance outcome between candidate and opponent.

### resolveObjectiveDirection

```ts
resolveObjectiveDirection(
  descriptors: ObjectiveDescriptor[],
  objectiveIndex: number,
): "max" | "min"
```

Resolves the objective direction for a given objective index.

If a descriptor omits `direction`, it is treated as maximization.

Parameters:
- `descriptors` - - Objective descriptors.
- `objectiveIndex` - - Objective index.

Returns: Normalized objective direction.

### shouldSkipSelfComparison

```ts
shouldSkipSelfComparison(
  candidateIndex: number,
  opponentIndex: number,
): boolean
```

Determines whether a pairwise comparison should be skipped.

Currently this skips only self-comparisons.

Parameters:
- `candidateIndex` - - Candidate genome index.
- `opponentIndex` - - Opponent genome index.

Returns: `true` if the pair should be skipped.

### updateDominanceForCandidate

```ts
updateDominanceForCandidate(
  dominanceState: DominanceState,
  valuesMatrixInput: number[][],
  descriptors: ObjectiveDescriptor[],
  candidateIndex: number,
  candidateIndices: number[],
): void
```

Updates dominance bookkeeping for a candidate against all opponents.

This iterates every opponent index and applies a pairwise dominance update.
Self-comparisons are ignored.

Parameters:
- `dominanceState` - - Dominance bookkeeping.
- `valuesMatrixInput` - - Matrix of objective values.
- `descriptors` - - Objective descriptors.
- `candidateIndex` - - Candidate genome index.
- `candidateIndices` - - Indices to compare against.

### updateStrictImprovement

```ts
updateStrictImprovement(
  hasStrictImprovement: boolean,
  isStrictlyBetter: boolean,
): boolean
```

Accumulates whether the candidate has any strict improvement across
objectives.

Parameters:
- `hasStrictImprovement` - - Current strict-improvement flag.
- `isStrictlyBetter` - - Whether the candidate strictly improves on the
current objective.

Returns: Updated strict-improvement flag.

### vectorDominates

```ts
vectorDominates(
  valuesA: number[],
  valuesB: number[],
  descriptors: ObjectiveDescriptor[],
): boolean
```

Determines whether vector A Pareto-dominates vector B.

A dominates B iff:
- A is **no worse** than B in every objective (respecting each objective’s
  direction: maximize/minimize), and
- A is **strictly better** in at least one objective.

Assumptions:
- `valuesA` and `valuesB` are aligned and have the same length.
- `descriptors` provides a descriptor for each objective index.
- If a descriptor has no `direction`, it defaults to `'max'`.

Parameters:
- `valuesA` - - Objective values for candidate A.
- `valuesB` - - Objective values for candidate B.
- `descriptors` - - Objective descriptors defining direction semantics.

Returns: `true` if A dominates B; otherwise `false`.

Example:

```ts
// Maximize accuracy, minimize latency:
vectorDominates([0.9, 120], [0.9, 150], [
  { accessor: () => 0, direction: 'max' },
  { accessor: () => 0, direction: 'min' },
]);
// => true (equal accuracy, lower latency)
```
