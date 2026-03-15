# neat/multiobjective/fronts

## neat/multiobjective/fronts/multiobjective.fronts.ts

### annotateGenomeRank

```ts
annotateGenomeRank(
  population: default[],
  genomeIndex: number,
  frontRank: number,
): void
```

Annotates a genome with its Pareto front rank.

Parameters:
- `population` - - Genome population.
- `genomeIndex` - - Index of the genome to annotate.
- `frontRank` - - Pareto front rank (0 = best front).

### appendFront

```ts
appendFront(
  paretoFronts: default[][],
  population: default[],
  currentFrontIndices: number[],
): void
```

Appends the current front (index list) as genome references to the
`paretoFronts` accumulator.

Parameters:
- `paretoFronts` - - Accumulator for Pareto fronts.
- `population` - - Genome population.
- `currentFrontIndices` - - Indices for the current front.

### buildNextFrontIndices

```ts
buildNextFrontIndices(
  population: default[],
  dominanceState: DominanceState,
  currentFrontIndices: number[],
  currentFrontRank: number,
): number[]
```

Builds the next front by applying rank annotations and dominance updates.

Parameters:
- `population` - - Genome population.
- `dominanceState` - - Dominance bookkeeping.
- `currentFrontIndices` - - Indices for the current front.
- `currentFrontRank` - - Rank to assign to the current front.

Returns: Indices for the next front.

### buildParetoFronts

```ts
buildParetoFronts(
  population: default[],
  dominanceState: DominanceState,
  maxFrontRankGuard: number,
): default[][]
```

Builds Pareto fronts from a precomputed dominance state.

This performs the “peeling” phase of fast non-dominated sorting:
- Start with the first front (all non-dominated genomes).
- For each front, reduce domination counts of the genomes it dominates.
- Any genome whose domination count becomes zero moves to the next front.

Side effects:
- Annotates each genome in `population` with `_moRank` (0 = best front).

Guard:
- Stops when `currentFrontRank > maxFrontRankGuard` to avoid pathological
  infinite/degenerate runs. If the guard triggers, the returned fronts may
  be incomplete.

Parameters:
- `population` - - Genome population (same ordering used by dominance
bookkeeping).
- `dominanceState` - - Dominance bookkeeping.
- `maxFrontRankGuard` - - Safety guard for ranking iterations.

Returns: Ordered Pareto fronts (rank order).

### collectNextFrontIndices

```ts
collectNextFrontIndices(
  dominanceState: DominanceState,
  genomeIndex: number,
  nextFrontIndices: number[],
): void
```

Collects indices that become non-dominated after removing the current
genome’s dominance influence.

Parameters:
- `dominanceState` - - Dominance bookkeeping.
- `genomeIndex` - - Index of the current genome.
- `nextFrontIndices` - - Accumulator for the next front.

### incrementFrontRank

```ts
incrementFrontRank(
  currentFrontRank: number,
): number
```

Increments the front rank counter.

Parameters:
- `currentFrontRank` - - Current front rank.

Returns: Incremented front rank.

### MAX_PARETO_FRONT_RANK_GUARD

Maximum number of Pareto fronts to allow during ranking before aborting.

This is a defensive guard against pathological conditions (e.g., corrupted
dominance bookkeeping) that could otherwise cause long/infinite loops.

### shouldStopFrontRanking

```ts
shouldStopFrontRanking(
  currentFrontRank: number,
  maxFrontRankGuard: number,
): boolean
```

Determines whether ranking should stop due to a safety guard.

Parameters:
- `currentFrontRank` - - Current front rank after increment.
- `maxFrontRankGuard` - - Safety guard for ranking iterations.

Returns: `true` if ranking should stop.
