# neat/multiobjective/objectives

## neat/multiobjective/objectives/multiobjective.objectives.ts

### buildGenomeValues

```ts
buildGenomeValues(
  genomeItem: default,
  descriptors: ObjectiveDescriptor[],
): number[]
```

Builds an objective vector for a single genome.

The resulting array order matches the `descriptors` order exactly.
Each component is read via {@link readObjectiveValue} so individual
objective accessors are fault-tolerant.

Parameters:
- `genomeItem` - - Genome to evaluate.
- `descriptors` - - Objective descriptors (vector schema).

Returns: Objective value vector (length equals `descriptors.length`).

### buildValuesMatrix

```ts
buildValuesMatrix(
  population: default[],
  descriptors: ObjectiveDescriptor[],
): number[][]
```

Builds a population-wide objective value matrix.

The resulting matrix is indexed as `[genomeIndex][objectiveIndex]` where
`genomeIndex` matches the input `population` order.

Parameters:
- `population` - - Genomes to evaluate (population order is preserved).
- `descriptors` - - Objective descriptors (column schema).

Returns: Objective values matrix.

### readObjectiveValue

```ts
readObjectiveValue(
  genomeItem: default,
  descriptor: ObjectiveDescriptor,
): number
```

Safely reads a single objective value for a given genome.

This wraps the descriptor `accessor` in a `try/catch` so that a buggy
objective function cannot crash multi-objective ranking.

Notes:
- If the accessor throws, this returns `0` (a neutral-ish fallback).
- Callers should prefer to surface accessor errors during development;
  this helper is intentionally defensive for long-running training loops.

Parameters:
- `genomeItem` - - Genome to evaluate.
- `descriptor` - - Objective descriptor providing an accessor.

Returns: Numeric objective value; `0` if the accessor throws.

Example:

```ts
const score = readObjectiveValue(genome, { accessor: (g) => g.score ?? 0 });
```
