# neat/multiobjective

Root orchestration for NEAT multi-objective ranking.

This chapter keeps the public ranking flow small and readable: collect the
active objective schema, precompute the population value matrix, resolve
pairwise dominance into fronts, assign NSGA-II style crowding distances, and
archive the leading fronts for later inspection.

The neighboring `objectives/`, `dominance/`, `fronts/`, and `crowding/`
chapters own the narrow mechanics. This file exists so a reader can learn
the ranking pipeline from top to bottom without digging through those lower-
level helpers first.

## neat/multiobjective/multiobjective.ts

### fastNonDominated

```ts
fastNonDominated(
  pop: default[],
): default[][]
```

Perform fast non-dominated sorting and compute crowding distances for a
population of networks (genomes). This implements a standard NSGA-II style
non-dominated sorting followed by crowding distance assignment.

The function annotates genomes with two fields used elsewhere in the codebase:
- `_moRank`: integer Pareto front rank (0 = best/frontier)
- `_moCrowd`: numeric crowding distance (higher is better; Infinity for
  boundary solutions)

Example
```ts
// inside a Neat class that exposes `_getObjectives()` and `options`
const fronts = fastNonDominated.call(neatInstance, population);
// fronts[0] is the Pareto-optimal set
```

Notes for documentation generation:
- Each objective descriptor returned by `_getObjectives()` must have an
  `accessor(genome: Network): number` function and may include
  `direction: 'max' | 'min'` to indicate optimization direction.
- Accessor failures are guarded and will yield a default value of 0.

Parameters:
- `this` - - Neat instance providing `_getObjectives()`, `options` and
`_paretoArchive` fields (function is meant to be invoked using `.call`)
- `pop` - - population array of `Network` genomes to be ranked

Returns: Array of Pareto fronts; each front is an array of `Network` genomes.
