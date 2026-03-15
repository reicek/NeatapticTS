# neat/multiobjective/shared

Shared contracts for the multi-objective ranking helpers.

These types describe the smallest stable surface the multi-objective helpers
need: how objectives read values from genomes, which runtime fields a
Neat-like host must expose for Pareto archiving, and the transient `_mo*`
annotations attached to genomes during ranking.

## neat/multiobjective/shared/multiobjective.types.ts

### NeatLikeWithMultiObjective

Minimal Neat-like interface required by the multi-objective helpers.

This intentionally models only the fields used for archiving Pareto fronts
and retrieving objective descriptors. It allows these helpers to be used
without depending on the full Neat class type.

### NetworkWithMOAnnotations

Extends a genome/network with multi-objective annotations.

These properties are used as transient metadata during selection.

### ObjectiveDescriptor

Describes how to evaluate a single objective for a genome.

The order of objective descriptors defines the order of each genome's
objective vector and therefore the columns of the values matrix.

Notes:
- `accessor` should be deterministic for a given genome state.
- `direction` controls Pareto dominance comparisons:
  - `'max'`: higher is better
  - `'min'`: lower is better
- If `direction` is omitted, it defaults to `'max'`.

Example:

```ts
const objectives: ObjectiveDescriptor[] = [
  { accessor: (g) => g.score ?? 0, direction: 'max' },
  { accessor: (g) => g.cost ?? 0, direction: 'min' },
];
```
