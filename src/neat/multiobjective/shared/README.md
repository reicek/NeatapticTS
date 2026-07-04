# neat/multiobjective/shared

Shared contracts for the multi-objective ranking helpers.

This chapter is the common language layer behind the multi-objective
pipeline. The root `multiobjective/` chapter explains the ranking story at a
high level, while `objectives/`, `dominance/`, `fronts/`, and `crowding/`
each own one narrow stage of the mechanics. This file exists so those helper
chapters can agree on a deliberately small contract surface instead of
depending on the full `Neat` controller shape.

The shared surface is organized into three practical contract families:

- objective descriptors define how one genome becomes one ordered value
  vector,
- the minimal host contract exposes only the objective schema and compact
  Pareto archive state,
- transient genome annotations carry rank and crowding evidence forward after
  sorting.

The main invariant to keep in mind is stable ordering. Descriptor order
becomes vector-column order, vector-column order feeds pairwise dominance,
and the same row positions are preserved through frontier peeling and
crowding assignment. If that ordering drifts, the later helper chapters would
still run, but they would be reasoning about the wrong objectives or genomes.

## neat/multiobjective/shared/multiobjective.types.ts

### NeatLikeWithMultiObjective

Minimal Neat-like interface required by the multi-objective helpers.

This host contract stays intentionally small so the multi-objective helpers
can be reused without depending on the entire `Neat` controller surface.
The boundary owns only two kinds of state:

- objective-schema access for the start of the ranking pass,
- optional Pareto-archive state for the end of the ranking pass.

Everything else stays outside this interface on purpose. `objectives/`,
`dominance/`, `fronts/`, and `crowding/` operate on prepared vectors,
bookkeeping structures, and annotated genomes rather than reaching back into
controller internals mid-pass.

### NetworkWithMOAnnotations

Extends a genome/network with multi-objective annotations.

These properties are transient ranking metadata. They are attached after the
multi-objective helpers compute fronts and crowding distances, then consumed
by later selection or inspection code as a compact summary of where a genome
landed on the current Pareto surface.

Treat these fields as derived evidence, not durable genome state. A later
ranking pass is free to recompute or overwrite them.

### ObjectiveDescriptor

Describes how to evaluate one objective for one genome.

`objectives/` uses these descriptors to assemble one ordered value vector per
genome. `dominance/` then compares those vectors column by column, so the
descriptor array is more than configuration data: it is the schema that tells
every later helper what each column means and whether larger or smaller
values should win.

Two rules matter most:

- keep descriptor order stable for the duration of one ranking pass,
- keep each accessor deterministic for a given genome state so pairwise
  comparisons do not change mid-pass.

Notes:

- `accessor` should return a finite numeric signal for the current genome.
- `direction` controls Pareto dominance comparisons:
  - `'max'`: higher is better
  - `'min'`: lower is better
- If `direction` is omitted, it defaults to `'max'` so single-score style
  objectives keep their usual interpretation.

Example:

```ts
const objectives: ObjectiveDescriptor[] = [
  { accessor: (genome) => genome.score ?? 0, direction: 'max' },
  { accessor: (genome) => genome.cost ?? 0, direction: 'min' },
];
```
