# neat/helpers

Helper utilities for the shared NEAT controller lifecycle.

This chapter exists so pool bootstrapping, parent-derived spawning, and
external genome registration live under one direct path instead of staying in
the shrinking `src/neat` root. The public `Neat` facade still exposes the
same methods, but the code now reads like a small controller chapter with one
responsibility: manage genomes as they enter or leave the active population.

## neat/helpers/neat.helpers.ts

### addGenome

```ts
addGenome(
  genome: GenomeWithMetadata,
  parents: number[] | undefined,
): void
```

Register an externally constructed genome (for example, deserialized,
custom-built, or imported from another run) into the active population.
Ensures lineage metadata and structural invariants are consistent with
internally spawned genomes.

Defensive design: if invariant enforcement fails, the genome is still added
on a best-effort basis so experiments remain reproducible and do not abort
mid-run.

Parameters:
- `this` - Bound NEAT instance.
- `genome` - Genome / network object to insert. Mutated in place to add
internal metadata fields (`_id`, `_parents`, `_depth`, `_reenableProb`).
- `parents` - Optional explicit list of parent genome IDs (for example, two
parents for crossover). If omitted, lineage metadata is left empty.

Example:

```ts
const imported = Network.fromJSON(saved);
neat.addGenome(imported, [parentA._id, parentB._id]);
```

### createPool

```ts
createPool(
  seedNetwork: GenomeWithMetadata | null,
): void
```

Create or reset the initial population pool for a NEAT run.

If a `seedNetwork` is supplied, every genome is a structural and weight clone
of that seed. This is useful for transfer learning or continuing evolution
from a known good architecture. When omitted, brand-new minimal networks are
synthesized using the configured input/output sizes and optional minimum
hidden layer size.

Design notes:
- Population size is derived from `options.popsize` (default 50).
- Each genome gets a unique sequential `_id` for reproducible lineage.
- When lineage tracking is enabled (`_lineageEnabled`), parent and depth
  fields are initialized for later analytics.
- Structural invariant checks are best effort. A single failure should not
  prevent other genomes from being created, hence the broad try/catch.

Parameters:
- `this` - Bound NEAT instance.
- `seedNetwork` - Optional prototype network to clone for every initial genome.

Example:

```ts
// Basic: create 50 fresh minimal networks
neat.createPool(null);

// Seeded: start with a known topology
const seed = new Network(neat.input, neat.output, { minHidden: 4 });
neat.createPool(seed);
```

### GenomeWithMetadata

Genome with NEAT-specific metadata and methods.

### MutationMethod

Mutation method with optional name.

### NeatControllerForHelpers

NEAT controller interface for helper functions.

### spawnFromParent

```ts
spawnFromParent(
  parentGenome: GenomeWithMetadata,
  mutateCount: number,
): Promise<GenomeWithMetadata>
```

Spawn (clone & mutate) a child genome from an existing parent genome.

The returned child is intentionally NOT auto-inserted into the population;
call {@link addGenome} (or the class method wrapper) once you decide to keep
it. This separation lets callers validate or score the child before it joins
the population.

Evolutionary rationale:
- Cloning preserves the full topology and weights of the parent.
- A configurable number of mutation passes are applied sequentially; each
  pass may alter structure (add/remove nodes or connections) or weights.
- Lineage annotations (`_parents`, `_depth`) enable later analytics such as
  diversity statistics, genealogy visualization, and pruning heuristics.

Robustness philosophy: individual mutation failures are silently ignored so a
single stochastic edge case does not derail evolutionary progress.

Parameters:
- `this` - Bound NEAT instance (inferred when used as a method).
- `parentGenome` - Parent genome/network to clone. Must implement either
`clone()` OR a pair of `toJSON()` / static `fromJSON()` for deep copying.
- `mutateCount` - Number of sequential mutation operations to attempt; each
iteration chooses a mutation method using the instance's selection
logic. Defaults to 1 for conservative structural drift.

Returns: A new genome (unregistered) whose score is reset and whose lineage
metadata references the parent.

Example:

```ts
// Assume `neat` is an instance implementing NeatLike and `parent` is a genome in neat.population
const child = neat.spawnFromParent(parent, 3); // apply 3 mutation passes
// Optionally inspect / filter the child before adding
neat.addGenome(child, [parent._id]);
```
