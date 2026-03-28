# neat/evolve/offspring

The evolve-time offspring boundary owns the narrow mechanics of turning
selected parents into one structurally valid child.

The surrounding `population/` chapter explains how the next generation is
assembled through elitism, provenance, and offspring allocation. This smaller
chapter answers the next question underneath that orchestration: once a pair
of parents has been requested, how does evolve recover from selection failure,
annotate lineage metadata, and preserve minimum structural invariants on the
resulting child?

Read this chapter when you want to understand:

- why offspring creation has its own fallback and metadata layer,
- how lineage depth defaults behave when parent metadata is incomplete,
- why structural cleanup runs immediately after crossover.

## neat/evolve/offspring/evolve.offspring.constants.ts

### LINEAGE_BASE_DEPTH

Baseline lineage depth when parent depth metadata is missing.

This keeps lineage annotation tolerant of older or narrower runtime surfaces
that do not carry full parent-depth metadata.

### LINEAGE_DEPTH_INCREMENT

Depth increment applied when deriving a child from its parents.

Offspring depth is always one generation deeper than the deepest available
parent depth so shallow lineage summaries remain monotonic.

### OFFSPRING_FALLBACK_INDEX

Index used when falling back to the first genome in the population.

The fallback stays explicit so parent selection failure still produces a
deterministic recovery path before the helper tries more permissive random
rescue.

## neat/evolve/offspring/evolve.offspring.utils.ts

### annotateOffspringMetadata

```ts
annotateOffspringMetadata(
  context: OffspringContext,
  offspring: default,
  parentOne: default,
  parentTwo: default,
): void
```

Attach runtime metadata to a newly crossed child.

The metadata step keeps offspring creation compatible with later lineage,
telemetry, and inbreeding reads without forcing the crossover call itself to
know about controller-level bookkeeping.

Parameters:
- `context` - - NEAT-like host containing population and options.
- `offspring` - - Newly crossed child genome.
- `parentOne` - - First selected parent.
- `parentTwo` - - Second selected parent.

Returns: Nothing.

### createOffspring

```ts
createOffspring(
  context: OffspringContext,
  selectParent: () => default,
): default
```

Create a child genome by crossing two parents selected via the provided callback.

This helper is the compact crossover pipeline beneath the larger population
builder. It asks for two parents, tolerates selection failure through a small
fallback path, crosses the parents, annotates runtime metadata such as ids and
lineage depth, and finally reapplies minimum structural invariants so the new
child is ready for the rest of the evolve loop.

Parameters:
- `context` - - NEAT-like host containing population and options.
- `selectParent` - - Callback to select a parent genome.

Returns: A newly created offspring genome ready for later mutation and scoring.

### enforceOffspringInvariants

```ts
enforceOffspringInvariants(
  context: OffspringContext,
  offspring: default,
): void
```

Reapply minimum structural invariants after crossover.

Crossover can produce a child that is technically valid for heredity but still
missing the controller's minimum hidden-node or dead-end guarantees. This
helper keeps that cleanup local to offspring creation so later population code
can treat returned children as already normalized.

Parameters:
- `context` - - NEAT-like host containing population and options.
- `offspring` - - Newly crossed child genome.

Returns: Nothing.

### OffspringContext

Minimal surface needed for offspring generation.

The context stays intentionally small so offspring creation can be reused by
the broader population-construction chapter without dragging the whole evolve
controller surface into this helper layer.

### safelySelectParent

```ts
safelySelectParent(
  context: OffspringContext,
  selectParent: () => default,
  populationFallback: default[] | undefined,
): default
```

Select a parent with deterministic and random fallback recovery.

Parent-selection helpers can fail during sparse populations, edge-case test
harnesses, or aggressive species filters. Instead of letting that failure
abort offspring creation immediately, this helper first falls back to a known
stable population index and then to a random population read when needed.

Parameters:
- `context` - - NEAT-like host containing population and options.
- `selectParent` - - Callback to select a parent genome.
- `populationFallback` - - Optional alternate population to read from.

Returns: A parent genome chosen from the preferred or fallback path.
