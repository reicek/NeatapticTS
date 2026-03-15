# neat/evolve/offspring

Index used when falling back to the first genome in the population.

## neat/evolve/offspring/evolve.offspring.constants.ts

### LINEAGE_BASE_DEPTH

Baseline lineage depth when parent depth metadata is missing.

### LINEAGE_DEPTH_INCREMENT

Depth increment applied when deriving a child from its parents.

### OFFSPRING_FALLBACK_INDEX

Index used when falling back to the first genome in the population.

## neat/evolve/offspring/evolve.offspring.utils.ts

### createOffspring

```ts
createOffspring(
  context: OffspringContext,
  selectParent: () => default,
): default
```

Create a child genome by crossing two parents selected via the provided callback.

Parameters:
- `context` - - NEAT-like host containing population and options.
- `selectParent` - - Callback to select a parent genome.

Returns: Newly created offspring genome.

### OffspringContext

Minimal surface needed for offspring generation.
