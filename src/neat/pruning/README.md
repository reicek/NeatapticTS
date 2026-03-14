# neat/pruning

Population-pruning orchestration for the NEAT controller.

The root pruning chapter keeps the two controller entrypoints together while
pushing the lower-level option resolution and metric math into `core/` and
the stable `Neat` wrappers into `facade/`.

- `core/` explains pruning options, population metrics, and adjustment math.
- `facade/` explains the lazy public `Neat` method wrappers.

## neat/pruning/pruning.ts

### applyAdaptivePruning

```ts
applyAdaptivePruning(): void
```

Run the adaptive pruning controller.

Adaptive pruning monitors a population-level metric and nudges a shared prune
level toward a configured sparsity target. This makes pruning responsive to
the actual population rather than a fixed generation schedule.

Returns: Nothing. The shared prune level and the genomes may be updated in place.

### applyEvolutionPruning

```ts
applyEvolutionPruning(): void
```

Apply evolution-time pruning to the current population.

This entrypoint is intended for the evolve loop. It reads scheduled pruning
options from the controller, computes the current target sparsity, and asks
each genome to prune itself using the configured method.

Returns: Nothing. Genomes are pruned in place when the schedule is active.
