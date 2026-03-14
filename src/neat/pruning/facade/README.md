# neat/pruning/facade

Public pruning facade helpers for the stable `Neat` entrypoint.

The pruning algorithms live in `pruning/pruning.ts`, while this facade keeps
the public class wrappers small and lazy-loaded so optional pruning behavior
does not clutter the main `Neat` surface.

## neat/pruning/facade/pruning.facade.ts

### applyAdaptivePruning

```ts
applyAdaptivePruning(
  host: NeatPruningFacadeHost,
): Promise<void>
```

Run adaptive pruning through the stable public `Neat` facade.

The facade keeps the lazy optional-loading behavior so callers can continue
to treat adaptive pruning as an additive maintenance feature rather than a
required runtime dependency.

Parameters:
- `host` - - `Neat` instance exposing adaptive pruning state and population metrics.

Returns: Promise that resolves after the best-effort adaptive pruning attempt finishes.

### applyEvolutionPruning

```ts
applyEvolutionPruning(
  host: NeatPruningFacadeHost,
): Promise<void>
```

Apply evolution-time pruning through the stable public `Neat` facade.

The underlying pruning module is loaded lazily so pruning remains optional
and the public facade preserves the same best-effort behavior as before this
extraction.

Parameters:
- `host` - - `Neat` instance exposing pruning options, generation state, and population.

Returns: Promise that resolves after the best-effort pruning attempt finishes.

### NeatPruningFacadeHost

Narrow `Neat` host surface required by the public pruning facade.

This stays intentionally small because the facade only forwards scheduled and
adaptive pruning calls into the extracted pruning implementation.
