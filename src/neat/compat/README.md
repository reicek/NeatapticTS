# neat/compat

Compatibility-distance orchestration for the NEAT controller.

The root compatibility chapter stays intentionally small so readers can find
the public distance entrypoints first, then drill into the lower-level
comparison mechanics inside `core/`.

- `core/` explains the cache lifecycle, innovation-list comparison, and the
  narrow runtime types used by compatibility checks.

## neat/compat/compat.ts

### _compatibilityDistance

```ts
_compatibilityDistance(
  genomeA: GenomeLike,
  genomeB: GenomeLike,
): number
```

Compute the NEAT compatibility distance between two genomes.

The helper keeps the top-level flow deliberately linear: refresh
generation-scoped caches, resolve sorted innovation lists, compare them, and
fold the resulting metrics into the final distance. The detailed list logic
lives in `core/` so this surface reads like the speciation contract rather
than an implementation dump.

Parameters:
- `this` - - NEAT context holding generation state, options, and caches.
- `genomeA` - - First genome to compare.
- `genomeB` - - Second genome to compare.

Returns: Compatibility distance where lower values mean more similar genomes.

### _fallbackInnov

```ts
_fallbackInnov(
  connection: ConnectionLike,
): number
```

Generate a deterministic fallback innovation id for a connection when the
connection does not provide an explicit innovation number.

This fallback encodes the `(from.index, to.index)` pair into one stable
number so compatibility distance can still compare legacy or partially
normalized genomes. Explicit innovation numbers remain the preferred source
of truth.

Parameters:
- `this` - - NEAT context kept for symmetry with the other compatibility helpers.
- `connection` - - Connection object expected to contain `from.index` and `to.index`.

Returns: Numeric innovation id derived from the directional endpoint pair.
