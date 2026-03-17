# neat/rng/core

Minimal host surface required by the RNG utilities.

This contract stays intentionally small so deterministic replay can be used
by the main `Neat` controller, tests, and diagnostics without coupling every
caller to the full runtime implementation. The contract carries only the live
RNG function, the persisted numeric state, enough population context to build
a default seed, and the option hooks that can override that default flow.

## neat/rng/core/rng.types.ts

### RngHost

Minimal host surface required by the RNG utilities.

This contract stays intentionally small so deterministic replay can be used
by the main `Neat` controller, tests, and diagnostics without coupling every
caller to the full runtime implementation. The contract carries only the live
RNG function, the persisted numeric state, enough population context to build
a default seed, and the option hooks that can override that default flow.

## neat/rng/core/rng.core.ts

Deterministic RNG primitives used by NEAT replay and diagnostics.

This chapter collects the low-level pieces: the host contract, the xorshift
constants, and the helpers that create, restore, and sample a reproducible
random stream.

### exportRngState

```ts
exportRngState(
  host: RngHost,
): number | undefined
```

Export the current RNG state for persistence.

Use this when deterministic replay must cross a broader boundary such as
JSON export, checkpointing, or test snapshots. The returned number is the
compact controller-facing representation of the current random stream.

Parameters:
- `host` - - Object holding RNG state.

Returns: The numeric RNG state or undefined when not set.

### getOrCreateRng

```ts
getOrCreateRng(
  host: RngHost,
): () => number
```

Return a cached RNG or create a deterministic xorshift RNG when absent.

This is the root runtime entrypoint for randomness. The helper resolves the
random stream in three ordered tiers:

1. reuse a previously created RNG when the stream already exists,
2. prefer a user-supplied RNG when the caller wants to own randomness
   directly,
3. otherwise create a deterministic xorshift32 stream from restored state,
   explicit seed, or a guarded default seed.

That order matters for replay. Once state has been restored, later random
draws should continue from the restored numeric state rather than silently
reseeding the controller.

Parameters:
- `host` - - Object holding RNG state and configuration.

Returns: A function that yields a uniform random value in [0, 1).

Example:

```ts
const rng = getOrCreateRng(neat);
const firstDraw = rng();
const checkpoint = snapshotRngState(neat);
```

### importRngState

```ts
importRngState(
  host: RngHost,
  state: string | number | undefined,
): void
```

Alias for restoring RNG state kept for compatibility with prior surface.

This exists so older callers can keep using the import-style name while the
underlying behavior remains the same replay boundary as `restoreRngState()`.

Parameters:
- `host` - - Object holding RNG state.
- `state` - - Numeric RNG state to restore.

### restoreRngState

```ts
restoreRngState(
  host: RngHost,
  state: string | number | undefined,
): void
```

Restore a previously captured RNG state.

Restoring state clears the cached RNG function so the next call to
`getOrCreateRng()` rebuilds the stream from the restored numeric position
instead of continuing from an older closure.

Parameters:
- `host` - - Object holding RNG state.
- `state` - - Numeric RNG state to restore.

Example:

```ts
const savedState = exportRngState(neat);
restoreRngState(neat, savedState);
```

### RngHost

Minimal host surface required by the RNG utilities.

This contract stays intentionally small so deterministic replay can be used
by the main `Neat` controller, tests, and diagnostics without coupling every
caller to the full runtime implementation. The contract carries only the live
RNG function, the persisted numeric state, enough population context to build
a default seed, and the option hooks that can override that default flow.

### sampleRandomSequence

```ts
sampleRandomSequence(
  host: RngHost,
  sampleCount: number,
): number[]
```

Produce a sequence of random samples using the host RNG.

This helper is mainly a diagnostics and testing convenience. It makes the
deterministic stream observable without forcing every caller to hand-roll its
own sampling loop, which is useful when comparing restored-state replay with
fresh execution.

Parameters:
- `host` - - Object holding RNG state.
- `sampleCount` - - Number of samples to generate.

Returns: Array of random samples in [0, 1).

Example:

```ts
const before = snapshotRngState(neat);
const samples = sampleRandomSequence(neat, 3);
restoreRngState(neat, before);
```

### snapshotRngState

```ts
snapshotRngState(
  host: RngHost,
): number | undefined
```

Snapshot the current RNG state for deterministic replay.

Use this when you want an in-memory checkpoint before a risky controller
action such as a mutation batch, debugging session, or deterministic test.
Unlike exporting a whole controller state, this is the smallest replay token:
it captures only the numeric RNG position.

Parameters:
- `host` - - Object holding RNG state.

Returns: The numeric RNG state or undefined when uninitialized.

## neat/rng/core/rng.constants.ts

Constants used by the deterministic xorshift RNG helper.

Read these values when you want to understand the fixed numeric choices that
shape seed guarding, time scrambling, and integer-to-float normalization.
The root RNG chapter uses these constants to make the replay contract
inspectable: none of these values are magic once you understand whether they
control initial seeding, xorshift mutation, or float normalization.

### RNG_DEFAULT_SEED_FALLBACK

Fallback seed used when the derived or restored seed would otherwise be zero.

Xorshift32 cannot advance from a zero state, so this constant is the guarded
non-zero escape hatch that keeps initialization and restore flows valid.

### RNG_NORMALIZATION_DIVISOR

Divisor used to normalize the 32-bit integer state into the `[0, 1)` range.

This is the final step that turns a deterministic integer state transition
into the floating-point random samples consumed by the controller.

### RNG_POPULATION_OFFSET

Minimum population offset added before time scrambling during default seeding.

The offset keeps empty or tiny populations from collapsing the derived seed
toward zero too easily during initialization.

### RNG_SHIFT_LEFT_PRIMARY

Left-shift used by the first xorshift32 mixing step.

### RNG_SHIFT_LEFT_SECONDARY

Left-shift used by the final xorshift32 mixing step.

### RNG_SHIFT_RIGHT_PRIMARY

Right-shift used by the middle xorshift32 mixing step.

### RNG_TIME_SCRAMBLE_CONSTANT

Constants used by the deterministic xorshift RNG helper.

Read these values when you want to understand the fixed numeric choices that
shape seed guarding, time scrambling, and integer-to-float normalization.
The root RNG chapter uses these constants to make the replay contract
inspectable: none of these values are magic once you understand whether they
control initial seeding, xorshift mutation, or float normalization.

## neat/rng/core/rng.utils.ts

### exportRngState

```ts
exportRngState(
  host: RngHost,
): number | undefined
```

Export the current RNG state for persistence.

Use this when deterministic replay must cross a broader boundary such as
JSON export, checkpointing, or test snapshots. The returned number is the
compact controller-facing representation of the current random stream.

Parameters:
- `host` - - Object holding RNG state.

Returns: The numeric RNG state or undefined when not set.

### getOrCreateRng

```ts
getOrCreateRng(
  host: RngHost,
): () => number
```

Return a cached RNG or create a deterministic xorshift RNG when absent.

This is the root runtime entrypoint for randomness. The helper resolves the
random stream in three ordered tiers:

1. reuse a previously created RNG when the stream already exists,
2. prefer a user-supplied RNG when the caller wants to own randomness
   directly,
3. otherwise create a deterministic xorshift32 stream from restored state,
   explicit seed, or a guarded default seed.

That order matters for replay. Once state has been restored, later random
draws should continue from the restored numeric state rather than silently
reseeding the controller.

Parameters:
- `host` - - Object holding RNG state and configuration.

Returns: A function that yields a uniform random value in [0, 1).

Example:

```ts
const rng = getOrCreateRng(neat);
const firstDraw = rng();
const checkpoint = snapshotRngState(neat);
```

### importRngState

```ts
importRngState(
  host: RngHost,
  state: string | number | undefined,
): void
```

Alias for restoring RNG state kept for compatibility with prior surface.

This exists so older callers can keep using the import-style name while the
underlying behavior remains the same replay boundary as `restoreRngState()`.

Parameters:
- `host` - - Object holding RNG state.
- `state` - - Numeric RNG state to restore.

### restoreRngState

```ts
restoreRngState(
  host: RngHost,
  state: string | number | undefined,
): void
```

Restore a previously captured RNG state.

Restoring state clears the cached RNG function so the next call to
`getOrCreateRng()` rebuilds the stream from the restored numeric position
instead of continuing from an older closure.

Parameters:
- `host` - - Object holding RNG state.
- `state` - - Numeric RNG state to restore.

Example:

```ts
const savedState = exportRngState(neat);
restoreRngState(neat, savedState);
```

### sampleRandomSequence

```ts
sampleRandomSequence(
  host: RngHost,
  sampleCount: number,
): number[]
```

Produce a sequence of random samples using the host RNG.

This helper is mainly a diagnostics and testing convenience. It makes the
deterministic stream observable without forcing every caller to hand-roll its
own sampling loop, which is useful when comparing restored-state replay with
fresh execution.

Parameters:
- `host` - - Object holding RNG state.
- `sampleCount` - - Number of samples to generate.

Returns: Array of random samples in [0, 1).

Example:

```ts
const before = snapshotRngState(neat);
const samples = sampleRandomSequence(neat, 3);
restoreRngState(neat, before);
```

### snapshotRngState

```ts
snapshotRngState(
  host: RngHost,
): number | undefined
```

Snapshot the current RNG state for deterministic replay.

Use this when you want an in-memory checkpoint before a risky controller
action such as a mutation batch, debugging session, or deterministic test.
Unlike exporting a whole controller state, this is the smallest replay token:
it captures only the numeric RNG position.

Parameters:
- `host` - - Object holding RNG state.

Returns: The numeric RNG state or undefined when uninitialized.
