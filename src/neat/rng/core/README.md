# neat/rng/core

Minimal host surface required by the RNG utilities.

This contract stays intentionally small so deterministic replay can be used
by the main `Neat` controller, tests, and diagnostics without coupling every
caller to the full runtime implementation.

## neat/rng/core/rng.types.ts

### RngHost

Minimal host surface required by the RNG utilities.

This contract stays intentionally small so deterministic replay can be used
by the main `Neat` controller, tests, and diagnostics without coupling every
caller to the full runtime implementation.

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

The helper respects a user-provided RNG at `options.rng` when present.
Otherwise it seeds a xorshift32 RNG using the current time and population
size, guarding against the invalid zero seed.

Parameters:
- `host` - - Object holding RNG state and configuration.

Returns: A function that yields a uniform random value in [0, 1).

### importRngState

```ts
importRngState(
  host: RngHost,
  state: string | number | undefined,
): void
```

Alias for restoring RNG state kept for compatibility with prior surface.

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

Parameters:
- `host` - - Object holding RNG state.
- `state` - - Numeric RNG state to restore.

### RngHost

Minimal host surface required by the RNG utilities.

This contract stays intentionally small so deterministic replay can be used
by the main `Neat` controller, tests, and diagnostics without coupling every
caller to the full runtime implementation.

### sampleRandomSequence

```ts
sampleRandomSequence(
  host: RngHost,
  sampleCount: number,
): number[]
```

Produce a sequence of random samples using the host RNG.

Parameters:
- `host` - - Object holding RNG state.
- `sampleCount` - - Number of samples to generate.

Returns: Array of random samples in [0, 1).

### snapshotRngState

```ts
snapshotRngState(
  host: RngHost,
): number | undefined
```

Snapshot the current RNG state for deterministic replay.

Parameters:
- `host` - - Object holding RNG state.

Returns: The numeric RNG state or undefined when uninitialized.

## neat/rng/core/rng.constants.ts

Constants used by the deterministic xorshift RNG helper.

Read these values when you want to understand the fixed numeric choices that
shape seed guarding, time scrambling, and integer-to-float normalization.

### RNG_DEFAULT_SEED_FALLBACK

Fallback seed used when the derived seed would be zero (xorshift cannot use 0).

### RNG_NORMALIZATION_DIVISOR

Divisor used to normalize the 32-bit integer state into [0, 1).

### RNG_POPULATION_OFFSET

Minimum population offset added before scrambling to avoid zero seeds.

### RNG_SHIFT_LEFT_PRIMARY

Bit-shift values for the xorshift32 variant.

### RNG_SHIFT_LEFT_SECONDARY

### RNG_SHIFT_RIGHT_PRIMARY

### RNG_TIME_SCRAMBLE_CONSTANT

Constants used by the deterministic xorshift RNG helper.

Read these values when you want to understand the fixed numeric choices that
shape seed guarding, time scrambling, and integer-to-float normalization.

## neat/rng/core/rng.utils.ts

### exportRngState

```ts
exportRngState(
  host: RngHost,
): number | undefined
```

Export the current RNG state for persistence.

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

The helper respects a user-provided RNG at `options.rng` when present.
Otherwise it seeds a xorshift32 RNG using the current time and population
size, guarding against the invalid zero seed.

Parameters:
- `host` - - Object holding RNG state and configuration.

Returns: A function that yields a uniform random value in [0, 1).

### importRngState

```ts
importRngState(
  host: RngHost,
  state: string | number | undefined,
): void
```

Alias for restoring RNG state kept for compatibility with prior surface.

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

Parameters:
- `host` - - Object holding RNG state.
- `state` - - Numeric RNG state to restore.

### sampleRandomSequence

```ts
sampleRandomSequence(
  host: RngHost,
  sampleCount: number,
): number[]
```

Produce a sequence of random samples using the host RNG.

Parameters:
- `host` - - Object holding RNG state.
- `sampleCount` - - Number of samples to generate.

Returns: Array of random samples in [0, 1).

### snapshotRngState

```ts
snapshotRngState(
  host: RngHost,
): number | undefined
```

Snapshot the current RNG state for deterministic replay.

Parameters:
- `host` - - Object holding RNG state.

Returns: The numeric RNG state or undefined when uninitialized.
