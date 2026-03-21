# architecture/network/deterministic

## architecture/network/deterministic/network.deterministic.utils.types.ts

### RNGSnapshot

Snapshot payload for RNG state restore flows.

### NetworkInternals

Internal deterministic network state shape used across deterministic utility modules.

### RNG_WEYL_INCREMENT

Fixed Weyl increment used to advance deterministic PRNG state.

### UINT32_NORMALIZER

Divisor used to normalize uint32 PRNG output into [0, 1).

## architecture/network/deterministic/network.deterministic.utils.ts

### setSeed

```ts
setSeed(
  seed: number,
): void
```

Sets deterministic randomness for a network by installing a seed-backed RNG.

Overview:
- Use this before training, mutation, or any stochastic operation when you need repeatable runs.
- The same seed and operation order produce the same random sequence and reproducible outcomes.
- This method delegates to setup utilities so behavior stays centralized across deterministic APIs.

Parameters:
- `this` - - Bound network instance whose RNG state is being initialized.
- `seed` - - Seed value used to derive deterministic RNG state (low 32 bits are applied).

Returns: Nothing.

Example:

```ts
network.setSeed(42);
```

### snapshotRNG

```ts
snapshotRNG(): RNGSnapshot
```

Captures the current deterministic RNG lifecycle state as a portable snapshot.

Overview:
- Use this before temporary experiments, branching simulations, or stateful debug sessions.
- The snapshot preserves enough information to resume from the same deterministic point later.
- This is useful when comparing alternate algorithm branches from an identical random timeline.

Parameters:
- `this` - - Bound network instance whose RNG lifecycle state is captured.

Returns: Snapshot containing deterministic progress metadata and RNG state payload.

Example:

```ts
const snapshot = network.snapshotRNG();
```

### restoreRNG

```ts
restoreRNG(
  fn: () => number,
): void
```

Restores deterministic RNG lifecycle behavior from a provided RNG function.

Overview:
- Use this when replaying deterministic flows after custom serialization, hydration, or test setup.
- The restored RNG function becomes the active random source used by the network lifecycle helpers.
- This keeps deterministic plumbing explicit when external code owns RNG reconstruction.

Parameters:
- `this` - - Bound network instance receiving the restored RNG lifecycle function.
- `fn` - - Deterministic RNG function to install (expected to return values in `[0, 1)`).

Returns: Nothing.

Example:

```ts
network.restoreRNG(restoredRandomFunction);
```

### getRNGState

```ts
getRNGState(): number | undefined
```

Returns the current deterministic RNG numeric state, when available.

Overview:
- Use this for lightweight checkpointing when full lifecycle snapshots are unnecessary.
- The value can be persisted and later reapplied through `setRNGState`.
- This is commonly used by tests that assert deterministic continuity across operations.

Parameters:
- `this` - - Bound network instance queried for deterministic RNG numeric state.

Returns: Numeric RNG state value, or `undefined` when no deterministic state exists yet.

Example:

```ts
const state = network.getRNGState();
```

### setRNGState

```ts
setRNGState(
  state: number,
): void
```

Applies a deterministic RNG numeric state to continue from a known checkpoint.

Overview:
- Pair this with `getRNGState` to pause/resume deterministic sequences.
- Useful for reproducible tests, multi-stage training workflows, and deterministic replay.
- Delegation keeps the write path consistent with the rest of deterministic state utilities.

Parameters:
- `this` - - Bound network instance receiving deterministic RNG state.
- `state` - - Numeric RNG state checkpoint to install.

Returns: Nothing.

Example:

```ts
network.setRNGState(savedState);
```

### getRandomFn

```ts
getRandomFn(): (() => number) | undefined
```

Returns the active deterministic RNG function currently attached to the network.

Overview:
- Use this when tooling or diagnostics need direct RNG access.
- Returning the function allows advanced integration code to inspect or reuse the random stream.
- For most persistence workflows, prefer `snapshotRNG` and `getRNGState` over direct function plumbing.

Parameters:
- `this` - - Bound network instance queried for active RNG function.

Returns: Active RNG function, or `undefined` when deterministic RNG is not initialized.

Example:

```ts
const randomFn = network.getRandomFn();
```

### RNGSnapshot

Snapshot payload for RNG state restore flows.

### network.deterministic.utils

Default export bundle for convenient named imports.

## architecture/network/deterministic/network.deterministic.setup.utils.ts

### setSeed

```ts
setSeed(
  seed: number,
): void
```

Seed the internal PRNG and install a deterministic random() implementation on the Network instance.

Parameters:
- `this` - - Bound Network instance.
- `seed` - - Any finite number; only its lower 32 bits are used.

Returns: Nothing.

### toUint32

```ts
toUint32(
  numericValue: number,
): number
```

Convert a numeric value into unsigned 32-bit state.

Parameters:
- `numericValue` - - Numeric value to normalize.

Returns: Unsigned 32-bit representation.

### setInternalSeedState

```ts
setInternalSeedState(
  internalState: DeterministicNetworkInternals,
  normalizedState: number,
): void
```

Assign normalized seed state to internal RNG storage.

Parameters:
- `internalState` - - Runtime network internals used by deterministic RNG pipeline.
- `normalizedState` - - Unsigned 32-bit state.

Returns: Nothing.

### setRandomFunction

```ts
setRandomFunction(
  internalState: DeterministicNetworkInternals,
  randomFunction: () => number,
): void
```

Assign the active random function reference.

Parameters:
- `internalState` - - Runtime network internals used by deterministic RNG pipeline.
- `randomFunction` - - PRNG function returning values in [0,1).

Returns: Nothing.

### createDeterministicRandomFunction

```ts
createDeterministicRandomFunction(
  internalState: DeterministicNetworkInternals,
): () => number
```

Create deterministic PRNG function bound to provided internal state holder.

Parameters:
- `internalState` - - Runtime network internals used by deterministic RNG pipeline.

Returns: PRNG function returning values in [0,1).

### advanceStateWithWeylIncrement

```ts
advanceStateWithWeylIncrement(
  currentState: number | undefined,
): number
```

Advance state using a fixed Weyl increment with uint32 wraparound.

Parameters:
- `currentState` - - Current state word (possibly undefined).

Returns: Next unsigned 32-bit state.

### mixStateWord

```ts
mixStateWord(
  stateWord: number,
): number
```

Mix state word with xorshift and multiplication steps.

Parameters:
- `stateWord` - - Unsigned 32-bit state word.

Returns: Mixed unsigned 32-bit word.

### toUnitInterval

```ts
toUnitInterval(
  unsignedWord: number,
): number
```

Convert unsigned 32-bit word to float in [0,1).

Parameters:
- `unsignedWord` - - Unsigned 32-bit word.

Returns: Unit-interval floating-point value.

## architecture/network/deterministic/network.deterministic.state.utils.ts

### getRNGState

```ts
getRNGState(): number | undefined
```

Get the current internal 32-bit RNG state value.

Parameters:
- `this` - - Bound Network instance.

Returns: Unsigned 32-bit state integer or undefined if generator not yet seeded or was reset.

### setRNGState

```ts
setRNGState(
  state: number,
): void
```

Explicitly set (override) the internal 32-bit RNG state without changing the generator function.

Parameters:
- `this` - - Bound Network instance.
- `state` - - Any finite number (only low 32 bits used). Ignored if not numeric.

Returns: Nothing.

### getRandomFn

```ts
getRandomFn(): (() => number) | undefined
```

Retrieve the active random function reference.

Parameters:
- `this` - - Bound Network instance.

Returns: Function producing numbers in [0,1). May be undefined if never seeded.

### isNumericState

```ts
isNumericState(
  candidateState: number,
): boolean
```

Check whether incoming state is numeric.

Parameters:
- `candidateState` - - Candidate state value.

Returns: True when state is numeric.

### toUint32

```ts
toUint32(
  numericState: number,
): number
```

Convert numeric state to unsigned 32-bit representation.

Parameters:
- `numericState` - - Numeric state value.

Returns: Unsigned 32-bit state.

## architecture/network/deterministic/network.deterministic.lifecycle.utils.ts

### snapshotRNG

```ts
snapshotRNG(): RNGSnapshot
```

Capture a snapshot of the RNG state together with the network's training step.

Parameters:
- `this` - - Bound Network instance.

Returns: Object containing current training step and 32-bit RNG state.

### restoreRNG

```ts
restoreRNG(
  fn: () => number,
): void
```

Restore a previously captured RNG function implementation and clear stored numeric state.

Parameters:
- `this` - - Bound Network instance.
- `fn` - - Function returning a pseudo-random number in [0,1).

Returns: Nothing.

### setRandomFunction

```ts
setRandomFunction(
  internalState: DeterministicNetworkInternals,
  randomFunction: () => number,
): void
```

Assign active random function reference.

Parameters:
- `internalState` - - Runtime network internals used by deterministic RNG pipeline.
- `randomFunction` - - PRNG function returning values in [0,1).

Returns: Nothing.

### clearStoredRngState

```ts
clearStoredRngState(
  internalState: DeterministicNetworkInternals,
): void
```

Clear stored numeric RNG state.

Parameters:
- `internalState` - - Runtime network internals used by deterministic RNG pipeline.

Returns: Nothing.
