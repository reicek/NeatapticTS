# architecture/network/deterministic

## architecture/network/deterministic/network.deterministic.utils.ts

### getRandomFn

`() => (() => number) | undefined`

Retrieve the active random function reference (for testing, instrumentation, or swapping).

Mutating the returned function's closure variables (if any) is not recommended; prefer using
higher-level APIs (setSeed / restoreRNG) to manage state.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *

Returns: Function producing numbers in [0,1). May be undefined if never seeded (call setSeed first).

### getRNGState

`() => number | undefined`

Get the current internal 32‑bit RNG state value.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *

Returns: Unsigned 32‑bit state integer or undefined if generator not yet seeded or was reset.

### network.deterministic.utils

Default export bundle for convenient named imports.

### restoreRNG

`(fn: () => number) => void`

Restore a previously captured RNG function implementation (advanced usage).

This does NOT rehydrate _rngState (it explicitly sets it to undefined). Intended for scenarios
where a caller has customly serialized a full RNG closure or wants to inject a deterministic stub.
If you only need to restore the raw state word produced by {@link snapshotRNG}, prefer
{@link setRNGState} instead.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `fn` - - Function returning a pseudo‑random number in [0,1). Caller guarantees determinism if required.

### RNGSnapshot

Snapshot payload for RNG state restore flows.

### setRNGState

`(state: number) => void`

Explicitly set (override) the internal 32‑bit RNG state without changing the generator function.

This is a low‑level operation; typical clients should call {@link setSeed}. Provided for advanced
replay functionality where the same PRNG algorithm is assumed but you want to resume exactly at a
known state word.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *
- `state` - - Any finite number (only low 32 bits used). Ignored if not numeric.

### setSeed

`(seed: number) => void`

Deterministic pseudo‑random number generation (PRNG) utilities for {@link Network}.

Why this module exists:
 - Facilitates reproducible evolutionary runs / gradient training by allowing explicit seeding.
 - Centralizes RNG state management & snapshot/restore operations (useful for rollbacks or
   deterministic tests around mutation sequences).
 - Keeps the core Network class focused by extracting ancillary RNG concerns.

Implementation notes:
 - Uses a small, fast 32‑bit xorshift / mix style generator (same semantics as the legacy inline version)
   combining an additive Weyl sequence step plus a few avalanche-style integer mixes.
 - Not cryptographically secure. Do not use for security / fairness sensitive applications.
 - Produces floating point numbers in [0,1) with 2^32 (~4.29e9) discrete possible mantissa states.

Public surface:
 - {@link setSeed}: Initialize deterministic generator with a numeric seed.
 - {@link snapshotRNG}: Capture current training step + raw internal RNG state.
 - {@link restoreRNG}: Provide an externally saved RNG function (advanced) & clear stored state.
 - {@link getRNGState} / {@link setRNGState}: Low-level accessors for the internal 32‑bit state word.
 - {@link getRandomFn}: Retrieve the active random() function reference (primarily for tests / tooling).

Design rationale:
 - Storing both a state integer (_rngState) and a function (_rand) allows hot-swapping alternative
   RNG implementations (e.g., for benchmarking or pluggable randomness strategies) without rewriting
   callsites inside Network algorithms.

### snapshotRNG

`() => import("C:/NeatapticTS/src/architecture/network/network.types").RNGSnapshot`

Capture a snapshot of the RNG state together with the network's training step.

Useful for implementing speculative evolutionary mutations where you may revert both the
structural change and the randomness timeline if accepting/rejecting a candidate.

Parameters:
- `this` - - Bound  {@link Network} instance.
 *

Returns: Object containing current training step & 32‑bit RNG state (both possibly undefined if unseeded).
