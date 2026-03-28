# neat/rng/facade

Public RNG maintenance facade helpers for the stable `Neat` entrypoint.

These helpers expose deterministic replay and lightweight random sampling
without forcing the main [src/neat.ts](src/neat.ts) facade to carry the
lower-level RNG state plumbing inline. Grouping them here keeps the public
class orchestration-focused while preserving the same debugging and replay
affordances that tests and examples already use.

Invariant: this boundary only manages deterministic RNG state progression
for replay and diagnostics. It does not change evolution policy, mutation
behavior, or selection semantics.

Read this facade as the stable wrapper layer above the RNG core chapter. The
core helpers own xorshift state mechanics and replay contracts; this file
translates those primitives into public `Neat`-shaped methods that are easier
to call from debugging sessions, tests, persistence flows, and examples.

The wrapper surface naturally falls into three pairs:

1. `snapshotRNGState()` and `sampleRandom()` support in-process diagnostics,
2. `exportRNGState()` and `restoreRNGState()` support persistence and replay,
3. `importRNGState()` preserves the legacy public name for the same restore
   boundary.

## neat/rng/facade/rng.facade.ts

### exportRNGState

```ts
exportRNGState(
  host: RngHost,
): number | undefined
```

Export the current RNG state for persistence or debugging.

Use this wrapper when the replay token is likely to leave the immediate call
site, such as checkpoint serialization, fixture capture, or bug-report
handoff. Compared with `snapshotRNGState()`, the emphasis here is portability
rather than momentary inspection.

Parameters:
- `host` - - `Neat` instance exposing RNG state.

Returns: Numeric RNG state or `undefined` when the RNG has not been initialized.

### importRNGState

```ts
importRNGState(
  host: RngHost,
  state: string | number | undefined,
): void
```

Import an RNG state using the legacy compatibility name.

This mirrors `restoreRNGState` so older callers can keep using the stable
public method name while the implementation stays centralized.

Read this as a compatibility alias, not a second persistence model. New
readers should understand it as the same replay boundary with older naming.

Parameters:
- `host` - - `Neat` instance exposing RNG state.
- `state` - - Numeric or string seed snapshot to restore.

Returns: Nothing. The helper mutates the host RNG state in place.

### NeatRngFacadeHost

Narrow `Neat` host surface required by the public RNG facade.

This stays intentionally small because the facade only forwards state
snapshot, restore, export, and sampling calls into the existing RNG helpers.

Keeping the host contract tiny protects the public wrapper boundary from the
rest of the controller. Replay helpers need RNG state, optional configuration,
and lightweight population context, but they do not need mutation, selection,
or telemetry internals.

### restoreRNGState

```ts
restoreRNGState(
  host: RngHost,
  state: string | number | undefined,
): void
```

Restore a previously captured RNG state.

The next RNG access will continue from the restored subsequence, which makes
it useful for replaying evolution bugs or comparing deterministic traces.

Pair this with `snapshotRNGState()` or `exportRNGState()` when you want to
prove that two runs consume the same future subsequence. The wrapper does not
draw a random number itself; it only repositions the next draw.

Parameters:
- `host` - - `Neat` instance exposing RNG state.
- `state` - - Numeric or string seed snapshot to restore.

Returns: Nothing. The helper mutates the host RNG state in place.

### sampleRandom

```ts
sampleRandom(
  host: RngHost,
  sampleCount: number,
): number[]
```

Produce deterministic random samples from the instance RNG.

This read-only helper is useful for tests and diagnostics that want to prove
two runs share the same random subsequence after restoring a snapshot.

Sampling advances the live stream, so this helper is best paired with a
snapshot-then-restore pattern when the caller wants to inspect a sequence
without permanently consuming that portion of the run's randomness budget.

Parameters:
- `host` - - `Neat` instance exposing RNG state.
- `sampleCount` - - Number of random values to generate.

Returns: Deterministic random samples in the range `[0, 1)`.

Example:

```ts
const snapshot = neat.snapshotRNGState();
const firstSequence = neat.sampleRandom(3);
neat.restoreRNGState(snapshot);
const repeatedSequence = neat.sampleRandom(3);
console.log(firstSequence, repeatedSequence);
```

### snapshotRNGState

```ts
snapshotRNGState(
  host: RngHost,
): number | undefined
```

Return the current opaque RNG numeric state used by the instance.

This is mainly useful for deterministic replay, test fixtures, and bug
reports that need to resume the same random subsequence.

Prefer this wrapper when the state is staying inside the current debugging or
test session. It captures the current numeric position without implying that
the token is about to cross a persistence boundary.

Parameters:
- `host` - - `Neat` instance exposing RNG state.

Returns: Numeric RNG state or `undefined` when the RNG has not been initialized.
