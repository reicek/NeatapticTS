# workers/eval-worker

Sim ↔ eval worker message contracts for the neatRacing in-band live
adaptation split (Phase 2, Step 02, slice P2-S2-A — tdd_sequence
`green-only`).

Spec anchors:
- §9.2 two-worker split: the simulation worker owns 60 Hz physics and
  per-tick control (E11(a)); this protocol carries every growth/variant
  evaluation request to the dedicated eval worker (E11(b)), whose
  replies the sim side queues and applies only at deterministic
  adaptation-tick boundaries — never on reply arrival (E11(c)).
- E11(d): eval requests are self-contained per evaluation window. The
  payload carries the AC-026 tuple seed and the sim tick the request was
  issued at, and every result echoes both. The echo is what lets the
  sim-side queue drop superseded-round replies after a round-boundary
  `stop`, so no reply from a cancelled round can mutate a new round.
- D2: variant RNG streams are derived by the eval worker itself from the
  payload tuple seed (§9.7.2 row-4 composition via the recycled
  reproducibility table in ../simulation-worker/reproducibility.utils).
  The eval worker NEVER consumes or advances sim-worker network RNG
  state, so the protocol ships serialized payloads only.
- TP-18: live-mode variant dispatch is bounded by the named constant
  NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT below — replacing the
  racing-curriculum demo's unbounded parallelVariantCount: 256.

Realm note: this module is deliberately realm-free (no `self`, no
`postMessage`): the sim and eval sides share these same contracts, and
Jest drives them headless. The worker-realm glue lives in
./eval-worker.ts — the eval side of the wire (session, bootstrap unit,
message adapter) — so both workers stay independently testable.

## workers/eval-worker/eval-worker.ts

### createEvalWorkerErrorReply

```ts
createEvalWorkerErrorReply(
  message: string,
): NeatRacingEvalWorkerResponse
```

Builds the single error reply for contract-level rejections.

Parameters:
- `message` - Static rejection vocabulary.

Returns: The error response to post.

### createEvalWorkerSession

```ts
createEvalWorkerSession(
  options: { evaluateVariant: NeatRacingEvalVariantUnit; },
): NeatRacingEvalWorkerSession
```

Creates a fresh eval-worker session (pure, no worker realm).

Overlapping eval requests are legal: each window is independent and
self-contained (its own tuple seed + variants + echoed identity), which
is what lets a Step 03 sim-side queue pipeline windows without the eval
worker holding cross-window state — and keeps every reply replayable
from its request alone (D2).

Parameters:
- `options` - The variant evaluation unit bound to this session.

Returns: The eval worker session state machine.

### createNeatRacingBootstrapVariantUnit

```ts
createNeatRacingBootstrapVariantUnit(): NeatRacingEvalVariantUnit
```

Creates the protocol-bootstrap variant unit used until the Step 03 NGE
growth engine lands (the remaining AC-029/AC-043 surface).

Educational note: the unit scores each slot from the variant stream
itself (D2(b)) — it draws the FIRST xorshift32 float of the stream
seeded with the per-variant child seed. The eval worker's end-to-end
behavior stays deterministic and replayable (same `(seed, variants)` →
identical scores) while still exercising the exact §9.7.2 derivation
table the real engine will receive. Step 03 swaps this unit through the
session's `evaluateVariant` option — the protocol, the seed derivation,
and the bounded dispatch all carry over unchanged.

Returns: The bootstrap evaluation unit.

### EvalWorkerCancelled

Private round-boundary cancellation sentinel thrown by in-flight task
wrappers after a `stop`. Never escapes the session: `handleEval`
converts it into "no reply" (E11(d)) rather than an error reply.

### NEATRACING_EVAL_WORKER_INTERNALS

Pure-function test seam for the eval worker (P2-S2-A green tests).

Educational note: like every established worker module in this repo
(flappy evolution, asciiMaze, the neatRacing sim worker), the entrypoint
exposes its internals so Jest can drive the protocol contract headless
without a worker realm, while production still boots through the module
side effect below.

### NeatRacingEvalVariantUnit

```ts
NeatRacingEvalVariantUnit(
  context: NeatRacingEvalVariantUnitContext,
): number | Promise<number>
```

A variant evaluation unit: produces one deterministic fitness score per
slot (sync or async). The Step 03 NGE growth engine plugs in here
through the session's `evaluateVariant` option without touching the
protocol or the bounded dispatch.

### NeatRacingEvalVariantUnitContext

Inputs the worker hands its variant-evaluation unit (D2 + §9.7.2).

### NeatRacingEvalWorkerPhase

Eval worker phases: ready until a round-boundary `stop` lands.

### NeatRacingEvalWorkerSession

Eval worker session contract (pure state machine, no worker realm).

### NeatRacingEvalWorkerSessionResponse

The eval-side response union (aliasing the protocol wire type so the
session's signature reads as its own contract).

### NeatRacingEvalWorkerSessionSnapshot

Session snapshot for green tests (read-only view of the phase).

## workers/eval-worker/eval.protocol.ts

### NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT

Live-mode parallel variant evaluation bound (TP-18).

Educational note: the racing-curriculum demo evaluated up to 256 weight
variants in a single Promise.all window
(examples/racing_curriculum/browser-entry/browser-entry.ts:
`parallelVariantCount: 256`). In-band live mode works inside the 2 ms
adaptation-tick budget (TP-18: NEATRACING_ADAPTATION_TICK_BUDGET_MS),
so the eval worker dispatches at most 16 concurrent variant evaluations
— mirroring the library default
DEFAULT_ACCELERATION_PARALLEL_VARIANT_COUNT in
src/acceleration/acceleration.constants.ts — and processes larger
request batches as sequential bounded windows via runBoundedConcurrency.

### NeatRacingEvalWorkerErrorPayload

eval → sim error payload (contract-level rejections).

### NeatRacingEvalWorkerEvalPayload

sim → eval request payload: one bounded evaluation window.

### NeatRacingEvalWorkerRequest

sim → eval request union.

### NeatRacingEvalWorkerResponse

eval → sim response union.

### NeatRacingEvalWorkerResultPayload

eval → sim result payload for one evaluation window.

### NeatRacingEvalWorkerTask

```ts
NeatRacingEvalWorkerTask(): T | Promise<T>
```

A lazy evaluation task factory: invoked only when a dispatch slot frees.

### NeatRacingEvalWorkerValidationError

Validation failure carrier mirroring the sim worker's `{ error }` shape.

### NeatRacingEvalWorkerVariant

One variant slot inside an eval request (D2: eval-owned, serialized).

The growth engine consumes these slots in Step 03; for now the protocol
pins only their transport shape so the sim side can be wired against a
stable contract.

### NeatRacingEvalWorkerVariantResult

One scored variant slot, reported in request order.

### runBoundedConcurrency

```ts
runBoundedConcurrency(
  tasks: readonly NeatRacingEvalWorkerTask<T>[],
  parallelCount: number,
): Promise<T[]>
```

Runs lazy task factories under a bounded parallel dispatch and resolves
the results in input order.

Educational note: mirrors the neatenstein bounded-concurrency harness
(examples/neatenstein/browser-entry/harness/bounded-concurrency.ts) so
both examples keep the same vocabulary without a cross-example import.
Concurrency never exceeds `parallelCount` (TP-18: live-mode dispatch
stays inside the 2 ms adaptation-tick budget) and larger batches run as
sequential bounded windows. The chain never reads the clock and never
mutates timers, so fake-timer tests install freely.

Parameters:
- `tasks` - Lazy factories; each is invoked at most once, only when a
slot is free.
- `parallelCount` - Dispatch bound; defaults to
NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT.

Returns: The results indexed exactly like the input tasks.

### validateNeatRacingEvalWorkerEvalPayload

```ts
validateNeatRacingEvalWorkerEvalPayload(
  payload: unknown,
): NeatRacingEvalWorkerValidationError | null
```

Validates one eval request payload (contract-level, static messages).

Returns a rejection-reason record when the payload violates the eval
protocol, or null when it is well-formed. Mirrors the sim worker's
validation-helper style so both workers reject malformed traffic with
their single `error` reply.

Validation surface: seed must be a finite number; requestTick must be a
non-negative integer; variants must be a non-empty array of objects each
carrying a non-negative integer variantIndex, an optional non-negative
integer patchSize, and a present (opaque) snapshot. The snapshot CONTENT
is deliberately unvalidated — it is the growth engine's input (Step 03),
and the protocol only guarantees transport (D2).

Parameters:
- `payload` - The decoded `eval` request payload as received on the
wire (untyped on purpose: the contract guards structured-clone traffic).

Returns: A rejection record, or null when the payload is well-formed.
