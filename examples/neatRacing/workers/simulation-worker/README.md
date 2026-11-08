# workers/simulation-worker

In-band live simulation worker for the neatRacing browser demo (Phase 2,
Step 01, slice P2-S1-A).

Educational note — why this file exists:
The phase-1 race loop lived on the browser main thread (`host/race-session.ts`):
every rAF frame advanced physics, race rules, and controllers inline, which
couples the frame budget of the render pipeline to the cost of the simulation.
The Phase-2 learning-document contract (LD P2-S1) moves that authority to a
single in-band live-worker file: the worker owns ALL simulation state, and
the host only paces frames and forwards the once-per-frame human control
sample (decision D5) as a message. Every piece of racing truth now crosses a
typed message boundary, so the ownership question ("which side owns this
state?") has one visible answer for every field.

What this worker owns per slice P2-S1-A:
- the 60 Hz fixed-step accumulator (`host/render-loop.utils.ts` reuse),
- the car fleet physics (`stepPhysics` per car through `stepPhysicsFleet`,
  whose single live-car path stays `stepPhysics`),
- the race-rules state machine (`createRaceRulesState` / `stepRaceRules`),
- the autopilot controllers (`stepAutopilotInput`) for the `ai` mode and
  every agent car beyond car 0,
- the per-tick forward lap-line crossing detector (ported phase-1 host
  boot logic, now owned where the car state it reads lives).

What the host keeps until later phases: rendering, HUD, selector, and the
keyboard-sampling seam — the host samples driving input once per drawn
frame (D5) and ships it to the worker with the frame timestamp.

Message protocol contract (slice A):
- host → worker `{ type: 'init', payload }` boots the session. The payload
  carries the AC-026 reproducibility tuple fields (`seed`, `agentCount`,
  `packSchemaVersion`, `opponentSnapshot`, `trackId`, `featureFlags`) plus
  the `trackSpec` geometry the sim needs to own its state. Replies exactly
  one `{ type: 'ready', payload }`.
- host → worker `{ type: 'frame', payload }` paces one drawn frame
  (`timestampMs` drives the accumulator; `sample` is the D5 human control
  sample for car 0 in `manual` mode). Replies exactly one
  `{ type: 'frames', payload }` whose payload snapshots the released ticks
  (`fromTick`..`toTick`), the post-frame fleet, and the post-frame race
  rules. Slice B replaces this payload with the packed
  `neatracing-packed-v1` budget; the message shape is the stable seam.
- host → worker `{ type: 'stop' }` halts the session. Replies nothing
  (fire-and-forget lifecycle message, matching the flappy stop precedent).
- anything else (unknown type, malformed payload, bad ordering) replies one
  `{ type: 'error', payload: { message } }`.

Determinism discipline (LD D1 / D5, AC-026):
- no `Math.random` anywhere in this file; every spawn is derived from the
  init seed and car index, and every controller is a pure step function.
- no clock reads inside the tick pipeline: wall-clock timestamps enter ONLY
  through `frame` messages and feed the release accumulator, never tick math.
- identical tuple + identical frame pacing + identical sample streams
  produce identical tick content (Level-2 ordered determinism), which the
  byte-parity slice (P2-S1-C) formalizes.

## workers/simulation-worker/simulation-worker.ts

### buildSimWorkerFramesPayload

```ts
buildSimWorkerFramesPayload(
  fromTick: number,
  sessionState: SimWorkerSessionState,
): NeatRacingSimWorkerFramesPayload
```

Builds the structured `frames` reply for one paced frame.

Parameters:
- `fromTick` - Total-tick counter before the frame's released ticks.
- `sessionState` - Running session bag (post-tick state).

Returns: The reply payload snapshot (fresh copies, no aliasing into later
tick output).

### createSimWorkerRunningState

```ts
createSimWorkerRunningState(
  payload: NeatRacingSimWorkerInitPayload,
): SimWorkerSessionState & { readyDetail: { seed: number; agentCount: number; packSchemaVersion: string; trackId: string; }; }
```

Boots the sim state for one session from a validated init payload.

Parameters:
- `payload` - Validated init payload.

Returns: The fresh internal session bag in the `running` phase, with the
boot detail guaranteed non-undefined (the running bag always carries it).

### createSimWorkerSession

```ts
createSimWorkerSession(): NeatRacingSimWorkerSession
```

Creates a pure sim-worker session state machine.

Educational note: one factory per session keeps every boot test isolated
without touching `globalThis.self`. The worker entrypoint below builds the
module-level session and is the ONLY place that wires `self.onmessage` /
`self.postMessage` — the transport adapter stays three lines.

Returns: A session whose `handleRequest` folds messages into state and
returns the ordered responses to post.

Example:

```ts
const session = createSimWorkerSession();
const [ready] = session.handleRequest({ type: 'init', payload });
```

### driveSimWorkerFrame

```ts
driveSimWorkerFrame(
  sessionState: SimWorkerSessionState,
  timestampMs: number,
  framePayload: NeatRacingSimWorkerFramePayload,
): void
```

Releases 0-4 fixed steps for one paced frame and mutates the running session.

Parameters:
- `sessionState` - Running session bag mutated in place.
- `timestampMs` - rAF-style frame timestamp driving the accumulator.
- `framePayload` - The paced frame payload (its optional D5 sample is
latched BEFORE ticking so zero-step frames still refresh the held manual
sample, matching the phase-1 sampling-outside-the-step-guard behavior).

### NEATRACING_SIM_WORKER_LAP_LINE_ALONG_EPS_M

Along-track deadband (meters from the lap-line plane) below which samples
do not update the detector baseline. Carried verbatim from the phase-1 host
detector.

### NEATRACING_SIM_WORKER_LAP_LINE_LATERAL_GATE_M

Lateral gate (meters from segment 0's chord) inside which a pose counts as
crossing the lap line. Carried verbatim from the phase-1 host detector so
the worker-owned port stays behaviorally identical.

### NEATRACING_SIM_WORKER_MAX_AGENTS

Largest legal fleet size for a live sim-worker session.

Educational note: the AC-033 demo fleet is three cars (human car 0 plus
two evolving agents), so this cap is deliberately generous rather than
load-bearing. The NGE eval-worker slice revisits the bound when real
populations flow through instead of hard-coding a tuning guess here.

### NEATRACING_SIM_WORKER_MIN_AGENTS

Smallest legal fleet size for a live sim-worker session (just the car).

### NEATRACING_SIM_WORKER_SPAWN_ALONG_LEAD_M

Along-track lead distance (meters past segment 0's START point) where the
first fleet car spawns, matching the phase-1 host spawn recipe exactly
(segment-0 chord heading, 2 m along it) so the AC-026 baseline run keeps
the same opening pose the host used.

### NEATRACING_SIM_WORKER_SPAWN_GRID_PITCH_M

Line-astern grid pitch (meters) between consecutive fleet spawn slots.

### NEATRACING_SIMULATION_WORKER_INTERNALS

Pure-function test seam for the sim worker (P2-S1-A green tests).

Educational note: like every established worker module in this repo
(flappy evolution, asciiMaze), the entrypoint exposes its internals so
Jest can drive the protocol contract headless without a worker realm,
while production still boots through the module side effect below.

### NeatRacingSimWorkerErrorPayload

Error payload for any rejected request.

### NeatRacingSimWorkerFramePayload

Per-frame pace message payload (one drawn frame, one reply).

### NeatRacingSimWorkerFramesPayload

Per-frame reply payload (slice A structured snapshots).

Slice B swaps this payload for the packed `neatracing-packed-v1` budget;
the tick-range bookkeeping below is what the pack must preserve.

### NeatRacingSimWorkerInitPayload

Init payload for the sim worker.

The AC-026 reproducibility tuple components ride alongside the track
geometry so one init message carries everything a replay bootstrap needs:
`trackId` names the tuple's track while `trackSpec` supplies the actual
owned geometry (segments for the surface resolver and lap detector,
spline samples for the autopilot controllers).

### NeatRacingSimWorkerPhase

Lifecycle phases of one sim-worker session.

### NeatRacingSimWorkerRaceMode

Race-mode knob the host boots the session with (selector state at init).

### NeatRacingSimWorkerReadyPayload

Boot acknowledgement payload echoing the session identity tuple.

### NeatRacingSimWorkerRequest

Host → worker request message union (the sim-worker protocol inbox).

### NeatRacingSimWorkerResponse

Worker → host response message union (the sim-worker protocol outbox).

### NeatRacingSimWorkerSession

Public session handle: a pure request/response state machine over the bag.

### NeatRacingSimWorkerSessionSnapshot

Read-model diagnostics snapshot of one sim-worker session.

### NEUTRAL_SIM_WORKER_CONTROL_SAMPLE

Neutral control sample replayed for car 0 in `manual` mode before the host
has sent any sample. Frozen: it is a constant, never a live accumulator.

### resolveSimWorkerFleetSpawnPoses

```ts
resolveSimWorkerFleetSpawnPoses(
  trackSpec: TrackSpec,
  agentCount: number,
): PhysicsCarSpawnPose[]
```

Resolves the deterministic fleet spawn poses from the owned track.

Educational note: car 0 inherits the phase-1 host recipe exactly (segment-0
chord heading, a 2 m lead along it), while cars behind it line up astern on
the same centerline at fixed grid pitch so the fleet start stays a function
of the init config alone (no RNG anywhere). The formal child-seed system is
the byte-parity slice's job (P2-S1-C); this slice derives per-car seeds by
the documented index offset when it builds the fleet.

Parameters:
- `trackSpec` - Owned track geometry.
- `agentCount` - Fleet size (>= 1).

Returns: One spawn pose per fleet slot, ordered car 0 first.

### resolveSimWorkerFleetTickInputs

```ts
resolveSimWorkerFleetTickInputs(
  cars: readonly PhysicsCarState[],
  raceMode: NeatRacingSimWorkerRaceMode,
  humanSample: PhysicsStepInput | undefined,
  splineSamples: readonly SplineSample[],
): PhysicsStepInput[]
```

Resolves the per-tick control inputs for the whole fleet.

Educational note: in `manual` mode car 0 runs on the host's D5 sample (the
latest received one is replayed across catch-up ticks), while in `ai` mode
car 0 AND every agent car run the worker-owned autopilot step function —
a pure function of the entering pose and the frozen spline samples, so the
whole control stream stays a deterministic function of sim state.

Parameters:
- `cars` - Fleet states entering this tick.
- `raceMode` - Session race mode.
- `humanSample` - Latest host sample (may be undefined pre-first sample).
- `splineSamples` - Frozen track spline samples for the autopilot.

Returns: Exactly one input per fleet car, ordered like the fleet.

### resolveSimWorkerLapLineDetectorStep

```ts
resolveSimWorkerLapLineDetectorStep(
  car: PhysicsCarState,
  prevSide: number | null,
  firstSegment: TrackSegment | undefined,
): { crossing: boolean; nextSide: number | null; }
```

One step of the worker-owned lap-line detector: decides whether the pose
ENTERING this tick crossed the finish plane forward, and carries the
detector baseline forward.

Educational note: this is the exact port of the phase-1 host detector
(browser-entry boot glue) — the finish line is the plane through segment
0's start point perpendicular to its chord; a crossing counts only inside
the lateral gate, samples inside the along-track deadband never update the
baseline, and leaving the gate resets it (a car that wandered off can never
bank a stale side). Phase-1 semantics kept: direction filtering is a
carried gap, so ANY sign flip counts.

Parameters:
- `car` - Fleet car state (the tick's entering pose).
- `prevSide` - Latched along-track side from the previous qualifying
sample, or null before the first one.
- `firstSegment` - Segment 0 of the owned track (or undefined when the
geometry is degenerate; the detector then never reports a crossing).

Returns: The crossing verdict plus the baseline to carry forward.

### SimWorkerSessionState

Internal mutable session bag: all sim truth owned by the worker lives here.

### SimWorkerValidationError

Validation failure carries a static rejection reason string.

### validateSimWorkerFramePayload

```ts
validateSimWorkerFramePayload(
  payload: unknown,
): SimWorkerValidationError | null
```

Validates a frame payload (object with a finite timestamp; an optional
sample carrying finite axes).

Parameters:
- `payload` - Candidate frame payload.

Returns: Rejection reason, or null when valid.

### validateSimWorkerInitPayload

```ts
validateSimWorkerInitPayload(
  payload: unknown,
): SimWorkerValidationError | null
```

Validates an init payload and returns a rejection reason, or null when the
payload may boot. The carried tuple fields (`opponentSnapshot`,
`featureFlags`) are intentionally unvalidated: slice A transports them, the
byte-parity slice pins their serialized shape.

Parameters:
- `payload` - Candidate init payload.

Returns: Rejection reason, or null when valid.

## workers/simulation-worker/frame.pack.service.ts

Frame pack service for the neatRacing simulation worker (Phase 2, Step 01,
slice P2-S1-B).

Educational note — why this file exists:
The P2-S1-A worker currently replies to every paced frame with structured
clones (plain `PhysicsCarState[]` objects). Structured cloning allocates a
fresh object graph per frame and pays the clone tax on the host<->worker
boundary every round trip. The Phase-2 learning-document contract moves the
per-frame transfer budget onto **preallocated buffers with explicit
offsets** (AC-042 partial): the worker packs each fleet snapshot into a
reusable slot buffer laid out as structure-of-arrays (SoA) channels, the
host reads (or unpacks) the same layout, and the only buffers that ever
cross `postMessage` are the slots preallocated once per session.

Wire contract (AC-025):
- The schema sentinel is `'neatracing-packed-v1'` (the NeatRacing analogue
  of the racing-curriculum `'racing-packed-v1'` sentinel); consumers must
  reject any other value via `assertNeatRacingPackedFrameSchemaVersion`
  so a future schema drift is caught at the message boundary instead of
  silently misreading packed bytes.
- EVERY packed frame channel is a `Float32Array` (D8a; the racing
  precedent packs every channel the same way). Float32 keeps the packing
  deterministic — pack/unpack round-trips are Float32-lossy but exactly
  reproducible, which the byte-parity slice (P2-S1-C) formalizes.
- The feature-flag bit table is the recycled D8b assignment: bit 0 =
  `tiresEnabled`, bit 1 = `radioEnabled`, bit 2 = `pitsEnabled`. NeatRacing
  pins tires = 0 and pits = 0 (no tire wear, no pit stops) and keeps
  radio = 1 (the observation tails survive), so the default bitfield
  encodes `{ tires: false, radio: true, pits: false }`.

Preallocated-buffer contract (AC-042 partial):
- `createNeatRacingFramePackPool` allocates ONE `ArrayBuffer` per slot,
  sized once from the packer's `carCapacity`, and partitions it into a
  rules span followed by `NEATRACING_PACKED_CAR_CHANNEL_COUNT` SoA channel
  spans, all at computed byte offsets. Views (`Float32Array` subarrays)
  are created exactly once per slot; packing writes into them in place.
- `packFrame` allocates NOTHING: it round-robins over the slots, mutates
  the slot's reused frame object, and fills the channels from a single
  fixed channel table. The double-buffered default (`poolSlots: 2`) keeps
  the previously packed frame readable while the next one packs because a
  packed slot stays valid until `poolSlots` further packs overwrite it.
- `resolveNeatRacingPackedFrameTransferList` collects the distinct buffers
  backing a packed frame so `postMessage(message, transferList)` transfers
  them zero-copy; with the single-buffer layout the list is always exactly
  one entry (`NEATRACING_PACKED_TRANSFER_BUFFER_COUNT`).

Recycling note (spec §11): the SoA channel pattern, the transfer-list
resolver, and the schema-sentinel guard are recycled from
`examples/racing_curriculum/workers/simulation-worker/`
(`simulation-worker.race-pack.service.ts` + `simulation-worker.snapshot.utils.ts`),
reshaped for the NeatRacing fleet + race-rules state and the single
preallocated slot buffer the AC-042 budget prefers.

Determinism discipline (D8a / AC-026): no `Math.random`, no clock reads;
packing is a pure state->bytes function of its input.

### assertNeatRacingPackedFrameSchemaVersion

```ts
assertNeatRacingPackedFrameSchemaVersion(
  frame: { schemaVersion: unknown; },
): void
```

Throws a `RangeError` when a received frame does not carry the
`'neatracing-packed-v1'` sentinel.

Consumers must call this before reading any channel so a schema mismatch
fails loudly at the boundary instead of silently misreading packed bytes.

Parameters:
- `frame` - The received frame-like object.

Example:

```ts
assertNeatRacingPackedFrameSchemaVersion(message.payload);
// now safe to read the channel views
```

### createNeatRacingFramePackPool

```ts
createNeatRacingFramePackPool(
  options: NeatRacingFramePackPoolOptions,
): NeatRacingFramePackPool
```

Creates a preallocated frame pack pool.

Educational note: the pool is the AC-042 unit — one `ArrayBuffer` per
slot allocated up front, partitioned into a rules span plus per-car SoA
channel spans at computed byte offsets. Every later `packFrame` is a
pure in-place fill: no view construction, no buffer churn, no per-frame
allocation in the worker's 60 Hz budget.

Parameters:
- `options` - Capacity and slot count; defaults documented on the type.

Returns: The pool handle (pack / unpack + diagnostics).

Example:

```ts
const pool = createNeatRacingFramePackPool({ carCapacity: 3 });
const packed = pool.packFrame({ fromTick: 0, toTick: 2, cars, rules });
worker.postMessage(
  { type: 'frames', payload: packed },
  resolveNeatRacingPackedFrameTransferList(packed),
);
const snapshot = pool.unpackFrame(packed);
```

### createPoolSlot

```ts
createPoolSlot(
  carCapacity: number,
): PoolSlot
```

Allocates one pool slot: a rules span at word 0 plus one span per channel
table entry, every view created exactly once.

Parameters:
- `carCapacity` - Fleet capacity to size the spans for.

Returns: The slot bundle (buffer + views + reusable frame object).

### decodeNeatRacingFeatureFlags

```ts
decodeNeatRacingFeatureFlags(
  bitfield: number,
): NeatRacingFeatureFlagState
```

Decodes a D8b bitfield into a feature-flag triple (masked to the table).

Parameters:
- `bitfield` - Arbitrary number; masked to the known three bits.

Returns: The decoded flag triple.

Example:

```ts
decodeNeatRacingFeatureFlags(0b011); // { tiresEnabled: true, radioEnabled: true, pitsEnabled: false }
```

### encodeNeatRacingFeatureFlags

```ts
encodeNeatRacingFeatureFlags(
  flags: NeatRacingFeatureFlagState,
): number
```

Encodes a feature-flag triple into the D8b bitfield.

Parameters:
- `flags` - The flag triple to encode (booleans; no coercion).

Returns: The bitfield (bit 0 tires, bit 1 radio, bit 2 pits).

Example:

```ts
encodeNeatRacingFeatureFlags({
  tiresEnabled: false,
  radioEnabled: true,
  pitsEnabled: false,
}); // 2 — the NeatRacing pinned default
```

### fillPackedFrameCarChannels

```ts
fillPackedFrameCarChannels(
  frame: NeatRacingPackedFrame,
  cars: readonly PhysicsCarState[],
): void
```

Fills every per-car SoA channel from the validated fleet, walking the
fixed channel table (zero allocation, deterministic float32 writes).

Parameters:
- `frame` - The slot frame whose channel views get filled.
- `cars` - The validated fleet snapshot (car 0 first).

### fillPackedFrameRulesChannel

```ts
fillPackedFrameRulesChannel(
  frame: NeatRacingPackedFrame,
  input: NeatRacingPackedFrameInput,
): void
```

Fills the continuous rules span of a packed frame.

Parameters:
- `frame` - The slot frame to mutate in place.
- `input` - The validated packing input.

### FramePackValidationError

Static validation error carrier for pack-side rejections.

### NEATRACING_DEFAULT_FEATURE_FLAG_BITFIELD

Numeric form of `NEATRACING_DEFAULT_FEATURE_FLAGS` under the D8b table.

### NEATRACING_DEFAULT_FEATURE_FLAGS

The feature flags NeatRacing pins by default: tires = 0 (no tire wear),
pits = 0 (no pit stops), radio = 1 (observation tails survive).

### NEATRACING_FEATURE_FLAG_BITS

The D8b feature-flag bit assignments, carried from the racing precedent:
bit 0 = `tiresEnabled`, bit 1 = `radioEnabled`, bit 2 = `pitsEnabled`.

### NEATRACING_FEATURE_FLAG_MASK

The full D8b bit mask covering all three recycled feature-flag bits.

### NEATRACING_FRAME_PACK_MAX_POOL_SLOTS

Largest legal pool slot count (bounded fan of preallocated slots).

### NEATRACING_FRAME_PACK_MIN_POOL_SLOTS

Smallest legal pool slot count (single reusable slot).

### NEATRACING_FRAME_PACK_POOL_SLOTS

Default double-buffering slot count for `createNeatRacingFramePackPool`.

### NEATRACING_PACK_NULL_LEVEL_SENTINEL

Numeric sentinel written for a null `nextRaceLevel` in the packed rules
header (a real level is a non-negative integer, so -1 can never collide).

### NEATRACING_PACK_SCHEMA_VERSION

Schema sentinel for forward-compatibility rejection on packed frames.

### NEATRACING_PACKED_CAR_CHANNEL_COUNT

SoA channel span count derived from the fixed channel table.

### NEATRACING_PACKED_CAR_CHANNELS

The SINGLE fixed channel mapping for the packed SoA layout.

Educational note: the channel list is the one source of truth for the
packed frame — the declaration order fixes the byte offsets, the field
names fix the wire shape, and each `reader` picks the physics field that
fills the channel. Packing, channel allocation, and finiteness validation
all walk this table so a new channel can never be added to the wire shape
without declaring its physics reader alongside it.

### NEATRACING_PACKED_MAX_CAR_CAPACITY

Largest legal packed fleet capacity.

Mirrors the sim worker's fleet bounds (`NEATRACING_SIM_WORKER_MAX_AGENTS`
in `./simulation-worker.ts`); duplicated as a local constant on purpose —
importing the worker module would drag its import-time `self.onmessage`
adapter into every packer consumer.

### NEATRACING_PACKED_MIN_CARS

Smallest legal packed fleet size (the single human car).

### NEATRACING_PACKED_RACE_STATUS_CODES

The recycled race-status code table: the fixed enum mapping packed into
the rules header (a plain number, exact across the boundary).

### NEATRACING_PACKED_RULES_FLOAT_COUNT

Float32 words the packed rules span occupies at the head of every slot.

### NEATRACING_PACKED_TRANSFER_BUFFER_COUNT

Buffers a packed frame contributes to a transfer list.

The slot layout packs every channel into ONE preallocated buffer, so the
resolved transfer list is always exactly a single entry — the AC-042
"single packed frame ArrayBuffer with offsets" story.

### NEATRACING_PACKED_WORD_BYTES

Byte size of one Float32 word in the packed slot layout.

### NeatRacingFeatureFlagState

A packed feature-flag triple under the recycled D8b bit table.

### NeatRacingFramePackPool

A preallocated pack pool owning one slot buffer per configured slot.

### NeatRacingFramePackPoolOptions

Options accepted by `createNeatRacingFramePackPool`.

### NeatRacingPackedFrame

A packed render frame on the wire (AC-025).

The discrete scalars ride as plain numbers (exact through structured
clone and transfer), while EVERY bulk channel is a `Float32Array` view
into the slot's single preallocated buffer at its computed byte offset.

### NeatRacingPackedFrameInput

The input a frame packing needs: the P2-S1-A frames payload + flags.

### NeatRacingPackedFrameRulesChannel

A rules span holding the continuous race-rules floats per packed frame.

### NeatRacingPackedFrameSnapshot

The unpacked plain-object reconstruction of one packed frame.

### PoolSlot

One preallocated pool slot: the slot buffer plus its once-created views
and the reused frame object that carries them on the wire.

### reconstructPackedFrameCar

```ts
reconstructPackedFrameCar(
  frame: NeatRacingPackedFrame,
  carIndex: number,
): PhysicsCarState
```

Reconstructs one physics car state from a packed frame's channels.

Parameters:
- `frame` - The packed frame to read (sentinel already checked).
- `carIndex` - Fleet index to reconstruct (below `agentCount`).

Returns: The reconstructed car (Float32-lossy physics fields).

### reconstructPackedFrameRules

```ts
reconstructPackedFrameRules(
  frame: NeatRacingPackedFrame,
): RaceRulesState
```

Reconstructs the race-rules state from a packed frame.

Parameters:
- `frame` - The packed frame to read (sentinel already checked).

Returns: The reconstructed rules (Float32-lossy continuous fields).

### resolveNeatRacingPackedChannelFirstFloatWord

```ts
resolveNeatRacingPackedChannelFirstFloatWord(
  channelIndex: number,
  carCapacity: number,
): number
```

First Float32 word of a SoA channel span (rules span comes first).

### resolveNeatRacingPackedFrameFloatWordCount

```ts
resolveNeatRacingPackedFrameFloatWordCount(
  carCapacity: number,
): number
```

Total Float32 words one packed slot buffer reserves for a car capacity.

### resolveNeatRacingPackedFrameTransferList

```ts
resolveNeatRacingPackedFrameTransferList(
  frame: NeatRacingPackedFrame,
): ArrayBuffer[]
```

Collects the distinct `ArrayBuffer`s backing a packed frame's channels
into a zero-copy `postMessage` transfer list.

With the single preallocated slot buffer every channel view shares, the
list is always exactly `NEATRACING_PACKED_TRANSFER_BUFFER_COUNT` entries;
the dedupe keeps the resolver correct even if a future layout splits the
channels across multiple slot buffers.

Parameters:
- `frame` - The packed frame whose buffers will be transferred.

Returns: Ordered, deduped `ArrayBuffer` list for `postMessage`.

Example:

```ts
worker.postMessage(
  { type: 'frames', payload: packedFrame },
  resolveNeatRacingPackedFrameTransferList(packedFrame),
);
```

### resolveNeatRacingPackedSlotByteLength

```ts
resolveNeatRacingPackedSlotByteLength(
  carCapacity: number,
): number
```

Total byte length of one packed slot buffer for a car capacity.

### validatePackedFrameCars

```ts
validatePackedFrameCars(
  cars: readonly PhysicsCarState[],
  carCapacity: number,
): FramePackValidationError | null
```

Validates the fleet part of a packing input against the packer bounds and
the channel table's finiteness contract.

Parameters:
- `cars` - The fleet snapshot to validate.
- `carCapacity` - The packer's preallocated fleet capacity.

Returns: A static rejection reason, or null when the fleet may pack.

### validatePackedFrameRules

```ts
validatePackedFrameRules(
  rules: RaceRulesState,
): FramePackValidationError | null
```

Validates the rules part of a packing input (status code membership and
finiteness of every packed field).

Parameters:
- `rules` - The race-rules snapshot to validate.

Returns: A static rejection reason, or null when the rules may pack.

### writePackedFrameHeader

```ts
writePackedFrameHeader(
  frame: NeatRacingPackedFrame,
  input: NeatRacingPackedFrameInput,
): void
```

Writes the frame header + discrete rules scalars of a packed frame.

Parameters:
- `frame` - The slot frame to mutate in place.
- `input` - The validated packing input.

## workers/simulation-worker/reproducibility.utils.ts

Reproducibility tuple utilities for the neatRacing simulation worker
(Phase 2, Step 01, slice P2-S1-C).

Educational note — why this file exists:
The spec's determinism contract (§5.3, §9.7) pins AC-026: two runs booted
from the SAME reproducibility tuple `(seed, agentCount, packSchemaVersion,
opponentSnapshot, trackId, featureFlags)` must produce byte-identical
frames across `NEATRACING_REPLAY_ADAPTATION_TICKS = 32` adaptation ticks
on the same runtime. The recycled runtime adaptation engine defaults to
`Math.random`, which would leave that contract numerically untestable, so
(D1) ALL sim-side randomness instead derives from the tuple seed through
the named §9.7.2 child-seed derivation table.

What this boundary owns:
- the §9.7.2 mix (`combineNeatRacingChildSeed`), mirroring the recycled
  track generator's combined-seed hash pattern (golden-ratio xor +
  FNV-style `Math.imul(..., 16777619)` folding, `>>> 0`);
- the seeded **xorshift32** stream every sim-side consumer must draw from
  (the `random:` source `createRuntimeAdaptationEngine` requires per §9.7.1);
- the five named derivation entries (`track`, `car-i network`,
  `adaptation-i`, `variant stream`, `rival-swap mutation stream`);
- the tuple's canonical serialized shape (static field order, sorted
  object keys, `null` for absent fields) plus a compact fingerprint;
- the packed-frame byte digest that formalizes the AC-026 parity claim.

Determinism rung (D1/D3, reproducibility-contracts language):
because every sim-side stream derives deterministically from the tuple
seed via the named table, the seed IS the RNG-state component of the
tuple — no separate RNG-state capture is needed at this boundary. The
parity claim is **same-runtime byte-exact** (Level 3 within the tuple
replay boundary); it is NOT cross-environment exact (floating-point
accumulation may differ across Node/browser runtimes). Visual-only
surfaces ride the separate `NEATRACING_VISUAL_SEED`, which sits OUTSIDE
the frame contract and is exempt from byte parity (§9.7.2 exemption).

### assertNonNegativeInteger

```ts
assertNonNegativeInteger(
  value: number,
  fieldName: string,
): void
```

Throws when a value is not an unsigned-safe non-negative integer.

Parameters:
- `value` - Candidate index/stride value.
- `fieldName` - Static field name for the rejection message.

### assertTupleNumericFields

```ts
assertTupleNumericFields(
  tuple: NeatRacingReproducibilityTuple,
): void
```

Validates the numeric identity fields of a tuple (seed finiteness and the
agentCount fleet bounds).

Reuses the frame packer's named fleet-range constants because the tuple
replays exactly the packed fleet size those constants bound.

Parameters:
- `tuple` - The tuple whose numeric fields validate.

### assertTupleStringFields

```ts
assertTupleStringFields(
  tuple: NeatRacingReproducibilityTuple,
): void
```

Validates the string identity fields of a tuple (both must be non-empty).

Parameters:
- `tuple` - The tuple whose string fields validate.

### canonicalizeReproducibilityJsonValue

```ts
canonicalizeReproducibilityJsonValue(
  value: unknown,
): unknown
```

Canonicalizes an arbitrary value into a JSON-safe, key-stable form.

Arrays stay in order (they carry positional semantics), plain objects
rebuild with sorted keys, and anything outside JSON material (functions,
symbols, BigInts, class instances) fails loudly instead of silently
collapsing the serialized shape.

Parameters:
- `value` - The value to canonicalize.

Returns: The canonical JSON-safe value.

### canonicalizeReproducibilityObject

```ts
canonicalizeReproducibilityObject(
  value: Record<string, unknown>,
): Record<string, unknown>
```

Canonicalizes one plain object into key-sorted order (undefined values
drop, matching JSON semantics), so reconstruction order can never change
the serialized shape.

Parameters:
- `value` - The plain object to canonicalize.

Returns: The canonical key-sorted object.

### canonicalizeReproducibilityScalar

```ts
canonicalizeReproducibilityScalar(
  value: unknown,
): unknown
```

Canonicalizes one JSON-safe scalar (`undefined`/`null` collapse to `null`,
booleans/strings pass, non-finite numbers fail loudly).

Parameters:
- `value` - The scalar to canonicalize.

Returns: The canonical scalar value.

### combineNeatRacingChildSeed

```ts
combineNeatRacingChildSeed(
  tupleSeed: number,
  surfaceTag: string,
): number
```

Combines the tuple seed and a named surface tag into one 32-bit child
seed (the §9.7.2 mix).

Mirrors the recycled track generator's combined-seed hash pattern: a
golden-ratio xor salts the tuple seed, then each tag character folds in
through `Math.imul(state ^ charCode, 16777619)` before the unsigned
`>>> 0` normalization. Same inputs always return the same unsigned value
on the same runtime (and on any IEEE 754 engine — integer-only math).

Parameters:
- `tupleSeed` - The AC-026 tuple seed (any finite number; normalized
through the 32-bit fold chain).
- `surfaceTag` - Named surface tag from the §9.7.2 table vocabulary.

Returns: The unsigned 32-bit child seed.

Example:

```ts
const childSeed = combineNeatRacingChildSeed(tuple.seed, 'adaptation-2');
const stream = createNeatRacingXorShift32Stream(childSeed);
```

### computeNeatRacingPackedFrameByteDigest

```ts
computeNeatRacingPackedFrameByteDigest(
  frame: NeatRacingPackedFrame,
): string
```

Computes the byte-parity witness digest of one packed frame (AC-026).

Two frames with EQUAL digests are treated as byte-identical by the parity
suite: the fold walks the pinned header scalars, then every channel's RAW
BYTES (Float32 storage bytes, not rounded values), so any byte-level
drift in the packed wire contract flips the digest. The digest is a
32-bit FNV-1a witness, not a cryptographic identity — a collision would
surface as a false PARITY, never a false mismatch. The schema sentinel is
guarded first so misread bytes fail loudly at the boundary.

Parameters:
- `frame` - The packed frame to digest.

Returns: An 8-character lowercase hex digest string.

Example:

```ts
const digestA = computeNeatRacingPackedFrameByteDigest(packFrameA);
const digestB = computeNeatRacingPackedFrameByteDigest(packFrameB);
expect(digestB).toBe(digestA); // AC-026 byte parity
```

### computeNeatRacingReproducibilityTupleFingerprint

```ts
computeNeatRacingReproducibilityTupleFingerprint(
  tuple: NeatRacingReproducibilityTuple,
): number
```

Folds the tuple's canonical serial into a compact unsigned 32-bit
fingerprint.

A convenience identity for logging/dedupe later slices; the canonical
STRING (not the fingerprint) stays the durable identity anchor. The fold
is the same FNV-1a family as every other digest here.

Parameters:
- `tuple` - The tuple to fingerprint.

Returns: The unsigned 32-bit fingerprint.

Example:

```ts
expect(computeNeatRacingReproducibilityTupleFingerprint(tupleA)).toBe(
  computeNeatRacingReproducibilityTupleFingerprint(tupleB),
); // same tuple → same fingerprint
```

### createNeatRacingXorShift32Stream

```ts
createNeatRacingXorShift32Stream(
  childSeed: number,
): NeatRacingXorShift32Stream
```

Creates a seeded xorshift32 stream (the D1 sim-side RNG source).

The stream is the ONLY permitted ambient-free random source sim-side
(§9.7.1): consumers pass it as `createRuntimeAdaptationEngine({ random })`
and draw car-network RNG through their derived seeds, so no sim-side code
path ever touches `Math.random`. The transition mirrors the recycled
track generator's xorshift32 (`<< 13`, `>>> 17`, `<< 5`); the zero-state
guard substitutes the named fallback so a zero seed stays deterministic.

Parameters:
- `childSeed` - Unsigned-safe child seed (a §9.7.2 derivation result).

Returns: A draw function returning floats in `[0, 1)`; identical seeds
produce identical sequences on the same runtime.

Example:

```ts
const random = createNeatRacingXorShift32Stream(
  deriveNeatRacingAdaptationChildSeed(tuple.seed, carIndex),
);
createRuntimeAdaptationEngine({ random });
```

### deriveNeatRacingAdaptationChildSeed

```ts
deriveNeatRacingAdaptationChildSeed(
  tupleSeed: number,
  carIndex: number,
): number
```

Derives the `adaptation-<carIndex>` child seed (§9.7.2 row 3).

Consumers build each car's runtime adaptation engine with an explicit
seeded stream from this seed (`createNeatRacingXorShift32Stream`), so the
adaptation-RNG path stays inside the replayed surface (D1 item 4).

Parameters:
- `tupleSeed` - The AC-026 tuple seed.
- `carIndex` - Fleet car index (non-negative integer).

Returns: The unsigned 32-bit child seed for the car's adaptation engine.

### deriveNeatRacingCarNetworkChildSeed

```ts
deriveNeatRacingCarNetworkChildSeed(
  tupleSeed: number,
  carIndex: number,
): number
```

Derives the `car-<carIndex>` network child seed (§9.7.2 row 2).

Consumers `setSeed()` every car network with this seed (the recycled
coevolution precedent), keeping network-internal draws inside the tuple.

Parameters:
- `tupleSeed` - The AC-026 tuple seed.
- `carIndex` - Fleet car index (non-negative integer).

Returns: The unsigned 32-bit child seed for the car's network.

### deriveNeatRacingRivalSwapChildSeed

```ts
deriveNeatRacingRivalSwapChildSeed(
  tupleSeed: number,
  eventIndex: number,
): number
```

Derives the `rival-swap` mutation-stream child seed for one swap event
(§9.7.2 row 5, E4).

Each swap event draws a disjoint mutation-stream slot (named
`NEATRACING_RIVAL_SWAP_SEED_STRIDE` spacing) and serializes its seed
into `opponentSnapshot`, so the reproducibility tuple replays swap
events exactly.

Parameters:
- `tupleSeed` - The AC-026 tuple seed.
- `eventIndex` - Swap-event index (non-negative integer).

Returns: The unsigned 32-bit seed for the event's mutation stream.

### deriveNeatRacingTrackChildSeed

```ts
deriveNeatRacingTrackChildSeed(
  tupleSeed: number,
): number
```

Derives the `track` surface child seed (§9.7.2 row 1).

Parameters:
- `tupleSeed` - The AC-026 tuple seed.

Returns: The unsigned 32-bit child seed for the track generator.

### deriveNeatRacingVariantStreamChildSeed

```ts
deriveNeatRacingVariantStreamChildSeed(
  tupleSeed: number,
  variantIndex: number,
  patchSize: number,
): number
```

Derives the `variant` stream child seed for one variant slot (§9.7.2 row 4).

Mirrors the recycled NGE composition (`base + variantIndex × stride`,
stride = `max(NGE_VARIANT_PATCH_SEED_OFFSET, max(1, patchSize) ×
NGE_VARIANT_PATCH_SEED_STRIDE_FACTOR)`); the eval-worker slice passes the
known patch size, while the default (`0`) falls back to the recycled
offset base.

Parameters:
- `tupleSeed` - The AC-026 tuple seed.
- `variantIndex` - Variant slot index (non-negative integer).
- `patchSize` - Patch size the slot evaluates (defaults to 0 when the
slot's patch size is not yet known).

Returns: The unsigned 32-bit seed for the variant's RNG stream.

### foldBytesIntoFnv1a

```ts
foldBytesIntoFnv1a(
  digest: number,
  view: ArrayBufferView<ArrayBufferLike>,
): number
```

Folds a typed array's raw bytes into an FNV-1a state.

Parameters:
- `digest` - Current fold state (unsigned 32-bit).
- `view` - Typed array whose bytes fold in, in order.

Returns: The advanced unsigned 32-bit fold state.

### foldPackedFrameChannelIntoDigest

```ts
foldPackedFrameChannelIntoDigest(
  digest: number,
  channelName: string,
  view: unknown,
): number
```

Validates one packed-frame channel view and folds its raw bytes into the
digest.

Parameters:
- `digest` - Current fold state.
- `channelName` - Static channel name for the rejection message.
- `view` - The channel view to walk.

Returns: The advanced fold state.

### foldPackedFrameChannelsIntoDigest

```ts
foldPackedFrameChannelsIntoDigest(
  digest: number,
  frame: NeatRacingPackedFrame,
): number
```

Folds every packed-frame channel view (rules span first, then the fixed
car SoA channel order) into the byte digest.

The view list mirrors the packer's fixed channel table declaration order;
a channel added to the wire shape must extend this walk in the same step.

Parameters:
- `digest` - Current fold state.
- `frame` - The packed frame whose channels fold in.

Returns: The advanced fold state.

### foldPackedFrameHeaderIntoDigest

```ts
foldPackedFrameHeaderIntoDigest(
  digest: number,
  frame: NeatRacingPackedFrame,
): number
```

Folds a packed frame's header scalars into the byte digest.

The header scalar order is PINNED (fromTick, toTick, agentCount,
featureFlags, level, lapsCompleted, raceStatusCode, nextRaceLevel) with
`;` separators so no concatenation ambiguity can shift fold boundaries.

Parameters:
- `digest` - Current fold state.
- `frame` - The packed frame whose header folds in.

Returns: The advanced fold state.

### foldTextIntoFnv1a

```ts
foldTextIntoFnv1a(
  digest: number,
  text: string,
): number
```

Folds a whole string (UTF-16 code units, in order) into an FNV-1a state.

Parameters:
- `digest` - Current fold state (unsigned 32-bit).
- `text` - The string to fold in, character by character.

Returns: The advanced unsigned 32-bit fold state.

### foldUint8IntoFnv1a

```ts
foldUint8IntoFnv1a(
  digest: number,
  value: number,
): number
```

Folds one unsigned char/byte value into an FNV-1a state.

Shared fold step for the child-seed mix (characters), the tuple
fingerprint (serialization characters), and the packed-frame digest
(channel bytes) — one hashing family across the boundary.

Parameters:
- `digest` - Current fold state (unsigned 32-bit).
- `value` - One character code or byte to fold in.

Returns: The advanced unsigned 32-bit fold state.

### isPlainJsonObject

```ts
isPlainJsonObject(
  value: unknown,
): boolean
```

Reports whether a value behaves as a plain JSON object (prototype is the
object prototype or null — built-ins like Date/Map fail).

Parameters:
- `value` - The candidate value.

Returns: True when the value is a plain object usable in JSON.

### NEATRACING_ADAPTATION_SURFACE_TAG_PREFIX

Prefix tag for the per-car adaptation seed: `adaptation-<carIndex>`.

### NEATRACING_BYTE_DIGEST_FNV_OFFSET_BASIS

FNV-1a offset basis starting every digest/fingerprint fold.

### NEATRACING_CAR_NETWORK_SURFACE_TAG_PREFIX

Prefix tag for the per-car network seed: `car-<carIndex>`.

### NEATRACING_CHILD_SEED_MIX_FNV_PRIME

FNV-style folding multiplier applied to every character/byte of a mixed
value, mirroring the recycled track generator's `Math.imul(..., 16777619)`
folding. Also drives the byte-digest and tuple fingerprint folds so one
hashing family covers the whole reproducibility boundary.

### NEATRACING_CHILD_SEED_MIX_GOLDEN_RATIO

Golden-ratio xor salt mixed into every child-seed derivation, mirroring
the recycled track generator's combined-seed pattern (`0x9e3779b9`).

### NEATRACING_REPLAY_ADAPTATION_TICKS

The byte-parity replay depth (TP-21, §9.7.1 item 4).

AC-026 asserts byte-identical frames across AT LEAST this many adaptation
ticks so the adaptation-engine RNG path stays inside the replayed surface.
At the present boundary each released fixed-step sim tick IS the
adaptation-tick unit; the adaptation cadence (Step 03) consumes the same
constant without redefining it.

### NEATRACING_RIVAL_SWAP_SEED_STRIDE

Per-event stride for the rival-swap mutation stream (§9.7.2, E4).

No recycled constant pins this stride; the value mirrors the NGE
variant base-slot spacing convention so every swap event owns a disjoint
mutation-stream slot. The rival propagation operator serializes the
swap-event seeds into `opponentSnapshot`, keeping them inside the tuple
contract.

### NEATRACING_RIVAL_SWAP_SURFACE_TAG

Tag for the shared rival-swap mutation stream base.

### NEATRACING_TRACK_SURFACE_TAG

Named §9.7.2 surface tags (the derivation vocabulary, single table).

Every stochastic surface's tag lives here so a tag rename is one edit
and the derivation table stays contract-pinned.

### NEATRACING_VARIANT_PATCH_SEED_OFFSET

Recycled NGE per-variant seed offset (D2, `NEATRACING_VARIANT_PATCH_SEED_OFFSET`
base in `src/neat/nge-juvenile`): the minimum per-variant stream spacing.

### NEATRACING_VARIANT_PATCH_SEED_STRIDE_FACTOR

Recycled NGE per-variant seed stride factor (D2): each variant patch slot
widens the stride by this factor.

### NEATRACING_VARIANT_STREAM_SURFACE_TAG

Tag for the shared variant stream base.

### NEATRACING_VISUAL_SEED

The visual-only seed base (TP-20, §9.7.2 exemption).

Starfield layers, pulses, and scenery placement derive from this base.
They never touch sim state, so they sit OUTSIDE the frame contract and
are exempt from AC-026 byte parity — the frame channels are the parity
surface, not the visual-channel seeds.

### NEATRACING_XORSHIFT32_ZERO_SEED_FALLBACK

Zero-state escape hatch for the xorshift32 stream.

Xorshift32 cannot advance from a zero state, so this non-zero constant
(the recycled track generator's zero-guard value) keeps a zero child seed
a valid, deterministic stream start instead of a dead stream.

### NeatRacingReproducibilityTuple

The AC-026 reproducibility tuple carried by the sim-worker init payload.

The field order of the interface mirrors the AC-026 wording; the canonical
serialized shape is pinned by `serializeNeatRacingReproducibilityTuple`.
`featureFlags` uses the worker payload's carried shape (an open boolean
record) and stays verbatim; `opponentSnapshot` is opaque at this boundary
(the NGE network-controller slice gives it its shape).

### NeatRacingXorShift32Stream

```ts
NeatRacingXorShift32Stream(): number
```

A seeded xorshift32 draw function returning floats in `[0, 1)`.

### serializeNeatRacingReproducibilityTuple

```ts
serializeNeatRacingReproducibilityTuple(
  tuple: NeatRacingReproducibilityTuple,
): string
```

Serializes the tuple into its canonical string shape (the replay
identity anchor).

The top-level field order is PINNED to the AC-026 ordering
(`seed, agentCount, packSchemaVersion, opponentSnapshot, trackId,
featureFlags`), nested objects rebuild key-sorted (insertion order can
never change the serial), and absent fields serialize as explicit `null`
so the shape is unambiguous at the boundary.

Parameters:
- `tuple` - The tuple to serialize.

Returns: The canonical JSON serial of the tuple.

Example:

```ts
const canonical = serializeNeatRacingReproducibilityTuple(tuple);
expect(canonical === serializeNeatRacingReproducibilityTuple(tuple)).toBe(true);
```

### throwNotJsonSerializable

```ts
throwNotJsonSerializable(
  value: unknown,
): never
```

Throws when a value is not JSON-serializable material for the tuple.

Parameters:
- `value` - The rejected value (for the static message surface).
