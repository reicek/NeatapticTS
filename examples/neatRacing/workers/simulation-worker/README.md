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
- the agent-car control path: the INJECTED NGE control seam
  (`./nge.fleet.control.ts` — observe -> activate -> steer through the
  LATEST adopted per-car network). The LEGACY SCRIPTED LANE is RETIRED
  (AC-029 — no dual path, no deferred cleanup): an agent car with NO
  injected controller has NO driver, and the fleet tick-input resolver
  raises the NAMED no-controller error instead of falling back to a
  scripted line,
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
  The payload additionally carries every rival-swap event released during
  that frame's ticks as an OPTIONAL `rivalSwaps` list (P4-S2, AC-032 —
  the live rival winner propagation); frames with NO rival crossing omit
  the field entirely, so untouched frames keep their exact prior shape.
- the NAMED no-controller error is the ONE error that converts into a
  protocol reply (M1, slice `P4-S2-rival-propagation`): a frame resolved
  for a session booted WITHOUT an injected controller replies one
  `{ type: 'error', payload: { message } }` carrying the SAME message
  text the resolver raises (`NEATRACING_SIM_WORKER_NO_CONTROLLER_ERROR`
  plus the ` (race mode: <mode>)` suffix) instead of escaping
  `handleRequest` as an uncaught error. Every OTHER worker error keeps
  its current behavior.
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

## workers/simulation-worker/nge.fleet.control.ts

Per-car NGE fleet-control seam for the neatRacing sim worker (Phase 4,
Step 23, slice `P4-S1-worker-seam`).

Educational note — what this module owns:
The spec §9.2 two-worker split keeps GROWTH (topology mutation, variant
evaluation, score windows, commit/rollback bookkeeping) inside the eval
worker and keeps INFERENCE inside the sim worker. This module is the sim
side of that split: it turns the live fleet state into ONE physics control
input per agent car, per tick, through the shipped observe -> activate ->
steer chain:

1. OBSERVE — the car's live `PhysicsCarState` is projected onto the
   observation seam state (`ai/observation.assembler.ts`), whose level
   ladder fixes the vector width (82 / 89 / 110 channels).
2. ACTIVATE — the level-shaped observation vector is pushed through the
   activate-only network seam (`ai/nge.controller.ts`), the single owner
   of the activation -> control law (output 0 -> throttle, output 1 ->
   steer, brake fixed at 0, NO turn-rate channel).
3. STEER — the produced control is published to the sim worker.

THE ADOPTED-SNAPSHOT LAW (AC-029): the seam NEVER caches the adopted
network. Every tick asks the relay for the CURRENTLY adopted per-car
network; when the relay hands back a DIFFERENT network object, the seam
seeds it from the §9.7.2 child-seed table (`car-<i>`) and rebuilds its
controller over it. The sim lane therefore follows the LATEST adopted
snapshot, and a snapshot adopted mid-race reaches the cars on the next
tick — a cached snapshot would strand the cars on a stale brain.

THE BORROWED-STREAM LAW (the seam does NOT own the adopted network):
the adopted network belongs to the EVAL side, so the seam BORROWS its
random stream instead of taking ownership of it. On adoption the seam
remembers the state the network carried, seeds it from the child-seed
table and hands the borrowed state straight back; on every tick the seam
restores the CAR's own stream state before the forward pass, re-captures
the advanced state afterwards, and returns the network to the state it was
found in. Two contractual consequences follow:
- every car draws from its OWN stream, even when the relay publishes ONE
  network instance for several cars (the per-car stream law), and
- the adopted network is left as it was found between ticks, so the sim
  lane never silently re-seeds the eval worker's brain.
The relay PRECONDITION follows from the same law: the published network
instance belongs to ONE car (a per-car-EXCLUSIVE instance). A network
without a deterministic stream (a pure activation function) simply skips
the stream handling — it has no stream to borrow.

THE DETERMINISM RUNG: the seam promises Level 2 (ordered deterministic) —
under a FROZEN adopted snapshot, the same tuple seed, the same fleet
inputs and the same per-car evaluation order produce byte-stable
controls. It does NOT promise Level 3 (replay exact) across an ADOPTION
SEQUENCE: the sequence of adopted snapshots is not captured in the AC-026
tuple, so a mid-race adoption is a replay boundary.

THE ZERO-ALLOCATION LAW (P9): the seam owns NO per-tick garbage. Its
per-car working set (the observation state, the roster entry and the
published control buffer) is allocated ONCE per car and rewritten in
place, so the control object the seam publishes is the SAME instance on
every tick. The published buffer is therefore REUSED: the caller copies
it before holding it (the sim-worker boundary publishes a fresh physics
input per car per tick, so a resolved tick's inputs stay observable after
the next tick resolves). The UPSTREAM shipped assembler and controller
allocate their own observation vector and control object per call — that
cost belongs to those modules, and it is recorded here rather than hidden.

### adoptNeatRacingFleetCarController

```ts
adoptNeatRacingFleetCarController(
  carCaches: Map<number, NeatRacingFleetCarCache>,
  carIndex: number,
  adopted: NeatRacingFleetCarController,
  tupleSeed: number,
  level: NeatRacingObservationLevel,
): NeatRacingFleetCarCache
```

Adopts a network for one fleet car: seeds it from the §9.7.2 child-seed
table and builds the controller over it.

Educational note: the child seed is `mix(tupleSeed, 'car-<i>')`, so every
car's network draws from its OWN stream inside the reproducibility tuple
(AC-026) — the fleet never shares one stream.

The seeding is a BORROW: the adopted network belongs to the eval side, so
the seam remembers the state the network carried, seeds it, captures the
car's own stream state, and hands the borrowed state straight back.

Parameters:
- `carCaches` - The seam's per-car cache.
- `carIndex` - The fleet car index.
- `adopted` - The newly adopted network.
- `tupleSeed` - The AC-026 tuple seed.
- `level` - The observation ladder level.

Returns: The fresh cache entry for the car (the reused observation state
and control buffer of a previous adoption are carried over).

### clampNeatRacingFleetSteerAxis

```ts
clampNeatRacingFleetSteerAxis(
  value: number,
): number
```

Clamps the steer axis into its physics domain `[-1, 1]`.

Parameters:
- `value` - The produced steer value.

Returns: The clamped steer (0 for a non-finite value).

### clampNeatRacingFleetThrottleAxis

```ts
clampNeatRacingFleetThrottleAxis(
  value: number,
): number
```

Clamps the throttle axis into its physics domain `[0, 1]`.

Parameters:
- `value` - The produced throttle value.

Returns: The clamped throttle (0 for a non-finite value).

### createNeatRacingFleetControl

```ts
createNeatRacingFleetControl(
  options: NeatRacingFleetControlOptions,
): NeatRacingFleetControlSeam
```

Creates the per-car NGE fleet-control seam.

Educational note: the seam is a small state machine over the adopted
snapshot. Its per-car cache holds the adopted network, the controller
built over it, the reused observation state and the published control
buffer. A tick rebuilds the controller ONLY when the relay hands back a
different network object (a fresh adoption), and it reuses everything
else — that is where the zero-per-tick-allocation law comes from.

Parameters:
- `options` - The tuple seed, the frozen track geometry, the relay and
the observation level.

Returns: The fleet-control seam the sim worker consumes.

Example:

```ts
const seam = createNeatRacingFleetControl({
  tupleSeed: 42,
  trackSpec,
  relay: { readCarController: (carIndex) => adopted.get(carIndex) ?? null },
});
const control = seam.computeControl(cars, 1);
```

### createNeatRacingFleetObservationState

```ts
createNeatRacingFleetObservationState(): MutableNeatRacingObservationState
```

Creates the empty (reused) observation state of one fleet car.

Returns: The fresh observation state.

### createNeatRacingFleetRosterEntry

```ts
createNeatRacingFleetRosterEntry(): MutableNeatRacingObservationCarState
```

Creates one empty roster entry.

Returns: The fresh roster entry.

### mapNeatRacingFleetControl

```ts
mapNeatRacingFleetControl(
  produced: PhysicsStepInput | null | undefined,
): PhysicsStepInput | null
```

Maps one produced control onto the physics control input the sim lane
publishes.

Educational note — the mapping law (AC-029): the two NGE output channels
map IDENTITY-wise onto the control axes (channel 0 -> throttle, channel 1
-> steer — the shipped `NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT = 2` law),
the throttle axis is clamped into `[0, 1]`, the steer axis into
`[-1, 1]`, the brake axis is fixed at zero, and NO turn-rate channel is
ever emitted (the steer axis is the sole turn channel, AC-093).

The mapping returns a FRESH physics input: a resolved tick's inputs must
stay observable after the next tick resolves (the adopted-snapshot
round-trip law).

Parameters:
- `produced` - The control the controller produced (null/undefined
when there is nothing to map).

Returns: The mapped physics control input, or null when there is nothing
to map.

### MutableNeatRacingObservationCarState

Mutable mirror of one roster entry (the seam rewrites it).

### MutableNeatRacingObservationState

Mutable mirror of the observation seam state (the seam rewrites it).

### NEATRACING_FLEET_CONTROL_BRAKE

The brake channel the NGE fleet-control seam always publishes: ZERO.

Educational note: the shipped NGE controller law (`ai/nge.controller.ts`)
maps the network's output channels onto the physics control axes and never
commands a brake — the sim lane slows down through the throttle axis only.

### NeatRacingFleetCarCache

The per-car working set of the fleet-control seam.

### NeatRacingFleetCarController

The per-car network surface the fleet-control seam drives.

Educational note: the seam stays inference-only (it never mutates the
network's STRUCTURE), but it DOES borrow the network's internal random
stream before the first forward pass, so the network's own random draws
stay inside the AC-026 reproducibility tuple. `setSeed` is OPTIONAL: a
network that carries no internal stream (a pure activation function)
simply skips the seeding step. The stream accessors are OPTIONAL too: a
network that exposes them lets the seam borrow AND return the stream
(the borrowed-stream law), so the adopted network is left exactly as it
was found.

### NeatRacingFleetControllerRelay

The relay that publishes the CURRENTLY adopted per-car network.

Educational note: the relay is the seam's ONLY window on the eval worker's
adopted snapshot. It is read once per car per tick (never cached), so the
sim lane follows the latest adoption without any snapshot-copy traffic
across the worker boundary.

The relay carries a PRECONDITION: the published network instance belongs
to ONE car (a per-car-EXCLUSIVE instance). The seam tolerates a shared
instance — the per-car stream law keeps every car on its own stream — but
the eval side keeps its bookkeeping per car, so an exclusive instance is
the intended shape.

### NeatRacingFleetControlOptions

The construction options of the per-car fleet-control seam.

### NeatRacingFleetControlSeam

The public fleet-control seam the sim worker consumes.

Educational note: the published control is a REUSED per-car buffer (the
zero-allocation law), so a consumer that HOLDS the control across ticks
must copy it first — the sim-worker boundary does exactly that.

### refreshNeatRacingFleetRoster

```ts
refreshNeatRacingFleetRoster(
  roster: MutableNeatRacingObservationCarState[],
  fleet: readonly PhysicsCarState[],
  teamLayout: readonly (number | undefined)[],
): void
```

Refreshes the reused roster from the live fleet.

Educational note: the roster drives the rival and teammate tails of the
observation vector; the array index IS the car index, so the focal car is
excluded from those tails by its position (the recycled roster contract).

Parameters:
- `roster` - The reused roster array.
- `fleet` - The live fleet states entering this tick.
- `teamLayout` - The level's team layout.

### resolveNeatRacingFleetTeamIndex

```ts
resolveNeatRacingFleetTeamIndex(
  teamLayout: readonly (number | undefined)[],
  carIndex: number,
): 0 | 1
```

Resolves the team index of one fleet car from the level's team layout.

Parameters:
- `teamLayout` - The level's team layout.
- `carIndex` - The fleet car index.

Returns: The team index (0 = team A, 1 = team B).

### writeNeatRacingFleetObservationState

```ts
writeNeatRacingFleetObservationState(
  observation: MutableNeatRacingObservationState,
  car: PhysicsCarState,
  carIndex: number,
  teamIndex: 0 | 1,
  roster: readonly NeatRacingObservationCarState[],
): void
```

Writes the live car kinematics into the reused observation state.

Educational note — the unit bridge (LD-15): `PhysicsCarState.yawRate` is
radians per 60 Hz tick, so the observation seam receives radians per
second (`yawRate / NEATRACING_FIXED_TIMESTEP_SECONDS`). The lateral speed
is the physics state's own lateral velocity (positive to the left of the
heading — the observation seam's convention), so the two sides can never
disagree about which way the car slides.

Parameters:
- `observation` - The reused observation state.
- `car` - The focal car state.
- `carIndex` - The focal car's fleet index.
- `teamIndex` - The focal car's team index.
- `roster` - The refreshed roster.

## workers/simulation-worker/simulation-worker.ts

### buildSimWorkerFramesPayload

```ts
buildSimWorkerFramesPayload(
  fromTick: number,
  sessionState: SimWorkerSessionState,
  rivalSwaps: readonly NeatRacingRivalSwapPayload[],
): NeatRacingSimWorkerFramesPayload
```

Builds the structured `frames` reply for one paced frame.

Parameters:
- `fromTick` - Total-tick counter before the frame's released ticks.
- `sessionState` - Running session bag (post-tick state).
- `rivalSwaps` - The frame's rival-swap events (empty when no rival
crossed; an empty list OMITS the optional payload field so untouched
frames keep their exact prior shape).

Returns: The reply payload snapshot (fresh copies, no aliasing into later
tick output).

### createSimWorkerRivalPropagationState

```ts
createSimWorkerRivalPropagationState(
  payload: NeatRacingSimWorkerInitPayload,
): SimWorkerRivalPropagationState | undefined
```

Resolves the worker-plane rival propagation state from one init payload
(P4-S2, AC-032).

Defensive resolution (the init `opponentSnapshot` is opaque at this
boundary): the level is read from the snapshot only when it is an integer
naming a pinned ladder rung (`NEATRACING_TEAM_LAYOUTS`), and the rivals
are that rung's non-zero team slots within the booted fleet size. A
snapshot that names no rivals (the solo rung, a missing snapshot, or a
malformed one) leaves the lane UNDEFINED — the 1-3-car baseline boots
never construct a network and never fire a swap.

Parameters:
- `payload` - Validated init payload.

Returns: The rival-propagation state, or undefined when the boot has no
rival lane.

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

### dispatchSimWorkerNetworkRequest

```ts
dispatchSimWorkerNetworkRequest(
  state: SimWorkerSessionState,
  request: { type: "network"; carIndex: number; },
): NeatRacingSimWorkerResponse
```

Answers ONE panel-network request with the elected car's network envelope
(P4-S2C, Step 24C — the docked-right panel network transport).

Educational note — THE OBSERVATIONAL COPY: the request is a READ. The
dispatcher never touches the RNG, never advances the simulation and never
mutates the rival ledger — it copies the stored rival mind's bytes into a
fresh hex string, so two identical requests answer byte-identically while
the ledger keeps its own bytes (the copy law).

Educational note — THE M1 REPLY LAW: every guard below answers with an
ERROR REPLY; no rejection ever escapes `handleRequest` as a throw (the
same shape the frame case uses for its own guards).

Parameters:
- `state` - Internal session bag (read-only on this path).
- `request` - The panel-network request (the carrier car index).

Returns: The single reply: the panel-network envelope or an error reply.

### dispatchSimWorkerRequest

```ts
dispatchSimWorkerRequest(
  state: SimWorkerSessionState,
  request: NeatRacingSimWorkerRequest,
): NeatRacingSimWorkerResponse[]
```

Routes one validated sim-worker request to its response list (the
`handleRequest` dispatch). Pure over the session bag apart from the
mutation the routed request performs.

Educational note — WHY the dispatch is a standalone function: `handleRequest`
wraps this call in the M1 selective catch (the NAMED no-controller error is
the ONE error that converts into a protocol `{ type: 'error' }` reply), and
the routing switch needs a provably-returning shape for that wrapper to own
the whole dispatch.

Parameters:
- `state` - Internal session bag (mutated by the routed request).
- `request` - Host request message (non-junk; type-routed below).

Returns: The ordered responses to post for this request.

### driveSimWorkerFrame

```ts
driveSimWorkerFrame(
  sessionState: SimWorkerSessionState,
  timestampMs: number,
  framePayload: NeatRacingSimWorkerFramePayload,
): readonly NeatRacingRivalSwapPayload[]
```

Releases 0-4 fixed steps for one paced frame and mutates the running session.

Parameters:
- `sessionState` - Running session bag mutated in place.
- `timestampMs` - rAF-style frame timestamp driving the accumulator.
- `framePayload` - The paced frame payload (its optional D5 sample is
latched BEFORE ticking so zero-step frames still refresh the held manual
sample, matching the phase-1 sampling-outside-the-step-guard behavior).

Returns: The rival-swap events released during this frame's ticks (the
shared empty list when no rival crossed; never a live tick-internal
array alias).

### encodeSimWorkerPanelWeightsHex

```ts
encodeSimWorkerPanelWeightsHex(
  weights: Float32Array<ArrayBufferLike>,
): string
```

Encodes one serialized weight plane as a byte-exact hex string (P4-S2C,
Step 24C — the panel-network envelope).

Educational note — THE OWNER-LOCAL MIRROR: the `promotion.rules.ts` and
`rival.propagation.ts` encoders are private to THEIR payloads; this
payload owns its own serialization path (the one serialization path per
payload law — never a shared dual path). Lowercase hex, two characters
per byte, no separators — the exact byte pattern the `Float32Array` holds,
never a `toJSON`.

Parameters:
- `weights` - The serialized weight plane to encode.

Returns: The hex string of the array's raw bytes in NATIVE byte order
(same-runtime byte parity only, never cross-environment exact).

### isSimWorkerFleetControlSeam

```ts
isSimWorkerFleetControlSeam(
  candidate: unknown,
): boolean
```

Checks whether a candidate value is a usable NGE control seam.

Educational note: the seam is a LIVE object, so the init payload carries a
shape check instead of a data check. A callable proxy over a function is
accepted as well as a plain object — the contract tests inject exactly
that shape.

Parameters:
- `candidate` - The candidate seam value.

Returns: True when the candidate exposes a callable `computeControl`.

### materializeSimWorkerRivalSpawnMind

```ts
materializeSimWorkerRivalSpawnMind(
  seed: number,
  level: NeatRacingObservationLevel,
  carIndex: number,
): Float32Array<ArrayBufferLike>
```

Materializes one rival's SPAWN mind as the flat serialized plane (P4-S2,
AC-032).

Educational note — the DETERMINISTIC RECONSTRUCTION precedent (the
promotion slice and the adaptation session both build agent-car minds
this way): the worker never reaches a live evolved network through the
seam — the controller seam holds the adopted networks behind
`computeControl` — so the worker reconstructs the rival's spawn mind from
the tuple: `new Network(level channel count, controller output count,
{ seed: deriveNeatRacingCarNetworkChildSeed(seed, carIndex) })`, the SAME
construction `createNeatRacingAdaptationCarStates` uses for the same car.
The instance is serialized positionally and discarded: only the flat
SerializedWeights plane enters the ledger, and a later replacement entry
swaps the ledger's bytes, not an instance.

Parameters:
- `seed` - The AC-026 tuple seed the session booted with.
- `level` - The ladder rung (selects the observation channel count).
- `carIndex` - The rival's fleet car index (the per-car child seed).

Returns: The rival's serialized spawn mind.

### NEATRACING_PANEL_NETWORK_ENVELOPE_KIND

The self-describing envelope kind the panel-network reply carries (P4-S2C,
Step 24C — the docked-right panel network transport).

Educational note — WHY A NAMED KIND: the reply travels over the SAME
outbox as the frame/ready/error messages, so the consumer must be able to
tell a panel-network envelope from any other payload WITHOUT guessing the
payload's shape. The discriminator is a named constant on the worker's own
surface (the one serialization path per payload law — the host codec
compares against this exact constant, never against a bare literal).

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

### NEATRACING_SIM_WORKER_NO_CONTROLLER_ERROR

The NAMED no-controller error (AC-029 — the retired scripted lane).

Educational note: the LEGACY SCRIPTED LANE is retired, so an agent car
without the injected NGE control seam has NO driver. The message NAMES the
missing controller (the contract pins the REASON, not one exact wording)
instead of falling back to a scripted drive.

Educational note — how to MATCH it: the THROWN message is this reason text
plus a diagnostic suffix (` (race mode: <mode>)`), so a consumer asserts
with `message.includes(NEATRACING_SIM_WORKER_NO_CONTROLLER_ERROR)` — an
equality check against the constant alone does NOT match. The suffix stays
because the resolver's race-mode parameter exists only to name the mode
here, and the resolver arity is pinned by the byte-frozen RED homes.

### NEATRACING_SIM_WORKER_SPAWN_ALONG_LEAD_M

Along-track lead distance (meters past segment 0's START point) where the
first fleet car spawns, matching the phase-1 host spawn recipe exactly
(segment-0 chord heading, 2 m along it) so the AC-026 baseline run keeps
the same opening pose the host used.

### NEATRACING_SIM_WORKER_SPAWN_GRID_PITCH_M

Line-astern grid pitch (meters) between consecutive fleet spawn slots.

### NEATRACING_SIM_WORKER_SPAWN_PATH_ANCHOR_EPS_M

Anchor tolerance (meters) under which the spline's first sample counts as
sitting exactly on segment 0's start point. The generated layouts publish
their first spline sample AT the segment-0 control point (the generator's
interpolation factor is zero there), while the hand-authored fixture rings
hang their sample list elsewhere - the tolerance separates the two without
any heuristic geometry.

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
owned geometry (segments for the surface resolver and lap detector).
The spline samples ride along for the REPLAY tuple, but the worker no
longer follows a scripted racing line (AC-029 — the legacy scripted lane
is retired).

### NeatRacingSimWorkerPanelNetworkReplyPayload

The panel-network reply payload (P4-S2C, Step 24C — the docked-right panel
network transport).

Educational note — THE CARRIER LAW: the envelope names the car it carries
(the elected rival), the two network dimensions of that car's rung (the
observation channel count and the controller output count) and the car's
serialized connection weights in the established hex law (lowercase, two
characters per byte, no separators, native byte order). The worker NEVER
mutates the ledger to answer the request: the reply is an observational
COPY of the stored rival mind.

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

Neutral control sample replayed for car 0 in BOTH race modes before the
host has sent any sample. Frozen: it is a constant, never a live
accumulator.

### resolveSimWorkerFleetSpawnPoses

```ts
resolveSimWorkerFleetSpawnPoses(
  trackSpec: TrackSpec,
  agentCount: number,
): PhysicsCarSpawnPose[]
```

Resolves the deterministic fleet spawn poses from the owned track.

Educational note: car 0 inherits the phase-1 host recipe exactly (a 2 m
lead past segment 0's start on the start straight), while cars behind it
line up astern at fixed grid pitch so the fleet start stays a function of
the init config alone (no RNG anywhere). The formal child-seed system is
the byte-parity slice's job (P2-S1-C); this slice derives per-car seeds by
the documented index offset when it builds the fleet. THE PATH REFINEMENT
(the P4-S2 live rival lane): on the generated layouts the grid is laid on
the ROAD-ANCHORED spline (the road the autopilot reads) instead of the
segment-0 control chord the spline diverges from; layouts whose spline
does not anchor at the segment-0 start (the hand-authored fixture rings)
keep the phase-1 chord walk verbatim.

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
  controller: NeatRacingFleetControlSeam | undefined,
): PhysicsStepInput[]
```

Resolves the per-tick control inputs for the whole fleet.

Educational note — the ONE live lane (AC-029, the retired scripted lane):
- car 0 is the HUMAN car in BOTH race modes (LD-12 lane isolation): it
  replays the latest host sample — or the neutral sample before the first
  one — and it is NEVER handed to a network.
- every AGENT car (index >= 1) runs the INJECTED NGE control seam
  (observe -> activate -> steer through the LATEST adopted network). An
  agent car with no adopted network YET gets the neutral control sample,
  never a scripted drive.
- there is NO scripted fallback lane: an agent car whose session carries
  no injected controller raises the NAMED no-controller error instead of
  driving a scripted line.

Every published input is a FRESH object: a resolved tick's inputs stay
observable after the next tick resolves (the adopted-snapshot round-trip
law), so the seam's reused control buffer is copied at this boundary.

Parameters:
- `cars` - Fleet states entering this tick.
- `raceMode` - Session race mode (the retired scripted lane made the
mode irrelevant to the control decision — car 0 is the human car in both
modes — so the slot only names the mode in the no-controller error).
- `humanSample` - Latest host sample (may be undefined pre-first sample).
- `splineSamples` - The RETIRED scripted lane's racing line (the
positional slot the five-argument call sites still fill; the worker
never reads it).
- `controller` - The injected NGE control seam driving the agent cars
(undefined only when the session booted without one).

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

### resolveSimWorkerRulesSeedLevel

```ts
resolveSimWorkerRulesSeedLevel(
  payload: NeatRacingSimWorkerInitPayload,
): number
```

Resolves the level the worker rules machine seeds from: the round's
carried boot level.

Educational note — the carried level (fix packet
`P4-S2A-debug-rung-seam-green` iteration 2): the host copies the round's
OWN boot rung into the init payload's `opponentSnapshot`
(`{ level, carCount }`), so the rules state machine — not just the fleet
ladder and the rival-propagation lane — seeds at the SAME rung. The
browser's mode-switch re-boot now carries a `?level=` debug rung across
the switch, and a solo lvl-1 rules state beside a rival-bearing lvl-2
fleet would desync the HUD read model from the round's honest layout.

Defensive resolution (the init `opponentSnapshot` is opaque at this
boundary, mirroring the rival-propagation lane's read): the level is
read only when it is an integer naming an in-ladder rung
(1..NEATRACING_HUMAN_MAX_LEVEL); a snapshot that names no rung (null, a
non-object, a malformed level or an out-of-band integer) keeps the
byte-identical prior default (NEATRACING_HUMAN_START_LEVEL — the worker
started every race at lvl 1 before the rung carried across the switch).

Parameters:
- `payload` - Validated init payload.

Returns: The rung the rules machine seeds from.

### resolveSimWorkerSplineSpawnGrid

```ts
resolveSimWorkerSplineSpawnGrid(
  path: { samples: readonly SplineSample[]; loopLengthM: number; },
  agentCount: number,
): PhysicsCarSpawnPose[]
```

Lays the line-astern grid along one road-anchored spawn path.

Educational note: the lead and the pitch stay EXACTLY the phase-1 law's
values (2 m lead, 12 m astern pitch) - only the PATH they are measured
along changes, from the control chord to the road the spline publishes.
Each pose reads its span chord's direction as the heading, so the spawn
heading IS the road direction the autopilot's own lookahead reads, and the
in-span interpolation keeps the pose on the lane-center curve between
samples. The walk wraps the closed loop, so an oversized agent count still
resolves deterministic poses (the init cap of 32 keeps the grid far below
one lap on the served layouts anyway).

Parameters:
- `path` - The road-anchored spawn path (samples + loop length).
- `agentCount` - Fleet size (>= 1).

Returns: One spawn pose per fleet slot, ordered car 0 first.

### resolveSimWorkerSplineSpawnPath

```ts
resolveSimWorkerSplineSpawnPath(
  trackSpec: TrackSpec,
): { samples: readonly SplineSample[]; loopLengthM: number; } | null
```

Resolves the road-anchored spawn path of one track, or `null` when the
spline does not anchor at the segment-0 start point.

THE LAW (the P4-S2 live rival lane): the generated layout's spline IS the
road every controller reads - the racing-line table, the observation
assembler and the shipped autopilot all walk it - and its FIRST sample
carries an interpolation factor of ZERO, so it sits exactly on segment 0's
control point while the samples continue along the road. A fleet grid laid
on the SEGMENT-0 CHORD instead drifts OFF that road as the chord diverges
from the lane-center curve (the chord is the control straight, the spline
is the sampled road), spawning cars where the shipped autopilot's own
throttle gate never opens: the aim error at pose start already exceeds the
full-throttle error while the car stands still, so it can neither
accelerate nor turn (a standing car never rotates). The spawn path
therefore walks the spline when the spline is road-anchored, and keeps the
phase-1 chord walk when it is not (the hand-authored fixture rings, whose
pinned poses the contract holds verbatim).

Parameters:
- `trackSpec` - Owned track geometry.

Returns: The ordered spline samples plus the closed-loop length, or `null`
when the spline does not anchor at segment 0's start (or degenerates).

### serializeSimWorkerRivalMindWeights

```ts
serializeSimWorkerRivalMindWeights(
  network: default,
): Float32Array<ArrayBufferLike>
```

Serializes one network's connection weights positionally — the recycled
`CarGenome.serialize()` plane: `network.connections[i].weight` in
connection order, mirroring the promotion slice's private
`serializeNetworkWeightsToFloat32Array` walk (fresh array; the caller
owns its bytes).

Parameters:
- `network` - The spawned network to serialize.

Returns: The positional connection-weight array.

### SimWorkerRivalPropagationState

The worker-plane rival propagation state (P4-S2, AC-032 — the LIVE
rival winner propagation).

Educational note — WHY THE FLAT SERIALIZED PLANE: the worker's drive path
never owns a `Network` instance per car (the controller seam holds the
adopted networks), so the session keeps the rival MINDS as the recycled
`CarGenome.serialize()` Float32Arrays — the exact flat serialized plane
the propagation operator computes on. Each materialized rival mind is the
spawn-mind reconstruction the adaptation plane itself uses
(`createNeatRacingAdaptationCarStates`): a network of the level's
observation channel count and controller output count, seeded from the
per-car child seed, serialized in positional connection order.

### SimWorkerSessionState

Internal mutable session bag: all sim truth owned by the worker lives here.

### SimWorkerValidationError

Validation failure carries a static rejection reason string.

### stepSimWorkerRivalPropagationTick

```ts
stepSimWorkerRivalPropagationTick(
  sessionState: SimWorkerSessionState,
  swaps: NeatRacingRivalSwapPayload[],
): void
```

Runs ONE tick of the worker-plane rival propagation lane (P4-S2, AC-032).

Each rival's lap-line detector sees its ENTERING pose (the same port the
car-0 detector runs: segment-0 chord projection, lateral gate, along
deadband, first-sample latch, sign-flip crossing), and a crossing rival
resolves ONE swap event through the operator — the crossing rival's
serialized ledger mind becomes the propagation base, every OTHER rival's
ledger entry is replaced by its bounded seeded nudge, and the event's
tuple record is collected for the frame's `frames` reply.

Educational note — the lane's place in the tick: the detectors run in the
SAME tick position as car 0's (the pose entering the tick, before fleet
physics), but rival crossings feed ONLY this lane — `stepRaceRules` keeps
car 0's detector as its sole input, so the main agent carries its own
lap state and is independent of rival success (AC-032). The operator
REFUSES a mid-lap call, so the lane passes `atLapLineBoundary: true` only
at a detected line crossing.

Parameters:
- `sessionState` - Running session bag (mutates the rival ledger, the
event counter and the per-rival detector baselines in place).
- `swaps` - The frame-local event list being collected (events pushed
in resolved order: rivals ascending per tick, ticks in session order).

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
