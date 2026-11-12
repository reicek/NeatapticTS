# ai

NeatRacing AI-mode goal state machine (slice P2-S4-A, AC-033).

THE LAW (spec section 9.4, LD-10 / LD-12): AI mode runs a MINIMUM of
`NEATRACING_AI_MIN_SESSION_LAPS` laps and then CONTINUES until the
current level's growth goal is met — that is the three-part promotion
predicate of `promotion.rules.ts` (the promotion thresholds) firing at a
forward crossing. Level 3 continues until the user stops the session.
Switching mode in EITHER direction restarts the race: the selector's
notify contract (`../host/mode-selector.ts`) IS the restart hook, and a
fresh AI goal session boots at the first rung of the ladder.

(1) THE LAP FLOOR IS A PRECONDITION, NOT A FOURTH CONDITION (OQ-002):
    floor membership is read through the SHARED predicate's eligibility
    gate — this module NEVER re-declares the floor constant, the
    threshold table, or the conditions: `promotion.rules.ts` owns them
    and this machine only asks it.

(2) COUNT-THEN-EVALUATE: the crossing that completes a lap increments
    `sessionLaps` BEFORE the predicate reads it, so the floor flips
    EXACTLY at the crossing that completes lap 20 (the promotion.rules
    GREEN fixture pins the promotion AT `sessionLaps` 20). The
    increment is the HOST's completed-circuit fact — the same
    `lapsSinceLastCrossing` field the promotion predicate's condition
    (a) reads — floored and clamped at 0 and NEVER a bare +1, so the
    OQ-002 floor counts COMPLETED laps.

(3) ONE EVALUATION PATH: every FORWARD finish-line crossing routes
    through `evaluateNeatRacingPromotion`. Non-forward crossings never
    count a lap and never evaluate — the predicate is evaluated at
    FORWARD crossings only.

(4) LEVEL 3 IS UNBOUNDED (LD-10): at the cap, crossings still count
    laps and the predicate verdict still runs (it reports `ladderCapped`
    and never promotes); the machine has NO terminal state — stopping is
    a HOST action (the user stops the session or switches mode).

(5) PROMOTION IS A ROUND BOUNDARY (E6 / D4): a promoted crossing
    re-bases the goal session to the next rung with `sessionLaps` 0, so
    the lap floor RE-APPLIES per round. The D4 carrier state (seed
    network, innovation counter, adapter reset) belongs to the
    `promotion.rules.ts` round-tuple primitives consumed by the host
    wiring (P2-S4-C), NOT to this bookkeeping machine.

(6) NON-FINITE MEASUREMENTS ARE NORMALIZED AT THE MACHINE BOUNDARY:
    `+Infinity` would PASS the shared predicate's comparisons (a
    `+Infinity` node count is `> threshold`, a `+Infinity` circuit
    count is `>= required`, a `+Infinity` lap pace is `> 0`), so the
    machine normalizes every measurement to finite-or-NaN FIRST: a
    non-finite measurement can never promote, while every forward
    crossing still routes through the ONE shared predicate.

(7) THE REALM: pure, headless and deterministic — no RNG, no clock
    reads, no DOM, no worker globals. The machine never mutates its
    inputs; the counted and promoted outcomes carry a FRESH session
    record.

## ai/ai-mode.goals.ts

### assertAiModeGoalLapCount

```ts
assertAiModeGoalLapCount(
  sessionLaps: unknown,
): void
```

Guards the lap counter of a session record.

WHY: the counter is the OQ-002 floor counter — a negative or fractional
value would silently misreport the remaining lap budget.

Parameters:
- `sessionLaps` - The candidate lap count.

### assertAiModeGoalLevel

```ts
assertAiModeGoalLevel(
  level: unknown,
): void
```

Guards the ladder rung of a session record.

WHY: the rung must be a real ladder rung. The TypeScript type already
narrows it, but a JavaScript caller can pass anything, so the runtime
check keeps out-of-band rungs loud.

Parameters:
- `level` - The candidate rung.

### assertAiModeGoalSession

```ts
assertAiModeGoalSession(
  session: unknown,
): void
```

Guards a session record handed in from the outside.

WHY: the goal machine is a pure realm, but a host can still hand it a
corrupted record. Rejecting a malformed session at the boundary keeps
the state machine total: every outcome it produces is derived from a
well-formed session.

Parameters:
- `session` - The candidate session record.

### createNeatRacingAiModeGoalSession

```ts
createNeatRacingAiModeGoalSession(
  level: NeatRacingObservationLevel,
): NeatRacingAiModeGoalSession
```

Creates a fresh AI-mode goal session on a ladder rung.

WHY: a fresh AI goal session boots at the first rung with an empty lap
counter — the lap floor is owed in full. The boot level is validated
here so that out-of-band rungs fail loudly at the boundary instead of
corrupting a round later on.

Parameters:
- `level` - The ladder rung the session starts on.

Returns: A fresh session record with `sessionLaps` 0.

Example:

```ts
const session = createNeatRacingAiModeGoalSession(NEATRACING_AI_MODE_START_LEVEL);
// { level: 1, sessionLaps: 0 }
```

### describeNeatRacingAiModeGoalPhase

```ts
describeNeatRacingAiModeGoalPhase(
  session: NeatRacingAiModeGoalSession,
): NeatRacingAiModeGoalPhase
```

Describes the phase the goal session is currently in.

WHY: the phase answers "what does this session still owe?" — it reads
the SHARED floor constant (`NEATRACING_AI_MIN_SESSION_LAPS`) and the
SHARED cap (`NEATRACING_PROMOTION_MAX_LEVEL`) instead of re-declaring
them, so the machine and the predicate can never disagree about the
floor.

Parameters:
- `session` - The session record to describe.

Returns: The current phase: `lapFloor` while the minimum lap budget is
still owed, `chasingGoal` once the floor is met and a rung above
exists, `unboundedCap` at the ladder top.

Example:

```ts
describeNeatRacingAiModeGoalPhase({ level: 1, sessionLaps: 0 });
// { kind: 'lapFloor', remainingLaps: 20 }
describeNeatRacingAiModeGoalPhase({ level: 1, sessionLaps: 20 });
// { kind: 'chasingGoal', promotionTargetLevel: 2 }
describeNeatRacingAiModeGoalPhase({ level: 3, sessionLaps: 20 });
// { kind: 'unboundedCap' }
```

### finiteOrNaN

```ts
finiteOrNaN(
  value: number,
): number
```

Normalizes a host measurement onto the shared predicate's own rejection
path.

WHY: every check a NaN touches inside `promotion.rules.ts` (`>`,
`>=`, `<=`) evaluates false, so a normalized measurement can never
pass its check. Only NaN, -Infinity and +Infinity are normalized —
every finite value passes through verbatim, so the machine re-declares
no threshold and loses no precision.

Parameters:
- `value` - The host measurement to normalize.

Returns: The measurement itself when it is finite, NaN otherwise.

### NEATRACING_AI_MODE_START_LEVEL

The ladder rung a fresh AI-mode goal session boots at (AC-033).

WHY: the AI lane's goal ladder starts at the FIRST rung — the same
numeric rung as the human lane's `NEATRACING_HUMAN_START_LEVEL` in
`../physics/race-rules.ts`, but deliberately NOT imported: the AI
lane's constants live here and the human lane's never leave
`race-rules.ts`.

Example:

```ts
const session = createNeatRacingAiModeGoalSession(NEATRACING_AI_MODE_START_LEVEL);
// { level: 1, sessionLaps: 0 }
```

### NeatRacingAiModeGoalCrossing

The host-owned facts about one finish-line crossing.

WHY: the goal machine does not detect crossings — the host's line
detector does. The machine only consumes the verdict plus the two
measurements the shared predicate needs, so the same lap counter and
the same drivability sample feed the goal decision and the promotion
decision.

Example:

```ts
const outcome = recordNeatRacingAiModeGoalCrossing(session, {
  forwardCrossing: true,
  lapsSinceLastCrossing: 1,
  nodeCount: 301,
  drivability: { meanForwardSpeed: 25, scriptedBaselineLapPace: 30, oscillationMetric: 0.2, offTrackFraction: 0.05 },
});
```

### NeatRacingAiModeGoalCrossingOutcome

The result of feeding one crossing to the goal machine.

WHY: the outcome is a discriminated union so a host can branch on the
KIND without inspecting the evaluation: `ignored` (nothing happened),
`crossingWithoutLap` (a forward crossing that completed no circuit —
the round is untouched), `lapCounted` (the lap was banked, the round
continues) and `promoted` (the round ended, the next rung begins).

Example:

```ts
if (outcome.kind === 'promoted') {
  session = outcome.session; // re-based on the next rung
}
```

### NeatRacingAiModeGoalCrossingWithoutLapOutcome

The outcome of a FORWARD crossing that completed NO full circuit.

WHY: the host reported `lapsSinceLastCrossing` below one lap (or a
non-finite value), so the round is untouched — NO lap is banked — but
the shared predicate still ran (ONE evaluation path): condition (a)
reports its failure honestly, and such a crossing can never promote
without a completed circuit.

Example:

```ts
recordNeatRacingAiModeGoalCrossing(session, { ...crossing, lapsSinceLastCrossing: 0 });
// { kind: 'crossingWithoutLap', session }
```

### NeatRacingAiModeGoalIgnoredOutcome

The outcome of a crossing that the machine ignores.

WHY: a non-forward crossing is not a lap and not a promotion trigger,
so the round is returned untouched and NO evaluation was run (the
shared predicate is a FORWARD-crossing evaluator).

Example:

```ts
recordNeatRacingAiModeGoalCrossing(session, { ...crossing, forwardCrossing: false });
// { kind: 'ignored', session }
```

### NeatRacingAiModeGoalLapCountedOutcome

The outcome of a forward crossing that banked a lap without promoting.

WHY: the lap is counted and the shared predicate still ran (and its
verdict is reported), but the round continues on the SAME rung — the
evaluation explains WHY the growth goal is still unmet.

Example:

```ts
outcome.evaluation.eligible; // true — the lap floor is met
outcome.evaluation.promoted; // false — the threshold was not crossed
```

### NeatRacingAiModeGoalPhase

The phase the goal session is currently in — what the session still owes
the user.

WHY: the phase is the human- and HUD-readable summary of the session.
It never re-derives the promotion conditions: it only reports whether
the shared floor has been met and whether a rung above still exists.

The three kinds:
- `lapFloor` — the session still owes the minimum lap budget
  (`remainingLaps > 0`); no promotion is even reachable yet.
- `chasingGoal` — the floor is met and a promotion rung exists: the
  session continues until the three-part predicate holds at a forward
  crossing.
- `unboundedCap` — the ladder top: the session keeps evolving until the
  user stops it or switches mode (no promotion target exists).

Example:

```ts
describeNeatRacingAiModeGoalPhase({ level: 2, sessionLaps: 20 });
// { kind: 'chasingGoal', promotionTargetLevel: 3 }
```

### NeatRacingAiModeGoalPromotedOutcome

The outcome of a forward crossing that promoted the session.

WHY: a promotion ends the round: the session is re-based on the next
rung with `sessionLaps` 0 (the lap floor re-applies per round) and the
evaluation carries the promotion verdict the host needs to swap the
round tuple.

Example:

```ts
outcome.session; // { level: 2, sessionLaps: 0 }
```

### NeatRacingAiModeGoalSession

The AI-mode goal session record — the whole state the machine owns.

WHY: the session is the ONLY thing the goal state machine carries. It
is a plain, immutable snapshot: every transition produces a fresh
record instead of mutating the previous one, so a host can keep the old
record for a replay or an audit trail.

Example:

```ts
const session = createNeatRacingAiModeGoalSession(NEATRACING_AI_MODE_START_LEVEL);
const phase = describeNeatRacingAiModeGoalPhase(session);
// { kind: 'lapFloor', remainingLaps: 20 }
```

### normalizeNeatRacingAiModeGoalDrivability

```ts
normalizeNeatRacingAiModeGoalDrivability(
  measurement: NeatRacingDrivabilityMeasurement,
): NeatRacingDrivabilityMeasurement
```

Normalizes a drivability measurement onto the shared predicate's own
rejection path.

WHY: the shared gate compares each of the four fields (`>`, `>=`,
`<=`), and a NaN field always fails its own comparison. Normalizing
the whole measurement at the machine boundary leaves the shared gate's
arithmetic untouched while making a non-finite sample unable to
promote.

Parameters:
- `measurement` - The host's drivability measurement.

Returns: The measurement with every field normalized to finite-or-NaN.

### recordNeatRacingAiModeGoalCrossing

```ts
recordNeatRacingAiModeGoalCrossing(
  session: NeatRacingAiModeGoalSession,
  crossing: NeatRacingAiModeGoalCrossing,
): NeatRacingAiModeGoalCrossingOutcome
```

Feeds one finish-line crossing to the goal state machine.

WHY: this is the single decision point of the AI-mode goal. A
non-forward crossing is ignored outright (no lap, no evaluation — the
shared predicate is a FORWARD-crossing evaluator). A forward crossing
COUNTS the host's completed-circuit fact FIRST and only then asks the
shared predicate, so the eligibility floor flips exactly at the
crossing that completes lap 20.

COUNT-THEN-EVALUATE: the banked increment is the HOST's
completed-circuit fact — `lapsSinceLastCrossing` floored and clamped
at 0 — never a bare +1. It is the SAME fact the shared predicate's
condition (a) consumes, so the machine's floor counter and the
predicate can never disagree: a forward crossing that completed no
circuit banks NO lap (the `crossingWithoutLap` outcome) and the
predicate reports condition (a) as failed.

The session record is NEVER mutated: the `ignored` and
`crossingWithoutLap` outcomes return the INPUT session reference (the
round is untouched), the counted outcome spreads the input record into
a fresh one, and the promoted outcome re-bases onto the next rung with
`sessionLaps` 0 (the lap floor re-applies per round). The D4 carrier
state (seed network, innovation counter, adapter reset) is NOT part of
this bookkeeping — the host consumes the `promotion.rules.ts` round-tuple
primitives for it (P2-S4-C).

The crossing measurements are NORMALIZED before the shared predicate
reads them: the machine runs every measurement through
`finiteOrNaN`, so an Infinity-laced measurement can never pass a
comparison it would otherwise pass (`+Infinity > threshold`,
`+Infinity >= 1` and `+Infinity >= required` were all true inside the
shared predicate). The normalization re-declares no threshold and
keeps the single evaluation path.

Parameters:
- `session` - The session record to advance.
- `crossing` - The host-owned crossing facts.

Returns: The outcome: `ignored` for a non-forward crossing,
`crossingWithoutLap` for a forward crossing that completed no
circuit, `lapCounted` when the round continues, `promoted` when the
round ended and the next rung began.

Example:

```ts
const outcome = recordNeatRacingAiModeGoalCrossing(session, crossing);
if (outcome.kind === 'promoted') {
  session = outcome.session; // re-based on the next rung
}
```

### resolveLadderRung

```ts
resolveLadderRung(
  level: number,
): NeatRacingObservationLevel
```

Narrows a level number onto the ladder's own rungs (1, 2 or 3).

WHY: the shared promotion evaluation reports its levels as plain
numbers, while the session record carries the ladder's rung type. This
narrowing keeps the module cast-free while still refusing an
out-of-band rung — a corrupted evaluation can never re-base a round
onto an unknown rung.

Parameters:
- `level` - The level number to narrow.

Returns: The ladder rung the level names.

## ai/nge.controller.ts

NeatRacing NGE controller — the recycled racing-curriculum network-controller
wrapper (spec P2-S3-A) migrated realm-free onto the NeatRacing observation
ladder and physics control contract.

THE E11 OWNERSHIP (spec §9.7): this module implements the SIM-side per-tick
control path of the two-worker split — observe -> activate -> steer — while
growth, variant evaluation, score windows, exhaustion state, oscillation
state, and commit/rollback bookkeeping stay with the separate eval-worker
adaptation engine (`./runtime.adaptation.ts`). The controller is
inference-only: it never mutates physics state, never draws ambient
randomness (D1: the network's own seeded stream drives any stochastic
forward pass), and never reads the adaptation engine's state.

## Recycled contract

The module is the realm-free migration of the racing-curriculum
`controller/nge.controller.ts`: the same activate-only network seam, the
same tier-guide evidence channels (optimal-line lateral offset at scalar
index 16, heading error at index 17), and the same degenerate single-car
self-radio seam. The NeatRacing deltas are the observation source and the
control sink:

- Observation: `assembleNeatRacingObservationVector` (the 82/89/110 channel
  ladder, AC-027) replaces the recycled 63/77-wide assembler. The recycled
  Tier-2 self-radio mapping becomes the ladder's self-radio tail, carried
  on levels 2 and 3.
- Control: the output is a NeatRacing `PhysicsStepInput`
  (`throttle`, `brake`, `steer`). THE STEER AXIS IS THE SOLE TURN CHANNEL
  (AC-093): no `turnRatePercent` is ever emitted. The throttle axis is
  clamped to its physics domain `[0, 1]` — negative network output commands
  zero thrust because reverse thrust is not an axis of this game — while
  the steer axis keeps the recycled `[-1, 1]` domain.

## Background reading

- NEAT and topology-evolving neuroevolution:
  K. O. Stanley and R. Miikkulainen, "Evolving Neural Networks through
  Augmenting Topologies," *Evolutionary Computation*, vol. 10, no. 2,
  pp. 99-127, 2002.
  [NEAT publications](https://nn.cs.utexas.edu/?neat-papers)

### clampNeatRacingControlValue

```ts
clampNeatRacingControlValue(
  value: number,
): number
```

Clamps a raw controller output into the `[-1, 1]` domain. Recycled
`clampControlValue` verbatim.

Parameters:
- `value` - Raw controller output.

Returns: Value clamped to `[-1, 1]`.

### clampNeatRacingSteerAxis

```ts
clampNeatRacingSteerAxis(
  value: number,
): number
```

Clamps the network's second output onto the physics steer axis in
`[-1, 1]` (positive steers counterclockwise). The steer axis is the sole
turn channel (AC-093), so no `turnRatePercent` is ever emitted.

Parameters:
- `value` - Raw network steer output.

Returns: Steer command clamped to `[-1, 1]`.

### clampNeatRacingThrottleAxis

```ts
clampNeatRacingThrottleAxis(
  value: number,
): number
```

Clamps the network's first output onto the physics throttle axis.

The `PhysicsStepInput.throttle` axis is defined in `[0, 1]` (positive
forward thrust), while NGE networks emit signed outputs. The clamp floors
the negative half at 0 — reverse thrust is not an axis of this game — and
keeps the recycled [-1, 1] saturation behavior on the positive half.

Parameters:
- `value` - Raw network throttle output.

Returns: Throttle command clamped to `[0, 1]`.

### createNeatRacingNgeController

```ts
createNeatRacingNgeController(
  network: NeatRacingControllerNetwork,
  options: NeatRacingNgeControllerOptions,
): NeatRacingNgeController
```

Creates the NeatRacing NGE controller wrapper around the public
`activate(...)` inference surface.

Per tick the controller assembles the level-shaped observation vector, runs
one public forward pass, maps the first two outputs onto the physics control
axes (throttle in `[0, 1]`, brake fixed at 0, steer in `[-1, 1]`, and NO
`turnRatePercent` — the steer axis is the sole turn channel, AC-093), and —
on radio-augmented levels — feeds the degenerate self-radio seam. The
recycled write-back order is preserved: the seam is written with the derived
self-monitoring payload BEFORE assembly, and with the network's own excess
outputs AFTER activation, so an externally supplied seam can stream both.

Parameters:
- `network` - Public network object that exposes `activate(...)`.
- `options` - Optional observation level and radio-channel overrides.

Returns: Stateful controller that maps network outputs to engine commands.

Example:

```ts
const controller = createNeatRacingNgeController(network, { level: 2 });
const { control, evidence } = controller.computeControlWithEvidence(
  { carX: 0, carY: 0, carHeading: 0 },
  trackSpec,
);
physics.step(control);
```

### createNeatRacingSingleCarRadioChannel

```ts
createNeatRacingSingleCarRadioChannel(
  radioDim: number,
): NeatRacingSingleCarRadioChannel
```

Creates the single-car radio seam used by radio-augmented controller levels.
Recycled `createSingleCarRadioChannel` verbatim: a fixed-width zero buffer
that pads and truncates through `writeSelf` and exposes `readSelf`.

Parameters:
- `radioDim` - Number of channels retained in the radio buffer.

Returns: Read/write self-monitoring radio channel.

### NEATRACING_NGE_CONTROLLER_OUTPUT_COUNT

Number of controller output channels mapped to the control axes.
Recycled value verbatim (`CONTROLLER_OUTPUT_COUNT`): the network emits
`[throttle, steer]` (plus the self-radio write tail on radio-augmented
levels).

### NEATRACING_NGE_DEFAULT_OBSERVATION_LEVEL

Default observation level for the NGE controller seam — the recycled solo
NGE driver default tier (`DEFAULT_CONTROLLER_TIER`).

### NEATRACING_NGE_GUIDANCE_ALPHA_TIER_ONE

Guidance overlay alpha for the solo NGE level: a faint optimal-line hint
while the network learns to drive. Recycled value verbatim
(`TIER_ONE_GUIDANCE_ALPHA`). Radio-augmented levels turn the hint off
entirely (alpha 0) so the network self-navigates.

### NEATRACING_NGE_OBSERVATION_INDEX_OPTIMAL_LINE_HEADING_ERROR

Observation channel index of the normalized optimal-line heading error.
Recycled value verbatim; see
{@link NEATRACING_NGE_OBSERVATION_INDEX_OPTIMAL_LINE_LATERAL_OFFSET}.

### NEATRACING_NGE_OBSERVATION_INDEX_OPTIMAL_LINE_LATERAL_OFFSET

Observation channel index of the normalized optimal-line lateral offset.
Recycled value verbatim; the NeatRacing assembler keeps the recycled
20-scalar base block order, so the index is level-independent.

### NEATRACING_NGE_RADIO_DIMENSION

Default self-radio channel count for the degenerate single-car radio seam.
Owner-local single source: the recycled radio width (7) is the assembler's
self-radio tail width, so the controller derives it instead of restating.

### NeatRacingControllerNetwork

Public network surface required by the NeatRacing NGE controller seam.
Recycled owner-local contract verbatim: the controller is inference-only and
accepts any object exposing `activate(...)`, so the NGE network stays free
to grow structure behind the seam.

### NeatRacingNgeController

Public controller surface consumed by the sim-worker per-tick wiring.

### NeatRacingNgeControllerOptions

Runtime options for the NeatRacing NGE controller seam. All fields are
optional. Level 1 is the default; the self-radio seam is exercised on
ladder levels 2 and 3 (the ladder's self-radio tail).

### NeatRacingNgeControllerTickEvidence

Tier-guide evidence sampled from one controller tick. Recycled structure
verbatim; the signals are read from the assembled observation vector's
optimal-line scalar channels.

### NeatRacingNgeControllerTickResult

Controller tick result including control and adaptation evidence.

### NeatRacingSingleCarRadioChannel

Mutable single-car self-radio seam (recycled `SingleCarRadioChannel`).

### normalizeNeatRacingControllerOutputs

```ts
normalizeNeatRacingControllerOutputs(
  controllerOutputs: number | readonly number[],
): readonly number[]
```

Normalizes raw network outputs into a flat, finite array. Recycled
`normalizeControllerOutputs` mapping: scalar shorthand becomes a two-channel
vector and non-finite entries collapse to 0 so downstream slicing stays safe.

Parameters:
- `controllerOutputs` - Raw controller network output.

Returns: Finite controller output vector.

### prepareRadioSeamState

```ts
prepareRadioSeamState(
  envState: NeatRacingObservationState,
  radioChannel: NeatRacingSingleCarRadioChannel,
): NeatRacingObservationState
```

Enriches the observation state with the current self-radio field before
radio-augmented assembly. Recycled `prepareObservationState` mapping: the
seam is first written with the derived self-monitoring payload, then the
buffer is published through the state's `radioField` tail.

Parameters:
- `envState` - Focal car observation seam state.
- `radioChannel` - Single-car self-monitoring seam.

Returns: Observation-ready seam state carrying the radio field.

### resolveNeatRacingNgeGuidanceAlpha

```ts
resolveNeatRacingNgeGuidanceAlpha(
  level: NeatRacingObservationLevel,
): number
```

Resolves the optimal-line overlay alpha for each NGE controller level.

The recycled `resolveGuidanceAlphaForTier` mapping, expressed on the
NeatRacing ladder: the solo NGE level keeps a faint hint, and the
radio-augmented levels turn the hint off entirely so the network
self-navigates once the self-radio seam is live. The scripted-baseline
full-overlay case (recycled alpha 1.0) belongs to the scripted controller
domain, not to this NGE seam.

Parameters:
- `level` - Observation ladder level driving the vector width.

Returns: Overlay alpha in `[0, 1]`.

Example:

```ts
resolveNeatRacingNgeGuidanceAlpha(1); // 0.35 — faint hint on the solo level
resolveNeatRacingNgeGuidanceAlpha(2); // 0 — self-navigation on radio levels
```

### resolveNeatRacingSelfMonitoringPayload

```ts
resolveNeatRacingSelfMonitoringPayload(
  envState: NeatRacingObservationState,
): Float32Array<ArrayBufferLike>
```

Builds the seven-channel self-monitoring payload consumed by the self-radio
seam. Recycled payload verbatim, with the recycled scales sourced from the
assembler's owner-local constants: forward speed (108), lateral speed (54),
total speed (108), yaw rate (1), slip angle (pi/2), centered progress, and
the raw optimal-line lateral offset (18).

Parameters:
- `envState` - Focal car observation seam state.

Returns: Ordered seven-channel self-monitoring payload.

### resolveTickEvidence

```ts
resolveTickEvidence(
  observationVector: Float32Array<ArrayBufferLike>,
): NeatRacingNgeControllerTickEvidence
```

Extracts the optimal-line guidance evidence channels from one assembled
observation vector. Recycled `resolveTickEvidence` verbatim: the lateral
error is the absolute normalized offset (channel 16), the heading error is
the absolute normalized angular error (channel 17), and the combined
guidance-need signal is the max of the two.

Parameters:
- `observationVector` - Active assembled observation vector.

Returns: Lateral error, heading alignment, and combined guidance-need signal.

## ai/promotion.rules.ts

NeatRacing AI promotion rules (slice `P2-S3-C`, AC-031).

THE LAW (spec section 9.4, LD-10 / LD-30, E1 / E6): promotion is NOT
gameable — a big node count alone (or one lucky crossing) never promotes.
The promotion rule is a THREE-PART PURE PREDICATE; ALL THREE conditions
must hold AT THE CROSSING TICK:

- (a) FORWARD FINISH-LINE CROSSING AFTER ONE COMPLETE CIRCUIT — the car
  crosses the start/finish line in the forward direction having completed at
  least one full lap since its last crossing (no back-and-forth over the
  line, no crossing on the first pass);
- (b) NODE COUNT ABOVE THE LEVEL'S THRESHOLD AT THE CROSSING TICK —
  `nodeCount > threshold(currentLevel)` (TP-07: 300 for the lvl 1 -> 2
  band, 600 for the lvl 2 -> 3 band);
- (c) DRIVABILITY GATE OVER THE CROSSING LAP — measured over the lap that
  JUST completed: the mean forward speed against the PINNED scripted-baseline
  lap pace, the TP-09 oscillation metric and the off-track fraction.

THE LAP FLOOR IS A PRECONDITION, NOT A FOURTH CONDITION (OQ-002): the
predicate is evaluated at forward finish-line crossings ONLY once
`sessionLaps >= NEATRACING_AI_MIN_SESSION_LAPS = 20`. The predicate itself
stays three-part; the floor only decides WHETHER the predicate runs at all.

THE NODE-THRESHOLD BAND IS LEVEL-RELATIVE AND CLIMBS ONE RUNG: the
threshold is the CURRENT level's promotion threshold — the very threshold
the oscillation law engages at (`resolveOscillationPromotionNodeThreshold`,
owned by `oscillation-penalty.utils.ts`) — so a promotion re-bases the floor
(300 -> 600) exactly as the oscillation law expects. A session therefore
climbs the ladder one rung per promotion (lvl 1 -> lvl 2 -> lvl 3) and the
ladder CAPS at level 3 (`NEATRACING_PROMOTION_MAX_LEVEL`), where no further
rung exists. The thresholds are NOT re-declared here (no dual path): this
module IMPORTS them.

THE INTENTIONAL BOUNDARY ASYMMETRY (spec section 9.5 — do NOT "fix"): the
oscillation law engages at `nodeCount >= threshold` while promotion fires
at `nodeCount > threshold`. The STRICT `>` is required here so the penalty
is already ramping at the promotion instant (no step change in the score).

THE PINNED BASELINE (N3): the drivability gate's speed reference is the
per-course profile-augmented scripted controller (AC-028's throttle profile
table) run on the SAME track/seed in a SCRIPTED-INPUT PROBE EPISODE — no
human input, no evolving agents. The probe's mean lap speed is a pure
deterministic function of the track tuple, so the gate is reproducible.

THE ROUND BOUNDARY (E6, D4): a promotion is a ROUND BOUNDARY, never a
mid-session event. The crossing network becomes the SEED NETWORK of the
next round; the next round starts from a FRESH reproducibility tuple (new
`agentCount`, new `opponentSnapshot` matching the new level's car count);
the adaptation-engine state RESETS; and the innovation counter CARRIES with
the seed network. NO MID-SESSION CAR MATERIALIZATION: the cars of the new
level exist from the FIRST tick of the new round.

THE TWO EXPLICIT D4 FIELDS: the fresh round tuple carries `seedNetwork`
(the crossing network's `CarGenome.serialize()` payload as a TYPED ARRAY
plus its NGE envelope — NOT `toJSON`, because typed-array serialization
avoids key-order sensitivity) and `carriedInnovationCounter` (the counter
value carried from the prior round). With those two named fields the round
N+1 session is REPLAYABLE FROM ITS OWN TUPLE ALONE.

THE REALM: pure, headless and deterministic — no RNG, no wall-clock reads,
no DOM, no worker globals. The probe episode is the ONE place that drives
the real physics, and it drives it deterministically (a fixed spawn seed, a
scripted input, no random source), so the same track tuple always yields
the same baseline pace.

### assertCarriedInnovationCounter

```ts
assertCarriedInnovationCounter(
  carriedInnovationCounter: number,
): void
```

Validates the carried innovation counter (D4) of a fresh round.

The counter is a top-level numeric identity field of the round tuple's
replay anchor, exactly parallel to the base tuple's `seed` (which the AC-026
serializer validates as finite): a non-finite, non-integer or negative
counter is rejected at the round boundary instead of being silently copied
into a `JSON.stringify` that would collapse it to `null` — leaving the
canonical serial and the restored round state disagreeing over the very
counter the D4 field exists to carry.

Parameters:
- `carriedInnovationCounter` - The counter to validate.

### buildNeatRacingNextRoundTuple

```ts
buildNeatRacingNextRoundTuple(
  options: NeatRacingNextRoundTupleOptions,
): NeatRacingPromotionRoundTuple
```

Builds the FRESH round tuple of the next round (E6, D4).

The fresh tuple carries the prior round's seed, track and pack-schema
identity, a NEW `agentCount` and a NEW `opponentSnapshot` matching the new
level's car count, plus the two explicit D4 fields: the `seedNetwork` (the
crossing network's typed-array payload and its NGE envelope) and the
`carriedInnovationCounter`.

Parameters:
- `options` - The fresh tuple's build input.

Returns: The fresh round tuple.

Example:

```ts
const nextRound = buildNeatRacingNextRoundTuple({
  base,
  targetLevel: 2,
  seedNetwork,
  carriedInnovationCounter: 42,
});
nextRound.agentCount; // 2 (the level-2 car count)
```

### buildNeatRacingPromotedRoundTuple

```ts
buildNeatRacingPromotedRoundTuple(
  options: NeatRacingPromotedRoundTupleOptions,
): NeatRacingPromotionRoundTuple
```

Builds the PROMOTED round's fresh tuple from the boundary's own evidence
(E6, D4, AC-031).

Educational note — what makes this the promotion-flavored variant: unlike
`buildNeatRacingNextRoundTuple` (which consumes an already-built seed
payload), this builder DERIVES the seed network from the promoted round's
own fingerprint — the first agent car's network is constructed from the
promoted tuple's seed and the promoted LEVEL's observation width (the same
width the live session rebuilds that car with), then its weights ship with
the frozen canonical NGE envelope. The innovation counter stays an INPUT
(the realm never reads a global): the host reads the live counter at the
crossing and hands it in, so the boundary records exactly the value the
prior round's evolution reached.

Parameters:
- `options` - The promoted tuple's build input.

Returns: The promoted round's fresh tuple (seed network and carried
innovation counter included).

Example:

```ts
const promoted = buildNeatRacingPromotedRoundTuple({
  base,
  targetLevel: 2,
  carriedInnovationCounter: 42,
});
promoted.seedNetwork.weights; // Float32Array (the car-1 genome)
promoted.carriedInnovationCounter; // 42
```

### canonicalizeEnvelopeJsonValue

```ts
canonicalizeEnvelopeJsonValue(
  value: unknown,
): unknown
```

Canonicalizes the NGE envelope into a key-sorted JSON value.

The base tuple's canonicalizer is private to its own module, so this module
owns the envelope's canonical form (a NEW payload, never a second path over
the AC-026 tuple). Keys are sorted, so the serial can never depend on the
envelope's insertion order. The STRICTNESS CONTRACT is the base's own (the
one canonical serial may not carry two regimes):

- Plain objects only (prototype `Object.prototype` or `null`): a Date, Map,
  Set or class instance has no canonical JSON form and is rejected instead
  of silently collapsing to `{}` — the base's `isPlainJsonObject` guard is
  mirrored one-for-one.
- Non-finite numbers are rejected (`numbers must be finite` — the base's
  own message) instead of being coerced to `null`, which would make the
  serial non-injective (NaN, null and an absent key could no longer be
  told apart by the replay anchor).
- Undefined-valued object keys DROP (JSON semantics — the base's
  `canonicalizeReproducibilityObject` drops them the same way) instead of
  materializing a `null` the base region would never have carried.

Parameters:
- `value` - The envelope (or any of its nested values).

Returns: The canonical JSON value.

### clampPromotionLevel

```ts
clampPromotionLevel(
  level: number,
): NeatRacingObservationLevel
```

Clamps a session level onto the ladder's own rungs (1, 2 or 3).

Parameters:
- `level` - The requested level.

Returns: The clamped level.

### collectFailedConditions

```ts
collectFailedConditions(
  conditions: NeatRacingPromotionConditions,
): NeatRacingPromotionConditionName[]
```

Collects the conditions that did not hold at the crossing tick.

Parameters:
- `conditions` - The three conditions' verdicts.

Returns: The names of the conditions that did not hold.

### createNeatRacingRoundBoundaryState

```ts
createNeatRacingRoundBoundaryState(
  targetLevel: number,
  carriedInnovationCounter: number,
): NeatRacingRoundBoundaryState
```

Creates the state a fresh round starts from (E6).

The adaptation-engine state RESETS (the oscillation predicate's hysteresis
state returns to its zeroed state), the session counters restart at zero
and the innovation counter CARRIES with the seed network. The fresh round's
cars are materialized FROM ITS FIRST TICK (no mid-session materialization).

Parameters:
- `targetLevel` - The level the fresh round runs at.
- `carriedInnovationCounter` - The innovation counter carried from the
prior round.

Returns: The fresh round's boundary state.

Example:

```ts
const state = createNeatRacingRoundBoundaryState(2, 42);
state.sessionLaps; // 0 (reset)
state.innovationCounter; // 42 (carried)
```

### createNeatRacingSeedNetworkPayload

```ts
createNeatRacingSeedNetworkPayload(
  weights: Float32Array<ArrayBufferLike>,
  envelope: NgeDnaCanonicalEnvelope,
): NeatRacingSeedNetworkPayload
```

Builds the seed-network payload (D4) of the fresh round.

The weights are copied out of the caller's array, so the payload owns its
typed array and a later mutation of the source can never change the round
tuple.

Parameters:
- `weights` - The crossing network's `CarGenome.serialize()` payload.
- `envelope` - The NGE envelope that travels with the weights.

Returns: The seed-network payload.

Example:

```ts
const payload = createNeatRacingSeedNetworkPayload(
  new Float32Array([0.5, -0.25]),
  envelope,
);
```

### encodeFloat32ArrayAsHex

```ts
encodeFloat32ArrayAsHex(
  weights: Float32Array<ArrayBufferLike>,
): string
```

Encodes a typed array as a byte-exact hex string (the typed-array
serialization D4 requires — never `toJSON`, whose key order would make the
serial key-order sensitive).

Parameters:
- `weights` - The typed array to encode.

Returns: The hex string of the array's raw bytes.

### evaluateNeatRacingDrivabilityGate

```ts
evaluateNeatRacingDrivabilityGate(
  measurement: NeatRacingDrivabilityMeasurement,
): NeatRacingDrivabilityVerdict
```

Evaluates the drivability gate (condition (c)) over the crossing lap.

The gate holds when ALL THREE checks hold: the mean forward speed reaches
the named fraction of the PINNED scripted-baseline lap pace, the TP-09
oscillation metric stays at or below its bound, and the off-track fraction
stays at or below its bound. A MISSING baseline (a non-positive pinned
pace) can never satisfy the speed check — an unpinned baseline must not
promote.

Parameters:
- `measurement` - The crossing lap's drivability measurement.

Returns: The gate's verdict.

Example:

```ts
evaluateNeatRacingDrivabilityGate({
  meanForwardSpeed: 30,
  scriptedBaselineLapPace: 40,
  oscillationMetric: 0.1,
  offTrackFraction: 0,
}).passed; // true (30 >= 0.75 * 40)
```

### evaluateNeatRacingPromotion

```ts
evaluateNeatRacingPromotion(
  input: NeatRacingPromotionEvaluationInput,
): NeatRacingPromotionEvaluation
```

Evaluates the THREE-PART promotion predicate at a crossing tick.

ALL THREE conditions must hold at the crossing tick: the forward
finish-line crossing after one complete circuit (a), the node count above
the level's threshold (b), and the drivability gate over the crossing lap
(c). The OQ-002 lap floor is a PRECONDITION of the evaluation, not a
fourth condition: below the floor the predicate never promotes.

Parameters:
- `input` - The crossing tick's evaluation input.

Returns: The predicate's verdict, carrying the per-condition verdicts and
the level the session ends up at.

Example:

```ts
const evaluation = evaluateNeatRacingPromotion({
  level: 1,
  sessionLaps: 20,
  nodeCount: 320,
  forwardCrossing: true,
  lapsSinceLastCrossing: 1,
  drivability: {
    meanForwardSpeed: 30,
    scriptedBaselineLapPace: 40,
    oscillationMetric: 0.1,
    offTrackFraction: 0,
  },
});
evaluation.promoted; // true
evaluation.toLevel; // 2
```

### findClosestSplineSampleIndex

```ts
findClosestSplineSampleIndex(
  car: PhysicsCarState,
  samples: readonly SplineSample[],
): number
```

Finds the spline sample the car is closest to (the course's own sample
order is the lap's progress measure).

Parameters:
- `car` - The car's current state.
- `samples` - The course's spline samples.

Returns: The index of the closest sample.

### isFinishLineCrossing

```ts
isFinishLineCrossing(
  furthestSampleIndex: number,
  currentSampleIndex: number,
  sampleCount: number,
): boolean
```

Resolves whether the car wrapped the start/finish line on the course's own
sample order (reached the last samples and returned to the first ones).

Parameters:
- `furthestSampleIndex` - The furthest sample index reached so far.
- `currentSampleIndex` - The sample index the car is closest to now.
- `sampleCount` - The course's sample count.

Returns: True when the car wrapped the finish line.

### isNeatRacingPromotionEvaluationEligible

```ts
isNeatRacingPromotionEvaluationEligible(
  sessionLaps: number,
): boolean
```

Resolves the OQ-002 session-lap floor's eligibility at a crossing tick.

The floor is a PRECONDITION of the evaluation, never a fourth condition of
the predicate: below the floor the predicate is not evaluated at all.

Parameters:
- `sessionLaps` - The laps the session has completed so far.

Returns: True when the predicate may be evaluated at this crossing.

Example:

```ts
isNeatRacingPromotionEvaluationEligible(20); // true
isNeatRacingPromotionEvaluationEligible(19); // false
```

### isPromotionRoundSnapshot

```ts
isPromotionRoundSnapshot(
  value: unknown,
): boolean
```

Guards the fresh round tuple's opponent snapshot's shape.

Parameters:
- `value` - The tuple's opponent snapshot.

Returns: True when the snapshot is a promotion round snapshot.

### NEATRACING_AI_MIN_SESSION_LAPS

OQ-002 session-lap floor of the AI lane: the promotion predicate is
evaluated at forward finish-line crossings ONLY once the session has
completed this many laps. It is a PRECONDITION of evaluation, never a
fourth condition of the three-part predicate.

### NEATRACING_DRIVABILITY_OFF_TRACK_MAX_FRACTION

TP-17 drivability gate — the highest fraction of the crossing lap the car
may spend off track. Arbitrary starting tuning point (like TP-07), tuned in
Phase 2.

### NEATRACING_DRIVABILITY_OSCILLATION_MAX

TP-17 drivability gate — the highest oscillation metric the crossing lap
may show. The metric keeps the TP-09 shape (the deadband
`RACING_OSCILLATION_MIN_MEAN_STEERING = 0.15` is preserved by the metric
detector). The bound sits BELOW the oscillation growth-commit gate
(`RACING_OSCILLATION_COMMIT_THRESHOLD = 0.5`), so a drivable car never
trips the commit boost. Arbitrary starting tuning point (like TP-07), tuned
in Phase 2.

### NEATRACING_DRIVABILITY_SPEED_FRACTION

TP-17 drivability gate — the fraction of the scripted-baseline lap pace the
crossing lap's mean forward speed must reach. Arbitrary starting tuning
point (like TP-07), tuned in Phase 2.

### NEATRACING_PROBE_DEFAULT_MAX_TICKS

The probe episode's default tick budget (at the 60 Hz physics step: 200 s
of driving — generous enough for several laps of a long course).

### NEATRACING_PROBE_LAP_WRAP_TOLERANCE

The sample-index tolerance of the probe's finish-line wrap detection: the
wrap counts when the car has reached the LAST samples of the course and
returns to the FIRST ones (the tolerance absorbs the sample-density
jitter of the closest-sample lookup).

### NEATRACING_PROBE_MIN_SAMPLE_COUNT

The smallest number of spline samples a course must own for the probe
episode to attempt a lap (a shorter course has no meaningful circuit).

### NEATRACING_PROBE_SPAWN_SEED

The probe episode's fixed spawn seed (the deterministic lane offset of the
spawned car; the probe never reads a random source).

### NEATRACING_PROMOTION_MAX_LEVEL

The highest rung of the AI level ladder (lvl 1 -> 2 -> 3). The ladder is
CAPPED here: at level 3 the session keeps evolving until the user stops it
or switches mode, so no promotion target exists beyond this level.

### NEATRACING_PROMOTION_MIN_CIRCUIT_LAPS

Condition (a)'s minimum circuit count: the car must have completed at least
this many full laps since its last finish-line crossing, so a back-and-forth
over the line and a crossing on the first pass never promote.

### NeatRacingDrivabilityCheckName

The names of the drivability gate's three checks.

### NeatRacingDrivabilityMeasurement

The drivability measurement of ONE completed lap (the crossing lap).

### NeatRacingDrivabilityVerdict

The drivability gate's verdict over the crossing lap.

### NeatRacingNextRoundTupleOptions

The fresh round tuple's build input.

### NeatRacingPromotedRoundTupleOptions

The promoted round-tuple builder's input (E6/D4).

### NeatRacingPromotionConditionName

The names of the predicate's three conditions (in evaluation order).

### NeatRacingPromotionConditions

The predicate's per-condition verdicts at the crossing tick.

### NeatRacingPromotionEvaluation

The predicate's verdict at the crossing tick.

### NeatRacingPromotionEvaluationInput

The crossing tick's evaluation input.

### NeatRacingPromotionRoundSnapshot

The fresh round's opponent snapshot: the new level and the car count the
new level materializes (the E6 "new `opponentSnapshot` matching the new
level's car count" requirement).

### NeatRacingPromotionRoundTuple

The fresh round's reproducibility tuple: the AC-026 tuple PLUS the two
explicit D4 fields (`seedNetwork` and `carriedInnovationCounter`), so the
round N+1 session is replayable from its own tuple alone.

### NeatRacingRoundBoundaryState

The state a fresh round starts from (E6): the adaptation-engine state is
RESET, the session counters restart at zero, and the innovation counter
CARRIES with the seed network.

### NeatRacingScriptedBaselineProbeOptions

The scripted-baseline probe episode's input.

### NeatRacingScriptedBaselineProbeResult

The scripted-baseline probe episode's result.

### NeatRacingSeedNetworkPayload

The SEED NETWORK payload (D4): the crossing network's serialized weights as
a TYPED ARRAY plus the NGE envelope that travels with it.

### resolveNeatRacingLevelCarCount

```ts
resolveNeatRacingLevelCarCount(
  level: number,
): number
```

Resolves the number of cars a level materializes (the E2 team layouts are
the single source of truth: level 1 owns one car, level 2 two, level 3
five).

Parameters:
- `level` - The session level.

Returns: The car's count materialized from the round's first tick.

Example:

```ts
resolveNeatRacingLevelCarCount(1); // 1
resolveNeatRacingLevelCarCount(3); // 5
```

### resolveNeatRacingPromotionTargetLevel

```ts
resolveNeatRacingPromotionTargetLevel(
  level: number,
  nodeCount: number,
): number
```

Resolves the rung the node-count band selects for a session level.

The band is LEVEL-RELATIVE: the threshold is the current level's promotion
threshold (`resolveOscillationPromotionNodeThreshold`), so a promotion
climbs exactly ONE rung and re-bases the floor (300 -> 600) exactly as the
oscillation law expects. The comparison is STRICT (`nodeCount > threshold`).
The ladder CAPS at {@link NEATRACING_PROMOTION_MAX_LEVEL}.

Parameters:
- `level` - The session level.
- `nodeCount` - The crossing network's node count at the crossing tick.

Returns: The level the session would climb to (the current level when the
band is not crossed or the ladder is capped).

Example:

```ts
resolveNeatRacingPromotionTargetLevel(1, 301); // 2
resolveNeatRacingPromotionTargetLevel(1, 300); // 1 (strict '>')
resolveNeatRacingPromotionTargetLevel(3, 900); // 3 (the ladder is capped)
```

### resolveProbeSpawnPose

```ts
resolveProbeSpawnPose(
  samples: readonly SplineSample[],
): PhysicsCarSpawnPose
```

Resolves the probe's spawn pose: the first spline sample's position, facing
the second sample (the course's own forward direction).

Parameters:
- `samples` - The course's spline samples.

Returns: The spawn pose.

### resolvePromotionConditions

```ts
resolvePromotionConditions(
  input: NeatRacingPromotionEvaluationInput,
  level: number,
): NeatRacingPromotionConditions
```

Resolves the predicate's three per-condition verdicts at the crossing tick.

Parameters:
- `input` - The crossing tick's evaluation input.
- `level` - The clamped session level.

Returns: The three conditions' verdicts.

### restoreNeatRacingRoundFromTuple

```ts
restoreNeatRacingRoundFromTuple(
  tuple: NeatRacingPromotionRoundTuple,
): NeatRacingRoundBoundaryState
```

Restores the fresh round's boundary state from the round's OWN tuple.

This is the D4 replay path: the tuple alone (its seed, its level snapshot
and its carried innovation counter) rebuilds the round's boundary state, so
round N+1 never depends on state that only the prior round held.

Parameters:
- `tuple` - The fresh round tuple.

Returns: The fresh round's boundary state.

Example:

```ts
const restored = restoreNeatRacingRoundFromTuple(nextRound);
restored.level; // the fresh round's level
```

### runNeatRacingScriptedBaselineProbeEpisode

```ts
runNeatRacingScriptedBaselineProbeEpisode(
  options: NeatRacingScriptedBaselineProbeOptions,
): NeatRacingScriptedBaselineProbeResult
```

Runs the PINNED scripted-baseline probe episode for one course.

The probe drives the per-course profile-augmented scripted controller
(AC-028's throttle profile table) on the SAME track/seed with a scripted
input only — no human input and no evolving agents — and measures the mean
forward speed of the lap it completes. The result is a pure deterministic
function of the track tuple, so the drivability gate's speed reference is
reproducible.

The lap is detected on the course's own sample order: the car has reached
the LAST spline samples and returns to the FIRST ones (the start/finish
wrap). A course that is too short to own a circuit, or a probe that never
completes a lap, reports `completedLap: false` and a zero pace — an
unpinned baseline can never promote.

Parameters:
- `options` - The probe's course and tick budget.

Returns: The probe's result.

Example:

```ts
const probe = runNeatRacingScriptedBaselineProbeEpisode({ trackSpec });
probe.completedLap; // true on a course with a real circuit
```

### serializeNeatRacingPromotionRoundTuple

```ts
serializeNeatRacingPromotionRoundTuple(
  tuple: NeatRacingPromotionRoundTuple,
): string
```

Serializes the fresh round tuple into its canonical string shape.

The AC-026 tuple's own canonical serial is REUSED for the base fields (one
serialization path, never a second one) and the two D4 fields are appended:
the `seedNetwork` (the typed-array weights encoded as a byte-exact hex
string plus the key-sorted NGE envelope — a typed array is NEVER `toJSON`ed,
because JSON key order would make the serial key-order sensitive) and the
`carriedInnovationCounter`.

Parameters:
- `tuple` - The fresh round tuple.

Returns: The canonical serial (the round's replay identity anchor).

Example:

```ts
serializeNeatRacingPromotionRoundTuple(nextRound);
// '{"seed":1,...,"seedNetwork":{"weights":"0000803f",...},"carriedInnovationCounter":42}'
```

### serializeNetworkWeightsToFloat32Array

```ts
serializeNetworkWeightsToFloat32Array(
  network: default,
): Float32Array<ArrayBufferLike>
```

Serializes one network's connection weights into a fresh typed array.

Educational note — why a manual positional walk: the D4 seed genome ships
as a `Float32Array` weight vector plus the canonical envelope (NOT the
`toJSON`-style compact serialization), so the boundary copies each
connection weight positionally. The fresh array detaches the payload from
the live network — later evolution of the source must never mutate the
shipped genome.

Parameters:
- `network` - The network whose connection weights serialize.

Returns: The fresh weight vector (one entry per connection).

## ai/rival.propagation.ts

NeatRacing rival propagation (slice `P2-S3-D`, AC-032).

THE LAW (spec section 9.6, E4): when a rival crosses the finish line, the
OTHER rivals' minds are replaced by WEIGHT-NUDGE MUTATIONS of the
crossing rival's serialized weights — one bounded MOD_WEIGHT round with a
SMALL delta range, seeded per swap event. The operator is a PURE
SERIALIZED-WEIGHTS PLANE:

- RECYCLED COEVOLUTION PROTOCOL (the racing curriculum's
  `createCarGenome`): the mind is `CarGenome.serialize()`'s Float32Array
  of connection weights, and the mutation mirrors the recycled MOD_WEIGHT
  semantics — per weight, one uniform delta sample added
  (`min + u * (max - min)`), here over the SMALL named range
  `[NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MIN, NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MAX]`.
  The src MOD_WEIGHT default range (plus/minus 1) is intentionally NOT
  reused: rival-to-rival propagation is a NUDGE (a TP tuning point, not
  a full re-roll), and the mutation RNG is the seeded xorshift32 stream
  (section 9.7.1), never `Math.random`.
- STRUCTURAL GROWTH STAYS WITH EACH RIVAL'S OWN ADAPTATION ENGINE: the
  operator touches WEIGHTS ONLY — its input and output are flat Float32
  arrays, it never imports `Network`, and it never adds, removes or
  rewires anything. NGE growth (nodes, topology) remains the per-rival
  adaptation engine's lane; this operator only re-seeds the weights.
- LAP/LINE-BOUNDARY SWAPS ONLY: the operator refuses every call whose
  `atLapLineBoundary` flag is not `true` — a mid-lap swap is a
  programming error, never a silent one (spec E4: swaps are applied at
  LAP/line boundaries, never mid-lap).
- THE MAIN AGENT IS STRUCTURALLY OUT OF REACH (the Blocked stop
  condition is a GUARD, not a promise): car 0 is the main agent's slot at
  EVERY level team layout (`[0]` / `[0, 1]` / `[0, 1, 1, 1, 1]` — AC-033
  "human takes car 0"), and the operator REJECTS any rival index at or
  below `NEATRACING_RIVAL_SWAP_MAIN_AGENT_CAR_INDEX`. Replacement minds
  are derived ONLY from the rival set minus the crossing rival, so no
  code path can return a main-agent mind. The main agent is NEVER
  touched and NEVER fully reset by rival success.
- LVL-2 IS A DOCUMENTED NO-OP: a single-rival session has nobody to
  propagate to — the crossing rival keeps its mind, the result carries
  ZERO replacements, and the swap payload records an empty propagated
  list. The lvl-3 case (four rivals) propagates to the THREE others.

THE SWAP PAYLOAD IS THE SECTION 9.7 REPLAY ANCHOR: the crossing-network
snapshot (carIndex + serialized weights) and the event's mutation RNG
seed are serialized into the reproducibility tuple's `opponentSnapshot`
AT EACH SWAP, so the tuple replays swap events exactly. The payload shape
is JSON-SAFE BY CONSTRUCTION (hex weights, plain numbers, arrays) — the
one shape the base tuple canonicalizer digests — and this module owns the
canonicalization for its own NEW payload: byte-exact hex, key-sorted
canonical serial (the D4 law from `promotion.rules.ts`, never a
`toJSON`). The replay rung is ORDERED-DETERMINISTIC LEVEL 2
(same-runtime): given the same delta-range constants and the same
underlying float32 math, the payload alone rebuilds the post-swap minds
(`restoreNeatRacingRivalSwapReplacementMinds`), with the draw order
pinned — targets ascending by car index, then weight index ascending.

REALM LAW: pure/headless — no ambient RNG, no wall clock, no DOM. The
only random source is the seeded xorshift32 stream.

### assertCrossingCarIndex

```ts
assertCrossingCarIndex(
  carIndex: unknown,
): void
```

Guards a crossing rival car index (a positive integer — never the main
agent's slot).

Parameters:
- `carIndex` - The crossing car index to guard.

### assertCrossingWeights

```ts
assertCrossingWeights(
  weights: unknown,
  prefix: string,
): void
```

Guards the crossing network's serialized weights (empty and non-finite
are loud rejections, mirroring the D4 finite-weights guard).

Parameters:
- `weights` - The weights to guard.
- `prefix` - The error-voice prefix of the calling lane.

### assertHexWeights

```ts
assertHexWeights(
  hex: unknown,
): void
```

Guards the hex crossing-weights field of a payload — STRUCTURE and the
DECODED PLANE both (fix-packet-P2-S3-D-iteration-1 FIX 2).

The structure check alone would let a hex string that decodes to a
non-finite float32 (NaN or an infinity) reach the canonical serial — a
certified replay anchor the restore path would then reject.
Strict-rejection parity: the serializer rejects exactly what the restore
rejects.

Parameters:
- `hex` - The candidate hex string.

### assertPayloadShape

```ts
assertPayloadShape(
  payload: NeatRacingRivalSwapPayload,
): void
```

Guards the payload's shape (the restore and serializer's entry guard).

Parameters:
- `payload` - The payload to guard.

### assertPropagatedCarIndices

```ts
assertPropagatedCarIndices(
  indices: unknown,
  crossingCarIndex: number,
): void
```

Guards the propagated rival car indices: positive rivals (never the main
agent's slot, never the crossing rival), strictly ascending (the pinned
Level 2 draw order's record must be canonical).

Parameters:
- `indices` - The propagated car indices.
- `crossingCarIndex` - The crossing rival's car index.

### assertSnapshotShape

```ts
assertSnapshotShape(
  snapshot: NeatRacingRivalSwapSnapshot,
): void
```

Guards the runtime snapshot's shape (the payload encoder's entry guard).

Parameters:
- `snapshot` - The runtime snapshot to guard.

### assertUnsigned32Seed

```ts
assertUnsigned32Seed(
  value: unknown,
  label: string,
): void
```

Guards an unsigned 32-bit seed value (the derivation table's integer
shape — a seed is a stored RNG-state component, not a free number).

Parameters:
- `value` - The seed to guard.
- `label` - The seed's name (for the static message surface).

### buildReplacementMinds

```ts
buildReplacementMinds(
  crossingWeights: Float32Array<ArrayBufferLike>,
  mutationSeed: number,
  targetCarIndices: readonly number[],
): NeatRacingRivalSwapReplacementMind[]
```

Builds one replacement mind per target rival (the ONE nudge round).

THE PINNED DRAW ORDER (Level 2 ordered-deterministic): targets ascend by
car index; within a target, weight indices ascend; each weight consumes
exactly one stream draw. Both the operator and the replay path funnel
through this helper, so the replay is the operator's own math.

Parameters:
- `crossingWeights` - The crossing rival's serialized weights.
- `mutationSeed` - The event's derived mutation seed.
- `targetCarIndices` - The targets, ascending (the caller guards).

Returns: One replacement mind per target (NEW arrays; input never
mutated, never aliased).

### decodeHexAsFloat32Array

```ts
decodeHexAsFloat32Array(
  hex: string,
): Float32Array<ArrayBufferLike>
```

Decodes a byte-exact hex string back into its Float32Array.

The caller's shape guard (`assertHexWeights`) rejects non-lowercase,
non-float32-aligned or empty strings BEFORE the decode, so the decode
itself stays total on guarded input.

Parameters:
- `hex` - The guarded hex string (lowercase, length a multiple of 8).

Returns: The decoded weights.

### encodeFloat32ArrayAsHex

```ts
encodeFloat32ArrayAsHex(
  weights: Float32Array<ArrayBufferLike>,
): string
```

Encodes a typed array as a byte-exact hex string.

The `promotion.rules.ts` encoder is private to ITS payload; this module
owns the canonicalization for its OWN payload, so the D4 encoder is
mirrored here (one serialization path per payload — never a shared dual
path). Lowercase hex, two characters per byte, no separators — the exact
byte pattern `Float32Array` holds, never a `toJSON`.

Parameters:
- `weights` - The typed array to encode.

Returns: The hex string of the array's raw bytes in NATIVE byte order
(little-endian on the supported runtimes — the D4 law's own convention,
pinned by `promotion.rules.test.ts` as `0000803f` for 1.0). NOT
endian-stable: the payload's byte parity is same-runtime (Level 2,
ordered-deterministic), never cross-environment exact.

### NEATRACING_RIVAL_SWAP_MAIN_AGENT_CAR_INDEX

The main agent's fleet car index (car 0 — AC-033; every level team layout
names it team 0's slot: `NEATRACING_TEAM_LAYOUTS` = `[0]` / `[0, 1]` /
`[0, 1, 1, 1, 1]`). The propagation operator never owns this slot: rival
car indices must be STRICTLY greater, so no replacement mind and no
crossing anchor can ever address the main agent. The Blocked stop
condition ("propagation touches the main agent state") is enforced in
code, not in prose.

### NEATRACING_RIVAL_SWAP_PAYLOAD_KIND

The swap payload's kind discriminator inside `opponentSnapshot`. The
reproducibility tuple's opponent snapshot is opaque at that boundary, so
the payload self-describes: a payload whose `kind` is not
`NEATRACING_RIVAL_SWAP_PAYLOAD_KIND` is rejected (no silent nulls).

### NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MAX

The rival-swap nudge delta range upper bound (see the lower bound — the
range is symmetric around zero).

### NEATRACING_RIVAL_SWAP_WEIGHT_DELTA_MIN

The rival-swap nudge delta range bounds (E4: "one bounded MOD_WEIGHT
round" with a SMALL delta range — a TP tuning point, NOT the library
default).

The recycled coevolution protocol mutates with the MOD_WEIGHT method's
`{ min, max }` shape; the src default (plus/minus 1) is a full-weight-
magnitude rewrite, wrong for rival-to-rival propagation where the
winner's mind is a starting point to be NUDGED, not re-rolled. The
symmetric range keeps every swap a bounded refinement. The sample mirrors
the method's contract: `min + u * (max - min)` with `u` drawn from the
event's seeded stream.

### NeatRacingRivalPropagationOptions

The operator call contract for one lap/line-boundary rival swap.

### NeatRacingRivalPropagationResult

The operator result: the runtime snapshot plus the replacement minds.

### NeatRacingRivalSwapCrossingSnapshot

The crossing-network snapshot: WHICH rival crossed and the mind it
crossed with. The snapshot owns its bytes — the weights are copied from
the caller's array so later runtime mutation can never change them.

### NeatRacingRivalSwapPayload

The tuple-carried swap record — the JSON-SAFE shape written into the AC-026
reproducibility tuple's `opponentSnapshot` at each swap (base-tuple
canonicalizer parity: plain objects, finite numbers, hex strings — no
typed arrays, whose canonical position in the tuple is the D4
`seedNetwork.weights` field only).

### NeatRacingRivalSwapReplacementMind

One replacement rival mind: the target rival's fleet car index plus its
post-swap serialized weights (the crossing mind with the bounded nudges
applied). The runtime applies the weights to the rival's network through
the Step 05 host wiring; this module stays on the flat serialized plane.

### NeatRacingRivalSwapSnapshot

The runtime-form swap snapshot for one rival swap event (the typed twin
of the tuple payload). Converted into the JSON-safe
`NeatRacingRivalSwapPayload` for the reproducibility tuple's
`opponentSnapshot` record.

### propagateNeatRacingRivalWinnerWeights

```ts
propagateNeatRacingRivalWinnerWeights(
  options: NeatRacingRivalPropagationOptions,
): NeatRacingRivalPropagationResult
```

Replaces every NON-crossing rival's mind with a bounded seeded
weight-nudge mutation of the crossing rival's serialized weights.

THE E4 OPERATOR: at a lap/line boundary a rival crossed the finish line
(the crossing rival). Its serialized mind — the recycled
`CarGenome.serialize()` Float32Array — becomes the base for ALL other
rivals: each target gets a NEW weights array holding the crossing weights
with ONE bounded MOD_WEIGHT-style round applied (per weight, one uniform
delta from the event's seeded stream, drawn in a pinned order: targets
ascending by car index, then weight index ascending). Structural growth
is untouched by construction — the plane is flat weights only.

The event's mutation seed derives from the section 9.7.2 rival-swap row
(`deriveNeatRacingRivalSwapChildSeed` — the named per-event stride keeps
every swap event's stream disjoint), and the result carries the snapshot
needed to serialize the seed + crossing weights into `opponentSnapshot`
so the reproducibility tuple replays the swap exactly.

Parameters:
- `options` - The swap call: the session's rival car indices (never a
main-agent index), the crossing rival's car index, the crossing
rival's serialized weights, the AC-026 tuple seed, the swap-event
index, and the LAP/line-boundary flag (a mid-lap call is rejected).

Returns: The swap result: the runtime snapshot plus one replacement mind
per non-crossing rival in ascending car order. The lvl-2 no-op (a
single rival) returns ZERO replacements and an empty propagated list.

Example:

```ts
const { snapshot, replacements } = propagateNeatRacingRivalWinnerWeights({
  rivalCarIndices: [1, 2, 3, 4],
  crossingCarIndex: 2,
  winnerWeights: crossing.carGenome.serialize(),
  tupleSeed: tuple.seed,
  eventIndex: nextRivalSwapEventIndex(session),
  atLapLineBoundary: true,
});
```

### restoreNeatRacingRivalSwapReplacementMinds

```ts
restoreNeatRacingRivalSwapReplacementMinds(
  payload: NeatRacingRivalSwapPayload,
): readonly NeatRacingRivalSwapReplacementMind[]
```

Rebuilds the post-swap replacement minds from the SERIALIZED PAYLOAD
ALONE (the section 9.7 replay path — "the reproducibility tuple replays
swap events exactly").

The payload carries the crossing network's hex weights, the event's
mutation seed and the propagated car indices; those three facts plus the
module's fixed delta range and the pinned draw order (targets ascending
by car index, then weight index ascending) fully determine every
replacement mind. A replay needs neither the original options, the tuple
seed nor any runtime state — only the payload that lived inside
`opponentSnapshot`. Same-runtime exactness: the math is the same float32
plane the operator wrote.

Parameters:
- `payload` - The swap payload (as stored in the tuple's
`opponentSnapshot` record; key order does not matter).

Returns: One replacement mind per propagated rival, in ascending car
order — byte-identical to the operator's own replacements for the same
payload.

Example:

```ts
const replacements = restoreNeatRacingRivalSwapReplacementMinds(
  tuple.opponentSnapshot as NeatRacingRivalSwapPayload,
);
```

### sampleSwapDelta

```ts
sampleSwapDelta(
  stream: NeatRacingXorShift32Stream,
): number
```

Samples ONE bounded nudge delta for the swap round (the recycled
MOD_WEIGHT uniform-sample contract on the small named range).

Parameters:
- `stream` - The event's seeded xorshift32 stream.

Returns: The delta (in the closed-ish `[MIN, MAX)` uniform range).

### serializeNeatRacingRivalSwapPayload

```ts
serializeNeatRacingRivalSwapPayload(
  payload: NeatRacingRivalSwapPayload,
): string
```

Serializes the swap payload into a canonical byte-exact string — the
replay anchor of one swap event.

THE ONE SERIALIZATION PATH for this module's payload: the canonical
object is rebuilt with keys in sorted order (`crossing`, `eventIndex`,
`kind`, `mutationSeed`, `propagatedCarIndices`; `crossing` sorts its own
two keys) so the serial can never depend on insertion order, and the
crossing weights ride as BYTE-EXACT HEX (never a `toJSON`). The payload
is validated first: a hand-rolled payload carrying non-finite numbers, a
typed array or a malformed hex string is rejected instead of silently
collapsing (the promotion module's strictness law, mirrored for this NEW
payload).

Parameters:
- `payload` - The swap payload (any key order).

Returns: The canonical JSON string (sorted keys, hex weights embedded).

Example:

```ts
const replayAnchor = serializeNeatRacingRivalSwapPayload(
  toNeatRacingRivalSwapPayload(swap.snapshot),
);
```

### SwapRecordGuardView

A possibly-malformed record handed to the shape guards (the base tuple
carries `opponentSnapshot` as `unknown` at the boundary). Every field is
validated at runtime instead of trusted — the guard view types carry
nothing but the object-ness itself.

### toNeatRacingRivalSwapPayload

```ts
toNeatRacingRivalSwapPayload(
  snapshot: NeatRacingRivalSwapSnapshot,
): NeatRacingRivalSwapPayload
```

Encodes the runtime swap snapshot into the JSON-safe tuple payload — the
ONE shape that may be written into the AC-026 reproducibility tuple's
`opponentSnapshot` record at a swap (the base canonicalizer digests plain
objects, finite numbers and strings; typed arrays belong in the D4
`seedNetwork.weights` slot only, so the crossing weights ride as
byte-exact hex here).

Parameters:
- `snapshot` - The runtime snapshot (the operator result's `snapshot`).

Returns: The tuple-carried swap payload (hex weights, plain numbers).

Example:

```ts
nextTuple.opponentSnapshot = toNeatRacingRivalSwapPayload(swap.snapshot);
```

## ai/runtime.adaptation.ts

### buildCandidateScoreWindow

```ts
buildCandidateScoreWindow(
  network: default,
  evidenceWindow: readonly (number | NeatRacingQualitySignal)[],
): number[]
```

Build a fresh candidate score window from the post-mutation network's
forward-pass outputs.

Unlike the shared {@link NeatRacingAdaptationTickInput.scoreHistory} (which
represents historical driving quality and is identical for both baseline
and candidate evaluations), this window is derived by activating the
post-mutation network on sample observations from the evidence window and
converting each output vector into a scalar quality score.  This ensures
the candidate score reflects the actual behavioral impact of the structural
mutation, not historical performance.

Each output vector is reduced to a scalar by taking the mean of its
absolute activation values.  A mutation that disrupts driving behavior
produces different activation magnitudes, yielding a different candidate
score window and therefore a different candidate score — even when the
historical scoreHistory is unchanged.

Parameters:
- `network` - Post-mutation candidate network to evaluate.
- `evidenceWindow` - Filtered rolling score history used to derive
sample observations for the forward passes.

Returns: Array of scalar quality scores, one per sample observation.

### buildGrowthBudget

```ts
buildGrowthBudget(
  network: default,
  limits: NeatRacingAdaptationLimits,
): NgeGrowthBudget
```

Build a growth budget from the runtime limits and live network.

Parameters:
- `network` - Live controller network.
- `limits` - Runtime adaptation limits.

Returns: NGE growth budget for the lifecycle apply phase.

### buildModuleMetricsSnapshot

```ts
buildModuleMetricsSnapshot(
  network: default,
  evidenceWindow: readonly (number | NeatRacingQualitySignal)[],
): NgeModuleMetricsSnapshot
```

Build a module metrics snapshot from the runtime evidence window.

Parameters:
- `network` - Live controller network.
- `evidenceWindow` - Filtered rolling score history.

Returns: NGE module metrics for the lifecycle focus scorer.

### buildPruneBudget

```ts
buildPruneBudget(
  network: default,
): NgePruneBudget
```

Build a prune budget from the live network.

Parameters:
- `network` - Live controller network.

Returns: NGE prune budget for the lifecycle apply phase.

### collectForwardPassOutputs

```ts
collectForwardPassOutputs(
  network: default,
  scoreHistory: readonly (number | NeatRacingQualitySignal)[],
): number[][]
```

Collects forward-pass outputs from the network by activating it on sample
observations drawn from the score history.  For numeric entries the scalar
is repeated to fill the input vector; for composite signals the five
signal fields are tiled or truncated to the input size.

Parameters:
- `network` - Candidate network to activate.
- `scoreHistory` - Rolling score window used to derive observations.

Returns: Array of output vectors, one per sample observation.

### createIdleOscillationEngagement

```ts
createIdleOscillationEngagement(
  level: number,
): OscillationEngagement
```

Builds the idle engagement verdict of a fresh engine.

The shared predicate is evaluated against an empty network, so the hard
node-count gate keeps the law fully disengaged until the engine's first
real tick evaluates it.

Parameters:
- `level` - The session level whose TP-07 node threshold gates the law.

Returns: The fully disengaged engagement verdict.

### createNeatRacingPerCarAdaptationEngines

```ts
createNeatRacingPerCarAdaptationEngines(
  tuple: NeatRacingReproducibilityTuple,
  carCount: number,
  options: Omit<NeatRacingAdaptationEngineOptions, "random" | "carId">,
): Map<number, NeatRacingAdaptationEngine>
```

Creates one independent runtime adaptation engine per car index, with every
per-car random stream DERIVED FROM THE AC-026 REPRODUCIBILITY TUPLE
(D1/D3, AC-026 support): car `i`'s engine draws from the xorshift32 stream
seeded by the `adaptation-<i>` surface child seed
(`deriveNeatRacingAdaptationChildSeed(tuple.seed, i)` through
`createNeatRacingXorShift32Stream`).

Each car in the fleet maintains its own adaptation state, cooldowns, and
cadence boundaries: the factory returns a `Map<number, NeatRacingAdaptationEngine>`
keyed by car index (0 to `carCount - 1`) where every engine has fully
independent closure-scoped state — no shared mutable state across cars, and
no ambient `Math.random` anywhere (D1).

Parameters:
- `tuple` - The AC-026 reproducibility tuple whose seed drives the
adaptation child-seed table.
- `carCount` - Number of cars to create engines for (non-negative integer).
- `options` - Optional engine options applied identically to every car's
engine (the `random` and `carId` fields are derived per car and cannot be
overridden here).

Returns: Map keyed by car index of independent seeded adaptation engines.

Example:

```ts
const engines = createNeatRacingPerCarAdaptationEngines(tuple, 3);
const car0Engine = engines.get(0); // adaptation-0 derived stream
const car1Engine = engines.get(1); // adaptation-1 derived stream
```

### createRuntimeAdaptationEngine

```ts
createRuntimeAdaptationEngine(
  options: NeatRacingAdaptationEngineOptions,
): NeatRacingAdaptationEngine
```

Creates a reusable per-tick adaptation engine for the NeatRacing racing
runtime loops.

The engine sequences one NGE grow-stabilize cycle per adaptation tick. It
enforces a configurable cadence policy so adaptation attempts do not fire on
every tick, applies a rollback cooldown after rejected candidates, and
respects a growth throttle that slows structural mutation as the network
grows. The growth-phase commit decision trusts the grow-stabilize cycle's
own commit flag, while the engine adds safety-limit checks and an
unconditional first-growth path so a brand-new network cannot stall.

D1 (normative): the engine REQUIRES an explicit seeded `random` stream —
usually `createNeatRacingXorShift32Stream(deriveNeatRacingAdaptationChildSeed(tupleSeed, carIndex))`
— and throws when it is omitted. The recycled ambient `Math.random` fallback
is banned sim-side and removed.

AC-043: every telemetry is stamped with the measured `tickCostMs` and
`withinTickBudget` against `NEATRACING_ADAPTATION_TICK_BUDGET_MS` (or the
`tickBudgetMs` override) through the injectable `now` clock. Budget fields
never gate decisions (replay-safe).

E10 (juvenile-only): the injected lifecycle runner throws for any stage
other than `'juvenile'`.

Parameters:
- `options` - Cadence, bounds, evaluation policy, the REQUIRED seeded
random stream, and optional budget/clock overrides.

Returns: Stateful runtime adaptation engine.

Example:

```ts
const engine = createRuntimeAdaptationEngine({
  random: createNeatRacingXorShift32Stream(
    deriveNeatRacingAdaptationChildSeed(tuple.seed, carIndex),
  ),
});
const telemetry = await engine.adaptOnTick({ tick, network, scoreHistory });
if (!telemetry.withinTickBudget) { /* eval-side cost telemetry *\/ }
```

### createTelemetry

```ts
createTelemetry(
  telemetry: Omit<NeatRacingAdaptationTelemetry, "adaptationPhase" | "stabilizationTicksSinceGrowth" | "tickCostMs" | "withinTickBudget" | "oscillationEngagement">,
  adaptationPhase: "growth" | "stabilization",
  stabilizationTicksSinceGrowth: number,
): Omit<NeatRacingAdaptationTelemetry, "tickCostMs" | "withinTickBudget" | "oscillationEngagement">
```

Stamps the phase fields onto a telemetry record (recycled helper).

### detectScoreWindowOscillation

```ts
detectScoreWindowOscillation(
  scores: readonly (number | NeatRacingQualitySignal)[],
): number
```

Detects score-window oscillation from a window of scalar scores or quality
signals.  The series is considered oscillating when the number of direction
changes (peaks or troughs) exceeds a length-scaled threshold.

Parameters:
- `scores` - Recent scores in chronological order.

Returns: Oscillation metric in [0, 1]; 0 means stable scores and values near
1 indicate strong up/down reward swings.

### detectSteeringOscillation

```ts
detectSteeringOscillation(
  angles: readonly number[],
): number
```

Detects steering oscillation from a window of steering angles.  A series is
considered oscillating when the number of sign changes exceeds a small,
length-scaled threshold and the mean steering magnitude is above a deadband,
which catches aggressive zig-zags without penalizing gentle corrections or
legitimate S-curves through chicanes.

Parameters:
- `angles` - Recent steering outputs in chronological order.

Returns: Oscillation metric in [0, 1]; 0 means smooth steering and values
near 1 indicate strong back-and-forth swings.

### evaluateRacingTrendScore

```ts
evaluateRacingTrendScore(
  network: default,
  scoreHistory: readonly (number | NeatRacingQualitySignal)[],
): number
```

Racing-specific trend evaluator that consumes a composite driving-quality
signal and accounts for network complexity via a forward pass.

Each history entry is either a legacy numeric score or a
{@link NeatRacingQualitySignal} that carries track progress, forward speed,
heading alignment, off-track penalty, and an optional physics reward.
The composite quality is folded into the same trend/mean combination used
by the default rolling-window evaluator.

The complexity bonus is computed from the variance of forward-pass outputs
across sample observations drawn from the score history.  A behaviorally-
neutral mutation (e.g. a disconnected dead-weight node) produces identical
forward-pass outputs and therefore zero variance, yielding no complexity
bonus — the evaluator correctly rejects it.  The bonus is additionally
gated on a non-negative driving-quality trend so that structural growth is
only rewarded when the car is not getting worse.

Parameters:
- `network` - Candidate network whose forward pass determines behavioral
complexity.
- `scoreHistory` - Rolling score window of numeric scores or composite
driving-quality signals.

Returns: Trend/mean score with performance-gated complexity bonus.

### evaluateRollingScoreWindow

```ts
evaluateRollingScoreWindow(
  network: default,
  scoreHistory: readonly number[],
): number
```

Lightweight default evaluator for rolling score history windows.
Recycled verbatim.

Parameters:
- `network` - Candidate network.
- `scoreHistory` - Rolling score window.

Returns: Combined trend/complexity score.

### evaluateTickOscillationEngagement

```ts
evaluateTickOscillationEngagement(
  network: default,
  evidenceWindow: readonly (number | NeatRacingQualitySignal)[],
  state: OscillationPredicateState,
  level: number,
): OscillationEngagement
```

Evaluates the SHARED AC-030 oscillation predicate for ONE adaptation tick.

The metric is the maximum of the steering oscillation and the score-window
oscillation, computed from the PRE-mutation network's forward-pass outputs
over the evidence window.  The steering deadband already lives inside
{@link detectSteeringOscillation}, so a gentle correction yields a metric of
exactly 0 and never engages the law.

Parameters:
- `network` - The PRE-mutation network of the tick.
- `evidenceWindow` - The tick's evidence window.
- `state` - The previous tick's predicate state (the hysteresis memory).
- `level` - The session level whose TP-07 node threshold gates the law.

Returns: The engagement verdict of the tick, carrying the next state.

### isCadenceReady

```ts
isCadenceReady(
  cadence: NeatRacingAdaptationCadenceOptions,
  tickInput: NeatRacingAdaptationTickInput,
  lastLapBoundary: number,
  lastSectorBoundary: number,
  storeBoundaryState: (nextLapBoundary: number, nextSectorBoundary: number) => void,
): boolean
```

Decides whether the current tick passes the cadence gate (recycled).

### NEATRACING_ADAPTATION_DEFAULT_LIMITS

Recycled default limits verbatim, published for the worker wiring and the
capacity-cap tests (the recycled `DEFAULT_LIMITS` was module-private).

### NEATRACING_ADAPTATION_MAX_CONNECTIONS

Hard connection capacity preserved from the recycled engine (AC-029).
Mirrors the NGE library's `NGE_MAX_EDGE_CAPACITY` so the adaptation limits
and the growth budget agree.

### NEATRACING_ADAPTATION_MAX_EPISODIC_SLOTS

Maximum number of episodic growth slots the NGE lifecycle may allocate
during adaptation (E10 juvenile budget). A positive value enables the
slot-expansion path; zero would silently disable growth.

### NEATRACING_ADAPTATION_MAX_NODES

Hard node capacity preserved from the recycled engine (AC-029). Mirrors the
NGE library's `NGE_MAX_NODE_CAPACITY` so the adaptation limits and the
growth budget agree.

### NEATRACING_ADAPTATION_MODULE_ID

Module identifier passed to the NGE lifecycle focus planner.

### NEATRACING_ADAPTATION_TICK_BUDGET_MS

The sim worker's per-tick adaptation budget in milliseconds (AC-029, TP-18,
AC-043). The sim-side wiring enforces it with
{@link resolveNeatRacingAdaptationTickBudget}; the engine's own telemetry
stamps carry `tickCostMs`/`withinTickBudget` against the same constant.

### NeatRacingAdaptationCadenceOptions

Cadence policy configuration for per-adaptation-tick checks.

### NeatRacingAdaptationEngine

Stateful runtime adaptation engine surface used by browser or worker loops.

### NeatRacingAdaptationEngineOptions

Engine options used by the NeatRacing runtime adaptation engine.

### NeatRacingAdaptationLimits

Hard bounds and cooldown controls for one adaptation step.

### NeatRacingAdaptationNetworkSizeSnapshot

Per-tick network size snapshot used by adaptation telemetry.

### NeatRacingAdaptationTelemetry

Adaptation telemetry emitted on every adaptation attempt.

### NeatRacingAdaptationTickBudgetState

Accounted per-tick adaptation budget state (the AC-043 sim-side fixture).

### NeatRacingAdaptationTickInput

Per-adaptation-tick input contract for adaptation checks.

### NeatRacingQualitySignal

Composite driving-quality signal used by the racing trend evaluator.
Recycled structure verbatim: spline-track progress, forward speed, heading
alignment with the track, and an off-track penalty describe how well the
car is driving on one tick.

### NeatRacingRuntimeAdaptationCadenceMode

Cadence modes supported by the runtime adaptation engine (recycled).

### NeatRacingRuntimeAdaptationOperation

Candidate mutation operations supported by the runtime adaptation engine.

### passesSafetyChecks

```ts
passesSafetyChecks(
  networkSize: NeatRacingAdaptationNetworkSizeSnapshot,
  limits: NeatRacingAdaptationLimits,
): boolean
```

Hard AC-029 capacity gate: candidate size must stay within both caps.

### RACING_COMPLEXITY_WEIGHT

Weight applied to network complexity (nodes + connections) in the racing
trend evaluator.  A small positive weight ensures the candidate (post-morph)
network scores slightly higher than the baseline (pre-morph) network when
structural mutations add capacity, allowing growth mutations to pass the
improvement threshold.  The weight is kept small so the driving-quality
trend remains the dominant signal.

### RACING_OSCILLATION_COMMIT_THRESHOLD

Oscillation gate of the additive growth-commit threshold boost: the boost is
granted only while the oscillation metric is strictly ABOVE this value.
At or below the gate the growth-commit threshold is left unchanged.

### RACING_OSCILLATION_IMPROVEMENT_THRESHOLD_BOOST

Small additive boost added to the growth-commit threshold while the shared
predicate is engaged and the oscillation metric is above
{@link RACING_OSCILLATION_COMMIT_THRESHOLD}.  The boost is scaled by the
hysteresis ramp (never a step change) and is NOT multiplied by the metric.

### RACING_OSCILLATION_MIN_MEAN_STEERING

Minimum mean absolute steering magnitude required for steering oscillation
to contribute to the oscillation metric.  Gentle corrections below this
deadband are treated as smooth steering, so legitimate S-curves through
chicanes are not penalized the same as aggressive zig-zags.

### RACING_OSCILLATION_PENALTY_WEIGHT

Weight applied to the oscillation metric when the shared predicate removes
score from a baseline or a variant (AC-030).  The penalty is MULTIPLICATIVE
and PROPORTIONAL to the score, so it damps oscillation without inverting a
positive score into a negative one.

### RACING_VARIANT_SCORER

```ts
RACING_VARIANT_SCORER(
  outputs: readonly number[][],
  target: readonly number[],
): number
```

Racing-specific variant scorer for the NGE grow-stabilize cycle.

This is a {@link VariantScorer}-compatible wrapper around
{@link scoreRacingVariant}.  The variant-scorer contract only passes
`outputs` and `target`, so the wrapper adds nothing: the AC-030
oscillation penalty is applied by the runtime engine (the shared
predicate's engagement, evaluated ONCE per adaptation tick), never inside
the scorer.

Parameters:
- `outputs` - Stack of network output vectors, one per input sample.
- `target` - Scalar target value for each sample (unused).

Returns: Positive driving-quality score (higher is better).

### reduceOutputToScalar

```ts
reduceOutputToScalar(
  outputVector: readonly number[],
): number
```

Reduce a multi-dimensional controller output vector to a scalar driving
quality proxy by averaging the absolute activation magnitudes.

This matches the reduction used by {@link buildCandidateScoreWindow} so that
the racing variant scorer and the rolling candidate scores live in the same
units.

Parameters:
- `outputVector` - Raw network output vector (e.g. `[throttle, steering]`).

Returns: Scalar proxy in the same units as the racing trend score.

### resolveAdaptiveHysteresis

```ts
resolveAdaptiveHysteresis(
  nodeCount: number,
): number
```

Resolve the adaptive hysteresis window count based on the live network
node count. Smaller networks use a lower threshold (2 consecutive
positive-quality windows) to accelerate early growth, while larger
networks require more sustained evidence (5 windows) before committing
to further structural expansion.

Parameters:
- `nodeCount` - Current total node count in the live network.

Returns: Hysteresis window count: 2 for ≤ 200 nodes, 3 for ≤ 500, 5 for > 500.

Example:

```ts
const hysteresis = resolveAdaptiveHysteresis(150);
console.log(hysteresis); // 2
```

### resolveBehavioralComplexity

```ts
resolveBehavioralComplexity(
  outputs: readonly number[][],
): number
```

Compute behavioral complexity as the total variance of forward-pass outputs
across sample observations.  If all samples produce identical outputs (e.g.
a dead-weight mutation that does not change the forward pass), the variance
is zero and the complexity bonus is correctly zero.

Parameters:
- `outputs` - Array of output vectors, one per sample observation.

Returns: Total variance across all output dimensions.

### resolveCadenceOptions

```ts
resolveCadenceOptions(
  cadence: NeatRacingAdaptationCadenceOptions | undefined,
): NeatRacingAdaptationCadenceOptions
```

Validates and normalizes the cadence options (recycled defaults).

### resolveEvidenceWindow

```ts
resolveEvidenceWindow(
  scoreHistory: readonly (number | NeatRacingQualitySignal)[],
): readonly (number | NeatRacingQualitySignal)[]
```

Filters non-finite entries out of the rolling score history.

### resolveLimitOptions

```ts
resolveLimitOptions(
  limits: Partial<NeatRacingAdaptationLimits> | undefined,
): NeatRacingAdaptationLimits
```

Validates and normalizes the capacity/cooldown limits (recycled defaults).

### resolveNeatRacingAdaptationTickBudget

```ts
resolveNeatRacingAdaptationTickBudget(
  tickStartedAtMs: number,
  nowMs: number,
  budgetMs: number,
): NeatRacingAdaptationTickBudgetState
```

Accounts one adaptation tick against the named `NEATRACING_ADAPTATION_TICK_BUDGET_MS`
budget.

This is the sim-worker wiring boundary's enforcement fixture (AC-043): the
sim worker records a tick start time, runs its per-tick adaptation work, and
calls this helper with the completion time to decide whether the residual
work stayed within budget. The engine itself stamps the same accounting into
every telemetry (`tickCostMs`, `withinTickBudget`) via its injectable clock.
Determinism note: budget fields are observability only and NEVER gate
adaptation decisions, so committed outcomes are identical across clocks
(D1 replay parity).

Parameters:
- `tickStartedAtMs` - Clock reading captured when the tick work started.
- `nowMs` - Clock reading captured when the accounting is called.
- `budgetMs` - The budget to account against; defaults to
`NEATRACING_ADAPTATION_TICK_BUDGET_MS` (2 ms).

Returns: The accounted budget state (elapsed, residual, within flag).

Example:

```ts
const startedAtMs = Date.now();
runSimSideAdaptationWork();
const budget = resolveNeatRacingAdaptationTickBudget(startedAtMs, Date.now());
if (!budget.withinBudget) { /* residual work exceeded 2 ms *\/ }
```

### resolveNetworkSizeSnapshot

```ts
resolveNetworkSizeSnapshot(
  network: default,
): NeatRacingAdaptationNetworkSizeSnapshot
```

Snapshots the live network size for telemetry.

### resolveObservationVector

```ts
resolveObservationVector(
  entry: number | NeatRacingQualitySignal,
  inputSize: number,
): number[]
```

Build an observation vector of the given size from a single score history
entry.  Numeric entries are repeated to fill the input; composite signals
tile their five fields (or truncate) to match the network input dimension.

Parameters:
- `entry` - Numeric score or composite driving-quality signal.
- `inputSize` - Number of input nodes in the candidate network.

Returns: Input vector suitable for `network.activate`.

### resolveSampleIndices

```ts
resolveSampleIndices(
  historyLength: number,
  maxSamples: number,
): number[]
```

Resolve evenly-spaced sample indices from the score history.

Parameters:
- `historyLength` - Total number of entries in the history.
- `maxSamples` - Maximum number of samples to select.

Returns: Array of indices into the score history.

### scoreRacingVariant

```ts
scoreRacingVariant(
  outputs: readonly number[][],
  _target: readonly number[],
): number
```

Internal racing variant scorer with an explicit neuron-count gate.

The network emits a 2-D controller vector (`[throttle, steering]`), but the
grow-stabilize evaluator expects the baseline and variant scores to share the
same positive driving-quality score space. This scorer mirrors the
{@link evaluateRacingTrendScore} baseline computation: it collapses each
output row with {@link reduceOutputToScalar}, then combines the mean trend
of the resulting scalar window with a behavioral-complexity bonus (gated on a
non-negative trend). Higher scores mean better driving quality, so a variant
can win the commit decision when it genuinely outperforms the baseline.

The `target` argument mirrors the {@link VariantScorer} contract but is
intentionally not used here; the score is derived from the candidate
network's own forward-pass outputs so that it lives in the same space as
the pre-mutation baseline.

The scorer is PURE: it carries NO oscillation penalty of its own.  The
AC-030 law lives in the shared predicate
(`./oscillation-penalty.utils`), whose engagement the runtime engine
evaluates ONCE per adaptation tick and applies symmetrically to the
baseline and the variant score.  A second penalty inside the scorer would
double-penalize the live path and mix two score spaces.

Parameters:
- `outputs` - Stack of network output vectors, one per input sample.
- `_target` - Scalar target value for each sample (unused).

Returns: Positive racing-trend quality score (higher is better).

### toDrivingQuality

```ts
toDrivingQuality(
  entry: number | NeatRacingQualitySignal,
): number
```

Convert one history entry into a scalar driving-quality score.

Legacy numeric entries pass through unchanged so existing callers and the
default engine can keep using raw score windows.  Composite signals are
weighted so that better progress, speed, and alignment increase the score,
while a larger off-track penalty decreases it.

Parameters:
- `entry` - Numeric score or composite driving-quality signal.

Returns: Scalar quality value for trend/mean scoring.

## ai/scripted.controller.ts

The deterministic scripted waypoint-rivals controller for the NeatRacing
voxel racer (`examples/neatRacing/ai` — Step 02, slice P2-S2-C).

AC-028 (spec §"AC Register"): "a deterministic scripted controller with a
per-course speed/throttle table; throttle constant 0.82 + proportional
steer". The waypoint advancement, the proportional steer term and the
throttle constant are RECYCLED from
`examples/racing_curriculum/controller/scripted.controller.ts` with every
recycled scale mirrored owner-local VERBATIM (advance radius 16, lookahead
4 samples, target projection 14, throttle constant 0.82, steer gain 2.2 —
the shared 2.2 rad⁻¹ heading-error-per-gain convention of
BOUNDARY-2). The NEW (E7, NOT recycled) functionality is the per-course
throttle table: a deterministic, segment-keyed throttle map derived once
per course layout from the published racing-line target speeds, floored by
a drivability bound. The controller holds throttle at the table entry for
the road segment the car is DRIVING and falls back to the recycled
constant for an empty course — this is not new intelligence, just
deterministic throttle bookkeeping derived from geometry this module does
not own. THE TURN CHANNEL (the round-2 physics finding): a PRESENT
`turnRatePercent` is AUTHORITATIVE in the NeatRacing physics (Step 14T/
AC-092-AC-093 — a present `0` commands EXACTLY zero turn), so this
controller emits NO turn-rate percent at all and the steer axis — the
recycled clamped demand — is the SOLE turn channel, resolved by the
physics steer-axis law (recycled parity: the recycled demand rides the
steer axis).

THE ONE SOURCE OF TRUTH (AC-086, LD-30 "track-relative features only"):
the throttle table is derived from the PUBLISHED racing-line table
(`resolveAutopilotRacingLine`) — the same memoized table the autopilot
drives on and the observation assembler reads. There is NO second geometry
and NO duplicated curvature math anywhere in this module; the table is a
pure deterministic function of the course's spline samples (the same
samples always produce the identical table) keyed by each sample's owning
chord `segmentIndex` metadata — no spline-density assumption.

THE REALM: pure headless seam — no self, no postMessage, no DOM, no RNG,
no clock reads. `computeNeatRacingScriptedControl` mutates ONLY the
caller-owned controller state (`targetSegmentIndex` advances in place, the
recycled mutation seam) and never the car snapshot or the track spec.

### computeNeatRacingScriptedControl

```ts
computeNeatRacingScriptedControl(
  car: NeatRacingScriptedCarSnapshot,
  trackSpec: TrackSpec,
  controllerState: NeatRacingScriptedControllerState,
): PhysicsStepInput
```

Produces the control output that steers a scripted rival toward its next
track waypoint — the NeatRacing port of the recycled
`computeScriptedControl`.

Algorithm (recycled waypoint plan preserved step-for-step):
1. Check whether the car is within `WAYPOINT_ADVANCE_RADIUS_M` of the
   current lane-center spline target — if so, advance to the next segment
   (modulo segment count); otherwise hold the (wrap-normalized) target.
2. Compute the wrapped heading error from the car's heading to the angle
   toward the lane-centered spline target — the recycled aim point of the
   CLOSEST spline sample (its inner-lane centerline projected forward
   along its own tangent).
3. Apply the proportional gain and clamp to [-1, 1] for the steer output.
4. Throttle reads the per-course table (AC-028) for the road segment the
   car is DRIVING — the closest sample's owning `segmentIndex` — falling
   back to the recycled constant for an empty course.

Mutates ONLY `controllerState.targetSegmentIndex` in place (the recycled
mutation seam); the car snapshot and the frozen track spec are never
touched.

Parameters:
- `car` - Current scripted-rival car snapshot (x, y, heading).
- `trackSpec` - Frozen track geometry (segment count + spline samples).
- `controllerState` - Mutable controller state (target index advances in place).

Returns: Physics step input for the next simulation step (no brake —
the scripted baseline is a throttle-only lane; NO turn-rate percent is
emitted because a PRESENT percent is authoritative in the physics —
`0` would command exactly zero turn — so the steer axis is the sole
turn channel).

Example:

```ts
const state = createNeatRacingScriptedControllerState();
const input = computeNeatRacingScriptedControl(car, trackSpec, state);
```

### createNeatRacingScriptedControllerState

```ts
createNeatRacingScriptedControllerState(): NeatRacingScriptedControllerState
```

Creates a fresh scripted controller state targeting the first segment.

Returns: Fresh controller state with `targetSegmentIndex` at 0.

Example:

```ts
const state = createNeatRacingScriptedControllerState();
const output = computeNeatRacingScriptedControl(car, trackSpec, state);
```

### deriveThrottleProfile

```ts
deriveThrottleProfile(
  splineSamples: readonly SplineSample[],
): ReadonlyMap<number, number>
```

Derives the per-course throttle table from the PUBLISHED racing line.

THE ONE SOURCE OF TRUTH: the published racing-line table
(`resolveAutopilotRacingLine`) is read per sample `globalIndex`; each
segment's binding speed is the minimum published target speed over the
segment's samples; the entry scales the recycled constant with the
top-speed-normalized factor clamped to the drivability floor. A sample
missing from the published table defaults to the physics top speed (the
constant-throttle behavior of a straight).

### findClosestSplineSampleIndex

```ts
findClosestSplineSampleIndex(
  car: NeatRacingScriptedCarSnapshot,
  splineSamples: readonly SplineSample[],
): number
```

Finds the closest lane-center spline sample for a car snapshot.

Mirrors the recycled nearest-scan exactly: linear scan over the sample
list, strict `<` distance comparisons (the FIRST best sample wins ties),
returning the sample's `globalIndex`.

### NEATRACING_SCRIPTED_PROFILE_MIN_FACTOR

Lower bound on the per-course throttle factor. The throttle axis is the
baseline rivals' ONLY forward-acceleration scale (the lane never brakes),
so the floor keeps the weakest corner segment away from a near-zero
command and the scripted plan keeps forward progress everywhere.

### NEATRACING_SCRIPTED_STEER_PROPORTIONAL_GAIN

Proportional gain applied to the wrapped heading-error term to compute
steer. A gain of 2.2 saturates full steer at approximately 26 degrees of
heading error. Recycled value verbatim (`STEER_PROPORTIONAL_GAIN`) and the
SAME rad⁻¹ convention as `NEATRACING_AUTOPILOT_STEER_GAIN_PER_RAD`
(BOUNDARY-2: one gain convention across NeatRacing controllers).

### NEATRACING_SCRIPTED_TARGET_LOOKAHEAD_SAMPLE_COUNT

Number of spline samples to look ahead within the active target segment.
Recycled value verbatim (`SCRIPTED_TARGET_LOOKAHEAD_SAMPLE_COUNT`).

### NEATRACING_SCRIPTED_TARGET_PROJECTION_DISTANCE_M

Forward tangent projection used to keep the scripted target lane-centered.
Recycled value verbatim (`SCRIPTED_TARGET_PROJECTION_DISTANCE_WORLD`).

### NEATRACING_SCRIPTED_THROTTLE_CONSTANT

Constant forward throttle scale of the scripted baseline — the throttle
ceiling and the empty-course fallback. Recycled value verbatim
(`SCRIPTED_THROTTLE_CONSTANT`, AC-028 pins the value 0.82).

### NEATRACING_SCRIPTED_WAYPOINT_ADVANCE_RADIUS_M

Distance (m) within which the car advances to the next waypoint. Set to
roughly 10% of a typical segment length for smooth progression.
Recycled value verbatim (`WAYPOINT_ADVANCE_RADIUS_WORLD`).

### NeatRacingScriptedCarSnapshot

The car snapshot the scripted controller reads — the same Pick shape as the
autopilot's `AutopilotCarSnapshot` minus speed (the recycled waypoint
baseline does not consume speed).

### NeatRacingScriptedControllerState

Persistent state for the scripted waypoint-following controller.

The `targetSegmentIndex` advances monotonically (modulo segment count) as
the car reaches successive waypoints around the closed loop — the recycled
mutation seam, preserved.

### resolveInnerLaneCenterlineOffsetWorld

```ts
resolveInnerLaneCenterlineOffsetWorld(
  splineSample: SplineSample,
): number
```

Resolves the lateral distance from the road centerline to the inner-lane
centerline for one sample (recycled helper semantics verbatim).

### resolveInnerLaneCenterlinePoint

```ts
resolveInnerLaneCenterlinePoint(
  splineSample: SplineSample,
  splineSampleFrame: { readonly tangentHeadingRadians: number; readonly normalX: number; readonly normalY: number; },
): { readonly x: number; readonly y: number; }
```

Resolves the inner-lane centerline point for one sample (recycled verbatim).

### resolveNeatRacingScriptedThrottleProfile

```ts
resolveNeatRacingScriptedThrottleProfile(
  splineSamples: readonly SplineSample[],
): ReadonlyMap<number, number>
```

THE PER-COURSE THROTTLE TABLE (AC-028, E7): resolves the deterministic,
segment-keyed throttle map for one course layout.

For every segment the table entry is the recycled throttle constant scaled
by the segment's BINDING speed factor — the minimum published racing-line
target speed over the segment's spline samples (the tightest corner within
the segment is the constraint) divided by the physics top speed, clamped to
`[NEATRACING_SCRIPTED_PROFILE_MIN_FACTOR, 1]`. Straight segments therefore
publish exactly the recycled constant; corner segments publish proportionally
less, floored for drivability.

THE DETERMINISM CONTRACT (drivability gate, AC-031): the table is a pure
deterministic function of the course's spline samples — the same sample
layout always derives the identical entries — so a scripted-baseline probe
episode on the same track tuple reproduces the same throttle plan every
run. The result is memoized per sample-list identity; distinct layout bytes
get distinct tables (different courses behave differently).

Parameters:
- `splineSamples` - The course's frozen lane-center spline samples.

Returns: Readonly map of `segmentIndex` → throttle in
`[0.35 × 0.82, 0.82]` (empty map for an empty course).

Example:

```ts
const profile = resolveNeatRacingScriptedThrottleProfile(trackSpec.splineSamples);
const segmentThrottle = profile.get(0) ?? NEATRACING_SCRIPTED_THROTTLE_CONSTANT;
```

### resolveProjectedTargetPoint

```ts
resolveProjectedTargetPoint(
  targetSplineSample: SplineSample,
  splineSamples: readonly SplineSample[],
): { readonly x: number; readonly y: number; }
```

Projects the lane-centered target forward along the target sample's tangent
by the recycled projection distance (recycled `resolveProjectedTargetPoint`).

### resolveSampleAt

```ts
resolveSampleAt(
  splineSamples: readonly SplineSample[],
  sampleIndex: number,
): SplineSample | undefined
```

Resolves the spline sample at a wrapped `globalIndex`, empty-safe.

### resolveSegmentStripMap

```ts
resolveSegmentStripMap(
  splineSamples: readonly SplineSample[],
): ReadonlyMap<number, readonly number[]>
```

Resolves the per-segment ordered sample strip map for a course, memoized by
the sample-list identity. Each strip carries the `globalIndex` values of
the samples owned by one chord `segmentIndex`, in list (sample) order —
the same order the published racing-line table uses.

THE INDEX-SPACE INVARIANT (recycled parity): the recycled controller reads
`trackSpec.splineSamples[globalIndex]` directly, i.e. globalIndex doubles
as the array position for the ordered production layouts the repo freezes
(`resolveSampleAt` applies exactly that recycled wrap). The strips carry
`globalIndex` so the waypoint walk resolves through the same index space
the recycled plan walks.

### resolveSplineSampleFrame

```ts
resolveSplineSampleFrame(
  splineSamples: readonly SplineSample[],
  sampleIndex: number,
): { readonly tangentHeadingRadians: number; readonly normalX: number; readonly normalY: number; }
```

Owner-local mirror of the recycled `SplineSampleFrame` (tangent + left normal).

### resolveTargetSplineSample

```ts
resolveTargetSplineSample(
  splineSamples: readonly SplineSample[],
  stripMap: ReadonlyMap<number, readonly number[]>,
  closestSplineSample: SplineSample,
  targetSegmentIndex: number,
): SplineSample
```

Resolves the target spline sample for the waypoint plan — the owner-local
port of the recycled `resolveTargetSplineSample` without the density-18
assumption: when the closest sample already sits in the target segment the
target walks the segment's OWN strip `LOOKAHEAD` samples ahead (capped at
the strip end); otherwise the target is the midpoint sample of the target
segment's strip.

### resolveWrappedSegmentIndex

```ts
resolveWrappedSegmentIndex(
  targetSegmentIndex: number,
  segmentCount: number,
): number
```

Wraps a target segment index into `[0, segmentCount)` (recycled verbatim).

### wrapAngleToMinusPiPi

```ts
wrapAngleToMinusPiPi(
  angleRadians: number,
): number
```

Wraps an angle in radians to the range [-π, π] (recycled helper verbatim).

### wrapSampleIndex

```ts
wrapSampleIndex(
  sampleCount: number,
  sampleIndex: number,
): number
```

Wraps a sample index into the sample list (recycled `wrapSplineSampleIndex`).

## ai/observation.assembler.ts

NeatRacing observation assembler (slice `P2-S2-B`, AC-027).

THE VECTOR: the deterministic per-car observation vector of the
NeatRacing voxel racer. AC-027 pins THREE prefixes of the SAME channel
layout, one per ladder level, so the NGE genotype/phenotype controller
of Step 03 can grow without any channel ever moving:

- Tier A (level 1) = 82 channels: base 70 + wheel slip 4 + rival gaps 8
- Tier B (level 2) = 89 channels: Tier A + self radio 7
- Tier C (level 3) = 110 channels: Tier B + teammate radio 3 x 7 = 21

THE PREFIX (the whole point): the base channels stay `[0..69]`, the
wheel-slip block stays `[70..73]`, the rival-gap block stays
`[74..81]`, the self-radio block stays `[82..88]` and the
teammate-radio block stays `[89..109]` on EVERY level. A Tier A
network's input space is a strict slice of Tier B's and Tier C's.

THE MIRROR: the base-70 block is the owner-local mirror of the
`racing_curriculum` observation vocabulary — the same 20 scalar,
40 look-ahead and 10 memory-trace channels with the SAME recycled
§11.1 scales, re-expressed for `PhysicsCarState` without a single
cross-example runtime import (only the shared frozen `TrackSpec` /
`SplineSample` TYPES flow in, exactly like `physics/ai.autopilot.ts`).

THE ONE SOURCE OF TRUTH (AC-086): the optimal-line lateral offset and
the curvature-derived target-speed channels are derived from the
PUBLISHED racing-line table (`resolveAutopilotRacingLine`), the same
memoized table the autopilot drives on. There is NO second geometry
anywhere in the repo.

THE WHEEL SLIP (LD-15): the physics publishes ONE car-level
`yawRate`/`slipAngle` pair, so the four per-wheel channels are the
DETERMINISTIC SIDE PROJECTION of that pair (see
`collectNeatRacingWheelSlipChannels` for the exact projection the
fixture pins).

THE REALM: this module is a PURE headless seam — no `self`, no
`postMessage`, no DOM, no RNG, no wall-clock reads. The same (state,
track, level) triple yields a byte-identical vector for the batch-83
replay contract.

### assembleNeatRacingObservationVector

```ts
assembleNeatRacingObservationVector(
  envState: NeatRacingObservationState,
  trackSpec: TrackSpec,
  options: NeatRacingObservationAssemblerOptions,
): Float32Array<ArrayBufferLike>
```

Assembles the deterministic NeatRacing observation vector of one focal car.

THE CONTRACT (AC-027): the returned vector is a prefix-preserving
Float32Array whose length is pinned by the ladder level — 82 (Tier A),
89 (Tier B) or 110 (Tier C) channels:

- `[0..69]` the recycled base block (20 scalars + 5 x 8 look-ahead + 10
  memory-trace) with the §11.1 scales reused verbatim;
- `[70..73]` the per-wheel slip channels derived from the LD-15 pair;
- `[74..81]` the four 2-channel rival-gap slots (body-frame longitudinal
  and lateral gap, roster order, zero-padded beyond the roster);
- `[82..88]` the raw self-radio tail (level 2 and 3);
- `[89..109]` the three 7-channel teammate radio slots (level 3 only),
  zero-padded beyond the same-team roster.

Parameters:
- `envState` - Focal car observation seam state (see `NeatRacingObservationState`).
- `trackSpec` - Frozen track geometry (the shared `TrackSpec` contract).
- `options` - The ladder level whose tail the vector must carry.

Returns: The normalized observation vector.

Example:

```ts
const vector = assembleNeatRacingObservationVector(
  { carX: 2, carY: 0, carHeading: 0, forwardSpeedWorld: 10 },
  trackSpec,
  { level: 3 },
);
vector.length; // 110
Array.from(vector.slice(0, 82)); // the Tier A prefix
```

### buildNeatRacingRivalGapSlot

```ts
buildNeatRacingRivalGapSlot(
  rival: NeatRacingObservationCarState,
  envState: NeatRacingObservationState,
): Float32Array<ArrayBufferLike>
```

Builds the two-channel body-frame gap slot of one rival.

Parameters:
- `rival` - The rival car state to encode.
- `envState` - Focal car observation seam state (the body-frame origin).

Returns: The two-channel slot `[longitudinal / 96, lateral / 96]`.

### buildNeatRacingTeammateSlot

```ts
buildNeatRacingTeammateSlot(
  teammate: NeatRacingObservationCarState,
  envState: NeatRacingObservationState,
): Float32Array<ArrayBufferLike>
```

Encodes one teammate's state into the recycled 7-channel radio slot.

Channel layout (recycled vocabulary verbatim):
`[posX / 96, posY / 96, sin(heading), speed / 108,
  relOffsetX / 96, relOffsetY / 96, sin(relHeading)]`.

Parameters:
- `teammate` - The teammate car state to encode.
- `envState` - Focal car observation seam state (the relative origin).

Returns: The seven-channel slot with the recycled normalization scales.

### clamp01NeatRacing

```ts
clamp01NeatRacing(
  value: number,
): number
```

Clamps a probability-like scalar to `[0, 1]`.

Parameters:
- `value` - Source scalar.

Returns: Value clamped to `[0, 1]` (non-finite input reads as 0).

### clampNeatRacingNormalizedValue

```ts
clampNeatRacingNormalizedValue(
  value: number,
): number
```

Clamps any scalar into the controller's accepted normalized range.

Parameters:
- `value` - Source scalar.

Returns: Value clamped to `[-1, 1]` (non-finite input reads as 0).

### collectNeatRacingBaseChannels

```ts
collectNeatRacingBaseChannels(
  observationContext: NeatRacingObservationContext,
): Float32Array<ArrayBufferLike>
```

Collects the 70 base channels (scalars + look-ahead + memory trace).

Parameters:
- `observationContext` - Resolved assembly context.

Returns: The normalized base block.

### collectNeatRacingLookAheadChannels

```ts
collectNeatRacingLookAheadChannels(
  observationContext: NeatRacingObservationContext,
): Float32Array<ArrayBufferLike>
```

Collects the 40 look-ahead channels used for track anticipation.

Slice `i` (0-based) reads the spline sample `closest + i x 18` (wrapped
around the closed ring) plus its neighbor, mirroring the recycled
per-segment layout: the sample and next-sample offsets to the car (96),
the tangent sin/cos, the lane width (24) and the car-to-sample gap (64).

Parameters:
- `observationContext` - Resolved assembly context.

Returns: Forty normalized look-ahead channels.

### collectNeatRacingMemoryTraceChannels

```ts
collectNeatRacingMemoryTraceChannels(
  observationContext: NeatRacingObservationContext,
): Float32Array<ArrayBufferLike>
```

Collects the 10-channel recurrent memory trace.

Parameters:
- `observationContext` - Resolved assembly context.

Returns: Ten normalized memory-trace channels (zeros when absent).

### collectNeatRacingRivalGapChannels

```ts
collectNeatRacingRivalGapChannels(
  envState: NeatRacingObservationState,
): Float32Array<ArrayBufferLike>
```

Collects the four 2-channel rival-gap slots (AC-027).

THE ENCODING: slot `s` channels are the body-frame gap of the `s`-th
tracked rival — `[longitudinal / 96, lateral / 96]`, where longitudinal
is positive AHEAD and lateral is positive LEFT of the focal heading.
Rivals are the opposing-team roster cars in deterministic roster order
(the recycled roster contract); missing slots stay zero-padded. The
position delta scale (96) mirrors the recycled body-frame offsets.

Parameters:
- `envState` - Focal car observation seam state (carries the roster).

Returns: Eight normalized rival-gap channels.

### collectNeatRacingScalarChannels

```ts
collectNeatRacingScalarChannels(
  observationContext: NeatRacingObservationContext,
): Float32Array<ArrayBufferLike>
```

Collects the 20 scalar channels that describe the current car state.

The channel ORDER and the §11.1 scales mirror the recycled vocabulary
verbatim: position (96), heading (sin/cos), body-frame speeds (108/54),
speed (108), yaw rate (1), slip angle (pi/2), progress probabilities,
boundary distances (24) and balance, hazard/waypoint distances (64),
optimal-line offset (18) and heading error (pi), target speed (108) and
the target-speed margin (108).

Parameters:
- `observationContext` - Resolved assembly context.

Returns: Twenty normalized scalar channels.

### collectNeatRacingSelfRadioChannels

```ts
collectNeatRacingSelfRadioChannels(
  radioField: readonly number[],
): Float32Array<ArrayBufferLike>
```

Collects the raw seven-channel self-radio tail (AC-027 Tier B).

THE TAIL: the recycled Tier-2 seam passes the provider-owned radio field
through WITHOUT renormalization (radio payloads are already bounded);
missing channels stay zero.

Parameters:
- `radioField` - Current self-radio field from the seam state.

Returns: The seven-channel raw radio tail.

### collectNeatRacingTeammateRadioChannels

```ts
collectNeatRacingTeammateRadioChannels(
  envState: NeatRacingObservationState,
): Float32Array<ArrayBufferLike>
```

Collects the 21-channel teammate radio tail (AC-027 Tier C).

THE SLOTS: up to three same-team roster cars (roster order, focal car
excluded) encode into the recycled 7-channel teammate slot; the remaining
slots stay zero-padded. The NeatRacing ladder teams are size 1 (team A)
or size 4 (team B), so the recycled self-broadcast rule for exactly
3-car teams is unreachable here and is not mirrored.

Parameters:
- `envState` - Focal car observation seam state (carries the roster).

Returns: The 21 normalized teammate radio channels.

### collectNeatRacingWheelSlipChannels

```ts
collectNeatRacingWheelSlipChannels(
  resolvedObservationState: ResolvedNeatRacingObservationState,
): Float32Array<ArrayBufferLike>
```

Collects the four per-wheel slip channels (AC-027, LD-15).

THE DERIVATION: NeatRacing publishes ONE car-level `slipAngle` and ONE
car-level `yawRate` (LD-15), so the per-wheel indicator is the DETERMINED
side projection of that shared pair rather than an invented fourth-scale
physical model:

1. the slip contributor normalizes the published slip angle by its
   recycled saturation scale (`pi/2` -> 1.0);
2. the yaw contributor normalizes the yaw rate by its recycled scale
   (1 rad/s -> 1.0);
3. the mean of the two becomes the slide intensity;
4. the wheel sides read it with opposing polarity (left +1, right -1)
   so opposite wheels see opposite signs of the same slide.

A straight drive keeps every channel exactly 0.

Parameters:
- `resolvedObservationState` - Fully resolved observation state.

Returns: Four normalized wheel-slip channels (FL, FR, RL, RR).

### createNeatRacingObservationContext

```ts
createNeatRacingObservationContext(
  envState: NeatRacingObservationState,
  trackSpec: TrackSpec,
): NeatRacingObservationContext
```

Builds the assembly context (nearest sample, track bounds, resolved state).

Parameters:
- `envState` - Focal car observation seam state.
- `trackSpec` - Frozen track geometry.

Returns: The resolved assembly context.

### findClosestNeatRacingSplineSampleIndex

```ts
findClosestNeatRacingSplineSampleIndex(
  envState: NeatRacingObservationState,
  trackSpec: TrackSpec,
): number
```

Finds the spline sample whose lane-center point is closest to the car.

Parameters:
- `envState` - Focal car observation seam state.
- `trackSpec` - Frozen track geometry.

Returns: Global index of the nearest spline sample.

### NEATRACING_OBSERVATION_BASE_CHANNEL_COUNT

Base channel count shared by every tier (the recycled owner-local block).

### NEATRACING_OBSERVATION_BOUNDARY_DISTANCE_WORLD_SCALE

Boundary distance and lane-width normalization scale (24 units -> 1.0).

### NEATRACING_OBSERVATION_CHANNELS_PER_LOOKAHEAD_SEGMENT

Per-segment look-ahead channel count (position, tail, tangent, width, gap).

### NEATRACING_OBSERVATION_CHANNELS_PER_TEAMMATE_SLOT

Per-teammate slot channel count (the recycled 7-channel radio slot).

### NEATRACING_OBSERVATION_DEFAULT_TARGET_SPEED_WORLD

Fallback curvature target speed when the racing-line table is absent.

### NEATRACING_OBSERVATION_DISTANCE_WORLD_SCALE

Distance normalization scale (64 world units -> 1.0).

### NEATRACING_OBSERVATION_LATERAL_SPEED_WORLD_SCALE

Lateral-speed normalization scale (54 world units/s -> 1.0).

### NEATRACING_OBSERVATION_LOOKAHEAD_CHANNEL_COUNT

Look-ahead channel count of the base block (5 segments x 8 channels).

### NEATRACING_OBSERVATION_LOOKAHEAD_SAMPLE_STRIDE

Look-ahead SAMPLE STRIDE between consecutive look-ahead slices.

The recycled vocabulary strides 18 samples per look-ahead segment — the
shared spline sample density — so slice `i` reads the sample
`closest + i x 18` (wrapped around the closed ring). The constant mirrors
that stride as a NeatRacing-owned number instead of importing the shared
runtime constant.

### NEATRACING_OBSERVATION_LOOKAHEAD_SEGMENT_COUNT

Look-ahead segment slice count of the base block.

### NEATRACING_OBSERVATION_MEMORY_TRACE_CHANNEL_COUNT

Recurrent memory-trace channel count of the base block.

### NEATRACING_OBSERVATION_OPTIMAL_LINE_LATERAL_OFFSET_WORLD_SCALE

Optimal-line lateral-offset normalization scale (18 units -> 1.0).

### NEATRACING_OBSERVATION_RIVAL_GAP_CHANNEL_COUNT

Rival-gap channel count appended after the wheel-slip block (AC-027).

### NEATRACING_OBSERVATION_RIVAL_GAP_CHANNELS_PER_SLOT

Per-rival gap channel count: [longitudinal body gap, lateral body gap].

### NEATRACING_OBSERVATION_RIVAL_GAP_SLOT_COUNT

Rival slot capacity covered by the 8 rival-gap channels.

AC-027: the 8 rival-gap channels cover the 4-rival case of the level 3
ladder (`[0, 1, 1, 1, 1]`), so the block holds FOUR slots of TWO
channels (at least 4 rival slots at level 3).

### NEATRACING_OBSERVATION_SCALAR_CHANNEL_COUNT

Scalar channel count of the base block (car state, tier and lane info).

### NEATRACING_OBSERVATION_SELF_RADIO_CHANNEL_COUNT

Self-radio channel count of the Tier B tail (raw recycled radio field).

### NEATRACING_OBSERVATION_SPEED_WORLD_SCALE

Speed normalization scale (108 world units/s -> 1.0).

### NEATRACING_OBSERVATION_TEAMMATE_RADIO_CHANNEL_COUNT

Teammate radio channel count of the Tier C tail (3 slots x 7 channels).

### NEATRACING_OBSERVATION_TEAMMATE_RADIO_SLOT_COUNT

Teammate radio slot capacity of the Tier C tail.

### NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT

Tier A (level 1) vector length: 70 + 4 + 8.

### NEATRACING_OBSERVATION_TIER_B_CHANNEL_COUNT

Tier B (level 2) vector length: 82 + 7.

### NEATRACING_OBSERVATION_TIER_C_CHANNEL_COUNT

Tier C (level 3) vector length: 89 + 21.

### NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE

World-position normalization scale (96 world units -> 1.0).

### NEATRACING_OBSERVATION_WHEEL_SLIP_CHANNEL_COUNT

Per-wheel slip channel count appended after the base block (AC-027).

### NEATRACING_OBSERVATION_WHEEL_SLIP_SLIP_ANGLE_SCALE

Slip-angle saturation scale of the wheel-slip derivation (LD-15).

`PhysicsCarState.slipAngle` is a signed angle in radians bounded to
`(-pi/2, pi/2)`, so this recycled saturation angle maps the published
slip onto the unit interval. Reused UNCHANGED from the recycled scalar
block (`slipAngle / (pi / 2)`), per the §11.1 no-renormalization clause.

### NEATRACING_OBSERVATION_WHEEL_SLIP_YAW_RATE_SCALE

Yaw-rate scale of the wheel-slip derivation (LD-15).

The wheel-slip contributor consumes `yawRateRadiansPerSecond` with the
recycled yaw-rate scale (1 rad/s -> 1.0), so the derivation reuses the
SAME normalization the recycled scalar channel 7 uses.

### NEATRACING_OBSERVATION_YAW_RATE_RADIANS_PER_SECOND_SCALE

Yaw-rate normalization scale (1 rad/s -> 1.0).

### NEATRACING_TEAM_LAYOUTS

The E2 team layouts of the 1/2/5-car ladder, pinned exactly as specified.

- level 1 (`[0]`): the solo rung — one car, no rivals, no teammates.
- level 2 (`[0, 1]`): two entries (team A vs team B).
- level 3 (`[0, 1, 1, 1, 1]`): the human/team A car 0 and FOUR team B
  rivals — the case the 8 rival-gap channels were sized for.

### NeatRacingObservationAssemblerOptions

Options selecting the ladder level whose tail the vector must carry.

### NeatRacingObservationCarState

One roster car as the seam consumes it for the rival and teammate tails.

The vocabulary mirrors the recycled per-car state: world pose, optional
speed and the team index. The roster ARRAY INDEX is the car index, so
the focal car is excluded by its array position exactly like the
recycled roster derivation.

### NeatRacingObservationContext

Assembly context threaded through the collector helpers.

### NeatRacingObservationLevel

NeatRacing observation level (the ladder rung that selects the tail).

The spec E2 mapping is level -> tier: level 1 -> Tier A (82 channels),
level 2 -> Tier B (89 channels) and level 3 -> Tier C (110 channels).

### NeatRacingObservationState

The observation seam state of ONE focal car.

Required fields cover the `PhysicsCarState` kinematics the vector needs;
every optional field mirrors a recycled observation extension and falls
back to its deterministic derived default when absent (see
`resolveNeatRacingObservationState`). The seam keeps the recycled field
names (`carX`, `carHeading`, `radioField`, `memoryTrace`, ...) so the
Step 03 controller wiring needs no translation layer, while remaining a
NeatRacing-owned type with NO cross-example import.

### NeatRacingSplineSampleFrame

Spline sample frame (tangent + left normal) mirrored owner-locally.

### NeatRacingTrackBounds

Bounding box + center used by the position channels (recycled contract).

### normalizeNeatRacingProbability

```ts
normalizeNeatRacingProbability(
  probabilityValue: number,
): number
```

Converts a probability-style scalar in `[0, 1]` to the controller's
symmetric `[-1, 1]` range.

Parameters:
- `probabilityValue` - Probability-like scalar.

Returns: Symmetric normalized value.

### normalizeNeatRacingSignedValue

```ts
normalizeNeatRacingSignedValue(
  value: number,
  scale: number,
): number
```

Converts a signed scalar into the normalized `[-1, 1]` range.

Parameters:
- `value` - Signed source scalar.
- `scale` - Absolute scale corresponding to magnitude 1.

Returns: Clamped normalized value.

### normalizeNeatRacingUnsignedValue

```ts
normalizeNeatRacingUnsignedValue(
  value: number,
  scale: number,
): number
```

Converts an unsigned scalar into the normalized `[0, 1]` range.

Parameters:
- `value` - Unsigned source scalar.
- `scale` - Maximum reference scale for value 1.

Returns: Normalized value in `[0, 1]`.

### ResolvedNeatRacingObservationState

Fully resolved observation state (no optional field left unresolved).

### resolveNeatRacingBoundaryBalance

```ts
resolveNeatRacingBoundaryBalance(
  boundaryDistanceLeftWorld: number,
  boundaryDistanceRightWorld: number,
): number
```

Computes the signed left-versus-right lane balance.

Parameters:
- `boundaryDistanceLeftWorld` - Distance to the left boundary.
- `boundaryDistanceRightWorld` - Distance to the right boundary.

Returns: Symmetric balance in `[-1, 1]` (positive means right of center).

### resolveNeatRacingFocalTeamIndex

```ts
resolveNeatRacingFocalTeamIndex(
  envState: NeatRacingObservationState,
): 0 | 1
```

Resolves the focal team index from the seam (override, roster entry, then 0).

Parameters:
- `envState` - Focal car observation seam state.

Returns: The focal team index (`0` or `1`).

### resolveNeatRacingLookAheadSample

```ts
resolveNeatRacingLookAheadSample(
  observationContext: NeatRacingObservationContext,
  lookAheadOffset: number,
): SplineSample
```

Resolves one look-ahead spline sample by wrapping around the closed ring.

Parameters:
- `observationContext` - Resolved assembly context.
- `lookAheadOffset` - Zero-based look-ahead slice offset.

Returns: The spline sample encoding into that slice.

### resolveNeatRacingMemoryTrace

```ts
resolveNeatRacingMemoryTrace(
  memoryTrace: readonly number[] | undefined,
): readonly number[]
```

Resolves the recurrent memory trace while preserving a fixed width.

### resolveNeatRacingObservationChannelCount

```ts
resolveNeatRacingObservationChannelCount(
  level: NeatRacingObservationLevel,
): number
```

Resolves the observation channel count of one ladder level.

Parameters:
- `level` - Ladder level (1, 2 or 3) to measure.

Returns: The pinned channel count: 82 for level 1, 89 for level 2 and 110 for level 3.

Example:

```ts
resolveNeatRacingObservationChannelCount(1); // 82
resolveNeatRacingObservationChannelCount(3); // 110
```

### resolveNeatRacingObservationState

```ts
resolveNeatRacingObservationState(
  envState: NeatRacingObservationState,
  trackSpec: TrackSpec,
  closestSplineSampleIndex: number,
): ResolvedNeatRacingObservationState
```

Resolves all optional observation fields, deriving safe defaults when the
live environment has not produced them yet.

THE OPTIMAL LINE (AC-086 single source of truth): the lateral-offset and
target-speed defaults are resolved from the PUBLISHED racing-line table
(`resolveAutopilotRacingLine`) at the nearest spline sample — the same
memoized table the autopilot drives on. The signed lateral offset of the
car relative to that line is the car's own centerline offset minus the
line's published offset (positive when the car sits LEFT of the line).

Parameters:
- `envState` - Focal car observation seam state.
- `trackSpec` - Frozen track geometry.
- `closestSplineSampleIndex` - Global index of the nearest spline sample.

Returns: The complete observation state ready for normalization.

### resolveNeatRacingRadioTail

```ts
resolveNeatRacingRadioTail(
  radioField: Float32Array<ArrayBufferLike> | readonly number[] | undefined,
): readonly number[]
```

Resolves the raw self-radio tail without applying any renormalization.

### resolveNeatRacingRelativeSplineSample

```ts
resolveNeatRacingRelativeSplineSample(
  splineSamples: readonly SplineSample[],
  sampleIndex: number,
  sampleOffset: number,
): SplineSample
```

Resolves one spline sample relative to a global index on the closed ring.

Parameters:
- `splineSamples` - Frozen closed-loop spline samples.
- `sampleIndex` - Base global sample index.
- `sampleOffset` - Positive ring offset in samples.

Returns: The wrapped spline sample.

### resolveNeatRacingSignedLateralOffsetWorld

```ts
resolveNeatRacingSignedLateralOffsetWorld(
  envState: NeatRacingObservationState,
  splineSample: SplineSample,
  splineSampleFrame: NeatRacingSplineSampleFrame,
): number
```

Resolves the car's signed lateral offset from a spline sample.

Parameters:
- `envState` - Focal car observation seam state.
- `splineSample` - The reference lane-center sample.
- `splineSampleFrame` - The sample's tangent/left-normal frame.

Returns: Signed lateral offset in world units (positive LEFT of travel).

### resolveNeatRacingSplineSampleFrame

```ts
resolveNeatRacingSplineSampleFrame(
  splineSamples: readonly SplineSample[],
  sampleIndex: number,
): NeatRacingSplineSampleFrame
```

Resolves the tangent + left-normal frame of one spline sample.

THE FRAME: the tangent is the chord toward the NEXT ring sample (the
deterministic discretization the spline ribbon implies) and the left
normal is the tangent's 90-degree CCW rotation `(x, y) -> (-y, x)`,
matching the frozen `SplineSample` convention of "positive along the
unit left normal".

Parameters:
- `splineSamples` - Frozen closed-loop spline samples.
- `sampleIndex` - Global index of the sample to frame.

Returns: The unit-length spline sample frame.

### resolveNeatRacingTeammates

```ts
resolveNeatRacingTeammates(
  envState: NeatRacingObservationState,
): readonly NeatRacingObservationCarState[]
```

Resolves the same-team teammates for the focal car in roster order.

Parameters:
- `envState` - Focal car observation seam state (carries the roster).

Returns: Up to three same-team cars, roster order, focal excluded.

### resolveNeatRacingTrackBounds

```ts
resolveNeatRacingTrackBounds(
  trackSpec: TrackSpec,
): NeatRacingTrackBounds
```

Resolves the track bounds used to normalize the world-position channels.

Parameters:
- `trackSpec` - Frozen track geometry.

Returns: Bounding box plus center point of the spline-sample ring.

### resolveNeatRacingTrackedRivals

```ts
resolveNeatRacingTrackedRivals(
  envState: NeatRacingObservationState,
): readonly NeatRacingObservationCarState[]
```

Resolves the tracked rivals for the focal car in deterministic roster order.

Parameters:
- `envState` - Focal car observation seam state (carries the roster).

Returns: Up to four opposing-team cars, roster order, focal excluded.

### wrapNeatRacingAngleToMinusPiPi

```ts
wrapNeatRacingAngleToMinusPiPi(
  angleRadians: number,
): number
```

Wraps an angle to the closed interval `[-pi, pi]`.

Parameters:
- `angleRadians` - Raw angle in radians.

Returns: Wrapped angle in `[-pi, pi]`.

### wrapNeatRacingSplineSampleIndex

```ts
wrapNeatRacingSplineSampleIndex(
  sampleCount: number,
  sampleIndex: number,
): number
```

Wraps a spline sample index around the closed ring.

Parameters:
- `sampleCount` - Number of spline samples on the ring.
- `sampleIndex` - Raw (possibly out-of-range) sample index.

Returns: The wrapped index in `[0, sampleCount)`.

## ai/panel.network.transport.ts

The panel network transport codec for the Neat Racing AI-mode network
panel (Step 24C, slice P4-S2C-transport).

Two pure laws over the sim worker's `panel-network` reply envelope:

1. The idempotent restore: `restoreNeatRacingPanelNetworkFromReply`
   validates the envelope's self-describing shape (the worker's
   `NEATRACING_PANEL_NETWORK_ENVELOPE_KIND` discriminator, an integer
   rival car index, positive channel/output counts and the EXISTING
   serialized-network hex law) and rebuilds a REAL `Network` whose
   positional connection weights byte-match the encoded spawn mind.
   The rebuild is IDEMPOTENT: the same envelope always rebuilds a
   structurally identical network (the constructor is a deterministic
   input-to-output layer build — same sizes, same connection order), so
   the weights ride positional assignment over
   `network.connections[i].weight`.

2. The carrier match: `resolveNeatRacingPanelFeedNetwork` hands the
   elected panel candidate ONLY a resolution whose car index equals the
   elected candidate's agent id; a pending (missing) resolution or a
   foreign car's resolution feeds the candidate's OWN network field (the
   F3 network-less-winner fallback law — the idle path stays).

The hex law is the established serialized-network envelope law
(promotion.rules.ts / rival.propagation.ts: lowercase hex, two
characters per byte, no separators, native byte order, one or more whole
float32 values) mirrored locally with the one-serialization-path
per-payload precedent. Same-runtime byte parity only, never
cross-environment exact.

### decodeHexAsFloat32Array

```ts
decodeHexAsFloat32Array(
  hex: string,
): Float32Array<ArrayBufferLike>
```

Decodes a byte-exact hex string back into its `Float32Array`.

The caller's hex law guard (lowercase, non-empty, whole float32 values)
rejects every malformed string BEFORE the decode, so the decode itself
stays total on guarded input.

Parameters:
- `hex` - The guarded hex string (lowercase, a multiple of eight
characters).

Returns: The decoded weights (native byte order).

### NeatRacingPanelNetworkResolution

The restored elected-winner resolution bound by its carrier car index.

### PanelNetworkReplyGuardView

The guard view of the validated envelope (the payload's own shape).

### resolveNeatRacingPanelFeedNetwork

```ts
resolveNeatRacingPanelFeedNetwork(
  elected: NeatRacingNetworkPanelCandidate,
  resolution: NeatRacingPanelNetworkResolution | undefined,
): default | undefined
```

Resolves the network the elected panel candidate is fed with (P4-S2C,
Step 24C — the docked-right panel network transport).

Educational note — THE CARRIER MATCH: only a resolution whose car index
equals the elected candidate's agent id feeds that candidate. A pending
resolution (undefined) and a foreign car's resolution both fall back to
the candidate's OWN network field — the F3 law: a network-less winner is
never replaced by a networked runner-up (AC-034).

Parameters:
- `elected` - The elected panel candidate (the most-forward rival).
- `resolution` - The restored resolution (undefined while the
transport is pending).

Returns: The network the candidate is fed with, or undefined when the
candidate owns none.

Example:

```ts
const feed = resolveNeatRacingPanelFeedNetwork(winner, resolution);
```

### restoreNeatRacingPanelNetworkFromReply

```ts
restoreNeatRacingPanelNetworkFromReply(
  payload: unknown,
): NeatRacingPanelNetworkResolution
```

Restores the sim worker's panel-network reply envelope into a REAL
network (P4-S2C, Step 24C — the docked-right panel network transport).

Educational note — WHY THE REBUILD IS IDEMPOTENT: the rebuilt network's
STRUCTURE comes from the envelope's own dimensions (the channel count and
the output count), never from a random seed, so the same envelope always
yields the same connection count and the same connection order; the
weights then ride positional assignment over `network.connections[i]
.weight`. The rebuild therefore reproduces the worker's serialized mind
byte-for-byte — same-runtime parity, never cross-environment exact.

Parameters:
- `payload` - The reply envelope (the worker's `panel-network`
payload); every malformed shape is a loud rejection.

Returns: The restored resolution (the carrier car index and the rebuilt
network).

Example:

```ts
const resolution = restoreNeatRacingPanelNetworkFromReply(reply.payload);
console.log(resolution.carIndex, resolution.network.connections.length);
```

## ai/racing-adaptation.session.ts

NeatRacing HOST ADAPTATION SESSION (Phase 4, Step 23, slice
`P4-S1-eval-wiring`).

THE SESSION (spec section 9.2, AC-043 session half): the AI lane needs ONE
owner that spans the whole adaptation lifecycle, and this module is that
owner:

- it boots ONE eval-worker session per AI-lane lifecycle (the §9.2
  two-worker split: the simulation lane stays 60 Hz physics + per-tick
  control, everything growth/variant lives in the eval worker);
- it drives ONE runtime-adaptation engine per AGENT car (the engines are
  created ONCE — their hysteresis, cooldown and AC-030 predicate memory
  live in their closures and are never rebuilt per tick);
- it evaluates the BOUNDED per-car variant window through the eval worker
  (the §9.6 shaped-fitness window: one slot per car, never more than
  `NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT` slots);
- it publishes the CURRENTLY adopted per-car network through the
  simulation-worker fleet-control seam (the AC-029 adopted-snapshot law:
  the relay reads the LIVE network of a car, never a cached copy).

LANE ISOLATION (LD-12): car 0 is the HUMAN sample. It owns NO network and
its engine is never driven (it is NEVER adapted), but it IS scored in the
shaped-fitness window, because the fitness law ranks the WHOLE grid.

THE AC-030 OSCILLATION LAW IS NOT RE-DERIVED HERE: the shared predicate
lives in `./oscillation-penalty.utils` and the ENGINE evaluates it EXACTLY
ONCE per adaptation tick; this session only REPORTS the engagement the
engine produced (a second evaluation would advance the hysteresis ramp
twice and break the law's single-evaluation invariant).

THE EVAL WORKER'S ROLE IN THIS SLICE: the engine's `evaluateScore` hook is
SYNCHRONOUS and the engine creates its candidate INTERNALLY, so the async
eval worker CANNOT score the engine's candidate before the engine's commit
decision. The eval worker therefore evaluates the SESSION-LEVEL variant
window (the per-car shaped fitness), while the engine owns growth, its own
score windows, exhaustion, commit and rollback.

THE ROUND-BOUNDARY LAW: the eval protocol's `stop` request is TERMINAL and
discards the in-flight window. A `reset()` therefore stops the outgoing
session (so a window that is still in flight can never publish into the
new round) and boots a FRESH session — it never leaves a terminal session
behind.

D1: no `Math.random` anywhere — every stream comes from the AC-026
reproducibility tuple (the per-car `car-<i>` network seeds and the
`adaptation-<i>` engine seeds).

### assertNeatRacingAdaptationSessionOptions

```ts
assertNeatRacingAdaptationSessionOptions(
  carCount: number,
  maxCars: number,
): void
```

Guards the session options: the fleet size must be a positive integer and
the grid the shaped fitness ranks against must cover the fleet.

Parameters:
- `carCount` - The fleet size (car 0 is the human sample).
- `maxCars` - The grid size the position term ranks against.

### assertNeatRacingAdaptationTickCars

```ts
assertNeatRacingAdaptationTickCars(
  cars: readonly NeatRacingAdaptationCarInput[],
  carCount: number,
): void
```

Guards the per-car inputs of one tick: every car index must name a car of
the session's fleet.

Parameters:
- `cars` - The per-car inputs of the tick.
- `carCount` - The fleet size of the session.

### buildNeatRacingAdaptationVariantWindow

```ts
buildNeatRacingAdaptationVariantWindow(
  cars: readonly NeatRacingAdaptationCarInput[],
): readonly NeatRacingEvalWorkerVariant[]
```

Builds the BOUNDED per-car variant window of one tick.

Educational note (TP-18): the window holds one slot per car and never
more than `NEATRACING_EVAL_WORKER_PARALLEL_VARIANT_COUNT` slots, so the
eval worker's dispatch stays inside the adaptation-tick budget.

Parameters:
- `cars` - The per-car inputs of the tick.

Returns: The variant slots to dispatch.

### createNeatRacingAdaptationCarStates

```ts
createNeatRacingAdaptationCarStates(
  tuple: NeatRacingReproducibilityTuple,
  carCount: number,
  level: NeatRacingObservationLevel,
): Map<number, NeatRacingAdaptationCarState>
```

Builds the per-car state of the session: ONE network and ONE engine per
car.

Educational note: the human sample (car 0) owns NO network — it replays
the recorded human input — while every agent car owns its OWN network
instance, seeded from the §9.7.2 `car-<i>` child seed so every car draws
from its own stream inside the reproducibility tuple (D1). The engine of
car 0 is created (the `adaptation-<i>` table stays aligned with the fleet
index) but is NEVER driven.

Parameters:
- `tuple` - The AC-026 reproducibility tuple.
- `carCount` - The fleet size.
- `level` - The observation ladder level (selects the network width).

Returns: The per-car state map, keyed by car index.

### createNeatRacingAdaptationEvalWorker

```ts
createNeatRacingAdaptationEvalWorker(
  maxCars: number,
): NeatRacingEvalWorkerSession
```

Boots the eval-worker session of the session and binds the §9.6 shaped
fitness as its variant evaluation unit.

Parameters:
- `maxCars` - The grid size the shaped fitness ranks the position
against.

Returns: The fresh eval-worker session (in the `ready` phase).

### createNeatRacingAdaptationSession

```ts
createNeatRacingAdaptationSession(
  options: NeatRacingAdaptationSessionOptions,
): NeatRacingAdaptationSession
```

Creates the host adaptation session of one AI-lane lifecycle.

Educational note — what the session owns: ONE eval-worker session (the
§9.2 growth/variant side), ONE runtime-adaptation engine per agent car
(the growth side), the BOUNDED per-car variant window that the eval
worker scores with the §9.6 shaped fitness, and the fleet-control seam
that publishes the CURRENTLY adopted per-car network to the simulation
lane (the AC-029 adopted-snapshot law).

LANE ISOLATION (LD-12): car 0 is the human sample — it owns no network
and its engine is never driven, but it IS scored in the shaped-fitness
window.

Parameters:
- `options` - The session options (the tuple, the track geometry, the
level, the fleet size and the grid size).

Returns: The adaptation session (its seam controller, its tick driver, its
snapshot and its reset).

Example:

```ts
const session = createNeatRacingAdaptationSession({
  tuple,
  trackSpec,
  level: 1,
  carCount: 3,
});
const results = await session.adaptOnTick({
  tick: 0,
  cars: [
    { carIndex: 0, scoreHistory: [], position: 2, timeSeconds: 30, lapProgress01: 0.5 },
    { carIndex: 1, scoreHistory: [], position: 1, timeSeconds: 28, lapProgress01: 0.6 },
  ],
});
const control = session.controller.computeControl(fleet, 1);
```

### NeatRacingAdaptationCarInput

The per-car input of one adaptation tick.

### NeatRacingAdaptationCarState

The per-car state of one adaptation session.

### NeatRacingAdaptationDrivingRecord

The driving record one variant slot carries across the eval wire.

Educational note: the eval protocol transports the slot payload verbatim
(`snapshot: unknown`), so the slot carries the three terms the §9.6
shaped fitness is a pure function of.

### NeatRacingAdaptationSession

The host adaptation session surface the AI lane consumes.

### NeatRacingAdaptationSessionOptions

The construction options of one host adaptation session.

### NeatRacingAdaptationSessionSnapshot

The plain-data snapshot of one adaptation session.

### NeatRacingAdaptationSessionTickInput

The input of one adaptation tick.

### NeatRacingAdaptationSessionTickResult

The per-car outcome of one adaptation tick.

### readNeatRacingAdaptationDrivingRecord

```ts
readNeatRacingAdaptationDrivingRecord(
  variant: NeatRacingEvalWorkerVariant,
): NeatRacingAdaptationDrivingRecord
```

Reads the driving record out of one variant slot.

Educational note: the eval protocol transports the slot payload verbatim
(`snapshot: unknown`), so the unit validates the shape it consumes instead
of trusting the wire.

Parameters:
- `variant` - The variant slot as received on the wire.

Returns: The driving record of the slot.

### readNeatRacingAdaptationEvalWindow

```ts
readNeatRacingAdaptationEvalWindow(
  responses: readonly NeatRacingEvalWorkerResponse[],
): readonly NeatRacingEvalWorkerVariantResult[] | null
```

Reads the scored window out of the eval worker's reply.

Educational note: the protocol answers a malformed window with ONE error
reply (a loud session failure) and answers a cancelled round with NO
reply at all (the window published nothing).

Parameters:
- `responses` - The replies the eval worker returned.

Returns: The scored slots, or null when the round was cancelled.

### readNeatRacingAdaptationNullable

```ts
readNeatRacingAdaptationNullable(
  source: ReadonlyMap<number, T>,
  carCount: number,
): readonly (T | null)[]
```

Reads a per-car map as a dense array indexed by car index, keeping the
absent cars explicitly null.

Parameters:
- `source` - The per-car map.
- `carCount` - The fleet size.

Returns: The dense per-car values (null for a car without an entry).

### readNeatRacingAdaptationNumbers

```ts
readNeatRacingAdaptationNumbers(
  source: ReadonlyMap<number, number>,
  carCount: number,
): readonly number[]
```

Reads a per-car numeric map as a dense array indexed by car index.

Parameters:
- `source` - The per-car map.
- `carCount` - The fleet size.

Returns: The dense per-car values (0 for a car without an entry).

### stopNeatRacingAdaptationEvalWorker

```ts
stopNeatRacingAdaptationEvalWorker(
  session: NeatRacingEvalWorkerSession,
): void
```

Stops an eval-worker session.

Educational note: the protocol's `stop` request is TERMINAL and discards
the in-flight window (E11(d)), so a reset that stops the outgoing session
guarantees that no superseded round can publish into the new one.

Parameters:
- `session` - The session to stop.

Returns: Nothing (the session is stopped in place).

## ai/fitness.shaping.utils.ts

NeatRacing fitness shaping (slice `P2-S3-E`, TP-08, AC-029 support).

THE LAW (spec section 9.1, E8): the growth-engine commit rule
(`bestVariantScore > baselineScore + threshold`) consumes ONE
HIGHER-IS-BETTER fitness number, while the recycled team-fitness resolver
(`resolveTeamFitness`) is LOWEST-POSITION-WINS. The recycled position
ordering therefore enters this module INVERTED — `maxCars − position` — so
finishing P1 scores the highest of the field. The shaped value is a PURE
function of three probe terms:

```text
fitness = W_POS · (maxCars − position) − W_TIME · time + W_LP · lapProgress01
```

Monotonic directions (the pinned red-test invariants, each term held with
the others fixed):

- POSITION: fitness strictly INCREASES as the position number DECREASES —
  each inverted position step pays exactly `W_POS` (P1 > P2 > ... > Pn).
- TIME: fitness strictly INCREASES as the elapsed race time DECREASES —
  each elapsed second pays exactly `W_TIME` (finishing faster is better).
- LAP PROGRESS: fitness strictly INCREASES with `lapProgress01` — each
  unit of progress pays exactly `W_LP` (more laps driven is better).

THE FORMULA SHAPE IS CONTRACT; THE WEIGHTS ARE TP: the sum-of-terms shape
above (including the inverted position term) is the deliverable contract,
while `NEATRACING_FITNESS_WEIGHT_POSITION`, `..._TIME` and
`..._LAP_PROGRESS` are TP-08 tuning points — arbitrary starting values,
retunable in Phase 2 as one-constant changes (spec risk R-04). A retune
MUST keep every weight strictly positive or the monotonic directions
above (and therefore the higher-is-better contract) break.

THE ACCUMULATION ORDER IS PINNED: the terms are summed left-to-right
(position -> time -> progress) so the same runtime keeps byte-identical
scores for identical inputs (same-runtime Level-2 ordered-deterministic;
never a cross-environment bit-exact claim — floating-point associativity
is not guaranteed across runtimes).

THE STRICT-REJECTION GUARDS keep the bounded-output contract: a
non-finite or out-of-range input throws a `RangeError`
(`neatRacing fitness shaping failed` voice) instead of silently shaping a
NaN into the commit rule, where a NaN score would disable the
`bestVariantScore > baselineScore + threshold` comparison. `lapProgress01`
callers are expected to be already-normalized by the observation
assembler's `normalizeNeatRacingProbability` clamp path; this module
surfaces out-of-band data loudly rather than re-clamping it.

THE CONSUMER: the eval worker's variant evaluation reports the
deterministic `score` per variant slot (AC-029 support) — this module is
the pure library-side shaping surface behind that protocol field; it owns
no RNG, no wall clock, no worker globals.

### assertFitnessFiniteNumber

```ts
assertFitnessFiniteNumber(
  value: number,
  label: string,
): void
```

Guards one shaped-fitness term value against non-finite data (NaN and the
infinities have no committable score meaning here).

Parameters:
- `value` - The term value to guard.
- `label` - The term's name (for the static message surface).

### assertFitnessLapProgress01

```ts
assertFitnessLapProgress01(
  lapProgress01: number,
): void
```

Guards the clamped lap progress (the observation assembler's
`normalizeNeatRacingProbability` path already clamps to `[0, 1]`; an
out-of-band value here is bad upstream data to surface, not noise to absorb).

Parameters:
- `lapProgress01` - The lap progress to guard.

### assertFitnessMaxCars

```ts
assertFitnessMaxCars(
  maxCars: number,
): void
```

Guards the grid size the position ranks against (a rank is only invertible
relative to a well-formed grid).

Parameters:
- `maxCars` - The grid size to guard.

### assertFitnessPosition

```ts
assertFitnessPosition(
  position: number,
  maxCars: number,
): void
```

Guards the 1-based finishing position against the well-formed grid.

Parameters:
- `position` - The position to guard.
- `maxCars` - The grid size the position ranks against.

### assertFitnessTimeSeconds

```ts
assertFitnessTimeSeconds(
  timeSeconds: number,
): void
```

Guards the elapsed race time (a race probe can never record negative time;
the time term is a monotonic penalty, so `0` is the best time).

Parameters:
- `timeSeconds` - The elapsed race time to guard.

### assertNeatRacingShapedFitnessInputs

```ts
assertNeatRacingShapedFitnessInputs(
  position: number,
  timeSeconds: number,
  lapProgress01: number,
  maxCars: number,
): void
```

Validates the whole shaped-fitness input bundle before any term is shaped.

`maxCars` is guarded first because it bounds the position range the
position guard checks against.

Parameters:
- `position` - The 1-based finishing position.
- `timeSeconds` - The elapsed race time in seconds.
- `lapProgress01` - The clamped lap progress.
- `maxCars` - The grid size the position ranks against.

### computeNeatRacingShapedFitness

```ts
computeNeatRacingShapedFitness(
  position: number,
  timeSeconds: number,
  lapProgress01: number,
  maxCars: number,
): number
```

Computes the shaped HIGHER-IS-BETTER fitness of one car probe.

The spec's formula shape is `fitness(position, time, lapProgress)`; this
module names the terms for the lane's data shapes (`position` is the
1-based finishing position, `timeSeconds` is elapsed race time in seconds
and `lapProgress01` is the assembler's clamped 0..1 lap progress) and
takes `maxCars` explicitly because the recycled position ordering is only
invertible relative to the grid size it ranks against.

Parameters:
- `position` - The car's 1-based finishing position in `[1, maxCars]`
(P1 = 1). A non-integer or out-of-grid value is rejected.
- `timeSeconds` - The elapsed race time in seconds (`>= 0`). Less time
is strictly higher fitness.
- `lapProgress01` - The clamped lap progress in `[0, 1]`. More progress
is strictly higher fitness.
- `maxCars` - The grid size the position ranks against — a positive
integer `>= 1` (the lvl-3 E2 layout carries 5 cars: 1 main agent plus
4 rivals).

Returns: The shaped fitness. Finite for every accepted input; strictly
monotonic in each term per the module-header directions.

Example:

```ts
// lvl-3 grid (5 cars): P1 after 41.5 s with 76% lap progress
const fitness = computeNeatRacingShapedFitness(1, 41.5, 0.76, 5);
// 10 * (5 - 1) - 0.01 * 41.5 + 1 * 0.76 === 40.345
```

### NEATRACING_FITNESS_WEIGHT_LAP_PROGRESS

Weight of the lap-progress term, paid per unit of `lapProgress01` (the
observation assembler's clamped 0..1 progress): a full lap of progress
scores exactly this weight.

TP-08 tuning point — arbitrary starting value chosen so one full lap of
progress ties a 100-second reduction in race time; retunable in Phase 2.
Must stay strictly positive (spec E8 monotonicity).

### NEATRACING_FITNESS_WEIGHT_POSITION

Weight of the INVERTED position term, paid per position step
(`maxCars − position`): P1 scores `maxCars − 1` steps above the last-placed
car, so the P1-vs-last gap scales with grid size (the lvl-3 five-car grid
spans 4 steps).

TP-08 tuning point — arbitrary starting value, NOT contract; retunable in
Phase 2. Must stay strictly positive (spec E8 monotonicity).

### NEATRACING_FITNESS_WEIGHT_TIME

Weight of the elapsed-race-time penalty, paid per elapsed second: less
race time is strictly higher fitness.

TP-08 tuning point — arbitrary starting value chosen so a full lvl-3 grid
position step (10 points) outweighs several minutes of race time; retunable
in Phase 2. Must stay strictly positive (spec E8 monotonicity).

## ai/oscillation-penalty.utils.ts

NeatRacing shared oscillation predicate (slice `P2-S3-B`, AC-030).

THE LAW (spec section 9.5, LD-31 / E3): the racing oscillation retune has
TWO coupled mechanisms — the MULTIPLICATIVE oscillation penalty inside the
variant scorer and the ADDITIVE growth-commit threshold boost — and BOTH
are driven by ONE shared predicate:

- ENGAGE iff `nodeCount >= the current level's promotion threshold`
  (TP-07: 300 at lvl 1 -> 2, 600 at lvl 2 -> 3) AND the oscillation signal
  has been sustained over the TP-09 grace window;
- HYSTERESIS (no step changes): the engagement ramps IN over the grace
  window while the signal is sustained and ramps OUT over the same window
  while it is absent, so a network crossing the node-count boundary never
  sees an instantaneous score jump (the original rollback-cascade trigger);
- EVALUATED ONCE PER ADAPTATION TICK (never per variant): the caller
  evaluates the predicate once per tick and applies the resulting
  engagement to the baseline AND the variant score SYMMETRICALLY, so the
  variant-vs-baseline comparison stays fair;
- THE DEADBAND IS PRESERVED: `RACING_OSCILLATION_MIN_MEAN_STEERING = 0.15`
  lives in the metric detectors (`detectSteeringOscillation` in
  `runtime.adaptation.ts`), so a gentle chicane correction still yields a
  metric of exactly 0 and is never penalized;
- BELOW-THRESHOLD NETWORKS ARE NEVER PENALIZED: the node-count gate is a
  HARD gate — under the level's promotion threshold the law is fully
  disengaged and the hysteresis state is reset, so a young network can
  never accumulate penalty debt and trip a rollback cascade (AC-030).

THE INTENTIONAL BOUNDARY ASYMMETRY (spec section 9.5 — do NOT "fix"): the
predicate engages at `nodeCount >= threshold` while promotion (section 9.4)
fires at `nodeCount > threshold`. The penalty is therefore already ramping
when a promotable crossing occurs, so there is NO step change in the
penalty at the promotion instant. The two operators MUST NOT be normalized
to a common comparison.

THE LEVEL RE-BASE ON PROMOTION (a consequence of the level-relative
floor — NOT a bug and NOT a second asymmetry): the threshold is the
CURRENT level's promotion threshold, so a level 1 -> 2 promotion re-bases
the floor 300 -> 600 and the law disengages instantly at that boundary.
The drop is a DISENGAGEMENT (the score rises; it never falls), so it can
never cascade into a rollback, and the round-boundary state reset (AC-031)
owns the state cleanup that follows a level change.

THE CONSTANT OWNERSHIP (no dual path): this module is the canonical owner
of the penalty weight, the commit gate, the commit boost and the TP-07
promotion node thresholds of the AI lane. `runtime.adaptation.ts` still
declares its own copies of the first three because that file is OUT of the
P2-S3-B packet boundary (READ-ONLY in this slice); the slice that wires
this predicate into the runtime MUST delete those three declarations there
and import them from here IN THE SAME STEP. This module deliberately does
NOT import them back from `runtime.adaptation.ts` — an import in that
direction would create a cycle once the runtime imports this predicate.

THE REALM: pure, headless and deterministic — no RNG, no wall-clock reads,
no DOM, no worker globals. The same (input, state) pair always yields the
same engagement.

### advanceOscillationState

```ts
advanceOscillationState(
  state: OscillationPredicateState,
  oscillationMetric: number,
): OscillationPredicateState
```

Advances the hysteresis state by one evaluation: an active oscillation
signal ramps the engagement up, an inactive signal ramps it down.  Neither
direction can move the ramp by more than one grace step, so the penalty
never steps.

Parameters:
- `state` - The previous tick's hysteresis state.
- `oscillationMetric` - The tick's (deadband-filtered) oscillation metric.

Returns: The next tick's hysteresis state.

### applyOscillationPenalty

```ts
applyOscillationPenalty(
  score: number,
  engagement: OscillationEngagement,
): number
```

Applies the oscillation penalty to ONE score.

The penalty is multiplicative and proportional to the score, so it damps
the oscillation without inverting a positive score into a negative one.
Apply the SAME engagement to the baseline and to the variant score — that
is what makes the comparison symmetric.

Parameters:
- `score` - The unpenalized score (baseline or variant).
- `engagement` - The engagement verdict of the current adaptation tick.

Returns: The penalized score.

Example:

```ts
const penalized = applyOscillationPenalty(10, engagement);
// 10 * (1 - engagement.penaltyFraction)
```

### applyOscillationPenaltySymmetrically

```ts
applyOscillationPenaltySymmetrically(
  baselineScore: number,
  variantScore: number,
  engagement: OscillationEngagement,
): SymmetricOscillationScores
```

Applies the oscillation penalty SYMMETRICALLY to the baseline and the
variant score of one adaptation tick.

AC-030 requires the penalty to be applied to both sides of the comparison:
the penalty is evaluated ONCE per adaptation tick (never per variant) and
the resulting engagement damps the baseline and the variant by the SAME
fraction, so the variant-vs-baseline ordering survives the penalty.

Parameters:
- `baselineScore` - The unpenalized baseline score.
- `variantScore` - The unpenalized variant score.
- `engagement` - The engagement verdict of the current adaptation tick.

Returns: The penalized baseline and variant scores.

Example:

```ts
const scores = applyOscillationPenaltySymmetrically(10, 11, engagement);
// { baselineScore: 10 * (1 - f), variantScore: 11 * (1 - f) }
```

### buildOscillationEngagement

```ts
buildOscillationEngagement(
  state: OscillationPredicateState,
  oscillationMetric: number,
): OscillationEngagement
```

Builds the engagement verdict from the advanced hysteresis state.

Parameters:
- `state` - The hysteresis state after this evaluation.
- `oscillationMetric` - The tick's (deadband-filtered) oscillation metric.

Returns: The engagement verdict of the tick.

### createDisengagedEngagement

```ts
createDisengagedEngagement(): OscillationEngagement
```

Builds the fully disengaged verdict of a below-threshold tick.  It also
resets the hysteresis state, so a network that drops under the promotion
threshold never carries engagement debt back over the boundary.

Returns: The disengaged engagement verdict.

### createOscillationPredicateState

```ts
createOscillationPredicateState(): OscillationPredicateState
```

Creates the initial (fully disengaged) hysteresis state of the shared
oscillation predicate.

The returned state is the seed of one adaptation session's predicate
memory; it carries no penalty debt, so the very first evaluation of a
session starts from zero engagement.

Returns: The fresh, fully disengaged predicate state.

Example:

```ts
const state = createOscillationPredicateState();
// { ramp: 0, sustainedTicks: 0 }
```

### evaluateOscillationEngagement

```ts
evaluateOscillationEngagement(
  input: OscillationEvaluationInput,
  state: OscillationPredicateState,
): OscillationEngagement
```

Evaluates the shared oscillation predicate for ONE adaptation tick.

The predicate is the single gate of BOTH coupled mechanisms: the
multiplicative score penalty (via {@link OscillationEngagement.penaltyFraction})
and the additive growth-commit threshold boost (via
{@link OscillationEngagement.commitBoost}).

The engagement is hysteretic: an active oscillation signal ramps the
engagement UP by one grace step per evaluation and an inactive signal ramps
it DOWN by one grace step per evaluation, so neither the node-count
boundary nor a single noisy tick can move the penalty in a step.

Parameters:
- `input` - The tick observation (node count, oscillation metric, level).
- `state` - The previous tick's hysteresis state (the session seed for
the first tick).

Returns: The engagement verdict, carrying the next tick's state.

Example:

```ts
const state = createOscillationPredicateState();
const engagement = evaluateOscillationEngagement(
  { nodeCount: 300, oscillationMetric: 0.8, level: 1 },
  state,
);
// engagement.ramp === 0.25 (one grace step of the four-step window)
```

### isOscillationSignalActive

```ts
isOscillationSignalActive(
  oscillationMetric: number,
): boolean
```

Reports whether the tick's oscillation metric carries an oscillation
signal.  A metric of exactly 0 is the deadband-filtered "smooth driving"
verdict and never engages the law.

Parameters:
- `oscillationMetric` - The tick's (deadband-filtered) oscillation metric.

Returns: True when an oscillation signal is present.

### NEATRACING_OSCILLATION_GRACE_TICKS

TP-09 grace window of the shared predicate, counted in PREDICATE
EVALUATIONS (adaptation ticks) and NOT in raw simulation ticks.  The
engagement ramps in over this window while the oscillation signal is
sustained and ramps out over the same window while it is absent, so a
single noisy adaptation tick can never engage (or disengage) the law.
Arbitrary starting tuning point (like TP-07), tuned in Phase 2.

### NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_2

TP-07 promotion node threshold of the lvl-1 -> lvl-2 band: the node count
at which the oscillation law engages for a session at level 1.  The
promotion itself fires at a STRICTLY LARGER node count (spec section 9.4)
— the asymmetry is intentional (see the module header).

### NEATRACING_PROMOTION_NODE_THRESHOLD_LVL_3

TP-07 promotion node threshold of the lvl-2 -> lvl-3 band.  It is also the
floor of the level-3 band, because the level ladder caps at level 3
(`NEATRACING_HUMAN_MAX_LEVEL`) and there is no further promotion to gate
on.

### OscillationEngagement

The shared predicate's verdict for one adaptation tick.

The verdict is evaluated ONCE per adaptation tick and then applied to the
baseline AND the variant score, which is what keeps the comparison fair.

### OscillationEvaluationInput

The observation the shared predicate consumes at one adaptation tick.

### OscillationPredicateState

The shared predicate's hysteresis state at one adaptation tick.

The state is the ONLY memory the predicate keeps: it is threaded from one
tick to the next by the caller, which is what makes the engagement
hysteretic (a ramp, never a step change).

### RACING_OSCILLATION_COMMIT_THRESHOLD

Oscillation gate of the additive growth-commit threshold boost: the boost is
granted only while the oscillation metric is strictly ABOVE this value.
At or below the gate the growth-commit threshold is left unchanged.

### RACING_OSCILLATION_IMPROVEMENT_THRESHOLD_BOOST

Small additive boost added to the growth-commit threshold while the shared
predicate is engaged and the oscillation metric is above
{@link RACING_OSCILLATION_COMMIT_THRESHOLD}.  The boost is scaled by the
hysteresis ramp (never a step change) and is NOT multiplied by the metric.

### RACING_OSCILLATION_PENALTY_WEIGHT

Weight applied to the oscillation metric when the shared predicate removes
score from a baseline or a variant (AC-030).  The penalty is MULTIPLICATIVE
and PROPORTIONAL to the score, so it damps oscillation without inverting a
positive score into a negative one.

### resolveCommitBoost

```ts
resolveCommitBoost(
  ramp: number,
  oscillationMetric: number,
): number
```

Resolves the additive growth-commit threshold boost of the tick: the boost
is granted only above the commit gate and is scaled by the hysteresis ramp
so it ramps in and out instead of stepping.

Parameters:
- `ramp` - The tick's engagement strength in [0, 1].
- `oscillationMetric` - The tick's (deadband-filtered) oscillation metric.

Returns: The additive growth-commit threshold boost.

### resolveOscillationPromotionNodeThreshold

```ts
resolveOscillationPromotionNodeThreshold(
  level: number,
): number
```

Resolves the TP-07 promotion node threshold that gates the oscillation law
for a session level.

Level 1 is gated by the lvl-1 -> lvl-2 band (300 nodes), level 2 by the
lvl-2 -> lvl-3 band (600 nodes) and level 3 keeps the lvl-2 -> lvl-3 band
as its floor because the level ladder caps at level 3.  An unknown level
falls back to the highest band, which is the conservative choice (a
network is penalized less eagerly, never more).

Parameters:
- `level` - The session level of the evaluated network.

Returns: The node count at which the oscillation law engages for that level.

Example:

```ts
resolveOscillationPromotionNodeThreshold(1); // 300
resolveOscillationPromotionNodeThreshold(2); // 600
```

### SymmetricOscillationScores

The symmetrically penalized score pair produced by
{@link applyOscillationPenaltySymmetrically}.
