# physics

Race rules (laps, checkpoints, level advancement) for the neatRacing
example (plan slice P1-S7-D; AC-024 partial; spec sections 4.1, 4.2,
20 OQ-001/OQ-002; TP-12/TP-13).

Session semantics (OQ-001 resolved values as named constants, TP-13): a
human race lasts NEATRACING_HUMAN_RACE_LAPS forward laps with an
arcade-style checkpoint budget that starts at
NEATRACING_CHECKPOINT_INITIAL_BUDGET_S and is extended by
NEATRACING_CHECKPOINT_PER_LAP_EXTENSION_S at every forward lap-line
crossing. The spec is silent on budget exhaustion, so the arcade-
standard rule is pinned here by design: a step whose remaining budget
reaches zero ends the race out-of-time, a crossing granted in the same
tick saves the run before that check (the extension applies first; the
line saves the second), and the budget is clamped at zero (never
negative).

Level advancement (OQ-002, TP-12): the first session starts at
NEATRACING_HUMAN_START_LEVEL and completing a race is the session
boundary — the NEXT race carries
min(level + 1, NEATRACING_HUMAN_MAX_LEVEL) in `nextRaceLevel` (capped
at lvl 3; a failed race never advances). Advancement only materializes
at the completion boundary; no mid-race level change ever happens.

Determinism: the module is pure — no `Math.random`, no clock reads,
the caller supplies `dtS`. The stepper mutates nothing and returns
fresh state records while a race runs, and steps on a terminal status
('raceComplete' / 'outOfTime') return the input record itself
unchanged, so the same (state, dt stream, crossing stream) reproduces
byte-identical `JSON.stringify` states on the same runtime
(same-runtime ordered-deterministic rung; cross-environment byte
equality is NOT claimed).

Scope notes: HUD display of laps/budget and the mode selector arrive
with P1-S7-E; AI-mode lap floors and the promotion predicate are
phase-2 rules and never live here. This module is the pure rules core
the boot glue (Step 13) wires into the host loop.

## physics/race-rules.ts

### applyForwardLapCrossing

```ts
applyForwardLapCrossing(
  state: RaceRulesState,
  forwardLapLineCrossing: boolean,
): RaceRulesState
```

Applies the one allowed forward lap-line crossing of this step: counts
the lap, grants the extension, and — exactly when the lap count
reaches NEATRACING_HUMAN_RACE_LAPS — marks the race complete and
records the NEXT race's carried level. No-op when the flag is false.

Parameters:
- `state` - Clock-decayed race state (never mutated).
- `forwardLapLineCrossing` - True only for a forward line crossing.

Returns: Fresh record with the crossing applied (and possibly the
completion boundary resolved), or the input record when no crossing.

### createRaceRulesState

```ts
createRaceRulesState(
  level: number,
): RaceRulesState
```

Creates the fresh race-rules state for one session.

The state is the deterministic seed of the pure stepper: a running
race with no laps, no raced time, the full initial checkpoint budget,
and no carried next-race level yet.

Parameters:
- `level` - Session level of the race. Defaults to
NEATRACING_HUMAN_START_LEVEL; follow-up sessions pass the level
carried by the previous race's `nextRaceLevel`.

Returns: The fresh running race state.

Example:

```ts
const fresh = createRaceRulesState();
// { level: 1, lapsCompleted: 0, checkpointBudgetS: 60,
//   elapsedS: 0, raceStatus: 'running', nextRaceLevel: null }
```

### decayRaceClock

```ts
decayRaceClock(
  state: RaceRulesState,
  dtS: number,
): RaceRulesState
```

Advances the race clock: spends `dtS` checkpoint seconds and races
`dtS` more seconds, returning a fresh record.

No clamping happens here — the same step's crossing may still extend
the budget, and the dying-tick contract requires the raw (pre-clamp)
balance to feed the extension.

Parameters:
- `state` - Current running race state (never mutated).
- `dtS` - Elapsed seconds of this step.

Returns: Fresh record with the decayed budget and the advanced elapsed clock.

### markRaceComplete

```ts
markRaceComplete(
  crossed: RaceRulesState,
): RaceRulesState
```

Marks the race complete exactly at the lap-count boundary and records
the NEXT race's carried level: min(level + 1, NEATRACING_HUMAN_MAX_LEVEL)
(level-capped session-boundary advancement; the completed race record
itself is never rewritten after this tick because terminal steps are
inert).

Parameters:
- `crossed` - Fresh record whose lap count just reached the limit.

Returns: Fresh terminal record with 'raceComplete' and the carried level.

### NEATRACING_CHECKPOINT_INITIAL_BUDGET_S

Initial human checkpoint budget in seconds (OQ-001 resolved, TP-13):
the arcade per-race time budget that every raced second spends down
and every lap-line crossing refills.

### NEATRACING_CHECKPOINT_PER_LAP_EXTENSION_S

Checkpoint extension granted at every forward lap-line crossing, in
seconds (OQ-001 resolved, TP-13): the arcade "the line extends your
clock" rule. The extension lands AFTER the step's decay, so a crossing
in the same tick the budget would die saves the run.

### NEATRACING_HUMAN_MAX_LEVEL

Level cap (OQ-002 resolved): the lvl-3 ceiling of the level ladder —
a completed race at the cap keeps the next race at lvl 3.

### NEATRACING_HUMAN_RACE_LAPS

Total forward laps of a human-mode race (OQ-002 resolved): the race
ends complete exactly at this lap count, never before; AI-mode session
floors are phase-2 rules and never read this constant.

### NEATRACING_HUMAN_START_LEVEL

Level of the FIRST session (OQ-002 resolved): human sessions start at
lvl 1 (one solo car per the level ladder, spec section 4.1).

### RaceRulesState

Full race-rules session state for one race. Small enough to
JSON-serialize byte-identically (the same-runtime determinism rung),
so replay suites can pin rule behavior across repeated runs.

### RaceRulesStepInput

Inputs of one race step. The host samples real frame time once per
step (D5 analog; replay suites pass binary-exact fixture deltas), and
the forward-crossing flag is set by the track-crossing detector only
for forward, once-per-circuit line passes.

### RaceStatus

Lifecycle status of one race session. A terminal status
('raceComplete' / 'outOfTime') freezes the whole state: further steps
on a terminal record return the input record unchanged.

### resolveBudgetExpiry

```ts
resolveBudgetExpiry(
  state: RaceRulesState,
  grantedExtension: boolean,
): RaceRulesState
```

Clamps the checkpoint budget at zero and resolves the out-of-time
flip: a still running race whose budget has reached zero ends out of
time — unless this same step granted a crossing (the line saves the
second), and never overriding a race completion marked this tick.

Parameters:
- `state` - Race state after decay and the optional crossing.
- `grantedExtension` - True when this step granted the crossing
extension (the same-tick escape hatch from the out-of-time flip).

Returns: Fresh record with the clamped budget and the resolved status.

### stepRaceRules

```ts
stepRaceRules(
  state: RaceRulesState,
  input: RaceRulesStepInput,
): RaceRulesState
```

Advances the race rules by one step and returns the NEXT state.

Step order is contracted: while the race runs, the checkpoint budget
first decays and the elapsed clock advances by `dtS`; then a forward
lap-line crossing — and only one — counts the lap and grants the
extension; the race is marked 'raceComplete' exactly when the lap
count reaches NEATRACING_HUMAN_RACE_LAPS (resolved at that crossing
tick, never before); the budget is clamped at zero; and a still
running race whose budget has reached zero ends 'outOfTime' unless
THIS step granted the crossing (the line saves the second). Steps on
a terminal status return the input record unchanged.

Parameters:
- `state` - Current race-rules state; never mutated.
- `input` - The step inputs (`dtS`, forward crossing flag).

Returns: The next state: a fresh record while the race runs, otherwise
the unchanged input record.

Example:

```ts
const next = stepRaceRules(fresh, {
  dtS: 1,
  forwardLapLineCrossing: false,
});
// budget 60 - 1 = 59 s, still running
```

## physics/drift.model.ts

Arcade drift model for the neatRacing example (Step 14N; slice
`P1-drift-feel-tune-model`; AC-047 / AC-079 / AC-087 / AC-089; Step 14S
slice `P1-momentum-law`; AC-091).

THE MODEL IS THE D1 COMMITTED-DRIFT MODEL. The drift is a MOMENTUM-DERIVED,
CLAMPED state; the body chases it; the countersteer gets its own
authority applied AGAINST the slide. The stage order is FIXED and is
the contract (integration artifact section 4.3):

  stage 0  the SOFTENED MIN_SPEED gate             (stepDriftCommitment)
  stage 1  the MOMENTUM commitment (DEMAND / EXIT) (stepDriftCommitment)
  stage 2  the grip budget factor                 (computeGripBudgetFactor)
  stage 3  the committed chase                    (computeNextLateralVelocity)
  stage 4  the countersteer authority             (computeCountersteerAuthority)
  stage 5  the snap-back at the zero crossing     (computeNextLateralVelocity)
  stage 6  the longitudinal speed scrub           (computeLongitudinalSpeedScrub)
  stage 7  the merged pose spring                 (stepDriftPoseSpring)

SIGN CONVENTION (production-pinned): `lateralVelocity > 0` means the
velocity vector lies to the LEFT of the heading, so
`slipAngle = atan2(lateralVelocity, speed)` is NEGATIVE in a LEFT turn
(the nose leads, the velocity lags). The visible pose therefore
shares the sign of the committed slip (`+committedSlip`), and a LEFT
turn yaws the body LEFT.

THE MOMENTUM LAW (Step 14S, AC-091): the drift is CAUSED by the MOMENTUM,
not COMMANDED by the wheel. The committed slide is the MOMENTUM EXCESS -
the lateral acceleration the commanded turn DEMANDS minus the lateral
acceleration the tires can HOLD at the current speed - scaled by the
declared depth budget. The SAME wheel command therefore slides DEEPER the
faster the car goes, the slide is EXACTLY 0 at a standstill, and the
demand never exceeds the declared depth budget. The OLD speed-proportional
ENTRY-RATE COMMAND (its named constant and its named function) is DELETED
in this change and REPLACED by the momentum law - there is NO deferred
cleanup.

THE ZERO-RATE CLAUSE IS SCOPED TO ENGAGEMENT ON PURPOSE (FP-14S-01): a
ZERO demanded turn rate produces an EXACTLY 0 committed slide AT
ENGAGEMENT - an UNCOMMITTED latch - because the ENGAGE branch of
`stepDriftCommitment` needs a POSITIVE momentum demand, so the latch is
never spent on a zero slide. UNDER A LIVE LATCH a ZERO momentum demand
HOLDS the committed depth (the HOLD branch of `stepDriftCommitment`):
the slide never SNAPS to zero under a live latch, and only the RELEASE -
the COUNTERSTEER or the NEUTRAL wheel - unwinds it. The UNSCOPED form
("the slide is EXACTLY 0 when no turn rate is demanded") is TRUE ONLY AT
ENGAGEMENT and is FALSE under a live latch, so it is NOT the contract.

THE LATCH (`turnSign`) IS THE TURN DIRECTION (+1 = LEFT), NEVER
`sign(slip)`. It is a three-state machine:

  - ENGAGE (uncommitted -> committed) needs ALL of: `|steer| >=
    STEER_ENGAGE_THRESHOLD`, `committed.slip` exactly 0, the
    RE-ARM TOKEN present and a POSITIVE momentum demand (a wheel
    command the momentum does NOT carry opens NO slide). The MIN_SPEED
    gate is SOFTENED (Step 14Q, AC-089): it is no longer a hard cliff
    on the COMMITMENT - the committed depth is the MOMENTUM DEMAND of
    the tick, so the slide grows with the weight at speed instead of
    freezing - while the VISIBLE slide (the lateral velocity) stays
    EXACTLY 0 there;
  - HOLD: while the latch is live the committed depth FOLLOWS the
    momentum demand; when the demand is EXACTLY 0 (the momentum no
    longer carries the commanded turn) the latch HOLDS the committed
    depth and only the RELEASE unwinds it, so a live slide never
    SNAPS to zero;
  - the RE-ARM TOKEN (`reArmed`) starts TRUE and is latched again
    when the wheel is seen NEUTRAL (`|steer| < STEER_NEUTRAL_THRESHOLD`)
    for at least one tick; a RELEASE clears it;
  - RELEASE (committed -> uncommitted) happens at the hard zero, and
    the slide UNWINDS when the driver releases the wheel: the
    COUNTERSTEER or the NEUTRAL wheel (`|steer| <
    STEER_NEUTRAL_THRESHOLD`) unwinds the committed slide, and the
    release CLEARS the token, so an opposite steer ALONE can never
    re-latch and no phantom mirrored drift can appear;
  - CONTRACTION: an opposite-sign steer applies the countersteer
    authority and NEVER flips `turnSign`.

The model stays PURE: no `Math.random`, no side effects, no hidden
globals. The same `(state, input)` always yields the same output ON THE
SAME RUNTIME (`Math.tan`, `Math.exp` and `Math.atan2` are
implementation-approximated, so the identity is same-engine, not
cross-engine), and the serialized zero channel is byte identical to
`createZeroDriftChannelState()` (AC-047D).

### computeCountersteerAuthority

```ts
computeCountersteerAuthority(
  committed: DriftCommittedState,
  input: DriftStepInput,
): number
```

D1 stage 4: the countersteer authority, applied AGAINST the slide.

The authority is an ACCELERATION (m/s^2) toward zero: it is multiplied by
`dt` to yield a velocity delta. The gate is the LATCHED TURN DIRECTION
(never `sign(slip)`), so the countersteer can never mirror the slide, and
the intent magnitude gate is the in-model fallback of D2.3.

Parameters:
- `committed` - The committed state (the slide and the latched turn).
- `input` - The tick input (the steer and the speed percentile).

Returns: The countersteer acceleration (m/s^2); 0 when the gate is closed.

### computeDriftReArmReleaseRateRadPerS

```ts
computeDriftReArmReleaseRateRadPerS(
  dt: number,
): number
```

The NEUTRAL release RATE (rad/s): the rate at which the committed slide
unwinds while the wheel is NEUTRAL (Step 14Q, AC-089).

The rate is DERIVED from the named hold window: the slide unwinds from
the full slide budget to the hard zero over
`NEATRACING_DRIFT_REARM_HOLD_TICKS` ticks, so the hold constant has a
real reader.

Parameters:
- `dt` - The tick duration (s).

Returns: The release rate (rad/s).

### computeDriftSlideDepthBudgetRad

```ts
computeDriftSlideDepthBudgetRad(
  speedMps: number,
): number
```

The committed slide DEPTH BUDGET (rad) at a speed: the speed-proportional
depth law (Step 14Q, AC-088).

The depth is PROPORTIONAL to the speed below the saturation speed and
SATURATES at the AC-087 budget (`MAX_SLIP_ANGLE_RAD`) above it, so a slow
car slides proportionally less and no car can exceed the declared budget.

THE FLOOR (the SOFTENED MIN_SPEED gate, AC-089): the budget never falls
to 0. It is FLOORED at the hard zero (`SNAP_BACK_ZERO_SLIP_RAD`), so the
softened gate can never HARD-ZERO the commitment: a car that coasts to a
standstill keeps its latch (the committed slip survives at the floor)
instead of losing it on the speed cliff. The VISIBLE slide is unaffected -
the CHASE keeps its exact-zero gate below `MIN_SPEED_MPS`, so the
published slide stays exactly 0 there (AC-047D).

The floor only binds below about 3.6 m/s (where the proportional budget
drops under it), so every speed the published lanes measure is untouched.

Parameters:
- `speedMps` - The absolute longitudinal speed (m/s).

Returns: The depth budget (rad) in `[SNAP_BACK_ZERO_SLIP_RAD,
MAX_SLIP_ANGLE_RAD]`.

### computeDriftSlideProxyRad

```ts
computeDriftSlideProxyRad(
  slipAngleRad: number,
): number
```

The deterministic slide-prominence proxy (rad): `|slip| * |GAIN - 1|`.

The proxy is the NAMED prominence floor's unit system (AC-087). It is
V-shaped in the gain with a NULL at GAIN = 1.0, so the gain must never
approach 1.0 (the 7g anti-null guard).

Parameters:
- `slipAngleRad` - The slip angle (rad).

Returns: The slide proxy (rad).

### computeDriftSlipResponseTauS

```ts
computeDriftSlipResponseTauS(
  speedMps: number,
): number
```

The committed chase RESPONSE time constant (s) at a speed.

The chase is a RELAXATION-LENGTH law (Step 14Q, AC-088): the time
constant scales with `TRACTION_TOP_SPEED_MPS / speed`, so the chase covers
a fixed DISTANCE per second and a fast car reacts FASTER (a slow car lags
more). The declared constant is the traction-top-speed reference value.

Parameters:
- `speedMps` - The absolute longitudinal speed (m/s).

Returns: The response time constant (s).

### computeGripBudgetFactor

```ts
computeGripBudgetFactor(
  slipAngle: number,
  speedMps: number,
): number
```

D1 stage 2: the tire saturation as a SPEED-AWARE BUDGET.

The budget is a function of the SLIP and the SPEED (Step 14Q, AC-088):
the same slide at a higher speed is a HARDER traction state, because the
TRACTION SEVERITY (the ONE shared scalar) grows with both. The speed
demand is charged by the SPEED SCRUB (stage 6) as well.

Parameters:
- `slipAngle` - The committed slip (rad).
- `speedMps` - The absolute longitudinal speed (m/s).

Returns: The grip budget factor in `(0, 1]`.

### computeLongitudinalSpeedScrub

```ts
computeLongitudinalSpeedScrub(
  slipAngle: number,
  speed: number,
  speedPercent: number,
): number
```

D1 stage 6: the longitudinal speed scrub.

The AC-079 law is re-homed from the retired traction loss onto the
saturating SPEED SCRUB: a sliding tire gives up the forward speed it
cannot hold. The term is zero on a straight line and grows with the slide
and with the speed, so the straight-line and low-speed handling stay
unchanged (AC-047D). The SERVICE owns the speed, so the SERVICE applies
it.

Parameters:
- `slipAngle` - The tick-start slip (rad).
- `speed` - The longitudinal speed (m/s).
- `speedPercent` - The normalized speed in `[0, 1]`.

Returns: The forward speed the tires cannot hold, in m/s per second.

### computeMomentumSlideDemandRad

```ts
computeMomentumSlideDemandRad(
  speedMps: number,
  demandedTurnRateRadPerS: number,
  availableGripShare: number,
): number
```

The MOMENTUM SLIDE DEMAND (rad): the drift depth the MOMENTUM demands
(Step 14S, AC-091).

THE CAUSE: the drift is produced by the MOMENTUM EXCESS - the lateral
acceleration the demanded turn DEMANDS minus the lateral acceleration the
tires can HOLD at the current speed - and NOT by the wheel command alone.
The excess is a SHARE in `[0, 1]` of the declared depth budget, so the
demand can never exceed the budget (the depth clamp never MASKS it) and
the demand is EXACTLY 0 at a standstill and EXACTLY 0 for a zero demanded
turn rate.

THE BASIS: the POST-traction `car.speed` (m/s) - the SAME basis the
traction law reads (`computeTractionSeverity` and
`computeDriftSlideDepthBudgetRad` both read the absolute speed) - and the
demanded turn rate in rad/s. The demanded rate is a RATE: the yaw law
resolves `steer * NEATRACING_PHYSICS_STEERING_RATE * speedPercent * dt`
(rad per tick), so the SAME product WITHOUT `dt` is the rate.

THE SHAPE: the demanded lateral acceleration is `speed * |rate|` (the
turn's own acceleration demand) and the holdable lateral acceleration is
`gripShare * NEATRACING_DRIFT_MOMENTUM_HOLD_ACCEL_MPS2` (the tires' grip
limit). The excess share is `1 - hold / demand`, clamped to `[0, 1]`:
below the grip limit the demand is 0 (the tires hold the turn), above it
the demand scales with the excess. The share is STRICTLY increasing in the
speed and in the rate, so the SAME wheel command slides DEEPER the faster
the car goes - the drift is a momentum effect, not a wheel command.

THE CROSSOVER (a DERIVED property, never a literal): the share reaches 0
where `speed * |rate| == gripShare * hold`, so the momentum stops carrying
the turn BELOW that speed. With the service-resolved rate (the rate
scales with the speed percent) and the full grip the crossover sits at
about 9.43 m/s; the commitment therefore opens a real slide from 10 m/s up
and the latch HOLDS the committed depth below the crossover (the RELEASE
law, not the momentum, unwinds a latched slide).

Parameters:
- `speedMps` - The post-traction longitudinal speed (m/s).
- `demandedTurnRateRadPerS` - The demanded turn rate (rad/s); the sign
is irrelevant (the magnitude drives the demand).
- `availableGripShare` - The available grip share in `(0, 1]` (1 = the
full grip).

Returns: The momentum slide demand (rad) in
`[0, NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD]`.

Examples:

// The 56 m/s AC-047A lane (the service-resolved 1.1118759778125444 rad/s)
// demands 0.3054164438299156 rad - above the ratified 0.2 rad floor.
computeMomentumSlideDemandRad(56, 1.1118759778125444, 1);

// The below-gate lane (10 m/s, the service-resolved 0.2247191011235955
// rad/s) demands 0.006198347107438023 rad - a REAL (non-zero) commitment
// below the MIN_SPEED gate, so the AC-089 softened gate keeps holding.
computeMomentumSlideDemandRad(10, 0.2247191011235955, 1);

### computeNextLateralVelocity

```ts
computeNextLateralVelocity(
  committed: DriftCommittedState,
  input: DriftStepInput,
): number
```

D1 stages 3-5: the committed chase, the countersteer authority and the
snap-back at the zero crossing.

Parameters:
- `committed` - The committed state AFTER the commitment step.
- `input` - The tick input.

Returns: The lateral velocity at tick end (m/s).

### computeSlipAngle

```ts
computeSlipAngle(
  lateralVelocity: number,
  speed: number,
): number
```

Resolves the bounded slip angle via `atan2(lateralVelocity, speed)`.

A stopped car reports zero slip so a bare `atan2` with zero speed
cannot create a phantom drift state.

Parameters:
- `lateralVelocity` - Lateral velocity in m/s.
- `speed` - Longitudinal speed in m/s.

Returns: Slip angle in `(-π/2, π/2)` rad, or `0` when stopped.

### computeTractionSeverity

```ts
computeTractionSeverity(
  slipAngleRad: number,
  speedMps: number,
): number
```

The TRACTION SEVERITY: the ONE speed-and-slide scalar of the drift law.

The severity is 0 on a straight line, lives in the closed unit interval
and is MONOTONIC in the slide depth. It is the shared "how much traction
is this car spending" measure: the deeper the slide AND the faster the
car, the higher the severity, and at the full slide budget on the traction
top speed it saturates at 1.

Parameters:
- `slipAngleRad` - The slip angle (rad); the sign is irrelevant.
- `speedMps` - The absolute longitudinal speed (m/s).

Returns: The severity in `[0, 1]`.

### createZeroDriftChannelState

```ts
createZeroDriftChannelState(): DriftChannelState
```

Creates the zeroed drift channel - the SPAWN contract.

The RE-ARM TOKEN starts TRUE (A-4): a fresh car may engage a drift as soon
as the driver turns the wheel. The KEY ORDER is part of the AC-047D
byte-identity contract - the serialized channel must stay byte-identical to
this factory's output.

The COLLISION BOUNCE reset is NOT this factory: a bounce is a RELEASE, not
a spawn, so it CLEARS the token via `zeroBouncedDriftChannel` in
`collision.utils.ts` (the committed slide and the pose are zeroed and
`reArmed` is false). A bounced car must not be able to re-latch on an
opposite steer alone.

### DriftChannelState

The nested drift channel carried by the physics car state: the committed
physical state and the visual pose accumulator.

### DriftCommittedState

The committed physical drift state: the slide, the latched TURN direction
and the RE-ARM token.

### DriftPoseSpringState

The visual pose accumulator - the ONLY visible-yaw source of the 14N
channel.

### DriftPoseStepInput

One visual pose tick: the tick-END sample the spring reads.

### DriftStepInput

One physical drift tick: the tick-START sample the model reads.

### NEATRACING_DRIFT_COUNTERSTEER_AUTHORITY_MPS2

The countersteer authority (m/s^2) applied AGAINST the slide. It is an
ACCELERATION: it is multiplied by `dt` to yield a velocity delta (REC-4).

### NEATRACING_DRIFT_COUNTERSTEER_MIN_SLIP_RAD

The minimum committed slip (rad) at which the countersteer authority is
applied - below it there is no slide left to catch.

### NEATRACING_DRIFT_COUNTERSTEER_MIN_STEER

The minimum countersteer steering magnitude (band 0.85-0.95): the in-model
fallback for the producer-declared countersteer intent (D2.3).

### NEATRACING_DRIFT_EXIT_RATE_RAD_PER_S

The rate (rad/s) at which the committed slip unwinds under a countersteer
(144 deg/s).

RECORDED DEVIATION (Step 14N, tune-model): the MERGE constant set and the
integration artifact name 2.0 rad/s, but 2.0 is INFEASIBLE under the RED
contract - the pose then settles after 35 ticks, OUTSIDE the ratified
30..34-tick band. The feasible band is 2.25..3.65 rad/s and this value
sits at its centre (33 ticks). The RED contract wins over the constant
set; the deviation is reported to the green slice.

### NEATRACING_DRIFT_FULL_STEER_TURN_RATE_RAD_PER_S

The RAW FULL-STEER DEMANDED TURN RATE at the TRACTION TOP SPEED (rad/s).

Step 14S (AC-091) makes the demanded turn rate an INPUT of the drift law:
the drift is caused by the MOMENTUM the commanded turn demands, so the
commitment CONSUMES the demanded turn rate. The demanded turn rate is a
RATE (rad/s) and NOT the per-tick heading delta (rad/tick) the yaw law
resolves - the two are NOT interchangeable (a unit mismatch is a silent
factor-of-sixty error).

THE VALUE IS THE YAW LAW'S OWN QUANTITY: the yaw law builds its demanded
rate from the steer-axis product `steer * NEATRACING_PHYSICS_STEERING_RATE
* speedPercent` (rad per tick once it is multiplied with `dt`), so the
SAME product WITHOUT `dt` is the demanded rate (rad/s).

MIRRORED, NOT IMPORTED: the model is PURE (it must not import the
service), so the steering rate is DECLARED here and MUST stay equal to
`NEATRACING_PHYSICS_STEERING_RATE` in `physics.step.service.ts` - the
same mirrored-constant precedent as
`NEATRACING_DRIFT_TRACTION_TOP_SPEED_MPS`.

THE VALUE IS MEASURED, NOT CARRIED: the Step 14S RED lane measured
exactly 2 rad/s for a full steer at the arcade top speed (the speed
percent is exactly 1 there). That MEASUREMENT is the RAW, FACTOR-FREE
steer-axis rate this constant names: the service softens its user rate
with the speed factor above the 10 m/s band, so the rate the SERVICE
resolves at the arcade top speed is 1.6 rad/s - the raw rate stays the
mirrored constant, and the drift model's ABSENT-rate fallback uses it.

### NEATRACING_DRIFT_GRIP_FALLOFF_RATE_PER_RAD

The rate at which the grip budget is consumed per radian of slip past the
threshold. The budget is a PROPORTIONAL law of the SLIP demand scaled by
the TRACTION SEVERITY (Step 14Q, AC-088): the deeper the slide AND the
faster the car, the less of the tire's grip budget is left for the chase.
The speed demand is ALSO charged by the SPEED SCRUB (stage 6) - the two
terms are independent (the tire budget vs the longitudinal loss).

### NEATRACING_DRIFT_GRIP_FALLOFF_THRESHOLD_RAD

The slip (rad) at which the tire grip budget starts to be consumed. Below
it the tire keeps its full lateral grip (`factor = 1`).

### NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD

The COMMITTED-SLIDE budget (rad). The 14N slide is a BUDGET of 0.45 rad
(25.8 deg), not a quarter turn (it was `Math.PI / 2`).

SPLIT CONTRACT (the GAP-N4 D1.3 escalation decision): this constant bounds
the COMMITTED slide only - `stepDriftCommitment` hard-clamps `committed.slip`
to +/- this value. The PUBLISHED slip (the slip the service publishes to
the renderer, the browser fixture and the published-slip pins) is a
DIFFERENT observable and MAY overshoot this budget by up to
`NEATRACING_DRIFT_PUBLISHED_SLIP_OVERSHOOT_RAD`. That overshoot is
INTRINSIC for every scrub coefficient c > 4/9: no coefficient that clears
the D1.3 speed-deficit floor keeps the published slip <= 0.45.

This constant is NEVER re-pinned - the 0.45 slide budget SURVIVES on the
committed slide.

### NEATRACING_DRIFT_MIN_SLIDE_PROXY_RAD

The NAMED slide-prominence floor (rad) on the deterministic unit proxy
`|slip| * |GAIN - 1|` (AC-087).

### NEATRACING_DRIFT_MIN_SPEED_DEFICIT_KMH

The D1.3 speed-deficit FLOOR (km/h): the drift must bleed at least this
much speed across a 600-tick drift. It ACCOMPANIES (and never replaces)
the re-pinned `TRACTION_LOSS_MAX_SPEED_KMH` ceiling - the ceiling is a
SPEED bound, this export is a SCRUB-RATE bound, and the D1.3 check needs
both.

### NEATRACING_DRIFT_MIN_SPEED_MPS

The minimum longitudinal speed (m/s) at which the drift model's VISIBLE
slide starts: below it the lateral velocity is held at exactly 0
(AC-047D).

Step 14Q (AC-089) SOFTENS the gate on the COMMITMENT: the committed
depth and the entry rate are PROPORTIONAL to the speed below the gate
(they are 0 at a standstill), so the slide ramps in instead of freezing
on a hard cliff. The gate still holds the VISIBLE slide at exactly 0.

### NEATRACING_DRIFT_MOMENTUM_HOLD_ACCEL_MPS2

The LATERAL ACCELERATION (m/s^2) a SLIDING tire can HOLD: about 0.2 g.

Step 14S (AC-091): the momentum excess is the lateral acceleration the
demanded turn DEMANDS minus the lateral acceleration the tires can HOLD.
The hold is the SLIDING tire's lateral capacity - a FRACTION of the ~1 g
peak grip, because a tire that is sliding cannot build its peak lateral
force - and it does NOT scale with the speed: a tire holds about the same
lateral acceleration at every speed.

CALIBRATED INSIDE THE MEASURED WINDOW: the Step 14S RED lane MEASURED the
feasible demand bounds and this hold is the calibration that lands the law
inside them (the value is never carried as a literal):

  - the 89 m/s reference lane (the service-resolved full-steer 1.6 rad/s
    at the full grip) demands 0.4436797752808989 rad, UNDER the 0.45 rad
    depth budget;
  - the 56 m/s AC-047A lane (the service-resolved 1.1118759778125444
    rad/s) demands 0.3054164438299156 rad, ABOVE the ratified 0.2 rad
    floor;
  - the 25 m/s strict-deeper pair (the service-resolved 0.5404636609301664
    rad/s and its double 1.0809273218603328 rad/s) demands
    0.1200195737277077 rad and 0.13044554944837675 rad, both strictly
    INSIDE the 0.14087152516904583 rad depth budget of that lane
    (otherwise the depth clamp would MASK the strict-deeper relation);
  - the 10 m/s below-gate lane (the service-resolved
    0.2247191011235955 rad/s) demands 0.006198347107438023 rad - a REAL
    (non-zero) commitment, so the ratified AC-089 softened gate keeps
    holding below the MIN_SPEED gate;
  - the standstill lane and the zero-rate lane demand EXACTLY 0 rad.

THE MEASURED COMMITTED SLIDES (the 180-tick model lanes): 0.44745539952559543
rad at the 89 m/s reference lane, 0.3087228648902781 rad at the 56 m/s
AC-047A lane, 0.1208114199849737 rad and 0.13084147257700976 rad at the
25 m/s strict-deeper pair and 0.006198347107438023 rad at the 10 m/s
below-gate lane.

### NEATRACING_DRIFT_POSE_ARTICULATION_SLEW_PER_S

The articulation slew (rad/s, band 15-30): the ONLY genuinely new constant
of the visual half, and the bound the per-tick pose-target step rides on.

### NEATRACING_DRIFT_POSE_DAMPING_RAD_PER_S

The pose-spring damping (rad/s): zeta = 0.35 (about 31 % first overshoot).

### NEATRACING_DRIFT_POSE_ENVELOPE_RAD

The DECLARED POSE ENVELOPE (rad): the largest body yaw the drift model
can declare.

RE-DERIVED (Step 14Q, AC-087) from the PUBLISHED slip ceiling - the OLD
declaration was derived from the COMMITTED ceiling (0.45 * 0.5 + 0.08 =
0.3050 rad), so the PUBLISHED slip (the committed slip plus the
chase-lag overshoot) could BREACH the declared bound. The pose target is
`publishedSlip * GAIN` plus the steer-yaw term, and the published slip is
what the pose spring actually reads (the pose is driven by the
PUBLISHED slip, never by the committed command), so the ceiling the
envelope is derived from MUST be the PUBLISHED one.

THE REGIME-INDEPENDENT CEILING: the published slip carries TWO named
ceilings - the ON-ROAD one (0.452 rad, the committed budget plus the
0.002 rad chase-lag overshoot) and the LOOSER OFF-ROAD one (0.493 rad,
the measured off-road peak plus its margin). The off-road drag drives the
same lag from a FALLING denominator, so the off-road published slip
reaches a DEEPER slide than the on-road one and the pose target can
exceed the on-road bound. The declaration is therefore derived from the
MAXIMUM of the two published ceilings, so it holds in EVERY regime:
`0.493 * 0.45 + 0.08` = 0.30185 rad.

The pose is NOT clamped to the envelope: the envelope DECLARES the bound,
it does not enforce it (a clamp would fight the spring and flatten the
pop, which IS the feature).

### NEATRACING_DRIFT_POSE_STEER_YAW_FULL_SLIP_RAD

The articulation ARM (rad of slip): the slip at which the steer-yaw term
reaches its full amplitude.

### NEATRACING_DRIFT_POSE_STEER_YAW_RAD

The articulation amplitude (rad): the steer-yaw term of the pose target.

### NEATRACING_DRIFT_POSE_STIFFNESS_RAD_PER_S2

The pose-spring stiffness (rad/s^2): omega = 18 rad/s (2.86 Hz).

### NEATRACING_DRIFT_PROXY_ENGAGEMENT_MIN_RAD

The drift-engagement gate (rad of proxy): the slide must be visible
before the drift is announced.

### NEATRACING_DRIFT_PROXY_MIN_FRAMES

The number of CONSECUTIVE frames the engagement gate must survive.

### NEATRACING_DRIFT_PUBLISHED_SLIP_MAX_RAD

The ON-ROAD PUBLISHED-SLIP CEILING (rad) - the DERIVED bound the
published slip is ASSERTED against in the ROAD regime.

DERIVED, NEVER A LITERAL: the ceiling is the COMMITTED slide budget
(`NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD` = 0.45 rad) plus the chase-lag
overshoot allowance
(`NEATRACING_DRIFT_PUBLISHED_SLIP_OVERSHOOT_RAD` = 0.002 rad), so a
re-pin of EITHER export moves the bound with it (0.452 rad today).

IT IS AN ASSERTION BOUND, NEVER A CLIP: nothing in the service clamps
the published slip to this value. The published slip keeps its
physical value, and the bound is the single source the jest pins and
the browser fixture mirror assert against.

THE REGIME (why the bound is road-scoped): the published slip is the
committed slide carried through the first-order chase, and the chase
LAGS a target that shrinks as the speed scrub bleeds speed, so the
published slip peaks slightly above the committed budget. On the road
the measured peak is 0.4511617369362305 rad - 0.00084 rad of headroom
under this ceiling. OFF the road the off-road drag drives the same lag
from a FALLING denominator, and the measured raw peak reaches
0.4872167352892035 rad (the post-RC-1 off-road envelope measurement),
so the OFF-ROAD regime carries its own DECLARED named ceiling instead
of this one.

Example:

```ts
const withinOnRoadBound =
  Math.abs(car.slipAngle) <= NEATRACING_DRIFT_PUBLISHED_SLIP_MAX_RAD;
```

### NEATRACING_DRIFT_PUBLISHED_SLIP_OFFROAD_MAX_RAD

The OFF-ROAD PUBLISHED-SLIP CEILING (rad) - the DECLARED named bound
the published slip is ASSERTED against in the OFF-ROAD regime (grass,
curb and the road/grass transitions).

WHY A SECOND, LOOSER BOUND: the off-road drag drives the same
chase-lag overshoot from a FALLING denominator, so the off-road
published slip peaks well above the ON-ROAD ceiling. The published
slip keeps its PHYSICAL value - this is an ASSERTION bound, NEVER a
clip - so the off-road regime is asserted against its own named
ceiling instead of being silently clipped into the on-road one.

DERIVED FROM A MEASUREMENT (the ratified deterministic rule: the
ceiling is the smallest 0.001-grid value at or above the measured raw
off-road peak plus a 0.005 rad margin): the W2a post-RC-1 off-road
envelope peak is 0.4872167352892035 rad (the jest lane, seed 42, the
top-speed start, a full-throttle full-steer drift input, swept over
the seven grass/curb/transition lanes), so 0.4872167352892035 + 0.005
= 0.4922167352892035 and the declared ceiling is 0.493 rad.

THE MARGIN absorbs the chase-lag drift of a lane that is not the
measured one. A re-measurement that moves the raw peak by more than
the margin FORCES a re-declaration of this value - a RECORDED
decision, never a silent clip.

Example:

```ts
const withinOffRoadBound =
  Math.abs(car.slipAngle) <= NEATRACING_DRIFT_PUBLISHED_SLIP_OFFROAD_MAX_RAD;
```

### NEATRACING_DRIFT_PUBLISHED_SLIP_OVERSHOOT_RAD

The PUBLISHED-SLIP observation tolerance (rad).

This export documents the PUBLISHED-slip OBSERVATION tolerance - it is NOT
the slide budget. The slide budget is
`NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD`, and it still hard-clamps the
COMMITTED slide.

MECHANISM (MEASURED, not inferred): the published slip is the committed
slide carried through the first-order chase (tau 45 ms). The chase LAGS
ABOVE a target that SHRINKS as the speed scrub bleeds speed, so the
published slip peaks slightly above the committed budget - the measured
`vLat / target` ratio (the target taken from the TICK-START speed) is
1.002317 at the peak, plus the post-scrub denominator on top of it.

CALIBRATION: the measured peak is 0.4511617369362305 rad at t42 with
`NEATRACING_DRIFT_SPEED_SCRUB_COEFF_PER_S` = 0.55. The calibration envelope
is EXHAUSTED at c ~= 0.61 (measured: the peak is 0.45182287 rad at
c = 0.61, so only 0.00018 rad of headroom is left under the 0.452 ceiling;
the exact crossing is at c ~= 0.626), so ANY coefficient change FORCES a
re-derivation of this tolerance.

The 0.45 slide budget SURVIVES on the COMMITTED slide: the hard clamp in
`stepDriftCommitment` (drift.model.ts:479-483) stays and is NEVER
re-pinned.

### NEATRACING_DRIFT_REARM_HOLD_TICKS

The number of ticks the NEUTRAL wheel HOLDS the committed slide before the
latch RELEASES it (Step 14Q, AC-089).

The countersteer-hold constant this replaces was DEAD: it was exported and
read by nobody, and the ratified `DriftCommittedState {slip, turnSign,
reArmed}` has no home for a debounce counter. Step 14Q deletes it and
replaces it with a constant that HAS a reader: the NEUTRAL release
unwinds the committed slide from the full slide budget to the hard zero
over this window, so the release rate is
`(MAX_SLIP_ANGLE_RAD - SNAP_BACK_ZERO_SLIP_RAD) / (REARM_HOLD_TICKS * dt)`.

The latch therefore RE-ARMS: the turn sign returns to 0, the slide unwinds
to the hard zero and the RE-ARM token becomes reachable again, instead of
freezing on the `Math.sign(0) === -turnSign` defect.

### NEATRACING_DRIFT_SLIDE_SATURATION_SPEED_MPS

The speed (m/s) at which the tire reaches the FULL slide budget: the
committed depth is `MAX_SLIP_ANGLE_RAD * min(1, speed / SATURATION_SPEED)`.

DERIVED (Step 14Q, AC-088): the slide depth is PROPORTIONAL to the speed
(a slow car slides less) and SATURATES at the declared budget, so the
committed slide is bounded by the AC-087 budget at every speed.

THE VALUE IS MEASURED, NOT CARRIED: it is the knee that puts the settled
EQUILIBRIUM speed deficit inside the ratified 14Q band. The equilibrium is
a MEASURED value and is never a carried constant.

THE MEASUREMENT (the pinned RED-7 lane: the road-only resolver, the full
throttle, the full LEFT steer, the arcade top speed as the entry speed,
dt 1/60): the deficit reads 32.91154453692649 km/h at 3000 ticks AND at
3600 ticks (the two readings are BIT-IDENTICAL - the lane has settled),
32.91154453591922 km/h at 2400 ticks, 32.91018739912641 km/h at 1200
ticks and the 600-tick TRANSIENT 32.556920153495916 km/h. The settled
value is 0.00054 km/h from the pinned 32.911 km/h equilibrium (inside the
0.002 km/h band) and the 600-tick transient is 0.354 km/h outside it, so
the anti-transient guard holds. The knee is sensitive: the settled
deficit moves about 319 km/h per rad of settled depth, so the knee is
re-derived (never carried) if the depth law or the scrub changes.

The measurement and the resulting pin are recorded in the plan's ledger.

### NEATRACING_DRIFT_SLIP_RESPONSE_TAU_S

The time constant (s) of the committed chase's exponential response at the
TRACTION TOP SPEED.

Step 14Q (AC-088) makes the response a RELAXATION-LENGTH law: the real
time constant is `TAU * (TRACTION_TOP_SPEED_MPS / speed)`, so the chase
covers a FIXED DISTANCE per second and a fast car reacts FASTER (a slower
car lags more). The declared constant stays the TOP-SPEED reference value.

### NEATRACING_DRIFT_SNAP_BACK_ZERO_SLIP_RAD

The committed slip (rad) at or below which the slide is a HARD ZERO: the
lateral velocity is held at exactly 0 there.

### NEATRACING_DRIFT_SPEED_SCRUB_COEFF_PER_S

The longitudinal speed-scrub coefficient (per rad of slip, per m/s of
speed, per unit of the speed percentile). It replaces the retired
proportional traction loss and it is zero on a straight line.

DERIVATION (the GAP-N4 D1.3 escalation decision; MEASURED, not carried):
the engine (17.8 m/s2 = MAX_SPEED / 5) WINS while c <= 4/9 = 0.444444 -
that is the exact clamp-bound crossover, and the landed 0.25 coefficient sat
inside that regime, so the 0.25 lane was green only because the D1.3 pin
was VACUOUS. Above the crossover the scrub wins and the interior
equilibrium is s* = sqrt(3520.4444 / c).

At c = 0.55 the terminal speed is 79.85792723069594 m/s (287.49 km/h) and
the D1.3 speed deficits are 32.55473 km/h

### NEATRACING_DRIFT_STEER_ENGAGE_THRESHOLD

The steering-axis magnitude that ENGAGES a drift (band 0.4-0.6).

### NEATRACING_DRIFT_STEER_NEUTRAL_THRESHOLD

The steering-axis magnitude below which the wheel counts as NEUTRAL and the
RE-ARM token is latched again (band 0.1-0.2).

### NEATRACING_DRIFT_TRACTION_TOP_SPEED_MPS

The TRACTION TOP SPEED (m/s) the speed-proportional laws are normalized
to: the arcade top speed (89 m/s).

The model is PURE: it does not import the service, so the reference speed
is DECLARED here and MUST stay equal to
`NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S`. Every speed-proportional law
in this file (the entry rate, the chase response rate, the severity and
the slide depth budget) is normalized by it, so a re-pin of the arcade top
speed FORCES a re-derivation of those laws - a RECORDED decision (a
ledger row), never a silent drift.

### NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD

The visual yaw gain: the base gain of the pose target. It owns the body
yaw (the pose target is `committedSlip * GAIN + POSE_STEER_YAW_RAD *
articulation`).

RE-DERIVED (Step 14Q, AC-087): 0.5 -> 0.45. The body yaw SHAPE is
retained (the gain still multiplies the committed slip) and the pose is
NOT clamped.

WHY DOWN AND NOT UP: the plan's 0.75 recommendation is INFEASIBLE against
the ratified pins. The slide-prominence proxy is `|slip| * |GAIN - 1|` and
it is V-shaped in the gain with a NULL at GAIN = 1.0, so the ratified
anti-null guard (`|GAIN - 1| >= 0.5`) forces `GAIN <= 0.5`. At 0.75 the
largest reachable proxy is `0.45 * 0.25 = 0.1125` rad, BELOW the
ratified 0.2 rad prominence floor - the drift would stop being announced,
a semantic break, not a re-pin. At 0.45 the guard still holds
(`|0.45 - 1| = 0.55 >= 0.5`) and the floor stays reachable
(`0.45 * 0.55 = 0.2475` rad, 24 % above the floor).

THE MEASURED FEEL (the pinned SCN-DRIFT-01 scenario, the 180-tick build
hold and the 120-tick countersteer, dt 1/60, the top speed): the far-side
pop is 9.0289 deg, the pose SETTLES in 33 ticks (inside the ratified
30..34 band), the per-tick pose-target step peaks at 2.0435 deg (under
the 2.2 deg bound), the per-tick articulation step peaks at 1.5279 deg
(exactly at the slew bound) and the prominence proxy peaks at 0.2475 rad
and stays above the 0.2 rad floor for 174 CONSECUTIVE ticks. The slide
never mirrors: the largest signed slip after the countersteer onset is 0.

The invalidated literal (`expect(GAIN).toBe(0.5)`) is reported to the
green slice as ONE ledger row.

### resolveDemandedTurnRateRadPerS

```ts
resolveDemandedTurnRateRadPerS(
  input: DriftStepInput,
): number
```

Resolves the DEMANDED turn rate (rad/s) of a tick (Step 14S, AC-091).

An EXPLICIT rate on the input is AUTHORITATIVE: an explicit 0 is a real
decision (the driver demands NO turn, so the momentum demands NO slide),
and an explicit non-zero rate is the turn the service resolved.

An ABSENT rate falls back to the STEERING AXIS: the RAW, FACTOR-FREE
steer-axis product `steer * NEATRACING_DRIFT_FULL_STEER_TURN_RATE_RAD_PER_S
* speedPercent` (rad/s) - the mirrored constant names the same raw
full-steer rate the yaw law builds its steer-axis product from. The
service softens its user rate with the speed factor above the 10 m/s
band, so this fallback is NOT the service's rate there; the PRODUCTION
call site always supplies the resolved rate, so the fallback only serves
a caller that carries no rate at all.

Parameters:
- `input` - The tick input.

Returns: The demanded turn rate (rad/s); the sign follows the steer axis.

### stepDriftCommitment

```ts
stepDriftCommitment(
  prev: DriftCommittedState,
  input: DriftStepInput,
): DriftCommittedState
```

D1 stages 0-1: the SOFTENED MIN_SPEED gate and the MOMENTUM commitment.

The latch is a three-state machine. ENGAGE needs the steer engaged, a
committed slip of exactly 0 and the RE-ARM token; the commitment is the
MOMENTUM DEMAND (Step 14S, AC-091): `slip = -turnSign * demand` (a LEFT
turn drives the slip NEGATIVE) and the committed depth is bounded by the
speed-proportional DEPTH BUDGET. RELEASE (at the hard zero, or when the
countersteer or the NEUTRAL wheel unwinds the slide) clears the token, so
an opposite steer ALONE can never re-latch and no phantom mirrored drift
can appear.

THE DRIFT IS CAUSED BY THE MOMENTUM, NOT COMMANDED BY THE WHEEL: the
committed slide is the MOMENTUM EXCESS of the tick (the lateral
acceleration the demanded turn demands minus the lateral acceleration the
tires can hold, scaled by the depth budget). The OLD speed-proportional
ENTRY-RATE COMMAND is DELETED (its named constant and its named function
are gone) - there is NO deferred cleanup.

The MIN_SPEED gate is SOFTENED, never removed (Step 14Q, AC-089): below
the gate the commitment is still allowed, but the momentum demand makes
the slide ramp in with the weight at speed instead of freezing on a hard
cliff. The VISIBLE slide stays exactly 0 below the gate - the CHASE keeps
the exact-zero gate (AC-047D).

Parameters:
- `prev` - The committed state at tick start (read-only).
- `input` - The tick input.

Returns: The committed state at tick end.

### stepDriftPoseSpring

```ts
stepDriftPoseSpring(
  prev: DriftPoseSpringState,
  input: DriftPoseStepInput,
): DriftPoseSpringState
```

D1 stage 7: the merged visual pose spring.

The pose target is `committedSlip * GAIN + POSE_STEER_YAW_RAD *
articulation`. The sign is `+committedSlip` - the upstream `-committed` is
a SIGN ERROR under the pinned LEFT = negative-slip convention - and the
articulation is the slew-limited steer-yaw term. The pose is NOT clamped:
a clamp would fight the spring and flatten the pop, which IS the feature.

Parameters:
- `prev` - The pose state at tick start (read-only).
- `input` - The pose tick input (the tick-END slip).

Returns: The pose state at tick end.

## physics/ai.autopilot.ts

NeatRacing AI autopilot (slice P1-physics-drive, user-gap patch batch 108;
Step 14M, slice `P1-ai-centerline`: AC-083..AC-086).

The step-14B user gap: the served `/` page must open as a running race,
not a parked scene. Phase 1 ships one deterministic arcade autopilot that
drives the human car around the generated closed loop while the race-mode
selector is on the `ai` option; switching to `manual` hands control back
to the keyboard sample seam with no other contract change.

Pure seam: the function reads only the car's kinematic snapshot and the
frozen track spline samples, and returns one analog step input. No
mutation, no RNG, no wall-clock reads, so the same (car, samples) pair
yields byte-identical inputs for the batch-83 replay contract.

Steering law: chase a lookahead sample on the lane centerline ahead of
the nearest spline sample, steer proportional to the wrapped heading
error (positive error steers counterclockwise per the step-input
contract), and brake when the error exceeds a named arcade threshold.

THE RACING LINE (Step 14M, AC-083..AC-086): the autopilot now CONSUMES
the deterministic racing-line table of
`examples/neatRacing/track/track.racing-line.ts` - the SAME table the
Phase 2 observation assembler and the scripted controller read, so there
is NO second geometry in the repo:

- the STEER TARGET is displaced by the table's SIGNED LATERAL OFFSET
  (positive to the LEFT of travel, exactly zero on a straight), so the
  AI drives the racing line instead of the raw spline point;
- a CENTERLINE-CORRECTING steer term is added, proportional to the
  car's OWN signed lateral offset to the road centerline and therefore
  EXACTLY zero on the centerline (the DECIDED R1 reading of AC-083);
- a CURVATURE FEED-FORWARD term is added, proportional to the table's
  published signed curvature of the road ahead and therefore exactly
  zero on a straight, so the AI commands the steer the corner ahead
  actually needs (the arcade model turns at a speed-proportional rate,
  so the pure-pursuit chase alone under-steers every corner);
- the THROTTLE/BRAKE follow the table's CURVATURE-DERIVED TARGET SPEED,
  so the AI slows for the corners ahead instead of entering them at the
  top speed (that published target is a BRAKING INTENT - the 12 m
  window is shorter than the braking distance, so the brake is issued
  before the corner and the car enters it above the target);

THE RETAINED LAW: the 12 m lookahead and the 2.2 rad^-1 heading-error
gain are UNCHANGED, the named `NEATRACING_AUTOPILOT_*` constants keep
their names, and the emergent speed behaviour on a straight is the same
as before (full throttle, no brake, at the top speed).

Turn-rate channel (Step 14T, AC-093): the autopilot drives the physics
through the DEDICATED turn-rate-percent channel, never through the user
steer axis. The percent it sets is the SIGNED, BEHAVIOUR-PRESERVING
re-expression of the rate the AI already asked for
(`100 x steerAI x speedPercent`), so the physics resolves
`rate = STEERING_RATE x (percent / 100)` - the AI keeps the FULL range
(a percent above 100 commands a rate above the nominal rate) and is
EXEMPT from the user steer-axis softening factor.

THE COMMAND SUFFICIENCY (Step 14Y, AC-098): the arcade law scales the
commanded rate WITH the speed, so the model realizes a CONSTANT 44.5 m
minimum turn radius - every corner tighter than that is UNHOLDABLE at
ANY speed, and the AI used to saturate its steer axis, brake to a
STANDSTILL at the corner entry and never complete a lap. The AI's turn
DEMAND is now the UNCLAMPED sum of the road-under-the-car curvature
feed-forward and the tracking chase, expressed through the SAME percent
channel: the demanded rate is `v x curvature` (the kinematic
requirement of the corner the car is ON) plus the tracking correction,
so a percent above 100 commands the rate the corner actually needs.
The published steer axis stays CLAMPED (the drift and the replay rails
read it), so the steer law itself is UNCHANGED - only the RATE it
commands grows, and only where the steer axis can no longer express
the road.

THE PURSUIT EXPECTATION (Step 14Y, AC-098): the road the AI follows
turns, so the aim point it chases sits OFF the tangent the car drives
on - that offset is EXPECTED geometry and not a driving mistake. The
expectation is the racing line's OWN chord angle at the car (the angle
between the line's chord to the aim point and the line's own tangent at
the car), READ from the same published line the AI aims at - never a
formula, never a tuned half-angle. The chase term, the throttle cut and
the error brake therefore read the EXCESS error (the heading error
minus that expectation), so a car that tracks the line reads ZERO error
in a corner and the AI can hold a corner at speed instead of coasting
and braking for the road it is following.

THE NEVER-STOP FLOOR (Step 14Y, AC-098(c)): at or below the physics'
own off-road cruise floor the AI never brakes, and on a road that turns
it never coasts either - so the AI can never buy its way out of a
corner by stopping in it.

THE ROAD-BOUND LINE (Step 14Y, AC-098(d)): the AI drives the RACING
LINE the table publishes, and that line is built on the track's SPLINE
- but the spline is NOT the road: the road is the CONTROL POLYGON the
physics classifies and contains against, and the Catmull-Rom spline
that smooths the polygon OVERSHOOTS it at every corner (MEASURED on
the served track: 42 of 144 samples, 32.3% of the arc, up to 6.33 m
BEYOND the road's own edge). An AI that drives the raw spline therefore
drives OFF the road for a third of the lap - onto the grass, where the
off-road floor can be lost and never regained - and no lap is ever
completed. The line the AI drives is therefore the spline HELD ON THE
ROAD: every sample is clamped LATERALLY into the road core the edge
seam publishes (the very core the physics classifies as road), so the
aim point, the lateral offset, the pursuit expectation and the
lookahead walk all read ONE line the car can really drive - while the
racing-line TABLE (the curvature, the target speed and the apex
offset) keeps being the published one, consumed through the same seam.

### AutopilotCarSnapshot

Minimal kinematic snapshot consumed by the autopilot step (no RNG fields).

### buildRoadBoundLine

```ts
buildRoadBoundLine(
  samples: readonly SplineSample[],
): readonly SplineSample[]
```

Builds the ROAD-BOUND LINE of a layout: the frozen samples with every
sample clamped LATERALLY into the road core the edge seam publishes.

THE CLAMP (Step 14Y, AC-098(d)): the sample's signed offset from its
OWN control chord (the road's centreline, positive to the LEFT of
travel) is clamped into the road core of the side the sample sits on -
the published half-width of that side MINUS the curb band, because the
physics classifies a point as ROAD only inside that core. The sample
is then moved LATERALLY onto its clamped offset, so the line keeps the
spline's shape and its parametrisation and only the off-road
overshoot is shaved.

THE SEAM, NOT A SECOND GEOMETRY (AC-082 / AC-086): the bound is the
one the edge seam publishes - the loop-global half-width PAIR of the
chord the sample owns - so the line the AI drives can never be off the
road the physics reads.

THE INACTIVE CLAMP IS BIT-IDENTICAL: a sample that is already inside
the road core keeps its EXACT published coordinates (the very same
sample object), so a layout whose spline already lies on its road
drives the very line it always drove.

Parameters:
- `samples` - Frozen track spline samples.

Returns: The road-bound line, one sample per published sample.

### clampRatio

```ts
clampRatio(
  value: number,
): number
```

Clamps a control ratio into the closed unit interval.

Parameters:
- `value` - Raw ratio.

Returns: Value clamped into `[0, 1]`.

### clampSteer

```ts
clampSteer(
  value: number,
): number
```

Clamps a steering command into the analog axis range.

Parameters:
- `value` - Raw steering command.

Returns: Value clamped into `[-1, 1]`.

### ControlChordRing

The CONTROL chord ring the frozen samples describe (the road's own
chords).

THE RECONSTRUCTION IS BIT-IDENTICAL to the spec's own chord list: the
first sample of a control segment carries an interpolation factor of
ZERO, so it sits EXACTLY on that segment's control point, and the
control chord of segment `k` runs from control point `k` to control
point `k + 1` with the width that segment publishes.

### lookaheadSampleIndex

```ts
lookaheadSampleIndex(
  samples: readonly SplineSample[],
  nearestIndex: number,
): number
```

Resolves the lookahead spline sample index a named distance ahead of the
nearest sample along the closed loop (wrapped walk accumulating the
centerline chord lengths).

Parameters:
- `samples` - Frozen track spline samples.
- `nearestIndex` - Index of the nearest sample (the walk origin).

Returns: The index of the lookahead sample.

### LookaheadWindow

The lookahead WINDOW reading of the racing line.

### nearestSampleIndex

```ts
nearestSampleIndex(
  samples: readonly SplineSample[],
  x: number,
  y: number,
): number
```

Resolves the index of the spline sample nearest to the car (squared
distance scan over the frozen list; the sample count is small enough
that a linear scan beats a maintained spatial index for this arcade
autopilot).

Parameters:
- `samples` - Frozen track spline samples.
- `x` - Car world x.
- `y` - Car world y.

Returns: The nearest sample index (0 when the list is empty).

### NEATRACING_AUTOPILOT_BRAKE

Autopilot brake axis under steering control (`[0, 1]`).

### NEATRACING_AUTOPILOT_BRAKE_GAIN_PER_SPEED_RATIO

The brake gain from the relative speed excess over the curvature-derived
target speed (1).

THE PROPORTIONAL BRAKE (Step 14M, AC-084): the brake is the speed
excess over the target speed as a RATIO of that target, so the AI
brakes in proportion to how far it is over the target and the brake
saturates at the full axis when the car is twice the target speed.

### NEATRACING_AUTOPILOT_CENTERLINE_STEER_GAIN_PER_M

The centerline-correcting steer gain from the signed lateral offset
(1/m).

THE CENTERLINE-CORRECTING TERM (Step 14M, AC-083): the term is
`-gain x the signed lateral offset to the road centerline` (the offset
is POSITIVE to the LEFT of travel), so it is EXACTLY zero on the
centerline and always steers the car BACK toward the centerline.

THE GAIN IS DERIVED, NOT TUNED: it is the RETAINED 2.2 rad^-1 heading
gain divided by the RETAINED 12 m lookahead, so the correction is the
lateral offset expressed as the heading error the retained lookahead
would produce (`offset / lookahead`) and scaled by the retained gain.
The classic Stanley cross-track term therefore reuses the two
constants the slice RETAINS, and the term stays exactly zero on the
centerline because the offset is.

### NEATRACING_AUTOPILOT_COMMAND_SUFFICIENCY_MARGIN

The command-sufficiency margin of the AI's turn demand (1).

THE MARGIN (Step 14Y, AC-098): the curvature feed-forward supplies the
EXACT kinematic requirement of the corner the car is on (`v x
curvature`), so a demand built from it alone MEETS the requirement with
NO slack - and a demand that merely meets it leaves the corner on the
boundary, where any tracking error and any float noise lose the
corner. The margin is the small slack that keeps the demanded rate
STRICTLY above the requirement, so the corner is HOLDABLE rather than
borderline.

THE VALUE: a tenth of slack - small enough that the AI still drives the
published racing line (the curvature it realizes stays within a tenth
of the road's own) and large enough that the demanded rate is never the
requirement minus a rounding error.

### NEATRACING_AUTOPILOT_CURVATURE_FEEDFORWARD_GAIN_M

The curvature feed-forward gain of the racing line (m).

THE FEED-FORWARD (Step 14M, AC-083..AC-085): the racing line publishes
the SIGNED curvature of the road ahead, and the physics realizes a turn
radius of `max speed / (steering rate x steer)`, so the steer command
that HOLDS a corner of radius `R` is `steer = (max speed / steering
rate) x curvature`. The gain is therefore the model's own minimum turn
radius, DERIVED from the physics constants - never a tuned number.

WHY THE CHASE ALONE IS NOT ENOUGH (MEASURED, not a formula): the
arcade model turns at a rate proportional to the speed, so the chase's
CONTROL SCALE is `max speed / (steering rate x the 2.2 rad^-1 gain)` =
`89 / (2 x 2.2)` = 20.227 m - wider than the retained 12 m lookahead.
That scale is a CONTROL magnitude, NOT a curvature law: the
pure-pursuit `L^2 / (2R)` offset formula does NOT describe this
proportional-heading law, so NO offset is derived from it. The offset
was MEASURED instead: on the R = 60 m lap fixture the retained chase
law alone settles at a STEADY state of 2.6368 m (inside the 3 m
AC-085 bound) but peaks at 3.3230 m on the corner-entry TRANSIENT
(outside it), and the PRE-14M model on the same fixture read 6.000 m -
the pre-14W containment's drivable edge (the containment is RETIRED
since AC-090 VERDICT 8). The feed-forward supplies the corner's own
steer command, so the AI holds the corner instead of under-steering
toward the band.

THE ZERO: the published curvature is EXACTLY zero on a straight, so the
term adds nothing on a straight and the emergent straight-line
behaviour is unchanged.

### NEATRACING_AUTOPILOT_FULL_BRAKE_ERROR_RAD

Full-brake heading-error threshold above which the car brakes.

### NEATRACING_AUTOPILOT_FULL_THROTTLE_ERROR_RAD

Full-throttle heading-error threshold below which the car holds throttle.

### NEATRACING_AUTOPILOT_LOOKAHEAD_M

Lookahead distance along the loop from the nearest spline sample (m).

### NEATRACING_AUTOPILOT_MINIMUM_DRIVING_SPEED_UNITS_PER_S

The AI's NEVER-STOP speed floor (m/s).

THE NEVER-STOP FLOOR (Step 14Y, AC-098(c)): the AI never BRAKES a car
at or below this speed, and on a road that turns it never COASTS one
either, so the AI can never buy its way out of a corner - or out of an
off-road excursion - by STOPPING in it.

WHY THE FLOOR IS THE PHYSICS' OWN, PLUS ONE BRAKING TICK (MEASURED,
not a formula): the physics floors an off-road car only while the car
ENTERS the tick at or above `NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S`
and is not braking - a coasting tick below that floor lets the car fall through
the floor's gate, and the off-road drag is stronger than the engine
(17.8 m/s^2 against 44.5 m/s^2), so the car then decays to a
standstill on the grass and CANNOT recover.

THE AI'S OWN BRAKE IS WHAT STRANDED IT THERE: the brake is NOT
floor-protected, so a brake applied AT the floor drops the car BELOW
it within a single tick and the floor's gate is lost for good - on the
served track the brake measured the car decaying 21.21 -> 18.10 ->
7.42 -> 0.30 -> 0.00 m/s and stopping for good, with the throttle
pinned at 1. The floor therefore sits ONE FULL-BRAKING TICK ABOVE the
physics' own floor: the model sheds
`|NEATRACING_PHYSICS_BRAKING_UNITS_PER_S2| x NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS`
= 1.4833 m/s in a single tick, so the AI RELEASES the brake while the
car is still above the floor and the physics' own floor holds it there
- the AI can never brake a car through the floor it must not lose.

THE VALUE IS DERIVED, NOT TUNED: it IS the physics' own off-road
cruise floor plus the ONE tick of braking the model can apply inside
it, so the AI never drives a car slower than the speed the physics
itself refuses to let a car fall below - and never brakes one into
that speed either.

### NEATRACING_AUTOPILOT_STEER_GAIN_PER_RAD

Proportional steering gain from wrapped heading error (rad^-1).

### NEATRACING_AUTOPILOT_THROTTLE

Autopilot throttle axis under steering control (`[0, 1]`).

### NEATRACING_AUTOPILOT_THROTTLE_CUT_GAIN_PER_SPEED_RATIO

The throttle-cut gain from the relative speed excess over the
curvature-derived target speed (1).

THE PROPORTIONAL THROTTLE CUT (Step 24, `P4-S2-DRIVE-LAW`): the
throttle sheds the speed excess through the SAME excess-ratio
machinery the AC-084 brake law reads - the cut is the excess over the
target speed as a ratio of that target - so the drive keeps slowing
authority on the ONE axis the fleet-control map cannot strip (the
AC-029 seam forces the brake to zero and drops the turn rate; the
throttle and the steer pass through untouched).

THE CEILING: the cut saturates at the full coast when the drive is
`1 + 1/gain` times the target speed, so the gain sets the speed
ceiling the throttle axis holds a corner at and the band between the
target and the ceiling still runs on proportional power.

THE VALUE (MEASURED CONSTRAINT, not tuned): the AC-084 note measured
the R = 60 m fixture entering 31% over the target and running 3.13 m
wide - OUTSIDE the 3 m AC-085 bound - so a ceiling that held a whole
corner that far over the published target would keep the drive on the
grip's losing side for the length of the corner. Eight puts the full
coast at 12.5% over the target - well inside the measured wide-run
account - and keeps a 12.5% band of proportional power between the
target and the ceiling, so the convoy still cruises at full power on
the straights the target speeds are published for.

THE FLOOR-STRANDING GUARD: the cut is gated by the never-stop floor
(the same gate the AC-084 brake law reads) - at or below the floor the
AI never slows through the new axis either, and the AC-084
never-brake-through-the-floor semantics stay untouched.

### NEATRACING_AUTOPILOT_TURN_RATE_PERCENT_SCALE

The percent scale of the AI turn-rate channel (percent per unit of the
signed AI turn command).

A percent of `100` commands the NOMINAL turn rate (the steering rate),
a percent of `0` commands no turn at all and a percent above `100`
commands a rate above the nominal one - the AI keeps the full rate
range, because the AI channel is never softened.

### racingLinePointOf

```ts
racingLinePointOf(
  samples: readonly SplineSample[],
  racingLine: readonly TrackRacingLinePoint[],
  index: number,
): RacingLinePointXY
```

Resolves the racing line's OWN point at a sample index.

THE POINT: the centerline sample displaced by the racing line's
published SIGNED LATERAL OFFSET along the chord's left normal - the
very point the AI aims at, so the pursuit expectation is measured
along the SAME line the AI follows.

Parameters:
- `samples` - Frozen track spline samples.
- `racingLine` - The published racing-line table of the layout.
- `index` - The sample index.

Returns: The racing line's own point at the index.

### RacingLinePointXY

A world point of the racing line (an offset-displaced sample).

### resolveAutopilotRacingLine

```ts
resolveAutopilotRacingLine(
  splineSamples: readonly SplineSample[],
): readonly TrackRacingLinePoint[]
```

Resolves the racing-line table of a track layout (AC-086).

THE TRACK-RELATIVE SURFACE: the table is the ONE published surface of
the signed lateral offset, the signed curvature profile and the
curvature-derived target speed, so the Phase 2 observation assembler
(AC-027) and the scripted controller (AC-028) read the SAME table the
autopilot drives on - there is NO second geometry and NO second source
of truth anywhere in the repo.

THE MEMO: the table is built once per layout and cached against the
frozen sample list, so the hot path never rebuilds it and never
allocates a TABLE (the per-frame step still builds its own small
step-input object and rescans the sample list).

Parameters:
- `splineSamples` - Frozen track spline samples (ordered closed loop).

Returns: The racing-line points, one per sample, in the sample order.

Example:

const line = resolveAutopilotRacingLine(trackSpec.splineSamples);
// line[0].targetSpeedUnitsPerS -> the curvature target speed (m/s)

### resolveControlChordRing

```ts
resolveControlChordRing(
  samples: readonly SplineSample[],
): ControlChordRing | null
```

Resolves the CONTROL chord ring the frozen samples describe.

THE RING (Step 14Y, AC-098(d)): the road the physics classifies and
contains against is the CONTROL POLYGON, and the samples name that
polygon's chords through their own metadata - so the AI rebuilds the
very chords the spec published instead of measuring the road a second
time.

THE FALLBACK: a sample list that names a SINGLE control segment
describes no control ring of its own (the hand-authored fixture rings
do), so the samples' OWN ring is the only ring it describes - and the
clamp is INACTIVE there (a sample sits ON the chord that starts at
it), so those layouts drive the very line they always drove.

Parameters:
- `samples` - Frozen track spline samples.

Returns: The control chord ring, or `null` when the samples name no
control ring of their own.

### resolveLookaheadWindow

```ts
resolveLookaheadWindow(
  samples: readonly SplineSample[],
  racingLine: readonly TrackRacingLinePoint[],
  nearestIndex: number,
  lookaheadIndex: number,
): LookaheadWindow
```

Resolves the lookahead WINDOW reading of the racing line.

THE WINDOW: the samples from the nearest one up to (and including) the
lookahead one, so the TIGHTEST curvature in the window ahead of the car
sets the target speed - the AI slows for a corner while it is still
straight, not once it is already in the corner.

THE TARGET IS A BRAKING INTENT, AND THE BRAKE IS SIZED FOR IT (Step
14Y, AC-098): the window is 12 m long, while the arcade brake rate
(`NEATRACING_PHYSICS_BRAKING_UNITS_PER_S2` = -89 m/s^2) needs
`v^2 / (2 x 89)` of distance to shed the excess - 31.1 m to drop from
89 m/s to the R = 60 m fixture's 48.75 m/s target, and about 18.5 m to
drop to the demo track's ~68 m/s target. Both are LONGER than the
12 m window, so a WEAK brake reaches the corner far above the
published target - and above the target the model's grip can no longer
hold, so the AI runs wide. The brake is therefore sized against the
distance the AI still has (the arc length to the constrained corner),
so it sheds the speed as fast as the model allows and arrives at the
published target. The window stays 12 m long (the Step 14M lanes pin
it): the DISTANCE to the constrained corner inside that window is what
the brake is sized against, and the surplus beyond it is made up on
the way in.

THE ONE SCAN (Step 14Y, AC-098): the window is scanned ONCE for all
three readings - the target speed, the tightest corner it spans and the
distance to that corner.

Parameters:
- `samples` - Frozen track spline samples.
- `racingLine` - The published racing-line table of the layout.
- `nearestIndex` - Index of the nearest sample (the window origin).
- `lookaheadIndex` - Index of the lookahead sample (the window end).

Returns: The window readings (the target speed, the tightest corner
and the distance to that corner).

### resolvePursuitExpectationRad

```ts
resolvePursuitExpectationRad(
  samples: readonly SplineSample[],
  racingLine: readonly TrackRacingLinePoint[],
  nearestIndex: number,
  aimX: number,
  aimY: number,
): number
```

Resolves the PURSUIT EXPECTATION of the racing line (rad).

THE EXPECTATION (Step 14Y, AC-098): the road the AI follows turns, so
the aim point it chases sits OFF the tangent the car drives on - that
offset is EXPECTED geometry and not a driving mistake. The expectation
is that EXPECTED angle, read from the racing line itself: the angle
between the line's own chord (from the line's point at the car to the
aim point) and the line's own tangent at the car.

WHY IT IS READ, NOT FORMULATED (Step 14Y, AC-098): the tangent-chord
theorem holds for a CONSTANT curvature, while the published line's
curvature VARIES along the window - so the expectation is measured on
the line the AI actually follows, and a car that tracks the line reads
EXACTLY zero error in every corner (the half-angle of the window's own
turn over-read the corner and drove the AI into it).

Parameters:
- `samples` - Frozen track spline samples.
- `racingLine` - The published racing-line table of the layout.
- `nearestIndex` - Index of the nearest sample (the car's sample).
- `aimX` - The world x of the aim point.
- `aimY` - The world y of the aim point.

Returns: The signed pursuit expectation (rad).

### resolveRoadBoundLine

```ts
resolveRoadBoundLine(
  splineSamples: readonly SplineSample[],
): readonly SplineSample[]
```

Resolves the ROAD-BOUND LINE of a track layout (the line the AI drives).

Parameters:
- `splineSamples` - Frozen track spline samples (ordered closed loop).

Returns: The road-bound line, one sample per published sample.

Example:

const line = resolveRoadBoundLine(trackSpec.splineSamples);
// line[i] is the published sample, held inside the published road core

### resolveSpeedPercent

```ts
resolveSpeedPercent(
  speed: number,
): number
```

Resolves the normalized speed percentile of a snapshot speed.

THE SAME PERCENTILE (Step 14T, AC-093): the AI reads the speed
percentile the physics service resolves for its own laws - the same
ratio, clamped the same way - so the AI's behaviour-preserving
re-expression resolves to exactly the turn rate the AI commanded
before the channel was adopted.

Parameters:
- `speed` - The car speed in units per second.

Returns: The normalized speed in `[0, 1]`.

### resolveTurnRatePercent

```ts
resolveTurnRatePercent(
  demand: number,
  speedPercent: number,
): number
```

Resolves the SIGNED turn-rate percent the autopilot commands.

THE DEMAND (Step 14Y, AC-098): the percent is the signed re-expression
of the AI's UNCLAMPED turn DEMAND - the pursuit chase plus the road's
own curvature feed-forward - so the physics service resolves
`rate = STEERING_RATE x (percent / 100)`.

THE UNCLAMPED DEMAND: the demand is NEVER clamped to the steer axis,
because that clamp is what made every corner tighter than the model's
44.5 m minimum turn radius unholdable (a clamped steer axis caps the
rate at `v / 44.5` whatever the road asks for). The demand is the SUM
of the two terms the steer law already reads, so wherever the steer
axis can express the demand the two numbers are the SAME and the
percent stays BIT-IDENTICAL to the pre-14Y percent - the demand only
exceeds the axis where the road asks for MORE than the axis can
express.

Parameters:
- `demand` - The unclamped turn demand (the pursuit chase plus the
curvature feed-forward of the road under the car).
- `speedPercent` - The normalized speed in `[0, 1]`.

Returns: The signed turn-rate percent.

### signedLateralOffsetM

```ts
signedLateralOffsetM(
  samples: readonly SplineSample[],
  nearestIndex: number,
  x: number,
  y: number,
): number
```

Measures the SIGNED lateral offset of a point from the road centerline.

THE MEASUREMENT: the offset is the signed perpendicular distance to the
centerline chord at the nearest spline sample, POSITIVE to the LEFT of
the travel direction (the repo's left-normal convention). It is
measured DIRECTLY from the samples - never inferred from any clamp or
seam (the containment is RETIRED: AC-090 VERDICT 8) - so an off-center
car always reads off-center.

THE ZERO: a car that sits exactly on the centerline chord reads EXACTLY
zero, which is what makes the centerline-correcting steer term zero on
the centerline.

Parameters:
- `samples` - Frozen track spline samples.
- `nearestIndex` - Index of the nearest sample (the chord origin).
- `x` - Car world x.
- `y` - Car world y.

Returns: The signed lateral offset (m); positive to the LEFT.

### stepAutopilotInput

```ts
stepAutopilotInput(
  car: AutopilotCarSnapshot,
  splineSamples: readonly SplineSample[],
): PhysicsStepInput
```

Computes one analog physics step input that chases the RACING LINE:
steer proportional to the wrapped heading error toward the racing-line
steer target (displaced by the signed lateral offset) MINUS the
centerline-correcting term, and throttle or brake from the
curvature-derived target speed and the named error thresholds.

THE TURN-RATE CHANNEL (Step 14T, AC-093): the input ALWAYS carries a
DEFINED, signed turn-rate percent - the behaviour-preserving
re-expression of the AI's own turn (`100 x steerAI x speedPercent`) -
so the physics drives the AI through the AI channel and the AI is
exempt from the user steer-axis softening factor.

THE RACING LINE (Step 14M, AC-083..AC-086): the seam reads the ONE
published racing-line table of the layout (memoized, so the hot path
allocates NO TABLE - it still builds its own small step-input object
and rescans the sample list) - the steer target is displaced by the
table's SIGNED LATERAL OFFSET, the steer command gains the CURVATURE
FEED-FORWARD of the road ahead, the throttle and the brake follow its
CURVATURE-DERIVED TARGET SPEED, and the car's own signed lateral
offset adds the centerline-correcting steer term (EXACTLY zero on the
centerline).

THE COMMAND SUFFICIENCY (Step 14Y, AC-098): the STEER AXIS keeps the
RETAINED clamped law (the published steer rail the drift and the
replay read), while the TURN DEMAND the percent channel carries is the
UNCLAMPED sum of the same two terms with the road-under-the-car
curvature - so the commanded rate is `v x curvature` (the kinematic
requirement of the corner the car is ON) plus the pursuit correction,
and a corner tighter than the model's 44.5 m minimum turn radius is
holdable at the speed the AI drives it at.

THE EXCESS ERROR (Step 14Y, AC-098): the throttle cut and the error
brake read the EXCESS heading error - the heading error minus the
pursuit expectation, the racing line's OWN chord angle at the car - so
the AI never coasts and brakes for the corner it is following.

THE NEVER-STOP FLOOR (Step 14Y, AC-098(c)): at or below the physics'
own off-road cruise floor the AI never brakes, and on a road that turns
it never coasts either - so the AI can never stop in a corner.

Deterministic pure seam — no RNG, no clock reads, no mutation.

Parameters:
- `car` - Kinematic car snapshot (x, y, heading, speed).
- `splineSamples` - Frozen track spline samples (ordered closed loop).

Returns: One step input for the 60 Hz fixed step.

Example:

const input = stepAutopilotInput(car, trackSpec.splineSamples);
// { throttle: 1, brake: 0, steer: 0.21, turnRatePercent: 12.3 }
// chasing the racing line of the loop

### wrapSignedAngle

```ts
wrapSignedAngle(
  radians: number,
): number
```

Wraps an angle into `[-π, π)` via the atan2 half-plane projection so
steering errors stay continuous across the ±π discontinuity.

Parameters:
- `radians` - Raw angle difference in radians.

Returns: Angle wrapped into `[-π, π)`.

## physics/physics.step.service.ts

Deterministic fixed-step physics step service for the neatRacing example
(plan slice P1-S7-B; AC-023; spec Â§5.1â5.3, LD-15, TP-01/TP-02/TP-04).

The simulation truth is a top-down 2D world: the car state carries world
`x`/`y` (meters; 1 world unit = 1 meter), `heading` (radians), a
longitudinal `speed` (meters/second; TP-01: ~89 u/s â‰ˆ 320 km/h top-speed
feel), and the LD-15 drift channel â `yawRate` (heading change in radians
PER 60 Hz tick, `rad/s = yawRate * 60`) and `slipAngle` (bounded
`atan2`-style angle between the velocity direction and the heading) â
exactly the fields Phase-2 sensors/observations consume unchanged.

The DRIFT CHANNEL is nested on the car state (AC-087):
`drift.committed` (the latched physical slide) and `drift.pose` (the
visible yaw accumulator) - the drift model in `./drift.model` owns every
drift law and this service only sequences the tick.

The TRACTION SEVERITY (`tractionSeverity`, AC-088) is the ONE shared
scalar published on the car state: it is derived from the POST-TRACTION
speed and the PUBLISHED slip, so the camera follow, the car pass, the
HUD and the debug seam all read the SAME number and no consumer
re-derives it.

The DEMANDED TURN RATE (Step 14S, AC-091) is resolved ONCE per tick and
is the ONE turn quantity the tick produces: the per-tick heading delta
is DERIVED from it (the rate times the fixed `dt`) and the SAME number
is fed to the drift commitment, so the yaw and the momentum law can
never disagree about the turn the driver demanded.

THE TURN-RATE LAW (Step 14T, AC-092/AC-093) resolves that one number on
TWO separate channels that never collide: the USER channel (the steer
axis times the steering rate times the speed percentile, softened by a
MILD speed-proportional reduction that never reaches zero) and the AI
channel (the OPTIONAL turn-rate percent, which is AUTHORITATIVE - it is
resolved BEFORE the user softening and never sees it). A PRESENT
percent wins; an ABSENT percent keeps the steer-axis law.

THE TURN SUFFICIENCY (Step 14Y, AC-098(a)) lifts the USER channel
inside a speed-localized BAND (the crawl band up to the band's upper
edge), so the channel can HOLD the tightest measured corner instead of
falling short of its kinematic requirement: the sufficiency peaks at
the corner's own target speed and it is exactly `1` outside the band,
so the crawl rate stays BIT-IDENTICAL and the ratified mild reduction
keeps its own full meaning above the band. The sufficiency belongs to
the USER channel alone - the AI channel is untouched by it.

The tuning baseline is the locked javascript-racer adaptation table
(spec Â§5.2), transcribed as named constants with their js-racer
derivations preserved: `ACCEL = maxSpeed / 5`,
`BRAKING = -maxSpeed`, `NATURAL_DECEL = -maxSpeed / 5`,
`OFF_ROAD_DECEL = -maxSpeed / 2` with the USER-named off-track speed
cap at `20 km/h` (AC-090 VERDICT 8 - the off-track limit the USER
named, `20 / 3.6` m/s, so a coasting off-road car never decelerates
to zero and never exceeds the USER-named limit),
the `STEERING_RATE = 2` steering rate scaled by the speed percentile
(TP-04: `dx = dt Â· 2 Â· speedPercent`, a standing car does not rotate).

Determinism (spec Â§5.3): the service is PURE â it calls no `Math.random`,
reads the surface ONCE from the POST-MOVE pose through the resolved
resolver, and
integrates in one fixed documented order, so the same
`(seed, control stream, surface stream)` produces byte-identical
JSON-serialized states on the same runtime (reproducibility rung
Level 2, ordered deterministic; cross-environment byte equality is NOT
claimed).

THE ONE SURFACE ORDER (Step 14W, AC-094 (c)): the surface the physics
integrates against is the surface the car is ON, never the surface it
was on at the tick start. The tick resolves the surface ONCE, from the
POST-MOVE pose (the pose the car reached in this tick), so a car that
has left the road really gets the band surface in the SAME tick and the
EXISTING named off-road law (the off-road deceleration constant and the
off-road speed-floor constant) actually applies. The reach pose itself
depends on the surface, so the reach is computed against the NEUTRAL
(road) surface FIRST and the tick is RE-INTEGRATED against the resolved
surface when that surface is not the neutral one - so the published tick
is always the tick that integrated against the surface the car is on.

Scope notes: off-track/ribbon detection is wired by the surface-resolver
consumer (P1-S7-C); per-wheel slip channels and the
worker-level reproducibility tuple are Phase-2 concerns and deliberately
out of scope here. `curb` shares the road longitudinal model in this
slice; curb-specific rumble shaping arrives with the collision/
off-track surface utils (P1-S7-C).

The CAR-TO-CAR COLLISION is wired HERE (AC-090): the named seam
`resolveLiveCarCollisions` is the ONE owner of the resolver call
(`resolveCarCollisions` in `./collision.utils`), and the pack tick
`stepPhysicsFleet` calls it AFTER the drift integration and BEFORE the
states are published. The single-car tick (`stepPhysics`) stays the hot
path of the live session and never resolves a fleet.

The TRACK BOUND IS GONE (AC-090 VERDICT 8, superseding the AC-081
containment): there is NO invisible wall near the track edge. A car
that leaves the ribbon really goes off track and keeps rolling - the
ONLY limit out there is the USER-named off-track SPEED cap (the named
`20 km/h` constant below), never a geometric bound, so the single-car
tick needs NO geometry seam at all.

### advanceCarState

```ts
advanceCarState(
  state: PhysicsCarState,
  clampedInput: PhysicsStepInput,
  surface: PhysicsSurfaceType,
): PhysicsCarState
```

Advances ONE car by ONE fixed tick against ONE surface.

THE PIPELINE (the documented order of the tick): the demanded turn rate
is resolved ONCE and drives both the yaw and the drift commitment, the
longitudinal speed is integrated through the locked speed-table rates
(with the surface's off-road law), the drift channel advances in its
load-bearing order, the published slip angle and the ONE shared
traction severity are derived from the POST-scrub speed, and the car is
translated along the velocity direction.

THE SURFACE IS THE ONLY INPUT THAT CHANGES THE NUMBERS: the off-road
deceleration and the off-road speed floor are the two surface-dependent
reads, so the same call with the same surface always produces the same
numbers. That is what lets the tick resolve the surface from the
POST-MOVE pose and re-integrate against it without changing a single
pinned tick.

Parameters:
- `state` - The tick-start car state (treated read-only; never
mutated).
- `clampedInput` - The clamped control sample applied to this tick.
- `surface` - The surface this tick integrates against.

Returns: The advanced car state (the published pose - the AC-081
containment seam is retired, so the reached pose IS the published
pose; AC-090 VERDICT 8).

### applyTrackContainment

```ts
applyTrackContainment(
  car: PhysicsCarState,
  trackSegments: readonly TrackSegment[] | undefined,
): PhysicsCarState
```

The RETIRED TRACK-CONTAINMENT seam (AC-090 VERDICT 8, superseding the
landed AC-081-WIRE seam of the Step 14I/14W iterations): the ONE named
owner of the ONE track bound - which is now NO BOUND AT ALL.

THE RETIREMENT (the USER observable, the 2026-11-05 verdict-8 record):
the landed iteration-2 non-reflecting clamp HELD the pose at the band
outer edge (288/288 post-touch ticks pinned EXACTLY 3.0 m past the
drivable edge while full throttle was applied - the plan's measured W1
record), so the boundary read as an INVISIBLE WALL. The USER named the
next-order defect verbatim: "the car should be able to go off track,
but with the speed limited to 20km/h". The measured classification
recorded the hold as a true pose-hold (NOT a slowdown), the corridor
beyond the bound as geometrically CLEAN (7.0 m to the AC-096 10 m
prop floor), and the published off-track floor as 4.005x the
USER-named limit - so the bound is RETIRED ENTIRELY and the ONLY
limit out there is the USER-named off-track speed cap (the named
`NEATRACING_PHYSICS_OFF_TRACK_SPEED_LIMIT_KMH` constant in this
module; the off-road decel law STAYS the slowdown mechanism).

THE NO-OP CONTRACT (AC-081-WIRE-NOOP, re-contracted by the
retirement): EVERY input is a no-op - an ABSENT geometry, an EMPTY
segment list and a FULL ribbon all leave the tick byte-identical,
and the returned car IS the reached car. The ONE-BOUND and
SHIPPED-TICK clauses carry over the retirement: this seam stays the
ONE named owner of the (now retired) bound contract, the second
parameter list stays the SHIPPED 4-ARG live-tick form (the served
closure and the contract lanes keep ONE tick form), and the AC-081
pose-independence clause is re-authored to hold TRUE BY
CONSTRUCTION: with no bound at all, no published value depends on
any pose.

Parameters:
- `car` - The advanced car state (treated read-only; never
mutated).
- `trackSegments` - The RETIRED ribbon geometry; accepted for the
shipped 4-arg live-tick form and never read (`void` marks the
intentional non-read of the retirement).

Returns: The SAME car state, unchanged.

### clampControlRatio

```ts
clampControlRatio(
  value: number,
): number
```

Clamps a `[0, 1]` control axis ratio (throttle, brake, speed percentile).

### clampSteeringRatio

```ts
clampSteeringRatio(
  value: number,
): number
```

Clamps the steering ratio into `[-1, 1]`.

### clampStepInput

```ts
clampStepInput(
  input: PhysicsStepInput,
): PhysicsStepInput
```

Clamps the three control axes into their contract ranges.

### createInitialPhysicsCarState

```ts
createInitialPhysicsCarState(
  seed: number | undefined,
  pose: PhysicsCarSpawnPose | undefined,
): PhysicsCarState
```

Creates the deterministic spawn car state for a given seed (spec Â§5.3
determinism tuple: the seed selects the spawn lane via a pure integer
mix Ã¢ no `Math.random` anywhere, so the same seed always yields the
byte-identical spawn state). The car starts standing (zero speed, zero
yaw, zero slip, a zeroed drift channel) heading straight along +x at the
world origin line.

Parameters:
- `seed` - Replay seed that fixes the spawn lane offset; defaults to
`NEATRACING_PHYSICS_DEFAULT_SPAWN_SEED` when omitted. Non-integer or
out-of-range values are truncated deterministically (`| 0` treats
`NaN` as `0`) Ã¢ never a silent nondeterminism source.
- `pose` - Optional explicit spawn pose; when supplied, the seed is
ignored because the explicit position already fixes the lane.

Returns: Fresh `PhysicsCarState` at `tick = 0`.

Example:

```ts
const car = createInitialPhysicsCarState(42);
console.log(car.speed === 0); // standing start
```

### integrateLongitudinalSpeed

```ts
integrateLongitudinalSpeed(
  speed: number,
  input: PhysicsStepInput,
  surface: PhysicsSurfaceType,
): number
```

Integrates the longitudinal speed for one tick per the Â§5.2 locked
table: braking wins and is never floor-protected (deliberate braking
may stop the car); passive ticks stack the off-road drag on grass and
floor the result at the off-road floor only while the car enters the
tick at/above it (js-racer limit-gate parity, so the floor actually
holds). Speeds are clamped to `[0, maxSpeed]`.

The floor gate is SHARED with the post-scrub floor of the tick
pipeline (RC-1) through `isOffRoadSpeedFloorEngaged`, so the
integrator and the drift scrub can never disagree about the floor.

### isOffRoadSpeedFloorEngaged

```ts
isOffRoadSpeedFloorEngaged(
  surface: PhysicsSurfaceType,
  input: PhysicsStepInput,
  tickStartSpeed: number,
): boolean
```

Resolves whether the OFF-ROAD SPEED FLOOR holds for the current tick.

THE GATE (js-racer limit-gate parity): the floor holds only while the
car ENTERS the tick at/above the floor and is NOT braking - a
deliberate brake is never floored and may stop the car.

ONE GATE, TWO READS (RC-1): the SAME gate guards the pre-scrub floor
(inside the longitudinal integrator) and the post-scrub floor (the
drift-scrub read in the tick pipeline), so the two can never disagree
about whether the floor applies to the tick. Before RC-1 the pre-scrub
floor was the only read, so the drift scrub pushed an off-road car
below the floor and the tick-start gate stopped holding it.

Parameters:
- `surface` - The resolved surface the tick integrates against.
- `input` - The clamped control sample applied to this tick.
- `tickStartSpeed` - The tick-start longitudinal speed in m/s.

Returns: True when the off-road speed floor applies to this tick.

### mixSeed

```ts
mixSeed(
  seed: number,
): number
```

Mixes a spawn seed through a deterministic `imul`-based avalanche
(pure integer arithmetic; identical inputs always produce identical
mixes on any runtime honoring the ES specification for `Math.imul`).

### NEATRACING_PHYSICS_ACCEL_UNITS_PER_S2

Forward acceleration rate: `maxSpeed / 5` (spec Â§5.2 locked table,
javascript-racer `ACCEL`), i.e. 0-to-top in ~5 s of full throttle.

### NEATRACING_PHYSICS_BRAKING_UNITS_PER_S2

Braking deceleration rate: `-maxSpeed` (spec Â§5.2 locked table,
javascript-racer `BRAKING`), i.e. a full brake pin stops the car in ~1 s.

### NEATRACING_PHYSICS_DEFAULT_SPAWN_SEED

Spawn seed used when a caller does not provide one (deterministic).

### NEATRACING_PHYSICS_FIXED_TIMESTEP_MS

Fixed simulation timestep in milliseconds (spec Â§5.1: deterministic
60 Hz fixed step inside the simulation worker; host pacing uses the
accumulator loop, not this service).

### NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS

Fixed simulation timestep in seconds, the `dt` of every integration
step in this service (js-racer parity: all rates are per-second values
scaled by `dt`).

### NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S

Top speed in meters/second â the locked tuning anchor (TP-01): ~320
km/h arcade feel with 1 world unit = 1 meter. Every derived tuning
constant below is expressed from this anchor, exactly as javascript-
racer derives `ACCEL`/`BRAKING`/`NATURAL_DECEL`/`OFF_ROAD_DECEL` from
its `maxSpeed`.

### NEATRACING_PHYSICS_NATURAL_DECEL_UNITS_PER_S2

Natural deceleration (coast drag): `-maxSpeed / 5` (spec Â§5.2 locked
table, javascript-racer `DECELERATION`), applied on ticks with neither
throttle nor brake.

### NEATRACING_PHYSICS_OFF_ROAD_DECEL_UNITS_PER_S2

Off-road deceleration drag: `-maxSpeed / 2` (spec Â§5.2 locked table,
javascript-racer `OFF_ROAD_DECELERATION`), stacked on top of the passive
rate on off-road ticks. `javascript-racer`'s value is transcribed with
the same `maxSpeed` derivation (G10 forbids only the raw segment-units
`5`, which this constant does not transcribe).

### NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S

Off-track speed cap in m/s: the USER-named 20 km/h (`20 / 3.6`).

While the car enters an off-road tick at/above this speed and is NOT
braking, the passive decelerations are floored here, so an off-road
car coasts down to the USER-named limit and HOLDS it instead of
decelerating to zero - and never runs faster than the USER named.

THE DECEL LAW STAYS THE SAND MECHANISM: the off-road deceleration
constant above is what actually slows the car down; this cap is the
CEILING the USER named, and it SUPERSEDES the retired `maxSpeed / 4`
floor (22.25 m/s = 80.1 km/h, ~4x the USER-named limit).

### NEATRACING_PHYSICS_OFF_TRACK_SPEED_LIMIT_KMH

The USER-named OFF-TRACK SPEED LIMIT, in km/h (AC-090 VERDICT 8).

THE USER'S OWN NUMBER: "the car should be able to go off track, but
with the speed limited to 20km/h". The off-track limit is therefore
20 km/h - a NAMED constant, never a magic number, and the ONE place
the off-track limit is authored.

### NEATRACING_PHYSICS_SPAWN_HEADING_RADIANS

Heading at spawn in radians: straight along +x (top-down race start).

### NEATRACING_PHYSICS_SPAWN_LATERAL_SPREAD_M

Lateral spawn spread in meters: the seeded spawn lane offset is drawn
deterministically from the spawn seed within `Â±spread / 2` so the same
seed always spawns on the same lane line.

### NEATRACING_PHYSICS_STEERING_RATE

Steering rate: `dx = dt Â· 2 Â· speedPercent` (spec Â§5.2 locked table,
TP-04 "cross the road in ~1 s at top speed"). Top-down adaptation: the
heading change per tick is `input.steer Â· STEERING_RATE Â·
speedPercent Â· dt` (radians per tick), so steering authority grows with
speed and a standing car does not rotate.

### NEATRACING_PHYSICS_TURN_RATE_REDUCTION_BAND_UNITS_PER_S

The low-speed handling band (m/s) BELOW which the USER turn rate is
NOT reduced (Step 14T, AC-092).

MEASURED (the RED decision, Step 14T): the slowest REAL drifting lane
of the build is the 10 m/s crawl lane (the momentum crossover sits at
about 9.43 m/s), so the reduction starts ABOVE it: the low-speed
handling keeps the nominal law BIT-IDENTICALLY and the reduction bites
on the high-speed lanes, where the handling actually needs it. The band
is the crawl band of the RED-measured window, so the crawl turn rate
stays at the existing nominal law.

### NEATRACING_PHYSICS_TURN_RATE_SPEED_FACTOR_FLOOR

The turn-rate speed-factor FLOOR (Step 14T, AC-092).

The floor is the RED-measured MILD bound (a quarter of the nominal rate
may never be taken away) and it is the hard guarantee that the speed
factor NEVER reaches zero: the user turn rate stays a REAL rate at
every speed, so the reduction can never silently switch the steering
off. The floor binds the FEEL and never the physics (at the floor the
top-speed full-steer lane still commits a 0.443 rad slide, 98.5 % of
the 0.45 rad depth budget).

### NEATRACING_PHYSICS_TURN_RATE_SPEED_REDUCTION_FRACTION

The USER turn-rate reduction at the TOP speed (Step 14T, AC-092).

MEASURED (the RED decision, Step 14T): the mild reduction takes a
FIFTH of the nominal turn rate away at the top speed, so the top-speed
USER turn rate reads `0.8 * 2 = 1.6` rad/s - above the RED-measured
mild floor (`0.75 * 2 = 1.5` rad/s) and far away from zero, so
turning AND drifting stay possible at every speed.

### NEATRACING_PHYSICS_TURN_RATE_SUFFICIENCY_BAND_END_UNITS_PER_S

The upper edge of the USER turn-rate sufficiency band (m/s) (Step 14Y,
AC-098(a)).

WHY THE SUFFICIENCY IS A BAND AND NOT A GLOBAL RAISE: the ratified
AC-092 reduction factor is measured against the NOMINAL law of the
SAME speed, so the rate may never grow above the crawl band's rate
(the factor may only FALL with the speed) - a global raise of the
nominal rate would invert that factor at the crawl band. The
sufficiency is therefore a BAND: it is raised inside the band and it
is back at zero at this edge, so the ratified factor stays monotonic
across the ratified speed ladder and the top-speed rate stays
strictly below the nominal law (the AC-092 rails).

THE VALUE: 18.0 m/s - the first speed above the RED-measured corner
speed where the ratified law's own rate is read again.

### NEATRACING_PHYSICS_TURN_RATE_SUFFICIENCY_CORNER_RADIUS_M

The tightest holdable corner radius of the USER channel (m) (Step 14Y,
AC-098(a)).

THE RED-MEASURED VALUE: the tightest MEASURED layout of the repo (the
R = 8 m stadium) is the corner the USER channel has to be able to
HOLD. A corner of curvature `kappa` demands the turn rate `v x kappa`
(the kinematic requirement) at the speed the racing line drives it at,
so the radius is the model-side half of the sufficiency: it turns the
corner into a turn RATE the commanded rate can be compared with.

### NEATRACING_PHYSICS_TURN_RATE_SUFFICIENCY_CORNER_SPEED_UNITS_PER_S

The target speed of that tightest corner (m/s) (Step 14Y, AC-098(a)).

THE RED-MEASURED VALUE: the published target speed of the tightest
corner of the tightest MEASURED layout of the repo (measured through
the racing line's own curvature-derived target speed). The
sufficiency PEAKS at this speed - it is the speed the corner is
driven at, so it is where the corner's kinematic requirement is
largest and where the USER channel is measured.

### NEATRACING_PHYSICS_TURN_RATE_SUFFICIENCY_MARGIN

The command-sufficiency margin of the USER channel (1) (Step 14Y,
AC-098(a)).

THE MARGIN (the same rationale as the AI's own command-sufficiency
margin): a commanded rate that MERELY meets the corner's kinematic
requirement leaves the corner on the boundary, where any float noise
loses it. The margin is the small slack that keeps the commanded rate
STRICTLY above the requirement, so the corner is HOLDABLE rather than
borderline.

### NEATRACING_PHYSICS_TURN_RATE_SUFFICIENCY_TRANSITION_HALF_WIDTH_UNITS_PER_S

The transition half-width of the USER turn-rate sufficiency window
(m/s) (Step 14Y, AC-098(a)).

THE SHAPE: the sufficiency is a raised-cosine window that peaks at
the RED-measured corner speed. This constant caps how far from the
peak the window is allowed to open on EITHER side. It keeps the R=8
hold fully boosted at 17.8 m/s while making the boost fade long before
the demo drive reaches the ~18.4 m/s point where the wide original
window pushed the car off the served track.

### PhysicsCarSpawnPose

Optional spawn-pose override for callers that already know the track-side
position and heading. When supplied, the pose fields replace the defaults;
the seed is ignored because the explicit position already fixes the lane.

### PhysicsCarState

Full deterministic car state for one 60 Hz fixed step. JSON-serializable
on purpose: the batch-83 replay contract compares byte-identical
`JSON.stringify` snapshots of this shape across repeated runs.

### PhysicsStepInput

Three-axis keyboard control sample applied for one fixed tick (D5: the
input is sampled once per frame by the host and replayed per tick, so
the same `PhysicsStepInput` object is passed to every catch-up step).

Throttle and brake are clamped into `[0, 1]`; brake takes precedence
over throttle. Steer is clamped into `[-1, 1]` (positive steers
counterclockwise). Axes are analog-capable so later input transports
can pass analog values without a contract change.

The OPTIONAL turn-rate percent (Step 14T, AC-093) is the AI channel: it
rides ALONGSIDE the steer axis (it never replaces it) and it is NOT
clamped, because the AI owns the full rate range (a percent may exceed
100 and it may be negative - the direction travels with the percent).

### PhysicsSurfaceType

Surface label for one tick, matching the track slab palette (`track.slabs`).

### republishSeparatedCar

```ts
republishSeparatedCar(
  car: PhysicsCarState,
): PhysicsCarState
```

Republishes a resolved car with the published scalars re-derived from
its POST-RESOLUTION pose.

The car-to-car collision (AC-090) recomposes the car around a new
world velocity (a new heading, a new speed, a zeroed slip and a zeroed
drift channel), so the published travel heading and the published
traction severity of the pre-resolution tick are STALE. They are
re-derived here - from the resolved values ONLY, never from the
pre-resolution ones - so the published state and the rendered frame
stay one frame of reference. (The road containment that shared this
republish path was RETIRED with the AC-081 bound - AC-090 VERDICT 8 -
so this republish now serves the car-to-car resolution ONLY.)

Parameters:
- `car` - The resolved car state (treated read-only).

Returns: A fresh state whose published scalars match the resolved pose.

### resolveDemandedTurnRateRadPerS

```ts
resolveDemandedTurnRateRadPerS(
  steer: number,
  speedPercent: number,
  turnRatePercent: number | undefined,
): number
```

Resolves the DEMANDED turn rate of a tick (radians per SECOND).

THE TWO CHANNELS (Step 14T, AC-092/AC-093): the rate is resolved on
TWO separate laws that never collide.

THE AI CHANNEL (a PRESENT turn-rate percent, AC-093): the rate is
resolved from the percent of the NOMINAL turn rate - INDEPENDENT of
the steer axis and NEVER softened by the user adjustment. The percent
is SIGNED, so the direction travels with the percent and never with the
steer axis. A percent of `100` commands the nominal rate, a percent of
`0` commands no turn at all and a percent above `100` commands a rate
above the nominal one (the AI keeps the full range).

THE USER CHANNEL (an ABSENT turn-rate percent, AC-092): the steer-axis
law (the steer axis times the steering rate times the speed percentile,
the TP-04 form) softened by the MILD speed-proportional factor. The
factor is applied LAST, in the SAME product order as the pre-14T law,
so every lane below the reduction band stays BIT-IDENTICAL.

THE TURN SUFFICIENCY (Step 14Y, AC-098(a)): inside the sufficiency band
the user rate is raised by a speed-localized sufficiency factor, so the
channel can HOLD the tightest measured corner instead of falling short
of its kinematic requirement. The factor is exactly `1` outside the
band - below the crawl band (the rate stays BIT-IDENTICAL) and above
the band's upper edge (the ratified mild reduction keeps its own full
meaning) - so the sufficiency never touches a ratified lane's reading.
The sufficiency belongs to the USER channel alone: the AI channel
above is untouched by it.

THE RATE is the one turn quantity a tick resolves; the per-tick heading
delta is DERIVED from it and the drift commitment CONSUMES it, so the
yaw and the momentum law read the SAME number (Step 14S, AC-091).

Parameters:
- `steer` - The clamped steering axis in `[-1, 1]`.
- `speedPercent` - The normalized speed in `[0, 1]`.
- `turnRatePercent` - The optional AI turn-rate percent, in percent.

Returns: The demanded turn rate (rad/s).

### resolveLiveCarCollisions

```ts
resolveLiveCarCollisions(
  fleet: readonly PhysicsCarState[],
): PhysicsCarState[]
```

The LIVE car-to-car collision seam (AC-090): the ONE named owner of the
car-to-car resolution inside the live physics tick.

The seam is the ONLY live call site of the resolver (`resolveCarCollisions`
in `./collision.utils`), so the resolution has a single owner and a single
call site: the pack tick calls it AFTER every member finished the drift
integration (so the whole fleet sits at ONE time base) and BEFORE the
states are published (so the published states and the rendered frame can
never disagree).

The resolver separates an overlapping pair and exchanges the world
velocities along the collision normal (LD-05, the arcade elastic
exchange). The seam adds the PUBLISH rule the resolver cannot know about:
a separated member is republished with its published scalars re-derived
from its RESOLVED pose, because `travelHeadingRad` and `tractionSeverity`
are documented as the exact values the integration used - a resolved car
must never publish the pre-resolution numbers.

Parameters:
- `fleet` - The advanced fleet states (treated read-only; never
mutated).

Returns: Fresh resolved states in the same order; a member that did NOT
overlap keeps every published value.

### resolvePassiveRate

```ts
resolvePassiveRate(
  throttle: number,
): number
```

Resolves the passive (non-braking) longitudinal rate for one tick.

### resolveSpawnLateralOffset

```ts
resolveSpawnLateralOffset(
  seed: number,
): number
```

Resolves the seeded spawn lane offset within `Â±spread / 2`.

### resolveSpeedPercent

```ts
resolveSpeedPercent(
  speed: number,
): number
```

Normalizes the longitudinal speed into the `[0, 1]` speed percentile.

### resolveSurfaceAtPostMove

```ts
resolveSurfaceAtPostMove(
  surfaceResolver: ((carState: PhysicsCarState) => PhysicsSurfaceType) | undefined,
  reached: PhysicsCarState,
): PhysicsSurfaceType
```

Resolves the surface of the POST-MOVE pose (Step 14W, AC-094 (c)).

THE ONE READ: the resolver is asked ONCE per tick, about the pose the
car REACHED in this tick (never the tick-start state), so the surface
the physics integrates against is the surface the car is ON. An absent
resolver keeps the tick on the road.

Parameters:
- `surfaceResolver` - The optional surface probe.
- `reached` - The pose the car reached in this tick.

Returns: The surface under the reached pose; road without a resolver.

### resolveTurnRateSpeedFactor

```ts
resolveTurnRateSpeedFactor(
  speedPercent: number,
): number
```

Resolves the USER turn-rate speed factor of a speed (a unitless factor).

THE NAMED SPEED-PROPORTIONAL FACTOR (Step 14T, AC-092): the factor is
`1` at a standstill and in the whole low-speed handling band (the
reduction starts ABOVE the band), and it falls LINEARLY with the speed
excess above the band until it reaches the measured mild value at the
top speed (`1 - 0.2 = 0.8` of the nominal rate). It is MONOTONIC in
the speed (the factor never grows with the speed) and it is floored
(the factor NEVER reaches zero, so the user turn rate stays a real rate
at every speed - the reduction can never switch the steering off).

THE BASIS (the RED-measured window, Step 14T): the factor is resolved
from the SAME speed percent the existing steer-axis law uses (the
normalized speed in `[0, 1]`), so the two laws can never disagree about
the speed of the tick. The MEASURED ladder reads `1` at the 10 m/s
crawl lane, `0.9620253164556962` at 25 m/s, `0.9113924050632911` at
45 m/s, `0.8556962025316456` at 67 m/s and `0.8` at the 89 m/s top
speed (the top-speed user turn rate is `1.6` rad/s, above the
RED-measured mild floor of `1.5` rad/s).

Parameters:
- `speedPercent` - The normalized speed in `[0, 1]`.

Returns: The speed factor in `(0, 1]`; `1` below the reduction band.

### resolveTurnRateSufficiencyFactor

```ts
resolveTurnRateSufficiencyFactor(
  speedPercent: number,
): number
```

Resolves the turn-sufficiency FACTOR of a speed (a unitless factor
`>= 1`).

THE FACTOR (Step 14Y, AC-098(a)): the factor is `1` outside the
sufficiency band, so the ratified rate is left BIT-IDENTICAL there, and
inside the band it scales the ratified rate by the window. At the
RED-MEASURED corner speed the factor lifts the commanded rate to the
corner's kinematic requirement (`speed / radius`) plus the sufficiency
margin, so the USER channel can HOLD that corner instead of falling
short of it.

THE BASIS: the peak is resolved from the ratified law of the corner
speed itself (the steer-axis law times the ratified mild factor), so
the sufficiency is measured against the very law it lifts and it never
needs its own copy of the reduction factor.

Parameters:
- `speedPercent` - The normalized speed in `[0, 1]`.

Returns: The sufficiency factor in `[1, ...]`; exactly `1` outside the band.

### resolveTurnRateSufficiencyWindow

```ts
resolveTurnRateSufficiencyWindow(
  speedPercent: number,
): number
```

Resolves the turn-sufficiency WINDOW of a speed (a unitless factor in
`[0, 1]`).

THE WINDOW (Step 14Y, AC-098(a)): the window is a raised cosine that is
EXACTLY zero at both edges of the sufficiency band (the crawl band
below, the band's upper edge above) and peaks at the RED-MEASURED
corner speed. The transition half-width caps how far from the peak the
cosine is allowed to open on each side, so the boost is concentrated on
the tight corner and fades before the higher-speed straightaway. The
window is continuous and smooth at the peak and at both edges, so the
commanded rate never steps.

Parameters:
- `speedPercent` - The normalized speed in `[0, 1]`.

Returns: The window in `[0, 1]`; `0` outside the sufficiency band.

### resolveYawRatePerTick

```ts
resolveYawRatePerTick(
  demandedTurnRateRadPerS: number,
): number
```

Resolves the per-tick heading delta (radians per tick) FROM the resolved
demanded turn rate: the rate times the fixed `dt`.

THE DERIVATION (Step 14S, AC-091): the per-tick delta is NEVER
re-derived from the steer axis - it is the single resolved demanded turn
rate scaled by the fixed timestep, in the SAME product order as the
pre-14S law, so the published heading values stay BIT-IDENTICAL while
the drift commitment reads the very same number.

Parameters:
- `demandedTurnRateRadPerS` - The demanded turn rate of this tick (rad/s).

Returns: The per-tick heading delta (rad per tick).

### stepPhysics

```ts
stepPhysics(
  state: PhysicsCarState,
  input: PhysicsStepInput,
  surfaceResolver: ((carState: PhysicsCarState) => PhysicsSurfaceType) | undefined,
  trackSegments: readonly TrackSegment[] | undefined,
): PhysicsCarState
```

Advances one car state by exactly one deterministic 60 Hz fixed tick
under the given control sample and surface.

The tick pipeline is fixed and side-effect free (spec Â§5.1âÂ§5.3):

1. resolve the surface ONCE from the POST-MOVE pose (Step 14W,
   AC-094 (c)): the tick integrates against the surface the car is ON,
   never the surface it was on at the tick start. The pose the car
   reaches is computed against the NEUTRAL (road) surface, the resolver
   is asked about THAT pose (omitted means road; the resolver sees the
   pose the car reached, never the tick-start state), and the tick is
   RE-INTEGRATED against the resolved surface when that surface is not
   the neutral one;
2. resolve the DEMANDED turn rate ONCE from the steer axis and the
   tick-start speed percentile (TP-04, standing cars do not rotate),
   DERIVE the per-tick yaw delta from that one number and integrate
   the heading - the SAME number is handed to the drift commitment in
   step 4a, so the yaw and the momentum law read ONE value;
3. integrate the longitudinal speed through the locked Â§5.2 rates â
   braking wins over throttle, passive coast drag otherwise, off-road
   drag stacked on grass with the passive off-road floor (deliberate
   braking is never floored and can stop the car), clamped to
   `[0, maxSpeed]`;

   the SAME off-road floor is applied again to the POST-SCRUB speed
   in step 4c (RC-1), so a drifting off-road car never decays below
   the floor and never traps at 0 m/s;
4. advance the DRIFT CHANNEL (AC-087) in a LOAD-BEARING order: 4a the
   commitment (the latched physical slide, with the MIN_SPEED gate and
   the RE-ARM token inside the drift model, fed the DEMANDED turn rate
   resolved in step 2), 4b the committed chase (the
   lateral velocity chases the committed slide), 4c the speed scrub
   (AC-079: a sliding tire cannot hold the forward speed, so a
   full-speed turn is scrubbed down instead of being taken at the whole
   top speed), 4d the published slip angle (the bounded LD-15 `atan2`),
   and 4e the pose spring (the visible yaw accumulator, stepped INSIDE
   the 60 Hz tick and never in the render path; its target is the
   PUBLISHED slip, the ONE shared scalar);
5. translate along the velocity direction (heading + slipAngle) so the
   drift arc bends the path, and PUBLISH that velocity heading as the
   state's travel direction (`travelHeadingRad`);
6. derive and PUBLISH the ONE shared traction severity
   (`tractionSeverity`) from the POST-TRACTION speed and the PUBLISHED
   slip, so the tires, the HUD, the camera and the debug seam all read
   the SAME number and NO consumer re-derives it;
7. (THE RETIRED SEAM - AC-090 VERDICT 8) the published pose is the
   REACHED pose: the AC-081 TRACK CONTAINMENT is retired with the
   invisible wall it made, so the car that left the ribbon keeps
   rolling under the USER-named off-track speed cap (the ONE named
   `NEATRACING_PHYSICS_OFF_TRACK_SPEED_LIMIT_KMH`) - the geometry
   argument rides the retired NO-OP seam and the SHIPPED 4-arg
   live-tick form is preserved;
8. count the tick.

Parameters:
- `state` - Current car state (treated read-only; never mutated).
- `input` - Control sample applied for this single tick.
- `surfaceResolver` - Optional surface probe; receives the POST-MOVE
pose (the pose the car reached in this tick - never the tick-start
state) and must return a `PhysicsSurfaceType`. When omitted the tick
behaves as on-road. Off-track detection (which road/curb/grass the
ribbon assigns) is the consumer's job (P1-S7-C).
- `trackSegments` - The RETIRED TRACK GEOMETRY parameter (the
ribbon segments). The AC-081 containment bound was RETIRED with the
invisible wall it made (AC-090 VERDICT 8: "it's now an invisible
wall ... the car should be able to go off track, but with the speed
limited to 20km/h"), so this argument is NEVER READ any more: an
ABSENT parameter, an EMPTY list and a FULL ribbon all produce THE
SAME tick (the retired seam is a NO-OP - the AC-081-WIRE-NOOP
contract re-contracted by the retirement). The parameter STAYS the
SHIPPED 4-arg live-tick form, so the served closure
(`browser-entry`), the served lane and the contract lanes keep ONE
tick form; the off-track limit is only ever the USER-named speed
cap.

Returns: A new `PhysicsCarState` (input argument is never mutated).

Example:

```ts
const car = stepPhysics(createInitialPhysicsCarState(42), {
  throttle: 1, brake: 0, steer: 0,
});
console.log(car.speed > 0); // one tick of full throttle
```

### stepPhysicsFleet

```ts
stepPhysicsFleet(
  fleet: readonly PhysicsCarState[],
  inputs: readonly PhysicsStepInput[],
  surfaceResolver: ((carState: PhysicsCarState) => PhysicsSurfaceType) | undefined,
): PhysicsCarState[]
```

Advances a whole fleet by exactly ONE deterministic 60 Hz fixed tick and
resolves the car-to-car overlaps (AC-090).

The ORDER is LOAD-BEARING: every member runs the SAME tick pipeline
FIRST (so the whole fleet sits at ONE time base) and the car-to-car
overlaps are resolved AFTER the drift integration and BEFORE the states
are published. The single-car tick (`stepPhysics`) stays the hot path of
the live session and never resolves a fleet; this pack tick is the entry
point a multi-car session drives, and it resolves the fleet EXACTLY ONCE
per tick (a per-car resolution would resolve the whole fleet N times per
tick).

Parameters:
- `fleet` - The fleet states at TICK START (treated read-only; never
mutated), index-aligned with `inputs`.
- `inputs` - The control sample of every member, index-aligned with
`fleet` (a missing sample is a caller error).
- `surfaceResolver` - Optional surface probe, shared by the fleet; it
receives the POST-MOVE pose the probed member reached in this tick
(never the tick-start state).

Returns: Fresh published states in the same order.

Example:

```ts
const fleet = stepPhysicsFleet([carA, carB], [inputA, inputB]);
console.log(fleet.length); // 2
```

### wrapHeading

```ts
wrapHeading(
  radians: number,
): number
```

Wraps a heading into `[-Ï€, Ï€)` with one O(1) pass (no spin-up loops).

## physics/ai.autopilot.fixtures.ts

Fixture track for the autopilot green tests: one straight closed loop
(N by the sample stride) of lane-center samples along +x at y = 0. The
closed loop is formed by the wrap from the last sample back to the
first, matching the frozen `SplineSample` order contract.

### createTrackSpecForAutopilotFixture

```ts
createTrackSpecForAutopilotFixture(): readonly SplineSample[]
```

Builds the straight-loop spline fixture: 40 samples, x = 0..39 at
y = 0, `globalIndex` equal to the array index.

Returns: The frozen sample list for direct `stepAutopilotInput` calls.

## physics/collision.utils.ts

Collision / boundary / off-track surface utils for the neatRacing
example (plan slice P1-S7-C; AC-024 partial, spec sections 4.3, 5.1
D6, 5.2, LD-04/LD-05).

Arcade collision model (LD-05): the car-vs-car bounce is frictionless
and perfectly elastic — kinetic energy and normal momentum are
conserved, the relative normal velocity reverses, tangential velocity
is preserved — with NO deformation and NO crash penalties or rewards
beyond the bounce itself (LD-04). The car-vs-track-boundary law is
NOT a bounce and NOT a wall: there is NO track-boundary containment
at all (the AC-090 VERDICT-8 supersession), so a car that leaves the
ribbon really goes off track and keeps rolling. Car bodies are
circles sized by the D6 fixed physics constant
{@link NEATRACING_CAR_COLLISION_RADIUS_M}, never derived from voxel
models, so `.bin` rebuilds stay gameplay neutral (renderer-only).

Surface bands ARE the slab rasterizer's bands (`../track/track.slabs`):
lateral bands measured from the segment chords as road
(`distance <= halfWidth - curb`), curb (`<= halfWidth`), the
DRIVABLE off-track band (`<= halfWidth + apron`) and the grass
beyond it (physics has no air surface, so every point off the
ribbon reads as off-track grass). Junctions resolve by the
rasterizer's priority rule (road before curb before grass). The
band WIDTHS are not mirrored here - they are READ from the
band-geometry owner ({@link NEATRACING_TRACK_GRASS_APRON_WIDTH_M}),
so the rasterizer and this classifier can never disagree about where
a band ends. The same band law is exposed as a NAMED, testable
classification ({@link TrackSurfaceClassification}) so a traction or
slide measurement can be asserted ROAD-ONLY instead of re-deriving
the bands at the call site.

THE EDGE IS NOT THIS MODULE'S TO DECIDE: the half-width of every
chord is the NAMED asymmetric half-width published by the edge seam
(`../track/track.edge`) - the inner half-width on the inner side of
a corner (the road never narrows) and the widened outer half-width on
the outer side - and the chord distance is the seam's canonical
measure. This module derives NO half-width of its own, so the
rasterizer, the tri-colour band paint and this module can never
disagree about how WIDE the road is. The physics plane IS the
generator plane (the car state reads the generator x/y directly), so
the generator-plane entry points of the seam apply here.

NO BOUND (AC-090 VERDICT 8, superseding AC-081/AC-094 (d)): the
track has NO invisible wall. The off-track band is DRIVABLE off-road
(the EXISTING 'grass' surface member) and the grass BEYOND the band
is drivable TOO - the car crosses the edge, keeps rolling, and is
limited only by the off-track SPEED cap (the NAMED 20 km/h off-track
limit owned by the physics step service), never by a geometric bound.
The road edge is therefore an INNER boundary only: a car that has
crossed the edge really ends up off the road, and no law ever reseats
it on the ribbon nor resets its longitudinal progress. Because there
is no bound at all, there is nothing that could MASK the AI steering
defect owned by Step 14M (AC-085) either: a car that has left the
ribbon is simply off the ribbon, and a car riding the curb is still
not road-only.

Determinism: every util is pure and state-free — inputs are never
mutated, resolved states are fresh records, and identical inputs on
the same runtime always produce byte-identical `JSON.stringify`
output (same-runtime reproducibility rung, ordered deterministic
arithmetic, no `Math.random`). The ONE exception is the
CONTENT-GUARDED edge-profile memo
({@link resolveEdgeProfileForSegments}): a pure cache that changes
only HOW OFTEN the published profile is resolved, NEVER the resolved
values themselves.

### classifySurfaceAt

```ts
classifySurfaceAt(
  pointX: number,
  pointY: number,
  edgeProfile: TrackEdgeProfile,
): TrackSurfaceClassification
```

Classifies one ground-plane point with the slab rasterizer's band and
priority rules: road core, then curb, then the DRIVABLE off-track band
(the grass apron), then the grass beyond it (physics has no air
surface, so beyond every band the car is off-track grass). The bands
are the rasterizer's UNION rule (a point is road when ANY segment's
road core holds it), and the result is the NAMED classification
record — the ONE band law behind both the bare resolver and the
classifier.

THE BAND IS THE SAME BAND THE RASTERIZER PAINTS (AC-094 (e)): the band
test is the rasterizer's own apron test
(`distance <= halfWidth + apron`), with the apron width READ from the
band-geometry owner, so a point on the band is the SAME surface in both
models and the band boundary is ONE boundary.

The half-width of every chord is the PUBLISHED half-width of the side
the point sits on, so the road is WIDER on the outer side of a corner
and never narrower on the inner side.

### classifyTrackSurface

```ts
classifyTrackSurface(
  car: PhysicsCarState,
  segments: readonly TrackSegment[],
): TrackSurfaceClassification
```

Classifies the ground under ONE car state: the single-shot form of
{@link createTrackSurfaceClassifier}, over the same band law and the
same pure function of the track segments.

Parameters:
- `car` - Car state, treated read-only; never mutated.
- `segments` - Closed-loop ribbon segments (ordered, read only).

Returns: The  {@link TrackSurfaceClassification} of the ground under
 *   the car.
 *

Example:

```ts
const lane = classifyTrackSurface(car, spec.segments);
console.log(lane.isRoadOnly); // the ROAD-ONLY measurement lane
```

### CollisionNormal

Unit collision normal pointing from car A toward car B.

### createTrackSurfaceClassifier

```ts
createTrackSurfaceClassifier(
  segments: readonly TrackSegment[],
): (car: PhysicsCarState) => TrackSurfaceClassification
```

Creates the pure NAMED surface classifier for a fixed track
geometry: the returned probe maps a car state to the
{@link TrackSurfaceClassification} of the ground under it, so a
traction or slide measurement can be asserted ROAD-ONLY (AC-087)
instead of re-deriving the bands at the call site.

The classifier is a PURE function of the track segments (the track
folder owns no off-track logic), it never mutates its inputs, and
identical inputs always produce the identical record.

Parameters:
- `segments` - Closed-loop ribbon segments (ordered, read only).

Returns: A pure `(car) => TrackSurfaceClassification` probe.

Example:

```ts
const classify = createTrackSurfaceClassifier(spec.segments);
const lane = classify(car);
console.log(lane.surface, lane.isRoadOnly); // 'road' true
```

### createTrackSurfaceResolver

```ts
createTrackSurfaceResolver(
  segments: readonly TrackSegment[],
): (car: PhysicsCarState) => PhysicsSurfaceType
```

Creates a pure surface resolver for a fixed track geometry: the
returned probe maps a car state to the physics surface under it with
the same nearest-chord lateral bands the slab rasterizer paints
(junction priority: road before curb before the off-track band; beyond
every band the car is off-track grass).

The resolver reads the SAME band law as the named classification
({@link createTrackSurfaceClassifier}) — there is ONE band
implementation, never two — and the existing callers are unchanged:
the probe still answers with the bare surface label, so it stays
safe to pass straight into `stepPhysics` as its optional surface
resolver.

Parameters:
- `segments` - Closed-loop ribbon segments (ordered, read only).

Returns: A pure `(car) => PhysicsSurfaceType` probe — the same
segments always resolve to the same surface.

Example:

```ts
const surfaceResolver = createTrackSurfaceResolver(spec.segments);
const surface = surfaceResolver(car); // 'road' | 'curb' | 'grass'
```

### displaceCar

```ts
displaceCar(
  car: PhysicsCarState,
  offsetX: number,
  offsetY: number,
): PhysicsCarState
```

Offsets a car state's world position (pure; fresh record).

### isEdgeProfileCurrent

```ts
isEdgeProfileCurrent(
  edgeProfile: TrackEdgeProfile,
  segments: readonly TrackSegment[],
): boolean
```

Reports whether a memoized edge profile still describes the CONTENT of
the caller's segments array.

The seam publishes a SNAPSHOT of the segments it resolved from, so a
caller that changes the geometry IN PLACE would otherwise keep reading
the profile resolved from the OLD content. The guard compares the live
segments against that snapshot - the five numeric fields of every
segment - and allocates NOTHING, so the hot path stays free of
per-frame garbage.

Parameters:
- `edgeProfile` - The memoized profile to check.
- `segments` - The live segments array the caller passed.

Returns: TRUE when the profile still describes the live array.

### NEATRACING_CAR_COLLISION_RADIUS_M

Fixed arcade collision radius of every car body, in meters (D6): one
named fixed physics constant for the whole fleet, deliberately NOT
derived from voxel models (`.bin` rebuilds are renderer-only
artifacts with zero sim-side effect). The 1.0 m half-span stands in
for the ~2 m arcade car body diameter.

### NEATRACING_TRACK_CURB_WIDTH_M

Width of the curb band on each side of the road core, in meters - the
PUBLISHED curb-band width of the tri-colour band.

THE BAND GEOMETRY IS PUBLISHED HERE, NOT MIRRORED: this module owns the
two band widths of the tri-colour band (the curb band and the grass
apron), and the physics surface classifier and the ONE containment bound
read them from HERE (through the re-export in
`../physics/collision.utils`), so the rasterizer and the physics model
can never disagree about where a band ends.

### NEATRACING_TRACK_GRASS_APRON_WIDTH_M

Width of the grass apron beyond the curb on each side, in meters - the
PUBLISHED width of the OFF-TRACK BAND (AC-094 (e)).

The apron IS the off-track band: it stays PAINTED as off-road (the paint
ladder is unchanged - the palette values, their order and their
cumulative semantics are untouched), and it is also the band the physics
surface classifier reads as the DRIVABLE off-road surface and the ONE
containment bound measures to (the band outer edge). Both read THIS
constant through the re-export in `../physics/collision.utils`, so the
rasterizer and the physics model share ONE band boundary.

### recomposeCarWithVelocity

```ts
recomposeCarWithVelocity(
  car: PhysicsCarState,
  velocityX: number,
  velocityY: number,
): PhysicsCarState
```

Rebuilds a car state around a target world velocity: the velocity-
aligned heading representation (speed = world-velocity magnitude,
heading = its direction, drift channels zeroed because the
frictionless bounce carries no slip or lateral component). Every
recomposed field stays inside the `PhysicsCarState` contract
(`speed >= 0`, zero slip/lat) so the next `stepPhysics` tick
integrates the bounced state exactly as written. A standing target
(magnitude within tolerance) keeps the heading untouched instead of
snapping to a degenerate `atan2` axis. The drift channel is zeroed on
both branches (a bounce is a release, so the latch and the RE-ARM token
are cleared).

### resolveCarCollisions

```ts
resolveCarCollisions(
  cars: readonly PhysicsCarState[],
): PhysicsCarState[]
```

Resolves every pairwise car-car overlap with the LD-05 arcade
elastic bounce: overlapping circular bodies (D6 radius) exchange
world velocities along the collision normal (equal masses, no
friction — kinetic energy and normal momentum are conserved, the
relative normal velocity reverses, tangential velocity is
preserved) and are then pushed apart to exactly the sum of the
radii, half per car. Only an approaching overlap takes an impulse —
a separating overlap still separates positionally without adding
energy. No deformation or reward applies (LD-04).

Pair resolution order follows car index order (A-B, A-C, B-C, ...),
so multi-car pile-ups resolve deterministically.

Parameters:
- `cars` - Car states, treated read-only; never mutated.

Returns: Fresh car states in the same order; non-overlapping cars
are returned byte-identical (same values, new records).

Example:

```ts
const [resolvedA, resolvedB] = resolveCarCollisions([carA, carB]);
console.log(
  Math.hypot(resolvedA.x - resolvedB.x, resolvedA.y - resolvedB.y),
); // at least 2 * NEATRACING_CAR_COLLISION_RADIUS_M
```

### resolveCollisionNormal

```ts
resolveCollisionNormal(
  rawOffsetX: number,
  rawOffsetY: number,
  centerDistance: number,
): CollisionNormal
```

Resolves the pair collision normal (points from car A toward car B),
falling back to the fixed +x tiebreak axis for coincident centers.

### ResolvedCarPair

One resolved car-car pair: fresh states that replace the pair's
slots in the resolved cars list, in the same order.

### resolveEdgeProfileForSegments

```ts
resolveEdgeProfileForSegments(
  segments: readonly TrackSegment[],
): TrackEdgeProfile
```

Resolves the published edge profile for a segments array, REUSING the
profile already resolved for that exact array identity.

The profile is resolved ONCE per segments-ARRAY IDENTITY and then
reused, so the live tick resolves NO profile and allocates NO array
and NO chord per car per frame. The reuse is CONTENT-GUARDED
({@link isEdgeProfileCurrent}): a memoized profile is handed back only
while the caller's array still holds the content the profile was
resolved from, so a changed geometry is NEVER served a stale profile -
it is re-resolved instead.

The returned profile is the SEAM'S OWN profile (never a copy of it), so
the published chords and the segment snapshot stay the ONE geometry
every reader measures the road through.

Parameters:
- `segments` - Closed-loop ribbon segments (ordered, read only).

Returns: The published edge profile for these segments.

### resolveNormalElasticExchange

```ts
resolveNormalElasticExchange(
  carA: PhysicsCarState,
  carB: PhysicsCarState,
  collisionNormal: CollisionNormal,
): VelocityExchange
```

Exchanges the paired world velocities along the collision normal for
the equal-mass frictionless arcade bounce: only an approaching pair
(`(vA - vB) · n > 0`) takes an impulse; a separating pair keeps its
velocities (no added energy).

### resolveOverlappingPair

```ts
resolveOverlappingPair(
  carA: PhysicsCarState,
  carB: PhysicsCarState,
): ResolvedCarPair | null
```

Resolves one overlapping car pair: geometry, normal elastic exchange,
and half-per-car re-separation. Returns `null` when the pair does not
overlap (kept untouched).

### resolvePairImpulsesInOrder

```ts
resolvePairImpulsesInOrder(
  cars: PhysicsCarState[],
): void
```

Applies the pairwise elastic exchange and separation in index order.

### resolveWorldVelocity

```ts
resolveWorldVelocity(
  car: PhysicsCarState,
): readonly [number, number]
```

World velocity `[vx, vy]` of a car state, in m/s: the heading-
longitudinal axis plus the perpendicular lateral axis — the exact
velocity algebra the red contract's `worldVelocityOf` helper (and
`stepPhysics` translation) shares, so bounces act on the same
velocity components the integrator moves.

### TrackSurfaceClassification

The NAMED, testable surface classification of the ground under a
car: the surface label plus the two predicates a measurement lane
asserts (Step 14P, AC-087).

The classification is the resolver's band law exposed as a RECORD, so
a slide, camera or traction measurement can assert that the surface
under the car is ROAD-ONLY instead of re-deriving the bands at the
call site. It is a PURE function of the track segments (the track
folder owns no off-track logic) and the existing resolver callers are
unchanged.

### VelocityExchange

Exchanged pair of world velocities after the normal-axis elastic
impulse (tangential components untouched).

### wrapHeading

```ts
wrapHeading(
  radians: number,
): number
```

Wraps a heading into `[-π, π)` with one O(1) pass — the local mirror
of the private header wrap in `physics.step.service` (the P1-S7-B
slice boundary owns that module, so this file re-states the wrap).

### zeroBouncedDriftChannel

```ts
zeroBouncedDriftChannel(): DriftChannelState
```

The fully ZEROED drift channel of a bounced car (P0-3, AC-079).

A bounce is a RELEASE, not a spawn: the latch is cleared AND the
RE-ARM token is cleared, so the next legitimate drift has to re-latch
from scratch (the drift model re-latches the token when the wheel goes
neutral). The spawn factory arms the token TRUE - that is the SPAWN
contract - so the bounce reset derives its own zeroed channel from the
factory shape instead of reusing the armed spawn channel.

Returns: A zeroed drift channel (every numeric field 0, the token
cleared).
