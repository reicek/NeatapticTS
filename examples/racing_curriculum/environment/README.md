# environment

## environment/environment.types.ts

### CarControlOutput

Control signal from the car's controller for one simulation tick.

Both values are normalised to [-1, 1] and clamped inside `stepEnvironment`.

### CarState

Per-car racing state tracked by the environment.

Tire health stays owner-local on each car so grip, pit restore, worker
projection, and observation assembly all read the same state regardless of
whether the active pack is solo, 2v2, or 3v3.

### EnvironmentState

Runtime state for one simulation episode.

The legacy top-level `carX`, `carY`, and `carHeading` fields stay in place so
the solo browser path remains stable. Multi-car packs layer the ordered `cars`
roster on top; the six-car 3v3 slice simply sets `cars.length = 6` while
reusing the same six-slot `pitOccupancy` and compatibility `pitStatus` shelf.

### PitOccupancyRecord

Fixed pit occupancy record for one team-owned pit slot.

`occupyingCarIndex = 255` means the slot is empty; otherwise
`remainingStopTicks` counts down the fixed stop duration before the
environment releases the car and restores all four tire channels back to
full health.

### PitOccupancyState

Fixed six-record pit shelf shared by Team A and Team B.

The canonical slot order is `[A0, A1, A2, B0, B1, B2]`. The shelf width does
not change with roster size, so single-car, 2v2, and 3v3 packs all reuse this
same six-slot state with the `255 = no car` sentinel and fixed stop-tick
contract.

### PitStrategyState

Pit/strategy sensory state appended to the Tier 4/5 observation tail.

All channels are normalized to `[0, 1]` (or zero when unavailable). The
assembler treats every field as optional so that Tier 1–3 code paths stay
unchanged until the race-pack service explicitly provides pit/strategy data.

The canonical channel order matches offsets `[95..102]` of the Tier 4/5
observation vector:
  1. `pitDistanceToEntrance01`
  2. `pitOccupancyStatus`
  3. `lapsSincePit`
  4. `teammatePitStatus`
  5. `tireDegradationRate`
  6. `estimatedLapsBeforeFailure`
  7. `reservedPitContext1`
  8. `reservedPitContext2`

`teammatePitStatus` is high whenever the team pit box is occupied by any
team member, including the querying car itself. Because each team has its
own box, the channel functions as a "team box busy" signal rather than a
strict teammate-other-than-self flag.

### RacingCarState

Concrete racing-car alias kept for the browser and worker seams.

### TireStateTuple

Ordered tire-health tuple for one car.

The layout is always `[frontLeft, frontRight, rearLeft, rearRight]`, with
each channel clamped to the closed `[0, 1]` interval.

## environment/environment.step.service.ts

### applyPitHold

```ts
applyPitHold(
  cars: readonly CarState[],
  pitOccupancy: PitOccupancyState,
  trackSpec: TrackSpec,
): CarState[]
```

Locks every car that is actively serving a pit stop to its assigned box center.

When a car enters a pit box in `resolvePitEntries`, the occupancy record already
starts counting down `remainingStopTicks`. This helper runs after entry
resolution so the same tick that claims the slot also teleports the car from
the entrance corridor to `boxCenter`, and every following tick keeps the car
parked there until the stop expires.

Parameters:
- `cars` - Updated car roster after separation.
- `pitOccupancy` - Pit occupancy shelf after entry resolution.
- `trackSpec` - Active track metadata.

Returns: Car roster with pitting cars pinned to their box centers.

### clampCarToTrackBounds

```ts
clampCarToTrackBounds(
  car: CarState,
  trackSpec: TrackSpec,
): CarState
```

Pulls a car's world position back onto the drivable ribbon if it has crossed
either the inner or outer track edge.

The ribbon is approximated by the nearest spline sample: the car is clamped
so its signed lateral offset stays within `[-halfWidth, halfWidth]`. Cars that
are stopped in a pit box are intentionally skipped, because pit stalls live
outside the drivable surface.

Parameters:
- `car` - Car whose position should be clamped.
- `trackSpec` - Frozen track geometry used for the boundary lookup.

Returns: Car state with its position bounded to the track ribbon.

### clampControlValue

```ts
clampControlValue(
  value: number,
): number
```

Clamps control values to the accepted controller range.

Parameters:
- `value` - Incoming control signal.

Returns: Value clamped to [-1, 1].

### clampUnitInterval

```ts
clampUnitInterval(
  value: number,
): number
```

Clamps a value to the closed `[0, 1]` interval.

Parameters:
- `value` - Incoming floating-point value.

Returns: Value clamped to the unit interval.

### clonePitOccupancy

```ts
clonePitOccupancy(
  pitOccupancy: PitOccupancyState,
): MutablePitOccupancyState
```

Clones the fixed six-slot pit shelf into a mutable array.

Parameters:
- `pitOccupancy` - Source pit occupancy state.

Returns: Mutable clone suitable for in-step updates.

### computeGuideDivergencePenalty

```ts
computeGuideDivergencePenalty(
  guideOffsetNormalized: number,
): number
```

Computes a divergence penalty when the car strays far from the guide line
while it is available.

Returns {@link GUIDE_DIVERGENCE_PENALTY} when the normalized lateral offset
exceeds {@link GUIDE_OFFSET_MODERATE_THRESHOLD}, and 0 otherwise.

Parameters:
- `guideOffsetNormalized` - Signed normalized lateral offset from the guide line.

Returns: Negative penalty or 0.

### computeGuideFollowReward

```ts
computeGuideFollowReward(
  guideOffsetNormalized: number,
): number
```

Computes a positive guide-following reward based on how close the car is
to the guide line.

Returns {@link GUIDE_FOLLOW_REWARD_CLOSE} when the normalized lateral offset
is very small (< {@link GUIDE_OFFSET_CLOSE_THRESHOLD}), a smaller
{@link GUIDE_FOLLOW_REWARD_MODERATE} when moderately close (<
{@link GUIDE_OFFSET_MODERATE_THRESHOLD}), and 0 when far from the guide.

Parameters:
- `guideOffsetNormalized` - Signed normalized lateral offset from the guide line.

Returns: Positive reward or 0.

### createEmptyPitOccupancy

```ts
createEmptyPitOccupancy(): PitOccupancyState
```

Creates an empty six-slot pit occupancy shelf.

Returns: Six-slot pit occupancy shelf with no cars assigned.

### createEmptyPitRecord

```ts
createEmptyPitRecord(): PitOccupancyRecord
```

Creates one empty pit occupancy record.

Returns: Empty record using the `255` no-car sentinel.

### createFallbackPrimaryCar

```ts
createFallbackPrimaryCar(): CarState
```

Creates a primary-car fallback used only when the roster is unexpectedly empty.

Returns: Neutral Team A fallback car.

### createInitialCars

```ts
createInitialCars(): readonly CarState[]
```

Creates the canonical six-car owner-local roster.

Returns: Ordered 3v3 car list `[A0, A1, A2, B0, B1, B2]` with full tire health.

### createInitialState

```ts
createInitialState(): EnvironmentState
```

Creates the canonical start-of-episode environment state.

Tier 5 seeds the owner-local 3v3 layout up front so tire decay, pit status,
observation assembly, and renderer fallbacks all share the same initial state.
Both `pitOccupancy` and the compatibility alias `pitStatus` point at the same
empty six-slot shelf, and every car starts with fully healthy tires.

Returns: Fresh environment state with six cars, healthy tires, and empty pits.

### decayTireState

```ts
decayTireState(
  current: readonly [number, number, number, number],
  lateralForce: number,
  longitudinalForce: number,
  speed: number,
): [number, number, number, number]
```

Decays all four tire channels using the pinned Tier 4 degradation formula.

The base wear for one step is
`|lateralForce| * lateralFactor + |longitudinalForce| * longitudinalFactor + |speed| * speedFactor`.
Each tire then multiplies that base wear by `1 + (1 - tireHealth) * accelerationFactor`,
so already-degraded tires lose grip faster than fresh tires under the same load.
The final value is always clamped to the closed `[0.0, 1.0]` interval.

Parameters:
- `current` - Current tire-health tuple `[FL, FR, RL, RR]`.
- `lateralForce` - Absolute lateral load for the current step.
- `longitudinalForce` - Absolute longitudinal load for the current step.
- `speed` - Scalar speed used by the degradation model.

Returns: New clamped tire-health tuple after one decay step.

Example:

```ts
const freshTires = decayTireState([1, 1, 1, 1], 0.2, 0.4, 10);
const wornTires = decayTireState([0.3, 0.3, 0.3, 0.3], 0.2, 0.4, 10);

wornTires[0] < freshTires[0]; // already-damaged tires decay faster
```

### detectWrongDirection

```ts
detectWrongDirection(
  beforeCars: readonly CarState[],
  afterCars: readonly CarState[],
  trackSpec: TrackSpec,
): readonly boolean[]
```

Detects cars that moved opposite to the track tangent during this step.

A car is flagged when its displacement vector has a negative dot product
with the forward tangent at its pre-step nearest sample. Stationary cars are
never flagged, so a parked car cannot accumulate wrong-direction penalties.

Parameters:
- `beforeCars` - Car roster before the kinematic update.
- `afterCars` - Car roster after the kinematic update.
- `trackSpec` - Active track geometry.

Returns: Per-car boolean flags; `true` means wrong-direction motion.

### EnvironmentControlInput

Accepted control input for one deterministic environment step.

### isCarStoppedInPit

```ts
isCarStoppedInPit(
  pitOccupancy: PitOccupancyState,
  carIndex: number,
): boolean
```

Returns whether the given car is currently waiting out an active pit stop.

Parameters:
- `pitOccupancy` - Current pit occupancy shelf.
- `carIndex` - Car index being checked.

Returns: True when the car is locked in a pit box this tick.

### isPointInsideAabb

```ts
isPointInsideAabb(
  x: number,
  y: number,
  axisAlignedBox: TrackAabb,
): boolean
```

Returns whether the point lies inside the provided axis-aligned rectangle.

Parameters:
- `x` - Point X coordinate.
- `y` - Point Y coordinate.
- `axisAlignedBox` - Rectangle to test.

Returns: True when the point is inside or on the rectangle boundary.

### normalizePitOccupancy

```ts
normalizePitOccupancy(
  pitOccupancy: PitOccupancyState,
): PitOccupancyState
```

Normalizes incoming pit occupancy state to the fixed six-slot layout.

Legacy two-slot inputs are mapped from `[teamA, teamB]` to
`[A0, A1, A2, B0, B1, B2]` by placing Team A at slot `0` and Team B at
slot `3`.

Parameters:
- `pitOccupancy` - Incoming pit occupancy state.

Returns: Six-slot normalized pit occupancy shelf.

### resolveCars

```ts
resolveCars(
  state: EnvironmentState,
): readonly CarState[]
```

Resolves the active car roster from legacy or Tier 4 state shapes.

Parameters:
- `state` - Current environment state.

Returns: Ordered car list to step for the current tick.

### resolveControls

```ts
resolveControls(
  control: EnvironmentControlInput,
  carCount: number,
): readonly CarControlOutput[]
```

Resolves the ordered control list for the active car roster.

Parameters:
- `control` - Single-car or per-car control input.
- `carCount` - Number of cars that will be stepped this tick.

Returns: Per-car control list aligned to the roster order.

### resolveMeanTireHealth

```ts
resolveMeanTireHealth(
  tireState: TireStateTuple,
): number
```

Computes the mean tire-health value used by the grip model.

Parameters:
- `tireState` - Ordered tire-health tuple.

Returns: Mean health across all four corners.

### resolveNearestSampleIndex

```ts
resolveNearestSampleIndex(
  x: number,
  y: number,
  trackSpec: TrackSpec,
): number
```

Finds the spline sample nearest to a world-space point.

Used by both boundary clamping and wrong-direction detection so both
features agree on the local track frame.

Parameters:
- `x` - Point X coordinate.
- `y` - Point Y coordinate.
- `trackSpec` - Active track geometry.

Returns: Index of the nearest spline sample.

### resolveOptimalLineLateralOffsetNormalized

```ts
resolveOptimalLineLateralOffsetNormalized(
  car: CarState,
  trackSpec: TrackSpec,
): number
```

Computes the normalized lateral offset from the car to its team's optimal
lane centerline (guide line).

Replicates the computation in the observation assembler: the signed lateral
offset from the nearest spline sample is subtracted by the team-specific
lane centerline offset, then normalized by
{@link GUIDE_OFFSET_NORMALIZATION_SCALE} world units. The result is a
signed value where 0 means the car is exactly on the guide line.

Parameters:
- `car` - Car whose offset should be computed.
- `trackSpec` - Frozen track geometry.

Returns: Normalized lateral offset; 0 when the track has no spline samples.

### resolvePitEntries

```ts
resolvePitEntries(
  cars: readonly CarState[],
  pitOccupancy: PitOccupancyState,
  trackSpec: TrackSpec,
  releasedCars: ReadonlySet<number>,
): PitOccupancyState
```

Detects new pit entries after the current tick's car updates complete.

Entry is based on any of the team's `entranceCorridor` axis-aligned boxes
inside the frozen `TrackSpec`. A car is admitted only when its mean tire
health is below `PIT_SERVICE_TIRE_HEALTH_THRESHOLD`, so freshly serviced
cars sitting at `boxCenter` (which may still be inside the same AABB) are
not immediately re-trapped. Cars may claim up to one own-team slot each,
and cars released earlier in the same tick cannot re-enter immediately.

Parameters:
- `cars` - Updated car roster.
- `pitOccupancy` - Pit occupancy shelf after ticking active stops.
- `trackSpec` - Active track metadata.
- `releasedCars` - Cars released this tick and therefore blocked from re-entry.

Returns: Final pit occupancy shelf for the next state.

### resolvePitOccupancy

```ts
resolvePitOccupancy(
  state: EnvironmentState,
): PitOccupancyState
```

Resolves the current pit occupancy shelf from either Tier 4 field name.

Parameters:
- `state` - Current environment state.

Returns: Six-slot pit occupancy shelf.

### RewardShapingStateExtensions

Optional runtime extensions carried on the environment state for reward shaping.

These fields are not part of the canonical {@link EnvironmentState} type but
may be set by callers (e.g. the browser harness) and are preserved across
step calls via object spread. `guidanceAlpha` controls whether guide-following
rewards and divergence penalties are active (Tier 0–1: alpha > 0, Tier 2+:
alpha = 0). `consecutiveBorderContactTicks` tracks per-car escalating
border-contact penalty state across ticks. `consecutiveWrongDirectionTicks`
tracks per-car escalating wrong-direction penalty state across ticks.

### separateCars

```ts
separateCars(
  cars: readonly CarState[],
): readonly CarState[]
```

Pushes overlapping car centers apart so bounding boxes never overlap.

Uses axis-aligned bounding boxes with half-extents {@link CAR_HALF_WIDTH}
along X and {@link CAR_HALF_LENGTH} along Y. For each overlapping pair the
required center-to-center distance is computed along the connecting line so
that either the X gap exceeds the combined half-widths or the Y gap exceeds
the combined half-lengths, whichever is smaller. A minimum Euclidean
separation of {@link CAR_MIN_CENTER_SEPARATION} is also enforced. The solver
iterates up to {@link SEPARATION_MAX_ITERATIONS} times to resolve cascading
overlaps in multi-car stacks.

Parameters:
- `cars` - Car roster after track-boundary clamping.

Returns: New roster with overlapping cars separated in place.

### stepCarKinematics

```ts
stepCarKinematics(
  car: CarState,
  control: CarControlOutput,
): CarState
```

Steps one car forward using the pinned Tier 4 grip and decay rules.

Parameters:
- `car` - Current car state.
- `control` - Controller output for the car.

Returns: New car state after one fixed timestep.

### stepEnvironment

```ts
stepEnvironment(
  state: EnvironmentState,
  control: EnvironmentControlInput,
): EnvironmentState
```

Advances the environment by exactly one fixed timestep.

The returned state is a new object; the input state is never mutated.
`tick` is incremented by exactly one. The stepping path is generic over the
resolved car roster length, so solo callers may still pass a single control
object while multi-car packs supply one control per car. In the 3v3 slice,
`state.cars.length = 6` and the same loop structure advances all six cars.

Pit lifecycle semantics are intentionally narrow and deterministic: each team
owns three pit slots (`255` means a slot is empty), a car that touches one of
its team's entrance-corridor AABBs claims an available owned slot for a fixed
four-tick stop, and the environment restores all four tire channels to `1`
when that stop expires. Entry stays deterministic: cars claim in roster order
and each car can claim at most one slot per tick.

Parameters:
- `state` - Current environment state.
- `control` - Single-car control or ordered per-car controls for this tick.

Returns: New environment state after one fixed timestep.

### stepEnvironmentBatch

```ts
stepEnvironmentBatch(
  state: EnvironmentState,
  control: CarControlOutput,
  stepCount: number,
): EnvironmentState
```

Advances the environment by a fixed number of deterministic timesteps.

Parameters:
- `state` - Current environment state.
- `control` - Controller output applied for each batched step.
- `stepCount` - Number of fixed timesteps to apply.

Returns: New environment state after the batch completes.

### tickPitOccupancy

```ts
tickPitOccupancy(
  pitOccupancy: PitOccupancyState,
  cars: CarState[],
  releasedCars: Set<number>,
): PitOccupancyState
```

Ticks active pit stops forward and restores tires on release.

A record remains active while `remainingStopTicks > 0`. When the counter
reaches zero, the occupying car's tire tuple is reset to `[1, 1, 1, 1]`, the
released car index is marked for same-tick re-entry blocking, and the record
returns to the `255 = no car` sentinel state.

Parameters:
- `pitOccupancy` - Current pit occupancy shelf.
- `cars` - Ordered car roster for the current tick.
- `releasedCars` - Mutable set filled with car indices released this tick.

Returns: Updated pit occupancy shelf after decrementing stop timers.
