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

### RacingCarState

Concrete racing-car alias kept for the browser and worker seams.

### TireStateTuple

Ordered tire-health tuple for one car.

The layout is always `[frontLeft, frontRight, rearLeft, rearRight]`, with
each channel clamped to the closed `[0, 1]` interval.

## environment/environment.step.service.ts

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
inside the frozen `TrackSpec`. Cars may claim up to one own-team slot each,
but cars released earlier in the same tick cannot re-enter immediately.

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
