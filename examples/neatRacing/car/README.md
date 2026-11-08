# car

Front-wheel steer animation helper for the Neat Racing voxel racer.

AC-044: the visual yaw of both front wheels follows the steering input
axis up to a named max lock angle. Wheel spin speed and direction are
independent of this visual yaw: the steer transform is a separate rotation
around the car's up axis applied after the G9 mount yaw and before the
spin rotation.

## car/wheel.steer.ts

### clampSteerAxis

```ts
clampSteerAxis(
  steerAxis: number,
): number
```

Clamp a steering-axis sample to the normalized [-1, 1] range.
Non-finite values are coerced to 0 so a bad input sample can never
produce an unbounded wheel angle.

Parameters:
- `steerAxis` - Raw steering axis; positive steers counterclockwise.

Returns: A clamped axis value in [-1, 1].

### computeWheelSteerAngleRad

```ts
computeWheelSteerAngleRad(
  steerAxis: number,
): number
```

Compute the visual front-wheel steer angle for a steering-axis input.

Positive input (left / A key) produces a positive angle, which rotates
the front wheels counterclockwise around the car's up axis. Negative
input (right / D key) produces a negative angle. Over-range inputs are
clamped so the wheels never exceed {@link NEATRACING_MAX_WHEEL_STEER_RAD}.

Parameters:
- `steerAxis` - Steering axis in [-1, 1]; values outside this range
are clamped.

Returns: Steer angle in radians to apply to the front wheels.

### NEATRACING_MAX_WHEEL_STEER_RAD

Visual max lock angle for front wheels in radians (~35°).

## car/generate-car.ts

Car render-pose generator for the live drive loop.

Bridges the deterministic physics state to the car/glass pass pose bags:
it fills a stable mutable pose object in place (P9: no per-frame
allocation churn) and computes the front-wheel steer angle from the
current steering input.

### computeWheelSteerAngleRad

```ts
computeWheelSteerAngleRad(
  steerAxis: number,
): number
```

Compute the visual front-wheel steer angle for a steering-axis input.

Positive input (left / A key) produces a positive angle, which rotates
the front wheels counterclockwise around the car's up axis. Negative
input (right / D key) produces a negative angle. Over-range inputs are
clamped so the wheels never exceed {@link NEATRACING_MAX_WHEEL_STEER_RAD}.

Parameters:
- `steerAxis` - Steering axis in [-1, 1]; values outside this range
are clamped.

Returns: Steer angle in radians to apply to the front wheels.

### fillCarRenderPose

```ts
fillCarRenderPose(
  pose: MutableNeatRacingCarRenderPose,
  worldX: number,
  worldZ: number,
  yawRad: number,
  wheelSpinRad: number,
  teamColorId: number,
  steerInput: number | undefined,
): void
```

Fill a mutable car render pose from the already-transformed scene values.

The caller (browser-entry) owns the top-down -> scene coordinate mapping
(worldZ and yawRad), wheel-spin accumulation, and team color selection.
This helper centralizes the pose-bag shape so both the opaque and glass
passes receive the same fields.

Parameters:
- `pose` - Mutable pose bag to fill in place.
- `worldX` - Car world X in meters.
- `worldZ` - Car world Z in meters.
- `yawRad` - Car yaw around world up in radians.
- `wheelSpinRad` - Accumulated visual wheel spin in radians.
- `teamColorId` - Team color id for the AC-014 remap.
- `steerInput` - Current steering-axis input; clamped to [-1, 1] by
 *    {@link computeWheelSteerAngleRad} . Defaults to 0 when absent.

### MutableNeatRacingCarRenderPose

Mutable car pose bag shared by the opaque car pass and the glass pass.
Properties are intentionally writable so one object can be refreshed
in place every frame while the composer holds a stable array reference.

### NEATRACING_MAX_WHEEL_STEER_RAD

Visual max lock angle for front wheels in radians (~35°).
