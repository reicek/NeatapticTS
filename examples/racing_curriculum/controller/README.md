# controller

Deterministic scripted waypoint-following controller for the Tier 0
racing curriculum demo.

The controller maintains a target waypoint index that advances around the
closed-loop track as the car approaches each endpoint. Steering is a
proportional heading-error term; throttle is held at a fixed constant.

This controller is intentionally simple and deterministic. It remains useful
as a baseline reference lane for regression comparisons against NGE-backed
controller behavior while preserving the same `computeScriptedControl`
integration seam.

## controller/nge.controller.ts

### clampControlValue

```ts
clampControlValue(
  value: number,
): number
```

Clamps controller outputs into the accepted environment range.

Parameters:
- `value` - Raw controller output.

Returns: Value clamped to [-1, 1].

### createNgeController

```ts
createNgeController(
  network: RacingControllerNetwork,
  options: NgeControllerOptions,
): NgeController
```

Creates the owner-local NGE controller wrapper around the public `activate(...)`
inference surface.

Parameters:
- `network` - Public network object that exposes `activate(...)`.
- `options` - Optional tier and radio-channel overrides.

Returns: Stateful controller that maps network outputs to throttle and steer.

### createSingleCarRadioChannel

```ts
createSingleCarRadioChannel(
  radioDim: number,
): SingleCarRadioChannel
```

Creates the owner-local single-car radio seam used by Tier 2 self-monitoring.

Parameters:
- `radioDim` - Number of channels retained in the radio buffer.

Returns: Read/write self-monitoring radio channel.

### NgeController

Public controller surface consumed by the browser harness.

### NgeControllerOptions

Runtime options for the owner-local NGE controller seam.

All fields are optional. Tier 1 is the default; providing `radioChannel`
and `radioDim` enables the degenerate single-car self-monitoring path used
by Tier 2.

### NgeControllerTickEvidence

Tier-guide evidence sampled from one controller tick.

### NgeControllerTickResult

Controller tick result including control and adaptation evidence.

### normalizeControllerOutputs

```ts
normalizeControllerOutputs(
  controllerOutputs: number | readonly number[],
): readonly number[]
```

Normalizes controller outputs into a two-element array.

Parameters:
- `controllerOutputs` - Raw network output.

Returns: Two-element control vector.

### prepareObservationState

```ts
prepareObservationState(
  envState: RacingObservationState,
  controllerTier: ObservationTier,
  radioChannel: SingleCarRadioChannel,
): RacingObservationState
```

Enriches the environment snapshot with the current self-radio field when Tier 2
is active.

Parameters:
- `envState` - Current environment snapshot.
- `controllerTier` - Active controller tier.
- `radioChannel` - Single-car self-monitoring seam.

Returns: Observation-ready environment snapshot.

### RacingControllerNetwork

Public network surface required by the owner-local NGE controller seam.

### resolveGuidanceAlphaForTier

```ts
resolveGuidanceAlphaForTier(
  tier: 0 | 1 | 2,
): number
```

Resolves the optimal-line overlay alpha for each curriculum tier.

- `0` — full overlay (alpha 1.0); used for the scripted baseline before NGE
  control is active.
- `1` — faint overlay (alpha 0.35); keeps a subtle optimal-line hint while
  the solo NGE driver learns.
- `2` — overlay off (alpha 0); the network must self-navigate without the
  hint once the radio seam is live.

Parameters:
- `tier` - Curriculum tier index (0 = scripted baseline, 1 = solo NGE, 2 = radio-augmented).

Returns: Overlay alpha in [0, 1].

### resolveSelfMonitoringPayload

```ts
resolveSelfMonitoringPayload(
  envState: RacingObservationState,
): Float32Array<ArrayBufferLike>
```

Builds the seven-channel self-monitoring payload consumed by the Tier 2 radio seam.

Parameters:
- `envState` - Current environment snapshot.

Returns: Ordered seven-channel self-monitoring payload.

### resolveTickEvidence

```ts
resolveTickEvidence(
  observationVector: Float32Array<ArrayBufferLike>,
): NgeControllerTickEvidence
```

Extracts Tier 1 center-guide evidence channels from one normalized observation vector.

Parameters:
- `observationVector` - Active normalized observation vector.

Returns: Lateral error, heading alignment, and combined guidance-need signal.

### SingleCarRadioChannel

Mutable self-radio seam used by Tier 2 single-car self-monitoring.

## controller/runtime.adaptation.ts

### createRuntimeAdaptationEngine

```ts
createRuntimeAdaptationEngine(
  options: RuntimeAdaptationEngineOptions,
): RuntimeAdaptationEngine
```

Creates a reusable per-tick adaptation engine for racing runtime loops.

Parameters:
- `options` - Optional cadence, bounds, and evaluation policy.

Returns: Stateful runtime adaptation engine.

### evaluateRollingScoreWindow

```ts
evaluateRollingScoreWindow(
  network: default,
  scoreHistory: readonly number[],
): number
```

Lightweight default evaluator for rolling score history windows.

Parameters:
- `network` - Candidate network.
- `scoreHistory` - Rolling score window.

Returns: Combined trend/complexity score.

### RuntimeAdaptationCadenceMode

Cadence modes supported by the runtime adaptation engine.

### RuntimeAdaptationCadenceOptions

Cadence policy configuration for per-tick adaptation checks.

### RuntimeAdaptationEngine

Stateful runtime adaptation engine surface used by browser or worker loops.

### RuntimeAdaptationEngineOptions

Engine options used by the racing runtime adaptation POC.

### RuntimeAdaptationLimits

Hard bounds and cooldown controls for one adaptation step.

### RuntimeAdaptationOperation

Candidate mutation operations supported by the runtime adaptation engine.

### RuntimeAdaptationTelemetry

Adaptation telemetry emitted on every adaptation attempt.

### RuntimeAdaptationTickInput

Per-tick input contract for adaptation checks.

### RuntimeNetworkSizeSnapshot

Per-step network size snapshot used by adaptation telemetry.

## controller/scripted.controller.ts

### computeScriptedControl

```ts
computeScriptedControl(
  envState: EnvironmentState,
  trackSpec: TrackSpec,
  controllerState: ScriptedControllerState,
): CarControlOutput
```

Produces a control output that steers the car toward its next track waypoint.

Algorithm:
1. Check whether the car is within `WAYPOINT_ADVANCE_RADIUS_WORLD` of the
   current lane-center spline target — if so, advance to the next segment.
2. Compute the heading error from the car's current heading to the angle
   toward the lane-centered spline target.
3. Apply a proportional gain and clamp to [-1, 1] for the steer output.
4. Return constant throttle + computed steer.

Mutates `controllerState.targetSegmentIndex` in place.

Parameters:
- `envState` - Current physics state.
- `trackSpec` - Frozen track geometry.
- `controllerState` - Mutable controller state (target index advances in place).

Returns: Control signals for the next simulation step.

Example:

```ts
const output = computeScriptedControl(envState, trackSpec, controllerState);
const nextState = stepEnvironment(envState, output);
```

### createScriptedControllerState

```ts
createScriptedControllerState(): ScriptedControllerState
```

Creates a fresh scripted controller state targeting the first segment.

Returns: Fresh controller state with `targetSegmentIndex` at 0.

Example:

```ts
const ctrl = createScriptedControllerState();
const output = computeScriptedControl(envState, spec, ctrl);
```

### ScriptedControllerState

Persistent state for the scripted waypoint-following controller.

The `targetSegmentIndex` advances monotonically (modulo segment count) as the
car reaches successive waypoints around the closed loop.

### wrapAngleToMinusPiPi

```ts
wrapAngleToMinusPiPi(
  angleRadians: number,
): number
```

Wraps an angle in radians to the range [-π, π].

Parameters:
- `angleRadians` - Raw angle in radians (any value).

Returns: Equivalent angle in [-π, π].

## controller/observation.assembler.ts

### appendOwnTireState

```ts
appendOwnTireState(
  baseVector: Float32Array<ArrayBufferLike>,
  tireState: TireStateTuple,
): Float32Array<ArrayBufferLike>
```

Appends the querying car's tire-health tuple to a base observation vector.

Parameters:
- `baseVector` - Base observation vector.
- `tireState` - Four-channel tire-health tuple ordered by wheel corner.

Returns: Observation vector with the tire-health tail appended.

### assembleNormalizedObservationVector

```ts
assembleNormalizedObservationVector(
  envState: RacingObservationState,
  trackSpec: TrackSpec,
  options: ObservationAssemblerOptions,
): Float32Array<ArrayBufferLike>
```

Builds the flat normalized observation vector used by the racing NGE controller.

The Tier 1 base vector contains 70 channels:
1. Twenty scalar channels describing the car and immediate driving context.
2. Forty channels describing five look-ahead track segments.
3. Ten recurrent memory-trace channels.

Tier 2 reuses the Tier 1 base and appends the seven self-radio channels at the
tail without re-normalizing them. Tier 3 reuses the same base and appends three
seven-channel teammate-radio slots. Tier 4 and 5 both keep the 95-channel tail
shape that appends the querying car's four tire channels.

Parameters:
- `envState` - Current environment snapshot plus optional Tier 1–5 fields.
- `trackSpec` - Frozen track geometry used to derive look-ahead features.
- `options` - Tier selector that decides which radio or tire tail is appended.

Returns: Normalized Tier 1 vector (70 channels), Tier 2 vector (77 channels), Tier 3 vector (91 channels), or Tier 4/5 vector (95 channels).

### assembleTier3Observation

```ts
assembleTier3Observation(
  envState: EnvironmentState & ObservationExtensions & { teammateRadioSlots?: readonly (Float32Array<ArrayBufferLike> | readonly number[])[] | undefined; },
  trackSpec: TrackSpec,
): Float32Array<ArrayBufferLike>
```

Builds the Tier 3 observation vector with a fixed teammate-radio extension.

The layout stays stable so controller weights can treat teammate radio rows
as a predictable suffix: `[0..69]` is the Tier 1 driving baseline,
`[70..76]` is teammate slot 0, `[77..83]` is teammate slot 1, and
`[84..90]` is teammate slot 2. `envState.teammateRadioSlots` feeds those
slots directly.

Missing slots are zero-padded. That is the honest Phase 3 2v2 behavior: a
race pack can supply at most two teammate rows, so the unused tail remains
silent rather than fabricating extra agents.

Parameters:
- `envState` - Current environment snapshot plus optional teammate radio rows.
- `trackSpec` - Frozen track geometry used to derive look-ahead features.

Returns: Ninety-one-channel Tier 3 observation vector.

Example:

```ts
const observation = assembleTier3Observation(
  {
    ...envState,
    teammateRadioSlots: [
      Float32Array.from([1, 0, 0, 0, 0, 0, 0]),
      Float32Array.from([0, 1, 0, 0, 0, 0, 0]),
    ],
  },
  trackSpec,
);

observation.length; // 91
observation.slice(84, 91); // zero-padded in 2v2
```

### assembleTier4Observation

```ts
assembleTier4Observation(
  envState: RacingObservationState,
  trackSpec: TrackSpec,
): Float32Array<ArrayBufferLike>
```

Builds the Tier 4 observation vector by appending own-car tire health.

The first 91 channels are byte-for-byte identical to Tier 3. The new suffix
occupies `[91..94]` and stores `[frontLeft, frontRight, rearLeft, rearRight]`.
That keeps every pre-existing Tier 3 feature aligned while exposing only the
querying car's four tire channels as the new Tier 4 sensory delta.

Parameters:
- `envState` - Current environment snapshot plus optional Tier 4 tire state.
- `trackSpec` - Frozen track geometry used to derive look-ahead features.

Returns: Ninety-five-channel Tier 4 observation vector.

Example:

```ts
const observation = assembleTier4Observation(
  { ...envState, tireState: [1, 0.9, 0.8, 0.7] },
  trackSpec,
);

observation.length; // 95
observation.slice(91, 95); // Float32Array [1, 0.9, 0.8, 0.7]
```

### assembleTier5Observation

```ts
assembleTier5Observation(
  envState: RacingObservationState,
  trackSpec: TrackSpec,
): Float32Array<ArrayBufferLike>
```

Builds the Tier 5 observation vector with the canonical 95-channel 3v3 layout.

Channel layout:
- `[0..69]` — 70-channel base observation containing pose, speed, track geometry,
  and recurrent memory trace.
- `[70..90]` — team radio (`3 × 7 = 21` channels). In 3v3 all three rows can be
  populated; smaller packs keep any missing row zero-padded.
- `[91..94]` — own-car tire state `[frontLeft, frontRight, rearLeft, rearRight]`.

Tier 5 is byte-stable with Tier 4: both tiers emit the same 95 floats in the
same order. The difference is radio population, not vector shape.

Parameters:
- `envState` - Current environment snapshot plus optional Tier 5 teammate radio rows and tire state.
- `trackSpec` - Frozen track geometry used to derive look-ahead features.

Returns: Ninety-five-channel Tier 5 observation vector with the stable Tier 4 byte layout.

Example:

```ts
const observation = assembleTier5Observation(
  {
    ...envState,
    teammateRadioSlots: [
      Float32Array.from([1, 0, 0, 0, 0, 0, 0]),
      Float32Array.from([0, 1, 0, 0, 0, 0, 0]),
      Float32Array.from([0, 0, 1, 0, 0, 0, 0]),
    ],
    tireState: [1, 0.9, 0.8, 0.7],
  },
  trackSpec,
);

observation.length; // TIER_ONE_CHANNEL_COUNT + TIER_THREE_TEAMMATE_RADIO_CHANNEL_COUNT + TIRE_CHANNEL_COUNT
observation.slice(
  TIER_ONE_CHANNEL_COUNT,
  TIER_ONE_CHANNEL_COUNT + TIER_THREE_TEAMMATE_RADIO_CHANNEL_COUNT,
); // three teammate radio rows
observation.slice(
  TIER_ONE_CHANNEL_COUNT + TIER_THREE_TEAMMATE_RADIO_CHANNEL_COUNT,
); // own-car tire channels
```

### clamp01

```ts
clamp01(
  value: number,
): number
```

Clamps a probability-like scalar to [0, 1].

Parameters:
- `value` - Source scalar.

Returns: Value clamped to [0, 1].

### clampNormalizedValue

```ts
clampNormalizedValue(
  value: number,
): number
```

Clamps any scalar into the controller's accepted normalized range.

Parameters:
- `value` - Source scalar.

Returns: Value clamped to [-1, 1].

### collectLookAheadChannels

```ts
collectLookAheadChannels(
  observationContext: ObservationContext,
): Float32Array<ArrayBufferLike>
```

Collects the 40 look-ahead segment channels used for track anticipation.

Parameters:
- `observationContext` - Resolved controller context.

Returns: Forty normalized channels describing five upcoming segments.

### collectMemoryTraceChannels

```ts
collectMemoryTraceChannels(
  observationContext: ObservationContext,
): Float32Array<ArrayBufferLike>
```

Collects the 10-channel recurrent memory trace.

Parameters:
- `observationContext` - Resolved controller context.

Returns: Ten normalized recurrent channels.

### collectScalarChannels

```ts
collectScalarChannels(
  observationContext: ObservationContext,
): Float32Array<ArrayBufferLike>
```

Collects the 20 scalar channels that describe the current car state.

Parameters:
- `observationContext` - Resolved controller context.

Returns: Twenty normalized scalar channels.

### createObservationContext

```ts
createObservationContext(
  envState: RacingObservationState,
  trackSpec: TrackSpec,
): ObservationContext
```

Creates the resolved controller context used by the assembler helpers.

Parameters:
- `envState` - Current environment snapshot.
- `trackSpec` - Frozen track geometry.

Returns: Stable context containing derived state and the closest segment index.

### createTier3ObservationOptions

```ts
createTier3ObservationOptions(): { readonly tier: 3; }
```

Creates the reusable `{ tier: 3 }` selector for the 91-channel observation layout.

Use this helper when a caller wants the teammate-aware Tier 3 vector without
re-allocating the options object by hand.

Returns: Immutable Tier 3 observation options object.

### createTier4ObservationOptions

```ts
createTier4ObservationOptions(): { readonly tier: 4; }
```

Creates the reusable `{ tier: 4 }` selector for the 95-channel observation layout.

Use this helper when callers need the canonical Tier 4 pack shape without
re-allocating the options object by hand.

Returns: Immutable Tier 4 observation options object.

### createTier5ObservationOptions

```ts
createTier5ObservationOptions(): { readonly tier: 5; }
```

Creates the reusable `{ tier: 5 }` selector for the 95-channel Tier 5 layout.

Pass this helper to `assembleNormalizedObservationVector` when the caller wants
the byte-stable Tier 4/5 observation shape while allowing all three teammate-radio
rows to be populated in a six-car 3v3 pack.

Returns: Immutable Tier 5 observation options object.

Example:

```ts
const options = createTier5ObservationOptions();
const observation = assembleNormalizedObservationVector(envState, trackSpec, options);

options.tier; // 5
observation.length; // TIER_ONE_CHANNEL_COUNT + TIER_THREE_TEAMMATE_RADIO_CHANNEL_COUNT + TIRE_CHANNEL_COUNT
```

### findClosestSplineSampleIndex

```ts
findClosestSplineSampleIndex(
  envState: RacingObservationState,
  trackSpec: TrackSpec,
): number
```

Finds the shared spline sample whose lane-center point is closest to the car.

Parameters:
- `envState` - Current environment snapshot.
- `trackSpec` - Frozen track geometry.

Returns: Index of the nearest spline sample.

### normalizeProbability

```ts
normalizeProbability(
  probabilityValue: number,
): number
```

Converts a probability-style scalar in [0, 1] to the controller's symmetric
[-1, 1] range.

Parameters:
- `probabilityValue` - Probability-like scalar.

Returns: Symmetric normalized value.

### normalizeSignedValue

```ts
normalizeSignedValue(
  value: number,
  scale: number,
): number
```

Converts a signed scalar into the normalized [-1, 1] range.

Parameters:
- `value` - Signed source scalar.
- `scale` - Absolute scale corresponding to magnitude 1.

Returns: Clamped normalized value.

### normalizeUnsignedValue

```ts
normalizeUnsignedValue(
  value: number,
  scale: number,
): number
```

Converts an unsigned scalar into the normalized [0, 1] range while keeping the
result inside the controller's accepted bounds.

Parameters:
- `value` - Unsigned source scalar.
- `scale` - Maximum reference scale for value 1.

Returns: Normalized value in [0, 1].

### ObservationAssemblerOptions

Options that select which suffix is appended to the 70-channel driving base.

`tier` decides whether callers receive only the base observation, the seven-channel
self-radio tail, the 21-channel teammate-radio tail, or the four-channel own-tire
suffix that keeps Tier 4 and Tier 5 byte-stable at 95 channels.

### ObservationExtensions

Additional Tier 1–3 sensory fields layered on top of the base environment state.

### ObservationTier

Tier selector for the owner-local observation seam.

- `1` — 70-channel base vector (20 scalar + 40 look-ahead + 10 memory-trace).
- `2` — Tier 1 plus the seven-channel self-radio tail for a 77-channel vector.
- `3` — Tier 1 plus three teammate-radio slots for a 91-channel vector; smaller
  packs keep any missing slot zero-padded.
- `4` — Tier 3 plus own-car tire health at `[91..94]` ordered as
  `[frontLeft, frontRight, rearLeft, rearRight]`, producing 95 channels.
- `5` — The same 95-channel byte layout as Tier 4, but the 3v3 six-car seam can
  fully populate all three teammate-radio rows before the tire tail is appended.

### RacingObservationState

Concrete environment shape consumed by the Tier 1–3 controller seam.

### resolveBoundaryBalance

```ts
resolveBoundaryBalance(
  boundaryDistanceLeftWorld: number,
  boundaryDistanceRightWorld: number,
): number
```

Computes the signed left-versus-right balance used by the controller to judge
how centered the car is within the lane.

Parameters:
- `boundaryDistanceLeftWorld` - Distance from the car to the left boundary.
- `boundaryDistanceRightWorld` - Distance from the car to the right boundary.

Returns: Symmetric balance in [-1, 1].

### resolveLookAheadSplineSample

```ts
resolveLookAheadSplineSample(
  trackSpec: TrackSpec,
  closestSplineSampleIndex: number,
  lookAheadOffset: number,
): SplineSample
```

Resolves one look-ahead spline sample by wrapping around the closed loop.

Parameters:
- `trackSpec` - Frozen track geometry.
- `closestSplineSampleIndex` - Nearest spline sample index for the car.
- `lookAheadOffset` - Positive offset from the nearest segment.

Returns: Spline sample to encode into the observation vector.

### resolveMemoryTrace

```ts
resolveMemoryTrace(
  memoryTrace: readonly number[] | undefined,
): readonly number[]
```

Resolves the recurrent memory trace while preserving a fixed width.

Parameters:
- `memoryTrace` - Incoming recurrent trace values, if any.

Returns: Ten-value trace ready for normalization.

### resolveObservationState

```ts
resolveObservationState(
  envState: RacingObservationState,
  trackSpec: TrackSpec,
  closestSplineSampleIndex: number,
): ResolvedObservationState
```

Resolves all optional Tier 1–2 observation fields, deriving safe defaults when
the live environment has not produced them yet.

Parameters:
- `envState` - Current environment snapshot.
- `trackSpec` - Frozen track geometry.
- `closestSegmentIndex` - Segment nearest to the car position.

Returns: Complete observation state ready for normalization.

### resolveRadioTail

```ts
resolveRadioTail(
  radioField: Float32Array<ArrayBufferLike>,
): Float32Array<ArrayBufferLike>
```

Resolves the raw Tier 2 radio tail without applying any additional normalization.

Parameters:
- `radioField` - Current self-radio buffer.

Returns: Seven-channel radio tail in the original order.

### resolveTrackBounds

```ts
resolveTrackBounds(
  trackSpec: TrackSpec,
): TrackBounds
```

Resolves the track bounds used to normalize car position channels.

Parameters:
- `trackSpec` - Frozen track geometry.

Returns: Bounding box plus center point.

### wrapAngleToMinusPiPi

```ts
wrapAngleToMinusPiPi(
  angleRadians: number,
): number
```

Wraps an angle to the closed interval [-π, π].

Parameters:
- `angleRadians` - Raw angle in radians.

Returns: Wrapped angle in [-π, π].

## controller/pathtracking.test.fixtures.ts

### BOUNDARY_DISTANCE_WORLD_SCALE

Shared boundary-distance scale used by observation-vector lane channels.

### buildSplineSamples

```ts
buildSplineSamples(
  trackSpec: TrackSpec,
): readonly SplineSample[]
```

Builds the sampled Catmull-Rom centerline that the renderer currently draws.

Parameters:
- `trackSpec` - Frozen track geometry.

Returns: Ordered spline samples with segment ownership metadata.

### createCurvedTrackSpec

```ts
createCurvedTrackSpec(): TrackSpec
```

Builds a deterministic curved track whose rendered spline diverges materially
from the raw polygon chord geometry.

Returns: Frozen owner-local `TrackSpec` used by the red seam contracts.

### createEnvironmentState

```ts
createEnvironmentState(
  overrides: Partial<EnvironmentState>,
): EnvironmentState
```

Creates a minimal base environment state for owner-local controller tests.

Parameters:
- `overrides` - Environment fields to override for a specific scenario.

Returns: Deterministic environment state for the red tests.

### DISTANCE_WORLD_SCALE

Shared distance scale used by observation-vector distance channels.

### FocalSampleSelection

Shared result shape returned by the focal-sample selectors.

### ObservationProbe

Probe position placed slightly toward the left boundary of a spline sample.

### resolveOffsetObservationProbe

```ts
resolveOffsetObservationProbe(
  splineSamples: readonly SplineSample[],
  focalSampleIndex: number,
): ObservationProbe
```

Places the observation probe slightly toward the left boundary of one spline sample.

Parameters:
- `splineSamples` - Ordered sampled centerline points.
- `focalSampleIndex` - Global index of the selected spline sample.

Returns: Probe position plus the exact lateral offset from the lane center.

### resolveSampleFrame

```ts
resolveSampleFrame(
  splineSamples: readonly SplineSample[],
  focalSampleIndex: number,
): SplineSampleFrame
```

Resolves the local tangent frame for one sampled spline point.

Parameters:
- `splineSamples` - Ordered sampled centerline points.
- `focalSampleIndex` - Global index of the sample to inspect.

Returns: Tangent heading plus the unit left normal.

### selectControllerFocalSample

```ts
selectControllerFocalSample(
  trackSpec: TrackSpec,
): FocalSampleSelection
```

Selects the owner-local spline sample with the strongest endpoint-vs-tangent gap.

Parameters:
- `trackSpec` - Frozen curved-track fixture.

Returns: The strongest controller seam sample plus its tangent frame.

### selectObservationFocalSample

```ts
selectObservationFocalSample(
  trackSpec: TrackSpec,
): FocalSampleSelection
```

Selects the owner-local spline sample with the strongest chord-midpoint gap.

Parameters:
- `trackSpec` - Frozen curved-track fixture.

Returns: The strongest observation seam sample plus its tangent frame.

### TestSampleFrame

Local tangent-frame information for one spline sample.

### TestSplineSample

One sampled spline point enriched with segment ownership metadata.

### wrapAngleToMinusPiPi

```ts
wrapAngleToMinusPiPi(
  angleRadians: number,
): number
```

Wraps an angle into the closed interval `[-π, π]`.

Parameters:
- `angleRadians` - Raw angle in radians.

Returns: Wrapped angle in `[-π, π]`.
