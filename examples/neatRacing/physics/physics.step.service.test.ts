/**
 * Contract tests for the Neat Racing physics step service (slice
 * P1-drift-feel-red, Step 14N; AC-023, AC-047B, AC-047D, AC-079,
 * AC-087, spec §5.1-5.3, LD-15, TP-01/02/04).
 *
 * RED PHASE (Step 14N, slice `P1-drift-feel-red`): the tuned service
 * carries the drift channel as ONE NESTED field (`state.drift.committed`
 * plus `state.drift.pose`) and retires the vestigial
 * `NEATRACING_PHYSICS_CENTRIFUGAL_MULTIPLIER`. This file asserts the NEW
 * contracts and the re-pinned green pins of the 14N re-baseline ledger
 * (G-8, G-9, G-10, G-14, G-17) against the CURRENT service.
 *
 * The file MUST COMPILE, so one missing export never masks a sibling
 * contract. BOTH RED-era reads below have since LANDED (the tune-model slice
 * exported `NEATRACING_DIFT_POSE_STEER_YAW_RAD` = 0.08 in drift.model.ts, and the
 * step service now carries the nested `state.drift` channel), so the wide-cast
 * and the local typed view are VESTIGIAL but harmless: the drift-model
 * constant is read through the NAMESPACE (`driftModel.NEATRACING_DIFT_POSE_STEER_YAW_RAD`)
 * instead of a named import, and `state.drift` is read through the local typed
 * view `driftChannelOf`.
 *
 * The SPLIT slip contract (GAP-N4 D1.3): the LANDED export
 * `NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD` (0.45 rad) bounds the COMMITTED
 * slide ONLY (the model clamps it), while the PUBLISHED slip may
 * OVERSHOOT it by the landed `NEATRACING_DRIFT_PUBLISHED_SLIP_OVERSHOOT_RAD`
 * (0.002 rad) because the chase spring lags the committed angle (the
 * measured published peak is 0.45116 rad). The published-slip envelope
 * (G-9, G-10, P0-2) is therefore DERIVED from the TWO named exports
 * (`MAX + OVERSHOOT` = 0.452 rad), never from a literal, and the
 * committed-slide pins are never touched. The 7e derivation contract pins
 * the constant itself (`MAX * GAIN === 0.2025`).
 *
 * The REGIME-AWARE published-slip bounds (R2): the ON-ROAD regime is
 * asserted against the DERIVED named export
 * `NEATRACING_DRIFT_PUBLISHED_SLIP_MAX_RAD` (0.452 rad) and the
 * OFF-ROAD regime against the DECLARED named export
 * `NEATRACING_DRIFT_PUBLISHED_SLIP_OFFROAD_MAX_RAD` (0.493 rad). The
 * off-road drag drives the same chase-lag overshoot from a FALLING
 * denominator, so the off-road published slip peaks higher (the measured
 * post-RC-1 off-road peak is 0.4872167 rad). There is NO publish-site
 * clip: the published slip keeps its physical value and the off-road
 * regime is asserted against its own recorded ceiling. Both bounds are
 * mirrored from the named exports, never from a literal.
 *
 * Determinism rung: same-runtime byte-identical via deterministic
 * serialization; the step service is pure and contains no Math.random
 * state (§5.3).
 *
 * GREEN contract for the implementer: './physics.step.service' exports the
 * named constants NEATRACING_PHYSICS_FIXED_TIMESTEP_MS,
 * NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
 * NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S, NEATRACING_PHYSICS_ACCEL_UNITS_PER_S2,
 * NEATRACING_PHYSICS_BRAKING_UNITS_PER_S2,
 * NEATRACING_PHYSICS_NATURAL_DECEL_UNITS_PER_S2,
 * NEATRACING_PHYSICS_OFF_ROAD_DECEL_UNITS_PER_S2,
 * NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S,
 * NEATRACING_PHYSICS_STEERING_RATE (the vestigial
 * NEATRACING_PHYSICS_CENTRIFUGAL_MULTIPLIER is GONE), the pure step
 * function `stepPhysics(state, input, surfaceResolver?)`, and the factory
 * `createInitialPhysicsCarState(seed?)`. The car state carries the nested
 * `drift` channel (`committed.slip`, `committed.turnSign`,
 * `committed.reArmed`, `pose.pose`, `pose.poseRate`, `pose.articulation`)
 * in place of the retired `driftYawOffset` field. The surface resolver
 * returns a `PhysicsSurfaceType` such as `'road' | 'curb' | 'grass'`; when
 * omitted the service treats the surface as road.
 *
 * STEP 14Q (slice `P1-drift-law-red`; AC-087 / AC-088 / AC-089) ADDS the
 * PUBLISHED traction-scalar contracts to this file (the plan's RED-5):
 * the state must publish the ONE shared traction severity as the named
 * field `tractionSeverity`, derived from the PUBLISHED slip and the
 * POST-traction speed, exactly 0 on a straight line, inside the closed
 * unit interval and MONOTONIC in the slide depth. The MIN_SPEED gate pin
 * below is RE-BASELINED in the same slice (the gate is SOFTENED, never
 * removed: the VISIBLE slide stays exactly zero below the gate while the
 * commitment is no longer frozen).
 *
 * STEP 14R (slice `P1-drift-collision-live`; AC-090 / AC-089) ADDS the
 * LIVE car-to-car collision contracts to this file (the plan's RED-4):
 * the resolver (`resolveCarCollisions` in `./collision.utils`) was
 * imported by NO production module, so a car-to-car overlap was DEAD in
 * the live tick. The contract is written against the LIVE step - the
 * pack tick that advances a whole fleet and publishes the RESOLVED
 * states - and NEVER against the resolver in isolation (the resolver
 * keeps its own unit coverage in `collision.utils.test.ts`). The live
 * tick keeps the LOAD-BEARING ORDER of AC-090: the drift integration
 * runs FIRST for every member, the car-to-car resolution runs AFTER it,
 * and the RESOLVED states are published LAST.
 *
 * The file MUST COMPILE (the rule of this file), so the
 * NOT-YET-EXISTING `stepPhysicsFleet` and `resolveLiveCarCollisions`
 * exports are read through a LOCAL widened view - exactly the
 * `computeTractionSeverityOf` pattern of the 14Q block.
 *
 * The file MUST COMPILE (the rule of this file), so the NOT-YET-EXISTING
 * `computeTractionSeverity` export is read through a LOCAL widened view
 * (`computeTractionSeverityOf`) and the NOT-YET-EXISTING state field
 * through `tractionSeverityOf` - exactly the `travelHeadingRadOf`
 * pattern of the 14P block. The named import of the missing export
 * lives in the MODEL test file (`drift.model.test.ts`), where the
 * artifact-mandated missing-import form applies.
 *
 * The MEASURED current-model baselines of the 14Q contracts live in the
 * plan ledger (they are never asserted here): the state publishes NO
 * traction field at all today, and the MIN_SPEED gate freezes the
 * commitment below 15 m/s (the committed slip and the turn sign are
 * exactly 0 below the gate).
 */

import { describe, expect, it, jest } from '@jest/globals';
import {
  createInitialPhysicsCarState,
  NEATRACING_PHYSICS_FIXED_TIMESTEP_MS,
  NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
  NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
  NEATRACING_PHYSICS_ACCEL_UNITS_PER_S2,
  NEATRACING_PHYSICS_BRAKING_UNITS_PER_S2,
  NEATRACING_PHYSICS_NATURAL_DECEL_UNITS_PER_S2,
  NEATRACING_PHYSICS_OFF_ROAD_DECEL_UNITS_PER_S2,
  NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S,
  NEATRACING_PHYSICS_OFF_TRACK_SPEED_LIMIT_KMH,
  NEATRACING_PHYSICS_STEERING_RATE,
  stepPhysics,
} from './physics.step.service';
import * as physicsStepService from './physics.step.service';
import {
  classifyTrackSurface,
  createTrackSurfaceResolver,
  NEATRACING_CAR_COLLISION_RADIUS_M,
  NEATRACING_TRACK_GRASS_APRON_WIDTH_M,
} from './collision.utils';
import {
  NEATRACING_DRIFT_COUNTERSTEER_MIN_STEER,
  NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD,
  NEATRACING_DRIFT_MIN_SPEED_DEFICIT_KMH,
  NEATRACING_DRIFT_PUBLISHED_SLIP_MAX_RAD,
  NEATRACING_DRIFT_PUBLISHED_SLIP_OFFROAD_MAX_RAD,
  NEATRACING_DRIFT_PUBLISHED_SLIP_OVERSHOOT_RAD,
  NEATRACING_DRIFT_SPEED_SCRUB_COEFF_PER_S,
  NEATRACING_DRIFT_STEER_ENGAGE_THRESHOLD,
  NEATRACING_DRIFT_STEER_NEUTRAL_THRESHOLD,
  NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD,
} from './drift.model';
import * as driftModelModule from './drift.model';
import type { DriftStepInput } from './drift.model';
import type {
  PhysicsCarState,
  PhysicsStepInput,
  PhysicsSurfaceType,
} from './physics.step.service';
import type {
  SplineSample,
  TrackSegment,
} from '../../racing_curriculum/track/track.generator.types';
import { generateTrack } from '../../racing_curriculum/track/track.generator';
import {
  buildTrackRacingLine,
  NEATRACING_RACING_LINE_STRAIGHT_CURVATURE_PER_M,
} from '../track/track.racing-line';
import {
  resolveNearestTrackChord,
  resolveTrackEdgeHalfWidthM,
  resolveTrackEdgeProfile,
  resolveTrackEdgeSideAt,
} from '../track/track.edge';

/** Minimal surface labels used by the fixture resolver. */
type FixtureSurface = 'road' | 'curb' | 'grass';

/**
 * A SURFACE PHASE (R2): the surface the car drives on and the number of
 * 60 Hz ticks it stays there.
 */
type SurfacePhase = readonly [FixtureSurface, number];

/**
 * A SURFACE SCHEDULE (R2): the ordered phases of one lane.
 */
type SurfaceSchedule = readonly SurfacePhase[];

/** Steered, full-throttle input used for the slip/drift/determinism fixtures. */
const FIXTURE_INPUT: PhysicsStepInput = { throttle: 1, brake: 0, steer: 1 };

/** Straight-line full-throttle input: the zero-yaw lane (merge 7b). */
const STRAIGHT_INPUT: PhysicsStepInput = { throttle: 1, brake: 0, steer: 0 };

/** Full-RIGHT countersteer input: the catch phase (AC-047B, P0-1). */
const COUNTER_INPUT: PhysicsStepInput = { throttle: 1, brake: 0, steer: -1 };

/** Low-speed full-steer input: the MIN_SPEED gate lane (D1). */
const LOW_SPEED_STEER_INPUT: PhysicsStepInput = { throttle: 0, brake: 0, steer: 1 };

/**
 * Full-throttle, full-steer input: the DRIFT-ENGAGED lane (RC-1).
 *
 * The steer magnitude (1) is above the engagement threshold (0.5), so the
 * drift model commits a slide and the service scrubs the forward speed.
 *
 * This fixture is DELIBERATELY NOT aliased to the shared `FIXTURE_INPUT`:
 * the RC-1 contract only holds while the steer magnitude stays above the
 * drift engagement threshold, so a future edit of the shared fixture must
 * not be able to silently de-vacuum the contract.
 */
const GRASS_DRIFT_INPUT: PhysicsStepInput = { throttle: 1, brake: 0, steer: 1 };

/**
 * Full-throttle, straight input: the NO-DRIFT control lane (RC-1).
 *
 * Identical to {@link GRASS_DRIFT_INPUT} except for the steer axis, so the
 * A/B pair isolates the drift scrub as the only difference. Kept
 * independent of the shared `STRAIGHT_INPUT` for the same non-vacuity
 * reason as {@link GRASS_DRIFT_INPUT}.
 */
const GRASS_STRAIGHT_INPUT: PhysicsStepInput = {
  throttle: 1,
  brake: 0,
  steer: 0,
};

/**
 * Float tolerance for the off-road floor comparisons (RC-1).
 *
 * The floor is an exact named export (the USER-named off-track limit,
 * 20 km/h / 3.6, the iteration-3 re-derivation), so the tolerance only
 * absorbs the float64 noise of the tick arithmetic - it never widens the
 * contract by a meaningful amount (the trapped speed is exactly 0).
 */
const OFF_ROAD_FLOOR_TOLERANCE = 1e-6;

/**
 * The RC-1 observation window: 600 ticks at 60 Hz (10 s).
 *
 * The window is long enough to cover the WHOLE collapse (the measured
 * trap is reached at tick 165), so a run that ends at 0 is a TRAP and not
 * a transient.
 */
const OFF_ROAD_FLOOR_TICKS = 600;

/**
 * The LOWER edge of the admissible scrub-coefficient band (R5).
 *
 * The 14N determinism review (R5: "the D1.3 deficit floor kills
 * c < 0.504") and the D-MEMO (`decision_d5.rationale`, "the c-band
 * [0.504, 0.626]") ratify 0.504 as the lower edge: below it the
 * D1.3 speed-deficit floor (the never-relaxed 20 km/h) stops
 * holding. NOTE: the exact terminal-speed crossover of that floor is
 * c = 0.505555 (the D1 decision), so the ratified edge is a slightly
 * loose round-down; the shipped coefficient (0.55) clears both by a
 * wide margin.
 */
const SCRUB_COEFF_BAND_LOWER = 0.504;

/**
 * The UPPER edge of the admissible scrub-coefficient band (R5).
 *
 * The derived 0.452 published-slip ceiling is crossed at c ~= 0.626
 * (the review's sweep: c = 0.626 -> 0.4519979733 rad, c = 0.65 ->
 * 0.4522611403 rad = VIOLATED), so 0.626 is the ratified upper edge.
 */
const SCRUB_COEFF_BAND_UPPER = 0.626;

/**
 * The swept OFF-ROAD surface envelope (R2).
 *
 * Every lane is a 300-tick (5 s) schedule that covers one regime of
 * the off-road regime space: the pure grass lane, the road -> grass
 * ENTRY transitions, the grass -> road EXIT transition, a full
 * off-road excursion with a road recovery, a curb phase inside the
 * off-road sweep, and the grass -> road -> grass lane that carries the
 * peak-reachable published slip.
 *
 * The lane set is NOT a cherry-pick: it carries the WORST lane of the
 * measured jest sweep (0.487217 rad) AND the lanes whose peak sits
 * ABOVE the off-road floor (the road -> grass -> road lane peaks at
 * 0.478325 rad at 31.11 m/s), so the off-road ceiling is exercised by
 * every regime and not only by the RC-1 collapse region.
 */
const OFF_ROAD_SLIP_ENVELOPE: readonly SurfaceSchedule[] = [
  [['grass', 300]],
  [['road', 42], ['grass', 258]],
  [['road', 120], ['grass', 180]],
  [['grass', 150], ['road', 150]],
  [['road', 60], ['grass', 90], ['road', 150]],
  [['grass', 100], ['curb', 100], ['grass', 100]],
  [['grass', 80], ['road', 40], ['grass', 180]],
];

/**
 * The ROAD / CURB control lanes (R2).
 *
 * The curb shares the road longitudinal model, so the three control
 * lanes are bit-identical to each other: their published peak is the
 * ratified on-road peak 0.4511617369362305 rad @t42, which clears the
 * derived on-road ceiling by 0.00084 rad. They are the NON-VACUITY
 * witness of the R2 contract: the ON-ROAD ceiling is a real bound that
 * the off-road lanes breach and the on-road lanes never breach.
 */
const ROAD_CURB_SLIP_CONTROL: readonly SurfaceSchedule[] = [
  [['road', 300]],
  [['curb', 300]],
  [['road', 42], ['curb', 258]],
];

/**
 * A car already at the arcade top speed: the drift-build fixture of the
 * P0-1 and P0-2 contracts (the slide is built from the top speed, not
 * from a standing start).
 */
const atTopSpeed = (): PhysicsCarState => ({
  ...createInitialPhysicsCarState(42),
  speed: NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
});

/**
 * The nested drift channel the tuned service will carry (RED view).
 *
 * The `state.drift` field does not exist yet, so every read of the
 * channel goes through this view: the channel is absent today and each
 * nested read yields `undefined`, which is what makes the channel
 * contracts RED for the RIGHT reason (a missing channel) instead of a
 * `TypeError` on `undefined.pose`.
 */
interface RedDriftChannelView {
  readonly committed: {
    readonly slip: number;
    readonly turnSign: number;
    readonly reArmed: boolean;
  };
  readonly pose: {
    readonly pose: number;
    readonly poseRate: number;
    readonly articulation: number;
  };
}

/** The absent pose sub-channel: every field reads as `undefined`. */
const ABSENT_POSE = {} as RedDriftChannelView['pose'];

/** The absent committed sub-channel: every field reads as `undefined`. */
const ABSENT_COMMITTED = {} as RedDriftChannelView['committed'];

/**
 * Reads the nested drift channel of a car state through the RED view.
 *
 * @param state - The car state to read.
 * @returns The drift channel, or an all-`undefined` stand-in while the
 *   service still carries the flat `driftYawOffset` field.
 */
const driftChannelOf = (state: PhysicsCarState): RedDriftChannelView => {
  const carrier = state as unknown as { drift?: Partial<RedDriftChannelView> };
  return {
    committed: carrier.drift?.committed ?? ABSENT_COMMITTED,
    pose: carrier.drift?.pose ?? ABSENT_POSE,
  };
};

/**
 * The drift-model namespace widened to a loose record, so the
 * NOT-YET-EXISTING export can be READ today: it is `undefined` today,
 * which is what makes the envelope-derivation contracts RED. A named
 * import of a missing export would be a file-level compile error that
 * masks every other contract in this file.
 */
const driftModel = driftModelModule as unknown as Record<string, number>;

/**
 * Runs the step service for a fixed number of 60 Hz ticks from a seeded
 * initial state and a constant surface.
 */
const runSteps = (
  steps: number,
  seed = 42,
  input = FIXTURE_INPUT,
  surface: FixtureSurface = 'road',
): PhysicsCarState => {
  let state = createInitialPhysicsCarState(seed);
  const surfaceResolver = (): FixtureSurface => surface;
  for (let step = 0; step < steps; step++) {
    state = stepPhysics(state, input, surfaceResolver);
  }
  return state;
};

/** Deterministic JSON serialization for byte-identity comparisons. */
const serializeState = (state: PhysicsCarState): string =>
  JSON.stringify(state);

/**
 * Runs the step service for a fixed number of 60 Hz ticks from an explicit
 * start state (the AC-079 fixtures start from a known speed).
 */
const runStepsFrom = (
  start: PhysicsCarState,
  steps: number,
  input = FIXTURE_INPUT,
  surface: FixtureSurface = 'road',
): PhysicsCarState => {
  let state = start;
  const surfaceResolver = (): FixtureSurface => surface;
  for (let step = 0; step < steps; step++) {
    state = stepPhysics(state, input, surfaceResolver);
  }
  return state;
};

/**
 * Runs the step service on a fixed surface and returns the per-tick speed
 * samples (one entry per COMPLETED tick, in tick order).
 *
 * The RC-1 off-road floor contract is a TRAJECTORY contract (the car
 * collapses on the way down, not only at the end), so this helper records
 * every tick-end speed instead of only the final state.
 *
 * @param start - The explicit start state (the lane always starts from a
 *   known speed).
 * @param steps - The number of 60 Hz ticks to run.
 * @param input - The constant control sample applied on every tick.
 * @param surface - The constant surface the resolver reports.
 * @returns The tick-end speed samples, in tick order.
 */
const runSpeedSamples = (
  start: PhysicsCarState,
  steps: number,
  input: PhysicsStepInput,
  surface: FixtureSurface,
): number[] => {
  let state = start;
  const surfaceResolver = (): FixtureSurface => surface;
  const samples: number[] = [];
  for (let step = 0; step < steps; step++) {
    state = stepPhysics(state, input, surfaceResolver);
    samples.push(state.speed);
  }
  return samples;
};

/**
 * Runs the full-RIGHT countersteer phase from an explicit start state and
 * reports the largest slip angle observed AFTER the countersteer onset.
 *
 * The anti-mirror probe: a slide that MIRRORS crosses to the opposite
 * sign, so the maximum slip after the onset must stay at or below zero.
 */
const runCountersteer = (
  start: PhysicsCarState,
  steps: number,
): { state: PhysicsCarState; maxSlipAfterCounterOnset: number } => {
  const surfaceResolver = (): FixtureSurface => 'road';
  let state = start;
  let maxSlipAfterCounterOnset = Number.NEGATIVE_INFINITY;
  for (let step = 0; step < steps; step++) {
    state = stepPhysics(state, COUNTER_INPUT, surfaceResolver);
    maxSlipAfterCounterOnset = Math.max(maxSlipAfterCounterOnset, state.slipAngle);
  }
  return { state, maxSlipAfterCounterOnset };
};

describe('physics step service constants (spec §5.2, AC-023)', () => {
  // [RED-PRESERVE-FIXED-TIMESTEP]
  it('exposes a deterministic 60 Hz fixed timestep', () => {
    expect(NEATRACING_PHYSICS_FIXED_TIMESTEP_MS).toBeCloseTo(1000 / 60, 10);
    expect(NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS).toBeCloseTo(1 / 60, 12);
  });

  // [RED-PRESERVE-TOP-SPEED]
  it('targets the ~89 units/s arcade top-speed feel (TP-01)', () => {
    expect(NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S).toBe(89);
  });

  // [RED-PRESERVE-LONGITUDINAL-CONSTANTS]
  it('derives accel, braking, and natural decel from maxSpeed (§5.2)', () => {
    expect(NEATRACING_PHYSICS_ACCEL_UNITS_PER_S2).toBe(
      NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S / 5,
    );
    expect(NEATRACING_PHYSICS_BRAKING_UNITS_PER_S2).toBe(
      -NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
    );
    expect(NEATRACING_PHYSICS_NATURAL_DECEL_UNITS_PER_S2).toBe(
      -NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S / 5,
    );
  });

  // [RED-PRESERVE-OFF-ROAD-CONSTANTS]
  it('derives the off-road decel from maxSpeed and the off-road speed cap from the USER-named 20 km/h limit (§5.2, AC-090 VERDICT 8)', () => {
    expect(NEATRACING_PHYSICS_OFF_ROAD_DECEL_UNITS_PER_S2).toBe(
      -NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S / 2,
    );
    // THE RE-AUTHOR (fix-packet-P1-sand-red-iteration-3): the off-track
    // FLOOR part of AC-094(c) was USER-SUPERSEDED - the floor constant
    // re-derives as the USER-named off-track speed cap (20 km/h / 3.6
    // m/s, the named verbatim USER constant), never the retired
    // `maxSpeed / 4` = 22.25 m/s floor that measured 4.005x the USER
    // value; the off-road decel law itself is UNCHANGED.
    expect(NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S).toBe(
      NEATRACING_PHYSICS_OFF_TRACK_SPEED_LIMIT_KMH / 3.6,
    );
  });

  // [RED-GONE-CENTRIFUGAL]
  it('retires the vestigial centrifugal multiplier (TP-02)', () => {
    // The multiplier was read by the retired `computeOutwardPush` helper;
    // with the helper gone the export must be gone too (14N section 7).
    expect(
      'NEATRACING_PHYSICS_CENTRIFUGAL_MULTIPLIER' in physicsStepService,
    ).toBe(false);
  });

  // [RED-STEERING-RATE]
  it('exposes the steering rate (TP-04)', () => {
    expect(NEATRACING_PHYSICS_STEERING_RATE).toBe(2);
  });
});

describe('physics step service determinism (spec §5.3)', () => {
  // [RED-PRESERVE-DETERMINISM]
  it('produces identical car state for the same seed and inputs across runs', () => {
    const runA = runSteps(60, 42, FIXTURE_INPUT, 'road');
    const runB = runSteps(60, 42, FIXTURE_INPUT, 'road');

    expect(serializeState(runA)).toBe(serializeState(runB));
  });
});

describe('physics step service slip and drift envelopes (LD-15)', () => {
  // [RED-G9-SLIP-ENVELOPE]
  it('enters a bounded non-zero slip angle under sustained steering', () => {
    // The sustained high-speed slide: the bound is only meaningful on a
    // slide that actually reaches the envelope, so the fixture builds it
    // from the top speed. The 180-tick capture now sits at 0.45042 rad
    // (the OLD model reached ~1.09 rad there, which is why the published
    // slip used to be unbounded).
    const state = runStepsFrom(atTopSpeed(), 180, FIXTURE_INPUT, 'road');

    expect(Math.abs(state.slipAngle)).toBeGreaterThan(0);
    // Re-pinned (G-9, GAP-N4 D1.3): the published slip is NOT bounded by
    // the committed budget alone. The 0.45 rad budget
    // (`NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD`) clamps the COMMITTED slide
    // inside the model, but the PUBLISHED slip follows the committed angle
    // through the chase spring, which LAGS it - so the published slip may
    // overshoot the committed budget by the landed
    // `NEATRACING_DRIFT_PUBLISHED_SLIP_OVERSHOOT_RAD` (0.002 rad). The
    // envelope is therefore DERIVED from the TWO named exports
    // (`MAX + OVERSHOOT` = 0.452 rad) and never from a literal.
    //
    // The CALIBRATION ENVELOPE: the published peak is 0.45116 rad (the
    // chase overshoot at the slide onset), so the derived bound clears it
    // by 0.00084 rad. MEASURED on the final bytes: the 180-tick capture
    // sits at 0.45041978773870417 rad and the run peak is
    // 0.4511617369362305 rad at t42.
    expect(Math.abs(state.slipAngle)).toBeLessThanOrEqual(
      NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD +
        NEATRACING_DRIFT_PUBLISHED_SLIP_OVERSHOOT_RAD,
    );
  });

  // [RED-PRESERVE-YAW-RATE-BOUND]
  it('enters a bounded non-zero yaw rate under sustained steering and throttle', () => {
    const state = runSteps(60, 42, FIXTURE_INPUT, 'road');

    expect(Math.abs(state.yawRate)).toBeGreaterThan(0);
    expect(Math.abs(state.yawRate)).toBeLessThan(1);
    expect(Number.isFinite(state.yawRate)).toBe(true);
  });

  // [RED-G8-ENVELOPE]
  it('keeps the visible yaw inside the derived envelope under sustained steering', () => {
    const state = runStepsFrom(atTopSpeed(), 180, FIXTURE_INPUT, 'road');

    // Re-pinned (G-8): the envelope is DERIVED from the constants - the
    // slip-only bound plus the steer-yaw articulation term - and it is
    // never a magic number. The pose is NOT clamped to the slip-only
    // bound.
    expect(Math.abs(driftChannelOf(state).pose.pose)).toBeLessThanOrEqual(
      NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD * NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD +
        driftModel.NEATRACING_DRIFT_POSE_STEER_YAW_RAD,
    );
  });

  // [RED-POSE-FINITE]
  it('carries a finite pose value under sustained steering', () => {
    const state = runStepsFrom(atTopSpeed(), 180, FIXTURE_INPUT, 'road');

    // The pose channel must carry a real number: the nested
    // `state.drift.pose` field does not exist yet, so the read is
    // `undefined` today.
    expect(Number.isFinite(driftChannelOf(state).pose.pose)).toBe(true);
  });

  // [RED-POSE-NONZERO]
  it('produces a non-zero finite pose under sustained steering', () => {
    const state = runStepsFrom(atTopSpeed(), 180, FIXTURE_INPUT, 'road');

    expect(Math.abs(driftChannelOf(state).pose.pose)).toBeGreaterThan(0);
    expect(Number.isFinite(driftChannelOf(state).pose.pose)).toBe(true);
  });

  // [RED-PRESERVE-OFF-ROAD-SPEED-FLOOR]
  it('enforces the off-road speed floor on grass (§5.2)', () => {
    const startSpeed = NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S;
    let state = createInitialPhysicsCarState(42);
    state = { ...state, speed: startSpeed };

    const surfaceResolver = (): FixtureSurface => 'grass';
    for (let step = 0; step < 120; step++) {
      state = stepPhysics(
        state,
        { throttle: 0, brake: 0, steer: 0 },
        surfaceResolver,
      );
    }

    expect(state.speed).toBeGreaterThanOrEqual(
      NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S - 1e-6,
    );
    expect(state.speed).toBeLessThan(startSpeed);
  });
});

describe('physics step service drift channel (AC-087, nested DriftChannelState)', () => {
  // [RED-D1-SPAWN-ZEROS]
  it('spawns with a zeroed drift channel', () => {
    const spawn = createInitialPhysicsCarState(42);

    expect(driftChannelOf(spawn).pose.pose).toBe(0);
    expect(driftChannelOf(spawn).committed.slip).toBe(0);
  });

  // [RED-7B-INTEGRATION]
  it('AC-087: the whole physics step keeps a zero straight-line yaw (merge 7b)', () => {
    const state = runSteps(600, 42, STRAIGHT_INPUT, 'road');

    // The EXACT identity, not a tolerance: the slip and the lateral
    // velocity are exactly zero on a straight line...
    expect(state.slipAngle).toBe(0);
    expect(state.lateralVelocity).toBe(0);
    // ...and so is the visible pose the new channel carries.
    expect(driftChannelOf(state).pose.pose).toBe(0);
  });

  // [RED-7B-CHANNEL-ZEROS]
  it('AC-087: the straight-line pose spring stays exactly zeroed (merge 7b)', () => {
    const state = runSteps(600, 42, STRAIGHT_INPUT, 'road');

    expect(driftChannelOf(state).pose.poseRate).toBe(0);
    expect(driftChannelOf(state).pose.articulation).toBe(0);
  });

  // [RED-7B-COMMITTED-ZEROS]
  it('AC-087: the straight-line commitment stays exactly zeroed (merge 7b)', () => {
    const state = runSteps(600, 42, STRAIGHT_INPUT, 'road');

    expect(driftChannelOf(state).committed.slip).toBe(0);
    expect(driftChannelOf(state).committed.turnSign).toBe(0);
    // RE-DERIVATION (fix-packet-P1-drift-feel-tune-state-iteration-1,
    // OBS-1 / GAP-N1): the re-arm TOKEN is not a drift readout. A-4 says
    // the token starts TRUE and is cleared ONLY on a RELEASE (the hard
    // zero or the speed-gate close). A pure straight line never releases,
    // so the ratified behavior IS `reArmed === true` here: the token
    // present means the car stays ARMED for a fresh engagement. The old
    // `false` pin asserted a release that never happened - an authoring
    // defect, not a model defect (the model token contracts in
    // drift.model.test.ts PASS with the TRUE-default semantics).
    expect(driftChannelOf(state).committed.reArmed).toBe(true);
  });

  // [RED-7E-ENVELOPE-DERIVATION]
  it('AC-087: the slip-only envelope is derived from the constants (merge 7e)', () => {
    // The envelope is DERIVED from the constants - it is not a magic
    // number. The re-pinned MAX_SLIP_ANGLE_RAD (0.45) times the visual
    // gain (0.45) is 0.2025 rad.
    expect(
      NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD * NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD,
    ).toBeCloseTo(0.2025, 10);
  });

  // [RED-7E-ENVELOPE-DERIVATION-POSE]
  it('AC-087: the declared pose envelope adds the steer-yaw term (merge 7e)', () => {
    // The DECLARED pose envelope is the slip-only bound PLUS the steer-yaw
    // articulation term (0.2025 + 0.0800 = 0.2825 rad): the pose is NOT
    // clamped to the slip-only bound.
    expect(
      NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD * NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD +
        driftModel.NEATRACING_DRIFT_POSE_STEER_YAW_RAD,
    ).toBeCloseTo(0.2825, 10);
  });

  // [RED-G17-AC047D]
  it('AC-047D: the straight-line drift channel is byte-identical across runs and zeroed', () => {
    const runA = runSteps(600, 42, STRAIGHT_INPUT, 'road');
    const runB = runSteps(600, 42, STRAIGHT_INPUT, 'road');

    // The determinism twin, anchored on the PRODUCTION serializer.
    expect(serializeState(runA)).toBe(serializeState(runB));
    // The new channel fields are all exactly zero on a straight line, so
    // the byte-identity survives the new state shape.
    expect(driftChannelOf(runA).pose.pose).toBe(0);
    expect(driftChannelOf(runA).committed.slip).toBe(0);
  });
});

describe('physics step service countersteer catch (AC-047B, P0-1)', () => {
  // [RED-P0-1-BUILD-SLIDE]
  it('builds a real LEFT slide from the top speed', () => {
    const drifted = runStepsFrom(atTopSpeed(), 180, FIXTURE_INPUT, 'road');

    // LEFT is negative in the production sign convention.
    expect(Math.abs(drifted.slipAngle)).toBeGreaterThan(0.4);
    expect(drifted.slipAngle).toBeLessThan(0);
  });

  // [RED-P0-1-HARD-ZERO]
  it('AC-047B: the countersteer unwinds the slide to the hard zero', () => {
    const drifted = runStepsFrom(atTopSpeed(), 180, FIXTURE_INPUT, 'road');
    const recovered = runStepsFrom(drifted, 20, COUNTER_INPUT, 'road');

    // THE HARD ZERO: the slide is snapped to exactly zero, not parked in
    // a mirror-image equilibrium of the same size.
    expect(recovered.slipAngle).toBe(0);
    expect(recovered.lateralVelocity).toBe(0);
  });

  // [RED-P0-1-ANTI-MIRROR]
  it('AC-047B: the countersteer never mirrors the slide to the opposite sign', () => {
    const drifted = runStepsFrom(atTopSpeed(), 180, FIXTURE_INPUT, 'road');
    const { maxSlipAfterCounterOnset } = runCountersteer(drifted, 120);

    // At no point during the unwind did the slip cross to the opposite
    // sign.
    expect(maxSlipAfterCounterOnset).toBeLessThanOrEqual(0);
  });
});

describe('physics step service longitudinal motion on road (AC-023)', () => {
  // [RED-PRESERVE-ACCELERATION]
  it('accelerates forward from a standing start under full throttle', () => {
    const ticks = 1;
    const input: PhysicsStepInput = { throttle: 1, brake: 0, steer: 0 };
    let state = createInitialPhysicsCarState(42);
    const spawnSpeed = state.speed;

    for (let step = 0; step < ticks; step++) {
      state = stepPhysics(state, input);
    }

    expect(state.speed).toBeGreaterThan(spawnSpeed);
  });

  // [RED-PRESERVE-FORWARD-TRAVEL]
  it('moves forward in x under full throttle with no steering', () => {
    const ticks = 10;
    const input: PhysicsStepInput = { throttle: 1, brake: 0, steer: 0 };
    let state = createInitialPhysicsCarState(42);
    const spawnX = state.x;

    for (let step = 0; step < ticks; step++) {
      state = stepPhysics(state, input);
    }

    expect(state.x).toBeGreaterThan(spawnX);
  });
});

describe('physics step service proportional traction loss (AC-079)', () => {
  // [RED-PRESERVE-SPEED-SCRUB]
  it('AC-079: a sustained full-speed turn scrubs the forward speed below the top speed', () => {
    // Re-pinned (GAP-N4 D1.3): the scrub is a SPEED DEFICIT against the
    // straight-line top speed, expressed in km/h. The OLD pin was
    // UNIT-MIXED (a m/s threshold compared against a km/h constant) and
    // is replaced by the ratified deficit shape.
    //
    // MEASURED on the final bytes (c = 0.55): the straight 600-tick
    // capture is 89.000 m/s (the arcade top speed) and the drift 600-tick
    // capture is 79.95702003056027 m/s, so the deficit is
    // (89.000 - 79.95702) * 3.6 = 32.55473 km/h.
    const straight = runSteps(600, 42, STRAIGHT_INPUT, 'road');
    const drift = runStepsFrom(atTopSpeed(), 600, FIXTURE_INPUT, 'road');
    const deficitKmh = (straight.speed - drift.speed) * 3.6;

    // The floor is the NEVER-RELAXED named export (20 km/h), never a
    // literal. The measured 32.55473 km/h clears it by 63 percent
    // (the ratified band was 28-46 km/h).
    expect(deficitKmh).toBeGreaterThanOrEqual(
      NEATRACING_DRIFT_MIN_SPEED_DEFICIT_KMH,
    );
    // The scrub is strictly positive: the slide is slower than the
    // straight line at the same tick count.
    expect(drift.speed).toBeLessThan(straight.speed);
    // The straight-line run still reaches the arcade top speed, so the
    // deficit is the SCRUB and not a slower top speed.
    expect(straight.speed).toBe(NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S);
  });

  // [RED-PRESERVE-SPEED-SCRUB-TERMINAL]
  it('AC-079: the sustained turn keeps scrubbing the forward speed toward its terminal equilibrium', () => {
    const straight = runSteps(600, 42, STRAIGHT_INPUT, 'road');
    const terminal = runStepsFrom(atTopSpeed(), 1200, FIXTURE_INPUT, 'road');

    // The 1200-tick terminal speed is STRICTLY below the straight-line
    // speed: the scrub does not wash out over time. MEASURED terminal:
    // 79.85792723069594 m/s (a 32.91146 km/h deficit against the
    // 89.000 m/s straight line, i.e. 287.49 km/h).
    expect(terminal.speed).toBeLessThan(straight.speed);
    // The scrub is bounded: the car still MOVES at the terminal speed.
    expect(terminal.speed).toBeGreaterThan(0);
  });

  // [RED-G10-BOUNDED-SLIDE]
  it('AC-079: the sustained high-speed slide stays bounded', () => {
    const state = runStepsFrom(atTopSpeed(), 1200);

    // Re-pinned (G-10, GAP-N4 D1.3): the bound is the PUBLISHED-slip
    // envelope, DERIVED from the two named exports: the 0.45 rad
    // committed budget (`NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD`) plus the
    // landed overshoot allowance
    // (`NEATRACING_DRIFT_PUBLISHED_SLIP_OVERSHOOT_RAD` = 0.002 rad). The
    // published slip follows the committed angle through the chase
    // spring, which LAGS it, so it may sit above the committed budget
    // for a few ticks. The old text was green only because the constant
    // was pi/2 - the re-pin is what turned this pin RED.
    //
    // The CALIBRATION ENVELOPE: the published peak is 0.45116 rad, so the
    // derived 0.452 rad bound clears it by 0.00084 rad. MEASURED on the
    // final bytes: the 1200-tick capture is 0.4500001875611939 rad and
    // the run peak is 0.4511617369362305 rad at t42.
    expect(Math.abs(state.slipAngle)).toBeLessThanOrEqual(
      NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD +
        NEATRACING_DRIFT_PUBLISHED_SLIP_OVERSHOOT_RAD,
    );
    expect(Math.abs(state.slipAngle)).toBeGreaterThan(0.2);
    expect(Number.isFinite(state.lateralVelocity)).toBe(true);
  });

  // [RED-P0-2-BOUNDED-SLIDE]
  it('AC-079: the sustained high-speed slide stays inside the declared envelope', () => {
    const state = runStepsFrom(atTopSpeed(), 1200);

    // The slide is a BUDGET: the committed angle is clamped, so it never
    // runs away past the declared envelope, and it never vanishes either.
    // The DECLARED ENVELOPE of the PUBLISHED slip is the committed budget
    // plus the landed overshoot allowance, DERIVED from the two named
    // exports (`NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD` = 0.45 rad on the
    // COMMITTED slide, `NEATRACING_DRIFT_PUBLISHED_SLIP_OVERSHOOT_RAD` =
    // 0.002 rad for the chase lag) - 0.452 rad, never a literal. The
    // committed-slide clamp itself is untouched.
    //
    // The CALIBRATION ENVELOPE: the published peak is 0.45116 rad (the
    // chase overshoot at the slide onset), so the derived bound clears it
    // by 0.00084 rad. MEASURED on the final bytes: the 1200-tick capture
    // is 0.4500001875611939 rad.
    expect(Math.abs(state.slipAngle)).toBeLessThanOrEqual(
      NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD +
        NEATRACING_DRIFT_PUBLISHED_SLIP_OVERSHOOT_RAD,
    );
    expect(Math.abs(state.slipAngle)).toBeGreaterThan(0.2);
  });

  // [RED-PRESERVE-STRAIGHT-TOP-SPEED]
  it('AC-079: a straight-line full-throttle run still reaches the top speed', () => {
    const state = runSteps(600, 42, { throttle: 1, brake: 0, steer: 0 });

    expect(state.speed).toBe(NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S);
    expect(state.slipAngle).toBe(0);
  });

  // [RED-D1-MIN-SPEED-GATE]
  it('AC-079: a low-speed car neither slides nor loses forward grip', () => {
    const start: PhysicsCarState = {
      ...createInitialPhysicsCarState(42),
      speed: 10,
    };

    // Coasting below the 15 m/s arcade drift threshold: the outward push
    // is zero there, so the car never slides and keeps all of its grip.
    const state = runStepsFrom(start, 120, LOW_SPEED_STEER_INPUT, 'road');

    // The MIN_SPEED gate lives INSIDE the model, so the lateral channel
    // is exactly zero (not merely small) below the threshold.
    expect(state.lateralVelocity).toBe(0);
    expect(state.slipAngle).toBe(0);
  });

  // [RED-14Q-D1-MIN-SPEED-GATE-SOFTENED]
  it('AC-089: below the MIN_SPEED gate the VISIBLE slide stays exactly zero while the commitment is softened instead of frozen', () => {
    // The RE-BASELINED pin. The DELETED [RED-D1-MIN-SPEED-GATE-UNCOMMITTED]
    // pin asserted the HARD CLIFF:
    //     expect(driftChannelOf(state).committed.slip).toBe(0);
    //     expect(driftChannelOf(state).committed.turnSign).toBe(0);
    // The MEASURED current-model behaviour below the 15 m/s gate is that
    // exact zero pair (the commitment never latches).
    //
    // The 14Q form (a RED decision, AC-089): the gate is SOFTENED into a
    // proportional ramp, NEVER removed. Below the gate the COMMITMENT is
    // no longer frozen (a non-zero committed slip), while the VISIBLE
    // slide stays exactly zero - the sibling [RED-D1-MIN-SPEED-GATE] pin
    // (the lateral velocity and the published slip are exactly 0) keeps
    // holding untouched. The turn sign is NOT asserted here: it is an
    // implementation detail of the ramp, and the latch-release law
    // (RED-4) owns it.
    const start: PhysicsCarState = {
      ...createInitialPhysicsCarState(42),
      speed: 10,
    };

    const state = runStepsFrom(start, 120, LOW_SPEED_STEER_INPUT, 'road');

    // The NON-VACUITY GUARD comes FIRST: the softened gate must produce
    // a REAL commitment, not a different flavour of zero.
    expect(
      Math.abs(driftChannelOf(state).committed.slip),
    ).toBeGreaterThan(0);
    // The VISIBLE slide stays exactly zero below the gate.
    expect(state.lateralVelocity).toBe(0);
    expect(state.slipAngle).toBe(0);
  });
});

/**
 * The RC-1 OFF-ROAD FLOOR contract (GREEN loop-back iteration 3; the
 * `ORCHESTRATOR RECORD - P1-drift-feel-green HALTED` root cause RC-1).
 *
 * ROOT CAUSE (MODEL, physics.step.service.ts): the longitudinal integrator
 * clamps the OFF-ROAD SPEED FLOOR against the TICK-START speed (the
 * js-racer limit-gate parity), but the DRIFT SCRUB is subtracted AFTER
 * that clamp (tick step 4c). A drifting off-road car is therefore pushed
 * BELOW the floor; on the next tick the tick-start gate no longer holds
 * the floor, so the car keeps decaying and ends up PERMANENTLY TRAPPED at
 * 0 m/s. The no-drift lane never exposes it, because with no slip the
 * scrub is exactly 0.
 *
 * THE INVARIANT: an off-road car under throttle NEVER decays below the
 * off-road floor, WITH OR WITHOUT a drift.
 *
 * FIXTURE: deterministic, seeded (seed 42), no mocks - a constant
 * `grass` surface resolver, a constant control sample and a fixed 60 Hz
 * tick count. The start state is the existing top-speed fixture, so the
 * lane begins far ABOVE the floor and must descend onto it. Rationale:
 * the trap is only reachable from a slide, and the start must be above
 * the floor for the tick-start gate to engage.
 */
describe('physics step service off-road floor under drift (RC-1)', () => {
  // [RED-RC1-FLOOR-WITH-DRIFT]
  it('RC-1: an off-road car under throttle never decays below the off-road floor while drifting', () => {
    const samples = runSpeedSamples(
      atTopSpeed(),
      OFF_ROAD_FLOOR_TICKS,
      GRASS_DRIFT_INPUT,
      'grass',
    );
    const minSpeed = Math.min(...samples);

    // The floor is the NAMED export (the USER-named off-track limit,
    // 20 km/h / 3.6, the iteration-3 re-derivation), never a literal.
    expect(minSpeed).toBeGreaterThanOrEqual(
      NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S -
        OFF_ROAD_FLOOR_TOLERANCE,
    );
  });

  // [RED-RC1-NO-ZERO-TRAP]
  it('RC-1: the drifting off-road car is never trapped at zero speed over 600 ticks', () => {
    const samples = runSpeedSamples(
      atTopSpeed(),
      OFF_ROAD_FLOOR_TICKS,
      GRASS_DRIFT_INPUT,
      'grass',
    );
    const zeroSpeedTicks = samples.filter((speed) => speed === 0).length;

    // The trap is a SUSTAINED zero, not a single zero tick.
    expect(zeroSpeedTicks).toBe(0);
    // The car is still moving when the window closes.
    expect(samples[samples.length - 1]).toBeGreaterThan(0);
  });

  // [RED-RC1-A-B-PIN]
  it('RC-1: the off-road floor holds identically with and without a drift (A/B)', () => {
    // A (the control): the same lane WITHOUT a drift coasts down to the
    // floor and HOLDS it, because with no slip the scrub is exactly 0.
    const withoutDrift = runStepsFrom(
      atTopSpeed(),
      OFF_ROAD_FLOOR_TICKS,
      GRASS_STRAIGHT_INPUT,
      'grass',
    );
    // B: the same lane WITH a drift must not end up WORSE than the
    // control - the drift may cost speed, it may not break the floor.
    const withDrift = runStepsFrom(
      atTopSpeed(),
      OFF_ROAD_FLOOR_TICKS,
      GRASS_DRIFT_INPUT,
      'grass',
    );

    // The control lane is a CHARACTERIZATION pin: the recorded behavior
    // is that the no-drift off-road car settles EXACTLY at the floor
    // (the CURRENT floor is the USER-named 20 km/h cap; the RC-1-era
    // record measured the same exact-settle at the then-floor 22.25).
    // A later fix must not re-derive that equilibrium.
    expect(withoutDrift.speed).toBeCloseTo(
      NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S,
      6,
    );
    expect(withDrift.speed).toBeGreaterThanOrEqual(
      withoutDrift.speed - OFF_ROAD_FLOOR_TOLERANCE,
    );
  });
});

/**
 * Runs a SURFACE SCHEDULE: a list of `[surface, tickCount]` phases, so a
 * lane can TRANSITION between surfaces (road -> grass -> road) exactly
 * like the track does under the car.
 *
 * The surface resolver reads the tick-start state, so the phase index is
 * the only state the schedule needs.
 */
const runSurfaceSchedule = (
  start: PhysicsCarState,
  phases: SurfaceSchedule,
  input: PhysicsStepInput,
): PhysicsCarState[] => {
  const schedule: FixtureSurface[] = [];
  for (const [surface, ticks] of phases) {
    for (let tick = 0; tick < ticks; tick++) {
      schedule.push(surface);
    }
  }
  let state = start;
  const surfaceResolver = (): FixtureSurface =>
    schedule[state.tick] ?? schedule[schedule.length - 1];
  const states: PhysicsCarState[] = [];
  for (let tick = 0; tick < schedule.length; tick++) {
    state = stepPhysics(state, input, surfaceResolver);
    states.push(state);
  }
  return states;
};

/**
 * The largest PUBLISHED slip angle observed on a surface schedule (R2).
 *
 * The published slip is the value the service publishes as
 * `state.slipAngle`. The sweep is deterministic: seed 42, the top-speed
 * start state, a constant full-throttle full-steer drift input and a
 * constant surface schedule - no mocks, no clock, no randomness.
 *
 * @param phases - The surface schedule of the lane.
 * @returns The largest absolute published slip on the lane, in radians.
 */
const peakPublishedSlip = (phases: SurfaceSchedule): number =>
  Math.max(
    ...runSurfaceSchedule(atTopSpeed(), phases, GRASS_DRIFT_INPUT).map(
      (state) => Math.abs(state.slipAngle),
    ),
  );

/**
 * The REGIME-AWARE PUBLISHED-SLIP BOUNDS over the swept envelope (R2;
 * the D6 FALLBACK RATIFICATION and the W2a CLOSED record).
 *
 * WHAT: the published slip is bounded, and the bound is REGIME-AWARE.
 * The ON-ROAD regime (the road/curb control lanes) is asserted against
 * the DERIVED named export `NEATRACING_DRIFT_PUBLISHED_SLIP_MAX_RAD`
 * (`NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD` = 0.45 rad plus
 * `NEATRACING_DRIFT_PUBLISHED_SLIP_OVERSHOOT_RAD` = 0.002 rad = 0.452
 * rad) and the OFF-ROAD regime (the grass/curb/transition sweep) is
 * asserted against the DECLARED named export
 * `NEATRACING_DRIFT_PUBLISHED_SLIP_OFFROAD_MAX_RAD` (0.493 rad). BOTH
 * bounds are named exports and BOTH are mirrored here - never from a
 * literal - so a re-pin of either export moves the assertion with it.
 *
 * WHY TWO BOUNDS (the ratified D6 fallback): the published slip is the
 * committed slide carried through the first-order chase, and the chase
 * LAGS a target that shrinks as the speed scrub bleeds speed. On the
 * road the denominator is stable, so the lag stays small (the measured
 * on-road peak is 0.4511617369362305 rad - 0.00084 rad of headroom
 * under the derived ceiling). Off the road the drag drives the same lag
 * from a FALLING denominator, so the published slip peaks higher. The
 * D6 plan first proposed a publish-site CLAMP, but the W1b measurement
 * FALSIFIED the clamp's off-road inertness, so the clamp was DROPPED:
 * the published slip keeps its PHYSICAL value (clipping it would fake
 * the velocityHeading trajectory) and the off-road regime is asserted
 * against a DECLARED ceiling instead. A DECLARED ceiling is a RECORDED
 * decision, never a silent clip.
 *
 * MEASURED (post-RC-1, the W2a record; the jest lane, seed 42, the
 * top-speed start, a constant full-throttle full-steer drift input,
 * 300-tick lanes; the tick numbers count COMPLETED ticks of the lane):
 * the off-road envelope peaks at 0.4872167352892035 rad and the per-lane
 * peaks are 0.4872167 (grass, tick 115), 0.4870472 (road -> grass, tick
 * 151), 0.4870997 (road -> grass, the 120-tick entry, tick 225),
 * 0.4872167 (grass -> road, tick 115), 0.4783253 (road -> grass -> road,
 * tick 150), 0.4869153 (grass -> curb -> grass, tick 256) and 0.4868913
 * (grass -> road -> grass, tick 170). The control lanes peak at
 * 0.4511617369362305 rad on tick 42. The off-road envelope therefore
 * EXCEEDS the on-road ceiling (0.452 rad) and stays under the off-road
 * ceiling (0.493 rad, a 0.0058 rad margin over the measured peak), while
 * the control lanes stay under BOTH - so the regime split is real and
 * the off-road ceiling is not a copy of the on-road one.
 *
 * WHAT THE RC-1 FIX MOVED: before the fix the off-road envelope peaked
 * at 0.503962 rad and breached the single 0.452 rad ceiling from the
 * FIRST off-road tick (tick 17, 77.53 m/s - far above the then-floor
 * 22.25 m/s), because the collapsing denominator drove the chase lag on
 * its own. After the fix the car no longer decays into the 0 m/s trap, so
 * the denominator stops collapsing and the peak MOVES into the
 * floor-holding regime (a RC-1-era record: the per-lane peaks landed
 * just above the then-floor at 22.42-22.64 m/s, except the road ->
 * grass -> road lane at 31.11 m/s; under the iteration-3 USER cap the
 * regime STARTS at the 5.5556 m/s settle). The lane SET and the lane
 * tick budgets are UNCHANGED
 * (seven off-road lanes, three controls); only the recorded peak values
 * and the peak ticks moved, and both are re-pinned above.
 */
describe('physics step service regime-aware published-slip bounds over the swept envelope (R2)', () => {
  // [RED-R2-OFFROAD-SLIP-CEILING]
  it('R2: the published slip on every off-road lane of the swept envelope stays within the declared off-road ceiling', () => {
    // Arrange
    const envelopePeak = Math.max(
      ...OFF_ROAD_SLIP_ENVELOPE.map((phases) => peakPublishedSlip(phases)),
    );

    // Act + Assert
    expect(envelopePeak).toBeLessThanOrEqual(
      NEATRACING_DRIFT_PUBLISHED_SLIP_OFFROAD_MAX_RAD,
    );
  });

  // [RED-R2-ONROAD-SLIP-CEILING]
  it('R2: the published slip on the road/curb control lanes stays within the derived on-road ceiling', () => {
    // Arrange
    const controlPeak = Math.max(
      ...ROAD_CURB_SLIP_CONTROL.map((phases) => peakPublishedSlip(phases)),
    );

    // Act + Assert
    expect(controlPeak).toBeLessThanOrEqual(
      NEATRACING_DRIFT_PUBLISHED_SLIP_MAX_RAD,
    );
  });

  // [GUARD-R2-REGIME-SPLIT-NON-VACUOUS]
  it('R2: the regime split is non-vacuous - the off-road envelope exceeds the on-road ceiling', () => {
    // Arrange
    const envelopePeak = Math.max(
      ...OFF_ROAD_SLIP_ENVELOPE.map((phases) => peakPublishedSlip(phases)),
    );

    // Act + Assert
    expect(envelopePeak).toBeGreaterThan(NEATRACING_DRIFT_PUBLISHED_SLIP_MAX_RAD);
  });
});

/**
 * The SCRUB-COEFFICIENT BAND guard (R5; the D-MEMO `wave_amendments`:
 * "R5 = the c-band guard (c in [0.504, 0.626] preserves the deficit
 * floor) - guard, not assertion").
 *
 * THIS IS A GUARD, NOT A RED CONTRACT: it is GREEN today by design and
 * it exists so that a re-pin of
 * `NEATRACING_DRIFT_SPEED_SCRUB_COEFF_PER_S` without a re-derivation
 * of the tolerances FAILS LOUDLY (the 14N determinism review, R5:
 * "assert the shipped NEATRACING_DRIFT_SPEED_SCRUB_COEFF_PER_S lies in
 * the band where the measured published peak stays <= 0.452 AND the
 * D1.3 deficit floor still holds (0.504 <= c <= 0.626)").
 *
 * THE BAND (the ratified GAP-N4 calibration baseline; the D-MEMO
 * `decision_d5.rationale` re-baselines "the c-band [0.504, 0.626]"):
 * - the LOWER edge 0.504 is where the D1.3 deficit floor binds (the
 *   review: "the D1.3 deficit floor kills c < 0.504");
 * - the UPPER edge 0.626 is where the derived 0.452 published-slip
 *   ceiling is crossed (the review's sweep: c = 0.626 ->
 *   0.4519979733 rad, c = 0.65 -> 0.4522611403 rad = VIOLATED).
 *
 * THE GUARDED PROPERTY: inside the band the D1.3 deficit floor holds.
 * The tie-back assertion below measures the deficit at the SHIPPED
 * coefficient (c = 0.55 -> 32.55473 km/h against the never-relaxed
 * 20 km/h floor, a 63 percent margin) so the band membership is not a
 * bare tautology: the band only matters while the property it
 * protects actually holds. The OTHER half of the band (the
 * published-slip ceiling) is witnessed by the R2 CONTROL lanes in the
 * sibling contract above, so it is not duplicated here.
 */
describe('physics step service scrub-coefficient band guard (R5)', () => {
  // [GUARD-R5-SCRUB-COEFF-BAND]
  it('R5: the shipped scrub coefficient stays inside the band that preserves the D1.3 deficit floor', () => {
    // Arrange
    const coefficient = NEATRACING_DRIFT_SPEED_SCRUB_COEFF_PER_S;
    const straight = runSteps(600, 42, STRAIGHT_INPUT, 'road');
    const drift = runStepsFrom(atTopSpeed(), 600, FIXTURE_INPUT, 'road');
    const deficitKmh = (straight.speed - drift.speed) * 3.6;

    // Act + Assert
    expect(coefficient).toBeGreaterThanOrEqual(SCRUB_COEFF_BAND_LOWER);
    expect(coefficient).toBeLessThanOrEqual(SCRUB_COEFF_BAND_UPPER);
    expect(deficitKmh).toBeGreaterThanOrEqual(
      NEATRACING_DRIFT_MIN_SPEED_DEFICIT_KMH,
    );
  });
});

/**
 * The RED discriminator tick (Step 14P, slice `P1-drift-rewire-red`).
 *
 * The drift ENGAGES at tick 52 (the 15 m/s MIN_SPEED gate), so the
 * pose-source discriminator has to be read on a tick INSIDE the slide.
 * Tick 61 is the tick the plan's `red_contract_api` section 4 measured.
 */
const RED_WAVE_DISCRIMINATOR_TICKS = 61;

/**
 * The RED hold length (Step 14P, slice `P1-drift-rewire-red`).
 *
 * The 200-tick full-steer hold of the plan's `red_contract_api`
 * section 4 (the measured current-model lane).
 */
const RED_WAVE_HOLD_TICKS = 200;

/**
 * Reads the NOT-YET-EXISTING travel heading off a car state.
 *
 * The field `travelHeadingRad` is the RED-1 contract (the plan's
 * `red_contract_api` section 1) and does not exist on `PhysicsCarState`
 * yet, so the read goes through this LOCAL widened view: the field is
 * `undefined` today, which is what makes the RED-1 contracts fail on
 * their OWN assertion instead of on a file-level compile error that
 * would mask every sibling contract in this file.
 *
 * @param state - The car state to read.
 * @returns The published travel heading, or `undefined` while the
 *   service publishes no travel direction at all.
 */
const travelHeadingRadOf = (state: PhysicsCarState): number | undefined =>
  (state as unknown as { travelHeadingRad?: number }).travelHeadingRad;

/**
 * Wraps an angle into the `[-PI, PI)` range, with the SAME convention
 * the service uses for the heading (`wrapHeading`).
 *
 * @param radians - The angle to wrap.
 * @returns The wrapped angle.
 */
const wrapRadians = (radians: number): number =>
  radians - 2 * Math.PI * Math.floor((radians + Math.PI) / (2 * Math.PI));

/**
 * Runs the step service for a fixed number of 60 Hz ticks from the
 * seeded spawn and returns the tick-END state TOGETHER WITH the state
 * it was stepped FROM.
 *
 * The RED-2 contract needs the PREVIOUS pose state (the pose-spring
 * input) next to the tick-end state, so the lane records both.
 *
 * @param steps - The number of 60 Hz ticks to run.
 * @param input - The constant control sample applied on every tick.
 * @param surface - The constant surface the resolver reports.
 * @returns The tick-start and the tick-end state of the last tick.
 */
const runStepsWithPrev = (
  steps: number,
  input: PhysicsStepInput = FIXTURE_INPUT,
  surface: FixtureSurface = 'road',
): { prev: PhysicsCarState; state: PhysicsCarState } => {
  const surfaceResolver = (): FixtureSurface => surface;
  let state = createInitialPhysicsCarState(42);
  let prev = state;
  for (let step = 0; step < steps; step++) {
    prev = state;
    state = stepPhysics(state, input, surfaceResolver);
  }
  return { prev, state };
};

/**
 * Builds the pose-spring reference for one tick from a chosen slip.
 *
 * The reference is the MODEL call the service is expected to make
 * (`stepDriftPoseSpring`), fed with the tick-start pose state, the
 * tick-start speed percentile and the chosen slip - the published slip
 * for the RED-2 contract, the committed slip for its discriminator.
 *
 * @param prev - The tick-start car state (the pose-spring input).
 * @param slipAngle - The slip that feeds the pose target (rad).
 * @returns The pose the spring would produce for that slip.
 */
const poseSpringReference = (
  prev: PhysicsCarState,
  slipAngle: number,
): number =>
  driftModelModule.stepDriftPoseSpring(prev.drift.pose, {
    steer: FIXTURE_INPUT.steer,
    speedPercent: Math.min(
      1,
      Math.max(0, prev.speed / NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S),
    ),
    slipAngle,
    dt: NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
  }).pose;

/**
 * The DRIFT RE-WIRING wave contracts (Step 14P, slice
 * `P1-drift-rewire-red`; the plan's RED-1 and RED-2 acceptance criteria
 * and the `red_contract_api` sections 1, 4 and 5).
 *
 * RED-1 (the state contract - the TRAVEL DIRECTION): the physics state
 * must publish the direction the car is actually TRAVELLING in, so the
 * camera and the body can read the SAME number. The named field
 * `travelHeadingRad` is the velocity heading the service ALREADY
 * computes to advance x and y and then DISCARDS (the G-M1 break line at
 * physics.step.service.ts:462/475). The field does NOT exist yet, so
 * every read goes through the LOCAL widened view `travelHeadingRadOf`:
 * a named import of a missing export would be a file-level compile
 * error that masks every sibling contract in this file.
 *
 * RED-2 (the ONE shared slip scalar and its sign): the pose spring must
 * be driven from the PUBLISHED slip (`state.slipAngle`) and NOT from the
 * COMMITTED slip (`state.drift.committed.slip`), so the visible crab
 * and the traction law can never read two different slides (the G-V3
 * defect). The published-slip SIGN convention (a LEFT steer yields a
 * NEGATIVE published slip) is pinned in the same contract, because the
 * audits could not verify it.
 *
 * MEASURED on the current model (seed 42, road, dt = 1/60, full
 * throttle, full LEFT steer = +1, the 200-tick hold of the plan's
 * `red_contract_api` section 4): the drift ENGAGES at tick 52 and the
 * RED-2 discriminator tick 61 reads the heading 0.20329298 rad, the
 * PUBLISHED slip -0.34897859 rad, the COMMITTED slip -0.45 rad, the
 * live pose -0.20404854 rad, the published-slip spring reference
 * -0.19950258 rad and the committed-slip spring reference
 * -0.20404854 rad - the live pose EQUALS the committed-slip reference
 * BIT-FOR-BIT (the G-V3 defect) and differs from the published-slip
 * reference by 0.00454596 rad. The 200-tick hold ends at the heading
 * 2.05209392 rad, the published slip -0.44354955 rad, the speed
 * 50.73672730 m/s and the travel heading 1.60854436 rad, with the
 * tick advance dx -0.03191262 m and dy 0.84500973 m. The straight lane
 * leaves the published slip EXACTLY 0.
 */
describe('drift re-wiring wave contracts (Step 14P, slice P1-drift-rewire-red)', () => {
  describe('RED-1 (the state contract - the travel direction)', () => {
    // [RED-1-TRAVEL-HEADING-PUBLISHED]
    it('publishes a finite travel heading on the car state', () => {
      // Arrange
      const { state } = runStepsWithPrev(RED_WAVE_HOLD_TICKS);
      const travelHeadingRad = travelHeadingRadOf(state);

      // Act + Assert
      expect(travelHeadingRad).toEqual(expect.any(Number));
      expect(Number.isFinite(travelHeadingRad)).toBe(true);
    });

    // [RED-1-TRAVEL-HEADING-WRAPPED]
    it('publishes the travel heading as the wrapped heading plus the published slip', () => {
      // Arrange
      const { state } = runStepsWithPrev(RED_WAVE_HOLD_TICKS);
      const travelHeadingRad = travelHeadingRadOf(state);
      const expectedTravelHeadingRad = wrapRadians(
        state.heading + state.slipAngle,
      );

      // Act + Assert
      expect(travelHeadingRad).toBeDefined();
      // The comparison is WRAP-INVARIANT: both sides are wrapped, so a
      // wrapped [-pi, pi) publication and a raw publication are BOTH
      // accepted.
      expect(wrapRadians(travelHeadingRad ?? Number.NaN)).toBeCloseTo(
        expectedTravelHeadingRad,
        12,
      );
    });

    // [RED-1-TRAVEL-HEADING-ADVANCE]
    it('advances x and y with exactly the published travel heading', () => {
      // Arrange
      const { prev, state } = runStepsWithPrev(RED_WAVE_HOLD_TICKS);
      const travelHeadingRad = travelHeadingRadOf(state);
      const advanceX =
        Math.cos(travelHeadingRad ?? Number.NaN) *
        state.speed *
        NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS;
      const advanceY =
        Math.sin(travelHeadingRad ?? Number.NaN) *
        state.speed *
        NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS;

      // Act + Assert
      expect(state.x - prev.x).toBeCloseTo(advanceX, 12);
      expect(state.y - prev.y).toBeCloseTo(advanceY, 12);
    });
  });

  describe('RED-2 (the shared slip scalar and its sign)', () => {
    // [RED-2-POSE-SPRING-SOURCE]
    it('drives the pose spring from the published slip, not from the committed slip', () => {
      // Arrange: the tick-61 discriminator (inside the slide).
      const { prev, state } = runStepsWithPrev(RED_WAVE_DISCRIMINATOR_TICKS);
      const publishedReference = poseSpringReference(prev, state.slipAngle);
      const committedReference = poseSpringReference(
        prev,
        state.drift.committed.slip,
      );

      // Act + Assert
      // The NON-VACUITY GUARD comes FIRST: the two references must
      // differ, so the equality below can never pass vacuously.
      expect(publishedReference).not.toBeCloseTo(committedReference, 6);
      // The live pose must EQUAL the PUBLISHED-slip reference ...
      expect(state.drift.pose.pose).toBeCloseTo(publishedReference, 12);
      // ... and must DIFFER from the committed-slip reference.
      expect(state.drift.pose.pose).not.toBeCloseTo(committedReference, 6);
    });

    // [RED-2-PUBLISHED-SLIP-SIGN]
    it('publishes a negative slip for a full left steer and an exact zero on a straight line', () => {
      // Arrange
      const { state: drifted } = runStepsWithPrev(RED_WAVE_DISCRIMINATOR_TICKS);
      const { state: straight } = runStepsWithPrev(
        RED_WAVE_HOLD_TICKS,
        STRAIGHT_INPUT,
      );

      // Act + Assert
      // The SIGN CONVENTION: a LEFT steer (steer = +1) yields a
      // NEGATIVE published slip.
      expect(drifted.slipAngle).toBeLessThan(0);
      // The STRAIGHT-LINE IDENTITY: the published slip is EXACTLY zero.
      expect(straight.slipAngle).toBe(0);
    });
  });
});

/* ------------------------------------------------------------------------- *
 * STEP 14Q (slice `P1-drift-law-red`): the PUBLISHED traction scalar.
 * ------------------------------------------------------------------------- */

/**
 * The 14Q deep committed slip (rad) of the monotonicity lane.
 *
 * The deep lane is the full AC-087 slide; the shallow lane is a light
 * slide. Both lanes start from the SAME tick-start speed (the arcade
 * top speed), so the only difference between them is the slide depth.
 */
const RED_DEEP_COMMITTED_SLIP_RAD = -0.45;

/** The 14Q shallow committed slip (rad) of the monotonicity lane. */
const RED_SHALLOW_COMMITTED_SLIP_RAD = -0.1;

/**
 * Reads the NOT-YET-EXISTING traction severity off a car state.
 *
 * The field `tractionSeverity` is the RED-5 contract and does not exist
 * on `PhysicsCarState` yet, so the read goes through this LOCAL widened
 * view: the field is `undefined` today, which is what makes the RED-5
 * contracts fail on their OWN assertion instead of on a file-level
 * compile error that would mask every sibling contract in this file.
 *
 * @param state - The car state to read.
 * @returns The published traction severity, or `undefined` while the
 *   service publishes no traction scalar at all.
 */
const tractionSeverityOf = (state: PhysicsCarState): number | undefined =>
  (state as unknown as { tractionSeverity?: number }).tractionSeverity;

/**
 * Reads the NOT-YET-EXISTING traction-severity law off the drift model.
 *
 * The law is the NOT-YET-EXISTING `computeTractionSeverity(slipAngleRad,
 * speedMps)` export: the severity is a NAMED function of the SPEED and
 * the PUBLISHED SLIP, so the published state value and the law value
 * must be the SAME number (the ONE shared scalar).
 */
const computeTractionSeverityOf = (
  driftModelModule as unknown as {
    computeTractionSeverity?: (
      slipAngleRad: number,
      speedMps: number,
    ) => number;
  }
).computeTractionSeverity;

/**
 * Runs ONE service tick from a chosen committed slip at the arcade top
 * speed.
 *
 * The monotonicity lane needs two lanes at the SAME tick-start speed with
 * two different published slips, so the lanes are built from an explicit
 * committed state instead of from a driven slide.
 *
 * @param committedSlipRad - The committed slip the lane starts from (rad).
 * @param speed - The tick-start speed of the lane (m/s).
 * @returns The tick-end car state of the lane.
 */
const runOneTickFromCommitted = (
  committedSlipRad: number,
  speed = NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
): PhysicsCarState => {
  const start: PhysicsCarState = {
    ...createInitialPhysicsCarState(42),
    speed,
    drift: {
      committed: {
        slip: committedSlipRad,
        turnSign: committedSlipRad < 0 ? 1 : -1,
        reArmed: true,
      },
      pose: { pose: 0, poseRate: 0, articulation: 0 },
    },
  };
  const surfaceResolver = (): FixtureSurface => 'road';
  return stepPhysics(start, FIXTURE_INPUT, surfaceResolver);
};

/**
 * The PROPORTIONAL traction-law contracts (Step 14Q, slice
 * `P1-drift-law-red`; the plan's RED-5 acceptance criterion).
 *
 * RED-5 (the published traction scalar): the physics state must publish
 * the ONE shared traction severity as the named field `tractionSeverity`.
 * The severity is a NAMED function of the SPEED and the PUBLISHED SLIP
 * (AC-088), so it is derived from the POST-traction speed (the value the
 * service publishes as `state.speed` after the scrub) and the published
 * slip; it is EXACTLY 0 on a straight line, it lives inside the closed
 * unit interval, and it is MONOTONIC in the slide depth. The state value
 * IS the law value: no consumer may re-derive it.
 *
 * The MEASURED current-model state: the service publishes NO traction
 * field at all, so every severity read is `undefined` today. The
 * monotonicity lane (two one-tick lanes from the same tick-start speed of
 * 89 m/s) reads the published slips -0.10528 rad (the deep lane) and
 * -0.04935 rad (the shallow lane) - the slip ordering the monotonicity
 * contract relies on.
 */
describe('the published traction scalar (Step 14Q, slice P1-drift-law-red)', () => {
  describe('RED-5 (the published traction scalar)', () => {
    // [RED-5-TRACTION-SEVERITY-PUBLISHED]
    it('publishes a finite traction severity on the car state', () => {
      // Arrange
      const { state } = runStepsWithPrev(RED_WAVE_HOLD_TICKS);
      const tractionSeverity = tractionSeverityOf(state);

      // Act + Assert
      expect(tractionSeverity).toEqual(expect.any(Number));
      expect(Number.isFinite(tractionSeverity)).toBe(true);
    });

    // [RED-5-TRACTION-SEVERITY-STRAIGHT-ZERO]
    it('publishes an EXACT zero traction severity on a straight line', () => {
      // Arrange
      const { state } = runStepsWithPrev(RED_WAVE_HOLD_TICKS, STRAIGHT_INPUT);

      // Act + Assert
      // The published slip is EXACTLY 0 on a straight line, so the
      // severity is EXACTLY 0 too.
      expect(state.slipAngle).toBe(0);
      expect(tractionSeverityOf(state)).toBe(0);
    });

    // [RED-5-TRACTION-SEVERITY-UNIT-INTERVAL]
    it('publishes a traction severity inside the closed unit interval', () => {
      // Arrange
      const { state } = runStepsWithPrev(RED_WAVE_HOLD_TICKS);
      const tractionSeverity = tractionSeverityOf(state);

      // Act + Assert
      expect(tractionSeverity).toBeGreaterThanOrEqual(0);
      expect(tractionSeverity).toBeLessThanOrEqual(1);
    });

    // [RED-5-TRACTION-SEVERITY-MONOTONIC]
    it('publishes a traction severity that is MONOTONIC in the published slip at a fixed speed', () => {
      // Arrange: two one-tick lanes from the SAME tick-start speed (the
      // arcade top speed) with two different committed slips. The
      // tick-END speeds differ only by the speed scrub (0.3 percent),
      // which the slip ordering dominates.
      const deep = runOneTickFromCommitted(RED_DEEP_COMMITTED_SLIP_RAD);
      const shallow = runOneTickFromCommitted(RED_SHALLOW_COMMITTED_SLIP_RAD);

      // Act + Assert
      // The NON-VACUITY GUARD comes FIRST: the two lanes must really
      // publish different slips, so the ordering below can never pass
      // vacuously.
      expect(Math.abs(deep.slipAngle)).toBeGreaterThan(
        Math.abs(shallow.slipAngle),
      );
      // The deeper slide publishes the LARGER severity.
      expect(tractionSeverityOf(deep)).toBeGreaterThan(
        tractionSeverityOf(shallow) ?? Number.NaN,
      );
    });

    // [RED-5-TRACTION-SEVERITY-SHARED-SCALAR]
    it('publishes the ONE shared traction severity: the state value IS the model law on the published slip and the post-traction speed', () => {
      // Arrange
      const { state } = runStepsWithPrev(RED_WAVE_HOLD_TICKS);
      const tractionSeverity = tractionSeverityOf(state);

      // Act + Assert
      // The NON-VACUITY GUARD comes FIRST: the law must exist, so the
      // identity below can never pass on a NaN comparison.
      expect(typeof computeTractionSeverityOf).toBe('function');
      // The published severity IS the law: the PUBLISHED slip and the
      // POST-traction speed (the value the service publishes as
      // `state.speed` after the scrub) produce the same number, so no
      // consumer can re-derive a different value.
      expect(tractionSeverity).toBeCloseTo(
        computeTractionSeverityOf?.(state.slipAngle, state.speed) ??
          Number.NaN,
        12,
      );
    });
  });
});

/* ------------------------------------------------------------------------- *
 * STEP 14R (slice `P1-drift-collision-live`): the LIVE car-to-car collision.
 * ------------------------------------------------------------------------- */

/**
 * The NOT-YET-EXISTING live pack tick, read through a LOCAL widened view
 * (the file MUST COMPILE: a missing export never masks a sibling
 * contract of this file).
 */
const stepPhysicsFleetOf = (
  physicsStepService as unknown as {
    stepPhysicsFleet?: (
      fleet: readonly PhysicsCarState[],
      inputs: readonly PhysicsStepInput[],
      surfaceResolver?: (carState: PhysicsCarState) => FixtureSurface,
    ) => PhysicsCarState[];
  }
).stepPhysicsFleet;

/**
 * The NOT-YET-EXISTING live collision seam, read through a LOCAL widened
 * view (the file MUST COMPILE: a missing export never masks a sibling
 * contract of this file).
 */
const resolveLiveCarCollisionsOf = (
  physicsStepService as unknown as {
    resolveLiveCarCollisions?: (
      fleet: readonly PhysicsCarState[],
    ) => PhysicsCarState[];
  }
).resolveLiveCarCollisions;

/** The NEUTRAL control sample: no throttle, no brake, no steer. */
const NEUTRAL_INPUT: PhysicsStepInput = { throttle: 0, brake: 0, steer: 0 };

/** The road surface probe of the live collision fixtures. */
const ROAD_SURFACE_RESOLVER = (): FixtureSurface => 'road';

/**
 * Builds a tick-start car state at a known world pose and speed.
 *
 * The spawn factory arms the drift channel, so the fixture starts from a
 * clean car (no slide, no lateral velocity) and the live tick owns every
 * published value from there on.
 *
 * @param x - The world x position of the car at tick start (m).
 * @param speed - The longitudinal speed of the car at tick start (m/s).
 * @returns The tick-start car state.
 */
const buildCollisionFixtureCar = (
  x: number,
  speed: number,
): PhysicsCarState => ({
  ...createInitialPhysicsCarState(42, { x, y: 0, heading: 0 }),
  speed,
});

/**
 * The LIVE car-to-car collision contracts (Step 14R, slice
 * `P1-drift-collision-live`; AC-090, AC-089, spec LD-05, D6).
 *
 * RED-4 (the live collision contract): the resolver (`resolveCarCollisions`
 * in `./collision.utils`) was imported by NO production module, so a
 * car-to-car overlap was DEAD in the LIVE tick - two overlapping cars
 * simply drove through each other. The contract below is written against
 * the LIVE step (the pack tick that advances a whole fleet and publishes
 * the RESOLVED states) and NEVER against the resolver in isolation: the
 * resolver keeps its own unit coverage in `collision.utils.test.ts`.
 *
 * The ORDER the live tick must keep (AC-090): the drift integration runs
 * FIRST for every member, the car-to-car resolution runs AFTER it, and
 * the RESOLVED states are published LAST. The resolution is a NAMED seam
 * with a SINGLE owner, so the resolver has exactly ONE live call site.
 */
describe('the LIVE car-to-car collision resolution (Step 14R, slice P1-drift-collision-live)', () => {
  describe('RED-4 (the live collision contract)', () => {
    // [RED-4-LIVE-COLLISION-SEAM]
    it('publishes the NAMED live collision seam (the resolver is no longer DEAD)', () => {
      // Act + Assert
      // The seam is the ONE named owner of the resolver call, so the
      // resolution has a single owner and a single live call site.
      expect(typeof resolveLiveCarCollisionsOf).toBe('function');
    });

    // [RED-4-LIVE-COLLISION-RESOLVED]
    it('resolves an overlapping car pair in the LIVE step (the positions separate and the speeds adjust)', () => {
      // Arrange: a fast car BEHIND a standing car, close enough to
      // overlap (the collision diameter is twice the collision radius).
      const fleet = [
        buildCollisionFixtureCar(0, 20),
        buildCollisionFixtureCar(0.5, 0),
      ];
      const inputs = [NEUTRAL_INPUT, NEUTRAL_INPUT];
      // The SAME tick WITHOUT the fleet: the baseline the live collision
      // has to change.
      const uncollided = fleet.map((car, index) =>
        stepPhysics(car, inputs[index], ROAD_SURFACE_RESOLVER),
      );

      // Act
      const published =
        stepPhysicsFleetOf?.(fleet, inputs, ROAD_SURFACE_RESOLVER) ?? [];

      // Assert
      // The NON-VACUITY GUARD comes FIRST: the live pack tick must exist,
      // so the assertions below can never pass on an empty result.
      expect(typeof stepPhysicsFleetOf).toBe('function');
      expect(published).toHaveLength(2);
      // The overlap is REAL in the baseline: the two advanced cars sit
      // closer than the collision diameter.
      const baselineDistance = Math.hypot(
        uncollided[1].x - uncollided[0].x,
        uncollided[1].y - uncollided[0].y,
      );
      expect(baselineDistance).toBeLessThan(
        2 * NEATRACING_CAR_COLLISION_RADIUS_M,
      );
      // The live step SEPARATES the pair to exactly the sum of the radii.
      const publishedDistance = Math.hypot(
        published[1].x - published[0].x,
        published[1].y - published[0].y,
      );
      expect(publishedDistance).toBeCloseTo(
        2 * NEATRACING_CAR_COLLISION_RADIUS_M,
        9,
      );
      // The speeds are ADJUSTED: the approaching car stops and the
      // standing car takes its momentum (the equal-mass elastic exchange
      // along the collision normal).
      expect(published[0].speed).toBe(0);
      expect(published[1].speed).toBeCloseTo(uncollided[0].speed, 9);
      // The speeds really CHANGED: neither member kept its baseline
      // speed, so the assertions above can never pass vacuously.
      expect(published[0].speed).not.toBe(uncollided[0].speed);
      expect(published[1].speed).not.toBe(uncollided[1].speed);
    });

    // [RED-4-LIVE-COLLISION-NOOP]
    it('leaves a NON-overlapping fleet untouched (the live seam is a no-op)', () => {
      // Arrange
      const fleet = [
        buildCollisionFixtureCar(0, 20),
        buildCollisionFixtureCar(10, 0),
      ];
      const inputs = [NEUTRAL_INPUT, NEUTRAL_INPUT];
      const expected = fleet.map((car, index) =>
        stepPhysics(car, inputs[index], ROAD_SURFACE_RESOLVER),
      );

      // Act
      const published =
        stepPhysicsFleetOf?.(fleet, inputs, ROAD_SURFACE_RESOLVER) ?? [];

      // Assert
      expect(typeof stepPhysicsFleetOf).toBe('function');
      // The live fleet tick publishes EXACTLY the single-car ticks of the
      // members, so a fleet that does not overlap is never disturbed.
      expect(JSON.stringify(published)).toBe(JSON.stringify(expected));
    });

    // [RED-4-LIVE-COLLISION-DETERMINISM]
    it('publishes byte-identical states for the SAME fleet (the determinism twin-run)', () => {
      // Arrange
      const fleet = [
        buildCollisionFixtureCar(0, 20),
        buildCollisionFixtureCar(0.5, 0),
      ];
      const inputs = [NEUTRAL_INPUT, NEUTRAL_INPUT];

      // Act
      const first =
        stepPhysicsFleetOf?.(fleet, inputs, ROAD_SURFACE_RESOLVER) ?? [];
      const second =
        stepPhysicsFleetOf?.(fleet, inputs, ROAD_SURFACE_RESOLVER) ?? [];

      // Assert
      expect(typeof stepPhysicsFleetOf).toBe('function');
      // The twin-run of the live collision tick is byte-identical: the
      // resolution adds no hidden state to the published fleet.
      expect(JSON.stringify(first)).toBe(JSON.stringify(second));
    });

    // [RED-4-LIVE-COLLISION-STRAIGHT-ZEROS]
    it('keeps the straight-line EXACT zeros in the live fleet tick (AC-089)', () => {
      // Arrange
      const fleet = [
        buildCollisionFixtureCar(0, 20),
        buildCollisionFixtureCar(10, 0),
      ];
      const inputs = [STRAIGHT_INPUT, STRAIGHT_INPUT];

      // Act
      const published =
        stepPhysicsFleetOf?.(fleet, inputs, ROAD_SURFACE_RESOLVER) ?? [];

      // Assert
      expect(typeof stepPhysicsFleetOf).toBe('function');
      expect(published).toHaveLength(2);
      // On a straight line the published slip stays EXACTLY 0, so the
      // published traction severity stays EXACTLY 0 too.
      for (const car of published) {
        expect(car.slipAngle).toBe(0);
        expect(car.tractionSeverity).toBe(0);
      }
    });
  });
});

/* ------------------------------------------------------------------------- *
 * STEP 14S (slice `P1-momentum-red`): the LIVE tick APPLIES the momentum.
 * ------------------------------------------------------------------------- */

/**
 * The 14S SERVICE window (60 Hz ticks).
 *
 * The service lanes are compared over the SAME fixed window, so the
 * window is a named constant and never a carried literal.
 */
const MOMENTUM_SERVICE_WINDOW_TICKS = 30;

/** The steer axis of the 14S service lanes (+1 = LEFT). */
const MOMENTUM_SERVICE_STEER_AXIS = 1;

/**
 * The speed percent of the 14S top-speed service lane.
 *
 * The lane starts at the arcade top speed, so the speed percent the
 * service resolves is EXACTLY 1 (89 / 89).
 */
const MOMENTUM_SERVICE_SPEED_PERCENT = 1;

/**
 * The 14T TOP-SPEED turn-rate speed factor of the 14S service lane.
 *
 * BASIS (the AC-089 re-baseline ledger row P-T-1, Step 14T, slice
 * `P1-turnrate-repin`): the OLD baseline was the UN-SOFTENED steer-axis
 * product (`2` rad/s at the arcade top speed). The 14T USER law applies
 * the mild speed-proportional factor LAST, so the top-speed lane (the
 * speed percent is EXACTLY 1) resolves the factor as EXACTLY `0.8` and
 * the demanded turn rate as EXACTLY `1.6` rad/s. The literal is DELETED
 * and REPLACED one-for-one: the assertion form, the exactness (`toBe`)
 * and the UNIT clause are UNCHANGED and the pin is NEVER relaxed.
 */
const MOMENTUM_SERVICE_TURN_RATE_SPEED_FACTOR = 0.8;

/**
 * The 14S LOW-speed service lane (m/s).
 *
 * 25 m/s is a REAL drifting lane (above the 15 m/s MIN_SPEED gate), so
 * the lane is never satisfied VACUOUSLY.
 */
const MOMENTUM_SERVICE_LOW_SPEED_MPS = 25;

/**
 * Reads the NOT-YET-EXISTING demanded turn rate off a drift input.
 *
 * The field `demandedTurnRateRadPerS` is the RED-4 contract and does not
 * exist on `DriftStepInput` yet, so the read goes through this LOCAL
 * widened view (the `tractionSeverityOf` pattern of the 14Q block): the
 * field is `undefined` today, which is what makes the RED-4 contracts
 * fail on their OWN assertion instead of on a file-level compile error
 * that would mask every sibling contract in this file.
 *
 * @param driftInput - The drift input the live tick fed the commitment.
 * @returns The demanded turn rate (rad/s), or `undefined` while the live
 *   tick feeds the commitment no turn rate at all.
 */
const demandedTurnRateOf = (driftInput: DriftStepInput): number | undefined =>
  (
    driftInput as unknown as {
      demandedTurnRateRadPerS?: number;
    }
  ).demandedTurnRateRadPerS;

/**
 * Runs a service lane and CAPTURES the drift input the live tick feeds
 * the drift commitment.
 *
 * THE RED-4 INSTRUMENT: the live tick is the ONLY place where the
 * service and the drift law meet, so the contract reads the input AT THE
 * SEAM (the service is spied on, the spy calls through, so the lane
 * behaves exactly as it does in production) instead of asserting on a
 * re-implementation of the service resolution.
 *
 * @param ticks - The number of 60 Hz ticks to run.
 * @param input - The constant control sample applied on every tick.
 * @param speed - The tick-start speed of the lane (m/s).
 * @returns The captured first-tick drift input, the first-tick published
 *   heading delta (rad) and the published slip of the last tick (rad).
 */
const runMomentumServiceLane = (
  ticks: number,
  input: PhysicsStepInput = FIXTURE_INPUT,
  speed: number = NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
): {
  firstDriftInput: DriftStepInput;
  firstHeadingDeltaRad: number;
  publishedSlipRad: number;
} => {
  const start: PhysicsCarState = {
    ...createInitialPhysicsCarState(42),
    speed,
    drift: {
      committed: { slip: 0, turnSign: 0, reArmed: true },
      pose: { pose: 0, poseRate: 0, articulation: 0 },
    },
  };
  const surfaceResolver = (): FixtureSurface => 'road';
  const spy = jest.spyOn(driftModelModule, 'stepDriftCommitment');
  try {
    let state = start;
    let firstHeadingDeltaRad = Number.NaN;
    for (let tick = 0; tick < ticks; tick += 1) {
      const next = stepPhysics(state, input, surfaceResolver);
      if (tick === 0) {
        firstHeadingDeltaRad = next.heading - state.heading;
      }
      state = next;
    }
    const firstCall = spy.mock.calls[0];
    return {
      firstDriftInput: firstCall
        ? firstCall[1]
        : ({} as unknown as DriftStepInput),
      firstHeadingDeltaRad,
      publishedSlipRad: state.slipAngle,
    };
  } finally {
    spy.mockRestore();
  }
};

/**
 * The LIVE-tick momentum contracts (Step 14S, slice `P1-momentum-red`;
 * the plan's RED-4 acceptance criterion).
 *
 * AC-091: the drift is CAUSED by the MOMENTUM. The live tick must
 * APPLY the momentum law end to end: the service RESOLVES the demanded
 * turn rate ONCE per tick, the SAME resolved number feeds BOTH the yaw
 * and the momentum law, and the drift commitment call site RECEIVES that
 * number - so the law is APPLIED, not merely exported.
 *
 * THE MEASURED current-model state: the drift commitment is fed EXACTLY
 * SIX fields (the steer, the speed, the speed percent, the dt, the
 * lateral velocity and the slip angle - physics.step.service.ts:458-464)
 * and the yaw resolves its OWN per-tick delta in a SEPARATE place
 * (`resolveYawRatePerTick`, physics.step.service.ts:700-706), so the two
 * do NOT share one resolved number today and the drift NEVER sees a turn
 * rate.
 *
 * THE "a STRICTLY LARGER demanded turn rate produces a STRICTLY DEEPER
 * slide" relation is NOT asserted HERE: it is OWNED BY RED-2 (the MODEL
 * lane), where the WIDENED internal drift input makes it exercisable and
 * red - at the service level the demanded turn rate is DERIVED from the
 * steer command and the speed percent, so that lane would be VACUOUS.
 *
 * THE PINNED UNIT (the RED-4 unit clause): the demanded turn rate is a
 * RATE in rad/s - the service resolves `steer * STEERING_RATE *
 * speedPercent` ONCE per tick, the momentum law consumes it, and the yaw
 * per-tick delta is DERIVED from the SAME resolved number (multiplied by
 * the fixed timestep), so the yaw behaviour stays BIT-IDENTICAL. A
 * per-tick delta and a rate are NOT interchangeable: a unit mismatch is
 * a silent factor-of-sixty error.
 */
describe('the LIVE tick APPLIES the momentum (Step 14S, slice P1-momentum-red)', () => {
  describe('RED-4 (the live tick APPLIES the momentum)', () => {
    // [RED-14S-4-DEMANDED-RATE-RECEIVED]
    it('AC-091: the live tick feeds the drift commitment the demanded turn rate it resolved for this tick (RED-4)', () => {
      // Arrange
      const lane = runMomentumServiceLane(MOMENTUM_SERVICE_WINDOW_TICKS);

      // Act + Assert
      // The NON-VACUITY GUARD comes FIRST: the lane must REALLY slide (a
      // published slip of 0 would satisfy the "the law is APPLIED" clause
      // VACUOUSLY).
      expect(Math.abs(lane.publishedSlipRad)).toBeGreaterThan(0);
      // The drift commitment call site RECEIVES the resolved demanded
      // turn rate. The MEASURED current-model state: the commitment is
      // fed EXACTLY SIX fields and NO turn rate, so the read is
      // `undefined` today.
      expect(demandedTurnRateOf(lane.firstDriftInput)).toEqual(
        expect.any(Number),
      );
    });

    // [RED-14S-4-ONE-RESOLVED-NUMBER]
    it('AC-091: the published heading delta is DERIVED from the SAME resolved demanded turn rate the drift law consumes (RED-4)', () => {
      // Arrange
      const lane = runMomentumServiceLane(MOMENTUM_SERVICE_WINDOW_TICKS);

      // Act + Assert
      // The NON-VACUITY GUARD comes FIRST: the lane must REALLY turn (a
      // heading delta of 0 would satisfy the relation VACUOUSLY).
      expect(lane.firstHeadingDeltaRad).toBeGreaterThan(0);
      // ONE resolved number: the published heading delta of the SAME
      // tick IS the resolved demanded turn rate resolved with the fixed
      // timestep - the yaw delta is DERIVED from the number the drift law
      // consumes, never a second re-derivation.
      //
      // The MEASURED current-model state: the drift never sees a turn
      // rate, so the relation has no number to hold.
      expect(lane.firstHeadingDeltaRad).toBeCloseTo(
        (demandedTurnRateOf(lane.firstDriftInput) ?? Number.NaN) *
          NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
        12,
      );
    });

    // [RED-14S-4-PINNED-UNIT]
    it('AC-091: the demanded turn rate the drift law consumes is a RATE (rad/s), not the per-tick heading delta (RED-4)', () => {
      // Arrange
      const lane = runMomentumServiceLane(MOMENTUM_SERVICE_WINDOW_TICKS);

      // Act + Assert
      // THE PINNED UNIT AND ITS BASIS: the service resolves the demanded
      // turn rate ONCE per tick as `steer * STEERING_RATE * speedPercent`
      // softened by the 14T mild speed factor (rad/s) - the SAME product
      // the yaw law already uses, WITHOUT the `dt` factor. The per-tick
      // heading delta is that number multiplied by the fixed timestep; a
      // per-tick delta is NOT a rate.
      //
      // The MEASURED post-14T state: the demanded turn rate of the
      // top-speed lane reads EXACTLY 1.6 rad/s (the OLD baseline was the
      // UN-SOFTENED 2 rad/s), because the 14T USER law applies the mild
      // speed-proportional factor LAST and the top-speed lane resolves
      // that factor as EXACTLY 0.8. The literal is DELETED and REPLACED
      // one-for-one (the AC-089 ledger row P-T-1): the assertion form,
      // the exactness and the UNIT clause are UNCHANGED.
      expect(demandedTurnRateOf(lane.firstDriftInput)).toBe(
        MOMENTUM_SERVICE_STEER_AXIS *
          NEATRACING_PHYSICS_STEERING_RATE *
          MOMENTUM_SERVICE_SPEED_PERCENT *
          MOMENTUM_SERVICE_TURN_RATE_SPEED_FACTOR,
      );
    });

    // [RED-14S-4-SPEED-DIMENSION-GUARD]
    it('AC-091: the published slip for the SAME steer command is STRICTLY DEEPER at a high speed than at a low speed over a fixed tick window (RED-4) - PRESERVATION GUARD', () => {
      // Arrange
      const highSpeedLane = runMomentumServiceLane(
        MOMENTUM_SERVICE_WINDOW_TICKS,
        FIXTURE_INPUT,
        NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
      );
      const lowSpeedLane = runMomentumServiceLane(
        MOMENTUM_SERVICE_WINDOW_TICKS,
        FIXTURE_INPUT,
        MOMENTUM_SERVICE_LOW_SPEED_MPS,
      );

      // Act + Assert
      // PRESERVATION GUARD, NOT A RED: the speed dimension ALREADY holds
      // today (the speed-proportional entry rate inside the
      // speed-proportional depth budget, clamped at drift.model.ts:817),
      // so this clause is the RAIL the new cause must stay on.
      // The NON-VACUITY GUARD comes FIRST: BOTH lanes must REALLY slide.
      expect(Math.abs(highSpeedLane.publishedSlipRad)).toBeGreaterThan(0);
      expect(Math.abs(lowSpeedLane.publishedSlipRad)).toBeGreaterThan(0);
      expect(Math.abs(highSpeedLane.publishedSlipRad)).toBeGreaterThan(
        Math.abs(lowSpeedLane.publishedSlipRad),
      );
    });

    // [RED-14S-4-STANDSTILL-GUARD]
    it('AC-091: the published slip is EXACTLY 0 at a standstill (RED-4) - PRESERVATION GUARD', () => {
      // Arrange - a standstill lane: the car never moves (no throttle),
      // so the speed stays EXACTLY 0 for the whole window.
      const standstillLane = runMomentumServiceLane(
        MOMENTUM_SERVICE_WINDOW_TICKS,
        LOW_SPEED_STEER_INPUT,
        0,
      );

      // Act + Assert
      // PRESERVATION GUARD, NOT A RED: there is NO momentum at rest, and
      // the MEASURED current-model state already publishes an EXACTLY 0
      // slip at a standstill.
      expect(standstillLane.publishedSlipRad).toBe(0);
    });

    // [RED-14S-4-EXACT-ZEROS-GUARD]
    it('AC-091: the straight-line exact zeros and the low-speed exact-zero gate stay green (RED-4) - PRESERVATION GUARD', () => {
      // Arrange
      const straightLane = runMomentumServiceLane(
        MOMENTUM_SERVICE_WINDOW_TICKS,
        STRAIGHT_INPUT,
        NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
      );
      const lowSpeedGateLane = runMomentumServiceLane(120, LOW_SPEED_STEER_INPUT, 10);

      // Act + Assert
      // PRESERVATION GUARDS, NOT REDS: the straight-line EXACT zeros and
      // the low-speed EXACT-zero gate (below the 15 m/s MIN_SPEED gate the
      // VISIBLE slide stays EXACTLY 0) are the ratified AC-047D / AC-079
      // behaviour and the momentum cause must not touch them.
      expect(straightLane.publishedSlipRad).toBe(0);
      expect(lowSpeedGateLane.publishedSlipRad).toBe(0);
    });
  });
});

/* ------------------------------------------------------------------------- *
 * STEP 14T (slice `P1-turnrate-red`): the TURN-RATE laws.
 * ------------------------------------------------------------------------- */

/**
 * The RED-3 WIDENED LOCAL VIEW of the physics step input.
 *
 * The turn-rate-percent channel is a TYPE MEMBER of `PhysicsStepInput`
 * (and NOT a module export), so it CANNOT be named by an import: the
 * RED shape is a named RUNTIME failure through this local view (the
 * Step 14R red-slice residual (2) lesson) - a MISSING-IMPORT failure
 * would be a file-level compile error that would mask EVERY sibling
 * contract in this file.
 */
interface TurnRatePercentInput extends PhysicsStepInput {
  /** The OPTIONAL turn-rate percent channel (Step 14T, AC-093). */
  turnRatePercent?: number;
}

/**
 * The 14T lane window (60 Hz ticks).
 *
 * The turn-rate lanes are measured over the SAME fixed window, so the
 * window is a named constant and never a carried literal (the Step 14S
 * window precedent).
 */
const TURN_RATE_LANE_TICKS = 30;

/**
 * The 14T TOP-SPEED lane (m/s).
 *
 * The arcade top speed (89 m/s) is the lane whose speed percent the
 * service resolves as EXACTLY 1, so it is the tightest comparison
 * against the nominal rate.
 */
const TURN_RATE_TOP_SPEED_MPS = NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S;

/**
 * The 14T CRAWL lane (m/s).
 *
 * MEASURED: the slowest REAL drifting lane of the current build. The
 * measured momentum crossover (the slowest speed at which the slide
 * commitment opens) sits at about 9.43 m/s - the 10 m/s crawl lane is
 * just above it, so the crawl lane is a REAL lane and never satisfied
 * vacuously.
 */
const TURN_RATE_CRAWL_SPEED_MPS = 10;

/**
 * The 14T speed ladder (m/s).
 *
 * The ladder the monotonicity rail walks: the crawl lane, a drifting
 * lane, two mid lanes and the top speed.
 */
const TURN_RATE_SPEED_LADDER_MPS = [10, 25, 45, 67, 89];

/**
 * The 14T NOMINAL turn rate (rad/s) - the COMPARISON AXIS.
 *
 * The nominal rate is the steer command times the steering rate
 * (`steer * NEATRACING_PHYSICS_STEERING_RATE`): the rate the mild law
 * reduces at the top speed. It is NEVER the low-speed rate (AC-092
 * keeps the existing speed-percent scaling, so the high-speed rate
 * stays ABOVE the low-speed one).
 */
const TURN_RATE_NOMINAL_RAD_PER_S = NEATRACING_PHYSICS_STEERING_RATE;

/**
 * The 14T MILD floor (the fraction of the nominal rate the top-speed
 * turn rate may never fall below).
 *
 * MEASURED (the RED decision, the Step 14Q gain-window precedent): the
 * reduction may never take a QUARTER of the nominal rate away, so the
 * top-speed turn rate stays at or above `0.75 * 2 = 1.5` rad/s. The
 * floor is a FEEL bound and it is CROSS-CHECKED against the measured
 * physics: at the floor the 89 m/s full-steer lane still commits a
 * slide of `0.44325842696629214` rad (98.5 % of the 0.45 rad depth
 * budget), so the floor binds the FEEL and NEVER the physics.
 */
const TURN_RATE_MILD_FLOOR_FRACTION = 0.75;

/**
 * The 14T CRAWL band floor (the fraction of the nominal law the crawl
 * turn rate may never fall below).
 *
 * MEASURED: the crawl slide commitment needs `speed * |rate|` above the
 * grip hold, so the crawl factor may never fall below
 * `0.2 / 0.2247191011235955 = 0.89006629` (the measured feasibility
 * floor - below it the 10 m/s commitment VANISHES and the crawl would
 * silently stop drifting). The band floor 0.9 sits just ABOVE that
 * measured floor, so the crawl band keeps the crawl drift alive while
 * the crawl rate stays inside the nominal law.
 */
const TURN_RATE_CRAWL_BAND_FLOOR_FRACTION = 0.9;

/**
 * The 14T top-speed committed-slide floor (rad).
 *
 * The ratified AC-047A floor (a committed slide above 0.2 rad at the
 * drifting speeds): the mild reduction may never take the top-speed
 * slide below it, so the drift is never disabled (the S-6 clause).
 */
const TURN_RATE_TOP_SPEED_SLIDE_FLOOR_RAD = 0.2;

/**
 * The float tolerance of the 14T band edges (rad/s).
 *
 * The commanded rate is MEASURED through the published heading delta
 * (the delta divided by the fixed timestep), so a reading carries a
 * float64 rounding of a few ULP. The tolerance absorbs that noise
 * ONLY - it is twelve orders of magnitude below the mild band.
 */
const TURN_RATE_FLOAT_EPSILON = 1e-12;

/**
 * The 14T USER lane input: the full LEFT arrow at full throttle.
 *
 * The user channel (the arrow keys) is the lane the mild law softens;
 * the AI channel (the turn-rate percent) is a SEPARATE law.
 */
const TURN_RATE_USER_INPUT: TurnRatePercentInput = {
  throttle: 1,
  brake: 0,
  steer: 1,
};

/**
 * The NOMINAL LAW rate of a speed (rad/s): the steer command times the
 * steering rate times the speed percent - the law the mild factor
 * reduces.
 *
 * @param speedMps - The tick-start speed of the lane (m/s).
 * @returns The nominal-law turn rate of that speed (rad/s).
 */
const nominalLawRateOf = (speedMps: number): number =>
  TURN_RATE_NOMINAL_RAD_PER_S *
  (speedMps / NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S);

/**
 * ONE 14T lane (the measured reading of a single lane).
 */
interface TurnRateLane {
  /** The commanded turn rate of the FIRST tick (rad/s). */
  commandedRateRadPerS: number;
  /** The demanded turn rate the live tick fed the drift commitment (rad/s). */
  demandedTurnRateRadPerS: number | undefined;
  /** The steer axis the live tick fed the drift commitment. */
  driftSteerAxis: number | undefined;
  /** The committed slide of the LAST tick (rad). */
  committedSlipRad: number;
  /** The published slip of the LAST tick (rad). */
  publishedSlipRad: number;
}

/**
 * Runs a 14T turn-rate lane and MEASURES the commanded turn rate.
 *
 * THE MEASUREMENT (the observable): the commanded turn rate is read
 * from the PUBLISHED heading delta of the FIRST tick (the delta divided
 * by the fixed timestep), so the lane measures the turn the car
 * actually got - never a re-implementation of the service law. The
 * FIRST tick is the only one that carries the lane speed, so the lane
 * speed IS the tick-start speed.
 *
 * THE SEAM (the drift input): the drift commitment is spied on (the
 * Step 14S helper pattern), so the lane can read the steer axis the
 * drift model receives and the demanded turn rate the live tick
 * resolved.
 *
 * @param speed - The tick-start speed of the lane (m/s).
 * @param input - The constant control sample applied on every tick.
 * @param ticks - The number of 60 Hz ticks to run.
 * @returns The measured reading of the lane.
 */
const runTurnRateLane = (
  speed: number,
  input: TurnRatePercentInput,
  ticks: number = TURN_RATE_LANE_TICKS,
): TurnRateLane => {
  const start: PhysicsCarState = {
    ...createInitialPhysicsCarState(42),
    speed,
    drift: {
      committed: { slip: 0, turnSign: 0, reArmed: true },
      pose: { pose: 0, poseRate: 0, articulation: 0 },
    },
  };
  const surfaceResolver = (): FixtureSurface => 'road';
  const spy = jest.spyOn(driftModelModule, 'stepDriftCommitment');
  try {
    let state = start;
    let commandedRateRadPerS = Number.NaN;
    for (let tick = 0; tick < ticks; tick += 1) {
      const next = stepPhysics(state, input, surfaceResolver);
      if (tick === 0) {
        commandedRateRadPerS =
          (next.heading - state.heading) /
          NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS;
      }
      state = next;
    }
    const firstCall = spy.mock.calls[0];
    const driftInput = firstCall
      ? firstCall[1]
      : ({} as unknown as DriftStepInput);
    return {
      commandedRateRadPerS,
      demandedTurnRateRadPerS: demandedTurnRateOf(driftInput),
      driftSteerAxis: driftInput.steer,
      committedSlipRad: state.drift.committed.slip,
      publishedSlipRad: state.slipAngle,
    };
  } finally {
    spy.mockRestore();
  }
};

/**
 * The TURN-RATE contracts (Step 14T, slice `P1-turnrate-red`).
 *
 * AC-092 (the turn rate follows the speed, S-6): the commanded turn
 * rate REDUCES slightly and PROPORTIONALLY with the speed - the factor
 * is 1 at a standstill and the MEASURED mild value at the top speed.
 * Outside the Step 14Y sufficiency boost band the mild law is unchanged;
 * inside the band the factor is lifted to hold the tightest corner. The
 * mild relation is MILD (the top-speed turn rate is never 0 and never below
 * the RED-measured fraction of the nominal rate, so turning AND drifting
 * remain possible at every speed). The COMPARISON AXIS IS THE NOMINAL
 * RATE, NEVER THE LOW-SPEED RATE: the existing speed-percent scaling is
 * KEPT, so the high-speed turn rate stays ABOVE the low-speed one.
 *
 * AC-093 (two separate laws on the same steer axis, S-7): the physics
 * input grows an OPTIONAL named turn-rate-percent channel - the AI
 * channel. When it is PRESENT the commanded turn rate is resolved from
 * that PERCENT of the nominal turn rate, INDEPENDENT of the steer axis;
 * when it is ABSENT the existing steer-axis law applies (the
 * backward-compatible default, and the USER channel).
 *
 * THE RED SET IS SPLIT INTO DISCRIMINATING REDS AND PRESERVATION
 * GUARDS: RED-1 (the top-speed user turn rate is STRICTLY LOWER than
 * the nominal), RED-3 (the percent channel does not exist) and RED-4
 * (in `ai.autopilot.test.ts`: the AI sets a DEFINED percent and READS
 * the snapshot speed) are RED today, while RED-2 is a LABELLED
 * PRESERVATION GUARD that holds today and is NOT counted as a red.
 * EVERY relational lane opens with a NON-VACUITY GUARD asserted BEFORE
 * the relational clause, because a zero yaw rate would satisfy a
 * strictly-lower clause VACUOUSLY.
 */
describe('the TURN-RATE laws (Step 14T, slice P1-turnrate-red)', () => {
  describe('RED-1 (the turn rate follows the speed, S-6)', () => {
    // [RED-14T-1-TOP-SPEED-STRICTLY-LOWER]
    it('AC-092: the top-speed USER turn rate is STRICTLY LOWER than the NOMINAL turn rate (RED-1)', () => {
      // Arrange
      const lane = runTurnRateLane(TURN_RATE_TOP_SPEED_MPS, TURN_RATE_USER_INPUT);

      // Act + Assert
      // The NON-VACUITY GUARD comes FIRST: the lane must REALLY turn (a
      // zero yaw rate would satisfy the strictly-lower clause
      // VACUOUSLY).
      expect(lane.commandedRateRadPerS).toBeGreaterThan(0);
      // THE DISCRIMINATING CLAUSE - the ONE fact that is FALSE today.
      //
      // The MEASURED current-model state: the top-speed turn rate IS the
      // nominal rate (the speed percent saturates at 1), so the user law
      // S-6 is INVERTED - the rate grows with the speed instead of
      // reducing. The MEASURED reading is the EXACT nominal 2 rad/s (the
      // (delta / dt) round trip is exact for this lane), and the mild
      // band floor leaves a margin of 0.5 rad/s - twelve orders of
      // magnitude above any float64 noise - so the clause stays RED
      // until the law really reduces the rate.
      expect(lane.commandedRateRadPerS).toBeLessThan(TURN_RATE_NOMINAL_RAD_PER_S);
    });

    // [RED-14T-1-MILD-FLOOR]
    it('AC-092: the top-speed USER turn rate stays MILD - never below the RED-measured fraction of the nominal rate (RED-1)', () => {
      // Arrange
      const lane = runTurnRateLane(TURN_RATE_TOP_SPEED_MPS, TURN_RATE_USER_INPUT);

      // Act + Assert
      // The NON-VACUITY GUARD comes FIRST: the lane must REALLY turn.
      expect(lane.commandedRateRadPerS).toBeGreaterThan(0);
      // The MILD rail (a RAIL, not the discriminating clause): the
      // reduction may never take a quarter of the nominal rate away, so
      // turning AND drifting stay possible at every speed (the user law:
      // never so much that it prevents turning or drifting).
      //
      // The MEASURED current-model state: the top-speed rate IS the
      // nominal one, so the rail is GREEN on arrival - it is the floor
      // the new law may never cross.
      expect(lane.commandedRateRadPerS).toBeGreaterThanOrEqual(
        TURN_RATE_MILD_FLOOR_FRACTION * TURN_RATE_NOMINAL_RAD_PER_S,
      );
    });

    // [RED-14T-1-FACTOR-MONOTONIC]
    it('AC-092: the speed-proportional reduction FACTOR stays NON-INCREASING outside the sufficiency boost band (RED-1)', () => {
      // Arrange
      const ladder = TURN_RATE_SPEED_LADDER_MPS.map((speed) => {
        const lane = runTurnRateLane(speed, TURN_RATE_USER_INPUT);
        return {
          speed,
          rateRadPerS: lane.commandedRateRadPerS,
          factor: lane.commandedRateRadPerS / nominalLawRateOf(speed),
        };
      });

      // Act + Assert
      // The NON-VACUITY GUARD comes FIRST: EVERY ladder lane must
      // REALLY turn (a zero yaw rate would make the factor undefined
      // and the relation vacuous).
      expect(
        Math.min(...ladder.map((entry) => entry.rateRadPerS)),
      ).toBeGreaterThan(0);
      // The factor NEVER grows with the speed across the ladder (the mild
      // law only reduces the rate, and it reduces it MORE the faster the
      // car goes). Every ladder speed lies OUTSIDE the Step 14Y
      // sufficiency boost band (~[17.6, 18.0] m/s), so this lane scopes
      // the monotonicity claim to the region where the mild law still
      // applies; the global monotonicity claim is retired because the
      // sufficiency boost lifts the factor inside the band.
      //
      // The MEASURED post-fix state: the factor does not grow across the
      // ladder (within a float64 ULP) - the clause is GREEN on the
      // post-fix bytes.
      expect(
        Math.max(
          ...ladder
            .slice(1)
            .map((entry, index) => entry.factor - ladder[index].factor),
        ),
      ).toBeLessThanOrEqual(TURN_RATE_FLOAT_EPSILON);
    });

    // [RED-14T-1-HIGH-ABOVE-LOW]
    it('AC-092: the high-speed USER turn rate stays ABOVE the low-speed one (the speed-percent scaling is kept) (RED-1)', () => {
      // Arrange
      const topLane = runTurnRateLane(
        TURN_RATE_TOP_SPEED_MPS,
        TURN_RATE_USER_INPUT,
      );
      const crawlLane = runTurnRateLane(
        TURN_RATE_CRAWL_SPEED_MPS,
        TURN_RATE_USER_INPUT,
      );

      // Act + Assert
      // The NON-VACUITY GUARD comes FIRST: BOTH lanes must REALLY turn.
      expect(topLane.commandedRateRadPerS).toBeGreaterThan(0);
      expect(crawlLane.commandedRateRadPerS).toBeGreaterThan(0);
      // The COMPARISON AXIS of the mild law is the NOMINAL rate, NEVER
      // the low-speed rate: the existing speed-percent scaling stays, so
      // the high-speed rate keeps climbing above the crawl one.
      expect(topLane.commandedRateRadPerS).toBeGreaterThan(
        crawlLane.commandedRateRadPerS,
      );
    });

    // [RED-14T-1-CRAWL-BAND]
    it('AC-092: the near-standstill USER turn rate stays at the existing nominal law inside the RED-measured band (RED-1)', () => {
      // Arrange
      const lane = runTurnRateLane(
        TURN_RATE_CRAWL_SPEED_MPS,
        TURN_RATE_USER_INPUT,
      );
      const crawlNominalLawRate = nominalLawRateOf(TURN_RATE_CRAWL_SPEED_MPS);

      // Act + Assert
      // The NON-VACUITY GUARD comes FIRST: the crawl lane must REALLY
      // turn (a zero yaw rate would satisfy the band vacuously).
      expect(lane.commandedRateRadPerS).toBeGreaterThan(0);
      // The CRAWL BAND (the RED-measured window): the crawl rate stays
      // at the existing nominal law (never ABOVE it - the law only
      // reduces) and never below the measured crawl floor (the crawl
      // handling is NOT degraded and the crawl drift stays alive).
      //
      // The MEASURED current-model state: the crawl rate IS the nominal
      // law (within a float64 ULP), so the band holds on arrival.
      expect(lane.commandedRateRadPerS).toBeLessThanOrEqual(
        crawlNominalLawRate + TURN_RATE_FLOAT_EPSILON,
      );
      expect(lane.commandedRateRadPerS).toBeGreaterThanOrEqual(
        TURN_RATE_CRAWL_BAND_FLOOR_FRACTION * crawlNominalLawRate -
          TURN_RATE_FLOAT_EPSILON,
      );
    });
  });

  describe('RED-2 (the user channel is softened, the drift is not disabled) - PRESERVATION GUARD', () => {
    // [RED-14T-2-STEER-AXIS-BINARY]
    it('AC-093: the steer AXIS the drift model reads stays BINARY (+/- 1) - the 0.5 / 0.15 / 0.9 gates keep their meaning (RED-2) - PRESERVATION GUARD', () => {
      // Arrange
      const leftLane = runTurnRateLane(
        TURN_RATE_TOP_SPEED_MPS,
        TURN_RATE_USER_INPUT,
      );
      const rightLane = runTurnRateLane(TURN_RATE_TOP_SPEED_MPS, {
        throttle: 1,
        brake: 0,
        steer: -1,
      });

      // Act + Assert
      // PRESERVATION GUARD, NOT A RED: this lane holds today, so it is
      // NOT counted as a red - it is the RAIL the mild law must stay on.
      // The NON-VACUITY GUARD comes FIRST: the lane must REALLY turn.
      expect(leftLane.commandedRateRadPerS).toBeGreaterThan(0);
      // The steer AXIS is the BINARY arrow value the drift model reads:
      // the mild law softens the commanded RATE, never the axis, so the
      // ratified engage (0.5), neutral (0.15) and countersteer (0.9)
      // gates keep their EXACT meaning and the drift is never silently
      // disabled.
      expect(leftLane.driftSteerAxis).toBe(1);
      expect(rightLane.driftSteerAxis).toBe(-1);
    });

    // [RED-14T-2-TOP-SPEED-SLIDE-COMMITS]
    it('AC-092: the top-speed USER turn rate STILL produces a committed slide (the drift is not disabled) (RED-2) - PRESERVATION GUARD', () => {
      // Arrange
      const lane = runTurnRateLane(
        TURN_RATE_TOP_SPEED_MPS,
        TURN_RATE_USER_INPUT,
        TURN_RATE_LANE_TICKS,
      );

      // Act + Assert
      // PRESERVATION GUARD, NOT A RED: the MEASURED current-model state
      // already commits a slide at the top speed (the committed depth
      // reads `0.44737300755695186` rad after the 30-tick window), so
      // this lane is GREEN on arrival - it is the RAIL the mild law must
      // stay on (the S-6 clause: never so much that it prevents turning
      // or drifting).
      // The NON-VACUITY GUARD comes FIRST: the lane must REALLY slide.
      expect(Math.abs(lane.publishedSlipRad)).toBeGreaterThan(0);
      // The committed slide stays at or above the ratified AC-047A floor
      // (0.2 rad at the drifting speeds), so the mild reduction may
      // never take the top-speed drift below it.
      expect(Math.abs(lane.committedSlipRad)).toBeGreaterThanOrEqual(
        TURN_RATE_TOP_SPEED_SLIDE_FLOOR_RAD,
      );
    });
  });

  describe('RED-3 (the AI turn-rate-percent channel, S-7)', () => {
    // [RED-14T-3-PERCENT-100-IS-NOMINAL]
    it('AC-093: a PRESENT turn-rate percent of 100 commands the NOMINAL turn rate - not the speed-scaled steer law (RED-3)', () => {
      // Arrange
      const lane = runTurnRateLane(TURN_RATE_CRAWL_SPEED_MPS, {
        throttle: 1,
        brake: 0,
        steer: 1,
        turnRatePercent: 100,
      });

      // Act + Assert
      // The NON-VACUITY GUARD comes FIRST: the lane must REALLY turn (a
      // zero yaw rate would satisfy the percent clause VACUOUSLY).
      expect(lane.commandedRateRadPerS).toBeGreaterThan(0);
      // THE DISCRIMINATING CLAUSE: 100 percent means the NOMINAL rate
      // (the steer command times the steering rate), INDEPENDENT of the
      // speed percent.
      //
      // The MEASURED current-model state: the percent channel DOES NOT
      // EXIST, so the crawl lane still runs the steer-axis law
      // (`0.2247191011235955` rad/s) and the clause is RED today.
      expect(lane.commandedRateRadPerS).toBeCloseTo(
        TURN_RATE_NOMINAL_RAD_PER_S,
        12,
      );
    });

    // [RED-14T-3-PERCENT-150-ABOVE-NOMINAL]
    it('AC-093: a turn-rate percent ABOVE 100 commands a rate ABOVE the nominal rate (the AI keeps full control) (RED-3)', () => {
      // Arrange
      const lane = runTurnRateLane(TURN_RATE_CRAWL_SPEED_MPS, {
        throttle: 1,
        brake: 0,
        steer: 1,
        turnRatePercent: 150,
      });

      // Act + Assert
      // The NON-VACUITY GUARD comes FIRST: the lane must REALLY turn.
      expect(lane.commandedRateRadPerS).toBeGreaterThan(0);
      // THE DISCRIMINATING CLAUSE: a percent above 100 commands a rate
      // above the nominal rate, so the AI can push its turn past the
      // nominal.
      //
      // The MEASURED current-model state: the percent channel DOES NOT
      // EXIST, so the crawl lane still runs the steer-axis law
      // (`0.2247191011235955` rad/s) and the clause is RED today.
      expect(lane.commandedRateRadPerS).toBeGreaterThan(
        TURN_RATE_NOMINAL_RAD_PER_S,
      );
    });

    // [RED-14T-3-PERCENT-0-NO-TURN]
    it('AC-093: a turn-rate percent of 0 commands NO turn at all (RED-3)', () => {
      // Arrange
      const controlLane = runTurnRateLane(
        TURN_RATE_CRAWL_SPEED_MPS,
        TURN_RATE_USER_INPUT,
      );
      const lane = runTurnRateLane(TURN_RATE_CRAWL_SPEED_MPS, {
        throttle: 1,
        brake: 0,
        steer: 1,
        turnRatePercent: 0,
      });

      // Act + Assert
      // The NON-VACUITY GUARD comes FIRST: the CONTROL lane (the same
      // steer WITHOUT the percent) must REALLY turn - otherwise the
      // zero-turn clause would be satisfied by a car that cannot turn
      // at all.
      expect(controlLane.commandedRateRadPerS).toBeGreaterThan(0);
      // THE DISCRIMINATING CLAUSE: 0 percent means NO turn - the
      // commanded rate is EXACTLY 0.
      //
      // The MEASURED current-model state: the percent channel DOES NOT
      // EXIST, so the steer axis still commands a turn
      // (`0.2247191011235955` rad/s) and the clause is RED today.
      expect(lane.commandedRateRadPerS).toBe(0);
    });

    // [RED-14T-3-PERCENT-STEER-INDEPENDENT]
    it('AC-093: the turn-rate percent resolves the SAME rate for either steer sign (the percent channel is INDEPENDENT of the steer axis) (RED-3)', () => {
      // Arrange
      const leftLane = runTurnRateLane(TURN_RATE_CRAWL_SPEED_MPS, {
        throttle: 1,
        brake: 0,
        steer: 1,
        turnRatePercent: 100,
      });
      const rightLane = runTurnRateLane(TURN_RATE_CRAWL_SPEED_MPS, {
        throttle: 1,
        brake: 0,
        steer: -1,
        turnRatePercent: 100,
      });

      // Act + Assert
      // The NON-VACUITY GUARD comes FIRST and is SIGN-AGNOSTIC: BOTH
      // lanes must REALLY turn (a zero yaw rate would satisfy the
      // relation VACUOUSLY). The guard is NEVER per-lane signed here,
      // because the direction travels with the PERCENT and never with
      // the steer axis - the same signed percent of 100 commands the
      // same signed nominal rate on BOTH lanes.
      expect(Math.abs(leftLane.commandedRateRadPerS)).toBeGreaterThan(0);
      expect(Math.abs(rightLane.commandedRateRadPerS)).toBeGreaterThan(0);
      // THE DISCRIMINATING CLAUSE: the percent channel is INDEPENDENT of
      // the steer axis - the SAME percent commands the SAME rate for
      // either steer sign (the percent is SIGNED, so the direction
      // travels with the percent and never with the axis).
      //
      // The MEASURED pre-14T state: the percent channel DOES NOT EXIST,
      // so the steer axis still commands `+0.2247191011235955` /
      // `-0.2247191011235955` rad/s and the clause is RED.
      expect(rightLane.commandedRateRadPerS).toBeCloseTo(
        leftLane.commandedRateRadPerS,
        12,
      );
      // THE DIRECTION-TRAVEL CLAUSES: the SIGNED percent of 100
      // commands the SIGNED NOMINAL rate on BOTH lanes - the mirrored
      // steer axis does NOT flip the commanded direction, because the
      // sign travels with the PERCENT.
      expect(leftLane.commandedRateRadPerS).toBeCloseTo(
        TURN_RATE_NOMINAL_RAD_PER_S,
        12,
      );
      expect(rightLane.commandedRateRadPerS).toBeCloseTo(
        TURN_RATE_NOMINAL_RAD_PER_S,
        12,
      );
    });

    // [RED-14T-3-PERCENT-100-NOT-SOFTENED]
    it('AC-093: the turn-rate percent channel is NEVER softened by the user adjustment (the two laws never collide) (RED-3) - PRESERVATION GUARD', () => {
      // Arrange
      const lane = runTurnRateLane(TURN_RATE_TOP_SPEED_MPS, {
        throttle: 1,
        brake: 0,
        steer: 1,
        turnRatePercent: 100,
      });

      // Act + Assert
      // PRESERVATION GUARD, NOT A RED: the MEASURED current-model state
      // reads the nominal rate at the top speed (the steer-axis law
      // saturates at 1 there), so this lane is GREEN on arrival - it is
      // the RAIL that pins the SEPARATION: a SOFTENED percent channel
      // would read BELOW the nominal rate at the top speed.
      // The NON-VACUITY GUARD comes FIRST: the lane must REALLY turn.
      expect(lane.commandedRateRadPerS).toBeGreaterThan(0);
      expect(lane.commandedRateRadPerS).toBeCloseTo(
        TURN_RATE_NOMINAL_RAD_PER_S,
        12,
      );
    });

    // [RED-14T-3-ABSENT-PERCENT-STEER-LAW]
    it('AC-093: an ABSENT turn-rate percent keeps the steer-axis law (the backward-compatible default, the USER channel) (RED-3) - PRESERVATION GUARD', () => {
      // Arrange
      const leftLane = runTurnRateLane(
        TURN_RATE_CRAWL_SPEED_MPS,
        TURN_RATE_USER_INPUT,
      );
      const rightLane = runTurnRateLane(TURN_RATE_CRAWL_SPEED_MPS, {
        throttle: 1,
        brake: 0,
        steer: -1,
      });

      // Act + Assert
      // PRESERVATION GUARD, NOT A RED: the MEASURED current-model state
      // runs the steer-axis law when no percent is present, so this lane
      // is GREEN on arrival.
      // The NON-VACUITY GUARD comes FIRST: BOTH lanes must REALLY turn
      // (the left lane turns LEFT, the right lane turns RIGHT, so each
      // lane is guarded by its OWN SIGN).
      expect(leftLane.commandedRateRadPerS).toBeGreaterThan(0);
      expect(rightLane.commandedRateRadPerS).toBeLessThan(0);
      // The steer-axis law is SYMMETRIC: the mirror steer commands the
      // mirror rate - the ABSENT percent falls back to the steer axis,
      // never to a 100 percent default.
      expect(rightLane.commandedRateRadPerS).toBeCloseTo(
        -leftLane.commandedRateRadPerS,
        12,
      );
    });
  });
});

/* ------------------------------------------------------------------------- *
 * STEP 14I (slice `P1-track-road-containment`): the AC-081 road-containment
 * seam in the LIVE tick - and its RETIREMENT.
 *
 * The AC-081 BOUND was LANDED (Step 14P) and WIRED (Step 14I) through the
 * FOURTH optional track-geometry parameter of `stepPhysics`. The AC-090
 * VERDICT-8 round (fix-packet-P1-sand-red-iteration-3) then RETIRED the
 * bound ENTIRELY: NO track-boundary containment at all - the road edge is
 * an INNER boundary, the ONLY off-track limit is the sand speed cap
 * (AC-094), and EVERY geometry input is a byte-identical NO-OP.
 *
 * This block is the RE-AUTHORED contract of the RETIREMENT (the retired
 * seam stays the ONE named owner of the shipped 4-arg form): the four
 * lanes measure (a) NO CONTAINMENT, (b) the ALL-inputs NO-OP, (c)
 * POSE-INDEPENDENCE (true by construction) and (d) PUBLISHED SCALARS
 * (the model law on the tick's own final state).
 * ------------------------------------------------------------------------- */

/**
 * The geometry-aware tick, read through a LOCAL widened view.
 *
 * The file MUST COMPILE (the rule of this file), so the NOT-YET-EXISTING
 * FOURTH track-geometry parameter of `stepPhysics` is read through a LOCAL
 * widened view - exactly the `stepPhysicsFleetOf` pattern of the 14R block.
 * A missing seam never masks a sibling contract of this file.
 */
const stepPhysicsWithGeometryOf = (
  physicsStepService as unknown as {
    stepPhysics?: (
      state: PhysicsCarState,
      input: PhysicsStepInput,
      surfaceResolver?: (carState: PhysicsCarState) => FixtureSurface,
      trackSegments?: readonly TrackSegment[],
    ) => PhysicsCarState;
  }
).stepPhysics;

/** The straight 8 m-wide drivable ribbon of the containment fixtures. */
const CONTAINMENT_RIBBON: readonly TrackSegment[] = [
  { startX: -50, startY: 0, endX: 50, endY: 0, width: 8 },
];

/** The drivable half width of the containment ribbon (m): 8 m / 2. */
const CONTAINMENT_DRIVABLE_HALF_WIDTH_M = 4;

/**
 * The OUTER edge of the off-track band on the containment ribbon (m):
 * the drivable half width plus the EXISTING named grass-apron width
 * (AC-094 (d) - the band width is an EXISTING published constant, never
 * a band-paint constant). With the bound RETIRED this anchor names the
 * pull-back horizon the lanes prove ABSENT: nothing ever clamps a car
 * back to it (the AC-090 VERDICT-8 observable: the road edge is an INNER
 * boundary, never a wall).
 */
const CONTAINMENT_BAND_OUTER_EDGE_M =
  CONTAINMENT_DRIVABLE_HALF_WIDTH_M + NEATRACING_TRACK_GRASS_APRON_WIDTH_M;

/** The world x of the containment fixture car (m): mid-ribbon. */
const CONTAINMENT_CAR_X_M = 25;

/** The lateral offset of the containment fixture car (m): ON the ribbon. */
const CONTAINMENT_CAR_ON_RIBBON_Y_M = 3.5;

/**
 * The lateral offset of the crossing containment fixture car (m): HALF a
 * metre INSIDE the band outer edge, so ONE tick at the fixture speed
 * really crosses the band outer edge (the crossing is the NON-VACUITY
 * GUARD of the lanes below; with the bound retired the car simply keeps
 * rolling outward - nothing clamps it back).
 */
const CONTAINMENT_CAR_IN_BAND_Y_M = CONTAINMENT_BAND_OUTER_EDGE_M - 0.5;

/**
 * The lateral offset of the pose-independence fixture car (m): BEYOND the
 * band outer edge - far from any pull-back horizon, so the lanes can
 * measure that the RETIRED seam moves NOTHING (the retired seam is a
 * byte-identical NO-OP: the car stands where it stood).
 */
const CONTAINMENT_CAR_OFF_RIBBON_Y_M = CONTAINMENT_BAND_OUTER_EDGE_M + 1;

/** The tick-start speed of the containment fixture car (m/s). */
const CONTAINMENT_CAR_SPEED_MPS = 50;

/** The committed slide of the slipping fixture (rad): a latched drift. */
const CONTAINMENT_FIXTURE_COMMITTED_SLIP_RAD = 0.2;

/** The published slip of the slipping fixture at tick start (rad). */
const CONTAINMENT_FIXTURE_SLIP_RAD = 0.1;

/** The lateral velocity of the slipping fixture at tick start (m/s). */
const CONTAINMENT_FIXTURE_LATERAL_VELOCITY_MPS = 3;

/** The slipping control sample: the wheel keeps the latched slide alive. */
const CONTAINMENT_SLIP_INPUT: PhysicsStepInput = {
  throttle: 1,
  brake: 0,
  steer: 1,
};

/**
 * The distance from a world point to the CONTAINMENT ribbon's centerline
 * (the nearest chord point of the straight fixture ribbon).
 *
 * The fixture ribbon is the straight chord from (-50, 0) to (50, 0), so the
 * nearest chord point of a point inside the chord span is that point
 * projected onto y = 0.
 *
 * @param x - The world x position of the point (m).
 * @param y - The world y position of the point (m).
 * @returns The distance from the point to the ribbon centerline (m).
 */
const containmentCenterlineDistance = (x: number, y: number): number => {
  const nearestX = Math.min(50, Math.max(-50, x));
  return Math.hypot(x - nearestX, y);
};

/**
 * The heading wrap into `[-π, π)` (the local mirror of the service wrap).
 *
 * @param radians - The heading to wrap (rad).
 * @returns The wrapped heading (rad).
 */
const wrapFixtureHeading = (radians: number): number =>
  radians - Math.PI * 2 * Math.floor((radians + Math.PI) / (Math.PI * 2));

/**
 * Builds a tick-start car state at a known world pose and speed for the
 * containment fixtures.
 *
 * @param x - The world x position of the car at tick start (m).
 * @param y - The world y position of the car at tick start (m).
 * @param heading - The nose heading of the car at tick start (rad).
 * @param speed - The longitudinal speed of the car at tick start (m/s).
 * @returns The tick-start car state.
 */
const buildContainmentFixtureCar = (
  x: number,
  y: number,
  heading: number,
  speed: number,
): PhysicsCarState => ({
  ...createInitialPhysicsCarState(42, { x, y, heading }),
  speed,
});

/**
 * Builds a SLIDING tick-start car: the drift channel is already latched, so
 * the tick publishes a NON-ZERO slip and a NON-ZERO traction severity
 * BEFORE the retired seam runs. With the containment RETIRED the seam
 * recomposes NOTHING, so the published scalars must already be the model
 * law on the tick's own final state (no second, post-seam reference
 * state exists).
 *
 * @param x - The world x position of the car at tick start (m).
 * @param y - The world y position of the car at tick start (m).
 * @param heading - The nose heading of the car at tick start (rad).
 * @param speed - The longitudinal speed of the car at tick start (m/s).
 * @returns The sliding tick-start car state.
 */
const buildSlidingContainmentFixtureCar = (
  x: number,
  y: number,
  heading: number,
  speed: number,
): PhysicsCarState => ({
  ...buildContainmentFixtureCar(x, y, heading, speed),
  slipAngle: CONTAINMENT_FIXTURE_SLIP_RAD,
  lateralVelocity: CONTAINMENT_FIXTURE_LATERAL_VELOCITY_MPS,
  drift: {
    committed: {
      slip: CONTAINMENT_FIXTURE_COMMITTED_SLIP_RAD,
      turnSign: 1,
      reArmed: true,
    },
    pose: { pose: 0, poseRate: 0, articulation: 0 },
  },
});

/**
 * The LIVE road-containment contracts (Step 14I, slice
 * `P1-track-road-containment`; AC-081, AC-081-WIRE, AC-081-WIRE-NOOP,
 * AC-081-WIRE-SEAM, AC-081-WIRE-SCALARS).
 */
describe('the LIVE road containment (Step 14I, slice P1-track-road-containment)', () => {
  describe('RED (the live containment wiring)', () => {
    // [RED-14I-CONTAINMENT]
    it('NO CONTAINMENT (AC-090 VERDICT 8): a car driven off the band KEEPS ROLLING past the band outer edge - the seam is a retired NO-OP', () => {
      // Arrange: a car driving ACROSS the ribbon from INSIDE the band, so
      // ONE tick really takes it past the band outer edge. The AC-081
      // wall this seam used to apply ("a car driven off the band comes
      // back ON the band outer edge") is RETIRED: the ONE off-track
      // limit is the sand speed cap (AC-094), never geometry.
      const car = buildContainmentFixtureCar(
        CONTAINMENT_CAR_X_M,
        CONTAINMENT_CAR_IN_BAND_Y_M,
        Math.PI / 2,
        CONTAINMENT_CAR_SPEED_MPS,
      );
      // The SAME tick WITHOUT the geometry: the free, unbounded roll.
      const unbounded = stepPhysics(car, NEUTRAL_INPUT);

      // Act
      const published = stepPhysicsWithGeometryOf?.(
        car,
        NEUTRAL_INPUT,
        ROAD_SURFACE_RESOLVER,
        CONTAINMENT_RIBBON,
      );

      // Assert
      // The NON-VACUITY GUARD comes FIRST: the seam must be callable, and
      // the unbounded tick must REALLY leave the band, so the clauses
      // below can never pass on a baseline that never left.
      expect(typeof stepPhysicsWithGeometryOf).toBe('function');
      expect(
        containmentCenterlineDistance(unbounded.x, unbounded.y),
      ).toBeGreaterThan(CONTAINMENT_BAND_OUTER_EDGE_M);
      // THE RETIREMENT (byte-identity): the published pose with the
      // geometry is EXACTLY the free roll - the retired seam is a NO-OP
      // for this input too.
      expect(JSON.stringify(published)).toBe(JSON.stringify(unbounded));
      // ... and the car keeps rolling OUTWARD: nothing clamps it back to
      // the band outer edge (the AC-090 VERDICT-8 observable: no
      // invisible wall anywhere near the track edge, the published pose
      // is the REACHED pose).
      expect(
        containmentCenterlineDistance(
          published?.x ?? Number.NaN,
          published?.y ?? Number.NaN,
        ),
      ).toBeGreaterThan(CONTAINMENT_BAND_OUTER_EDGE_M);
    });

    // [RED-14I-NOOP]
    it('NO-OP: an ABSENT geometry, an EMPTY segment list and a FULL ribbon leave the tick BYTE-IDENTICAL (and NOTHING throws)', () => {
      // Arrange
      const car = buildContainmentFixtureCar(
        CONTAINMENT_CAR_X_M,
        CONTAINMENT_CAR_ON_RIBBON_Y_M,
        Math.PI / 2,
        CONTAINMENT_CAR_SPEED_MPS,
      );

      // Act
      // The unbounded tick (NO surface probe, NO geometry).
      const unbounded = stepPhysics(car, NEUTRAL_INPUT);
      // The SAME tick WITH a surface probe but NO geometry.
      const withoutGeometry = stepPhysicsWithGeometryOf?.(
        car,
        NEUTRAL_INPUT,
        ROAD_SURFACE_RESOLVER,
      );
      // The SAME tick with an EMPTY segment list.
      const emptyGeometry = stepPhysicsWithGeometryOf?.(
        car,
        NEUTRAL_INPUT,
        ROAD_SURFACE_RESOLVER,
        [],
      );
      // The SAME tick with a FULL ribbon (the retired seam reads
      // NOTHING, so even the geometry it "could" read changes nothing).
      const fullGeometry = stepPhysicsWithGeometryOf?.(
        car,
        NEUTRAL_INPUT,
        ROAD_SURFACE_RESOLVER,
        CONTAINMENT_RIBBON,
      );

      // Assert
      expect(typeof stepPhysicsWithGeometryOf).toBe('function');
      // NOTHING throws: the retired seam owns EVERY input and reads
      // NOTHING from it - an empty list, a full ribbon and an absent
      // geometry all produce THE SAME tick.
      expect(() =>
        stepPhysicsWithGeometryOf?.(
          car,
          NEUTRAL_INPUT,
          ROAD_SURFACE_RESOLVER,
          [],
        ),
      ).not.toThrow();
      expect(() =>
        stepPhysicsWithGeometryOf?.(
          car,
          NEUTRAL_INPUT,
          ROAD_SURFACE_RESOLVER,
          CONTAINMENT_RIBBON,
        ),
      ).not.toThrow();
      // The empty list is a BYTE-IDENTICAL no-op...
      expect(JSON.stringify(emptyGeometry)).toBe(JSON.stringify(unbounded));
      // ... so is the absent geometry...
      expect(JSON.stringify(withoutGeometry)).toBe(JSON.stringify(unbounded));
      // ... and so is the FULL ribbon (the ALL-inputs NO-OP: the
      // retirement's pose-independence, true by construction).
      expect(JSON.stringify(fullGeometry)).toBe(JSON.stringify(unbounded));
    });

    // [RED-14I-POSE-INDEPENDENCE]
    it('POSE-INDEPENDENCE: the SAME car pose reaches the SAME position for EVERY heading and EVERY slip angle - and NOTHING moves it (the retired seam is a NO-OP)', () => {
      // Arrange: a STANDING car BEYOND the band outer edge. A standing car
      // does not translate, so the pose the (retired) seam sees is the
      // SAME world position for every heading and every slip angle - the
      // lane then measures that NOTHING moves the car (the AC-081
      // pose-independence clause carries over: with the bound retired it
      // is true BY CONSTRUCTION, and a bound that creeps back would
      // break the second clause).
      const headings = [0, Math.PI / 4, Math.PI / 2, Math.PI, -Math.PI / 2];
      const slipAngles = [0, 0.1, -0.1];

      // Act
      const contained = headings.flatMap((heading) =>
        slipAngles.map((slipAngle) => {
          const car: PhysicsCarState = {
            ...buildContainmentFixtureCar(
              CONTAINMENT_CAR_X_M,
              CONTAINMENT_CAR_OFF_RIBBON_Y_M,
              heading,
              0,
            ),
            slipAngle,
          };
          const published = stepPhysicsWithGeometryOf?.(
            car,
            NEUTRAL_INPUT,
            ROAD_SURFACE_RESOLVER,
            CONTAINMENT_RIBBON,
          );
          return { heading, slipAngle, x: published?.x, y: published?.y };
        }),
      );

      // Assert
      expect(typeof stepPhysicsWithGeometryOf).toBe('function');
      // The lane really covered the matrix (a vacuous matrix would make
      // the equality below a tautology).
      expect(contained).toHaveLength(headings.length * slipAngles.length);
      // EVERY pose reaches the SAME point - the car stands where it
      // stood for EVERY heading and EVERY slip angle.
      const first = contained[0];
      for (const pose of contained) {
        expect(pose.x).toBe(first.x);
        expect(pose.y).toBe(first.y);
      }
      // And the position is WHERE THE CAR STOOD: the retired seam moves
      // NOTHING, so the standing car stays BEYOND the band outer edge
      // (the centerline distance is STILL the standing offset - no wall
      // pulled it to the bound; the AC-090 VERDICT-8 observable).
      expect(first.x).toBeCloseTo(CONTAINMENT_CAR_X_M, 9);
      expect(first.y).toBeCloseTo(CONTAINMENT_CAR_OFF_RIBBON_Y_M, 9);
      expect(
        containmentCenterlineDistance(
          first.x ?? Number.NaN,
          first.y ?? Number.NaN,
        ),
      ).toBeCloseTo(CONTAINMENT_CAR_OFF_RIBBON_Y_M, 9);
    });

    // [RED-14I-PUBLISHED-SCALARS]
    it('PUBLISHED SCALARS: the published scalars are THE MODEL LAW on the tick\'s own final state, byte-identical with and without the geometry (the containment is retired)', () => {
      // Arrange: a SLIDING car that leaves the band, so the tick
      // publishes a NON-ZERO slip and a NON-ZERO traction severity.
      const car = buildSlidingContainmentFixtureCar(
        CONTAINMENT_CAR_X_M,
        CONTAINMENT_CAR_IN_BAND_Y_M,
        Math.PI / 2,
        CONTAINMENT_CAR_SPEED_MPS,
      );
      // The SAME tick WITHOUT the geometry: the free, unbounded roll.
      const unbounded = stepPhysics(car, CONTAINMENT_SLIP_INPUT);

      // Act
      const published = stepPhysicsWithGeometryOf?.(
        car,
        CONTAINMENT_SLIP_INPUT,
        ROAD_SURFACE_RESOLVER,
        CONTAINMENT_RIBBON,
      );

      // Assert
      // The NON-VACUITY GUARD comes FIRST: the tick really does publish a
      // slip and a severity, so the invariants below can never pass
      // vacuously.
      expect(typeof stepPhysicsWithGeometryOf).toBe('function');
      expect(Math.abs(unbounded.slipAngle)).toBeGreaterThan(0);
      expect(unbounded.tractionSeverity).toBeGreaterThan(0);
      // THE RETIREMENT (byte-identity): the geometry changes NOTHING -
      // with the containment retired there is no second, post-seam
      // reference state, so the published scalars can only be the
      // integration's own numbers.
      expect(JSON.stringify(published)).toBe(JSON.stringify(unbounded));
      // The published travel heading is the model law on the published
      // state: the wrapped sum of the published heading and the
      // published slip.
      expect(published?.travelHeadingRad).toBeCloseTo(
        wrapFixtureHeading(
          (published?.heading ?? Number.NaN) +
            (published?.slipAngle ?? Number.NaN),
        ),
        12,
      );
      // The published traction severity is the model law on the
      // published slip and the published speed.
      expect(published?.tractionSeverity).toBeCloseTo(
        computeTractionSeverityOf?.(
          published?.slipAngle ?? Number.NaN,
          published?.speed ?? Number.NaN,
        ) ?? Number.NaN,
        12,
      );
    });
  });
});

/* ------------------------------------------------------------------------- *
 * STEP 14W (slice `P1-sand-red`): the BOUNDARY RE-CONTRACT (AC-094).
 *
 * THE USER DIRECTIVE: "touching the edge teleports you to the middle of the
 * track" must STOP, and the off-track surface must be drivable and
 * SAND-LIKE instead of a wall that throws the car back onto the ribbon.
 *
 * THE CLASSIFICATION THIS SLICE RECORDS: BOUNDARY/MODEL - the LIVE TICK
 * and the BOUNDARY LAW own the pose, and the defect is NEVER a visual or
 * a presentation defect.
 *
 * AC tags: AC-094 (a - the road edge is an INNER boundary: a car that
 * crossed it really ends up off the road), AC-094 (b - the off-track band
 * is DRIVABLE and SAND-LIKE: the band speed is strictly lower than the
 * road speed for the same input, and the car KEEPS ROLLING),
 * AC-094 (d - the ONE bound moves to the outer edge of the band, whose
 * width is the EXISTING named grass-apron constant), AC-081 (RECONCILED -
 * the RETAINED half: the ONE-BOUND and the SHIPPED-TICK clauses; the
 * SUPERSEDED half: "leaving the track is no longer allowed"),
 * AC-090 (the user gate is NOT claimed - this lane closes nothing and
 * records no verdict).
 *
 * RED-2 (DISCRIMINATING - the lane FAILS today on its own assertion):
 * after the crossing tick the published lateral distance EXCEEDS the
 * drivable half-width. The NON-VACUITY GUARD comes FIRST (the Step 14Q
 * style): the probe MUST really cross the drivable edge inside ONE tick
 * (a probe that stays on the road would satisfy the clause VACUOUSLY) and
 * the LIVE tick MUST really apply the containment seam (the positive
 * control below). The lane deliberately does NOT assert "the bound
 * changed the pose": that clause BREAKS in the green state, where the
 * car legitimately crosses the edge and is no longer reseated.
 *
 * RED-3 (DISCRIMINATING - the lane FAILS today on its own assertion):
 * the band is SAND-LIKE in the LIVE tick: (a) the surface is resolved
 * from the POST-MOVE pose (so the off-road machinery is REACHABLE in the
 * live tick), (b) the band speed is STRICTLY LOWER than the road speed
 * for the SAME deterministic input, and (c) the car KEEPS ROLLING. The
 * A/B pair shares the SAME start speed, the SAME heading and the SAME
 * input - ONLY the lateral offset differs, so the road lane ends ON the
 * road and the band lane ends ON the band. The NON-VACUITY GUARD comes
 * FIRST: neither lane may be a standstill and the two setups must really
 * differ (comparing a standstill with a roll would satisfy the speed
 * clause VACUOUSLY).
 *
 * The lanes are RUNTIME-ONLY: they drive the EXISTING live tick, so every
 * red is a NAMED RUNTIME failure and NEVER an import failure.
 * ------------------------------------------------------------------------- */

/** One recorded surface question of the recording resolver. */
interface BandSurfaceProbe {
  /** The world x of the probed pose (m). */
  readonly x: number;
  /** The world y of the probed pose (m). */
  readonly y: number;
  /** The surface the production classifier resolved for that pose. */
  readonly surface: PhysicsSurfaceType;
}

/**
 * The drivable half width of the band ribbon (m): the 8 m ribbon's half.
 */
const BAND_DRIVABLE_HALF_WIDTH_M = CONTAINMENT_DRIVABLE_HALF_WIDTH_M;

/**
 * The OUTER edge of the off-track band (m): the drivable edge plus the
 * EXISTING named grass-apron width (AC-094 (d) - the band width is an
 * EXISTING published constant, never a band-paint constant).
 */
const BAND_OUTER_EDGE_M =
  BAND_DRIVABLE_HALF_WIDTH_M + NEATRACING_TRACK_GRASS_APRON_WIDTH_M;

/**
 * The crossing probe's tick-start lateral offset (m): INSIDE the
 * drivable surface, close enough to the edge that ONE tick crosses it.
 */
const BAND_CROSSING_START_LATERAL_M = BAND_DRIVABLE_HALF_WIDTH_M - 0.6;

/**
 * The tick-start speed of the crossing probe (m/s).
 */
const BAND_CROSSING_SPEED_MPS = 60;

/**
 * The crossing input: full throttle, straight.
 *
 * DELIBERATELY NOT aliased to the shared `STRAIGHT_INPUT`: the band
 * contract only holds while the throttle is applied (a passive tick would
 * make the off-road speed law indistinguishable from the road law), so a
 * future edit of the shared fixture must not be able to silently
 * de-vacuum this contract.
 */
const BAND_CROSSING_INPUT: PhysicsStepInput = {
  throttle: 1,
  brake: 0,
  steer: 0,
};

/**
 * Builds a tick-start car at a known lateral offset, heading and speed.
 *
 * @param lateralM - The world y of the car at tick start (m).
 * @param heading - The nose heading of the car at tick start (rad).
 * @param speed - The tick-start speed of the car (m/s).
 * @returns The tick-start car state.
 */
const buildBandFixtureCar = (
  lateralM: number,
  heading: number,
  speed: number,
): PhysicsCarState => ({
  ...createInitialPhysicsCarState(42, {
    x: CONTAINMENT_CAR_X_M,
    y: lateralM,
    heading,
  }),
  speed,
});

/**
 * Creates a recording surface resolver over the PRODUCTION classifier:
 * it answers exactly what the production classifier answers and RECORDS
 * every question, so the lane can observe WHICH pose the live tick asked
 * about (the state carries no surface field of its own).
 *
 * @param segments - The ribbon the classifier reads (read only).
 * @returns The recording resolver and its recorded probes.
 */
const createRecordingSurfaceResolver = (
  segments: readonly TrackSegment[],
): {
  readonly resolver: (car: PhysicsCarState) => PhysicsSurfaceType;
  readonly probes: BandSurfaceProbe[];
} => {
  const resolve = createTrackSurfaceResolver(segments);
  const probes: BandSurfaceProbe[] = [];
  return {
    resolver: (car: PhysicsCarState): PhysicsSurfaceType => {
      const surface = resolve(car);
      probes.push({ x: car.x, y: car.y, surface });
      return surface;
    },
    probes,
  };
};

/**
 * The BOUNDARY RE-CONTRACT contracts (Step 14W, slice `P1-sand-red`;
 * AC-094).
 */
describe('the BOUNDARY RE-CONTRACT (Step 14W, slice P1-sand-red)', () => {
  // [RED-14W-CROSSING]
  it('RED-2 (DISCRIMINATING): a car that has crossed the drivable edge REALLY ends up off the road (the published lateral distance EXCEEDS the drivable half-width after the crossing tick)', () => {
    // Arrange: the crossing probe - INSIDE the drivable surface, heading
    // OUTWARD, at a speed that crosses the edge inside ONE tick.
    const start = buildBandFixtureCar(
      BAND_CROSSING_START_LATERAL_M,
      Math.PI / 2,
      BAND_CROSSING_SPEED_MPS,
    );
    const surfaceResolver = createTrackSurfaceResolver(CONTAINMENT_RIBBON);

    // Act: the UNBOUNDED tick (the pose the car REALLY reached) and the
    // LIVE tick with the ribbon geometry (the shipped containment seam).
    const unbounded = stepPhysics(
      start,
      BAND_CROSSING_INPUT,
      surfaceResolver,
    );
    const published = stepPhysics(
      start,
      BAND_CROSSING_INPUT,
      surfaceResolver,
      CONTAINMENT_RIBBON,
    );

    // The NON-VACUITY GUARD comes FIRST: the probe MUST really cross the
    // drivable edge inside ONE tick (a probe that stays inside it would
    // satisfy the clause VACUOUSLY) ...
    expect([
      Math.abs(unbounded.y) > BAND_DRIVABLE_HALF_WIDTH_M,
      Math.abs(unbounded.y) <= BAND_OUTER_EDGE_M,
      Math.abs(start.y) <= BAND_DRIVABLE_HALF_WIDTH_M,
    ]).toEqual([true, true, true]);

    // ... and the LIVE tick really runs the RETIRED seam as a NO-OP (the
    // positive control, RE-AUTHORED by fix-packet-P1-sand-red-iteration-3:
    // a probe FAR beyond the band outer edge is NEVER pulled back to the
    // bound - the published pose is BYTE-IDENTICAL to the free roll, so
    // there is NO wall anywhere near the track edge: the AC-090
    // VERDICT-8 observable. The RETIRING positive control also proves the
    // seam really runs in this tick: a re-landed bound would clamp the
    // far probe to BAND_OUTER_EDGE_M and FAIL this clause).
    const farProbe = buildBandFixtureCar(
      BAND_OUTER_EDGE_M + 10,
      Math.PI / 2,
      BAND_CROSSING_SPEED_MPS,
    );
    const farPublished = stepPhysics(
      farProbe,
      BAND_CROSSING_INPUT,
      surfaceResolver,
      CONTAINMENT_RIBBON,
    );
    expect(JSON.stringify(farPublished)).toBe(
      JSON.stringify(
        stepPhysics(farProbe, BAND_CROSSING_INPUT, surfaceResolver),
      ),
    );
    expect(Math.abs(farPublished.y)).toBeGreaterThan(BAND_OUTER_EDGE_M);

    // RELATIONAL CLAUSE (DISCRIMINATING): the published lateral distance
    // from the centreline EXCEEDS the drivable half-width - the road edge
    // is an INNER boundary, never a wall that teleports the car back onto
    // the ribbon (RED-2, the USER-named observable: the car REALLY ends
    // up off the road, where the sand law now governs it).
    expect(Math.abs(published.y)).toBeGreaterThan(BAND_DRIVABLE_HALF_WIDTH_M);
  });

  // [RED-14W-SAND]
  it('RED-3 (DISCRIMINATING): the band is SAND-LIKE in the LIVE tick (the surface is resolved from the POST-MOVE pose, the band speed is STRICTLY LOWER than the road speed for the SAME input, and the car KEEPS ROLLING)', () => {
    // Arrange: TWO lanes with the SAME start speed, the SAME heading and
    // the SAME input - ONLY the lateral offset differs, so the road lane
    // ends ON the road and the band lane ends ON the band.
    const roadStart = buildBandFixtureCar(
      0,
      Math.PI / 2,
      BAND_CROSSING_SPEED_MPS,
    );
    const bandStart = buildBandFixtureCar(
      BAND_CROSSING_START_LATERAL_M,
      Math.PI / 2,
      BAND_CROSSING_SPEED_MPS,
    );
    const surfaceResolver = createTrackSurfaceResolver(CONTAINMENT_RIBBON);
    const roadLane = createRecordingSurfaceResolver(CONTAINMENT_RIBBON);
    const bandLane = createRecordingSurfaceResolver(CONTAINMENT_RIBBON);

    // Act: the two lanes (with the geometry, so the LIVE tick applies the
    // containment seam) and the UNBOUNDED baselines (the poses the cars
    // REALLY reached).
    const roadPublished = stepPhysics(
      roadStart,
      BAND_CROSSING_INPUT,
      roadLane.resolver,
      CONTAINMENT_RIBBON,
    );
    const bandPublished = stepPhysics(
      bandStart,
      BAND_CROSSING_INPUT,
      bandLane.resolver,
      CONTAINMENT_RIBBON,
    );
    const roadReached = stepPhysics(
      roadStart,
      BAND_CROSSING_INPUT,
      surfaceResolver,
    );
    const bandReached = stepPhysics(
      bandStart,
      BAND_CROSSING_INPUT,
      surfaceResolver,
    );

    // The NON-VACUITY GUARD comes FIRST: the band lane MUST really reach
    // the band inside ONE tick and the road lane MUST really stay on the
    // road (a probe that never reached the band would satisfy the
    // clauses VACUOUSLY) ...
    expect([
      Math.abs(bandReached.y) > BAND_DRIVABLE_HALF_WIDTH_M,
      Math.abs(bandReached.y) <= BAND_OUTER_EDGE_M,
      Math.abs(roadReached.y) <= BAND_DRIVABLE_HALF_WIDTH_M,
    ]).toEqual([true, true, true]);

    // ... the surface model MUST really resolve a NON-ROAD surface for the
    // pose the band lane reached (a probe whose surface is the road would
    // satisfy the surface clause VACUOUSLY) ...
    const bandReachedClassification = classifyTrackSurface(
      { ...bandStart, x: bandReached.x, y: bandReached.y },
      CONTAINMENT_RIBBON,
    );
    expect(bandReachedClassification.surface).toBe('grass');

    // ... the two lanes MUST really share the SAME start speed, the SAME
    // heading and the SAME input, and NEITHER may be a standstill
    // (comparing a standstill with a roll would satisfy the speed clause
    // VACUOUSLY).
    expect([
      bandStart.speed === roadStart.speed,
      bandStart.heading === roadStart.heading,
      roadPublished.speed > 0,
      bandPublished.speed > 0,
    ]).toEqual([true, true, true, true]);

    // RELATIONAL CLAUSE (a) (DISCRIMINATING): the LIVE tick resolved the
    // surface for the pose the car REACHED (the POST-MOVE pose), so the
    // off-road machinery is REACHABLE in the live tick. TODAY the live
    // tick resolves the surface at the TICK START, so it never asks about
    // a band pose and this lane FAILS HERE.
    const bandProbes = bandLane.probes.filter(
      (probe) => Math.abs(probe.y) > BAND_DRIVABLE_HALF_WIDTH_M,
    );
    expect(bandProbes.length).toBeGreaterThan(0);
    expect(bandProbes.every((probe) => probe.surface === 'grass')).toBe(true);

    // RELATIONAL CLAUSE (b) (DISCRIMINATING): the band speed is STRICTLY
    // LOWER than the road speed for the SAME deterministic input. TODAY
    // the live tick resolves the surface at the tick start (the road), so
    // the off-road law never applies and the two speeds are EQUAL - this
    // lane FAILS HERE.
    expect(bandPublished.speed).toBeLessThan(roadPublished.speed);

    // RELATIONAL CLAUSE (c) (the RAIL): the car KEEPS ROLLING on the
    // band - the band is drivable sand, never a standstill.
    expect(bandPublished.speed).toBeGreaterThan(0);
  });
});

/* ------------------------------------------------------------------------- *
 * STEP 14W (slice `P1-sand-red`, ITERATION 2): the RED-I2 lanes - the
 * SAND-BAND OUTER-EDGE REBOUND.
 *
 * THE MEASURED DEFECT (the 2026-11-04 RED MEASUREMENT ROUND of the plan's
 * `fix-packet-P1-sand-red-iteration-2`, BEFORE any production edit): the
 * "teleport" that survived the iteration-1 band contract is NOT a visual
 * defect and NOT the iteration-1 bound LOCATION - it is the ELASTIC
 * REBOUND of the sand-band OUTER EDGE (the former
 * `reboundPastBandOuterEdge` in `./collision.utils`, REMOVED WITH the
 * bound by iteration 3): a car that crosses the band outer edge had its
 * OUTWARD velocity component REFLECTED and was RECOMPOSED around the
 * reflected velocity, so the published heading SNAPPED (the MEASURED
 * one-tick snap was 1.2870 rad) and the reflected velocity carried the
 * car BACK onto the ribbon (the MEASURED lateral distance fell from
 * 17.64 m to 13.59 m - straight back over the drivable edge), which
 * destroyed the sand-like slow roll (the MEASURED off-road speed never
 * came near the then-floor 22.25 m/s: it bottomed out at 57.18 m/s).
 *
 * THE RETIREMENT (fix-packet-P1-sand-red-iteration-3, AC-090 VERDICT 8):
 * the bound AND its rebound are RETIRED ENTIRELY - the track boundary
 * contains NOTHING, the published pose is the reached pose, and the ONE
 * off-track limit is the USER-named sand cap (AC-094). The lanes below
 * are RE-AUTHORED IN-STEP onto the retirement: their anchors are the
 * poses the car REALLY reaches (the band-outer-edge excess and the first
 * grass tick), never the retired touch.
 *
 * THE LANES BELOW (the served lane lives in
 * `./host/race-session.sand-red.test.ts`):
 *
 * RED-I2-a (DISCRIMINATING, RE-AUTHORED BY ITERATION 3): the published
 *   heading delta never SNAPS - for EVERY tick the published heading
 *   change stays inside the yaw-rate channel (`abs(yawRate) * dt`); the
 *   retired seam leaves the yaw to the model (the OLD measured snap was
 *   1.2870 rad).
 * RED-I2-b (DISCRIMINATING, RE-AUTHORED BY ITERATION 3): once the car is
 *   off the ribbon it NEVER reenters the drivable surface on its own
 *   (measured from the FIRST GRASS tick on).
 * RED-I2-c (DISCRIMINATING, RE-AUTHORED BY ITERATION 3 against the
 *   USER-named off-track speed): the sand roll SETTLES AT THE USER-NAMED
 *   OFF-TRACK LIMIT 20 km/h = 5.5556 m/s (the USER's OWN number); the
 *   published 22.25 m/s floor is MEASURED 4.005x the USER value and was
 *   SUPERSEDED by the 2026-11-05 verdict-8 record.
 * RED-I2-preserve (PRESERVATION GUARD - GREEN today, able to fail): the
 *   per-tick displacement stays inside ONE integration envelope on EVERY
 *   tick (re-authored by iteration 3: the envelope is measured on the
 *   WHOLE lane and no mechanism may regress into a hard position
 *   teleport).
 *
 * THE LANES ARE RUNTIME-ONLY: they drive the EXISTING live tick, so every
 * red is a NAMED RUNTIME failure and NEVER an import failure.
 *
 * RED-I3-wall / RED-I3-cap (ITERATION 3, the describe block right below
 * the RED-I2 one): the band-outer-edge WALL + the USER-named off-track
 * speed CAP contracts. The MEASURED classification (W1 the pose hold vs
 * W2 the clean corridor vs S1 the speed-law contradiction vs S2 the
 * settle window) and its numbers live in the plan's 2026-11-05 RED
 * MEASUREMENT ROUND record (fix-packet-P1-sand-red-iteration-3), which
 * was written BEFORE any production or contract edit, per AC-094.
 * ------------------------------------------------------------------------- */

/**
 * The served demo-track fixture (the browser-entry generation input: the
 * REAL served geometry the defect was measured on).
 */
const SAND_I2_TRACK_FIXTURE = {
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'large',
} as const;

/**
 * The outward heading bias of the sharp-exit probe (rad): the car leaves
 * the ribbon across the sand band with a wide, pose-controlled bias.
 */
const SAND_I2_OUTWARD_BIAS_RAD = 0.6;

/** The sharp-exit probe's tick-start speed (m/s). */
const SAND_I2_EXIT_SPEED_MPS = 60;

/**
 * The sharp-exit probe's input: full throttle and NO steer.
 *
 * DELIBERATELY NOT aliased to a shared fixture: the yaw contract of
 * RED-I2-a is only observable while the steer axis stays DISENGAGED (a
 * steered tick publishes a non-zero yaw rate, so the yaw envelope would be
 * satisfied by the driver and never by the bound).
 */
const SAND_I2_EXIT_INPUT: PhysicsStepInput = {
  throttle: 1,
  brake: 0,
  steer: 0,
};

/** The live tick's fixed step (s). */
const SAND_I2_TICK_SECONDS = NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS;

/** The float64 epsilon of the heading arithmetic (rad). */
const SAND_I2_YAW_EPSILON_RAD = 1e-9;

/** The float64 epsilon of the displacement arithmetic (m). */
const SAND_I2_MOVE_EPSILON_M = 1e-6;

/* NOTE: the iteration-2 speed epsilon (`SAND_I2_SPEED_EPSILON_MPS`) was
 * removed in-step by the iteration-3 re-author: the speed clause reads the
 * NAMED off-road speed export now, so no raw speed epsilon is asserted. */

/**
 * The observation window of the anchor lanes (ticks): long enough to span
 * the band-outer-edge anchor with margin (the lane's pre-anchor ticks and
 * the anchor itself both live inside it).
 */
const SAND_I2_TOUCH_WINDOW_TICKS = 32;

/**
 * The sand-roll observation budget (ticks): the plan's fixed budget for
 * the RED-I2-c window.
 */
const SAND_I2_ROLL_BUDGET_TICKS = 300;

/* ------------------------------------------------------------------------- *
 * STEP 14W (slice `P1-sand-red`, ITERATION 3): the USER-NAMED OFF-TRACK
 * SPEED + THE WALL PROBES (fix-packet-P1-sand-red-iteration-3).
 *
 * THE MEASURED CLASSIFICATION (the plan's 2026-11-05 RED MEASUREMENT
 * ROUND, recorded BEFORE any production edit): (W1) the landed
 * iteration-2 clamp HOLDS the pose EXACTLY on the band outer edge -
 * 288/288 post-touch ticks read 3.0 m past the drivable edge to 1e-9
 * while full throttle is applied (max published excess 0.0 vs the
 * free-rolling twin at 79.52 m) - the hold IS the wall the USER named;
 * (W2) the corridor beyond the bound is CLEAN (7.0 m to the AC-096
 * 10 m prop floor, 9.3342 m to the AC-096 ledger's worst measured prop
 * body 12.3342 m); (S1) the published off-road floor 22.25 m/s
 * = 80.1 km/h is 4.005x the USER-named limit; (S2) under the EXISTING
 * off-road decel law the settle to the USER value takes ~38 ticks from
 * the stall speed and ~127 ticks from the lane's measured 61.63 m/s
 * grass entry - inside the existing roll budget either way.
 * ------------------------------------------------------------------------- */

/**
 * THE USER-NAMED OFF-TRACK SPEED LIMIT (m/s): 20 km/h = 5.5556 m/s
 * (20 / 3.6). THE USER's OWN NUMBER, recorded verbatim in the plan's
 * 2026-11-05 VERDICT 8 record ("...with the speed limited to 20km/h").
 * This SUPERSEDES the off-track floor part of AC-094(c)
 * (`NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S`, 22.25 m/s,
 * measured 4.005x the USER value); the EXISTING off-road decel law
 * (`NEATRACING_PHYSICS_OFF_ROAD_DECEL_UNITS_PER_S2`) STAYS the slowdown
 * mechanism, so the sand feel survives.
 */
const SAND_I3_USER_OFF_TRACK_LIMIT_MPS = 5.5556;

/**
 * The enforce window of the USER-named limit (m/s): wide enough to accept
 * both the exact 20/3.6 expansion (5.555555...) and a 5.5556 named
 * literal (both ARE the USER's named value), narrow enough to reject any
 * slack above ~20.02 km/h (the published 22.25 m/s floor sits 16,694 of
 * these windows above the USER value).
 */
const SAND_I3_CAP_EPSILON_MPS = 1e-3;

/**
 * The wall excess epsilon (m): the published past-edge excess at the band
 * outer edge is MEASURED EXACTLY 0.0 today (the clamp lands the pose at
 * exactly the bound, residual <= 1e-9), so 1e-4 separates "the car left
 * the bound behind" from "the car was held at it" - a resolved fix grows
 * the excess by ~0.06 m per tick ALREADY at the USER cap (the 5.5556 m/s
 * roll through the lane's outward bias), ~600x the window.
 */
const SAND_I3_WALL_EXCESS_EPSILON_M = 1e-4;

/**
 * The hold epsilon of the band-outer-edge pin (m): the MEASURED clamp
 * residual at the bound is <= 1e-9, so 1e-9 is the float boundary of
 * "pinned on the bound" (NEVER a slowdown - a slowdown would show a
 * growing excess).
 */
const SAND_I3_PIN_EPSILON_M = 1e-9;

/**
 * The genuine road-speed entry margin over the USER-named limit (m/s):
 * the cap contract is non-vacuous only when the car ENTERS the grass
 * ABOVE a road-speed reading (the MEASURED live-tick entry is
 * 61.6317 m/s and the MEASURED served entry is 38.8273 m/s - both far
 * above 6.0556), never when it crawls onto the sand already inside the
 * cap (a cap enforced at the cap's own entry speed would prove nothing).
 */
const SAND_I3_ENTRY_MARGIN_MPS = 0.5;

/** One recorded tick of the sharp-exit lane. */
interface SandI2Tick {
  /** The tick index (0 is the first tick after the start pose). */
  readonly index: number;
  /** The published pose of the tick (the pose the world sees). */
  readonly published: PhysicsCarState;
  /** The surface the PRODUCTION classifier resolved for the published pose. */
  readonly surface: PhysicsSurfaceType;
  /** The published displacement of the tick (m). */
  readonly moveM: number;
  /** The wrapped published heading delta of the tick (rad). */
  readonly yawDeltaRad: number;
  /** The published speed of the tick (m/s). */
  readonly speedMps: number;
  /** TRUE when the published pose diverged from the free-rolling twin. */
  readonly boundMovedPose: boolean;
}

/** The recorded sharp-exit lane and its MEASURED summary. */
interface SandI2Lane {
  /** The recorded ticks. */
  readonly ticks: readonly SandI2Tick[];
  /**
   * The index of the tick in which the published pose diverged from the
   * free-rolling twin. MEASURED under the RETIRED seam (iteration 3): -1
   * for EVERY tick (the byte-identical NO-OP); a re-landed bound would
   * set it at the touch.
   */
  readonly touchIndex: number;
  /** The consecutive distinct surfaces the lane crossed. */
  readonly surfaceLadder: readonly PhysicsSurfaceType[];
  /**
   * The surface of the tick right BEFORE the band-outer-edge anchor (the
   * drivable-edge crossing; RE-AUTHORED BY ITERATION 3 - the OLD touch
   * anchor is retired WITH the bound).
   */
  readonly preAnchorSurface: PhysicsSurfaceType;
  /** The largest heading delta OVER the yaw-rate envelope (rad). */
  readonly maxYawViolationRad: number;
  /** The largest displacement OVER the integration envelope (m). */
  readonly maxMoveViolationM: number;
  /** The smallest displacement of the lane (m). */
  readonly minMoveM: number;
  /**
   * The smallest displacement BEFORE the band-outer-edge anchor (m;
   * RE-AUTHORED BY ITERATION 3 - the OLD pre-touch anchor is retired
   * WITH the bound).
   */
  readonly minPreAnchorMoveM: number;
  /** The smallest published speed of the lane (m/s). */
  readonly minSpeedMps: number;
  /** The published speed of the last tick of the lane (m/s). */
  readonly finalSpeedMps: number;
  /**
   * The tick indices (from the FIRST GRASS tick on) on which the car was
   * back on the ribbon (RE-AUTHORED BY ITERATION 3 - the OLD touch
   * anchor is retired WITH the bound, and a vacuous -1 anchor would make
   * this clause EMPTY forever).
   */
  readonly reentryTickIndices: readonly number[];
  /** The tick indices of the FIRST continuous off-road run (the sand roll). */
  readonly offRoadRunTickIndices: readonly number[];
  /** The speed of the LAST off-road tick of the sand roll (m/s). */
  readonly lastOffRoadSpeedMps: number;
  /**
   * THE RE-AUTHORED ANCHOR (iteration 3): the first published tick whose
   * past-edge distance reaches the band outer edge (the 3.0 m apron), or
   * -1 when the lane never reaches it. THE OLD touch anchor is RETIRED
   * WITH the bound (the retired seam never diverges the poses, so a
   * touch anchor would sit at -1 forever and every lane built on it
   * would be VACUOUS). FIX-NEUTRAL: the car reaches that distance when
   * the bound HOLDS it AND when the bound is moved farther away or
   * retired (the free roll crosses 3.0 m on the way out) - the anchor
   * alone claims nothing about the wall.
   */
  readonly firstBoundEdgeExcessTickIndex: number;
  /**
   * THE ITERATION-3 PROBE: the count of post-anchor ticks whose published
   * past-edge distance sits EXACTLY on today's band outer edge
   * (`|pastEdge - 3.0| <= 1e-9`) - THE MEASURED WALL HOLD. MEASURED
   * today: 288/288 post-touch ticks (ticks 12..299 of the roll budget).
   */
  readonly pinnedExcessTickCount: number;
  /**
   * THE ITERATION-3 PROBE: the LARGEST published past-edge distance
   * BEYOND today's band outer edge over the post-anchor window (m).
   * MEASURED today: 0.0 (the clamp never lets the car one epsilon past
   * its own bound). -Infinity when the anchor never fires (the reaching
   * guard of the wall lane fails first, so this is never asserted on).
   */
  readonly maxPublishedExcessM: number;
  /**
   * THE ITERATION-3 PROBE: the LARGEST free-rolling past-edge distance
   * BEYOND the band outer edge over the post-anchor window (m; measured
   * data only - it is NEVER asserted on, because a moved bound also
   * leaves the two trajectories apart and only the USER observable
   * decides). MEASURED today: 79.5197 m by tick 299.
   */
  readonly maxUnboundedExcessM: number;
  /**
   * THE ITERATION-3 PROBE: the LARGEST free-rolling MINUS published
   * past-edge deficit over the post-anchor window (m; THE MEASURED WALL
   * HOLD DISTANCE - measured data only, never asserted on, for the same
   * fix-neutrality reason as `maxUnboundedExcessM`). MEASURED today:
   * 79.5197 m at the end of the roll budget.
   */
  readonly maxWallDeficitM: number;
  /**
   * THE ITERATION-3 PROBE: the published speed at the FIRST grass tick
   * (m/s; the genuine road-speed entry guard of the cap lane). MEASURED
   * today: 61.6317 m/s (served wiring: 38.8273 m/s) - both far above
   * the USER limit + margin.
   */
  readonly speedAtFirstGrassMps: number;
  /**
   * THE ITERATION-3 PROBE: the first OFF-ROAD tick whose speed is at or
   * below the USER-named limit + cap window (m/s), or -1 when the lane
   * never settles. Under the RETIRED seam + the AC-094 USER cap a settled
   * lane reports the settle tick (the floor gate decays into the 5.5556
   * m/s settlement); the PRE-FIX measurement recorded -1 - the then-
   * published floor gate pinned 22.25 m/s forever (4.005x the USER
   * value).
   */
  readonly firstSettleTickIndex: number;
  /**
   * THE ITERATION-3 PROBE: the max published speed from the settle tick
   * on, measured over the CONTINUING off-road run (m/s). SENTINEL RULE:
   * a lane that never settles reports the MAX off-road speed of the
   * whole run (a truthful "the cap never held", never a vacuous pass),
   * and a lane with no grass tick reports NaN.
   */
  readonly postSettleMaxSpeedMps: number;
  /**
   * THE ITERATION-3 PROBE: the min published speed from the settle tick
   * on over the continuing off-road run (m/s; THE CAR STILL ROLLS - the
   * cap is a roll, never a trap). SENTINEL RULE: NaN when the lane never
   * settles (the settle clause fails first, so this is never asserted
   * on) or when the lane has no grass tick at all.
   */
  readonly postSettleMinSpeedMps: number;
}

/**
 * Runs the sharp-exit lane on the REAL served geometry: the car leaves the
 * ribbon through the sand band with a wide outward heading bias, so the
 * live tick really crosses the band outer edge.
 *
 * The lane runs the PUBLISHED tick (the live tick WITH the ribbon
 * geometry) and the UNBOUNDED tick (the pose the car REALLY reached)
 * side by side: the first tick in which the two poses differ is the TOUCH
 * - the tick in which the ONE bound moved the published pose.
 *
 * @param tickBudget - The number of ticks the lane runs.
 * @returns The recorded lane and its measured summary.
 */
const buildSandI2Lane = (tickBudget: number): SandI2Lane => {
  const trackSpec = generateTrack(SAND_I2_TRACK_FIXTURE);
  const segments = trackSpec.segments;
  const surfaceResolver = createTrackSurfaceResolver(segments);

  // The LONGEST chord is the straightest ribbon, so the outward exit is a
  // clean straight-line crossing of the sand band.
  let longest = segments[0];
  let longestLengthM = 0;
  for (const segment of segments) {
    const lengthM = Math.hypot(
      segment.endX - segment.startX,
      segment.endY - segment.startY,
    );
    if (lengthM > longestLengthM) {
      longestLengthM = lengthM;
      longest = segment;
    }
  }
  const unitX = (longest.endX - longest.startX) / longestLengthM;
  const unitY = (longest.endY - longest.startY) / longestLengthM;
  const midX = (longest.startX + longest.endX) / 2;
  const midY = (longest.startY + longest.endY) / 2;
  const normalX = -unitY;
  const normalY = unitX;
  // The OUTWARD side: the side whose normal points AWAY from the ring
  // centre (the ribbon is a closed loop, so the midpoint radius decides).
  const midRadiusM = Math.hypot(midX, midY);
  const exitSide =
    Math.hypot(midX + normalX, midY + normalY) > midRadiusM ? 1 : -1;
  const chordHeadingRad = Math.atan2(unitY, unitX);
  const outwardAngleRad =
    exitSide === 1
      ? Math.asin(SAND_I2_OUTWARD_BIAS_RAD)
      : Math.asin(-SAND_I2_OUTWARD_BIAS_RAD);
  const startCar: PhysicsCarState = {
    ...createInitialPhysicsCarState(trackSpec.seed, {
      x: midX + exitSide * normalX * 10 - unitX * 6,
      y: midY + exitSide * normalY * 10 - unitY * 6,
      heading: chordHeadingRad + outwardAngleRad,
    }),
    speed: SAND_I2_EXIT_SPEED_MPS,
  };

  const ticks: SandI2Tick[] = [];
  // THE ITERATION-3 PROBE LEDGER: the free-rolling twin's poses, recorded
  // IN LOCKSTEP with the published ticks (same array index) so the wall
  // deficit can be measured post-loop WITHOUT touching the loop below.
  const unboundedPoses: PhysicsCarState[] = [];
  let published = startCar;
  let unbounded = startCar;
  let previousYawRate = 0;
  let touchIndex = -1;
  let maxYawViolationRad = Number.NEGATIVE_INFINITY;
  let maxMoveViolationM = Number.NEGATIVE_INFINITY;
  let minMoveM = Number.POSITIVE_INFINITY;
  let minSpeedMps = Number.POSITIVE_INFINITY;
  for (let index = 0; index < tickBudget; index += 1) {
    const poseBefore = published;
    const nextPublished = stepPhysics(
      published,
      SAND_I2_EXIT_INPUT,
      surfaceResolver,
      segments,
    );
    const nextUnbounded = stepPhysics(
      unbounded,
      SAND_I2_EXIT_INPUT,
      surfaceResolver,
    );
    const boundMovedPose =
      nextPublished.x !== nextUnbounded.x || nextPublished.y !== nextUnbounded.y;
    if (boundMovedPose && touchIndex < 0) {
      touchIndex = index;
    }
    const moveM = Math.hypot(
      nextPublished.x - poseBefore.x,
      nextPublished.y - poseBefore.y,
    );
    const yawDeltaRad = Math.abs(
      Math.atan2(
        Math.sin(nextPublished.heading - poseBefore.heading),
        Math.cos(nextPublished.heading - poseBefore.heading),
      ),
    );
    if (touchIndex < 0 || index >= touchIndex) {
      // THE YAW ENVELOPE (the plan's law): the published heading delta
      // never exceeds the yaw-rate channel of the PREVIOUS tick.
      maxYawViolationRad = Math.max(
        maxYawViolationRad,
        yawDeltaRad - Math.abs(previousYawRate) * SAND_I2_TICK_SECONDS,
      );
    }
    maxMoveViolationM = Math.max(
      maxMoveViolationM,
      moveM - nextPublished.speed * SAND_I2_TICK_SECONDS,
    );
    minMoveM = Math.min(minMoveM, moveM);
    minSpeedMps = Math.min(minSpeedMps, nextPublished.speed);
    ticks.push({
      index,
      published: nextPublished,
      surface: classifyTrackSurface(nextPublished, segments).surface,
      moveM,
      yawDeltaRad,
      speedMps: nextPublished.speed,
      boundMovedPose,
    });
    unboundedPoses.push(nextUnbounded);
    published = nextPublished;
    unbounded = nextUnbounded;
    previousYawRate = nextPublished.yawRate;
  }

  // The consecutive distinct surfaces the lane crossed.
  const surfaceLadder: PhysicsSurfaceType[] = [];
  for (const tick of ticks) {
    if (surfaceLadder[surfaceLadder.length - 1] !== tick.surface) {
      surfaceLadder.push(tick.surface);
    }
  }

  // -----------------------------------------------------------------
  // THE RE-AUTHORED ANCHORS (fix-packet-P1-sand-red-iteration-3): the
  // OLD "touch" anchor moved WITH the bound (the retired seam can never
  // diverge the published pose from the free roll, so a touch-index
  // anchor would sit at -1 forever and every lane built on it would be
  // VACUOUS). The anchors are now poses the car REALLY reaches: the
  // first past-edge excess at the band outer edge and the first grass
  // tick.
  //
  // THE PAST-EDGE READER is the AC-096 nearest-chord reader
  // (resolveNearestTrackChord + resolveTrackEdgeHalfWidthM +
  // resolveTrackEdgeSideAt) - the same reader the AC-096 clearance rows
  // were measured with, so the LANED measurement record and the
  // contracts can never disagree about which band-outer-edge distance
  // the wall hold sits at.
  // -----------------------------------------------------------------
  const edgeProfile = resolveTrackEdgeProfile(segments);
  const pastEdgeMOf = (car: PhysicsCarState): number => {
    const nearest = resolveNearestTrackChord(edgeProfile, car.x, car.y);
    const chord = edgeProfile.chords[nearest.segmentIndex];
    if (chord === undefined) {
      throw new Error(
        'buildSandI2Lane: the edge profile does not own the nearest chord',
      );
    }
    const side = resolveTrackEdgeSideAt(
      edgeProfile,
      nearest.segmentIndex,
      car.x,
      car.y,
    );
    return nearest.distanceM - resolveTrackEdgeHalfWidthM(chord, side);
  };
  const publishedPastEdgeSeriesM = ticks.map((tick) =>
    pastEdgeMOf(tick.published),
  );
  const unboundedPastEdgeSeriesM = unboundedPoses.map((pose) =>
    pastEdgeMOf(pose),
  );

  // The band-outer-edge anchor: the FIRST published tick whose past-edge
  // distance reaches today's bound (the 3.0 m apron). FIX-NEUTRAL: it
  // fires whether the bound HOLDS the car there or the car ROLLS THROUGH
  // it - the anchor alone claims nothing about the wall.
  const firstBoundEdgeExcessTickIndex = publishedPastEdgeSeriesM.findIndex(
    (pastEdgeM) =>
      pastEdgeM >=
      NEATRACING_TRACK_GRASS_APRON_WIDTH_M - SAND_I3_WALL_EXCESS_EPSILON_M,
  );

  // The FIRST GRASS tick anchor: the first tick the PRODUCTION classifier
  // resolves as the off-road band. FIX-NEUTRAL: it fires for any
  // off-road trajectory, whatever the off-road mechanism is.
  const firstGrassIndex = ticks.findIndex((tick) => tick.surface === 'grass');

  // The reentry ticks (RE-AUTHORED): from the FIRST GRASS tick on (the
  // car is REALLY off the ribbon from there), every later tick whose
  // surface is drivable again (the road core or the curb band). The OLD
  // anchor (the touch) is retired WITH the bound - from the vacuous -1
  // on, that clause would be EMPTY forever. Under the RETIRED seam the
  // outward roll never reenters; a re-landed bound that reflects the
  // outward velocity (the OLD MEASURED rebound) carries the car back
  // over the drivable edge and FAILS this clause.
  const reentryTickIndices = ticks
    .filter((tick) => firstGrassIndex >= 0 && tick.index >= firstGrassIndex)
    .filter((tick) => tick.surface !== 'grass')
    .map((tick) => tick.index);

  // The FIRST continuous off-road run: THE sand roll the lane measures.
  // It ends at the reentry (today: the rebound interrupts the roll), at
  // the USER-named off-track limit (THE ITERATION-3 RE-AUTHOR of the
  // stop condition - the published 22.25 m/s floor is SUPERSEDED), or
  // at the tick budget.
  const offRoadRunTickIndices: number[] = [];
  let lastOffRoadSpeedMps = Number.NaN;
  const firstOffRoadIndex = ticks.findIndex((tick) => tick.surface === 'grass');
  if (firstOffRoadIndex >= 0) {
    for (let index = firstOffRoadIndex; index < ticks.length; index += 1) {
      const tick = ticks[index];
      if (tick.surface !== 'grass') {
        break;
      }
      offRoadRunTickIndices.push(tick.index);
      lastOffRoadSpeedMps = tick.speedMps;
      // THE ITERATION-3 RE-AUTHOR of the run's stop condition (the packet
      // orders every floor-supersession site re-authored in-step): the
      // roll now ends at the USER-named off-track limit, so the settle
      // is measured AT the USER value - never at the published 22.25 m/s
      // floor the USER superseded (22.25 = 4.005x the USER value).
      if (
        tick.speedMps <=
        SAND_I3_USER_OFF_TRACK_LIMIT_MPS + SAND_I3_CAP_EPSILON_MPS
      ) {
        break;
      }
    }
  }

  // The smallest displacement BEFORE the band-outer-edge anchor (the
  // non-vacuity guard of RED-I2-a and RED-I3-wall, RE-AUTHORED BY
  // ITERATION 3: the car REALLY moved on every pre-anchor tick; the OLD
  // pre-touch anchor is retired WITH the bound).
  let minPreAnchorMoveM = Number.POSITIVE_INFINITY;
  if (firstBoundEdgeExcessTickIndex >= 0) {
    for (const tick of ticks) {
      if (tick.index < firstBoundEdgeExcessTickIndex) {
        minPreAnchorMoveM = Math.min(minPreAnchorMoveM, tick.moveM);
      }
    }
  }

  // The surface right before the band-outer-edge anchor (the re-authored
  // drivable-edge crossing record of RED-I2-a).
  const preAnchorTick =
    firstBoundEdgeExcessTickIndex > 0
      ? ticks[firstBoundEdgeExcessTickIndex - 1]
      : undefined;

  // -----------------------------------------------------------------
  // THE ITERATION-3 PROBES (fix-packet-P1-sand-red-iteration-3): the
  // wall/cap measurements, computed post-loop from the RECORDED ticks.
  // The PAST-EDGE READER and the two anchors (the band-outer-edge excess
  // and the first grass tick) live directly after the record loop above.
  // -----------------------------------------------------------------

  // The wall hold: every post-anchor tick whose published past-edge
  // distance sits EXACTLY on today's bound (|excess - 3.0| <= 1e-9).
  // MEASURED today: 288/288 post-anchor ticks pinned, max excess 0.0,
  // the free-rolling twin 79.5197 m past the bound by tick 299.
  let pinnedExcessTickCount = 0;
  let maxPublishedExcessM = Number.NEGATIVE_INFINITY;
  let maxUnboundedExcessM = Number.NEGATIVE_INFINITY;
  let maxWallDeficitM = Number.NEGATIVE_INFINITY;
  if (firstBoundEdgeExcessTickIndex >= 0) {
    for (
      let tickIndex = firstBoundEdgeExcessTickIndex;
      tickIndex < ticks.length;
      tickIndex += 1
    ) {
      const publishedExcessM =
        (publishedPastEdgeSeriesM[tickIndex] ?? Number.NaN) -
        NEATRACING_TRACK_GRASS_APRON_WIDTH_M;
      const unboundedExcessM =
        (unboundedPastEdgeSeriesM[tickIndex] ?? Number.NaN) -
        NEATRACING_TRACK_GRASS_APRON_WIDTH_M;
      if (Math.abs(publishedExcessM) <= SAND_I3_PIN_EPSILON_M) {
        pinnedExcessTickCount += 1;
      }
      maxPublishedExcessM = Math.max(maxPublishedExcessM, publishedExcessM);
      maxUnboundedExcessM = Math.max(maxUnboundedExcessM, unboundedExcessM);
      maxWallDeficitM = Math.max(
        maxWallDeficitM,
        unboundedExcessM - publishedExcessM,
      );
    }
  }

  // The cap: the settle to the USER-named limit, measured ONLY over the
  // off-road ticks (the on-road opening ticks run far under the cap and
  // would read a VACUOUS settle). firstGrassIndex is the moved-up first
  // grass tick anchor.
  const speedAtFirstGrassMps =
    firstGrassIndex >= 0
      ? (ticks[firstGrassIndex]?.speedMps ?? Number.NaN)
      : Number.NaN;
  let firstSettleTickIndex = -1;
  let settleOffsetInTicks = -1;
  if (firstGrassIndex >= 0) {
    for (let tickIndex = firstGrassIndex; tickIndex < ticks.length; tickIndex += 1) {
      const tick = ticks[tickIndex];
      if (
        tick === undefined ||
        tick.speedMps >
          SAND_I3_USER_OFF_TRACK_LIMIT_MPS + SAND_I3_CAP_EPSILON_MPS
      ) {
        continue;
      }
      firstSettleTickIndex = tick.index;
      settleOffsetInTicks = tickIndex;
      break;
    }
  }

  // THE SENTINEL RULES (mirrored in the interface JSDoc): a lane that
  // never settles reports firstSettleTickIndex = -1, postSettleMaxSpeedMps
  // = the MAX off-road speed of the whole run (a truthful "the cap never
  // held", never a vacuous pass) and postSettleMinSpeedMps = NaN (the
  // settle clause fails FIRST, so the rolling clause is never asserted
  // on). A settled lane reports the max/min over the CONTINUING off-road
  // run from the settle tick on (the cap is an OFF-TRACK cap).
  let postSettleMaxSpeedMps = Number.NaN;
  let postSettleMinSpeedMps = Number.NaN;
  if (firstGrassIndex >= 0) {
    if (settleOffsetInTicks < 0) {
      for (let tickIndex = firstGrassIndex; tickIndex < ticks.length; tickIndex += 1) {
        const tick = ticks[tickIndex];
        if (tick === undefined || tick.surface !== 'grass') {
          break;
        }
        postSettleMaxSpeedMps = Number.isNaN(postSettleMaxSpeedMps)
          ? tick.speedMps
          : Math.max(postSettleMaxSpeedMps, tick.speedMps);
      }
    } else {
      for (let tickIndex = settleOffsetInTicks; tickIndex < ticks.length; tickIndex += 1) {
        const tick = ticks[tickIndex];
        if (tick === undefined || tick.surface !== 'grass') {
          break;
        }
        postSettleMaxSpeedMps = Number.isNaN(postSettleMaxSpeedMps)
          ? tick.speedMps
          : Math.max(postSettleMaxSpeedMps, tick.speedMps);
        postSettleMinSpeedMps = Number.isNaN(postSettleMinSpeedMps)
          ? tick.speedMps
          : Math.min(postSettleMinSpeedMps, tick.speedMps);
      }
    }
  }

  const lastTick = ticks[ticks.length - 1];
  return {
    ticks,
    touchIndex,
    surfaceLadder,
    preAnchorSurface: preAnchorTick?.surface ?? 'road',
    maxYawViolationRad,
    maxMoveViolationM,
    minMoveM,
    minPreAnchorMoveM,
    minSpeedMps,
    finalSpeedMps: lastTick?.speedMps ?? Number.NaN,
    reentryTickIndices,
    offRoadRunTickIndices,
    lastOffRoadSpeedMps,
    firstBoundEdgeExcessTickIndex,
    pinnedExcessTickCount,
    maxPublishedExcessM,
    maxUnboundedExcessM,
    maxWallDeficitM,
    speedAtFirstGrassMps,
    firstSettleTickIndex,
    postSettleMaxSpeedMps,
    postSettleMinSpeedMps,
  };
};

/**
 * The RED-I2 contracts (slice `P1-sand-red`, ITERATION 2; the plan's
 * `fix-packet-P1-sand-red-iteration-2`).
 */
describe('the SAND-BAND OUTER-EDGE REBOUND (Step 14W iteration 2, slice P1-sand-red)', () => {
  // [RED-I2-A]
  it('RED-I2-a (DISCRIMINATING, RE-AUTHORED BY ITERATION 3): the published heading delta NEVER snaps - for EVERY tick the published heading change stays inside the yaw-rate channel (the OLD clamp snapped 1.2870 rad in one tick; the retired seam leaves the yaw to the model)', () => {
    // Arrange: the sharp-exit lane on the REAL served geometry.
    const lane = buildSandI2Lane(SAND_I2_TOUCH_WINDOW_TICKS);

    // The NON-VACUITY GUARDS come FIRST (they PASS today and are able to
    // fail): the lane REALLY crosses the ribbon (road -> curb -> grass) ...
    expect(lane.surfaceLadder.slice(0, 3)).toEqual(['road', 'curb', 'grass']);
    // ... the pose right before the band-outer-edge anchor is REALLY past
    // the drivable edge (the crossing record, RE-AUTHORED BY ITERATION 3:
    // the anchor is the reaching of the band outer edge, not the retired
    // touch) ...
    expect(lane.preAnchorSurface).toBe('grass');
    expect(lane.firstBoundEdgeExcessTickIndex).toBeGreaterThanOrEqual(0);
    // ... and the car REALLY moved on EVERY pre-anchor tick (a standstill
    // would satisfy the yaw clause VACUOUSLY).
    expect(lane.minPreAnchorMoveM).toBeGreaterThan(0);

    // THE DISCRIMINATING CLAUSE: the published heading delta NEVER exceeds
    // the yaw-rate channel (`abs(yawRate) * dt + 1e-9`). TODAY the seam is
    // RETIRED (it recomposes NOTHING), so the yaw is pure integration and
    // the clause measures the WHOLE lane; a RE-LANDED bound that
    // recomposes the car around a reflected velocity would still fail it
    // (the OLD MEASURED one-tick snap was 1.2870 rad).
    expect(lane.maxYawViolationRad).toBeLessThanOrEqual(
      SAND_I2_YAW_EPSILON_RAD,
    );
  });

  // [RED-I2-B]
  it('RED-I2-b (DISCRIMINATING, RE-AUTHORED BY ITERATION 3): once the car is off the ribbon it NEVER reenters the drivable surface on its own (the retired seam: the outward roll keeps going; the OLD clamp carried the car back over the drivable edge)', () => {
    // Arrange: the sharp-exit lane on the REAL served geometry.
    const lane = buildSandI2Lane(SAND_I2_TOUCH_WINDOW_TICKS);

    // The NON-VACUITY GUARDS come FIRST (they PASS today and are able to
    // fail): the lane REALLY left the ribbon (the off-road band is a real
    // surface of this lane) ...
    expect(lane.offRoadRunTickIndices.length).toBeGreaterThan(0);
    // ... the car REALLY left the WHOLE band behind (the re-authored
    // anchor: the past-edge excess reaches today\'s band outer edge; the
    // OLD touch anchor is retired WITH the bound) ...
    expect(lane.firstBoundEdgeExcessTickIndex).toBeGreaterThanOrEqual(0);
    // ... and the car REALLY moved on EVERY tick (a standstill would
    // satisfy the reentry clause VACUOUSLY).
    expect(lane.minMoveM).toBeGreaterThan(0);

    // THE DISCRIMINATING CLAUSE: NO tick from the first grass tick on may
    // report a drivable surface again - the off-road band is drivable
    // sand, so the car must NEVER be carried back onto the ribbon without
    // user steer. TODAY the seam is RETIRED and the outward roll keeps
    // going (the clause measures the WHOLE post-grass lane); a RE-LANDED
    // bound that reflects the outward velocity would still fail it (the
    // OLD MEASURED reentry: the curb at tick 18 and the road at tick 19).
    expect(lane.reentryTickIndices).toEqual([]);
  });

  // [RED-I2-C]
  it('RED-I2-c (DISCRIMINATING, RE-AUTHORED BY ITERATION 3 against the USER-named off-track speed): the sand roll SETTLES AT THE USER-NAMED LIMIT 20 km/h = 5.5556 m/s - the then-published 22.25 m/s floor measured 4.005x the USER value and is SUPERSEDED (the PRE-FIX stall was the 22.25 floor; this lane discriminated the USER-named value)', () => {
    // Arrange: the sharp-exit lane on the REAL served geometry, observed
    // over the plan's fixed roll budget.
    const lane = buildSandI2Lane(SAND_I2_ROLL_BUDGET_TICKS);

    // The NON-VACUITY GUARDS come FIRST (they PASS today and are able to
    // fail): the lane REALLY left the ribbon (a lane that never reached
    // the off-road band would satisfy the speed clause VACUOUSLY) ...
    expect(lane.offRoadRunTickIndices.length).toBeGreaterThan(0);
    // ... the lane REALLY ran the sand roll (the roll is a measured run of
    // off-road ticks, never a single tick) ...
    expect(lane.offRoadRunTickIndices.length).toBeGreaterThan(1);
    // ... and the car NEVER came to a standstill (the "stopped" state must
    // NOT be entered: the off-track cap is a roll, never a trap).
    expect(lane.minSpeedMps).toBeGreaterThan(0);

    // THE DISCRIMINATING CLAUSE (THE ITERATION-3 RE-AUTHOR): the sand roll
    // ends at the USER-NAMED off-track limit - the measured window's last
    // off-road speed must be at (or below) 5.5556 m/s (the exact 20/3.6
    // expansion and a 5.5556 named literal BOTH pass; any slack above
    // ~20.02 km/h fails). The PRE-FIX floor gate held the speed at
    // exactly 22.25 m/s = 4.005x the USER value (MEASURED 2026-11-05:
    // min = final = 22.25 across the whole 300-tick window) - that was
    // the discriminating failure this lane was authored against. This
    // lane still DISCRIMINATES the sand-law preservation INDEPENDENTLY of
    // RED-I2-b: a change that retires the bound but breaks the off-road
    // deceleration still fails here.
    expect(lane.lastOffRoadSpeedMps).toBeLessThanOrEqual(
      SAND_I3_USER_OFF_TRACK_LIMIT_MPS + SAND_I3_CAP_EPSILON_MPS,
    );
  });

  // [RED-I2-PRESERVE]
  it('RED-I2-preserve (PRESERVATION GUARD - GREEN today, able to fail): the per-tick displacement stays inside ONE integration envelope on EVERY tick (no mechanism ever TELEPORTS the position; the OLD touch anchor is retired WITH the bound and the envelope is measured on the WHOLE lane)', () => {
    // Arrange: the sharp-exit lane on the REAL served geometry.
    const lane = buildSandI2Lane(SAND_I2_TOUCH_WINDOW_TICKS);

    // The NON-VACUITY GUARDS come FIRST (they PASS today and are able to
    // fail): the car REALLY moved (a standstill would satisfy the
    // envelope clause VACUOUSLY) ...
    expect(lane.minMoveM).toBeGreaterThan(0);
    // ... the retired seam NEVER diverged the published pose from the
    // free roll (THE RETIREMENT PROOF: the touch stays at -1 for EVERY
    // tick - the NEGATIVE guard is able to fail the moment a bound
    // creeps back) ...
    expect(lane.touchIndex).toBe(-1);
    // ... and the lane REALLY ran past the WHOLE band (the re-authored
    // anchor; the OLD touch anchor is retired WITH the bound).
    expect(lane.firstBoundEdgeExcessTickIndex).toBeGreaterThanOrEqual(0);

    // THE PRESERVATION CLAUSE: EVERY tick's displacement stays inside the
    // integration envelope of the tick that produced it
    // (`move_t <= speed_t * dt + 1e-6`; the clause measures the WHOLE
    // lane and FAILS the moment a mechanism truly teleports the pose -
    // a re-landed clamp snapping the pose farther than its own tick
    // speed would violate it).
    expect(lane.maxMoveViolationM).toBeLessThanOrEqual(SAND_I2_MOVE_EPSILON_M);
  });
});

/**
 * The RED-I3 contracts (slice `P1-sand-red`, ITERATION 3; the plan's
 * `fix-packet-P1-sand-red-iteration-3`). TWO discriminating contracts
 * probe the TWO measured defects the USER named after the iteration-2
 * verdict: the band-outer-edge reads as an INVISIBLE WALL (the car is
 * HELD at today's bound while the throttle is applied - W1) and the
 * off-track speed is read as UNLIMITED-at-80.1-km/h instead of the
 * USER-named 20 km/h (S1). The lanes are RUNTIME-ONLY and FIX-NEUTRAL
 * (a retired bound and a moved bound BOTH pass them; only a held
 * band-edge bound and any slack above the USER value fail).
 */
describe('the SAND-BAND OUTER-EDGE WALL + THE USER-NAMED OFF-TRACK SPEED CAP (Step 14W iteration 3, slice P1-sand-red)', () => {
  // [RED-I3-WALL]
  it('RED-I3-wall (DISCRIMINATING): NO INVISIBLE WALL at the band outer edge - the car LEAVES today\'s bound location behind and is NEVER held there while the throttle is applied (MEASURED 2026-11-05: 288/288 post-anchor ticks pinned at exactly 3.0 m past the drivable edge, max published excess 0.0 vs the free-rolling twin at 79.52 m)', () => {
    // Arrange: the sharp-exit lane over the FULL roll budget (the
    // MEASURED hold spans ticks 12..299 of this window).
    const lane = buildSandI2Lane(SAND_I2_ROLL_BUDGET_TICKS);

    // The NON-VACUITY GUARDS come FIRST (they PASS today and are able to
    // fail): the lane REALLY crosses the ribbon (road -> curb -> grass;
    // a lane that never leaves the ribbon would satisfy the excess
    // clauses VACUOUSLY) ...
    expect(lane.surfaceLadder.slice(0, 3)).toEqual(['road', 'curb', 'grass']);
    // ... the car REALLY moved on every pre-anchor tick (a standstill
    // would satisfy the excess clauses VACUOUSLY) ...
    expect(lane.minPreAnchorMoveM).toBeGreaterThan(0);
    // ... and the car REALLY reaches today's band outer edge (the
    // reaching guard MEASURED today: the anchor fires at tick 12; a
    // lane that never reaches the bound would satisfy the excess
    // clauses VACUOUSLY).
    expect(lane.firstBoundEdgeExcessTickIndex).toBeGreaterThanOrEqual(0);

    // THE DISCRIMINATING CLAUSE 1: the pose is NOT held at today's
    // bound - the published past-edge distance MUST grow BEYOND the
    // bound (excess > 1e-4 m). The PRE-FIX clamp pinned the pose at
    // EXACTLY the bound forever (MEASURED 2026-11-05 max published
    // excess: 0.0, the residual <= 1e-9) - that was the discriminating
    // failure this lane was authored against.
    expect(lane.maxPublishedExcessM).toBeGreaterThan(
      SAND_I3_WALL_EXCESS_EPSILON_M,
    );

    // THE DISCRIMINATING CLAUSE 2: NO post-anchor tick sits EXACTLY on
    // today's bound (|excess - 3.0| <= 1e-9 is the MEASURED clamp
    // residual). The PRE-FIX run pinned 288/288 post-anchor ticks there
    // - that was the discriminating failure. FIX-NEUTRAL: a retired
    // bound and a bound moved farther away BOTH leave this count at 0
    // (only a HELD band-edge bound fails).
    expect(lane.pinnedExcessTickCount).toBe(0);

    // THE PRESERVATION CLAUSE: the wall defect must NEVER be "resolved"
    // by a position teleport - the per-tick displacement stays inside
    // ONE integration envelope on EVERY tick (the SAME preservation law
    // RED-I2-preserve guards; the MEASURED worst case today is 0 m
    // against the 1e-6 envelope, so this clause is GREEN today and
    // constrains the fix: the car must GROW past the bound tick by
    // tick, never SNAP past it).
    expect(lane.maxMoveViolationM).toBeLessThanOrEqual(SAND_I2_MOVE_EPSILON_M);
  });

  // [RED-I3-CAP]
  it('RED-I3-cap (DISCRIMINATING): the off-track roll SETTLES AT the USER-named 20 km/h = 5.5556 m/s limit and HOLDS it while the car keeps rolling (MEASURED 2026-11-05: the then-published 22.25 m/s floor = 4.005x the USER value; the settle never happened under the pre-fix floor gate)', () => {
    // Arrange: the sharp-exit lane over the FULL roll budget (the
    // MEASURED settle window: ~38 ticks from the 22.25 stall, ~127
    // ticks from the 61.63 m/s grass entry - both inside this budget).
    const lane = buildSandI2Lane(SAND_I2_ROLL_BUDGET_TICKS);

    // The NON-VACUITY GUARDS come FIRST (they PASS today and are able to
    // fail): the lane REALLY ran the sand roll (a lane that never left
    // the ribbon would satisfy the settle clauses VACUOUSLY) ...
    expect(lane.offRoadRunTickIndices.length).toBeGreaterThan(1);
    // ... and the car REALLY entered the grass at a road-speed reading,
    // far ABOVE the USER-named limit (MEASURED today: 61.6317 m/s at
    // the first grass tick; a crawl already inside the cap would
    // satisfy the settle clause VACUOUSLY - THE CAP MUST BE EARNED BY
    // THE DECAY LAW, never by the entry speed).
    expect(lane.speedAtFirstGrassMps).toBeGreaterThan(
      SAND_I3_USER_OFF_TRACK_LIMIT_MPS + SAND_I3_ENTRY_MARGIN_MPS,
    );

    // THE DISCRIMINATING CLAUSE 1: the roll SETTLES within the roll
    // budget - some off-road tick reaches the USER-named limit (speed
    // <= 5.5556 + 1e-3). The PRE-FIX floor gate pinned the speed at
    // EXACTLY 22.25 m/s forever (MEASURED 2026-11-05: min = final =
    // 22.25, ratio 4.005x the USER value) - that was the discriminating
    // failure this lane was authored against.
    expect(lane.firstSettleTickIndex).toBeGreaterThanOrEqual(0);

    // THE DISCRIMINATING CLAUSE 2: the settle HOLDS - the max speed
    // from the settle tick on stays inside the USER-named limit window
    // (<= 5.5556 + 1e-3; both the exact 20/3.6 expansion and a 5.5556
    // literal pass, any slack above ~20.02 km/h fails).
    expect(lane.postSettleMaxSpeedMps).toBeLessThanOrEqual(
      SAND_I3_USER_OFF_TRACK_LIMIT_MPS + SAND_I3_CAP_EPSILON_MPS,
    );

    // THE PRESERVATION CLAUSE: the car STILL ROLLS at the cap (the min
    // speed from the settle tick on stays > 0 m/s - the cap is a slow
    // sand roll, never a trap; the USER still names it a speed
    // LIMIT, not a stop).
    expect(lane.postSettleMinSpeedMps).toBeGreaterThan(0);
  });
});

/* ------------------------------------------------------------------------- *
 * STEP 14Y (slice `P1-turnfeel-red`): the TURNING FEEL + AI DRIVEABILITY
 * wave (AC-098).
 *
 * THE POST-FIX LAW THIS BLOCK RECORDS: the USER turn-rate law keeps the
 * ratified mild reduction above the crawl band and adds a raised-cosine
 * sufficiency boost confined to ~[17.6, 18.0] m/s. At the R = 8 m stadium's
 * target speed (~17.8 m/s) the law commands 2.4475 rad/s against the 2.225
 * rad/s kinematic requirement (a 1.1x margin), while the mild reduction
 * remains untouched outside the boost band. The AI - which drives the SAME
 * geometry - now keeps moving and completes laps.
 *
 * RED-1 (DISCRIMINATING) probes the hold on the tightest MEASURED layout
 * of the repo (the R = 8 m stadium), NOT on the served layout: the served
 * layout's tightest corner (R = 116.5 m) is ALREADY holdable - the second,
 * LABELLED lane below is the GREEN record of that fact - so a served-track
 * reading of RED-1 would be VACUOUS, and RED-1's own non-vacuity guard
 * forbids a vacuous contract.
 *
 * RED-5 IS NOT A RED: every RED-5 lane in this file is a LABELLED
 * PRESERVATION GUARD that holds today and is NEVER counted as a red.
 * ------------------------------------------------------------------------- */

/**
 * The RED-14Y stadium road width (m) - the same 12 m road the shared repo
 * fixture uses.
 */
const TURNFEEL_ROAD_WIDTH_M = 12;

/**
 * The RED-14Y tight-corner radius (m).
 *
 * THE TIGHTEST MEASURED LAYOUT of the repo: an R = 8 m corner is far
 * tighter than the nominal-law turn radius, so the lane measures a REAL
 * corner and never a straight. The post-fix sufficiency boost is what
 * makes this corner holdable.
 */
const TURNFEEL_TIGHT_CORNER_RADIUS_M = 8;

/** The RED-14Y tight-corner straight length (m). */
const TURNFEEL_TIGHT_CORNER_STRAIGHT_M = 40;

/**
 * The RED-14Y crawl-band ABSOLUTE near-standstill rate (rad/s).
 *
 * THE N-7 CARRY-FORWARD (the load-bearing pin): the rate the USER channel
 * commands at the crawl-band speed with the full steer axis, MEASURED
 * through the published heading delta. It is an ABSOLUTE pin (never a
 * derived one), so a global raise of the nominal turn rate can NEVER reach
 * the crawl band - the near-standstill feel stays ratifiable.
 */
const TURNFEEL_CRAWL_BAND_ABSOLUTE_RAD_PER_S = 0.2247191011235955;

/**
 * The RED-14Y drift-ENGAGE lane speed (m/s).
 *
 * The lane runs above the drift's own minimum-speed gate (15 m/s), so the
 * engagement decision the lane measures is really the STEER gate and never
 * a speed gate.
 */
const TURNFEEL_DRIFT_ENGAGE_SPEED_MPS = 40;

/**
 * The RED-14Y drift ENGAGE threshold steer (the engaging side of the
 * boundary).
 */
const TURNFEEL_DRIFT_ENGAGE_STEER = 0.5;

/**
 * The RED-14Y drift NON-ENGAGE threshold steer (the holding side of the
 * boundary): just BELOW the engage gate, so the boundary is really probed
 * and never assumed.
 */
const TURNFEEL_DRIFT_NO_ENGAGE_STEER = 0.49;

/**
 * Builds the RED-14Y stadium fixture: a straight along +x, a right
 * half-circle, a straight along -x and a left half-circle, all traversed
 * COUNTER-CLOCKWISE.
 *
 * THE LOCAL FIXTURE (owned by this red slice): the builder is PORTED from
 * the sibling AI test file's fixture (a test file never imports another
 * test file), so the two blocks measure the SAME geometry.
 *
 * @param radiusM - The corner radius (m).
 * @param straightM - The straight length (m).
 * @returns The frozen spline samples of the closed loop.
 */
const buildTurnfeelStadiumSamples = (
  radiusM: number,
  straightM: number,
): readonly SplineSample[] => {
  const samples: SplineSample[] = [];
  const straightSteps = Math.round(straightM);
  const arcSteps = Math.max(4, Math.round(Math.PI * radiusM));
  const push = (x: number, y: number): void => {
    samples.push({
      x,
      y,
      width: TURNFEEL_ROAD_WIDTH_M,
      segmentIndex: 0,
      sampleIndexWithinSegment: samples.length,
      globalIndex: samples.length,
    });
  };
  for (let index = 0; index < straightSteps; index += 1) {
    push(index, -radiusM);
  }
  for (let step = 0; step < arcSteps; step += 1) {
    const angle = -Math.PI / 2 + (step / arcSteps) * Math.PI;
    push(straightM + radiusM * Math.cos(angle), radiusM * Math.sin(angle));
  }
  for (let index = 0; index < straightSteps; index += 1) {
    push(straightM - index, radiusM);
  }
  for (let step = 0; step < arcSteps; step += 1) {
    const angle = Math.PI / 2 + (step / arcSteps) * Math.PI;
    push(radiusM * Math.cos(angle), radiusM * Math.sin(angle));
  }
  return samples;
};

/**
 * The frozen tight-corner fixture (the R = 8 m stadium - the tightest
 * MEASURED layout of the repo).
 */
const TURNFEEL_TIGHT_SAMPLES = buildTurnfeelStadiumSamples(
  TURNFEEL_TIGHT_CORNER_RADIUS_M,
  TURNFEEL_TIGHT_CORNER_STRAIGHT_M,
);

/**
 * The tightest (max |curvature|) racing-line point of a layout.
 */
interface TurnfeelCorner {
  /** The index of the tightest racing-line point. */
  readonly index: number;
  /** The SIGNED curvature of that point (1/m). */
  readonly curvaturePerM: number;
  /** The published target speed of that point (m/s). */
  readonly targetSpeedMps: number;
  /** The SIGNED lateral offset of that point (m). */
  readonly offsetM: number;
  /** The kinematic turn rate the corner DEMANDS (rad/s). */
  readonly requiredRateRadPerS: number;
}

/**
 * Resolves the tightest (max |curvature|) racing-line point of a layout.
 *
 * THE KINEMATIC REQUIREMENT (never a literal): a corner of curvature
 * `kappa` demands the turn rate `v x kappa` at the speed `v`, so the
 * required rate is COMPUTED from the two PUBLISHED quantities - the
 * racing line's curvature and its target speed.
 *
 * @param samples - The frozen spline samples of the layout.
 * @returns The tightest racing-line point of the layout.
 */
const tightestCornerOf = (
  samples: readonly SplineSample[],
): TurnfeelCorner => {
  const line = buildTrackRacingLine(samples);
  let index = 0;
  let curvaturePerM = 0;
  for (let candidate = 0; candidate < line.length; candidate += 1) {
    const magnitude = Math.abs(line[candidate].curvaturePerM);
    if (magnitude > curvaturePerM) {
      curvaturePerM = magnitude;
      index = candidate;
    }
  }
  const point = line[index];
  return {
    index,
    curvaturePerM: point.curvaturePerM,
    targetSpeedMps: point.targetSpeedUnitsPerS,
    offsetM: point.offsetM,
    requiredRateRadPerS: point.targetSpeedUnitsPerS * curvaturePerM,
  };
};

/**
 * The reading of the drift ENGAGE boundary lane.
 */
interface DriftEngageLane {
  /** The steer axis the drift model received (the RAW steer, never softened). */
  readonly driftSteerAxis: number | undefined;
  /** The committed slide of the LAST tick (rad). */
  readonly committedSlipRad: number;
}

/**
 * Runs a drift ENGAGE boundary lane and MEASURES the boundary.
 *
 * THE MEASUREMENT (the observable): the drift commitment is spied on (the
 * Step 14T lane pattern), so the lane reads the RAW steer axis the drift
 * model receives - a softened axis would silently re-scale the 0.5 gate -
 * and the committed slide of the lane's LAST tick says whether the slide
 * ENGAGED at all.
 *
 * @param steer - The steer axis of the lane.
 * @returns The measured reading of the lane.
 */
const runDriftEngageLane = (steer: number): DriftEngageLane => {
  const start: PhysicsCarState = {
    ...createInitialPhysicsCarState(42),
    speed: TURNFEEL_DRIFT_ENGAGE_SPEED_MPS,
    drift: {
      committed: { slip: 0, turnSign: 0, reArmed: true },
      pose: { pose: 0, poseRate: 0, articulation: 0 },
    },
  };
  const surfaceResolver = (): FixtureSurface => 'road';
  const spy = jest.spyOn(driftModelModule, 'stepDriftCommitment');
  try {
    let state = start;
    for (let tick = 0; tick < TURN_RATE_LANE_TICKS; tick += 1) {
      state = stepPhysics(
        state,
        { throttle: 1, brake: 0, steer },
        surfaceResolver,
      );
    }
    const firstCall = spy.mock.calls[0];
    const driftInput = firstCall
      ? firstCall[1]
      : ({} as unknown as DriftStepInput);
    return {
      driftSteerAxis: driftInput.steer,
      committedSlipRad: state.drift.committed.slip,
    };
  } finally {
    spy.mockRestore();
  }
};

/**
 * The TURNING FEEL contracts (Step 14Y, slice `P1-turnfeel-red`; AC-098).
 *
 * AC-098 (the turning feel + the AI driveability): the commanded turn rate
 * must be able to HOLD the corner the car is driving, at the speed the car
 * is driving it at - the arcade feel realizes a CONSTANT 44.5 m minimum turn
 * radius today, so every corner tighter than that is unholdable at ANY
 * speed.
 *
 * THE RED SET: RED-1 is DISCRIMINATING (it fails today on its own
 * assertion, on the tightest MEASURED layout of the repo). RED-5 is a
 * LABELLED PRESERVATION GUARD SET that holds today and is NEVER counted as
 * a red: it pins the crawl band's ABSOLUTE rate (the N-7 carry-forward), the
 * mild reduction's RELATIVE bite above the band and the drift gates' EXACT
 * meaning.
 */
describe('the TURNING FEEL wave (Step 14Y, slice P1-turnfeel-red; AC-098)', () => {
  describe('RED-1 (the USER channel can HOLD the tightest corner)', () => {
    // [RED-14Y-1-USER-HOLDS-TIGHTEST-CORNER]
    it('AC-098: the USER channel commanded rate can HOLD the tightest corner of the tightest MEASURED layout (the commanded rate meets the kinematic requirement) (RED-14Y-1-USER-HOLDS-TIGHTEST-CORNER)', () => {
      // Arrange
      const corner = tightestCornerOf(TURNFEEL_TIGHT_SAMPLES);
      const lane = runTurnRateLane(
        corner.targetSpeedMps,
        TURN_RATE_USER_INPUT,
      );

      // Act + Assert
      // The NON-VACUITY GUARD comes FIRST, and it has TWO parts: the
      // probe really IS a corner (never a straight) and the corner
      // demands MORE than the model's own nominal law at that speed (so
      // the sufficiency clause below can never be satisfied VACUOUSLY by
      // a corner that asks for LESS than the law already commands).
      expect(Math.abs(corner.curvaturePerM)).toBeGreaterThan(
        NEATRACING_RACING_LINE_STRAIGHT_CURVATURE_PER_M,
      );
      expect(corner.requiredRateRadPerS).toBeGreaterThan(
        nominalLawRateOf(corner.targetSpeedMps),
      );
      // THE DISCRIMINATING CLAUSE: the commanded rate must MEET the
      // kinematic requirement of the corner.
      //
      // THE MEASURED post-fix state: the USER law commands 2.4475 rad/s
      // at the R = 8 m corner's target speed (~17.8 m/s), against a
      // required 2.225 rad/s (a 1.1x margin) - the clause is GREEN on the
      // post-fix bytes.
      expect(lane.commandedRateRadPerS).toBeGreaterThanOrEqual(
        corner.requiredRateRadPerS,
      );
    });

    // [RED-14Y-1-SERVED-LAYOUT-READING]
    it('AC-098: the SERVED layout reading of the same contract (the served track\'s tightest corner is ALREADY holdable) (RED-14Y-1-SERVED-LAYOUT-READING) - PRESERVATION GUARD (GREEN on arrival, NOT a red)', () => {
      // Arrange
      const spec = generateTrack({
        seed: 42,
        layoutVersion: 1,
        sizeBucket: 'large',
      });
      const corner = tightestCornerOf(spec.splineSamples);
      const lane = runTurnRateLane(
        corner.targetSpeedMps,
        TURN_RATE_USER_INPUT,
      );

      // Act + Assert
      // PRESERVATION GUARD, NOT A RED: the MEASURED current-model state
      // already HOLDS the served track's tightest corner (the commanded
      // 1.3027 rad/s meets the required 0.5830 rad/s), so this lane is
      // GREEN on arrival.
      //
      // THE RECORD (WHY the RED-1 lane probes the TIGHTEST MEASURED
      // layout instead of the served one): the served layout's tightest
      // corner has a 116.5 m radius, so the served track is ALREADY
      // holdable - a served-track reading of RED-1 would be VACUOUS, and
      // RED-1's own non-vacuity guard forbids that.
      // The NON-VACUITY GUARD comes FIRST: the lane must REALLY turn.
      expect(lane.commandedRateRadPerS).toBeGreaterThan(0);
      expect(lane.commandedRateRadPerS).toBeGreaterThanOrEqual(
        corner.requiredRateRadPerS,
      );
    });
  });

  describe('RED-5 (the PRESERVATION GUARDS - NOT reds)', () => {
    // [RED-14Y-5A-CRAWL-BAND-ABSOLUTE]
    it('AC-098: the crawl-band ABSOLUTE near-standstill rate is UNCHANGED (the N-7 carry-forward pin) (RED-14Y-5A-CRAWL-BAND-ABSOLUTE) - PRESERVATION GUARD (NOT a red)', () => {
      // Arrange
      const lane = runTurnRateLane(
        TURN_RATE_CRAWL_SPEED_MPS,
        TURN_RATE_USER_INPUT,
      );

      // Act + Assert
      // PRESERVATION GUARD, NOT A RED: the MEASURED current-model state
      // already commands exactly that rate, so this lane is GREEN on
      // arrival.
      //
      // THE N-7 CARRY-FORWARD (the load-bearing ABSOLUTE pin): the crawl
      // band's near-standstill rate is PINNED to its measured value, so a
      // global raise of the nominal turn rate can never reach the crawl
      // band - the crawl feel (the near-standstill handling) stays
      // ratifiable.
      expect(lane.commandedRateRadPerS).toBeCloseTo(
        TURNFEEL_CRAWL_BAND_ABSOLUTE_RAD_PER_S,
        12,
      );
    });

    // [RED-14Y-5B-TOP-SPEED-REDUCTION-BITES]
    it('AC-092: the mild reduction still BITES above the crawl band (the top-speed USER rate stays STRICTLY BELOW the nominal law of that speed) (RED-14Y-5B-TOP-SPEED-REDUCTION-BITES) - PRESERVATION GUARD (NOT a red)', () => {
      // Arrange
      const lane = runTurnRateLane(
        TURN_RATE_TOP_SPEED_MPS,
        TURN_RATE_USER_INPUT,
      );

      // Act + Assert
      // PRESERVATION GUARD, NOT A RED: the MEASURED current-model state
      // already reads 1.6 rad/s against the nominal 2 rad/s, so this lane
      // is GREEN on arrival.
      //
      // THE RELATIVE FORM (why this lane is NOT the 14T lane): the
      // comparison axis is the DERIVED nominal law of the top speed
      // (`nominalLawRateOf`), never the nominal CONSTANT, so the clause
      // SURVIVES a global raise of the nominal turn rate - only the
      // crawl-band ABSOLUTE pin (RED-14Y-5A) breaks under a raise.
      //
      // THE 14T CRAWL-BAND RELATIVE CLAUSE is NOT duplicated here: it is
      // already pinned by `[RED-14T-1-CRAWL-BAND]`.
      expect(lane.commandedRateRadPerS).toBeGreaterThan(0);
      expect(lane.commandedRateRadPerS).toBeLessThan(
        nominalLawRateOf(TURN_RATE_TOP_SPEED_MPS),
      );
    });

    // [RED-14Y-5C-DRIFT-GATES]
    it('AC-098: the drift gates keep their EXACT meaning (the 0.5 / 0.15 / 0.9 steer thresholds are unchanged) (RED-14Y-5C-DRIFT-GATES) - PRESERVATION GUARD (NOT a red)', () => {
      // Act + Assert
      // PRESERVATION GUARD, NOT A RED: the MEASURED current-model state
      // already publishes exactly those thresholds, so this lane is GREEN
      // on arrival - it is the RAIL a turn-rate change must stay on (a
      // change that touched the steer axis would silently re-scale every
      // gate).
      expect(NEATRACING_DRIFT_STEER_ENGAGE_THRESHOLD).toBe(0.5);
      expect(NEATRACING_DRIFT_STEER_NEUTRAL_THRESHOLD).toBe(0.15);
      expect(NEATRACING_DRIFT_COUNTERSTEER_MIN_STEER).toBe(0.9);
    });

    // [RED-14Y-5C-DRIFT-ENGAGE-AT-GATE]
    it('AC-098: the drift ENGAGE boundary keeps its EXACT meaning (the engage gate still ENGAGES the slide) (RED-14Y-5C-DRIFT-ENGAGE-AT-GATE) - PRESERVATION GUARD (NOT a red)', () => {
      // Arrange
      const engageLane = runDriftEngageLane(TURNFEEL_DRIFT_ENGAGE_STEER);

      // Act + Assert
      // PRESERVATION GUARD, NOT A RED: the MEASURED current-model state
      // already commits the slide at the engage gate, so this lane is
      // GREEN on arrival.
      // The NON-VACUITY GUARD comes FIRST: the drift model must REALLY
      // receive the RAW steer axis (a softened axis would make the gate
      // meaningless).
      expect(engageLane.driftSteerAxis).toBe(TURNFEEL_DRIFT_ENGAGE_STEER);
      // The ENGAGE side of the boundary: the slide COMMITS.
      expect(Math.abs(engageLane.committedSlipRad)).toBeGreaterThan(0);
    });

    // [RED-14Y-5C-DRIFT-NO-ENGAGE-BELOW-GATE]
    it('AC-098: the drift ENGAGE boundary keeps its EXACT meaning (just BELOW the engage gate the slide never commits) (RED-14Y-5C-DRIFT-NO-ENGAGE-BELOW-GATE) - PRESERVATION GUARD (NOT a red)', () => {
      // Arrange
      const noEngageLane = runDriftEngageLane(TURNFEEL_DRIFT_NO_ENGAGE_STEER);

      // Act + Assert
      // PRESERVATION GUARD, NOT A RED: the MEASURED current-model state
      // never commits the slide below the gate, so this lane is GREEN on
      // arrival.
      // The NON-VACUITY GUARD comes FIRST: the drift model must REALLY
      // receive the RAW steer axis on this lane too.
      expect(noEngageLane.driftSteerAxis).toBe(TURNFEEL_DRIFT_NO_ENGAGE_STEER);
      // The HOLDING side of the boundary: the slide NEVER commits.
      expect(noEngageLane.committedSlipRad).toBe(0);
    });

    // [RED-14Y-5C-STEER-AXIS-BINARY]
    it('AC-098: the steer AXIS stays BINARY above the crawl band (the mirrored steer commands the mirrored rate) (RED-14Y-5C-STEER-AXIS-BINARY) - PRESERVATION GUARD (NOT a red)', () => {
      // Arrange
      const leftLane = runTurnRateLane(
        TURN_RATE_TOP_SPEED_MPS,
        TURN_RATE_USER_INPUT,
      );
      const rightLane = runTurnRateLane(TURN_RATE_TOP_SPEED_MPS, {
        throttle: 1,
        brake: 0,
        steer: -1,
      });

      // Act + Assert
      // PRESERVATION GUARD, NOT A RED: the MEASURED current-model state
      // already mirrors the rate, so this lane is GREEN on arrival.
      //
      // THE LANE (why it runs ABOVE the band): the crawl-band mirroring
      // is already pinned by `[RED-14T-3-ABSENT-PERCENT-STEER-LAW]`, so
      // this lane measures the regime where the mild factor actually
      // bites - the mirroring must survive the factor too.
      // The NON-VACUITY GUARD comes FIRST: BOTH lanes must REALLY turn.
      expect(leftLane.commandedRateRadPerS).toBeGreaterThan(0);
      expect(rightLane.commandedRateRadPerS).toBeLessThan(0);
      // The steer AXIS is BINARY and SYMMETRIC: the mirrored steer
      // commands the mirrored rate.
      expect(rightLane.commandedRateRadPerS).toBeCloseTo(
        -leftLane.commandedRateRadPerS,
        12,
      );
    });
  });
});
