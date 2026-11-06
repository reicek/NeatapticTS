/**
 * Deterministic green tests for the P1 AI autopilot seam (user-gap patch
 * batch 108): the served race page must open as a running race, so the
 * `ai` mode samples `stepAutopilotInput` instead of the keyboard seam.
 *
 * The fixture is a straight +x lane-center loop; every expectation is a
 * numeric single-expect behavioral check and the same (car, samples) pair
 * must yield identical inputs across repeats (batch-83 replay rung).
 *
 * @module
 */

import {
  NEATRACING_AUTOPILOT_FULL_BRAKE_ERROR_RAD,
  NEATRACING_AUTOPILOT_LOOKAHEAD_M,
  resolveAutopilotRacingLine,
  stepAutopilotInput,
  type AutopilotCarSnapshot,
} from './ai.autopilot';
import { createTrackSpecForAutopilotFixture } from './ai.autopilot.fixtures';
import {
  createTrackSurfaceResolver,
} from './collision.utils';
import {
  createInitialPhysicsCarState,
  NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
  NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
  NEATRACING_PHYSICS_STEERING_RATE,
  NEATRACING_PHYSICS_TURN_RATE_REDUCTION_BAND_UNITS_PER_S,
  stepPhysics,
  type PhysicsCarState,
  type PhysicsStepInput,
} from './physics.step.service';
import { NEATRACING_TRACK_GRASS_APRON_WIDTH_M } from '../track/track.slabs';
import { NEATRACING_RACING_LINE_STRAIGHT_CURVATURE_PER_M } from '../track/track.racing-line';
import { generateTrack } from '../../racing_curriculum/track/track.generator';
import type {
  SplineSample,
  TrackSegment,
} from '../../racing_curriculum/track/track.generator.types';

/** Fixture straight loop for steering/throttle law checks. */
const STRAIGHT_LOOP = createTrackSpecForAutopilotFixture();

/**
 * The MEASURED steer of the 2 m-off-center lane (rad).
 *
 * THE RE-PIN (Step 14M, slice `P1-ai-centerline`): AC-083 adds the
 * CENTERLINE-CORRECTING steer term, so the TWO pre-existing pins that
 * asserted the LEGACY chase-only value (`atan2(2, 12) * 2.2`) are
 * INVALIDATED BY DESIGN (the RED slice carried them FORWARD unchanged
 * for exactly this reason). The legacy literal is DELETED and REPLACED
 * one-for-one by this MEASURED value - NEVER RELAXED: the assertion
 * form, the precision and the lane semantics are UNCHANGED.
 *
 * THE NAMED BASIS (the two terms the seam now commands on the STRAIGHT
 * fixture, where the racing line's curvature is EXACTLY zero, so the
 * curvature feed-forward contributes NOTHING):
 *
 * - the RETAINED heading-error chase term `atan2(2, 12) * 2.2` =
 *   0.363327090312179,
 * - the CENTERLINE-CORRECTING term `2 m x (2.2 rad^-1 / 12 m)` =
 *   0.3666666666666667,
 *
 * so the MEASURED steer of the lane is 0.7299937569788457.
 */
const AI_OFF_CENTER_STEER_PIN_RAD = 0.7299937569788457;

describe('stepAutopilotInput (P1 ai-mode race automation)', () => {
  it('returns a neutral input for an empty sample list (no loop, no drive)', () => {
    const input = stepAutopilotInput({ x: 0, y: 0, heading: 0, speed: 0 }, []);
    // The 14T AI turn-rate-percent channel is ALWAYS emitted (the AC-089
    // ledger row P-T-3), so the neutral input carries the DEFINED
    // `turnRatePercent: 0`. The literal is DELETED and REPLACED
    // one-for-one: the assertion form and the other three fields are
    // UNCHANGED.
    expect(input).toEqual({
      throttle: 0,
      brake: 0,
      steer: 0,
      turnRatePercent: 0,
    });
  });

  it('drives straight ahead with full throttle on the centerline (no steer)', () => {
    const car = { x: 0, y: 0, heading: 0, speed: 10 };
    const input = stepAutopilotInput(car, STRAIGHT_LOOP);
    // The 14T AI turn-rate-percent channel is ALWAYS emitted (the AC-089
    // ledger row P-T-4) and the centerline steer is EXACTLY 0, so the
    // emitted percent is EXACTLY +0. The literal is DELETED and REPLACED
    // one-for-one: the assertion form and the other three fields are
    // UNCHANGED.
    expect(input).toEqual({
      throttle: 1,
      brake: 0,
      steer: 0,
      turnRatePercent: 0,
    });
  });

  it('steers right (negative) when the car nose points off-line to the left', () => {
    const car = { x: 0, y: 0, heading: Math.PI / 8, speed: 10 };
    const input = stepAutopilotInput(car, STRAIGHT_LOOP);
    expect(input.steer).toBeCloseTo(-(Math.PI / 8) * 2.2, 9);
    expect(input.throttle).toBe(0);
    expect(input.brake).toBe(0);
  });

  it('steers left (positive) when the car sits off-line to the right', () => {
    const car = { x: 5, y: -2, heading: 0, speed: 10 };
    const input = stepAutopilotInput(car, STRAIGHT_LOOP);
    // THE RE-PIN (Step 14M, AC-083): the LEGACY literal
    // `atan2(2, 12) * 2.2` is INVALIDATED BY DESIGN (the
    // centerline-correcting term ADDS to the chase term on this lane),
    // so it is DELETED and REPLACED one-for-one by the MEASURED literal
    // `AI_OFF_CENTER_STEER_PIN_RAD` - NEVER RELAXED: the assertion form,
    // the precision and the lane semantics are UNCHANGED.
    expect(input.steer).toBeCloseTo(AI_OFF_CENTER_STEER_PIN_RAD, 9);
    expect(input.throttle).toBe(1);
    expect(input.brake).toBe(0);
  });

  it('brakes and cuts throttle past the full-brake error threshold', () => {
    const car = { x: 0, y: 0, heading: Math.PI / 2, speed: 30 };
    const input = stepAutopilotInput(car, STRAIGHT_LOOP);
    expect(Math.PI / 2).toBeGreaterThan(
      NEATRACING_AUTOPILOT_FULL_BRAKE_ERROR_RAD,
    );
    expect(input.throttle).toBe(0);
    expect(input.steer).toBe(-1);
  });

  it('is deterministic for repeated identical inputs (replay rung)', () => {
    const car = {
      x: STRAIGHT_LOOP[2].x,
      y: STRAIGHT_LOOP[2].y - 1.5,
      heading: 0.4,
      speed: 8,
    };
    const first = stepAutopilotInput(car, STRAIGHT_LOOP);
    const second = stepAutopilotInput(car, STRAIGHT_LOOP);
    expect(JSON.stringify(first)).toBe(JSON.stringify(second));
  });
});

/* ------------------------------------------------------------------------- *
 * STEP 14T (slice `P1-turnrate-red`): the AI turn-rate-percent channel.
 * ------------------------------------------------------------------------- */

/**
 * The RED-4 WIDENED LOCAL VIEW of the autopilot step input.
 *
 * The turn-rate-percent channel is a TYPE MEMBER of `PhysicsStepInput`
 * (and NOT a module export), so it CANNOT be named by an import: the
 * RED shape is a named RUNTIME failure through this local view (the
 * Step 14R red-slice residual (2) lesson) - a MISSING-IMPORT failure
 * would be a file-level compile error that would mask EVERY sibling
 * contract in this file.
 */
interface AiTurnRateInput extends PhysicsStepInput {
  /** The OPTIONAL turn-rate percent channel (Step 14T, AC-093). */
  turnRatePercent?: number;
}

/**
 * The 14T AI snapshot: the car sits off-line to the RIGHT of the loop
 * centerline, so the autopilot steers LEFT (a POSITIVE, non-zero steer).
 *
 * The lane is NON-VACUOUS: the autopilot steer reads
 * `AI_OFF_CENTER_STEER_PIN_RAD` (0.7299937569788457 - the RETAINED
 * chase term plus the AC-083 centerline-correcting term), so a zero
 * steer can never satisfy the rate clauses.
 */
const AI_TURN_RATE_SNAPSHOT: AutopilotCarSnapshot = {
  x: 5,
  y: -2,
  heading: 0,
  speed: 10,
};

/**
 * The 14T AI FAST lane speed (m/s).
 *
 * The lane isolates the SPEED reading: the geometry (and so the steer
 * law) is unchanged, so the ONLY thing that may differ between the two
 * AI inputs is the speed-derived turn rate.
 */
const AI_TURN_RATE_FAST_SPEED_MPS = 40;

/**
 * The 14T AI nominal turn rate (rad/s) - the COMPARISON AXIS.
 *
 * The nominal rate is the steer command times the steering rate
 * (`steer * NEATRACING_PHYSICS_STEERING_RATE`): the rate a 100 percent
 * turn-rate channel commands, independent of the speed.
 */
const AI_TURN_RATE_NOMINAL_RAD_PER_S = NEATRACING_PHYSICS_STEERING_RATE;

/**
 * The 14T AI rate-adoption tolerance (rad/s).
 *
 * The rate the physics commands is MEASURED through the published
 * heading delta (the delta divided by the fixed timestep), so the
 * reading carries a float64 rounding of a few ULP. The tolerance
 * absorbs that noise ONLY.
 */
const AI_RATE_ADOPTION_TOLERANCE_RAD_PER_S = 1e-9;

/**
 * Reads the NOT-YET-EXISTING turn-rate percent off an autopilot input.
 *
 * The field `turnRatePercent` is the AC-093 contract and does not exist
 * on `PhysicsStepInput` yet, so the read goes through this LOCAL
 * widened view: the field is `undefined` today, which is what makes the
 * RED-4 contracts fail on their OWN assertion instead of on a
 * file-level compile error that would mask every sibling contract in
 * this file.
 *
 * @param input - The autopilot input the seam produced.
 * @returns The turn-rate percent, or `undefined` while the seam sets
 *   none at all.
 */
const aiTurnRatePercentOf = (input: PhysicsStepInput): number | undefined =>
  (input as AiTurnRateInput).turnRatePercent;

/**
 * Runs ONE 14T AI lane: the autopilot seam produces the input and the
 * physics service turns it into a MEASURED turn rate.
 *
 * THE MEASUREMENT (the observable): the commanded turn rate is read
 * from the PUBLISHED heading delta of the FIRST tick (the delta divided
 * by the fixed timestep), so the lane measures the turn the car
 * actually got - never a re-implementation of the service law. The
 * FIRST tick is the only one that carries the lane speed, so the lane
 * speed IS the tick-start speed.
 *
 * @param snapshot - The autopilot snapshot the lane starts from.
 * @returns The produced autopilot input and the measured turn rate of
 *   the first tick (rad/s).
 */
const runAiTurnRateLane = (
  snapshot: AutopilotCarSnapshot,
): { input: PhysicsStepInput; commandedRateRadPerS: number } => {
  const input = stepAutopilotInput(snapshot, STRAIGHT_LOOP);
  const start: PhysicsCarState = {
    ...createInitialPhysicsCarState(42),
    x: snapshot.x,
    y: snapshot.y,
    heading: snapshot.heading,
    speed: snapshot.speed,
    drift: {
      committed: { slip: 0, turnSign: 0, reArmed: true },
      pose: { pose: 0, poseRate: 0, articulation: 0 },
    },
  };
  const surfaceResolver = (): 'road' => 'road';
  const next = stepPhysics(start, input, surfaceResolver);
  return {
    input,
    commandedRateRadPerS:
      (next.heading - start.heading) /
      NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
  };
};

/**
 * The AI turn-rate-percent contracts (Step 14T, slice `P1-turnrate-red`).
 *
 * AC-093 (two separate laws on the same steer axis, S-7): the AI does
 * NOT steer through the USER channel - it drives the physics through the
 * DEDICATED turn-rate-percent channel, so the AI owns the turn RATE
 * outright while the user keeps the arrow-key steer axis. The AI must
 * therefore SET a DEFINED percent, and that percent must READ the
 * snapshot speed (the AI's turn rate follows the speed it actually has).
 *
 * THE BEHAVIOUR-PRESERVING RE-EXPRESSION (the design seam the AI slice
 * will implement): the percent is SIGNED and resolved BEFORE the user
 * softening, so `rate = STEERING_RATE * (percent / 100)`. The AI's
 * behaviour-preserving percent is `100 * steerAI * speedPercent`, which
 * resolves to exactly today's rate - the two laws never collide and the
 * AI keeps the FULL range (a percent above 100 commands a rate above the
 * nominal rate).
 *
 * THE RED SET: the two RED-4 lanes below are RED today (the channel
 * does not exist and the seam never reads the speed), while the two
 * PRESERVATION GUARDS are GREEN on arrival - they pin the AI steer law
 * and the AI turn rate so the adoption stays BEHAVIOR-PRESERVING.
 */
describe('the AI turn-rate-percent channel (Step 14T, slice P1-turnrate-red)', () => {
  describe('RED-4 (the two laws never collide, and the AI adoption is behavior-preserving)', () => {
    // [RED-14T-4-AI-SETS-PERCENT]
    it('AC-093: the autopilot sets a DEFINED turn-rate percent on its physics input (RED-4)', () => {
      // Arrange
      const lane = runAiTurnRateLane(AI_TURN_RATE_SNAPSHOT);

      // Act + Assert
      // The NON-VACUITY GUARD comes FIRST: the AI must REALLY steer (a
      // zero steer would satisfy the percent clause VACUOUSLY).
      expect(lane.input.steer).not.toBe(0);
      // THE DISCRIMINATING CLAUSE: the AI drives the physics through the
      // turn-rate-percent channel, so it SETS a defined (finite) percent.
      //
      // The MEASURED current-model state: the seam returns ONLY
      // `{ throttle, brake, steer }`, so the percent is `undefined` today
      // and the clause is RED.
      expect(
        Number.isFinite(aiTurnRatePercentOf(lane.input)),
      ).toBe(true);
    });

    // [RED-14T-4-AI-READS-SPEED]
    it('AC-093: the autopilot turn rate READS the snapshot speed (the AI output differs between two speeds on the same geometry) (RED-4)', () => {
      // Arrange
      const slowLane = runAiTurnRateLane(AI_TURN_RATE_SNAPSHOT);
      const fastLane = runAiTurnRateLane({
        ...AI_TURN_RATE_SNAPSHOT,
        speed: AI_TURN_RATE_FAST_SPEED_MPS,
      });

      // Act + Assert
      // The NON-VACUITY GUARD comes FIRST: the AI must REALLY steer on
      // BOTH lanes (a zero steer would satisfy the relation VACUOUSLY).
      expect(slowLane.input.steer).not.toBe(0);
      expect(fastLane.input.steer).not.toBe(0);
      // THE DISCRIMINATING CLAUSE: the AI turn rate follows the speed
      // the car actually has, so the SAME geometry at two speeds must
      // NOT produce the same AI command.
      //
      // The MEASURED current-model state: the seam NEVER reads the
      // snapshot speed, so both lanes produce the IDENTICAL input and the
      // clause is RED.
      expect(fastLane.input).not.toEqual(slowLane.input);
    });

    // [RED-14T-4-AI-RATE-PRESERVED]
    it('AC-093: the AI turn rate the physics commands is PRESERVED by the percent channel (the adoption is behavior-preserving) (RED-4) - PRESERVATION GUARD', () => {
      // Arrange
      const lane = runAiTurnRateLane(AI_TURN_RATE_SNAPSHOT);
      const speedPercent =
        AI_TURN_RATE_SNAPSHOT.speed / NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S;

      // Act + Assert
      // PRESERVATION GUARD, NOT A RED: the MEASURED current-model state
      // already commands exactly that rate (the steer-axis law), so this
      // lane is GREEN on arrival - it is the RAIL the AI slice must keep
      // (the percent channel must not change the AI's driving).
      // The NON-VACUITY GUARD comes FIRST: the lane must REALLY turn.
      expect(lane.commandedRateRadPerS).toBeGreaterThan(0);
      // The commanded rate stays `steer * STEERING_RATE * speedPercent` -
      // the behaviour the AI had BEFORE the channel was adopted.
      expect(
        Math.abs(
          lane.commandedRateRadPerS -
            lane.input.steer * AI_TURN_RATE_NOMINAL_RAD_PER_S * speedPercent,
        ),
      ).toBeLessThanOrEqual(AI_RATE_ADOPTION_TOLERANCE_RAD_PER_S);
    });

    // [RED-14T-4-AI-STEER-LAW-UNCHANGED]
    it('AC-093: the AI STEER law is UNCHANGED by the percent channel (the two laws never collide) (RED-4) - PRESERVATION GUARD', () => {
      // Arrange
      const lane = runAiTurnRateLane(AI_TURN_RATE_SNAPSHOT);

      // Act + Assert
      // PRESERVATION GUARD, NOT A RED: the MEASURED current-model state
      // reads exactly that steer, so this lane is GREEN on arrival.
      // The NON-VACUITY GUARD comes FIRST: the AI must REALLY steer.
      expect(lane.input.steer).not.toBe(0);
      // The steer law keeps the RETAINED heading-error chase term and
      // GAINS the AC-083 centerline-correcting term, so the LEGACY pin
      // (`atan2(2, 12) * 2.2`) is INVALIDATED BY DESIGN: it is DELETED
      // and REPLACED one-for-one by the MEASURED literal
      // `AI_OFF_CENTER_STEER_PIN_RAD` - NEVER RELAXED (the assertion
      // form and the precision are UNCHANGED). The percent channel is a
      // SECOND channel, never a REPLACEMENT of the steer axis.
      expect(lane.input.steer).toBeCloseTo(
        AI_OFF_CENTER_STEER_PIN_RAD,
        12,
      );
    });
  });
});

/* ------------------------------------------------------------------------- *
 * STEP 14M (slice `P1-ai-centerline`): the AI centerline-keeping wave.
 * The RED contracts were authored by the sibling slice
 * `P1-ai-centerline-red`; the lane IDs keep the RED prefix.
 * ------------------------------------------------------------------------- */

/**
 * The Step 14M fixture road width (m) - the same 12 m road the shared
 * repo fixture uses.
 */
const CENTERLINE_ROAD_WIDTH_M = 12;

/**
 * The tight-corner radius (m) of the curvature fixture.
 *
 * TIGHT on purpose: a wide corner curves almost like a straight, so a
 * loose "is this a corner?" threshold could classify it as a straight and
 * make the curvature contracts VACUOUS. At R = 8 m the curvature is
 * 0.125 m^-1 - far from any straight.
 */
const CENTERLINE_TIGHT_CORNER_RADIUS_M = 8;

/** The straight length (m) of the tight-corner fixture. */
const CENTERLINE_TIGHT_CORNER_STRAIGHT_M = 40;

/** The lap-loop radius (m) of the centerline-measurement fixture. */
const CENTERLINE_LAP_RADIUS_M = 60;

/** The straight length (m) of the lap-loop fixture. */
const CENTERLINE_LAP_STRAIGHT_M = 60;

/**
 * The lap length in ticks (20 s of the 60 Hz fixed step - a little over
 * two full laps of the lap-loop fixture).
 */
const CENTERLINE_LAP_TICK_COUNT = 1200;

/** The drivable half-width of the fixture road (m) - `width / 2`. */
const CENTERLINE_DRIVABLE_HALF_WIDTH_M = CENTERLINE_ROAD_WIDTH_M / 2;

/**
 * The OUTER EDGE of the drivable band (m) - the drivable half-width plus
 * the PUBLISHED grass-apron width (the width of the drivable off-road
 * band that surrounds the road).
 *
 * THE FORMER ONE BOUND (Step 14W, slice `P1-sand-service`, AC-094
 * (d)/(g)) - RETIRED (fix-packet-P1-sand-red-iteration-3, AC-090 VERDICT
 * 8): the ONE containment bound once sat HERE, on the outer edge of the
 * band, NOT on the drivable edge, and the band is DRIVABLE, so a car
 * inside the band was never carried back to the road. The BAND GEOMETRY
 * survives as the band law; the anti-masking guard below is measured
 * against this band geometry.
 */
const CENTERLINE_BAND_OUTER_EDGE_M =
  CENTERLINE_DRIVABLE_HALF_WIDTH_M + NEATRACING_TRACK_GRASS_APRON_WIDTH_M;

/**
 * The AC-085 centerline bound (m).
 *
 * THE BOUND IS INSIDE THE BAND (the anti-masking rule): the band law's
 * drivable surface reaches past the drivable edge (the drivable edge
 * plus the published grass apron), so a centerline bound of 3 m - well
 * inside that band geometry - can still fail while nothing clamps the
 * defect away. A masked defect is a FAILED acceptance.
 */
const CENTERLINE_MAX_OFFSET_M = 3;

/**
 * The road-surface resolver of the Step 14M lanes (the fixture is road
 * everywhere, so the surface never changes under the car).
 */
const CENTERLINE_ROAD_SURFACE_RESOLVER = (): 'road' => 'road';

/**
 * Builds the Step 14M stadium fixture: a straight along +x, a right
 * half-circle, a straight along -x and a left half-circle, all traversed
 * COUNTER-CLOCKWISE.
 *
 * THE LOCAL FIXTURE (owned by this red slice): the shared repo fixture
 * (`ai.autopilot.fixtures.ts`) is a STRAIGHT loop only, so the curvature
 * and the lap contracts need a corner of their own. The builder below is
 * the MEASURED fixture the recorded numbers in the plan come from - it
 * is not a copy of any production layout.
 *
 * @param radiusM - The corner radius (m).
 * @param straightM - The straight length (m).
 * @returns The frozen spline samples of the closed loop.
 */
const buildStadiumSamples = (
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
      width: CENTERLINE_ROAD_WIDTH_M,
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

/** The frozen tight-corner fixture (a straight that ends in an R = 8 m corner). */
const CENTERLINE_TIGHT_CORNER_SAMPLES = buildStadiumSamples(
  CENTERLINE_TIGHT_CORNER_RADIUS_M,
  CENTERLINE_TIGHT_CORNER_STRAIGHT_M,
);

/** The frozen lap-loop fixture (the R = 60 m stadium the lap lanes drive). */
const CENTERLINE_LAP_SAMPLES = buildStadiumSamples(
  CENTERLINE_LAP_RADIUS_M,
  CENTERLINE_LAP_STRAIGHT_M,
);

/**
 * Builds the track chords of a fixture loop (the chords feed the live
 * tick's RETIRED geometry seam - the former containment bound READ
 * them; the seam is a NO-OP since AC-090 VERDICT 8).
 *
 * @param samples - The frozen spline samples of the loop.
 * @returns The closed-loop chords, one per sample pair.
 */
const buildCenterlineSegments = (
  samples: readonly SplineSample[],
): readonly TrackSegment[] =>
  samples.map((from, index) => {
    const to = samples[(index + 1) % samples.length];
    return {
      startX: from.x,
      startY: from.y,
      endX: to.x,
      endY: to.y,
      width: from.width,
    };
  });

/**
 * Measures the SIGNED lateral offset of a point from the road centerline.
 *
 * THE MEASUREMENT (the observable): the offset is the signed
 * perpendicular distance to the NEAREST centerline chord, POSITIVE to the
 * LEFT of the travel direction (the repo's left-normal convention). It
 * is measured DIRECTLY - never inferred from a clamp or a seam - so an
 * off-center car always reads off-center.
 *
 * @param samples - The frozen spline samples of the loop.
 * @param x - The measured world x.
 * @param y - The measured world y.
 * @returns The signed lateral offset (m).
 */
const measureLateralOffsetM = (
  samples: readonly SplineSample[],
  x: number,
  y: number,
): number => {
  let bestDistance = Infinity;
  let bestSigned = 0;
  const total = samples.length;
  for (let index = 0; index < total; index += 1) {
    const from = samples[index];
    const to = samples[(index + 1) % total];
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    const length = Math.hypot(dx, dy);
    const lengthSquared = length * length;
    const t =
      lengthSquared === 0
        ? 0
        : Math.max(
            0,
            Math.min(
              1,
              ((x - from.x) * dx + (y - from.y) * dy) / lengthSquared,
            ),
          );
    const px = from.x + t * dx;
    const py = from.y + t * dy;
    const ex = x - px;
    const ey = y - py;
    const distance = Math.hypot(ex, ey);
    if (distance < bestDistance) {
      bestDistance = distance;
      bestSigned = (ex * -dy + ey * dx) / length;
    }
  }
  return bestSigned;
};

/**
 * Resolves the autopilot's OWN lookahead target for a point.
 *
 * WHY THE FIXTURE MIRRORS THE LAW: the AC-083 lanes isolate the LATERAL
 * term, so the heading error has to be EXACTLY zero - the fixture
 * therefore aims the car at the very sample the seam aims at. AC-083
 * RETAINS the 12 m lookahead, so the mirrored law is a stable contract
 * and not a re-implementation that may drift.
 *
 * @param samples - The frozen spline samples of the loop.
 * @param x - The car world x.
 * @param y - The car world y.
 * @returns The lookahead sample the seam aims at.
 */
const resolveAutopilotLookaheadTarget = (
  samples: readonly SplineSample[],
  x: number,
  y: number,
): SplineSample => {
  let nearestIndex = 0;
  let bestDistance = Infinity;
  for (let index = 0; index < samples.length; index += 1) {
    const sample = samples[index];
    const distance = (sample.x - x) ** 2 + (sample.y - y) ** 2;
    if (distance < bestDistance) {
      bestDistance = distance;
      nearestIndex = index;
    }
  }
  const total = samples.length;
  let remaining = NEATRACING_AUTOPILOT_LOOKAHEAD_M;
  let cursor = nearestIndex;
  for (let step = 0; step < total; step += 1) {
    const from = samples[cursor];
    const to = samples[(cursor + 1) % total];
    remaining -= Math.hypot(to.x - from.x, to.y - from.y);
    cursor = (cursor + 1) % total;
    if (remaining <= 0) {
      break;
    }
  }
  return samples[cursor];
};

/**
 * Aims a car at the autopilot's own lookahead target (the heading that
 * makes the seam's heading error EXACTLY zero).
 *
 * @param samples - The frozen spline samples of the loop.
 * @param x - The car world x.
 * @param y - The car world y.
 * @returns The heading (rad) that aims the car at the lookahead target.
 */
const aimAtLookaheadHeadingRad = (
  samples: readonly SplineSample[],
  x: number,
  y: number,
): number => {
  const target = resolveAutopilotLookaheadTarget(samples, x, y);
  return Math.atan2(target.y - y, target.x - x);
};

/**
 * Reads the commanded drive of an autopilot input (the throttle minus the
 * brake).
 *
 * THE MONOTONIC PROXY: the throttle and the brake are the ONLY speed
 * commands the AI has, so the commanded drive is the observable "the AI
 * asks for less speed" signal.
 *
 * @param input - The autopilot input to read.
 * @returns The commanded drive (`throttle - brake`).
 */
const commandedDriveOf = (input: PhysicsStepInput): number =>
  input.throttle - input.brake;

/**
 * The AI centerline-keeping contracts (Step 14M, slice
 * `P1-ai-centerline`; the RED contracts were authored by the sibling
 * slice `P1-ai-centerline-red`).
 *
 * AC-083 (the centerline-correcting steer term): the autopilot exposes a
 * SIGNED LATERAL OFFSET to the road centerline and steers BACK toward the
 * centerline proportionally to that offset. The new term is ZERO on the
 * centerline, and the 12 m lookahead and the 2.2 rad^-1 gain are
 * RETAINED.
 *
 * AC-084 (the curvature-derived target speed): the target speed comes
 * from the CURVATURE AHEAD instead of the binary throttle/brake law, and
 * the emergent speed behaviour on a straight is unchanged.
 *
 * AC-085 (the measurement contract): the center-keeping contract is
 * INDEPENDENT of the Step 14I containment - RETIRED by
 * fix-packet-P1-sand-red-iteration-3 (AC-090 VERDICT 8) - so the
 * measurement must still be able to FAIL with nothing clamping the car,
 * and a masked defect is a FAILED acceptance.
 *
 * THE RED SET: the three RED lanes below (the two lateral lanes and the
 * corner-slows lane) and the lap lane were RED when the sibling slice
 * `P1-ai-centerline-red` authored them, while the three labelled guards
 * were GREEN on arrival - they pin the behaviour the change must
 * PRESERVE (the zero-offset steer rail, the straight-line speed behaviour
 * and the anti-masking property of the measurement). The implementing
 * slice `P1-ai-centerline` turns the RED set GREEN and ADDS the tight
 * fixture lane (AI-8), because the AC-085 bound has to hold on the
 * tightest measured layout too.
 */
describe('the AI centerline-keeping wave (Step 14M, slice P1-ai-centerline)', () => {
  describe('the centerline-correcting steer term (AC-083)', () => {
    // [RED-14M-AI-1-CENTERLINE-ZERO]
    it('AC-083: the centerline-correcting steer term is ZERO on the centerline (the car already aims at the lookahead target) (RED-14M-AI-1-CENTERLINE-ZERO) - PRESERVATION GUARD', () => {
      // Arrange
      const x = 5;
      const y = 0;
      const heading = aimAtLookaheadHeadingRad(STRAIGHT_LOOP, x, y);

      // Act
      const input = stepAutopilotInput(
        { x, y, heading, speed: 40 },
        STRAIGHT_LOOP,
      );

      // Assert
      // PRESERVATION GUARD, NOT A RED: the MEASURED current-model state
      // already commands exactly zero here (the heading error is exactly
      // zero), so the lane is GREEN on arrival - it is the RAIL the two
      // RED lanes below are measured against.
      //
      // THE AC-083 CLAUSE (still GREEN after the change): the
      // centerline-correcting term is EXACTLY zero here too - the car's
      // own signed lateral offset to the road centerline is EXACTLY zero
      // on the centerline - and the racing line's curvature feed-forward
      // is EXACTLY zero on the straight, so the whole steer command stays
      // EXACTLY zero.
      expect(input.steer).toBe(0);
    });

    // [RED-14M-AI-2-RIGHT-OFFSET-STEERS-LEFT]
    it('AC-083: the autopilot steers BACK toward the centerline (2 m to the RIGHT of the centerline steers LEFT) (RED-14M-AI-2-RIGHT-OFFSET-STEERS-LEFT)', () => {
      // Arrange
      const x = 5;
      const y = -2;
      const heading = aimAtLookaheadHeadingRad(STRAIGHT_LOOP, x, y);

      // Act
      const input = stepAutopilotInput(
        { x, y, heading, speed: 40 },
        STRAIGHT_LOOP,
      );

      // Assert
      // THE MEASURED state (Step 14M, slice `P1-ai-centerline`): the
      // heading error is EXACTLY zero on this lane, so the steer is the
      // centerline-correcting term ALONE - `2 m x (2.2 rad^-1 / 12 m)` =
      // 0.3666666666666667, a POSITIVE (left) steer - so the clause is
      // GREEN (it was RED: the steer was EXACTLY 0, the seam had no
      // lateral term at all).
      expect(input.steer).toBeGreaterThan(0);
    });

    // [RED-14M-AI-3-LEFT-OFFSET-STEERS-RIGHT]
    it('AC-083: the autopilot steers BACK toward the centerline (2 m to the LEFT of the centerline steers RIGHT) (RED-14M-AI-3-LEFT-OFFSET-STEERS-RIGHT)', () => {
      // Arrange
      const x = 5;
      const y = 2;
      const heading = aimAtLookaheadHeadingRad(STRAIGHT_LOOP, x, y);

      // Act
      const input = stepAutopilotInput(
        { x, y, heading, speed: 40 },
        STRAIGHT_LOOP,
      );

      // Assert
      // THE MEASURED state (Step 14M, slice `P1-ai-centerline`): the
      // heading error is EXACTLY zero on this lane, so the steer is the
      // centerline-correcting term ALONE - `-2 m x (2.2 rad^-1 / 12 m)` =
      // -0.3666666666666667, a NEGATIVE (right) steer - so the clause is
      // GREEN (it was RED: the steer was EXACTLY 0).
      expect(input.steer).toBeLessThan(0);
    });
  });

  describe('the curvature-derived target speed (AC-084)', () => {
    // [RED-14M-AI-4-STRAIGHT-FULL-THROTTLE]
    it('AC-084: the emergent straight-line speed behaviour is UNCHANGED (full throttle, no brake, at the top speed) (RED-14M-AI-4-STRAIGHT-FULL-THROTTLE) - PRESERVATION GUARD', () => {
      // Arrange
      const input = stepAutopilotInput(
        {
          x: 20,
          y: -CENTERLINE_TIGHT_CORNER_RADIUS_M,
          heading: 0,
          speed: NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
        },
        CENTERLINE_TIGHT_CORNER_SAMPLES,
      );

      // Act + Assert
      // PRESERVATION GUARD, NOT A RED: the MEASURED current-model state
      // already commands exactly that (throttle 1, brake 0), so the lane
      // is GREEN on arrival - it pins the behaviour AC-084 must keep.
      expect(input.throttle).toBe(1);
      expect(input.brake).toBe(0);
    });

    // [RED-14M-AI-5-CORNER-SLOWS]
    it('AC-084: the autopilot SLOWS for the curvature AHEAD (the tight corner commands less drive than the straight) (RED-14M-AI-5-CORNER-SLOWS)', () => {
      // Arrange
      // The two lanes differ ONLY in the curvature ahead: the same speed,
      // the same stadium, the same straight - 14 m apart.
      const straightInput = stepAutopilotInput(
        {
          x: 20,
          y: -CENTERLINE_TIGHT_CORNER_RADIUS_M,
          heading: 0,
          speed: NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
        },
        CENTERLINE_TIGHT_CORNER_SAMPLES,
      );
      const cornerInput = stepAutopilotInput(
        {
          x: 34,
          y: -CENTERLINE_TIGHT_CORNER_RADIUS_M,
          heading: 0,
          speed: NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
        },
        CENTERLINE_TIGHT_CORNER_SAMPLES,
      );

      // Act + Assert
      // THE MEASURED state (Step 14M, slice `P1-ai-centerline`): the
      // straight lane commands a drive of 1 (throttle 1, brake 0) and the
      // corner lane a drive of -1 (throttle 0, brake 1), so the clause is
      // GREEN (it was RED: both lanes commanded a drive of 1, because the
      // seam never read the speed and the throttle/brake were binary).
      expect(commandedDriveOf(cornerInput)).toBeLessThan(
        commandedDriveOf(straightInput),
      );
    });
  });

  describe('the centerline measurement (AC-085)', () => {
    // [RED-14M-AI-6-CENTERLINE-BOUND]
    it('AC-085: the AI-driven car holds the road centerline (the measured lateral offset stays inside the named bound) (RED-14M-AI-6-CENTERLINE-BOUND)', () => {
      // Arrange
      const segments = buildCenterlineSegments(CENTERLINE_LAP_SAMPLES);
      let state = createInitialPhysicsCarState(42, {
        x: 5,
        y: -CENTERLINE_LAP_RADIUS_M,
        heading: 0,
      });
      let maxAbsOffsetM = 0;

      // Act
      for (let tick = 0; tick < CENTERLINE_LAP_TICK_COUNT; tick += 1) {
        const input = stepAutopilotInput(state, CENTERLINE_LAP_SAMPLES);
        state = stepPhysics(
          state,
          input,
          CENTERLINE_ROAD_SURFACE_RESOLVER,
          segments,
        );
        // The Step 14I containment (AC-081) is RETIRED (AC-090 VERDICT
        // 8): the geometry rides the retired NO-OP seam, so the drive is
        // boundary-free and the measurement can still FAIL (AC-085) -
        // nothing clamps the defect away.
        maxAbsOffsetM = Math.max(
          maxAbsOffsetM,
          Math.abs(
            measureLateralOffsetM(
              CENTERLINE_LAP_SAMPLES,
              state.x,
              state.y,
            ),
          ),
        );
      }

      // Assert
      // THE MEASURED state (Step 14M, slice `P1-ai-centerline`): the
      // racing-line consumption holds the car INSIDE the named bound -
      // the MEASURED lateral band over the 1200-tick lap is 2.0640 m
      // (the bound is 3 m), where the PRE-14M model drifted onto the
      // drivable edge (6.000 m - the pre-14W containment's limit) and
      // the clause was RED.
      expect(maxAbsOffsetM).toBeLessThanOrEqual(CENTERLINE_MAX_OFFSET_M);
    });

    // [RED-14M-AI-8-TIGHT-CORNER-BOUND]
    it('AC-085: the AC-085 bound holds on the TIGHT fixture too (the R = 8 m corner never pushes the AI outside the named bound) (RED-14M-AI-8-TIGHT-CORNER-BOUND)', () => {
      // Arrange
      // The SAME measurement as AI-6, on the TIGHT fixture (R = 8 m): the
      // AC-085 bound is a property of the CONTRACT, so it has to hold on
      // the tightest layout the repo measures, not only on the lap
      // layout. The lane was authored by the implementing slice
      // `P1-ai-centerline` (the RED slice authored AI-1..AI-7).
      const segments = buildCenterlineSegments(
        CENTERLINE_TIGHT_CORNER_SAMPLES,
      );
      let state = createInitialPhysicsCarState(42, {
        x: 5,
        y: -CENTERLINE_TIGHT_CORNER_RADIUS_M,
        heading: 0,
      });
      let maxAbsOffsetM = 0;

      // Act
      for (let tick = 0; tick < CENTERLINE_LAP_TICK_COUNT; tick += 1) {
        const input = stepAutopilotInput(
          state,
          CENTERLINE_TIGHT_CORNER_SAMPLES,
        );
        state = stepPhysics(
          state,
          input,
          CENTERLINE_ROAD_SURFACE_RESOLVER,
          segments,
        );
        // The Step 14I containment is RETIRED here too (AC-090 VERDICT
        // 8): the geometry rides the retired NO-OP seam, so the drive is
        // boundary-free and the measurement can still FAIL (AC-085).
        maxAbsOffsetM = Math.max(
          maxAbsOffsetM,
          Math.abs(
            measureLateralOffsetM(
              CENTERLINE_TIGHT_CORNER_SAMPLES,
              state.x,
              state.y,
            ),
          ),
        );
      }

      // Assert
      // THE MODEL LIMIT (the recorded reading): the Step 14Y sufficiency
      // boost now holds the R = 8 m corner, so the AI keeps moving
      // through it (the minimum corner speed is ~22.3 m/s on the post-fix
      // bytes). The AI still stays INSIDE the 3 m AC-085 bound; the
      // measured lateral band here is a centerline-keeping reading, not a
      // standstill indicator.
      //
      // FP-14M-05 is RETIRED by the P1-turnfeel fix: the pre-fix
      // standstill band (1.3554 m) no longer occurs, so this lane now
      // guards the AC-085 centerline bound with the containment RETIRED
      // (AC-090 VERDICT 8 - the drive is boundary-free). The drivable judgement still belongs to the R = 60 m
      // lap fixture of AI-6.
      expect(maxAbsOffsetM).toBeLessThanOrEqual(CENTERLINE_MAX_OFFSET_M);
    });

    // [RED-14M-AI-7-NO-CONTAINMENT-MASK]
    it('AC-085: the centerline measurement is NEVER masked by the retired seam (the band geometry is wider than the bound, and ANY off-center car still reads off-center) (RED-14M-AI-7-NO-CONTAINMENT-MASK) - ANTI-MASKING GUARD', () => {
      // Arrange
      const centerlineY = -CENTERLINE_LAP_RADIUS_M;
      const insideCar = createInitialPhysicsCarState(42, {
        x: 5,
        y: centerlineY + 4,
        heading: 0,
      });
      const outsideCar = createInitialPhysicsCarState(42, {
        x: 5,
        y: centerlineY - 8,
        heading: 0,
      });
      const beyondBandCar = createInitialPhysicsCarState(42, {
        x: 5,
        y: centerlineY - 12,
        heading: 0,
      });

      // Assert
      // The drivable surface is WIDER than the centerline bound, so the
      // band law can never clamp the measured defect away.
      expect(CENTERLINE_DRIVABLE_HALF_WIDTH_M).toBeGreaterThan(
        CENTERLINE_MAX_OFFSET_M,
      );
      // THE FORMER ONE BOUND (Step 14W, slice `P1-sand-service`, AC-094
      // (d)/(g)) - RETIRED (AC-090 VERDICT 8): the ONE containment bound
      // once sat on the OUTER edge of the drivable band - WIDER than the
      // centerline bound still - so a masked defect stays impossible.
      expect(CENTERLINE_BAND_OUTER_EDGE_M).toBeGreaterThan(
        CENTERLINE_MAX_OFFSET_M,
      );
      // A car INSIDE the drivable surface but 4 m off the centerline
      // still READS 4 m off the centerline: the measurement is taken,
      // never inferred from a clamp.
      expect(
        Math.abs(
          measureLateralOffsetM(
            CENTERLINE_LAP_SAMPLES,
            insideCar.x,
            insideCar.y,
          ),
        ),
      ).toBeCloseTo(4, 9);
      // A car 8 m off the centerline sits INSIDE the off-track band: the
      // former band bound never contained it, and the RETIRED seam does
      // not either - it still READS 8 m off the centerline.
      //
      // THE RETIRED CONTRACT (re-authored, NEVER deleted): the drivable
      // edge used to carry the car BACK onto the road ("reseated at the
      // drivable edge"); the STEP-14W band bound never did that (the band
      // is DRIVABLE) and the RETIREMENT removed the bound entirely - the
      // defect is READ, never masked.
      expect(
        Math.abs(
          measureLateralOffsetM(
            CENTERLINE_LAP_SAMPLES,
            outsideCar.x,
            outsideCar.y,
          ),
        ),
      ).toBeCloseTo(8, 9);
      // A car 12 m off the centerline is BEYOND the band: the FORMER ONE
      // bound would have contained it AT THE BAND OUTER EDGE; the
      // RETIRED seam leaves it where it stands and the measurement still
      // reads the FULL 12 m defect (the bound is retired, never moved
      // back into the tick).
      expect(
        Math.abs(
          measureLateralOffsetM(
            CENTERLINE_LAP_SAMPLES,
            beyondBandCar.x,
            beyondBandCar.y,
          ),
        ),
      ).toBeCloseTo(12, 9);
    });
  });
});

/* ------------------------------------------------------------------------- *
 * STEP 14Y (slice `P1-turnfeel-red`): the TURNING FEEL + AI DRIVEABILITY
 * wave (AC-098).
 *
 * AC-098 (the turning feel AND the AI driveability): the commanded turn
 * rate must be able to HOLD the corner the car is driving, at the speed the
 * car drives it at. The Step 14Y fix adds a raised-cosine sufficiency
 * boost confined to ~[17.6, 18.0] m/s; at the R = 8 m stadium's target
 * speed (~17.8 m/s) the USER law now commands enough rate to hold the
 * corner (2.4475 rad/s vs 2.225 rad/s required), while the ratified mild
 * reduction above the crawl band stays untouched outside the boost window.
 *
 * THE AI HALF (this file): the AI drives the SAME geometry through the
 * AC-093 turn-rate-percent channel, which the physics resolves FIRST and
 * NEVER softens. With the P1-turnfeel fix the AI now commands a sufficient
 * rate through the percent channel (2.2893 rad/s vs 2.225 rad/s required),
 * keeps moving through the R = 8 m corner and completes laps of the
 * served track.
 *
 * THE RED SET: RED-2 (both clauses), RED-3 and RED-4 are DISCRIMINATING
 * lanes that turned GREEN on the post-fix bytes (they were RED before the
 * P1-turnfeel fix), with their non-vacuity guards asserted FIRST. RED-5 is
 * a LABELLED PRESERVATION GUARD SET that holds on the post-fix bytes and
 * is NEVER counted as a red.
 *
 * RED-5e IS NOT RE-AUTHORED HERE: the AC-085 anti-masking guard already
 * exists in the Step 14M block of THIS file - `[RED-14M-AI-7-NO-CONTAINMENT-
 * MASK]` (the drivable half-width, the band outer edge and any off-center
 * reading) - so it is REFERENCED, never duplicated.
 *
 * RED-5d EXTENDS (never duplicates) the Step 14T lane `[RED-14T-4-AI-RATE-
 * PRESERVED]`: that lane pins the AI rate against the steer-axis law
 * INSIDE the crawl band, where the user factor is EXACTLY 1 - so it can
 * never DETECT a softened AI channel. The lane below measures the AI
 * channel ABOVE the band, where a softened channel reads a different rate.
 * ------------------------------------------------------------------------- */

/**
 * The RED-14Y AI served-layout seed (the served race page's layout key).
 */
const TURNFEEL_AI_SERVED_SEED = 42;

/**
 * The RED-14Y AI served-layout version (the served race page's layout key).
 */
const TURNFEEL_AI_SERVED_LAYOUT_VERSION = 1;

/**
 * The RED-14Y AI served-layout size bucket (the served race page's layout
 * key).
 */
const TURNFEEL_AI_SERVED_SIZE_BUCKET = 'large';

/**
 * The RED-14Y AI served-layout spawn offset along the first segment (m).
 *
 * THE SERVED-PAGE MIRROR: the served page drops the car 2 m along the
 * first segment, so the RED-4 lane starts exactly where the served race
 * starts.
 */
const TURNFEEL_AI_SERVED_SPAWN_OFFSET_M = 2;

/**
 * The RED-14Y AI served-layout drive length (ticks) - 60 s of the 60 Hz
 * fixed step.
 */
const TURNFEEL_AI_SERVED_LAP_TICK_COUNT = 3600;

/**
 * The RED-14Y AI target-speed bisection iterations.
 *
 * THE DETERMINISM: the bisection runs a FIXED number of iterations (never
 * a tolerance loop), so the measured target speed is bit-identical on every
 * run.
 */
const TURNFEEL_AI_TARGET_BISECTION_ITERATIONS = 60;

/**
 * The RED-14Y AI percent-channel tolerance (rad/s).
 *
 * The reading is measured through the published heading delta, so it
 * carries a float64 rounding of a few ULP. The tolerance absorbs that
 * noise ONLY: a SOFTENED AI channel differs by ~0.007 rad/s at the lane's
 * speed, which is SIX ORDERS of magnitude above this tolerance, so the
 * lane can still FAIL.
 */
const TURNFEEL_AI_PERCENT_CHANNEL_TOLERANCE_RAD_PER_S = 1e-9;

/**
 * The RED-14Y AI road-surface resolver (the fixtures are road everywhere).
 */
const TURNFEEL_AI_ROAD_SURFACE_RESOLVER = (): 'road' => 'road';

/**
 * The index of the sample NEAREST to a point.
 *
 * THE SHARED LOOKUP: the corner window and the lap observable both need
 * the car's nearest sample, so the lookup lives in ONE helper.
 *
 * @param samples - The frozen spline samples of the layout.
 * @param x - The measured world x.
 * @param y - The measured world y.
 * @returns The index of the nearest sample.
 */
const turnfeelNearestIndexOf = (
  samples: readonly SplineSample[],
  x: number,
  y: number,
): number => {
  let best = 0;
  let bestDistance = Infinity;
  for (let index = 0; index < samples.length; index += 1) {
    const sample = samples[index];
    const distance = (sample.x - x) ** 2 + (sample.y - y) ** 2;
    if (distance < bestDistance) {
      bestDistance = distance;
      best = index;
    }
  }
  return best;
};

/**
 * The perimeter of a closed sample ring (m).
 *
 * @param samples - The frozen spline samples of the layout.
 * @returns The perimeter of the closed ring (m).
 */
const turnfeelPerimeterOf = (samples: readonly SplineSample[]): number => {
  let total = 0;
  for (let index = 0; index < samples.length; index += 1) {
    const from = samples[index];
    const to = samples[(index + 1) % samples.length];
    total += Math.hypot(to.x - from.x, to.y - from.y);
  }
  return total;
};

/**
 * The TEST-SIDE lap observable of a layout.
 */
interface TurnfeelLapObservable {
  /** Feeds one published car position into the observable. */
  advance: (x: number, y: number) => void;
  /** The counted laps (whole rings of arc progress). */
  readonly laps: number;
  /** The accumulated arc progress (m). */
  readonly progressM: number;
}

/**
 * Creates the TEST-SIDE lap observable of a layout.
 *
 * THE OBSERVABLE (never a production counter): the lap count is derived
 * from the PUBLISHED car snapshot and the track geometry ONLY - the car's
 * arc position is projected onto the closed sample ring and a whole ring
 * of arc progress counts as ONE lap. No production lap counter and no new
 * published state is involved.
 *
 * @param samples - The frozen spline samples of the layout.
 * @returns The lap observable of the layout.
 */
const createTurnfeelLapObservable = (
  samples: readonly SplineSample[],
): TurnfeelLapObservable => {
  const total = samples.length;
  const cumulative: number[] = [0];
  for (let index = 0; index < total; index += 1) {
    const from = samples[index];
    const to = samples[(index + 1) % total];
    cumulative.push(
      cumulative[index] + Math.hypot(to.x - from.x, to.y - from.y),
    );
  }
  const perimeterM = cumulative[total];
  let lastArcM: number | null = null;
  let progressM = 0;
  let laps = 0;
  const projectArcM = (x: number, y: number): number => {
    let bestDistance = Infinity;
    let bestArc = 0;
    for (let index = 0; index < total; index += 1) {
      const from = samples[index];
      const to = samples[(index + 1) % total];
      const dx = to.x - from.x;
      const dy = to.y - from.y;
      const lengthSquared = dx * dx + dy * dy;
      const t =
        lengthSquared === 0
          ? 0
          : Math.max(
              0,
              Math.min(
                1,
                ((x - from.x) * dx + (y - from.y) * dy) / lengthSquared,
              ),
            );
      const px = from.x + t * dx;
      const py = from.y + t * dy;
      const distance = Math.hypot(x - px, y - py);
      if (distance < bestDistance) {
        bestDistance = distance;
        bestArc = cumulative[index] + t * Math.sqrt(lengthSquared);
      }
    }
    return bestArc;
  };
  return {
    advance: (x: number, y: number): void => {
      const arcM = projectArcM(x, y);
      if (lastArcM === null) {
        lastArcM = arcM;
        return;
      }
      let deltaM = arcM - lastArcM;
      if (deltaM < -perimeterM / 2) {
        deltaM += perimeterM;
      } else if (deltaM > perimeterM / 2) {
        deltaM -= perimeterM;
      }
      const before = Math.floor(progressM / perimeterM);
      progressM += deltaM;
      const after = Math.floor(progressM / perimeterM);
      laps += after - before;
      lastArcM = arcM;
    },
    get laps(): number {
      return laps;
    },
    get progressM(): number {
      return progressM;
    },
  };
};

/**
 * The tightest (max |curvature|) racing-line point of a layout.
 */
interface TurnfeelRacingLineCorner {
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
 * required rate is COMPUTED from the two PUBLISHED quantities - the racing
 * line's curvature and its target speed.
 *
 * @param samples - The frozen spline samples of the layout.
 * @returns The tightest racing-line point of the layout.
 */
const tightestRacingLinePointOf = (
  samples: readonly SplineSample[],
): TurnfeelRacingLineCorner => {
  const line = resolveAutopilotRacingLine(samples);
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
 * A pose on the layout (a world position and a heading).
 */
interface TurnfeelPose {
  /** The world x. */
  readonly x: number;
  /** The world y. */
  readonly y: number;
  /** The heading (rad). */
  readonly heading: number;
}

/**
 * Resolves the pose of the racing line at a sample index.
 *
 * THE POSE (never a literal): the racing line is DISPLACED from the
 * centerline by its SIGNED lateral offset (positive to the LEFT of the
 * travel direction), so the pose is the centerline sample moved along the
 * chord's left normal - the very point the AI aims at.
 *
 * @param samples - The frozen spline samples of the layout.
 * @param index - The index of the racing-line point.
 * @returns The pose of the racing-line point.
 */
const turnfeelRacingLinePoseOf = (
  samples: readonly SplineSample[],
  index: number,
): TurnfeelPose => {
  const line = resolveAutopilotRacingLine(samples);
  const from = samples[index];
  const to = samples[(index + 1) % samples.length];
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const length = Math.hypot(dx, dy);
  const offsetM = line[index].offsetM;
  return {
    x: from.x + (-dy / length) * offsetM,
    y: from.y + (dx / length) * offsetM,
    heading: Math.atan2(dy, dx),
  };
};

/**
 * The reading of ONE AI turn-rate lane.
 */
interface TurnfeelAiLane {
  /** The autopilot input the seam produced. */
  readonly input: PhysicsStepInput;
  /** The MEASURED commanded turn rate of the first tick (rad/s). */
  readonly commandedRateRadPerS: number;
}

/**
 * Measures the AI's commanded turn rate at a pose and a speed.
 *
 * THE MEASUREMENT (the observable): the autopilot seam produces the input
 * and the physics service turns it into a MEASURED turn rate, read from
 * the PUBLISHED heading delta of the FIRST tick (the delta divided by the
 * fixed timestep) - never a re-implementation of the service law. The
 * FIRST tick is the only one that carries the lane speed, so the lane
 * speed IS the tick-start speed.
 *
 * @param pose - The pose the lane starts from.
 * @param speedMps - The snapshot speed of the lane (m/s).
 * @param samples - The frozen spline samples of the layout.
 * @returns The produced autopilot input and the measured turn rate.
 */
const measureAiTurnRateLane = (
  pose: TurnfeelPose,
  speedMps: number,
  samples: readonly SplineSample[],
): TurnfeelAiLane => {
  const input = stepAutopilotInput(
    { x: pose.x, y: pose.y, heading: pose.heading, speed: speedMps },
    samples,
  );
  const start: PhysicsCarState = {
    ...createInitialPhysicsCarState(42),
    x: pose.x,
    y: pose.y,
    heading: pose.heading,
    speed: speedMps,
    drift: {
      committed: { slip: 0, turnSign: 0, reArmed: true },
      pose: { pose: 0, poseRate: 0, articulation: 0 },
    },
  };
  const next = stepPhysics(start, input, TURNFEEL_AI_ROAD_SURFACE_RESOLVER);
  return {
    input,
    commandedRateRadPerS:
      (next.heading - start.heading) / NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
  };
};

/**
 * The reading of the AI's OWN target speed at a pose.
 */
interface TurnfeelTargetSpeedReading {
  /** The MEASURED target speed (m/s). */
  readonly targetSpeedMps: number;
  /** Whether the AI commands a speed brake at the given speed. */
  readonly brakesAt: (speedMps: number) => boolean;
}

/**
 * MEASURES the AI's OWN speed-brake transition at a pose, through its
 * OWN drive law.
 *
 * THE OBSERVABLE (never a re-implementation): the AI commands a SPEED
 * BRAKE iff the snapshot speed is ABOVE its own speed-brake transition.
 * After the P1-turnfeel fix this transition sits at the "never-stop"
 * floor (~23.7 m/s on the R = 8 m corner), not at the published racing-
 * line target speed (~17.8 m/s). The bisection is DETERMINISTIC (a fixed
 * iteration count, never a tolerance loop).
 *
 * @param pose - The pose the measurement runs at.
 * @param samples - The frozen spline samples of the layout.
 * @returns The measured transition speed and the brake predicate.
 */
const measureAiTargetSpeedMps = (
  pose: TurnfeelPose,
  samples: readonly SplineSample[],
): TurnfeelTargetSpeedReading => {
  const brakesAt = (speedMps: number): boolean => {
    const input = stepAutopilotInput(
      { x: pose.x, y: pose.y, heading: pose.heading, speed: speedMps },
      samples,
    );
    return input.brake > 0;
  };
  let low = 0;
  let high = NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S;
  if (brakesAt(low) || !brakesAt(high)) {
    throw new Error(
      'the AI drive law exposes no speed-brake transition at this pose - the target speed is NOT measurable here',
    );
  }
  for (
    let iteration = 0;
    iteration < TURNFEEL_AI_TARGET_BISECTION_ITERATIONS;
    iteration += 1
  ) {
    const mid = (low + high) / 2;
    if (brakesAt(mid)) {
      high = mid;
    } else {
      low = mid;
    }
  }
  return { targetSpeedMps: high, brakesAt };
};

/**
 * The TURNING FEEL + AI DRIVEABILITY contracts (Step 14Y, slice
 * `P1-turnfeel-red`; AC-098).
 *
 * AC-098 (the turning feel AND the AI driveability): the commanded turn
 * rate must be able to HOLD the corner the car is driving, at the speed the
 * car drives it at. The Step 14Y fix replaced the CONSTANT 44.5 m minimum
 * turn radius with a narrow sufficiency boost around 17.8 m/s, so the
 * tightest measured corner is now holdable and the AI no longer stalls.
 *
 * THE RED SET: RED-2 (both clauses), RED-3 and RED-4 are DISCRIMINATING
 * lanes that are GREEN on the post-fix bytes (they were RED before the
 * P1-turnfeel fix), with their non-vacuity guard asserted FIRST. RED-5 is
 * a LABELLED PRESERVATION GUARD SET that holds on the post-fix bytes and
 * is NEVER counted as a red.
 */
describe('the TURNING FEEL + AI DRIVEABILITY wave (Step 14Y, slice P1-turnfeel-red; AC-098)', () => {
  describe('RED-2 (the AI commanded rate can hold the line it TARGETS)', () => {
    // [RED-14Y-2A-AI-HOLDS-TARGETED-LINE]
    it('AC-098: the AI commanded rate can HOLD the line it TARGETS (the AI turn rate meets the kinematic requirement of the tightest corner) (RED-14Y-2A-AI-HOLDS-TARGETED-LINE)', () => {
      // Arrange
      const corner = tightestRacingLinePointOf(
        CENTERLINE_TIGHT_CORNER_SAMPLES,
      );
      const pose = turnfeelRacingLinePoseOf(
        CENTERLINE_TIGHT_CORNER_SAMPLES,
        corner.index,
      );

      // Act
      const lane = measureAiTurnRateLane(
        pose,
        corner.targetSpeedMps,
        CENTERLINE_TIGHT_CORNER_SAMPLES,
      );

      // Assert
      // The NON-VACUITY GUARD comes FIRST, and it has TWO parts: the AI
      // really TARGETS a corner (never a straight) and the racing line
      // really DIFFERS from the centerline there (an AI driving the
      // centerline would satisfy the clause VACUOUSLY).
      expect(Math.abs(corner.curvaturePerM)).toBeGreaterThan(
        NEATRACING_RACING_LINE_STRAIGHT_CURVATURE_PER_M,
      );
      expect(Math.abs(corner.offsetM)).toBeGreaterThan(0);
      // THE DISCRIMINATING CLAUSE: the AI's commanded rate must MEET the
      // kinematic requirement of the corner it targets.
      //
      // THE MEASURED post-fix state: the AI commands ~2.2893 rad/s through
      // the percent channel, while the R = 8 m corner demands 2.225 rad/s
      // at the AI's target speed (a surplus of ~0.0643 rad/s) - the clause
      // is GREEN on the post-fix bytes.
      expect(lane.commandedRateRadPerS).toBeGreaterThanOrEqual(
        corner.requiredRateRadPerS,
      );
    });

    // [RED-14Y-2B-TARGET-SPEED-REACHABLE]
    it('AC-098: the AI TARGET speed is REACHABLE (the corner is holdable AT the speed the AI asks for) (RED-14Y-2B-TARGET-SPEED-REACHABLE)', () => {
      // Arrange
      const corner = tightestRacingLinePointOf(
        CENTERLINE_TIGHT_CORNER_SAMPLES,
      );
      const pose = turnfeelRacingLinePoseOf(
        CENTERLINE_TIGHT_CORNER_SAMPLES,
        corner.index,
      );
      const reading = measureAiTargetSpeedMps(
        pose,
        CENTERLINE_TIGHT_CORNER_SAMPLES,
      );

      // Act
      const laneAtTarget = measureAiTurnRateLane(
        pose,
        corner.targetSpeedMps,
        CENTERLINE_TIGHT_CORNER_SAMPLES,
      );

      // Assert
      // The NON-VACUITY GUARD: the AI's own speed-brake transition is
      // REAL (it does NOT brake at low speed and DOES brake at high speed),
      // and it now sits ABOVE the published racing-line target speed. That
      // proves the published target (~17.8 m/s) is inside the reachable
      // envelope, rather than an extrapolation.
      expect(reading.brakesAt(reading.targetSpeedMps * 0.99)).toBe(false);
      expect(reading.brakesAt(reading.targetSpeedMps * 1.01)).toBe(true);
      expect(reading.targetSpeedMps).toBeGreaterThan(corner.targetSpeedMps);
      // THE DISCRIMINATING CLAUSE: at the PUBLISHED racing-line target
      // speed, the AI's commanded rate must MEET the kinematic requirement
      // of the corner.
      //
      // THE MEASURED post-fix state: the AI commands enough rate at the
      // published target speed to hold the R = 8 m corner - the clause is
      // GREEN on the post-fix bytes.
      expect(
        corner.targetSpeedMps * Math.abs(corner.curvaturePerM),
      ).toBeLessThanOrEqual(laneAtTarget.commandedRateRadPerS);
    });
  });

  describe('RED-3 (the AI KEEPS MOVING through the corner)', () => {
    // [RED-14Y-3-AI-KEEPS-MOVING-IN-CORNER]
    it('AC-098: the AI KEEPS MOVING through the tightest corner (a STANDSTILL band is a FAILURE) (RED-14Y-3-AI-KEEPS-MOVING-IN-CORNER)', () => {
      // Arrange
      // THE SAME DRIVE as the Step 14M tight-fixture lane (AI-8): the AI
      // drives the R = 8 m stadium for the shared 1200-tick window with
      // the containment RETIRED (AC-090 VERDICT 8): the drive is
      // boundary-free.
      const segments = buildCenterlineSegments(CENTERLINE_TIGHT_CORNER_SAMPLES);
      const racingLine = resolveAutopilotRacingLine(
        CENTERLINE_TIGHT_CORNER_SAMPLES,
      );
      let state = createInitialPhysicsCarState(42, {
        x: 5,
        y: -CENTERLINE_TIGHT_CORNER_RADIUS_M,
        heading: 0,
      });
      let cornerTickCount = 0;
      let firstCornerTick = -1;
      let speedAtFirstCornerTick = 0;
      let minSpeedInCorner = Infinity;

      // Act
      for (let tick = 0; tick < CENTERLINE_LAP_TICK_COUNT; tick += 1) {
        const input = stepAutopilotInput(
          state,
          CENTERLINE_TIGHT_CORNER_SAMPLES,
        );
        // The corner window is the set of ticks whose NEAREST racing-line
        // sample really IS a corner (never a straight).
        const nearest = turnfeelNearestIndexOf(
          CENTERLINE_TIGHT_CORNER_SAMPLES,
          state.x,
          state.y,
        );
        const curvaturePerM = Math.abs(
          racingLine[nearest].curvaturePerM,
        );
        if (curvaturePerM > NEATRACING_RACING_LINE_STRAIGHT_CURVATURE_PER_M) {
          cornerTickCount += 1;
          if (firstCornerTick < 0) {
            firstCornerTick = tick;
            speedAtFirstCornerTick = state.speed;
          }
        }
        state = stepPhysics(
          state,
          input,
          CENTERLINE_ROAD_SURFACE_RESOLVER,
          segments,
        );
        // The Step 14I containment is RETIRED (AC-090 VERDICT 8): the
        // geometry rides the retired NO-OP seam, so the drive is
        // boundary-free and the measurement can still FAIL (AC-085).
        if (
          curvaturePerM > NEATRACING_RACING_LINE_STRAIGHT_CURVATURE_PER_M
        ) {
          minSpeedInCorner = Math.min(minSpeedInCorner, state.speed);
        }
      }

      // Assert
      // The NON-VACUITY GUARD comes FIRST, and it has TWO parts: the
      // window really COVERS a corner and the AI really was MOVING when it
      // entered that corner (an AI that never moved could never fail the
      // clause below).
      expect(cornerTickCount).toBeGreaterThan(0);
      expect(speedAtFirstCornerTick).toBeGreaterThan(0);
      // THE DISCRIMINATING CLAUSE: the AI must KEEP MOVING through the
      // corner - a STANDSTILL band is a failure.
      //
      // THE MEASURED post-fix state: the AI keeps moving through the
      // R = 8 m corner (minimum corner speed ~22.3 m/s) - the clause is
      // GREEN on the post-fix bytes.
      expect(minSpeedInCorner).toBeGreaterThan(0);
    });
  });

  describe('RED-4 (the AI completes a LAP)', () => {
    // [RED-14Y-4-LAP-OBSERVABLE-COUNTS-A-REAL-LAP]
    it('AC-098: the test-side lap observable COUNTS a real lap (a synthetic full circuit reports EXACTLY one lap) (RED-14Y-4-LAP-OBSERVABLE-COUNTS-A-REAL-LAP) - NON-VACUITY GUARD', () => {
      // Arrange
      // THE SYNTHETIC CIRCUIT: the SAME geometry the AI lane drives (the
      // served layout's own sample polyline), walked all the way round.
      const spec = generateTrack({
        seed: TURNFEEL_AI_SERVED_SEED,
        layoutVersion: TURNFEEL_AI_SERVED_LAYOUT_VERSION,
        sizeBucket: TURNFEEL_AI_SERVED_SIZE_BUCKET,
      });
      const samples = spec.splineSamples;
      const observable = createTurnfeelLapObservable(samples);

      // Act
      for (let index = 0; index < samples.length; index += 1) {
        observable.advance(samples[index].x, samples[index].y);
      }
      observable.advance(samples[0].x, samples[0].y);

      // Assert
      // THE NON-VACUITY GUARD of the RED-4 lane below: the observable
      // REALLY counts a lap, so a counter that never increments can never
      // satisfy the AI clause VACUOUSLY.
      expect(turnfeelPerimeterOf(samples)).toBeGreaterThan(0);
      expect(observable.laps).toBe(1);
    });

    // [RED-14Y-4-AI-COMPLETES-A-LAP]
    it('AC-098: the AI completes a LAP of the served track (the test-side lap observable reports at least one lap) (RED-14Y-4-AI-COMPLETES-A-LAP)', () => {
      // Arrange
      // THE SERVED-PAGE MIRROR: the served layout, the served spawn (2 m
      // along the first segment, on its heading) and the served drive law.
      const spec = generateTrack({
        seed: TURNFEEL_AI_SERVED_SEED,
        layoutVersion: TURNFEEL_AI_SERVED_LAYOUT_VERSION,
        sizeBucket: TURNFEEL_AI_SERVED_SIZE_BUCKET,
      });
      const samples = spec.splineSamples;
      const firstSegment = spec.segments[0];
      const spawnHeading = Math.atan2(
        firstSegment.endY - firstSegment.startY,
        firstSegment.endX - firstSegment.startX,
      );
      const surfaceResolver = createTrackSurfaceResolver(spec.segments);
      const observable = createTurnfeelLapObservable(samples);
      let state = createInitialPhysicsCarState(spec.seed, {
        x:
          firstSegment.startX +
          Math.cos(spawnHeading) * TURNFEEL_AI_SERVED_SPAWN_OFFSET_M,
        y:
          firstSegment.startY +
          Math.sin(spawnHeading) * TURNFEEL_AI_SERVED_SPAWN_OFFSET_M,
        heading: spawnHeading,
      });

      // Act
      for (let tick = 0; tick < TURNFEEL_AI_SERVED_LAP_TICK_COUNT; tick += 1) {
        const input = stepAutopilotInput(state, samples);
        state = stepPhysics(state, input, surfaceResolver, spec.segments);
        observable.advance(state.x, state.y);
      }

      // Assert
      // THE DISCRIMINATING CLAUSE: the AI completes a lap of the served
      // track.
      //
      // THE MEASURED post-fix state: the AI closes the lap on the served
      // track in ~26.03 s and completes 2 laps within the test window -
      // the clause is GREEN on the post-fix bytes.
      expect(observable.laps).toBeGreaterThanOrEqual(1);
    });
  });

  describe('RED-5 (the PRESERVATION GUARDS - NOT reds)', () => {
    // [RED-14Y-5D-AI-PERCENT-NOT-SOFTENED]
    it('AC-093: the AI percent channel is NEVER softened by the user factor (the resolved rate is exactly (percent / 100) x the steering rate, ABOVE the crawl band) (RED-14Y-5D-AI-PERCENT-NOT-SOFTENED) - PRESERVATION GUARD (NOT a red)', () => {
      // Arrange
      const corner = tightestRacingLinePointOf(
        CENTERLINE_TIGHT_CORNER_SAMPLES,
      );
      const pose = turnfeelRacingLinePoseOf(
        CENTERLINE_TIGHT_CORNER_SAMPLES,
        corner.index,
      );

      // Act
      const lane = measureAiTurnRateLane(
        pose,
        corner.targetSpeedMps,
        CENTERLINE_TIGHT_CORNER_SAMPLES,
      );
      const percent = aiTurnRatePercentOf(lane.input) ?? Number.NaN;
      const percentLawRateRadPerS =
        (percent / 100) * NEATRACING_PHYSICS_STEERING_RATE;

      // Assert
      // PRESERVATION GUARD, NOT A RED: the MEASURED current-model state
      // already resolves exactly that rate, so this lane is GREEN on
      // arrival.
      //
      // THE NON-VACUITY GUARD comes FIRST, and it has TWO parts: the lane
      // must REALLY turn (the steer saturates) and the lane's speed must
      // be ABOVE the crawl band - INSIDE the band the user factor is
      // EXACTLY 1, so a softened AI channel would be indistinguishable
      // there (that is why the Step 14T lane cannot detect it).
      expect(Math.abs(lane.input.steer)).toBe(1);
      expect(corner.targetSpeedMps).toBeGreaterThan(
        NEATRACING_PHYSICS_TURN_RATE_REDUCTION_BAND_UNITS_PER_S,
      );
      // THE CLAUSE: the AI channel resolves the rate from the percent
      // ALONE - the user factor never touches it.
      expect(
        Math.abs(lane.commandedRateRadPerS - percentLawRateRadPerS),
      ).toBeLessThanOrEqual(TURNFEEL_AI_PERCENT_CHANNEL_TOLERANCE_RAD_PER_S);
    });

    // RED-5e IS NOT RE-AUTHORED HERE (see the block banner): the AC-085
    // anti-masking guard already exists in the Step 14M block of this file
    // - `[RED-14M-AI-7-NO-CONTAINMENT-MASK]` - so it is REFERENCED, never
    // duplicated. That lane pins the drivable half-width, the band outer
    // edge and an off-center reading, so the center-keeping
    // measurement stays ABLE TO FAIL with nothing clamping the car
    // (AC-090 VERDICT 8 - the containment is retired).
  });
});
