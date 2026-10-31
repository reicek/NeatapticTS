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
  stepAutopilotInput,
  type AutopilotCarSnapshot,
} from './ai.autopilot';
import { createTrackSpecForAutopilotFixture } from './ai.autopilot.fixtures';
import { resolveTrackBoundaryCar } from './collision.utils';
import {
  createInitialPhysicsCarState,
  NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
  NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
  NEATRACING_PHYSICS_STEERING_RATE,
  stepPhysics,
  type PhysicsCarState,
  type PhysicsStepInput,
} from './physics.step.service';
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
 * The AC-085 centerline bound (m).
 *
 * THE BOUND IS INSIDE THE CONTAINMENT (the anti-masking rule): the
 * Step 14I containment only guarantees that the car stays on the
 * drivable surface (6 m from the centerline), so a centerline bound of
 * 3 m - HALF the drivable half-width - can still fail while the
 * containment is in place. A masked defect is a FAILED acceptance.
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
 * Builds the track chords of a fixture loop (the Step 14I containment
 * reads the chords, not the samples).
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
 * is measured DIRECTLY - never inferred from the containment - so a
 * contained-but-off-center car still reads off-center.
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
 * INDEPENDENT of the Step 14I containment - the measurement must still
 * be able to FAIL with the containment in place, and a masked defect is a
 * FAILED acceptance.
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
        state = stepPhysics(state, input, CENTERLINE_ROAD_SURFACE_RESOLVER);
        // The Step 14I containment (AC-081) is IN PLACE for the whole
        // lap, so the measurement has to be able to FAIL with the
        // containment in place (AC-085).
        state = resolveTrackBoundaryCar(state, segments);
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
      // containment's own drivable edge (6.000 m - the containment's
      // limit) and the clause was RED.
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
        state = stepPhysics(state, input, CENTERLINE_ROAD_SURFACE_RESOLVER);
        // The Step 14I containment is IN PLACE here too, so the
        // measurement has to be able to FAIL with the containment in
        // place (AC-085).
        state = resolveTrackBoundaryCar(state, segments);
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
      // THE MODEL LIMIT (the recorded reading): the arcade model realizes
      // a minimum turn radius of `max speed / (steering rate x steer)` =
      // 44.5 m at the full steer axis, so the R = 8 m corner is BEYOND
      // the kinematic limit of the model - NO controller can drive it at
      // speed. The AI therefore brakes down at the corner entry (the
      // racing line's curvature-derived target speed there is 17.8 m/s
      // and the heading error saturates the full brake), and the MEASURED
      // lateral band over the whole lap is 1.3554 m - INSIDE the 3 m
      // AC-085 bound.
      //
      // THE 1.3554 m BAND IS A STANDSTILL BAND (FP-14M-05): the AI brakes
      // down to a STOP at the corner entry and STAYS there (the R = 8 m
      // corner is beyond the model's 44.5 m kinematic limit, so the brake
      // saturates), so this lane guards the CONTRACT - the offset bound
      // with the Step 14I containment IN PLACE - and CANNOT detect a
      // driving-quality regression on a tight corner. The drivable
      // judgement belongs to the R = 60 m lap fixture of AI-6.
      expect(maxAbsOffsetM).toBeLessThanOrEqual(CENTERLINE_MAX_OFFSET_M);
    });

    // [RED-14M-AI-7-NO-CONTAINMENT-MASK]
    it('AC-085: the centerline measurement is NEVER masked by the containment (the drivable edge is wider than the bound, and a contained-but-off-center car still reads off-center) (RED-14M-AI-7-NO-CONTAINMENT-MASK) - ANTI-MASKING GUARD', () => {
      // Arrange
      const segments = buildCenterlineSegments(CENTERLINE_LAP_SAMPLES);
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

      // Act
      const containedInside = resolveTrackBoundaryCar(insideCar, segments);
      const containedOutside = resolveTrackBoundaryCar(outsideCar, segments);

      // Assert
      // The drivable surface is WIDER than the centerline bound, so the
      // containment can never clamp the measured defect away.
      expect(CENTERLINE_DRIVABLE_HALF_WIDTH_M).toBeGreaterThan(
        CENTERLINE_MAX_OFFSET_M,
      );
      // A car INSIDE the drivable surface but 4 m off the centerline
      // still READS 4 m off the centerline: the measurement is taken,
      // never inferred from the containment.
      expect(
        Math.abs(
          measureLateralOffsetM(
            CENTERLINE_LAP_SAMPLES,
            containedInside.x,
            containedInside.y,
          ),
        ),
      ).toBeCloseTo(4, 9);
      // A car 8 m off the centerline is reseated by the containment at the
      // drivable edge - the limit the centerline contract must stay well
      // inside of.
      expect(
        Math.abs(
          measureLateralOffsetM(
            CENTERLINE_LAP_SAMPLES,
            containedOutside.x,
            containedOutside.y,
          ),
        ),
      ).toBeCloseTo(CENTERLINE_DRIVABLE_HALF_WIDTH_M, 9);
    });
  });
});
