/**
 * Green tests for the NeatRacing deterministic scripted waypoint-rivals
 * controller (`examples/neatRacing/ai/scripted.controller.ts` — Step 02,
 * slice P2-S2-C, TDD lane green-only).
 *
 * Contracts pinned here:
 * - AC-028: the per-course throttle TABLE is a deterministic, segment-keyed
 *   function of the course layout (straight segments publish the recycled
 *   constant 0.82 verbatim; corner segments publish proportionally less,
 *   floored at the drivability bound); different courses derive different
 *   tables. Derived from the ONE published racing-line source
 *   (`resolveAutopilotRacingLine`), never from second geometry.
 * - The recycled waypoint plan: advance radius 16 (in-state, modulo wrapped),
 *   lookahead 4, target projection 14, proportional steer gain 2.2 with
 *   [-1, 1] clamp, empty-course fallback {0.82, 0 brake, 0 steer}.
 * - The output shape is the `PhysicsStepInput` seam the simulation worker
 *   consumes (throttle/brake/steer finite numbers).
 * - THE PHYSICS PROOF (AC-092/AC-093): a PRESENT `turnRatePercent` is
 *   authoritative in the physics — a present `0` commands EXACTLY zero
 *   turn — so the controller emits NONE and the steer axis alone turns the
 *   car, proven through the real `stepPhysics` tick.
 *
 * @module
 */

import type {
  SplineSample,
  TrackSpec,
} from '../../racing_curriculum/track/track.generator.types';
import {
  NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
  type PhysicsCarState,
  stepPhysics,
} from '../physics/physics.step.service';
import { resolveAutopilotRacingLine } from '../physics/ai.autopilot';
import { createTrackSpecForAutopilotFixture } from '../physics/ai.autopilot.fixtures';
import {
  NEATRACING_SCRIPTED_PROFILE_MIN_FACTOR,
  NEATRACING_SCRIPTED_STEER_PROPORTIONAL_GAIN,
  NEATRACING_SCRIPTED_TARGET_LOOKAHEAD_SAMPLE_COUNT,
  NEATRACING_SCRIPTED_TARGET_PROJECTION_DISTANCE_M,
  NEATRACING_SCRIPTED_THROTTLE_CONSTANT,
  NEATRACING_SCRIPTED_WAYPOINT_ADVANCE_RADIUS_M,
  computeNeatRacingScriptedControl,
  createNeatRacingScriptedControllerState,
  resolveNeatRacingScriptedThrottleProfile,
} from './scripted.controller';
import { NEATRACING_RACING_LINE_LATERAL_ACCELERATION_LIMIT_M_PER_S2 } from '../track/track.racing-line';

// ── Fixtures (straight loop reused from the autopilot fixture builder; the
// circle is author-local because the controller must see a genuinely curved
// course) ─────────────────────────────────────────────────────────────────────

const CIRCLE_RADIUS_M = 100;
const CIRCLE_SAMPLE_COUNT = 40;
const FIXTURE_WIDTH_M = 12;

/** The straight closed loop: 40 samples along +x at y = 0 (autopilot fixture). */
const STRAIGHT_LOOP_SAMPLES = createTrackSpecForAutopilotFixture();

/** One-chord straight loop spec (segment 0 owns all 40 samples). */
const STRAIGHT_LOOP_TRACK_SPEC: TrackSpec = {
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'test-fixture-straight-loop',
  segments: [
    { startX: 0, startY: 0, endX: 40, endY: 0, width: FIXTURE_WIDTH_M },
  ],
  splineSamples: STRAIGHT_LOOP_SAMPLES,
};

/**
 * Metadata-split two-segment straight loop: the same straight geometry with
 * samples 0..19 owned by segment 0 and 20..39 by segment 1 — proves the
 * throttle table is keyed by the samples' OWN segment metadata, not by a
 * spline-density assumption.
 */
function createSplitSegmentStraightSamples(): readonly SplineSample[] {
  return STRAIGHT_LOOP_SAMPLES.map((sample) => ({
    ...sample,
    segmentIndex: sample.globalIndex < 20 ? 0 : 1,
    sampleIndexWithinSegment: sample.globalIndex % 20,
  }));
}

const SPLIT_SEGMENT_TRACK_SPEC: TrackSpec = {
  seed: 44,
  layoutVersion: 1,
  sizeBucket: 'test-fixture-split-loop',
  segments: [
    { startX: 0, startY: 0, endX: 20, endY: 0, width: FIXTURE_WIDTH_M },
    { startX: 20, startY: 0, endX: 40, endY: 0, width: FIXTURE_WIDTH_M },
  ],
  splineSamples: createSplitSegmentStraightSamples(),
};

/**
 * Circular closed loop (radius 100, CCW travel): a genuinely curved course —
 * the published racing line binds target speed below the top speed, so the
 * throttle table must scale below the recycled constant.
 */
function createCircularLoopSamples(): readonly SplineSample[] {
  const samples: SplineSample[] = [];
  for (let index = 0; index < CIRCLE_SAMPLE_COUNT; index += 1) {
    const angle = (index * 2 * Math.PI) / CIRCLE_SAMPLE_COUNT;
    samples.push({
      x: CIRCLE_RADIUS_M + CIRCLE_RADIUS_M * Math.cos(angle),
      y: CIRCLE_RADIUS_M * Math.sin(angle),
      width: FIXTURE_WIDTH_M,
      segmentIndex: 0,
      sampleIndexWithinSegment: index,
      globalIndex: index,
    });
  }
  return samples;
}

const CIRCULAR_LOOP_SAMPLES = createCircularLoopSamples();

const CIRCULAR_LOOP_TRACK_SPEC: TrackSpec = {
  seed: 43,
  layoutVersion: 1,
  sizeBucket: 'test-fixture-circular-loop',
  segments: [
    { startX: 0, startY: 0, endX: 200, endY: 0, width: FIXTURE_WIDTH_M },
  ],
  splineSamples: CIRCULAR_LOOP_SAMPLES,
};

/** The empty course. */
const EMPTY_TRACK_SPEC: TrackSpec = {
  seed: 0,
  layoutVersion: 0,
  sizeBucket: 'test-fixture-empty',
  segments: [],
  splineSamples: [],
};

/** The inner-lane centerline y of the straight fixture fixtures (width 12). */
const STRAIGHT_INNER_LANE_Y = FIXTURE_WIDTH_M / 2 - FIXTURE_WIDTH_M / 4;

describe('per-course throttle profile (AC-028 table)', () => {
  it('publishes exactly the recycled throttle constant for a straight course', () => {
    const profile = resolveNeatRacingScriptedThrottleProfile(
      STRAIGHT_LOOP_SAMPLES,
    );

    expect(profile.size).toBe(1);
    expect(profile.get(0)).toBe(NEATRACING_SCRIPTED_THROTTLE_CONSTANT);
  });

  it('derives identical tables from identical course layouts (determinism)', () => {
    const first = resolveNeatRacingScriptedThrottleProfile(
      STRAIGHT_LOOP_SAMPLES,
    );
    const rebuilt = resolveNeatRacingScriptedThrottleProfile(
      createTrackSpecForAutopilotFixture(),
    );

    // Same course bytes → the identical table through a fresh derivation.
    expect([...rebuilt.entries()]).toEqual([...first.entries()]);
    // Same layout instance → the SAME memoized table object.
    expect(resolveNeatRacingScriptedThrottleProfile(STRAIGHT_LOOP_SAMPLES)).toBe(
      first,
    );
  });

  it('keys the table by segment metadata without a density assumption', () => {
    const splitSamples = createSplitSegmentStraightSamples();
    const profile = resolveNeatRacingScriptedThrottleProfile(splitSamples);

    expect([...profile.keys()]).toEqual([0, 1]);
    expect(profile.get(0)).toBe(NEATRACING_SCRIPTED_THROTTLE_CONSTANT);
    expect(profile.get(1)).toBe(NEATRACING_SCRIPTED_THROTTLE_CONSTANT);
  });

  it('binds curved segments below the constant from the published racing line', () => {
    const profile = resolveNeatRacingScriptedThrottleProfile(
      CIRCULAR_LOOP_SAMPLES,
    );
    const circularEntry = profile.get(0) ?? NaN;
    const racingLine = resolveAutopilotRacingLine(CIRCULAR_LOOP_SAMPLES);
    const bindingSpeed = CIRCULAR_LOOP_SAMPLES.reduce(
      (minimum, sample) =>
        Math.min(minimum, racingLine[sample.globalIndex].targetSpeedUnitsPerS),
      Number.POSITIVE_INFINITY,
    );
    const expectedFactor = Math.min(
      1,
      Math.max(
        NEATRACING_SCRIPTED_PROFILE_MIN_FACTOR,
        bindingSpeed / NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
      ),
    );

    // First-principles: the circle publishes binding ≈ sqrt(aLimit · R).
    expect(bindingSpeed).toBeCloseTo(
      Math.sqrt(NEATRACING_RACING_LINE_LATERAL_ACCELERATION_LIMIT_M_PER_S2 * CIRCLE_RADIUS_M),
      6,
    );
    expect(circularEntry).toBeCloseTo(
      expectedFactor * NEATRACING_SCRIPTED_THROTTLE_CONSTANT,
      12,
    );
    // Different course → a different table entry than the straight one.
    expect(circularEntry).not.toBe(
      resolveNeatRacingScriptedThrottleProfile(STRAIGHT_LOOP_SAMPLES).get(0),
    );
  });

  it('bounds every entry inside the drivability band', () => {
    const profiles = [
      resolveNeatRacingScriptedThrottleProfile(STRAIGHT_LOOP_SAMPLES),
      resolveNeatRacingScriptedThrottleProfile(
        createSplitSegmentStraightSamples(),
      ),
      resolveNeatRacingScriptedThrottleProfile(CIRCULAR_LOOP_SAMPLES),
    ];

    for (const profile of profiles) {
      for (const throttle of profile.values()) {
        expect(throttle).toBeGreaterThanOrEqual(
          NEATRACING_SCRIPTED_PROFILE_MIN_FACTOR * NEATRACING_SCRIPTED_THROTTLE_CONSTANT,
        );
        expect(throttle).toBeLessThanOrEqual(NEATRACING_SCRIPTED_THROTTLE_CONSTANT);
      }
    }
  });

  it('returns an empty profile for an empty course', () => {
    expect(resolveNeatRacingScriptedThrottleProfile([]).size).toBe(0);
  });
});

describe('computeNeatRacingScriptedControl (recycled waypoint plan)', () => {
  it('guards an empty course with the recycled constant fallback', () => {
    const state = createNeatRacingScriptedControllerState();

    expect(
      computeNeatRacingScriptedControl(
        { x: 0, y: 0, heading: 0 },
        EMPTY_TRACK_SPEC,
        state,
      ),
    ).toEqual({
      throttle: NEATRACING_SCRIPTED_THROTTLE_CONSTANT,
      brake: 0,
      steer: 0,
    });
    expect(state.targetSegmentIndex).toBe(0);
  });

  it('steers exactly 0 on the inner-lane centerline of a straight loop', () => {
    const output = computeNeatRacingScriptedControl(
      { x: 2, y: STRAIGHT_INNER_LANE_Y, heading: 0 },
      STRAIGHT_LOOP_TRACK_SPEC,
      createNeatRacingScriptedControllerState(),
    );

    expect(output).toEqual({
      throttle: NEATRACING_SCRIPTED_THROTTLE_CONSTANT,
      brake: 0,
      steer: 0,
    });
  });

  it('proportionally steers from the recycled gain toward the lane-centered aim', () => {
    // Recycled aim: the CLOSEST sample's (2, 0) inner-lane centerline →
    // (2, 3), projected 14 along its +x tangent → (16, 3). Signed error
    // stays inside (-π, π), so the wrap is the identity here.
    const output = computeNeatRacingScriptedControl(
      { x: 2, y: 0, heading: 0.4 },
      STRAIGHT_LOOP_TRACK_SPEC,
      createNeatRacingScriptedControllerState(),
    );

    expect(output.steer).toBeCloseTo(
      (Math.atan2(3, 14) - 0.4) * NEATRACING_SCRIPTED_STEER_PROPORTIONAL_GAIN,
      12,
    );
    expect(output.steer).toBeLessThan(0);
  });

  it('saturates steer at ±1 for large heading errors', () => {
    const facingBackwards = computeNeatRacingScriptedControl(
      { x: 2, y: 0, heading: Math.PI },
      STRAIGHT_LOOP_TRACK_SPEC,
      createNeatRacingScriptedControllerState(),
    );
    const headingAwayNegatively = computeNeatRacingScriptedControl(
      { x: 2, y: 0, heading: -0.35 },
      STRAIGHT_LOOP_TRACK_SPEC,
      createNeatRacingScriptedControllerState(),
    );

    expect(facingBackwards.steer).toBe(-1);
    expect(headingAwayNegatively.steer).toBe(1);
  });

  it('physics proof: the emitted steer demand turns the car through the real stepPhysics tick', () => {
    // THE TURN-CHANNEL CONTRACT (AC-092/AC-093): a PRESENT turnRatePercent
    // is authoritative in the physics — even `0` commands EXACTLY zero
    // turn — so the controller must emit NONE. The proof drives the real
    // physics; the same input with the defective present-zero percent is
    // the discriminator (it must NOT turn).
    const proofCar: PhysicsCarState = {
      x: 2,
      y: 0,
      heading: 0.4,
      speed: 20,
      yawRate: 0,
      slipAngle: 0,
      travelHeadingRad: 0.4,
      tractionSeverity: 0,
      lateralVelocity: 0,
      drift: {
        committed: { slip: 0, turnSign: 0, reArmed: true },
        pose: { pose: 0, poseRate: 0, articulation: 0 },
      },
      tick: 0,
    };
    const output = computeNeatRacingScriptedControl(
      proofCar,
      STRAIGHT_LOOP_TRACK_SPEC,
      createNeatRacingScriptedControllerState(),
    );

    // The FULL car state steered (as the snapshot subset) toward the aim:
    // same proportional law the unit pin above carries.
    expect(output.steer).toBeCloseTo(
      (Math.atan2(3, 14) - 0.4) * NEATRACING_SCRIPTED_STEER_PROPORTIONAL_GAIN,
      12,
    );
    // NO turn-rate percent is emitted (the channel is deliberately absent).
    expect('turnRatePercent' in output).toBe(false);

    // ROAD proof (the default surface is 'road'): the emitted input rotates
    // the heading toward the aim — steer < 0, so the heading decreases.
    const turned = stepPhysics(proofCar, output);
    expect(turned.heading).toBeLessThan(proofCar.heading);
    expect(turned.yawRate).toBeLessThan(0);

    // THE DEFECT DISCRIMINATOR: the same input carrying a PRESENT 0 percent
    // commands EXACTLY zero turn (the repo's own AC-093 clause) — the
    // regression class that left waypoint rivals driving straight forever.
    const suppressed = stepPhysics(proofCar, { ...output, turnRatePercent: 0 });
    expect(suppressed.heading).toBe(proofCar.heading);
    expect(suppressed.yawRate).toBe(0);
  });

  it('advances the waypoint in place within the radius, wrapping modulo segments', () => {
    const state = createNeatRacingScriptedControllerState();
    const ringBack = createNeatRacingScriptedControllerState();
    ringBack.targetSegmentIndex = 1;

    // Within radius 16 of the lookahead target (14, 0): 0 → 1 (2 segments).
    computeNeatRacingScriptedControl(
      { x: 10, y: 3, heading: 0 },
      SPLIT_SEGMENT_TRACK_SPEC,
      state,
    );
    // Reaching segment 1's lookahead target (34, 0): 1 → 0 (ring wrap).
    computeNeatRacingScriptedControl(
      { x: 30, y: 3, heading: 0 },
      SPLIT_SEGMENT_TRACK_SPEC,
      ringBack,
    );

    expect(state.targetSegmentIndex).toBe(1);
    expect(ringBack.targetSegmentIndex).toBe(0);
  });

  it('holds and wrap-normalizes a negative target beyond the radius', () => {
    const state = createNeatRacingScriptedControllerState();
    state.targetSegmentIndex = -1;

    computeNeatRacingScriptedControl(
      { x: 0, y: 20, heading: 0 },
      SPLIT_SEGMENT_TRACK_SPEC,
      state,
    );

    // ((-1 % 2) + 2) % 2 = 1: normalized to the valid ring, held (no advance).
    expect(state.targetSegmentIndex).toBe(1);
  });

  it('feeds throttle from the per-course table of the road under the car', () => {
    const circularProfile = resolveNeatRacingScriptedThrottleProfile(
      CIRCULAR_LOOP_SAMPLES,
    );
    const state = createNeatRacingScriptedControllerState();

    // Car parked exactly on circle sample 0 (200, 0) → segment 0.
    const output = computeNeatRacingScriptedControl(
      { x: 200, y: 0, heading: Math.PI / 2 },
      CIRCULAR_LOOP_TRACK_SPEC,
      state,
    );

    expect(output.throttle).toBe(circularProfile.get(0));
    expect(output.brake).toBe(0);
  });

  it('mutates only the controller state, never the car or course', () => {
    const car = { x: 200, y: 0, heading: Math.PI / 2 };
    const carSnapshot = { ...car };
    const sampleCountBefore = CIRCULAR_LOOP_TRACK_SPEC.splineSamples.length;

    computeNeatRacingScriptedControl(
      car,
      CIRCULAR_LOOP_TRACK_SPEC,
      createNeatRacingScriptedControllerState(),
    );

    expect(car).toEqual(carSnapshot);
    expect(CIRCULAR_LOOP_TRACK_SPEC.splineSamples.length).toBe(sampleCountBefore);
  });

  it('is deterministic across repeated identical calls', () => {
    const first = computeNeatRacingScriptedControl(
      { x: 2, y: 0, heading: 0.4 },
      STRAIGHT_LOOP_TRACK_SPEC,
      createNeatRacingScriptedControllerState(),
    );
    const second = computeNeatRacingScriptedControl(
      { x: 2, y: 0, heading: 0.4 },
      STRAIGHT_LOOP_TRACK_SPEC,
      createNeatRacingScriptedControllerState(),
    );

    expect(second).toEqual(first);
  });
});

describe('recycled scale pins (module contract)', () => {
  it('pins the recycled throttle and steer scales verbatim', () => {
    expect(NEATRACING_SCRIPTED_THROTTLE_CONSTANT).toBe(0.82);
    expect(NEATRACING_SCRIPTED_STEER_PROPORTIONAL_GAIN).toBe(2.2);
  });

  it('pins the recycled waypoint geometry verbatim', () => {
    expect(NEATRACING_SCRIPTED_WAYPOINT_ADVANCE_RADIUS_M).toBe(16);
    expect(NEATRACING_SCRIPTED_TARGET_LOOKAHEAD_SAMPLE_COUNT).toBe(4);
    expect(NEATRACING_SCRIPTED_TARGET_PROJECTION_DISTANCE_M).toBe(14);
  });
});
