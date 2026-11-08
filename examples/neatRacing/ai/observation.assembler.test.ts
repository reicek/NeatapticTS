/**
 * Green contracts for the NeatRacing observation assembler (slice
 * `P2-S2-B`, AC-027): tiered channel counts of 82/89/110 with
 * prefix-preserving tails and one tier per test fixture.
 *
 * The fixture is the straight +x lane-center loop shared with the
 * autopilot green tests (40 samples, x = 0..39, at y = 0, width 12), so
 * every expectation is a FIRST-PRINCIPLES numeric check: the racing line
 * of a straight ring has exactly zero offset on every sample and the
 * curvature target speed equals the physics top speed, the nearest
 * sample of the focal car is sample 2 and the tangent heading is
 * exactly 0. Channel values are pinned with float32-domain precision
 * (the vector is a Float32Array) but exact zeros and lengths use `toBe`.
 *
 * @module
 */

import {
  assembleNeatRacingObservationVector,
  NEATRACING_OBSERVATION_BASE_CHANNEL_COUNT,
  NEATRACING_OBSERVATION_BOUNDARY_DISTANCE_WORLD_SCALE,
  NEATRACING_OBSERVATION_DISTANCE_WORLD_SCALE,
  NEATRACING_OBSERVATION_LOOKAHEAD_CHANNEL_COUNT,
  NEATRACING_OBSERVATION_MEMORY_TRACE_CHANNEL_COUNT,
  NEATRACING_OBSERVATION_OPTIMAL_LINE_LATERAL_OFFSET_WORLD_SCALE,
  NEATRACING_OBSERVATION_RIVAL_GAP_CHANNEL_COUNT,
  NEATRACING_OBSERVATION_SCALAR_CHANNEL_COUNT,
  NEATRACING_OBSERVATION_SELF_RADIO_CHANNEL_COUNT,
  NEATRACING_OBSERVATION_SPEED_WORLD_SCALE,
  NEATRACING_OBSERVATION_TEAMMATE_RADIO_CHANNEL_COUNT,
  NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT,
  NEATRACING_OBSERVATION_TIER_B_CHANNEL_COUNT,
  NEATRACING_OBSERVATION_TIER_C_CHANNEL_COUNT,
  NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
  NEATRACING_OBSERVATION_WHEEL_SLIP_CHANNEL_COUNT,
  NEATRACING_OBSERVATION_WHEEL_SLIP_SLIP_ANGLE_SCALE,
  NEATRACING_OBSERVATION_WHEEL_SLIP_YAW_RATE_SCALE,
  NEATRACING_TEAM_LAYOUTS,
  resolveNeatRacingObservationChannelCount,
  type NeatRacingObservationCarState,
  type NeatRacingObservationState,
} from './observation.assembler';
import { createTrackSpecForAutopilotFixture } from '../physics/ai.autopilot.fixtures';
import { NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S } from '../physics/physics.step.service';
import type { TrackSpec } from '../../racing_curriculum/track/track.generator.types';

/* ------------------------------------------------------------------------- *
 * Fixture constants (first-principles values of the straight loop)
 * ------------------------------------------------------------------------- */

/** Straight-loop spline samples (40 samples, x = 0..39, y = 0, width 12). */
const FIXTURE_SAMPLES = createTrackSpecForAutopilotFixture();

/** Fixture track spec wrapping the straight loop in the frozen contract. */
const FIXTURE_TRACK_SPEC: TrackSpec = {
  seed: 42,
  layoutVersion: 1,
  sizeBucket: 'test-fixture',
  segments: [],
  splineSamples: FIXTURE_SAMPLES,
};

/** Track-bounds center x of the fixture (bbox of x = 0..39 -> 19.5). */
const FIXTURE_TRACK_CENTER_X = 19.5;

/** Focal car x of the primary fixtures (parked on sample 2). */
const FIXTURE_CAR_X = 2;

/** Fixture lane half width (width 12 -> 6). */
const FIXTURE_HALF_WIDTH_M = 6;

/** Focal car forward speed world units/s of the primary fixtures. */
const FIXTURE_SPEED_WORLD = 10;

/** Curvature target speed of the straight fixture (the physics top speed). */
const FIXTURE_TOP_SPEED_WORLD = NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S;

/** The 5-car level 3 ladder roster positions (x, y) along the straight loop. */
const LADDER_ROSTER_POSITIONS: readonly (readonly [number, number])[] = [
  [FIXTURE_CAR_X, 0],
  [7, 0],
  [2, -3],
  [-3, 0],
  [1, 2.5],
];

/** The ladder roster forward speeds (focal + four team B rivals). */
const LADDER_ROSTER_SPEEDS_WORLD: readonly number[] = [10, 8, 6, 5, 7];

/** Float32-domain comparison precision for normalized channel pins. */
const FLOAT32_PIN_PRECISION = 6;

/**
 * Converts an expected float64 channel block through the SAME Float32Array
 * storage the assembler writes into, so numeric comparisons stay
 * bit-identical to the implementation path instead of debating float32
 * rounding per value.
 */
function float32ChannelBlock(values: readonly number[]): number[] {
  return Array.from(Float32Array.from(values));
}

/* ------------------------------------------------------------------------- *
 * Fixture builders
 * ------------------------------------------------------------------------- */

/** Builds the primary observation state (focal parked on sample 2). */
function createBaseState(): NeatRacingObservationState {
  return {
    carX: FIXTURE_CAR_X,
    carY: 0,
    carHeading: 0,
    forwardSpeedWorld: FIXTURE_SPEED_WORLD,
    lateralSpeedWorld: 0,
  };
}

/**
 * Builds the level 3 ladder roster state: the `[0, 1, 1, 1, 1]` E2 layout
 * with the focal car on sample 2 (team A) and the four team B rivals
 * placed at deterministic body-frame-offset positions.
 */
function createLadderRosterState(): NeatRacingObservationState {
  const ladderTeams = NEATRACING_TEAM_LAYOUTS[3];

  return {
    ...createBaseState(),
    cars: LADDER_ROSTER_POSITIONS.map(
      ([carX, carY], rosterIndex): NeatRacingObservationCarState => {
        return {
          carX,
          carY,
          carHeading: 0,
          speedWorld: LADDER_ROSTER_SPEEDS_WORLD[rosterIndex],
          teamIndex: ladderTeams[rosterIndex] as 0 | 1,
        };
      },
    ),
  };
}

/**
 * Builds the teammate-roster state: the focal car sits in a SAME-TEAM
 * roster so the Tier C radio tail carries two real teammates and one
 * zero-padded slot (the human-ladder team is size 1, so the assembler
 * must stay padded there; this roster exercises the populated path).
 */
function createTeammateRosterState(): NeatRacingObservationState {
  return {
    carX: FIXTURE_CAR_X,
    carY: 0,
    carHeading: 0,
    forwardSpeedWorld: FIXTURE_SPEED_WORLD,
    cars: [
      {
        carX: FIXTURE_CAR_X,
        carY: 0,
        carHeading: 0,
        speedWorld: FIXTURE_SPEED_WORLD,
        teamIndex: 1,
      },
      {
        carX: 9,
        carY: 0,
        carHeading: 0,
        speedWorld: 12,
        teamIndex: 1,
      },
      {
        carX: 3,
        carY: -2,
        carHeading: 0.5,
        speedWorld: 11,
        teamIndex: 1,
      },
      {
        carX: 4,
        carY: 1,
        carHeading: 1,
        teamIndex: 0,
      },
      {
        carX: 0,
        carY: 1,
        carHeading: -1,
        teamIndex: 0,
      },
    ],
  };
}

/**
 * Builds the expected 8-channel rival-gap block of the ladder roster in
 * body frame (heading 0, so longitudinal = dx and lateral = dy).
 */
function expectedLadderRivalGapChannels(): number[] {
  const focal = LADDER_ROSTER_POSITIONS[0]!;
  return [
    [7 - focal[0], 0], // rival 1: 5 ahead
    [2 - focal[0], -3], // rival 2: 3 right
    [-3 - focal[0], 0], // rival 3: 5 behind
    [1 - focal[0], 2.5], // rival 4: 1 behind, 2.5 left
  ].flatMap(([deltaX, deltaY]) => [
    deltaX / NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
    deltaY / NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
  ]);
}

/* ------------------------------------------------------------------------- *
 * Tier A (level 1, 82 channels)
 * ------------------------------------------------------------------------- */

describe('assembleNeatRacingObservationVector (Tier A, level 1)', () => {
  it('emits 82 finite normalized channels: base 70 + wheel slip 4 + rival gaps 8', () => {
    const observationVector = assembleNeatRacingObservationVector(
      createBaseState(),
      FIXTURE_TRACK_SPEC,
      { level: 1 },
    );
    const channels = Array.from(observationVector);

    expect(channels.length).toBe(NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT);
    expect(channels.every((value) => Number.isFinite(value))).toBe(true);
    expect(
      channels.every((value) => value >= -1 && value <= 1),
    ).toBe(true);
  });

  it('pins the base-state scalar channels 0..6 (pose + body-frame speeds)', () => {
    const observationVector = assembleNeatRacingObservationVector(
      createBaseState(),
      FIXTURE_TRACK_SPEC,
      { level: 1 },
    );

    expect(Array.from(observationVector.slice(0, 7))).toEqual(
      float32ChannelBlock([
        (FIXTURE_CAR_X - FIXTURE_TRACK_CENTER_X) /
          NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE, // ch0 x offset
        0, // ch1 y offset
        0, // ch2 sin(heading)
        1, // ch3 cos(heading)
        FIXTURE_SPEED_WORLD / NEATRACING_OBSERVATION_SPEED_WORLD_SCALE, // ch4 forward
        0, // ch5 lateral
        FIXTURE_SPEED_WORLD / NEATRACING_OBSERVATION_SPEED_WORLD_SCALE, // ch6 speed
      ]),
    );
  });

  it('pins the derived scalars 7..19 (LD-15, progress, boundaries, optimal line)', () => {
    const observationVector = assembleNeatRacingObservationVector(
      createBaseState(),
      FIXTURE_TRACK_SPEC,
      { level: 1 },
    );

    expect(Array.from(observationVector.slice(7, 20))).toEqual(
      float32ChannelBlock([
        0, // ch7 yaw rate (rad/s, absent -> 0)
        0, // ch8 slip angle (absent -> 0)
        -0.9, // ch9 progress01 = 2/40 -> 0.05 x 2 - 1
        -0.9, // ch10 lapProgress01 falls back to progress01
        FIXTURE_HALF_WIDTH_M /
          NEATRACING_OBSERVATION_BOUNDARY_DISTANCE_WORLD_SCALE, // ch11 left
        FIXTURE_HALF_WIDTH_M /
          NEATRACING_OBSERVATION_BOUNDARY_DISTANCE_WORLD_SCALE, // ch12 right
        0, // ch13 boundary balance ((6-6)/12)
        1, // ch14 hazard distance defaults to the distance scale (64/64)
        0, // ch15 waypoint distance (car on sample 2)
        0, // ch16 optimal-line lateral offset (0 - straight racing offset 0)
        0, // ch17 optimal-line heading error (tangent 0 - heading 0)
        FIXTURE_TOP_SPEED_WORLD / NEATRACING_OBSERVATION_SPEED_WORLD_SCALE, // ch18 target
        (FIXTURE_TOP_SPEED_WORLD - FIXTURE_SPEED_WORLD) /
          NEATRACING_OBSERVATION_SPEED_WORLD_SCALE, // ch19 margin
      ]),
    );
  });

  it('pins the 40 look-ahead channels 20..59 (stride 18 on the straight ring)', () => {
    const observationVector = assembleNeatRacingObservationVector(
      createBaseState(),
      FIXTURE_TRACK_SPEC,
      { level: 1 },
    );

    const lookAheadSampleIndices = [2, 20, 38, 16, 34];
    const expectedLookAheadChannels = lookAheadSampleIndices.flatMap(
      (sampleIndex) => [
        (sampleIndex - FIXTURE_CAR_X) /
          NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
        0,
        ((sampleIndex + 1) % FIXTURE_SAMPLES.length - FIXTURE_CAR_X) /
          NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
        0,
        0, // sin(tangent) = sin(0)
        1, // cos(tangent)
        FIXTURE_SAMPLES[sampleIndex]!.width /
          NEATRACING_OBSERVATION_BOUNDARY_DISTANCE_WORLD_SCALE,
        (sampleIndex - FIXTURE_CAR_X) /
          NEATRACING_OBSERVATION_DISTANCE_WORLD_SCALE,
      ],
    );

    expect(
      Array.from(
        observationVector.slice(
          NEATRACING_OBSERVATION_SCALAR_CHANNEL_COUNT,
          NEATRACING_OBSERVATION_SCALAR_CHANNEL_COUNT +
            NEATRACING_OBSERVATION_LOOKAHEAD_CHANNEL_COUNT,
        ),
      ),
    ).toEqual(float32ChannelBlock(expectedLookAheadChannels));
  });

  it('zero-pads the memory trace channels 60..69 when absent', () => {
    const observationVector = assembleNeatRacingObservationVector(
      createBaseState(),
      FIXTURE_TRACK_SPEC,
      { level: 1 },
    );
    const memoryChannels = Array.from(
      observationVector.slice(
        NEATRACING_OBSERVATION_SCALAR_CHANNEL_COUNT +
          NEATRACING_OBSERVATION_LOOKAHEAD_CHANNEL_COUNT,
        NEATRACING_OBSERVATION_BASE_CHANNEL_COUNT,
      ),
    );

    expect(memoryChannels.length).toBe(
      NEATRACING_OBSERVATION_MEMORY_TRACE_CHANNEL_COUNT,
    );
    expect(memoryChannels.every((value) => value === 0)).toBe(true);
  });

  it('clamps the provided memory trace into [-1, 1] and zero-pads to 10', () => {
    const observationVector = assembleNeatRacingObservationVector(
      { ...createBaseState(), memoryTrace: [2, -2, 0.25] },
      FIXTURE_TRACK_SPEC,
      { level: 1 },
    );

    expect(Array.from(observationVector.slice(60, 70))).toEqual([
      1,
      -1,
      0.25,
      0,
      0,
      0,
      0,
      0,
      0,
      0,
    ]);
  });

  it('derives the four wheel-slip channels 70..73 from the LD-15 pair with side polarity', () => {
    const slipAngleRadians = 0.2;
    const yawRateRadiansPerSecond = 0.4;
    const observationVector = assembleNeatRacingObservationVector(
      {
        ...createBaseState(),
        slipAngleRadians,
        yawRateRadiansPerSecond,
      },
      FIXTURE_TRACK_SPEC,
      { level: 1 },
    );
    const expectedSlideIntensity =
      (slipAngleRadians / NEATRACING_OBSERVATION_WHEEL_SLIP_SLIP_ANGLE_SCALE +
        yawRateRadiansPerSecond /
          NEATRACING_OBSERVATION_WHEEL_SLIP_YAW_RATE_SCALE) /
      2;

    const [frontLeft, frontRight, rearLeft, rearRight] = Array.from(
      observationVector.slice(
        NEATRACING_OBSERVATION_BASE_CHANNEL_COUNT,
        NEATRACING_OBSERVATION_BASE_CHANNEL_COUNT +
          NEATRACING_OBSERVATION_WHEEL_SLIP_CHANNEL_COUNT,
      ),
    );

    expect(frontLeft).toBeCloseTo(expectedSlideIntensity, FLOAT32_PIN_PRECISION);
    expect(frontRight).toBeCloseTo(
      -expectedSlideIntensity,
      FLOAT32_PIN_PRECISION,
    );
    expect(rearLeft).toBeCloseTo(expectedSlideIntensity, FLOAT32_PIN_PRECISION);
    expect(rearRight).toBeCloseTo(
      -expectedSlideIntensity,
      FLOAT32_PIN_PRECISION,
    );
  });

  it('reads exactly zero on every wheel-slip channel for a straight drive', () => {
    const observationVector = assembleNeatRacingObservationVector(
      createBaseState(),
      FIXTURE_TRACK_SPEC,
      { level: 1 },
    );
    const wheelSlipChannels = Array.from(
      observationVector.slice(
        NEATRACING_OBSERVATION_BASE_CHANNEL_COUNT,
        NEATRACING_OBSERVATION_BASE_CHANNEL_COUNT +
          NEATRACING_OBSERVATION_WHEEL_SLIP_CHANNEL_COUNT,
      ),
    );

    expect(wheelSlipChannels).toEqual([0, 0, 0, 0]);
  });

  it('fills rival-gap channels 74..81 with body-frame pairs in roster order', () => {
    const observationVector = assembleNeatRacingObservationVector(
      createLadderRosterState(),
      FIXTURE_TRACK_SPEC,
      { level: 1 },
    );
    const rivalGapChannels = Array.from(
      observationVector.slice(
        NEATRACING_OBSERVATION_BASE_CHANNEL_COUNT +
          NEATRACING_OBSERVATION_WHEEL_SLIP_CHANNEL_COUNT,
        NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT,
      ),
    );

    expect(rivalGapChannels.length).toBe(
      NEATRACING_OBSERVATION_RIVAL_GAP_CHANNEL_COUNT,
    );
    expect(rivalGapChannels).toEqual(
      float32ChannelBlock(expectedLadderRivalGapChannels()),
    );
  });

  it('zero-fills the rival-gap channels for a roster-free solo state', () => {
    const observationVector = assembleNeatRacingObservationVector(
      createBaseState(),
      FIXTURE_TRACK_SPEC,
      { level: 1 },
    );
    const rivalGapChannels = Array.from(
      observationVector.slice(
        NEATRACING_OBSERVATION_BASE_CHANNEL_COUNT +
          NEATRACING_OBSERVATION_WHEEL_SLIP_CHANNEL_COUNT,
        NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT,
      ),
    );

    expect(rivalGapChannels).toEqual([0, 0, 0, 0, 0, 0, 0, 0]);
  });

  it('derives lane balance and optimal-line offset for an off-center car (AC-086 source)', () => {
    const observationVector = assembleNeatRacingObservationVector(
      { ...createBaseState(), carY: 1 },
      FIXTURE_TRACK_SPEC,
      { level: 1 },
    );

    const boundaryLeft = Array.from(observationVector.slice(11, 12))[0]!;
    const boundaryRight = Array.from(observationVector.slice(12, 13))[0]!;
    const balance = Array.from(observationVector.slice(13, 14))[0]!;
    const waypointDistance = Array.from(observationVector.slice(15, 16))[0]!;
    const optimalLineOffset = Array.from(observationVector.slice(16, 17))[0]!;

    expect(boundaryLeft).toBeCloseTo(
      (FIXTURE_HALF_WIDTH_M - 1) /
        NEATRACING_OBSERVATION_BOUNDARY_DISTANCE_WORLD_SCALE,
      FLOAT32_PIN_PRECISION,
    );
    expect(boundaryRight).toBeCloseTo(
      (FIXTURE_HALF_WIDTH_M + 1) /
        NEATRACING_OBSERVATION_BOUNDARY_DISTANCE_WORLD_SCALE,
      FLOAT32_PIN_PRECISION,
    );
    expect(balance).toBeCloseTo(2 / 12, FLOAT32_PIN_PRECISION);
    expect(waypointDistance).toBeCloseTo(
      1 / NEATRACING_OBSERVATION_DISTANCE_WORLD_SCALE,
      FLOAT32_PIN_PRECISION,
    );
    expect(optimalLineOffset).toBeCloseTo(
      1 / NEATRACING_OBSERVATION_OPTIMAL_LINE_LATERAL_OFFSET_WORLD_SCALE,
      FLOAT32_PIN_PRECISION,
    );
  });
});

/* ------------------------------------------------------------------------- *
 * Tier B (level 2, 89 channels)
 * ------------------------------------------------------------------------- */

describe('assembleNeatRacingObservationVector (Tier B, level 2)', () => {
  const radioField = [0.11, -0.22, 0.33, -0.44, 0.55, -0.66, 0.77];

  it('emits 89 channels that prefix-preserve the Tier A vector', () => {
    const state = createLadderRosterState();
    const tierAVector = assembleNeatRacingObservationVector(
      state,
      FIXTURE_TRACK_SPEC,
      { level: 1 },
    );
    const tierBVector = assembleNeatRacingObservationVector(
      state,
      FIXTURE_TRACK_SPEC,
      { level: 2 },
    );

    expect(tierBVector.length).toBe(NEATRACING_OBSERVATION_TIER_B_CHANNEL_COUNT);
    expect(Array.from(tierBVector.slice(0, NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT))).toEqual(
      Array.from(tierAVector),
    );
  });

  it('appends the raw seven-channel self-radio tail at 82..88 without renormalization', () => {
    const observationVector = assembleNeatRacingObservationVector(
      { ...createBaseState(), radioField },
      FIXTURE_TRACK_SPEC,
      { level: 2 },
    );
    const radioTail = Array.from(
      observationVector.slice(
        NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT,
        NEATRACING_OBSERVATION_TIER_B_CHANNEL_COUNT,
      ),
    );

    expect(radioTail.length).toBe(
      NEATRACING_OBSERVATION_SELF_RADIO_CHANNEL_COUNT,
    );
    expect(radioTail).toEqual(Array.from(Float32Array.from(radioField)));
  });

  it('zero-fills the self-radio tail when the state has no radio field', () => {
    const observationVector = assembleNeatRacingObservationVector(
      createBaseState(),
      FIXTURE_TRACK_SPEC,
      { level: 2 },
    );
    const radioTail = Array.from(
      observationVector.slice(
        NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT,
        NEATRACING_OBSERVATION_TIER_B_CHANNEL_COUNT,
      ),
    );

    expect(radioTail).toEqual([0, 0, 0, 0, 0, 0, 0]);
  });
});

/* ------------------------------------------------------------------------- *
 * Tier C (level 3, 110 channels)
 * ------------------------------------------------------------------------- */

describe('assembleNeatRacingObservationVector (Tier C, level 3)', () => {
  it('emits 110 channels that prefix-preserve the Tier B vector', () => {
    const state = createLadderRosterState();
    const tierBVector = assembleNeatRacingObservationVector(
      state,
      FIXTURE_TRACK_SPEC,
      { level: 2 },
    );
    const tierCVector = assembleNeatRacingObservationVector(
      state,
      FIXTURE_TRACK_SPEC,
      { level: 3 },
    );

    expect(tierCVector.length).toBe(NEATRACING_OBSERVATION_TIER_C_CHANNEL_COUNT);
    expect(Array.from(tierCVector.slice(0, NEATRACING_OBSERVATION_TIER_B_CHANNEL_COUNT))).toEqual(
      Array.from(tierBVector),
    );
  });

  it('covers the 4-rival [0,1,1,1,1] ladder with all eight rival-gap channels populated', () => {
    const observationVector = assembleNeatRacingObservationVector(
      createLadderRosterState(),
      FIXTURE_TRACK_SPEC,
      { level: 3 },
    );
    const rivalGapChannels = Array.from(
      observationVector.slice(
        NEATRACING_OBSERVATION_BASE_CHANNEL_COUNT +
          NEATRACING_OBSERVATION_WHEEL_SLIP_CHANNEL_COUNT,
        NEATRACING_OBSERVATION_BASE_CHANNEL_COUNT +
          NEATRACING_OBSERVATION_WHEEL_SLIP_CHANNEL_COUNT +
          NEATRACING_OBSERVATION_RIVAL_GAP_CHANNEL_COUNT,
      ),
    );

    expect(rivalGapChannels).toEqual(
      float32ChannelBlock(expectedLadderRivalGapChannels()),
    );
  });

  it('keeps the teammate tail zero for the human ladder car (no same-team entries)', () => {
    const observationVector = assembleNeatRacingObservationVector(
      createLadderRosterState(),
      FIXTURE_TRACK_SPEC,
      { level: 3 },
    );
    const teammateChannels = Array.from(
      observationVector.slice(
        NEATRACING_OBSERVATION_TIER_B_CHANNEL_COUNT,
        NEATRACING_OBSERVATION_TIER_C_CHANNEL_COUNT,
      ),
    );

    expect(teammateChannels.length).toBe(
      NEATRACING_OBSERVATION_TEAMMATE_RADIO_CHANNEL_COUNT,
    );
    expect(teammateChannels.every((value) => value === 0)).toBe(true);
  });

  it('encodes the recycled 7-channel teammate slots from same-team roster cars', () => {
    const observationVector = assembleNeatRacingObservationVector(
      createTeammateRosterState(),
      FIXTURE_TRACK_SPEC,
      { level: 3 },
    );
    const teammateChannels = Array.from(
      observationVector.slice(
        NEATRACING_OBSERVATION_TIER_B_CHANNEL_COUNT,
        NEATRACING_OBSERVATION_TIER_C_CHANNEL_COUNT,
      ),
    );

    const teammateOne = teammateChannels.slice(0, 7);
    const teammateTwo = teammateChannels.slice(7, 14);
    const paddedSlot = teammateChannels.slice(14, 21);

    // THE RECYCLED SLOT: [posX/96, posY/96, sin(heading), speed/108,
    // relOffsetX/96, relOffsetY/96, sin(relHeading)].
    expect(teammateOne).toEqual(
      float32ChannelBlock([
        9 / NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE, // posX
        0, // posY
        0, // sin(heading 0)
        12 / NEATRACING_OBSERVATION_SPEED_WORLD_SCALE, // speed
        (9 - FIXTURE_CAR_X) / NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE, // relOffsetX
        0, // relOffsetY
        0, // sin(heading delta 0)
      ]),
    );
    expect(teammateTwo[0]).toBeCloseTo(
      3 / NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
      FLOAT32_PIN_PRECISION,
    );
    expect(teammateTwo[1]).toBeCloseTo(
      -2 / NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
      FLOAT32_PIN_PRECISION,
    );
    expect(teammateTwo[2]).toBeCloseTo(
      Math.sin(0.5),
      FLOAT32_PIN_PRECISION,
    );
    expect(teammateTwo[3]).toBeCloseTo(
      11 / NEATRACING_OBSERVATION_SPEED_WORLD_SCALE,
      FLOAT32_PIN_PRECISION,
    );
    expect(teammateTwo[4]).toBeCloseTo(
      1 / NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
      FLOAT32_PIN_PRECISION,
    );
    expect(teammateTwo[5]).toBeCloseTo(
      -2 / NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
      FLOAT32_PIN_PRECISION,
    );
    expect(teammateTwo[6]).toBeCloseTo(
      Math.sin(0.5),
      FLOAT32_PIN_PRECISION,
    );
    expect(paddedSlot).toEqual([0, 0, 0, 0, 0, 0, 0]);
  });

  it('pads the rival-gap block beyond the tracked rivals of the teammate roster', () => {
    const observationVector = assembleNeatRacingObservationVector(
      createTeammateRosterState(),
      FIXTURE_TRACK_SPEC,
      { level: 3 },
    );
    const rivalGapChannels = Array.from(
      observationVector.slice(
        NEATRACING_OBSERVATION_BASE_CHANNEL_COUNT +
          NEATRACING_OBSERVATION_WHEEL_SLIP_CHANNEL_COUNT,
        NEATRACING_OBSERVATION_BASE_CHANNEL_COUNT +
          NEATRACING_OBSERVATION_WHEEL_SLIP_CHANNEL_COUNT +
          NEATRACING_OBSERVATION_RIVAL_GAP_CHANNEL_COUNT,
      ),
    );

    // Rival 1 at (+2, +1), rival 2 at (-2, +1); slots 2 and 3 zero-padded.
    expect(rivalGapChannels).toEqual(
      float32ChannelBlock([
        2 / NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
        1 / NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
        -2 / NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
        1 / NEATRACING_OBSERVATION_TRACK_POSITION_WORLD_SCALE,
        0,
        0,
        0,
        0,
      ]),
    );
  });
});

/* ------------------------------------------------------------------------- *
 * Module contracts
 * ------------------------------------------------------------------------- */

describe('observation.assembler module contracts', () => {
  it('pins the E2 team ladder [0] / [0, 1] / [0, 1, 1, 1, 1]', () => {
    expect(NEATRACING_TEAM_LAYOUTS).toEqual({
      1: [0],
      2: [0, 1],
      3: [0, 1, 1, 1, 1],
    });
  });

  it('maps ladder level 1 to the Tier A count of 82', () => {
    expect(resolveNeatRacingObservationChannelCount(1)).toBe(
      NEATRACING_OBSERVATION_TIER_A_CHANNEL_COUNT,
    );
  });

  it('maps ladder level 2 to the Tier B count of 89', () => {
    expect(resolveNeatRacingObservationChannelCount(2)).toBe(
      NEATRACING_OBSERVATION_TIER_B_CHANNEL_COUNT,
    );
  });

  it('maps ladder level 3 to the Tier C count of 110', () => {
    expect(resolveNeatRacingObservationChannelCount(3)).toBe(
      NEATRACING_OBSERVATION_TIER_C_CHANNEL_COUNT,
    );
  });

  it('rejects an empty spline-sample loop (frozen TrackSpec contract)', () => {
    expect(() => {
      assembleNeatRacingObservationVector(
        createBaseState(),
        { ...FIXTURE_TRACK_SPEC, splineSamples: [] },
        { level: 1 },
      );
    }).toThrow(RangeError);
  });
});
