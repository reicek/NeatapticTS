/**
 * ============================================================================
 * P5-S28-red (plan step 28) - THE MEASURED DEPARTURE BASELINE (AC-107).
 *
 * THE GAP (what is missing today): NOTHING in the served code MEASURES how
 * often the player car leaves the drivable surface on the served geometry.
 * The EXISTING off-track counter reads the COARSE verdict
 * (`surfaceResolver(car) !== 'road'` - the curb band and the whole drivable
 * grass band count as "off track"), while the ADVISORY departure hint this
 * step ships reads the FINE verdict (`TrackSurfaceClassification.isDrivable
 * === false` - the car really left the drivable surface). The two verdicts
 * are DIFFERENT laws over the SAME published classifier, so the departure
 * rate has to be MEASURED per verdict - otherwise the GREEN lane compares
 * its advisory hint against a HAND-WRITTEN expectation instead of a
 * measured baseline.
 *
 * THE GREEN TARGET: the ADVISORY "return to the track" hint, derived from
 * the ONE published FINE classification (the curb band and the drivable
 * grass band are STILL drivable; only the open grass beyond the band is a
 * departure), published as a HUD overlay that changes NO pose, NO speed and
 * NO penalty. THIS file is the BASELINE that GREEN compares against: it
 * pins the MEASURED current departure rate of the served geometry, and it
 * pins the two invariants the GREEN lane MUST NOT break - the NO-RECOVERY
 * law (a departed car is NEVER reseated, NEVER teleported and NEVER
 * auto-reset) and the PENALTY REGRESSION GUARD (the EXISTING coarse
 * off-track counter still reads the coarse verdict and the fine verdict is
 * never blurred into it).
 *
 * THE MEASURED-BASELINE LAW: every number pinned below was MEASURED on the
 * SERVED geometry (`resolveTrackVariantSpec(NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID)`)
 * BEFORE any production edit, and is NEVER hand-written: the measurement
 * run that produced this file logged the drive (600 driven frames, 389 fine
 * departure frames, 394 coarse off-road frames, first departure frame 211)
 * and the measured values were then pinned as the named module constants
 * below. The GREEN lane re-measures the SAME drive and compares against
 * THESE constants, so a drift between the served geometry and this
 * baseline is a NAMED runtime failure instead of a silent redefinition of
 * the departure rate.
 *
 * THE FINE / COARSE DISTINCTION IS LOAD-BEARING AND MUST NOT BE BLURRED:
 * - the FINE verdict (`isDrivable === false`) means the car has LEFT the
 *   drivable surface (the open grass beyond the band outer edge);
 * - the COARSE verdict (`surface !== 'road'`) is the EXACT predicate the
 *   EXISTING off-track counter reads (the curb band and the drivable band
 *   both count as off-road for the penalty);
 * this file pins BOTH, side by side, on the same served geometry.
 *
 * THE DRIVE PROTOCOL (the ONE measured baseline, documented so the GREEN
 * lane re-runs the EXACT same drive): the player car starts at the
 * MIDPOINT of the FIRST chord of the served ribbon (`spec.segments[0]`),
 * heading ALONG that chord (`atan2(endY - startY, endX - startX)`), and is
 * driven for 600 fixed ticks (60 Hz, so 10 s) with the ONE control sample
 * `{ throttle: 1, brake: 0, steer: 0 }` through
 * `stepPhysics(state, input, surfaceResolver, spec.segments)`. EVERY
 * driven frame is then classified through the ONE published classifier
 * (`createTrackSurfaceClassifier(spec.segments)`) AT THE POSE THE CAR
 * REACHED IN THAT FRAME (the same pose the live off-track counter reads),
 * and the frames are counted per verdict: the FINE departure frames (the
 * car left the drivable surface) and the COARSE off-road frames (the
 * verdict the EXISTING off-track counter reads).
 *
 * THE BAND LADDER (the four PUBLISHED band poses, walked PERPENDICULARLY
 * away from the first chord's midpoint in 0.25 m steps up to 40 m): the
 * measured offsets on the served geometry are the road core at 0 m, the
 * curb at 14.75 m, the DRIVABLE grass band at 15.75 m and the open grass
 * at 18.75 m - the walk keeps the FIRST pose of each band as the ONE
 * published classifier reports it, so the ladder is NEVER hand-placed.
 *
 * Determinism: the served geometry is a pure function of the catalog entry,
 * the physics tick is pure, and this file reads NO wall clock and NO RNG
 * (`Math.random` / `Date.now` never appear) - so the measured baseline is
 * byte-reproducible on every run.
 *
 * Red-phase discipline: this file imports ONLY modules that EXIST today
 * (the published classifier, the step service, the catalog), so every
 * failure it can produce is a RUNTIME ASSERTION failure and NEVER an
 * import / module-resolution failure.
 */

import { describe, expect, it } from '@jest/globals';
import {
  createInitialPhysicsCarState,
  stepPhysics,
} from './physics.step.service';
import type {
  PhysicsCarState,
  PhysicsStepInput,
} from './physics.step.service';
import {
  createTrackSurfaceClassifier,
  createTrackSurfaceResolver,
} from './collision.utils';
import {
  NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID,
  resolveTrackVariantSpec,
} from '../track/track.catalog';
import type { TrackSegment } from '../../racing_curriculum/track/track.generator.types';

/**
 * The SERVED ribbon of the boot-default demo track: the closed-loop chord
 * chain every measurement below is resolved against.
 */
const SERVED_SEGMENTS: readonly TrackSegment[] = resolveTrackVariantSpec(
  NEATRACING_DEMO_TRACK_CATALOG_DEFAULT_ID,
).segments;

/** The length of the measured baseline drive, in fixed ticks (60 Hz = 10 s). */
const BASELINE_DRIVE_TICKS = 600;

/** The ONE control sample of the measured baseline drive. */
const BASELINE_DRIVE_INPUT: PhysicsStepInput = {
  throttle: 1,
  brake: 0,
  steer: 0,
};

/** The lateral step of the band walk, in meters. */
const BAND_WALK_STEP_M = 0.25;

/** The maximum lateral offset of the band walk, in meters. */
const BAND_WALK_MAX_OFFSET_M = 40;

/** The speed of the DEPARTED car state, in meters per second. */
const DEPARTED_CAR_SPEED_M_PER_S = 10;

/**
 * The MEASURED number of driven frames of the baseline drive.
 *
 * MEASURED on the served geometry before any production edit (600 fixed
 * ticks at 60 Hz = 10 s of driving) - never hand-written.
 */
const MEASURED_DRIVEN_FRAMES = 600;

/**
 * The MEASURED number of frames whose FINE verdict reads
 * `isDrivable === false` (the car LEFT the drivable surface).
 *
 * MEASURED on the served geometry before any production edit - never
 * hand-written.
 */
const MEASURED_FINE_DEPARTURE_FRAMES = 389;

/**
 * The MEASURED number of frames whose COARSE verdict reads
 * `surface !== 'road'` (the verdict the EXISTING off-track counter reads).
 *
 * MEASURED on the served geometry before any production edit - never
 * hand-written.
 */
const MEASURED_COARSE_OFF_ROAD_FRAMES = 394;

/**
 * The MEASURED index of the FIRST frame whose FINE verdict reads
 * `isDrivable === false` (the frame the car left the drivable surface on).
 *
 * MEASURED on the served geometry before any production edit - never
 * hand-written.
 */
const MEASURED_FIRST_DEPARTURE_FRAME = 211;

/**
 * The measured outcome of the baseline drive.
 */
interface BaselineDriveMeasurement {
  /** The number of driven frames (the ticks that were stepped). */
  readonly drivenFrames: number;
  /** The frames whose FINE verdict reads `isDrivable === false`. */
  readonly fineDepartureFrames: number;
  /** The frames whose COARSE verdict reads `surface !== 'road'`. */
  readonly coarseOffRoadFrames: number;
  /** The first frame whose FINE verdict reads `isDrivable === false`. */
  readonly firstDepartureFrame: number | null;
}

/**
 * One lateral band pose on the served geometry, addressed by the signed
 * lateral offset from the first chord's midpoint.
 */
interface BandPose {
  /** The signed lateral offset from the first chord's midpoint, in meters. */
  readonly offsetM: number;
  /** The world x of the pose, in meters. */
  readonly x: number;
  /** The world y of the pose, in meters. */
  readonly y: number;
  /** The heading of the pose, in radians (along the outward lateral normal). */
  readonly heading: number;
}

/**
 * The four PUBLISHED band poses of the served geometry.
 */
interface PublishedBandPoses {
  /** The road core (the road-only lane). */
  readonly roadCore: BandPose;
  /** The curb band. */
  readonly curb: BandPose;
  /** The DRIVABLE off-track band (the grass apron). */
  readonly drivableBand: BandPose;
  /** The open grass BEYOND the drivable band. */
  readonly openGrass: BandPose;
}

/**
 * The pose of the first chord's midpoint, heading ALONG that chord.
 *
 * @returns The midpoint coordinates and the along-chord heading, in radians.
 */
function resolveFirstChordPose(): {
  readonly x: number;
  readonly y: number;
  readonly heading: number;
} {
  const chord = SERVED_SEGMENTS[0];
  const deltaX = chord.endX - chord.startX;
  const deltaY = chord.endY - chord.startY;
  return {
    x: (chord.startX + chord.endX) / 2,
    y: (chord.startY + chord.endY) / 2,
    heading: Math.atan2(deltaY, deltaX),
  };
}

/**
 * Builds a standing car state at one world pose.
 *
 * @param x - The world x of the pose, in meters.
 * @param y - The world y of the pose, in meters.
 * @param heading - The heading of the pose, in radians.
 * @returns A fresh standing car state at the pose.
 */
function buildStandingCarState(
  x: number,
  y: number,
  heading: number,
): PhysicsCarState {
  return createInitialPhysicsCarState(0, { x, y, heading });
}

/**
 * Builds the car state that occupies one band pose.
 *
 * @param pose - The band pose the car occupies.
 * @returns A fresh standing car state at the band pose.
 */
function buildCarStateAtBandPose(pose: BandPose): PhysicsCarState {
  return buildStandingCarState(pose.x, pose.y, pose.heading);
}

/**
 * MEASURES the baseline drive on the served geometry: the player car starts
 * at the midpoint of the FIRST chord, heading along that chord, and is
 * driven for {@link BASELINE_DRIVE_TICKS} fixed ticks with the ONE control
 * sample. EVERY driven frame is classified through the ONE published
 * classifier and counted per verdict.
 *
 * @returns The measured frame counts and the first departure frame.
 */
function measureBaselineDrive(): BaselineDriveMeasurement {
  const classifier = createTrackSurfaceClassifier(SERVED_SEGMENTS);
  const surfaceResolver = createTrackSurfaceResolver(SERVED_SEGMENTS);
  const startPose = resolveFirstChordPose();
  let car = buildStandingCarState(
    startPose.x,
    startPose.y,
    startPose.heading,
  );
  let fineDepartureFrames = 0;
  let coarseOffRoadFrames = 0;
  let firstDepartureFrame: number | null = null;

  for (let frame = 0; frame < BASELINE_DRIVE_TICKS; frame += 1) {
    car = stepPhysics(
      car,
      BASELINE_DRIVE_INPUT,
      surfaceResolver,
      SERVED_SEGMENTS,
    );
    const classification = classifier(car);
    if (!classification.isDrivable) {
      fineDepartureFrames += 1;
      if (firstDepartureFrame === null) {
        firstDepartureFrame = frame;
      }
    }
    if (classification.surface !== 'road') {
      coarseOffRoadFrames += 1;
    }
  }

  return {
    drivenFrames: BASELINE_DRIVE_TICKS,
    fineDepartureFrames,
    coarseOffRoadFrames,
    firstDepartureFrame,
  };
}

/**
 * Builds the four PUBLISHED band poses on the served geometry by walking
 * PERPENDICULAR offsets away from the first chord's midpoint (offset 0 is
 * the road core) in {@link BAND_WALK_STEP_M} steps up to
 * {@link BAND_WALK_MAX_OFFSET_M} meters, and keeping the FIRST pose of
 * each band as the published classifier reports it.
 *
 * @returns The road core, the curb, the drivable band and the open grass
 *   poses of the served geometry.
 * @throws When the served geometry exposes no pose of one of the bands
 *   within the walked offsets.
 */
function buildPublishedBandPoses(): PublishedBandPoses {
  const classifier = createTrackSurfaceClassifier(SERVED_SEGMENTS);
  const chord = SERVED_SEGMENTS[0];
  const deltaX = chord.endX - chord.startX;
  const deltaY = chord.endY - chord.startY;
  const chordLengthM = Math.hypot(deltaX, deltaY);
  const normalX = -deltaY / chordLengthM;
  const normalY = deltaX / chordLengthM;
  const midpointX = (chord.startX + chord.endX) / 2;
  const midpointY = (chord.startY + chord.endY) / 2;

  const poseAtOffset = (offsetM: number): BandPose => {
    const lateralSign = offsetM === 0 ? 1 : Math.sign(offsetM);
    return {
      offsetM,
      x: midpointX + normalX * offsetM,
      y: midpointY + normalY * offsetM,
      heading: Math.atan2(normalY * lateralSign, normalX * lateralSign),
    };
  };

  const roadCore = poseAtOffset(0);
  let curb: BandPose | null = null;
  let drivableBand: BandPose | null = null;
  let openGrass: BandPose | null = null;
  const stepCount = Math.round(BAND_WALK_MAX_OFFSET_M / BAND_WALK_STEP_M);

  for (let stepIndex = 1; stepIndex <= stepCount; stepIndex += 1) {
    const magnitudeM = stepIndex * BAND_WALK_STEP_M;
    for (const lateralSign of [1, -1]) {
      const pose = poseAtOffset(lateralSign * magnitudeM);
      const classification = classifier(buildCarStateAtBandPose(pose));
      if (curb === null && classification.surface === 'curb') {
        curb = pose;
      }
      if (
        drivableBand === null &&
        classification.surface === 'grass' &&
        classification.isDrivable
      ) {
        drivableBand = pose;
      }
      if (openGrass === null && !classification.isDrivable) {
        openGrass = pose;
      }
    }
    if (curb !== null && drivableBand !== null && openGrass !== null) {
      break;
    }
  }

  if (curb === null || drivableBand === null || openGrass === null) {
    throw new Error(
      'the served geometry exposes no complete band ladder within ' +
        `${BAND_WALK_MAX_OFFSET_M} m of the first chord's midpoint`,
    );
  }

  return { roadCore, curb, drivableBand, openGrass };
}

/**
 * Builds a DEPARTED car state: a car that has LEFT the drivable surface on
 * the served geometry (the open grass beyond the band outer edge), rolling
 * away from the ribbon with a non-zero speed.
 *
 * @returns A fresh departed car state.
 */
function buildDepartedCarState(): PhysicsCarState {
  const openGrass = buildPublishedBandPoses().openGrass;
  return {
    ...buildCarStateAtBandPose(openGrass),
    speed: DEPARTED_CAR_SPEED_M_PER_S,
  };
}

describe('the MEASURED current departure rate (AC-107-RED-BASELINE)', () => {
  it('pins the MEASURED driven-frame counts of the served drive', () => {
    const measurement = measureBaselineDrive();
    expect(measurement.drivenFrames).toBe(MEASURED_DRIVEN_FRAMES);
    expect(measurement.fineDepartureFrames).toBe(
      MEASURED_FINE_DEPARTURE_FRAMES,
    );
    expect(measurement.coarseOffRoadFrames).toBe(
      MEASURED_COARSE_OFF_ROAD_FRAMES,
    );
  });

  it('pins the MEASURED departure rate of the served drive', () => {
    const measurement = measureBaselineDrive();
    expect(
      measurement.fineDepartureFrames / measurement.drivenFrames,
    ).toBeCloseTo(
      MEASURED_FINE_DEPARTURE_FRAMES / MEASURED_DRIVEN_FRAMES,
      12,
    );
  });

  it('pins the MEASURED first departure frame of the served drive', () => {
    const measurement = measureBaselineDrive();
    expect(measurement.firstDepartureFrame).toBe(
      MEASURED_FIRST_DEPARTURE_FRAME,
    );
  });
});

describe('the no-recovery law (RED-S28-2)', () => {
  it('starts the DEPARTED fixture genuinely OFF the drivable surface', () => {
    const classifier = createTrackSurfaceClassifier(SERVED_SEGMENTS);
    expect(classifier(buildDepartedCarState()).isDrivable).toBe(false);
  });

  it('NEVER reseats, NEVER teleports and NEVER auto-resets a departed car', () => {
    const departed = buildDepartedCarState();
    const surfaceResolver = createTrackSurfaceResolver(SERVED_SEGMENTS);
    const classifier = createTrackSurfaceClassifier(SERVED_SEGMENTS);
    const tickedWithRetiredSeam = stepPhysics(
      departed,
      BASELINE_DRIVE_INPUT,
      surfaceResolver,
      SERVED_SEGMENTS,
    );
    const tickedWithoutRetiredSeam = stepPhysics(
      departed,
      BASELINE_DRIVE_INPUT,
      surfaceResolver,
    );
    expect(tickedWithRetiredSeam).toEqual(tickedWithoutRetiredSeam);
    expect(classifier(tickedWithRetiredSeam).isDrivable).toBe(false);
    expect(tickedWithRetiredSeam.tick).toBe(departed.tick + 1);
  });
});

describe('the penalty regression guard (RED-S28-3)', () => {
  it('reads the COARSE off-track verdict as FALSE on the road core and TRUE on the curb', () => {
    const poses = buildPublishedBandPoses();
    const surfaceResolver = createTrackSurfaceResolver(SERVED_SEGMENTS);
    expect(
      surfaceResolver(buildCarStateAtBandPose(poses.roadCore)) !== 'road',
    ).toBe(false);
    expect(
      surfaceResolver(buildCarStateAtBandPose(poses.curb)) !== 'road',
    ).toBe(true);
  });

  it('reads the COARSE off-track verdict as TRUE on the drivable band and on the open grass', () => {
    const poses = buildPublishedBandPoses();
    const surfaceResolver = createTrackSurfaceResolver(SERVED_SEGMENTS);
    expect(
      surfaceResolver(buildCarStateAtBandPose(poses.drivableBand)) !== 'road',
    ).toBe(true);
    expect(
      surfaceResolver(buildCarStateAtBandPose(poses.openGrass)) !== 'road',
    ).toBe(true);
  });

  it('reads the FINE drivable verdict as TRUE on the road core, the curb and the drivable band', () => {
    const poses = buildPublishedBandPoses();
    const classifier = createTrackSurfaceClassifier(SERVED_SEGMENTS);
    expect(
      classifier(buildCarStateAtBandPose(poses.roadCore)).isDrivable,
    ).toBe(true);
    expect(
      classifier(buildCarStateAtBandPose(poses.curb)).isDrivable,
    ).toBe(true);
    expect(
      classifier(buildCarStateAtBandPose(poses.drivableBand)).isDrivable,
    ).toBe(true);
  });

  it('reads the FINE drivable verdict as FALSE only beyond the drivable band', () => {
    const poses = buildPublishedBandPoses();
    const classifier = createTrackSurfaceClassifier(SERVED_SEGMENTS);
    expect(
      classifier(buildCarStateAtBandPose(poses.openGrass)).isDrivable,
    ).toBe(false);
  });
});
