/**
 * @jest-environment jsdom
 */

/**
 * RED-phase contract tests for the Neat Racing chase camera (slice
 * P1-drift-rewire-red; Step 14P; the RED-3 .. RED-7 contracts).
 *
 * The re-wiring wave points the chase camera at the PUBLISHED travel
 * direction instead of the car NOSE, so the 3D view slides with the car
 * instead of freezing on the nose line. These contracts are the RED home of
 * that behaviour; the GREEN home is the new module
 * `examples/neatRacing/browser-entry/chase-camera.ts` plus the three live
 * call sites in `browser-entry.ts` (the camera placement, the look target
 * and the starfield yaw).
 *
 * RED FORM (the 14N import-fails form): the not-yet-existing module is
 * imported BY NAME, so the suite fails to compile with
 * `TS2307: Cannot find module './chase-camera'`. That IS the red - the
 * contract module does not exist yet, and NO placeholder export is authored
 * here to make the import resolve.
 *
 * THE PINNED CONTRACT API (the plan's `red_contract_api`, section 2 - the
 * green slice must export these names VERBATIM):
 *
 * - `NEATRACING_CHASE_CAMERA_FOLLOW_DT_S` - the named tick dt (1/60 s);
 * - `NEATRACING_CHASE_CAMERA_BACK` (5.5), `NEATRACING_CHASE_CAMERA_HEIGHT`
 *   (2.4) and `NEATRACING_CHASE_CAMERA_LOOKAHEAD` (8) - the chase geometry,
 *   moved out of the browser entry UNCHANGED;
 * - `createChaseCameraFollowState()` - the follow state, EXACTLY two fields
 *   (`offsetRad`, `lastTick`);
 * - `resolveChaseCameraAimRad(headingRad, followOffsetRad)` - the ONE aim
 *   law: the nose heading plus the follow offset;
 * - `stepChaseCameraFollow(follow, step)` - the follow step, where `step` is
 *   `{ ticks, headingRad, travelHeadingRad, speedMps, tick }`;
 * - `resolveChaseCameraBasis(carX, carY, aimRad)` - the camera basis
 *   (`cameraWorld`, `lookTarget`, `forwardVector`).
 *
 * THE MEASURED CURRENT-MODEL NUMBERS (the plan's `red_contract_api`,
 * section 4 - seed 42, road, dt 1/60, full throttle, full LEFT steer, a
 * 200-tick hold) are the fixtures below. The drift engages at tick 52; at
 * tick 61 the car runs at 18.04974 m/s with a 0.203293 rad nose heading, a
 * -0.34897859 rad PUBLISHED slip (the TRAVEL DIVERGENCE) and a -0.45 rad
 * committed slip. The CURRENT camera is a pure function of the NOSE, so the
 * 0.34897859 rad of off-axis travel produces EXACTLY ZERO screen-space
 * translation today - the frozen-camera defect these contracts close.
 *
 * The fixtures are deterministic: no RNG, no timers, no wall-clock reads.
 */

import { describe, expect, it } from '@jest/globals';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

import {
  NEATRACING_CHASE_CAMERA_BACK,
  NEATRACING_CHASE_CAMERA_FOLLOW_DT_S,
  NEATRACING_CHASE_CAMERA_HEIGHT,
  NEATRACING_CHASE_CAMERA_LOOKAHEAD,
  createChaseCameraFollowState,
  resolveChaseCameraAimRad,
  resolveChaseCameraBasis,
  stepChaseCameraFollow,
} from './chase-camera';
// The NAMESPACE import (the drift-camera RED slice): the three names the
// GREEN slice will publish do NOT exist yet, and a NAMED import of a
// missing export breaks the COMPILE (ts-jest diagnostics are ON). The
// namespace import lets the contracts read those names at RUNTIME, so a
// missing export fails as a legible ASSERTION and never as an import
// failure.
import * as chaseCameraModule from './chase-camera';
import {
  NEATRACING_CAR_SCENE_YAW_OFFSET_RAD,
  carSceneYawRad,
} from './browser-entry';
import { NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS } from '../physics/physics.step.service';

/** The float64 precision used by the camera-math assertions (repo convention). */
const CAMERA_PRECISION_DIGITS = 12;

/**
 * The MEASURED tick-61 nose heading (rad) of the full-LEFT 200-tick hold
 * (seed 42, road, dt 1/60, full throttle, steer +1).
 */
const MEASURED_NOSE_HEADING_RAD = 0.203293;

/**
 * The MEASURED tick-61 PUBLISHED slip (rad) - the TRAVEL DIVERGENCE between
 * the nose heading and the velocity heading.
 */
const MEASURED_PUBLISHED_SLIP_RAD = -0.34897859;

/**
 * The MEASURED tick-61 speed (m/s) of the same hold.
 */
const MEASURED_SPEED_MPS = 18.04974;

/**
 * The MEASURED tick-61 travel heading (rad): the velocity heading the
 * physics already computes and then discards (the nose heading plus the
 * published slip).
 */
const MEASURED_TRAVEL_HEADING_RAD =
  MEASURED_NOSE_HEADING_RAD + MEASURED_PUBLISHED_SLIP_RAD;

/** The deterministic car position (m) the basis contracts are read at. */
const CAR_X = 10;
const CAR_Y = -4;

/** The source of the browser entry (the three live call sites). */
const BROWSER_ENTRY_PATH = resolve(
  process.cwd(),
  'examples',
  'neatRacing',
  'browser-entry',
  'browser-entry.ts',
);

/**
 * A chase-camera basis as the test reads it. The test keeps a structural
 * view so the assertions never depend on the exported type declaration.
 */
interface ChaseCameraBasisView {
  /** The camera eye in scene space. */
  cameraWorld: readonly number[];
  /** The look target in scene space. */
  lookTarget: readonly number[];
  /** The normalized camera forward in scene space. */
  forwardVector: readonly number[];
}

/**
 * The step descriptor the follow consumes. The plan pins its FIVE fields
 * (not the production type name), so the test keeps a local view.
 */
interface ChaseCameraFollowStepView {
  /** The number of physics ticks this frame advances the follow by. */
  ticks: number;
  /** The car nose heading (rad). */
  headingRad: number;
  /** The published travel heading (rad). */
  travelHeadingRad: number;
  /** The car speed (m/s). */
  speedMps: number;
  /** The physics tick index this step belongs to. */
  tick: number;
}

/**
 * Wraps an angle into the `[-pi, pi)` band.
 *
 * @param angleRad - The angle to wrap (rad).
 * @returns The wrapped angle (rad).
 */
function wrapRad(angleRad: number): number {
  const twoPi = 2 * Math.PI;
  return ((((angleRad + Math.PI) % twoPi) + twoPi) % twoPi) - Math.PI;
}

/**
 * The SIGNED SHORTEST-PATH delta from one angle to another - the law the
 * follow uses so a heading crossing plus or minus pi never spins the camera
 * by 2 pi.
 *
 * @param fromRad - The angle travelled from (rad).
 * @param toRad - The angle travelled to (rad).
 * @returns The shortest signed delta (rad).
 */
function signedShortestDeltaRad(fromRad: number, toRad: number): number {
  return wrapRad(toRad - fromRad);
}

/**
 * The top-down bearing (rad) of a scene-space direction or offset. The scene
 * mirrors the top-down y into z (`worldZ = -y`), so the scene forward of a
 * heading h is `(cos h, 0, -sin h)` and its top-down bearing is
 * `atan2(-z, x)`.
 *
 * @param vector - The scene-space vector (x, y, z).
 * @returns The top-down bearing (rad).
 */
function topDownBearingRad(vector: readonly number[]): number {
  return Math.atan2(-vector[2], vector[0]);
}

/**
 * The top-down bearing (rad) of a scene-space point around the car.
 *
 * @param point - The scene-space point (x, y, z).
 * @param carX - The car x coordinate (m).
 * @param carY - The car top-down y coordinate (m).
 * @returns The top-down bearing of the point around the car (rad).
 */
function bearingAroundCarRad(
  point: readonly number[],
  carX: number,
  carY: number,
): number {
  return topDownBearingRad([point[0] - carX, 0, point[2] + carY]);
}

/**
 * The PRE-WAVE chase-camera basis: the exact basis the browser entry built
 * from the NOSE heading before the re-wiring, rebuilt here from the SAME
 * named constants so the straight-line identity is asserted against a
 * re-derived expectation instead of against the implementation under test.
 *
 * @param carX - The car x coordinate (m).
 * @param carY - The car top-down y coordinate (m).
 * @param headingRad - The nose heading (rad).
 * @returns The pre-wave basis.
 */
function preWaveBasis(
  carX: number,
  carY: number,
  headingRad: number,
): ChaseCameraBasisView {
  const forwardX = Math.cos(headingRad);
  const sceneForwardZ = -Math.sin(headingRad);
  const carSceneZ = -carY;
  const cameraWorld = [
    carX - forwardX * NEATRACING_CHASE_CAMERA_BACK,
    NEATRACING_CHASE_CAMERA_HEIGHT,
    carSceneZ - sceneForwardZ * NEATRACING_CHASE_CAMERA_BACK,
  ];
  const lookTarget = [
    carX + forwardX * NEATRACING_CHASE_CAMERA_LOOKAHEAD,
    NEATRACING_CHASE_CAMERA_HEIGHT / 2,
    carSceneZ + sceneForwardZ * NEATRACING_CHASE_CAMERA_LOOKAHEAD,
  ];
  const toTargetX = lookTarget[0] - cameraWorld[0];
  const toTargetY = lookTarget[1] - cameraWorld[1];
  const toTargetZ = lookTarget[2] - cameraWorld[2];
  const distance = Math.hypot(toTargetX, toTargetY, toTargetZ) || 1;
  return {
    cameraWorld,
    lookTarget,
    forwardVector: [
      toTargetX / distance,
      toTargetY / distance,
      toTargetZ / distance,
    ],
  };
}

/**
 * Asserts a scene-space vector matches an expected vector element by
 * element at float64 precision.
 *
 * @param actual - The vector read from the implementation.
 * @param expected - The expected vector.
 */
function expectVectorCloseTo(
  actual: readonly number[],
  expected: readonly number[],
): void {
  expect(actual).toHaveLength(expected.length);
  expected.forEach((entry, index) => {
    expect(actual[index]).toBeCloseTo(entry, CAMERA_PRECISION_DIGITS);
  });
}

/**
 * Asserts a chase-camera basis matches an expected basis at float64
 * precision (the repo's camera-math convention).
 *
 * @param actual - The basis read from the implementation.
 * @param expected - The expected basis.
 */
function expectBasisCloseTo(
  actual: ChaseCameraBasisView,
  expected: ChaseCameraBasisView,
): void {
  expectVectorCloseTo(actual.cameraWorld, expected.cameraWorld);
  expectVectorCloseTo(actual.lookTarget, expected.lookTarget);
  expectVectorCloseTo(actual.forwardVector, expected.forwardVector);
}

/**
 * Builds a follow step descriptor for the measured tick-61 hold.
 *
 * @param ticks - The number of physics ticks the step advances by.
 * @param tick - The physics tick index of the step.
 * @param speedMps - The car speed (m/s).
 * @returns The step descriptor.
 */
function measuredStep(
  ticks: number,
  tick: number,
  speedMps: number = MEASURED_SPEED_MPS,
): ChaseCameraFollowStepView {
  return {
    ticks,
    headingRad: MEASURED_NOSE_HEADING_RAD,
    travelHeadingRad: MEASURED_TRAVEL_HEADING_RAD,
    speedMps,
    tick,
  };
}

/**
 * The source of the chase-camera LAW module: the file that will carry the
 * three named exports the GREEN slice publishes (the cap, the follow gain
 * and the release smooth fraction) and the JSDoc that states their
 * measured basis.
 */
const CHASE_CAMERA_PATH = resolve(
  process.cwd(),
  'examples',
  'neatRacing',
  'browser-entry',
  'chase-camera.ts',
);

/**
 * The absolute tolerance (rad) of the camera RELATIONAL comparisons (the
 * cap, the response and the release offsets).
 */
const CAMERA_RAD_TOLERANCE = 1e-9;

/**
 * The absolute tolerance (fraction) of the camera RELEASE-FRACTION
 * comparisons.
 */
const CAMERA_FRACTION_TOLERANCE = 1e-9;

/**
 * The MEASURED peak |offset| (rad) of a 1200-tick hold of the measured
 * tick-61 slide under the PRE-WAVE law: the worst yaw aim offset the
 * current camera law allows.
 */
const MEASURED_PRE_WAVE_PEAK_OFFSET_RAD = 0.34897858999999976;

/**
 * The RECORDED cap (rad) of the yaw aim offset: EXACTLY HALF of the
 * measured pre-wave peak (the half is USER-NAMED).
 */
const RECORDED_MAX_AIM_OFFSET_RAD = 0.17448929499999988;

/**
 * The MEASURED pre-wave follow gain: the module-private gain the current
 * law uses (the GREEN slice promotes it to a NAMED export).
 */
const MEASURED_PRE_WAVE_FOLLOW_GAIN = 1;

/**
 * The MEASURED one-tick response (rad) from the neutral aim under the
 * pre-wave law.
 */
const MEASURED_PRE_WAVE_FIRST_TICK_OFFSET_RAD = 0.0366367775241712;

/**
 * The MEASURED number of ticks the pre-wave camera needs to first reach
 * 95% of the recorded cap.
 */
const MEASURED_PRE_WAVE_TICKS_TO_CAPPED_OFFSET = 6;

/**
 * The MEASURED one-tick release close fraction at the PEAK offset: the
 * pre-wave release closes 10.498% of the offset per tick, so it does NOT
 * close in a single tick.
 */
const MEASURED_PRE_WAVE_PEAK_RELEASE_CLOSE_FRACTION = 0.10498288025082125;

/**
 * The MEASURED one-tick release close fraction when the release starts
 * from the tick-2 offset (0.06942732062073578 rad). It is SMALLER than the
 * peak fraction because the pre-wave release rate is multiplied by the
 * camera's OWN offset (the snap-back enabler).
 */
const MEASURED_PRE_WAVE_EARLY_RELEASE_CLOSE_FRACTION = 0.020885751435015254;

/**
 * The RECORDED smooth release fraction: the RED-DECIDED one-tick close
 * fraction of the release (strictly below the measured 0.10498 peak close
 * fraction).
 */
const RECORDED_RELEASE_SMOOTH_FRACTION = 0.05;

/**
 * The RECORDED cap-reach fraction: the threshold (95% of the cap) used for
 * the time-to-cap comparison.
 */
const RECORDED_CAP_REACH_FRACTION = 0.95;

/**
 * The RECORDED probe length (ticks) of the measured hold: the
 * deterministic length of the hold the peak is measured over.
 */
const RECORDED_PROBE_TICK_COUNT = 1200;

/**
 * The observation of ONE release tick applied to a held measured slide.
 */
interface MeasuredReleaseObservation {
  /** The follow offset (rad) BEFORE the release tick. */
  offsetBeforeRad: number;
  /** The follow offset (rad) AFTER the release tick. */
  offsetAfterRad: number;
  /** The fraction of the offset the release tick closed. */
  closeFraction: number;
}

/**
 * The trace of a measured hold followed by a run of release ticks.
 */
interface MeasuredReleaseTrace {
  /**
   * The offsets (rad) of the release trace: the hold's last offset first,
   * then the offset after every release tick.
   */
  offsetsRad: number[];
  /** Whether any |offset| grew against its predecessor. */
  grew: boolean;
  /** Whether the offset ever changed sign. */
  flipped: boolean;
}

/**
 * Reads a numeric export of the chase-camera module at RUNTIME.
 *
 * WHY: the three names the GREEN slice will publish do NOT exist yet, and
 * a NAMED import of a missing export breaks the COMPILE. A missing name
 * must therefore fail as a legible RUNTIME assertion, so this helper reads
 * the namespace and answers `NaN` for an absent name.
 *
 * @param name - The export name to read.
 * @returns The exported number, or `NaN` when the name is absent.
 */
function readNumericExport(name: string): number {
  const value = (chaseCameraModule as unknown as Record<string, unknown>)[
    name
  ];
  return typeof value === 'number' && Number.isFinite(value)
    ? value
    : Number.NaN;
}

/**
 * Extracts the JSDoc block that documents ONE named export.
 *
 * WHY: the JSDoc guard must bind the "MEASURED" note to the JSDoc of
 * THAT export - a loose source regex would also be satisfied by a JSDoc
 * on an EARLIER export, which is weaker than the AC ("a derived JSDoc").
 *
 * @param source - The source text of the module.
 * @param exportName - The name of the exported constant.
 * @returns The JSDoc block immediately above the export, or an empty
 *   string when the export or its JSDoc is absent.
 */
function jsDocAbove(source: string, exportName: string): string {
  const exportIndex = source.indexOf(`export const ${exportName}`);
  if (exportIndex < 0) {
    return '';
  }
  const before = source.slice(0, exportIndex);
  const start = before.lastIndexOf('/**');
  const end = before.lastIndexOf('*/');
  if (start < 0 || end < start) {
    return '';
  }
  return before.slice(start, end + 2);
}

/**
 * Runs the MEASURED tick-61 slide as a hold probe and records the follow
 * offset after EVERY tick (ONE tick per step).
 *
 * @param tickCount - The number of ticks the probe holds.
 * @returns The offset (rad) after each tick, in tick order.
 */
function runMeasuredHoldProbe(tickCount: number): number[] {
  const offsetsRad: number[] = [];
  let follow = createChaseCameraFollowState();
  for (let tick = 1; tick <= tickCount; tick += 1) {
    follow = stepChaseCameraFollow(follow, measuredStep(1, tick));
    offsetsRad.push(follow.offsetRad);
  }
  return offsetsRad;
}

/**
 * The largest |offset| of a probe trace.
 *
 * @param offsets - The probe trace.
 * @returns The peak |offset| (rad), or 0 for an empty trace.
 */
function peakAbsOffsetRad(offsets: readonly number[]): number {
  let peakRad = 0;
  for (const offsetRad of offsets) {
    const absRad = Math.abs(offsetRad);
    if (absRad > peakRad) {
      peakRad = absRad;
    }
  }
  return peakRad;
}

/**
 * The 1-based tick index at which a probe trace first reaches a target
 * |offset|.
 *
 * @param offsets - The probe trace.
 * @param targetRad - The target |offset| (rad).
 * @returns The 1-based tick index, or `Infinity` when the trace never
 *   reaches the target.
 */
function ticksToReachOffsetRad(
  offsets: readonly number[],
  targetRad: number,
): number {
  for (let index = 0; index < offsets.length; index += 1) {
    if (Math.abs(offsets[index]) >= targetRad) {
      return index + 1;
    }
  }
  return Number.POSITIVE_INFINITY;
}

/**
 * Holds the measured slide for a number of ticks and then applies ONE
 * release tick (the travel heading falls back onto the nose heading).
 *
 * @param holdTicks - The number of measured ticks to hold.
 * @returns The offset before and after the release tick (rad) and the
 *   one-tick close fraction.
 */
function releaseFromMeasuredHold(
  holdTicks: number,
): MeasuredReleaseObservation {
  let follow = createChaseCameraFollowState();
  for (let tick = 1; tick <= holdTicks; tick += 1) {
    follow = stepChaseCameraFollow(follow, measuredStep(1, tick));
  }
  const offsetBeforeRad = follow.offsetRad;
  const released = stepChaseCameraFollow(follow, {
    ...measuredStep(1, holdTicks + 1),
    travelHeadingRad: MEASURED_NOSE_HEADING_RAD,
  });
  const offsetAfterRad = released.offsetRad;
  const closeFraction = 1 - Math.abs(offsetAfterRad) / Math.abs(offsetBeforeRad);
  return { offsetBeforeRad, offsetAfterRad, closeFraction };
}

/**
 * Holds the measured slide and then runs a number of release ticks (the
 * tick index keeps running), recording the offsets of the release.
 *
 * @param holdTicks - The number of measured ticks to hold.
 * @param releaseTicks - The number of release ticks to run.
 * @returns The release offsets (rad), whether any |offset| grew and whether
 *   the offset ever changed sign.
 */
function measuredReleaseTrace(
  holdTicks: number,
  releaseTicks: number,
): MeasuredReleaseTrace {
  let follow = createChaseCameraFollowState();
  for (let tick = 1; tick <= holdTicks; tick += 1) {
    follow = stepChaseCameraFollow(follow, measuredStep(1, tick));
  }
  const offsetsRad: number[] = [follow.offsetRad];
  let grew = false;
  let flipped = false;
  let previousAbsRad = Math.abs(follow.offsetRad);
  let previousSign = Math.sign(follow.offsetRad);
  for (let index = 0; index < releaseTicks; index += 1) {
    follow = stepChaseCameraFollow(follow, {
      ...measuredStep(1, holdTicks + index + 1),
      travelHeadingRad: MEASURED_NOSE_HEADING_RAD,
    });
    offsetsRad.push(follow.offsetRad);
    const absRad = Math.abs(follow.offsetRad);
    if (absRad > previousAbsRad) {
      grew = true;
    }
    if (Math.sign(follow.offsetRad) !== previousSign) {
      flipped = true;
    }
    previousAbsRad = absRad;
    previousSign = Math.sign(follow.offsetRad);
  }
  return { offsetsRad, grew, flipped };
}

describe('chase camera (P1-drift-rewire-red; Step 14P; RED-3 .. RED-7)', () => {
  describe('RED-3 - ONE aim law (the camera placement, the look target and the starfield yaw)', () => {
    // [RED-3-AIM-LAW]
    it('resolves the camera aim as the nose heading plus the follow offset', () => {
      // Arrange: the measured tick-61 nose heading and follow offset.
      const noseHeadingRad = MEASURED_NOSE_HEADING_RAD;
      const followOffsetRad = MEASURED_PUBLISHED_SLIP_RAD;
      // Act
      const aimRad = resolveChaseCameraAimRad(noseHeadingRad, followOffsetRad);
      // Assert
      expect(aimRad).toBeCloseTo(
        noseHeadingRad + followOffsetRad,
        CAMERA_PRECISION_DIGITS,
      );
    });

    // [RED-3-STARFIELD-YAW]
    it('keeps the starfield yaw on the same aim with the retained pi/2 scene offset', () => {
      // Arrange
      const aimRad = resolveChaseCameraAimRad(
        MEASURED_NOSE_HEADING_RAD,
        MEASURED_PUBLISHED_SLIP_RAD,
      );
      // Act
      const starfieldYawRad = carSceneYawRad(aimRad);
      // Assert
      expect(starfieldYawRad).toBeCloseTo(
        NEATRACING_CAR_SCENE_YAW_OFFSET_RAD + aimRad,
        CAMERA_PRECISION_DIGITS,
      );
    });

    // [RED-3-SCENE-OFFSET]
    it('retains the pi/2 scene yaw offset', () => {
      // Act
      const sceneYawOffsetRad = NEATRACING_CAR_SCENE_YAW_OFFSET_RAD;
      // Assert
      expect(sceneYawOffsetRad).toBe(Math.PI / 2);
    });

    // [RED-3-BASIS-IS-THE-AIM]
    it('resolves the camera basis from the aim alone (the one law every call site reads)', () => {
      // Arrange: two (nose, offset) pairs that resolve to the SAME aim.
      const aimRad = 0.2;
      // Act
      const first = resolveChaseCameraBasis(
        CAR_X,
        CAR_Y,
        resolveChaseCameraAimRad(0.4, -0.2),
      );
      const second = resolveChaseCameraBasis(
        CAR_X,
        CAR_Y,
        resolveChaseCameraAimRad(0.35, -0.15),
      );
      // Assert
      expect(first.cameraWorld).toHaveLength(3);
      expect(aimRad).toBeCloseTo(0.2, CAMERA_PRECISION_DIGITS);
      expectBasisCloseTo(first, second);
    });

    // [RED-3-CALL-SITES]
    it('has the browser entry read the shared chase-camera module (the one aim law)', () => {
      // Act
      const browserEntrySource = readFileSync(BROWSER_ENTRY_PATH, 'utf-8');
      // Assert: the three live call sites (the camera placement, the look
      // target and the starfield yaw) can only agree if they read the ONE
      // shared law, so the browser entry must import the shared module.
      expect(browserEntrySource).toMatch(/from\s+'\.\/chase-camera(\.js)?'/);
    });
  });

  describe('RED-4 - the 3D view moves (the frozen-camera trap)', () => {
    // [RED-4-GEOMETRY-CONSTANTS]
    it('keeps the pre-wave chase-camera geometry constants', () => {
      // Assert
      expect(NEATRACING_CHASE_CAMERA_BACK).toBe(5.5);
      expect(NEATRACING_CHASE_CAMERA_HEIGHT).toBe(2.4);
      expect(NEATRACING_CHASE_CAMERA_LOOKAHEAD).toBe(8);
    });

    // [RED-4-STRAIGHT-LINE-IDENTITY]
    it('keeps the camera basis identical to the pre-wave basis when the published slip is zero', () => {
      // Arrange: the straight lane, where the published slip is EXACTLY 0
      // and the aim is the nose heading.
      const noseAlignedBasis = resolveChaseCameraBasis(
        CAR_X,
        CAR_Y,
        MEASURED_NOSE_HEADING_RAD,
      );
      // Act
      const preWave = preWaveBasis(CAR_X, CAR_Y, MEASURED_NOSE_HEADING_RAD);
      // Assert
      expectBasisCloseTo(noseAlignedBasis, preWave);
    });

    // [RED-4-CAMERA-BEARING]
    it('turns the camera bearing by the travel divergence under a published slip', () => {
      // Arrange
      const noseBasis = resolveChaseCameraBasis(
        CAR_X,
        CAR_Y,
        MEASURED_NOSE_HEADING_RAD,
      );
      const travelBasis = resolveChaseCameraBasis(
        CAR_X,
        CAR_Y,
        MEASURED_TRAVEL_HEADING_RAD,
      );
      // Act
      const noseBearingRad = bearingAroundCarRad(
        noseBasis.cameraWorld,
        CAR_X,
        CAR_Y,
      );
      const travelBearingRad = bearingAroundCarRad(
        travelBasis.cameraWorld,
        CAR_X,
        CAR_Y,
      );
      // Assert
      expect(signedShortestDeltaRad(noseBearingRad, travelBearingRad)).toBeCloseTo(
        MEASURED_PUBLISHED_SLIP_RAD,
        CAMERA_PRECISION_DIGITS,
      );
    });

    // [RED-4-LOOK-TARGET]
    it('turns the look target off the nose line by the travel divergence', () => {
      // Arrange
      const noseBasis = resolveChaseCameraBasis(
        CAR_X,
        CAR_Y,
        MEASURED_NOSE_HEADING_RAD,
      );
      const travelBasis = resolveChaseCameraBasis(
        CAR_X,
        CAR_Y,
        MEASURED_TRAVEL_HEADING_RAD,
      );
      // Act
      const noseTargetBearingRad = bearingAroundCarRad(
        noseBasis.lookTarget,
        CAR_X,
        CAR_Y,
      );
      const travelTargetBearingRad = bearingAroundCarRad(
        travelBasis.lookTarget,
        CAR_X,
        CAR_Y,
      );
      // Assert
      expect(
        signedShortestDeltaRad(noseTargetBearingRad, travelTargetBearingRad),
      ).toBeCloseTo(MEASURED_PUBLISHED_SLIP_RAD, CAMERA_PRECISION_DIGITS);
    });

    // [RED-4-FORWARD-VECTOR]
    it('turns the camera forward vector by the travel divergence', () => {
      // Arrange
      const noseBasis = resolveChaseCameraBasis(
        CAR_X,
        CAR_Y,
        MEASURED_NOSE_HEADING_RAD,
      );
      const travelBasis = resolveChaseCameraBasis(
        CAR_X,
        CAR_Y,
        MEASURED_TRAVEL_HEADING_RAD,
      );
      // Act
      const noseForwardBearingRad = topDownBearingRad(noseBasis.forwardVector);
      const travelForwardBearingRad = topDownBearingRad(
        travelBasis.forwardVector,
      );
      // Assert
      expect(
        signedShortestDeltaRad(noseForwardBearingRad, travelForwardBearingRad),
      ).toBeCloseTo(MEASURED_PUBLISHED_SLIP_RAD, CAMERA_PRECISION_DIGITS);
    });
  });

  describe('RED-5 - the PROPORTIONAL follow law (S-1 / S-4, not a fixed tau)', () => {
    // [RED-5-SPEED-PROPORTIONAL]
    it('follows proportionally to the speed at the same slide', () => {
      // Arrange: the same slide at the measured speed and at double the
      // speed.
      const slow = stepChaseCameraFollow(
        createChaseCameraFollowState(),
        measuredStep(1, 1, MEASURED_SPEED_MPS),
      );
      const fast = stepChaseCameraFollow(
        createChaseCameraFollowState(),
        measuredStep(1, 1, MEASURED_SPEED_MPS * 2),
      );
      // Act
      const followRateRatio = Math.abs(fast.offsetRad) / Math.abs(slow.offsetRad);
      // Assert: a fixed time constant would leave the ratio at 1.0.
      expect(followRateRatio).toBeCloseTo(2, 1);
    });

    // [RED-5-NO-SNAP]
    it('moves the aim by a bounded step per tick (no snap)', () => {
      // Arrange
      const stepped = stepChaseCameraFollow(
        createChaseCameraFollowState(),
        measuredStep(1, 1),
      );
      // Act
      const offsetAfterOneTickRad = Math.abs(stepped.offsetRad);
      // Assert: one tick must not reach the whole travel divergence.
      expect(offsetAfterOneTickRad).toBeLessThan(
        Math.abs(MEASURED_PUBLISHED_SLIP_RAD),
      );
    });

    // [RED-5-MONOTONIC-RELEASE]
    it('releases toward the nose-aligned aim monotonically without overshoot', () => {
      // Arrange: a follow that already carries a slide offset, then a
      // release (the travel heading falls back onto the nose heading).
      const slid = stepChaseCameraFollow(
        createChaseCameraFollowState(),
        measuredStep(30, 30),
      );
      const releaseStep: ChaseCameraFollowStepView = {
        ...measuredStep(1, 31),
        travelHeadingRad: MEASURED_NOSE_HEADING_RAD,
      };
      // Act
      const released = stepChaseCameraFollow(slid, releaseStep);
      const releasedAgain = stepChaseCameraFollow(released, {
        ...releaseStep,
        tick: 32,
      });
      // Assert
      expect(Math.abs(released.offsetRad)).toBeLessThan(
        Math.abs(slid.offsetRad),
      );
      expect(Math.abs(releasedAgain.offsetRad)).toBeLessThan(
        Math.abs(released.offsetRad),
      );
      expect(Math.sign(releasedAgain.offsetRad)).toBe(Math.sign(slid.offsetRad));
    });
  });

  describe('RED-6 - the dt is pinned (frame-rate independence)', () => {
    // [RED-6-DT-SOURCE]
    it('steps the follow with the named physics tick dt', () => {
      // Assert: the dt is NAMED, not inlined, and it is the physics tick.
      expect(NEATRACING_CHASE_CAMERA_FOLLOW_DT_S).toBe(
        NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
      );
    });

    // [RED-6-FRAME-RATE-INDEPENDENCE]
    it('produces the identical aim for sixty ticks delivered as one step or as sixty steps', () => {
      // Arrange
      const singleStep = stepChaseCameraFollow(
        createChaseCameraFollowState(),
        measuredStep(60, 60),
      );
      // Act
      let stepped = createChaseCameraFollowState();
      for (let tick = 1; tick <= 60; tick += 1) {
        stepped = stepChaseCameraFollow(stepped, measuredStep(1, tick));
      }
      // Assert
      expect(stepped.offsetRad).toBeCloseTo(
        singleStep.offsetRad,
        CAMERA_PRECISION_DIGITS,
      );
    });

    // [RED-6-ZERO-STEP]
    it('holds the aim on a zero-step frame', () => {
      // Arrange
      const slid = stepChaseCameraFollow(
        createChaseCameraFollowState(),
        measuredStep(30, 30),
      );
      // Act
      const held = stepChaseCameraFollow(slid, measuredStep(0, 31));
      // Assert
      expect(held.offsetRad).toBe(slid.offsetRad);
    });
  });

  describe('RED-7 - the signed wrap and the tick-regression reset', () => {
    // [RED-7-SHORTEST-PATH]
    it('follows the shortest path across the pi wrap instead of spinning by 2 pi', () => {
      // Arrange: the nose heading sits just below +pi and the travel heading
      // just above -pi - the same turn, wrapped. The raw delta is -6 rad; the
      // shortest path is +0.28318530 rad.
      const wrappedStep: ChaseCameraFollowStepView = {
        ...measuredStep(1, 1),
        headingRad: 3.0,
        travelHeadingRad: -3.0,
      };
      // Act
      const wrapped = stepChaseCameraFollow(
        createChaseCameraFollowState(),
        wrappedStep,
      );
      // Assert
      expect(wrapped.offsetRad).toBeGreaterThan(0);
      expect(Math.abs(wrapped.offsetRad)).toBeLessThan(Math.PI / 2);
    });

    // [RED-7-TICK-REGRESSION]
    it('resets the follow offset to exactly zero on a tick regression (no stale offset)', () => {
      // Arrange: a follow that already carries a slide offset.
      const slid = stepChaseCameraFollow(
        createChaseCameraFollowState(),
        measuredStep(30, 30),
      );
      // Act: a new car or a mode switch rewinds the tick counter.
      const reset = stepChaseCameraFollow(slid, measuredStep(1, 1));
      // Assert
      expect(reset.offsetRad).toBe(0);
      expect(
        resolveChaseCameraAimRad(MEASURED_NOSE_HEADING_RAD, reset.offsetRad),
      ).toBeCloseTo(MEASURED_NOSE_HEADING_RAD, CAMERA_PRECISION_DIGITS);
    });

    // [RED-7-STATE-SHAPE]
    it('carries exactly the two named follow-state fields', () => {
      // Act
      const follow = createChaseCameraFollowState();
      // Assert
      expect(Object.keys(follow).sort()).toEqual(['lastTick', 'offsetRad']);
    });
  });
});

/**
 * RED-phase contract tests for the DRIFT CAMERA (slice `P1-driftcam-red`;
 * Step 14X; AC-097).
 *
 * WHAT THIS BLOCK IS: the RED contracts of the drift-camera wave. The
 * camera law (`chase-camera.ts`) currently lets the yaw aim offset run to
 * 0.34897858999999976 rad (the MEASURED pre-wave peak) and releases it
 * with a MULTIPLICATIVE snap-back enabler (the release rate is multiplied
 * by the camera's OWN offset, so the camera snaps back harder the further
 * it already sits from the nose). The GREEN slice (a later, different
 * slice) will publish a NAMED cap, a NAMED lower follow gain and a NAMED
 * smooth release fraction.
 *
 * RED FORM (the RUNTIME-fails form): the three names the GREEN slice will
 * publish do NOT exist yet, so they are read through the NAMESPACE import
 * at RUNTIME by `readNumericExport` (an absent name reads `NaN` and fails
 * the guard legibly). The contracts therefore fail at RUNTIME, NEVER at
 * import time: a NAMED import of a not-yet-existing export would break the
 * COMPILE and hide the contract behind a module-resolution error.
 *
 * THE NUMBERS: every constant in this block is MEASURED (from the pre-wave
 * law) or DECIDED (by this RED slice); none of them is carried over. The
 * RED lanes are RED BY ASSERTION against the EXISTING law.
 *
 * RED-4 IS A LABELLED PRESERVATION GUARD: it is GREEN today, it is NEVER
 * counted as a red, and it fences the SCOPE of the camera change (the AIM
 * only - the BODY presentation stays on the published pose).
 */
describe('drift camera (P1-driftcam-red; Step 14X; RED-1 .. RED-4)', () => {
  describe('RED-1 - the yaw aim offset is CAPPED at at most half of the measured pre-wave peak', () => {
    // [RED-1-RECORDED-PREMISE]
    it('records a cap that is exactly half of the measured pre-wave peak', () => {
      // Arrange: the measured pre-wave peak and the recorded cap.
      const measuredPeakRad = MEASURED_PRE_WAVE_PEAK_OFFSET_RAD;
      const recordedCapRad = RECORDED_MAX_AIM_OFFSET_RAD;
      // Act
      const doubledCapRad = recordedCapRad * 2;
      // Assert: the PREMISE of the cap (this lane is a GREEN recorded
      // premise, never a red): the peak really is above the cap and the cap
      // really is exactly half of the peak.
      expect(measuredPeakRad).toBeGreaterThan(recordedCapRad);
      expect(doubledCapRad).toBeCloseTo(
        measuredPeakRad,
        CAMERA_PRECISION_DIGITS,
      );
    });

    // [RED-1-AIM-CAP]
    it('never lets the yaw aim offset exceed the recorded cap', () => {
      // Arrange: the measured hold probe (the aim really moves).
      const probe = runMeasuredHoldProbe(RECORDED_PROBE_TICK_COUNT);
      // Act: the peak of the trace IS the worst-case yaw aim offset.
      const peakOffsetRad = peakAbsOffsetRad(probe);
      // Assert: the GUARDS first (the probe must really move the aim and
      // there must really be something to cap), then the cap itself.
      expect(peakOffsetRad).toBeGreaterThan(0);
      expect(MEASURED_PRE_WAVE_PEAK_OFFSET_RAD).toBeGreaterThan(
        RECORDED_MAX_AIM_OFFSET_RAD,
      );
      expect(peakOffsetRad).toBeLessThanOrEqual(
        RECORDED_MAX_AIM_OFFSET_RAD + CAMERA_RAD_TOLERANCE,
      );
    });

    // [RED-1-NAMED-CAP]
    it('publishes the cap as the named export', () => {
      // Arrange: the named export is read at RUNTIME (an absent name reads
      // NaN, so the guard below fails legibly instead of crashing).
      const capRad = readNumericExport(
        'NEATRACING_CHASE_CAMERA_MAX_AIM_OFFSET_RAD',
      );
      // Act
      // Assert: the GUARD first (the export must EXIST and be a finite
      // positive number), then the named value and its relation to the
      // measured peak.
      expect(capRad).toBeGreaterThan(0);
      expect(capRad).toBeCloseTo(RECORDED_MAX_AIM_OFFSET_RAD, 8);
      expect(capRad).toBeLessThanOrEqual(
        MEASURED_PRE_WAVE_PEAK_OFFSET_RAD / 2 + CAMERA_RAD_TOLERANCE,
      );
    });

    // [RED-1-NAMED-CAP-JSDOC]
    it('documents the measured basis of the three named exports', () => {
      // Arrange: the law source (the named exports live there).
      const chaseCameraSource = readFileSync(CHASE_CAMERA_PATH, 'utf-8');
      // Act: the JSDoc that documents EACH named export is extracted
      // (the guard is ANCHORED to the export's OWN JSDoc - a loose source
      // regex would also be satisfied by a JSDoc on an EARLIER export).
      // Assert: each of the three named exports is documented by a JSDoc
      // that states the MEASURED basis of its value.
      expect(
        jsDocAbove(
          chaseCameraSource,
          'NEATRACING_CHASE_CAMERA_MAX_AIM_OFFSET_RAD',
        ),
      ).toMatch(/measur/i);
      expect(
        jsDocAbove(chaseCameraSource, 'NEATRACING_CHASE_CAMERA_FOLLOW_GAIN'),
      ).toMatch(/measur/i);
      expect(
        jsDocAbove(
          chaseCameraSource,
          'NEATRACING_CHASE_CAMERA_RELEASE_SMOOTH_FRACTION',
        ),
      ).toMatch(/measur/i);
    });
  });

  describe('RED-2 - the response is DESENSITISED and MUCH SLOWER', () => {
    // [RED-2-FOLLOW-GAIN]
    it('publishes a follow gain BELOW the measured pre-wave gain', () => {
      // Arrange: the named export is read at RUNTIME (an absent name reads
      // NaN, so the clause below fails legibly).
      const gain = readNumericExport('NEATRACING_CHASE_CAMERA_FOLLOW_GAIN');
      // Act
      // Assert: the GUARD first (the measured pre-wave gain must be a real
      // positive gain, so the comparison is meaningful), then the clause.
      expect(MEASURED_PRE_WAVE_FOLLOW_GAIN).toBeGreaterThan(0);
      expect(gain).toBeLessThan(MEASURED_PRE_WAVE_FOLLOW_GAIN);
    });

    // [RED-2-DESENSITISED-RESPONSE]
    it('moves the aim LESS on the first tick than the measured pre-wave response', () => {
      // Arrange: the neutral follow and the measured tick-61 step.
      const neutralFollow = createChaseCameraFollowState();
      // Act
      const firstTick = stepChaseCameraFollow(
        neutralFollow,
        measuredStep(1, 1),
      );
      const responseRad = Math.abs(firstTick.offsetRad);
      // Assert: the GUARDS first (the probe must really move the aim and
      // the measured pre-wave response must be a real response), then the
      // clause: the camera must react LESS to the same slide.
      expect(responseRad).toBeGreaterThan(0);
      expect(MEASURED_PRE_WAVE_FIRST_TICK_OFFSET_RAD).toBeGreaterThan(0);
      expect(responseRad).toBeLessThan(
        MEASURED_PRE_WAVE_FIRST_TICK_OFFSET_RAD,
      );
    });

    // [RED-2-SLOWER-TIME-TO-CAP]
    it('needs MORE ticks to reach the capped offset than the measured pre-wave law', () => {
      // Arrange: the measured hold probe.
      const probe = runMeasuredHoldProbe(RECORDED_PROBE_TICK_COUNT);
      // Act
      const runtimeTicks = ticksToReachOffsetRad(
        probe,
        RECORDED_CAP_REACH_FRACTION * RECORDED_MAX_AIM_OFFSET_RAD,
      );
      // Assert: the GUARDS first (the measured pre-wave tick count must be
      // real and the camera must really reach the capped offset, else there
      // is no time to compare), then the clause: the response must be MUCH
      // SLOWER.
      expect(MEASURED_PRE_WAVE_TICKS_TO_CAPPED_OFFSET).toBeGreaterThan(0);
      expect(Number.isFinite(runtimeTicks)).toBe(true);
      expect(runtimeTicks).toBeGreaterThan(
        MEASURED_PRE_WAVE_TICKS_TO_CAPPED_OFFSET,
      );
    });
  });

  describe('RED-3 - the release is MONOTONIC and SMOOTH (NO SNAP-BACK)', () => {
    // [RED-3-RECORDED-PREMISE]
    it('records the measured pre-wave release fractions as the baseline of the release clauses', () => {
      // Arrange: the RECORDED pre-wave release fractions.
      const earlyFraction = MEASURED_PRE_WAVE_EARLY_RELEASE_CLOSE_FRACTION;
      const peakFraction = MEASURED_PRE_WAVE_PEAK_RELEASE_CLOSE_FRACTION;
      // Act
      // Assert: the PREMISE (this lane is a GREEN recorded premise, never
      // a red): the recorded fractions are real and the recorded rate
      // GROWS with the camera's own offset - that growth IS the
      // multiplicative snap-back enabler the clause below deletes. The
      // lane asserts the RECORDED constants only (never the runtime
      // pre-wave behaviour, which the law change moves by design).
      expect(earlyFraction).toBeGreaterThan(0);
      expect(peakFraction).toBeGreaterThan(earlyFraction);
      expect(RECORDED_RELEASE_SMOOTH_FRACTION).toBeLessThan(peakFraction);
    });

    // [RED-3-SMOOTH-FRACTION]
    it('closes the aim by NO MORE than the recorded smooth fraction per release tick', () => {
      // Arrange: the measured hold and ONE release tick.
      const release = releaseFromMeasuredHold(240);
      // Act
      // Assert: the GUARDS first (the probe must really produce a release
      // of a real offset and a NON-ZERO close fraction - a zero fraction
      // would leave nothing to smooth), then the clause.
      expect(Math.abs(release.offsetBeforeRad)).toBeGreaterThan(0);
      expect(release.closeFraction).toBeGreaterThan(0);
      expect(release.closeFraction).toBeLessThanOrEqual(
        RECORDED_RELEASE_SMOOTH_FRACTION + CAMERA_FRACTION_TOLERANCE,
      );
    });

    // [RED-3-NAMED-SMOOTH-FRACTION]
    it('publishes the release smooth fraction as the named export', () => {
      // Arrange: the named export is read at RUNTIME (an absent name reads
      // NaN, so the guard below fails legibly).
      const fraction = readNumericExport(
        'NEATRACING_CHASE_CAMERA_RELEASE_SMOOTH_FRACTION',
      );
      // Act
      // Assert: the GUARD first (the export must EXIST and be a finite
      // positive fraction), then the named value.
      expect(fraction).toBeGreaterThan(0);
      expect(fraction).toBeCloseTo(RECORDED_RELEASE_SMOOTH_FRACTION, 8);
    });

    // [RED-3-NO-MULTIPLICATIVE-ENABLER]
    it('does NOT close the aim faster the FURTHER the camera already sits from the nose', () => {
      // Arrange: the release at the peak offset and the release at the
      // early (tick-2) offset.
      const atPeak = releaseFromMeasuredHold(240);
      const atEarly = releaseFromMeasuredHold(2);
      // Act
      // Assert: the GUARDS first (BOTH probes must REALLY release a
      // NON-ZERO fraction - a zero fraction would leave the comparison
      // vacuous), then the clause: the release rate must NOT grow with
      // the camera's own offset (the multiplicative enabler is deleted at
      // its source, so the peak offset may never close faster than the
      // early one).
      expect(atPeak.closeFraction).toBeGreaterThan(0);
      expect(atEarly.closeFraction).toBeGreaterThan(0);
      expect(atPeak.closeFraction).toBeLessThanOrEqual(
        atEarly.closeFraction + CAMERA_FRACTION_TOLERANCE,
      );
    });

    // [RED-3-MONOTONIC-RELEASE]
    it('approaches the nose-aligned aim monotonically without overshoot (GREEN RAIL)', () => {
      // Arrange: the measured hold and the release trace.
      const trace = measuredReleaseTrace(240, 240);
      // Act
      // Assert: this lane is GREEN today and is a RAIL, NEVER a red: the
      // release must never grow the offset and never flip its sign.
      expect(trace.grew).toBe(false);
      expect(trace.flipped).toBe(false);
      expect(
        Math.abs(trace.offsetsRad[trace.offsetsRad.length - 1]),
      ).toBeLessThan(Math.abs(trace.offsetsRad[0]));
    });
  });

  describe('RED-4 - LABELLED PRESERVATION GUARD: the camera change touches the AIM ONLY (this lane is GREEN today and is NEVER counted as a red)', () => {
    // [RED-4-STATE-SHAPE]
    it('keeps exactly the two named follow-state fields', () => {
      // Act
      const follow = createChaseCameraFollowState();
      // Assert
      expect(Object.keys(follow).sort()).toEqual(['lastTick', 'offsetRad']);
    });

    // [RED-4-PUBLISHED-API]
    it('keeps the four published numeric constants', () => {
      // Act
      const publishedTypes = [
        typeof NEATRACING_CHASE_CAMERA_FOLLOW_DT_S,
        typeof NEATRACING_CHASE_CAMERA_BACK,
        typeof NEATRACING_CHASE_CAMERA_HEIGHT,
        typeof NEATRACING_CHASE_CAMERA_LOOKAHEAD,
      ];
      // Assert
      expect(publishedTypes).toEqual([
        'number',
        'number',
        'number',
        'number',
      ]);
    });

    // [RED-4-PUBLISHED-FUNCTIONS]
    it('keeps the four published functions', () => {
      // Act
      const publishedTypes = [
        typeof createChaseCameraFollowState,
        typeof resolveChaseCameraAimRad,
        typeof resolveChaseCameraBasis,
        typeof stepChaseCameraFollow,
      ];
      // Assert
      expect(publishedTypes).toEqual([
        'function',
        'function',
        'function',
        'function',
      ]);
    });

    // [RED-4-PUBLISHED-TRAVEL-HEADING]
    it('still consumes the PUBLISHED travel heading (and holds the aim on a straight lane)', () => {
      // Arrange: the measured slide and the straight lane.
      const slid = stepChaseCameraFollow(
        createChaseCameraFollowState(),
        measuredStep(1, 1),
      );
      const straight = stepChaseCameraFollow(
        createChaseCameraFollowState(),
        {
          ...measuredStep(1, 1),
          travelHeadingRad: MEASURED_NOSE_HEADING_RAD,
        },
      );
      // Act
      // Assert
      expect(slid.offsetRad).not.toBe(0);
      expect(Math.sign(slid.offsetRad)).toBe(
        Math.sign(MEASURED_PUBLISHED_SLIP_RAD),
      );
      expect(straight.offsetRad).toBe(0);
    });

    // [RED-4-BODY-PRESENTATION-UNTOUCHED]
    it('leaves the BODY presentation on the published pose (never on the chase follow)', () => {
      // Arrange: the browser entry source (the three live call sites).
      const browserEntrySource = readFileSync(BROWSER_ENTRY_PATH, 'utf-8');
      // Act: the source is matched against the body presentation.
      // Assert: the body yaw still comes from the published pose and NO
      // body field may be assigned from the chase follow.
      expect(browserEntrySource).toMatch(
        /carPoseBag\.driftYawOffsetRad = car\.drift\.pose\.pose/,
      );
      expect(browserEntrySource).not.toMatch(
        /carPoseBag\.[A-Za-z]+ = [^;]*chaseFollow/,
      );
    });
  });
});
