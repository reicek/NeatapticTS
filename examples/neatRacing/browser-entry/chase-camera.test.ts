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
