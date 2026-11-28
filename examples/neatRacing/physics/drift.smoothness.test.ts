/**
 * RED-phase contract tests for the SMOOTH drift (Step 30, slice
 * `P5-S30-red`; AC-109-RED).
 *
 * THIS HOME PINS FOUR CONTRACTS, ONE PER `it`:
 *
 * - RED-S30-1 (the DAMPING contract): the damping ratio of the visual
 *   pose spring is MEASURED from the two PUBLISHED pose coefficients and
 *   is asserted to be at least 0.7, so the UNDER-DAMPED 0.35 spring fails;
 * - RED-S30-2 (the COUNTERSTEER CONTINUITY contract): the countersteer
 *   authority is asserted to be CONTINUOUS in the steering input, so the
 *   hard `|steer| >= 0.9` gate and the 70 m/s^2 step fail;
 * - RED-S30-3 (the SLIP HYSTERESIS contract): the PUBLISHED slip is
 *   asserted to be HELD (a depth hysteresis) instead of re-derived every
 *   tick, so the per-tick re-derivation fails;
 * - RED-S30-4 (the YAW-RATE FEEDBACK contract): the heading advance is
 *   asserted to be fed back from the MEASURED yaw rate, so the absent
 *   steering smoothing / yaw-rate feedback fails.
 *
 * AND ONE ALREADY-PASSING INVARIANT PIN (it passes today and MUST survive
 * the green): RED-S30-AI-UNCOUPLED - the AI FITNESS DOMAIN
 * (`examples/neatRacing/ai/oscillation-penalty.utils.ts`) stays UNCOUPLED
 * from the physics drift seam.
 *
 * THE RED DECISIONS (why this file fails BY ASSERTION and never by a
 * compile error):
 *
 * - EVERY probe targets an EXISTING module: the pure drift model
 *   (`./drift.model`) and the step service (`./physics.step.service`).
 *   NOTHING that does not exist yet is imported, so no contract can be
 *   masked by an import failure;
 * - the countersteer FULL SCALE is a LOCAL LITERAL
 *   (`MEASURED_COUNTERSTEER_FULL_SCALE_MPS2`): the green lane may delete
 *   or rename the published constants, and an import failure is FORBIDDEN
 *   in this lane;
 * - the published slip is NEVER asserted against the 0.45 rad COMMITTED
 *   slide budget: 0.45 rad bounds the COMMITTED slide and NEVER clips the
 *   PUBLISHED slip, so such an assertion would fail for the WRONG reason;
 * - the assertions use the HOUSE OBJECT FORM (`expect({...}).toEqual({...})`)
 *   with the MEASURED numbers carried on BOTH sides of the comparison (a
 *   READOUT that rides along in the diff) and the CONTRACT carried by a
 *   boolean flag: the failure message shows the measured numbers and the
 *   flag is the field that fails, so every red is a NAMED runtime failure.
 *
 * THE MEASURED BASELINE (read off the live bytes on 2026-11-22, BEFORE any
 * production edit):
 *
 * - the pose spring damping ratio is `12.6 / (2 * sqrt(324))` = 0.35
 *   EXACTLY (the contract wants at least 0.7);
 * - the countersteer authority jumps by 63 m/s^2 (90 % of the 70 m/s^2
 *   full scale) between the `|steer| = 0.899` and the `|steer| = 0.9`
 *   samples of the 1001-sample sweep (the contract wants at most 7, i.e.
 *   10 % of the full scale);
 * - the 300-tick sustained-drift lane publishes a slip that is RE-DERIVED
 *   on EVERY tick: 0 of the 300 ticks differ from
 *   `computeSlipAngle(lateralVelocity, speed)` (the contract wants at
 *   least ONE held tick), and 273 of the 300 per-tick re-derivation steps
 *   move the slip by at most 0.0005 rad, so ANY non-zero chatter band
 *   produces holds once the green lands;
 * - the heading advance of the RELEASE tick (the wheel goes neutral) is
 *   EXACTLY 0 rad, while the last steer tick advances the heading by
 *   0.02554904338790065 rad (the contract wants the release tick to keep
 *   rotating, and never to advance MORE than the pre-release tick).
 *
 * DETERMINISM: no `Math.random`, no `Date.now`, no timers and no async -
 * the lane is a pure function of the seeded inputs.
 */

import { describe, expect, it } from '@jest/globals';
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import {
  computeCountersteerAuthority,
  computeSlipAngle,
  NEATRACING_DRIFT_POSE_DAMPING_RAD_PER_S,
  NEATRACING_DRIFT_POSE_STIFFNESS_RAD_PER_S2,
} from './drift.model';
import type { DriftCommittedState, DriftStepInput } from './drift.model';
import {
  createInitialPhysicsCarState,
  NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
  stepPhysics,
} from './physics.step.service';
import type {
  PhysicsCarState,
  PhysicsStepInput,
  PhysicsSurfaceType,
} from './physics.step.service';

/**
 * The minimum damping ratio of the visual pose spring (the contract of
 * RED-S30-1).
 *
 * WHY 0.7: a ratio below 0.7 RINGS - the body overshoots and comes back
 * (about 31 % first overshoot at the measured 0.35), and a ringing body
 * yaw IS the wobble the user sees. At 0.7 the spring is still lively but
 * it never rings.
 */
const MIN_POSE_DAMPING_RATIO = 0.7;

/**
 * The MEASURED countersteer full scale (m/s^2) - a LOCAL LITERAL.
 *
 * WHY A LOCAL LITERAL: the published countersteer constants
 * (`NEATRACING_DRIFT_COUNTERSTEER_AUTHORITY_MPS2` and
 * `NEATRACING_DRIFT_COUNTERSTEER_MIN_STEER`) may be DELETED or RENAMED by
 * the green lane, and an import failure is FORBIDDEN in the RED lane - so
 * this file never imports them. The value is the MEASURED full scale of
 * the published authority (70 m/s^2) and it is only ever read as the
 * REFERENCE the continuity limit is quoted against.
 */
const MEASURED_COUNTERSTEER_FULL_SCALE_MPS2 = 70;

/**
 * The maximum allowed jump between two ADJACENT steering samples of the
 * countersteer sweep (m/s^2) - 10 % of the measured 70 m/s^2 full scale.
 *
 * WHY 10 %: a CONTINUOUS ramp over the full scale moves the authority by
 * at most ~0.7 m/s^2 per 0.001 of the steering axis, so a 7 m/s^2 bound
 * is GENEROUS: it lets a smooth ramp through and still rejects the
 * bang-bang step (63 m/s^2 today).
 */
const COUNTERSTEER_CONTINUITY_LIMIT_MPS2 = 7;

/** The number of steering samples of the countersteer continuity sweep. */
const COUNTERSTEER_SWEEP_SAMPLES = 1001;

/** The number of ticks of the sustained-drift lane (RED-S30-3). */
const DRIFT_LANE_TICKS = 300;

/** The number of full-steer ticks before the release tick (RED-S30-4). */
const YAW_STEER_PHASE_TICKS = 120;

/** The number of straight ticks after the release tick (RED-S30-4). */
const YAW_STRAIGHT_TICKS = 3;

/** The road surface resolver: the sustained-drift lane never leaves it. */
const ROAD_SURFACE_RESOLVER = (): PhysicsSurfaceType => 'road';

/**
 * Wraps an angle into `(-PI, PI]`.
 *
 * The heading advance is measured across a WRAPPED heading, so the
 * difference of two wrapped headings is wrapped again before it is
 * compared - otherwise a heading that crosses the `-PI / PI` seam would
 * read a huge fake advance.
 *
 * @param angleRad - The raw angle (rad).
 * @returns The same angle wrapped into `(-PI, PI]`.
 */
const wrapToPi = (angleRad: number): number => {
  const twoPi = 2 * Math.PI;
  let wrapped = angleRad % twoPi;
  if (wrapped > Math.PI) {
    wrapped -= twoPi;
  }
  if (wrapped <= -Math.PI) {
    wrapped += twoPi;
  }
  return wrapped;
};

/**
 * Reads the module specifiers of every IMPORT statement of a source.
 *
 * The scan looks at the IMPORT STATEMENTS only (not at the prose), so a
 * comment that merely NAMES the AI domain can never fake a coupling.
 *
 * @param source - The source text of one module.
 * @returns The module specifiers, in source order.
 */
const readImportSpecifiers = (source: string): string[] => {
  const importPattern =
    /(?:^|\n)\s*import\s+(?:type\s+)?(?:[^'"]*?\s+from\s+)?['"]([^'"]+)['"]/g;
  const specifiers: string[] = [];
  let match = importPattern.exec(source);
  while (match !== null) {
    specifiers.push(match[1]);
    match = importPattern.exec(source);
  }
  return specifiers;
};

describe('drift smoothness contracts (AC-109-RED)', () => {
  // [RED-S30-1]
  it('AC-109-RED: [RED-S30-1] the visual pose spring is damped at least at a 0.7 damping ratio', () => {
    // Arrange: the damping ratio is MEASURED from the two PUBLISHED pose
    // spring coefficients, never from a literal - a re-pin of EITHER
    // export moves the measured ratio with it.
    const dampingRatio =
      NEATRACING_DRIFT_POSE_DAMPING_RAD_PER_S /
      (2 * Math.sqrt(NEATRACING_DRIFT_POSE_STIFFNESS_RAD_PER_S2));

    // Act + Assert
    // The MEASURED numbers ride along in the object so the failure
    // message names them (0.35 today); the CONTRACT is the flag.
    expect({
      dampingRatio,
      minimumDampingRatio: MIN_POSE_DAMPING_RATIO,
      damped: dampingRatio >= MIN_POSE_DAMPING_RATIO,
    }).toEqual({
      dampingRatio,
      minimumDampingRatio: MIN_POSE_DAMPING_RATIO,
      damped: true,
    });
  });

  // [RED-S30-2]
  it('AC-109-RED: [RED-S30-2] the countersteer authority is continuous in the steering input', () => {
    // Arrange: a committed LEFT slide the driver is catching, probed at a
    // constant speed and a constant speed percentile.
    const committed: DriftCommittedState = {
      slip: -0.3,
      turnSign: 1,
      reArmed: true,
    };
    const probeInput = (steer: number): DriftStepInput => ({
      steer,
      speed: 60,
      speedPercent: 1,
      dt: 1 / 60,
      lateralVelocity: 0,
      slipAngle: -0.3,
    });

    // Act: sweep the steering axis from 0 down to -1 in 1001 steps and
    // measure the largest jump between two ADJACENT samples.
    let maxJumpMps2 = 0;
    let previousAuthorityMps2 = computeCountersteerAuthority(
      committed,
      probeInput(0),
    );
    for (let sample = 1; sample < COUNTERSTEER_SWEEP_SAMPLES; sample += 1) {
      const steer = -sample / 1000;
      const authorityMps2 = computeCountersteerAuthority(
        committed,
        probeInput(steer),
      );
      maxJumpMps2 = Math.max(
        maxJumpMps2,
        Math.abs(authorityMps2 - previousAuthorityMps2),
      );
      previousAuthorityMps2 = authorityMps2;
    }

    // Act + Assert
    // The MEASURED jump rides along in the object so the failure message
    // names it (63 m/s^2 today); the CONTRACT is the `continuous` flag.
    expect({
      maxJumpMps2,
      limitMps2: COUNTERSTEER_CONTINUITY_LIMIT_MPS2,
      fullScaleMps2: MEASURED_COUNTERSTEER_FULL_SCALE_MPS2,
      continuous: maxJumpMps2 <= COUNTERSTEER_CONTINUITY_LIMIT_MPS2,
    }).toEqual({
      maxJumpMps2,
      limitMps2: COUNTERSTEER_CONTINUITY_LIMIT_MPS2,
      fullScaleMps2: MEASURED_COUNTERSTEER_FULL_SCALE_MPS2,
      continuous: true,
    });
  });

  // [RED-S30-3]
  it('AC-109-RED: [RED-S30-3] the published slip is held inside the chatter band instead of re-derived every tick', () => {
    // Arrange: the sustained-drift lane - the arcade top speed, the full
    // throttle and the full LEFT steer on the road.
    let state: PhysicsCarState = {
      ...createInitialPhysicsCarState(42),
      speed: NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
    };
    const steerInput: PhysicsStepInput = { throttle: 1, brake: 0, steer: 1 };

    // Act: on EVERY tick compare the PUBLISHED slip with the slip
    // RE-DERIVED from the published state. A tick on which the two
    // differ IS a hysteresis HOLD: the published slip was held while the
    // underlying state moved on.
    let heldTicks = 0;
    for (let tick = 0; tick < DRIFT_LANE_TICKS; tick += 1) {
      state = stepPhysics(state, steerInput, ROAD_SURFACE_RESOLVER);
      const rederivedSlipRad = computeSlipAngle(
        state.lateralVelocity,
        state.speed,
      );
      if (state.slipAngle !== rederivedSlipRad) {
        heldTicks += 1;
      }
    }

    // Act + Assert
    // NO slip-envelope bound is asserted here: the 0.45 rad budget bounds
    // the COMMITTED slide and NEVER clips the PUBLISHED slip, so such an
    // assertion would fail for the WRONG reason.
    // The MEASURED held-tick count rides along in the object so the
    // failure message names it (0 today); the CONTRACT is the flag.
    expect({
      heldTicks,
      laneTicks: DRIFT_LANE_TICKS,
      holdsPublishedSlip: heldTicks > 0,
    }).toEqual({
      heldTicks,
      laneTicks: DRIFT_LANE_TICKS,
      holdsPublishedSlip: true,
    });
  });

  // [RED-S30-4]
  it('AC-109-RED: [RED-S30-4] the release tick keeps advancing the heading from the measured yaw rate', () => {
    // Arrange: the steer phase of the sustained-drift lane.
    let state: PhysicsCarState = {
      ...createInitialPhysicsCarState(42),
      speed: NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
    };
    const steerInput: PhysicsStepInput = { throttle: 1, brake: 0, steer: 1 };
    const straightInput: PhysicsStepInput = { throttle: 1, brake: 0, steer: 0 };

    // Act (the steer phase): the last steer tick is the PRE-RELEASE
    // advance the release tick may never exceed.
    let preReleaseAdvanceRad = 0;
    for (let tick = 0; tick < YAW_STEER_PHASE_TICKS; tick += 1) {
      const headingBeforeRad = state.heading;
      state = stepPhysics(state, steerInput, ROAD_SURFACE_RESOLVER);
      preReleaseAdvanceRad = wrapToPi(state.heading - headingBeforeRad);
    }

    // Act (the RELEASE tick): the wheel goes neutral, so the demanded
    // turn rate is zero - but the body keeps rotating while the yaw
    // decays, so the advance is fed back from the MEASURED yaw rate.
    const headingBeforeReleaseRad = state.heading;
    state = stepPhysics(state, straightInput, ROAD_SURFACE_RESOLVER);
    const releaseAdvanceRad = wrapToPi(state.heading - headingBeforeReleaseRad);

    // Act (the straight ticks after the release): the yaw may only decay.
    const straightAdvancesRad: number[] = [];
    for (let tick = 0; tick < YAW_STRAIGHT_TICKS; tick += 1) {
      const headingBeforeRad = state.heading;
      state = stepPhysics(state, straightInput, ROAD_SURFACE_RESOLVER);
      straightAdvancesRad.push(wrapToPi(state.heading - headingBeforeRad));
    }
    const firstStraightAdvanceRad = straightAdvancesRad[0];

    // Act + Assert
    // The release advance is NOT asserted to be STRICTLY LESS than the
    // pre-release advance: a feedback implementation may reuse the
    // previous tick's measured rate and be EXACTLY equal.
    // The MEASURED advances ride along in the object so the failure
    // message names them (0 rad release advance today); the CONTRACT is
    // carried by the two flags.
    expect({
      releaseAdvanceRad,
      preReleaseAdvanceRad,
      firstStraightAdvanceRad,
      bodyKeepsRotating: releaseAdvanceRad > 0,
      yawOnlyDecays: releaseAdvanceRad <= preReleaseAdvanceRad,
    }).toEqual({
      releaseAdvanceRad,
      preReleaseAdvanceRad,
      firstStraightAdvanceRad,
      bodyKeepsRotating: true,
      yawOnlyDecays: true,
    });
  });
});

describe('drift smoothness invariants (the carried-forward pins)', () => {
  // [RED-S30-AI-UNCOUPLED]
  it('AC-109-RED: [RED-S30-AI-UNCOUPLED] the physics seam never imports the AI fitness domain', () => {
    // Arrange: the source scan over the NON-TEST physics sources. The
    // `*.test.ts` files are EXCLUDED, so the scan can never read this
    // file's own import statements.
    const directory = join(process.cwd(), 'examples', 'neatRacing', 'physics');
    const files = readdirSync(directory).filter(
      (name) => name.endsWith('.ts') && !name.endsWith('.test.ts'),
    );

    // Act: collect every import that reaches into the AI domain.
    const aiDomainImports: string[] = [];
    for (const name of files) {
      const source = readFileSync(join(directory, name), 'utf8');
      for (const specifier of readImportSpecifiers(source)) {
        if (specifier.includes('../ai/')) {
          aiDomainImports.push(`${name}: ${specifier}`);
        }
      }
    }

    // Act + Assert
    // The scanned-file count rides along in the object so the pin reports
    // how wide the scan was; the CONTRACT is the `uncoupled` flag.
    expect({
      scannedFiles: files.length,
      aiDomainImports,
      uncoupled: aiDomainImports.length === 0,
    }).toEqual({
      scannedFiles: files.length,
      aiDomainImports: [],
      uncoupled: true,
    });
  });
});
