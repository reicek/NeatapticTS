/**
 * RED-phase contract tests for the Neat Racing arcade drift FEEL
 * sharpening (slice P1-drift-feel-red; Step 14N; AC-087 / AC-089; the
 * recorded direction D-11: A BIGGER SLIDE).
 *
 * This file is the RED home of the drift-model contract set. It is written
 * AGAINST the ratified 14N public API (the integration artifact section 4):
 *
 * - the drift channel is `DriftChannelState { committed, pose }`;
 * - the physical commitment is `DriftCommittedState { slip, turnSign,
 *   reArmed }` - the turn sign is the TURN direction (+1 = LEFT) and is
 *   NEVER `sign(slip)`;
 * - the visual accumulator is `DriftPoseSpringState { pose, poseRate,
 *   articulation }` and it is the ONLY visible-yaw source;
 * - the physical tick input CHANGED shape: there is no `yawRatePerTick` and
 *   no `centrifugalMultiplier` any more, and the tick-start
 *   `lateralVelocity` / `slipAngle` are INPUTS.
 *
 * RED FORM (the artifact-mandated import-fails form): the not-yet-existing
 * exports are imported BY NAME, so the suite fails to compile with
 * `TS2305: Module '"./drift.model"' has no exported member 'X'`. That IS
 * the red - no contract is masked by a fixture error.
 *
 * The temporary PASS 1 measurement case is REMOVED in this step (the packet
 * requires it): the measured current-model baselines live in the plan
 * ledger, they are never asserted here.
 *
 * STEP 14Q (slice `P1-drift-law-red`; AC-087 / AC-088 / AC-089) ADDS the
 * PROPORTIONAL traction-law contracts to this file: the re-opened
 * speed-blind grip budget (RED-1), the speed-proportional slide DEPTH
 * (RED-2), the speed-proportional entry RAMP and the softened MIN_SPEED
 * gate (RED-3), the neutral-wheel RELEASE (RED-4), the speed-proportional
 * slip RESPONSE (RED-6) and the EQUILIBRIUM speed deficit (RED-7). The
 * RED-5 contracts (the published traction scalar) live in the SERVICE
 * test file, because the published state field is a service contract.
 *
 * The 14Q RED FORM (the artifact-mandated import-fails form) is KEPT: the
 * not-yet-existing `computeTractionSeverity` export is imported BY NAME,
 * so this file fails to compile with `TS2305: Module '"./drift.model"'
 * has no exported member 'computeTractionSeverity'`. That IS the red - no
 * contract is masked by a fixture error, and NO placeholder export is
 * created to make the import resolve.
 *
 * The MEASURED current-model baselines of the 14Q contracts live in the
 * plan red_evidence block (they are never asserted here): the grip budget factor is
 * 0.4166666666666667 at BOTH 0.9 rad and 1 / 0.2 speed (the speed-blind
 * G-2 pin), the committed slide is exactly 0.45 rad at EVERY speed above
 * the gate (89 / 45 / 30 / 25 / 20 / 16 / 15.5 m/s), the entry ramp needs
 * the same 1 / 4 / 8 ticks to reach 0.05 / 0.2 / 0.45 rad at EVERY speed,
 * the neutral wheel keeps the slide frozen at -0.45 rad with the turn sign
 * latched at 1, the one-tick chase fraction is 0.26555632807026885 at
 * EVERY speed, and the equilibrium speed deficit is 32.9153237248284 km/h
 * (the 600-tick transient reads 32.55472788998302 km/h).
 */

import { describe, expect, it } from '@jest/globals';
import { readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

import * as driftModel from './drift.model';
import {
  NEATRACING_DRIFT_GRIP_FALLOFF_RATE_PER_RAD,
  NEATRACING_DRIFT_GRIP_FALLOFF_THRESHOLD_RAD,
  NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD,
  NEATRACING_DRIFT_MIN_SLIDE_PROXY_RAD,
  NEATRACING_DRIFT_MIN_SPEED_MPS,
  NEATRACING_DRIFT_POSE_ARTICULATION_SLEW_PER_S,
  NEATRACING_DRIFT_POSE_DAMPING_RAD_PER_S,
  NEATRACING_DRIFT_POSE_STEER_YAW_FULL_SLIP_RAD,
  NEATRACING_DRIFT_POSE_STEER_YAW_RAD,
  NEATRACING_DRIFT_POSE_STIFFNESS_RAD_PER_S2,
  NEATRACING_DRIFT_PROXY_ENGAGEMENT_MIN_RAD,
  NEATRACING_DRIFT_PROXY_MIN_FRAMES,
  NEATRACING_DRIFT_PUBLISHED_SLIP_MAX_RAD,
  NEATRACING_DRIFT_PUBLISHED_SLIP_OFFROAD_MAX_RAD,
  NEATRACING_DRIFT_PUBLISHED_SLIP_OVERSHOOT_RAD,
  NEATRACING_DRIFT_STEER_ENGAGE_THRESHOLD,
  NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD,
  computeCountersteerAuthority,
  computeDriftSlideDepthBudgetRad,
  computeDriftSlideProxyRad,
  computeGripBudgetFactor,
  computeLongitudinalSpeedScrub,
  computeNextLateralVelocity,
  computeSlipAngle,
  computeTractionSeverity,
  createZeroDriftChannelState,
  stepDriftCommitment,
  stepDriftPoseSpring,
} from './drift.model';
import type {
  DriftChannelState,
  DriftCommittedState,
  DriftPoseSpringState,
  DriftPoseStepInput,
  DriftStepInput,
} from './drift.model';
import {
  createInitialPhysicsCarState,
  NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
  NEATRACING_PHYSICS_STEERING_RATE,
  stepPhysics,
} from './physics.step.service';
import type {
  PhysicsCarState,
  PhysicsStepInput,
  PhysicsSurfaceType,
} from './physics.step.service';

/** Deterministic JSON serialization of a drift channel (byte-identity helper). */
const serializeDriftChannel = (channel: DriftChannelState): string =>
  JSON.stringify(channel);

/** SCN-DRIFT-01: 60 Hz; 180 build ticks at full LEFT (steer +1), then 120 counter ticks at full RIGHT (steer -1), at the arcade top speed. */
const SCN_DRIFT_01_DT_S = 1 / 60;
/** The arcade top speed of SCN-DRIFT-01 (m/s). */
const SCN_DRIFT_01_SPEED_MPS = 89;
/** The build phase of SCN-DRIFT-01 (full LEFT, 3.000 s). */
const SCN_DRIFT_01_BUILD_TICKS = 180;
/** The counter phase of SCN-DRIFT-01 (full RIGHT, 2.000 s). */
const SCN_DRIFT_01_COUNTER_TICKS = 120;

/**
 * The pinned pose-spring point of merge 7a (omega = 18 rad/s, zeta = 0.35).
 *
 * The pop, the settle and the per-tick step are pinned TOGETHER at this ONE
 * point: a green pop bought with a different point is a contract violation.
 */
const POSE_POINT = {
  stiffnessRadPerS2: NEATRACING_DRIFT_POSE_STIFFNESS_RAD_PER_S2,
  dampingRadPerS: NEATRACING_DRIFT_POSE_DAMPING_RAD_PER_S,
  articulationRad: NEATRACING_DRIFT_POSE_STEER_YAW_RAD,
  armSlipRad: NEATRACING_DRIFT_POSE_STEER_YAW_FULL_SLIP_RAD,
  slewPerS: NEATRACING_DRIFT_POSE_ARTICULATION_SLEW_PER_S,
};

/**
 * The MEASURED post-RC-1 raw off-road published-slip envelope peak (rad).
 *
 * PROVENANCE (the W2a measurement: seed 42, the top-speed start, a
 * full-throttle full-steer drift input, swept over the seven
 * grass/curb/transition lanes): the off-road drag drives the same
 * chase-lag overshoot from a FALLING denominator, so the off-road
 * published slip peaks well above the ON-ROAD ceiling.
 *
 * This is a MEASUREMENT, not a tuned knob - it is the INPUT of the
 * off-road declaration rule and it is NEVER a candidate ceiling.
 */
const MEASURED_OFF_ROAD_RAW_PEAK_RAD = 0.4872167352892035;

/**
 * The off-road declaration margin (rad) added to the measured raw peak
 * before the ceiling is declared.
 *
 * The margin absorbs the chase-lag drift of a lane that is not the
 * measured one; a re-measurement that moves the raw peak by more than the
 * margin FORCES a re-declaration of the ceiling - a RECORDED decision,
 * never a silent clip.
 */
const OFF_ROAD_DECLARATION_MARGIN_RAD = 0.005;

/**
 * The declaration grid step (rad): the off-road ceiling is declared on a
 * 0.001 rad grid, and `OFF_ROAD_DECLARATION_STEPS_PER_RAD` is its
 * reciprocal.
 */
const OFF_ROAD_DECLARATION_GRID_RAD = 0.001;

/**
 * The number of declaration grid steps per radian - the scale factor of the
 * declaration rule (`ceil((peak + margin) * steps) / steps`).
 */
const OFF_ROAD_DECLARATION_STEPS_PER_RAD = 1000;

/**
 * The 14N physical tick input.
 *
 * The 14N slice drops `yawRatePerTick` and `centrifugalMultiplier` and adds
 * the tick-start `lateralVelocity` / `slipAngle`. The shape is expressed as
 * an `Omit` so this file still COMPILES against the pre-14N `DriftStepInput`
 * - the only RED reason allowed is the missing export, never a fixture
 * mismatch.
 */
type DriftStepSample = Omit<
  DriftStepInput,
  'yawRatePerTick' | 'centrifugalMultiplier'
> & {
  lateralVelocity: number;
  slipAngle: number;
};

/**
 * Builds a 14N physical tick input.
 *
 * The return type is deliberately loose: the fixtures must not raise a
 * secondary diagnostic while the module is RED.
 */
const makeStepInput = (overrides: Partial<DriftStepSample> = {}): any => ({
  steer: 0,
  speed: SCN_DRIFT_01_SPEED_MPS,
  speedPercent: 1,
  dt: SCN_DRIFT_01_DT_S,
  lateralVelocity: 0,
  slipAngle: 0,
  ...overrides,
});

/** Builds a 14N pose-spring tick input (the tick-END slip feeds the pose). */
const makePoseInput = (overrides: Partial<DriftPoseStepInput> = {}): any => ({
  steer: 0,
  speedPercent: 1,
  slipAngle: 0,
  dt: SCN_DRIFT_01_DT_S,
  ...overrides,
});

/**
 * Builds a committed state for the pure-model fixtures.
 *
 * The zero state carries the latch (`turnSign` 0, `reArmed` false); the
 * fixtures override only the fields they exercise. A committed fixture is
 * ARMED (`reArmed` true): the re-arm token gates a fresh ENGAGEMENT, and a
 * fixture that stands for an already-latched drift must not be blocked by
 * the engagement gate.
 */
const makeCommittedState = (
  slip: number,
  turnSign: number,
): DriftCommittedState => ({
  ...createZeroDriftChannelState().committed,
  slip,
  turnSign,
  reArmed: true,
});

/** One committed (physical) tick sample of a drift run. */
interface CommittedTickSample {
  /** The committed physical state AFTER the tick. */
  committed: DriftCommittedState;
  /** The lateral velocity AFTER the tick (m/s). */
  lateralVelocity: number;
  /** The tick-start slip angle (rad). */
  slipAngleRad: number;
  /** The tick-start lateral velocity (m/s). */
  lateralVelocityAtStart: number;
}

/** The outcome of a committed (physical) drift run. */
interface CommittedRun {
  /** The committed state after the last tick. */
  committed: DriftCommittedState;
  /** The lateral velocity after the last tick (m/s). */
  lateralVelocity: number;
  /** Every tick sample, in order. */
  samples: CommittedTickSample[];
  /** The maximum absolute committed slip over the run (rad). */
  maxAbsCommittedSlipRad: number;
  /** The maximum absolute lateral velocity over the run (m/s). */
  maxAbsLateralVelocityMps: number;
}

/**
 * Drives the pure 14N physical commitment API for a fixed number of ticks.
 *
 * The channel carries the committed state and the pose spring; the lateral
 * velocity is a TICK INPUT, so it is threaded through the loop by hand.
 */
const runCommittedTicks = (
  ticks: number,
  steer: number,
  speed: number,
  options: {
    /** The channel to continue from (defaults to a fresh zero channel). */
    start?: DriftChannelState;
    /** The lateral velocity to continue from (defaults to 0). */
    startLateralVelocity?: number;
    /** The tick-start slip angle (defaults to 0). */
    slipAngleRad?: number;
    /**
     * The tick-start speed PERCENT (defaults to the speed / top speed).
     *
     * The 14Q speed-axis contract needs a lane where the percent and the
     * absolute speed DISAGREE, so the percent is overridable.
     */
    speedPercent?: number;
  } = {},
): CommittedRun => {
  const channel = options.start ?? createZeroDriftChannelState();
  let committed: DriftCommittedState = channel.committed;
  let lateralVelocity = options.startLateralVelocity ?? 0;
  const slipAngleRad = options.slipAngleRad ?? 0;
  const samples: CommittedTickSample[] = [];
  let maxAbsCommittedSlipRad = 0;
  let maxAbsLateralVelocityMps = 0;

  for (let tick = 0; tick < ticks; tick += 1) {
    const input = makeStepInput({
      steer,
      speed,
      speedPercent: options.speedPercent ?? speed / SCN_DRIFT_01_SPEED_MPS,
      dt: SCN_DRIFT_01_DT_S,
      lateralVelocity,
      slipAngle: slipAngleRad,
    });
    committed = stepDriftCommitment(committed, input);
    lateralVelocity = computeNextLateralVelocity(committed, input);
    samples.push({
      committed,
      lateralVelocity,
      slipAngleRad,
      lateralVelocityAtStart: input.lateralVelocity,
    });
    maxAbsCommittedSlipRad = Math.max(
      maxAbsCommittedSlipRad,
      Math.abs(committed.slip),
    );
    maxAbsLateralVelocityMps = Math.max(
      maxAbsLateralVelocityMps,
      Math.abs(lateralVelocity),
    );
  }

  return {
    committed,
    lateralVelocity,
    samples,
    maxAbsCommittedSlipRad,
    maxAbsLateralVelocityMps,
  };
};

/**
 * The longest CONSECUTIVE run of samples at or above a threshold.
 *
 * The run length - not the total count - is the observable: a single-frame
 * spike can never satisfy a consecutive-run gate.
 */
const longestRunAtOrAbove = (values: number[], threshold: number): number => {
  let longest = 0;
  let current = 0;
  for (const value of values) {
    if (value >= threshold) {
      current += 1;
      longest = Math.max(longest, current);
    } else {
      current = 0;
    }
  }
  return longest;
};

/** One SCN-DRIFT-01 tick sample. */
interface DriftScenarioSample {
  /** The global tick index (0-based; the build phase first). */
  tick: number;
  /** The phase of the tick. */
  phase: 'build' | 'counter';
  /** The committed physical state after the tick. */
  committed: DriftCommittedState;
  /** The lateral velocity after the tick (m/s). */
  lateralVelocity: number;
  /** The pose spring state after the tick. */
  pose: DriftPoseSpringState;
  /** The pose target derived from the post-tick state (rad). */
  poseTargetRad: number;
  /** The articulation-only term of the pose target (rad). */
  articulationTargetRad: number;
  /** The slide proxy of the post-tick committed slip (rad). */
  proxyRad: number;
}

/** The SCN-DRIFT-01 run result. */
interface DriftScenarioRun {
  /** The pose point the run was driven at. */
  point: typeof POSE_POINT;
  /** The far-side pop over the counter phase (deg). */
  popDeg: number;
  /** The first counter tick after which the pose stays within 0.5 deg of the final pose. */
  settleTicks: number;
  /** The largest per-tick pose-target step over all ticks (deg). */
  maxStepDeg: number;
  /** The largest per-tick articulation-target step over all ticks (deg). */
  maxArticulationStepDeg: number;
  /** The largest absolute pose over the run (rad). */
  maxAbsPoseRad: number;
  /** The largest slide proxy over the run (rad). */
  maxProxyRad: number;
  /** The longest consecutive run at or above the prominence floor (build phase). */
  longestRunAboveFloorTicks: number;
  /** The longest consecutive run at or above the engagement gate (build phase). */
  longestRunAboveEngagementGateTicks: number;
  /** The pose at the end of the build phase (rad). */
  poseAtHold: number;
  /** The committed slip at the end of the build phase (rad). */
  slipAtHold: number;
  /** The first build tick at which the proxy reaches 63 percent of the hold proxy. */
  ticksTo63PctOfHold: number;
  /** The largest (signed) committed slip over the counter phase (rad). */
  maxSlipAfterCounterOnset: number;
  /** The pose at the last counter tick (rad). */
  poseFinal: number;
  /** The latched turn sign at the end of the build phase. */
  turnSignAtHold: number;
  /** The latched turn sign at the end of the counter phase. */
  turnSignAfterCounter: number;
  /** The committed slip after 20 counter ticks (rad). */
  slipAfter20Counter: number;
  /** The lateral velocity after 20 counter ticks (m/s). */
  lateralAfter20Counter: number;
  /** The largest absolute committed slip over the counter phase (rad). */
  maxAbsSlipDuringCounter: number;
  /** The lateral velocity at the end of the build phase (m/s). */
  lateralVelocityAtHold: number;
  /** The drift channel after the last tick (the byte-identity subject). */
  finalChannel: DriftChannelState;
  /** Every tick sample, in order. */
  samples: DriftScenarioSample[];
}

/**
 * SCN-DRIFT-01: the arcade-refs / gaming research scenario.
 *
 * Phase A (BUILD): 180 ticks (3.000 s) at full LEFT (steer +1).
 * Phase B (COUNTER): 120 ticks (2.000 s) at full RIGHT (steer -1), continuing
 * from the Phase-A channel AND its lateral velocity.
 *
 * The measurement conventions are CONTRACTS, not implementation details
 * (D-5 / D-6 / D-8 / D-9):
 *
 * - `poseTarget(t) = slipAngle(t) * VISUAL_YAW_GAIN_RAD + POSE_STEER_YAW_RAD * articulation(t)`
 *   with the POST-tick articulation carried in the state;
 * - `poseFinal` is the pose at the LAST Phase-B tick;
 * - `popDeg` is `max |pose(t) - poseFinal|` over the Phase-B ticks;
 * - `settleTicks` is the first Phase-B tick after which `|pose - poseFinal|`
 *   is below 0.5 deg for EVERY remaining tick;
 * - `maxStepDeg` is `max |poseTarget(t) - poseTarget(t-1)|` over ALL ticks and
 *   `maxArticulationStepDeg` is the same maximum restricted to the
 *   articulation term.
 *
 * The sign convention is production-pinned: steer +1 = LEFT and a LEFT turn
 * produces a NEGATIVE slip.
 */
const runDriftScenario = (point = POSE_POINT): DriftScenarioRun => {
  const samples: DriftScenarioSample[] = [];
  let channel = createZeroDriftChannelState();
  let lateralVelocity = 0;
  let previousTargetRad = 0;
  let previousArticulationTargetRad = 0;
  let maxStepDeg = 0;
  let maxArticulationStepDeg = 0;
  let maxAbsPoseRad = 0;
  let maxProxyRad = 0;
  let tick = 0;

  const drive = (
    steer: number,
    ticks: number,
    phase: 'build' | 'counter',
  ): void => {
    for (let step = 0; step < ticks; step += 1) {
      const input = makeStepInput({
        steer,
        speed: SCN_DRIFT_01_SPEED_MPS,
        speedPercent: 1,
        dt: SCN_DRIFT_01_DT_S,
        lateralVelocity,
        slipAngle: channel.committed.slip,
      });
      const committed = stepDriftCommitment(channel.committed, input);
      const nextLateralVelocity = computeNextLateralVelocity(committed, input);
      const pose = stepDriftPoseSpring(
        channel.pose,
        makePoseInput({
          steer,
          speedPercent: 1,
          slipAngle: committed.slip,
          dt: SCN_DRIFT_01_DT_S,
        }),
      );
      const poseTargetRad =
        committed.slip * NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD +
        NEATRACING_DRIFT_POSE_STEER_YAW_RAD * pose.articulation;
      const articulationTargetRad =
        NEATRACING_DRIFT_POSE_STEER_YAW_RAD * pose.articulation;
      const proxyRad =
        Math.abs(committed.slip) *
        Math.abs(NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD - 1);

      if (tick > 0) {
        maxStepDeg = Math.max(
          maxStepDeg,
          Math.abs(poseTargetRad - previousTargetRad) * (180 / Math.PI),
        );
        maxArticulationStepDeg = Math.max(
          maxArticulationStepDeg,
          Math.abs(articulationTargetRad - previousArticulationTargetRad) *
            (180 / Math.PI),
        );
      }
      previousTargetRad = poseTargetRad;
      previousArticulationTargetRad = articulationTargetRad;
      maxAbsPoseRad = Math.max(maxAbsPoseRad, Math.abs(pose.pose));
      maxProxyRad = Math.max(maxProxyRad, proxyRad);

      samples.push({
        tick,
        phase,
        committed,
        lateralVelocity: nextLateralVelocity,
        pose,
        poseTargetRad,
        articulationTargetRad,
        proxyRad,
      });

      channel = { committed, pose };
      lateralVelocity = nextLateralVelocity;
      tick += 1;
    }
  };

  drive(1, SCN_DRIFT_01_BUILD_TICKS, 'build');
  drive(-1, SCN_DRIFT_01_COUNTER_TICKS, 'counter');

  const buildSamples = samples.filter((sample) => sample.phase === 'build');
  const counterSamples = samples.filter((sample) => sample.phase === 'counter');
  const holdSample = buildSamples[buildSamples.length - 1];
  const lastCounterSample = counterSamples[counterSamples.length - 1];
  const poseFinal = lastCounterSample.pose.pose;

  // The far-side pop: max |pose(t) - poseFinal| over the counter phase.
  const popDeg =
    counterSamples.reduce(
      (max, sample) => Math.max(max, Math.abs(sample.pose.pose - poseFinal)),
      0,
    ) *
    (180 / Math.PI);

  // The settle tick: the first counter tick after which the pose stays
  // within 0.5 deg of the final pose for EVERY remaining tick.
  const settleLimitRad = (0.5 * Math.PI) / 180;
  let settleTicks = -1;
  for (let index = 0; index < counterSamples.length; index += 1) {
    const settled = counterSamples
      .slice(index)
      .every(
        (sample) => Math.abs(sample.pose.pose - poseFinal) < settleLimitRad,
      );
    if (settled) {
      settleTicks = index;
      break;
    }
  }
  if (settleTicks < 0) {
    settleTicks = counterSamples.length;
  }

  const buildProxies = buildSamples.map((sample) => sample.proxyRad);
  const longestRunAboveFloorTicks = longestRunAtOrAbove(
    buildProxies,
    NEATRACING_DRIFT_MIN_SLIDE_PROXY_RAD,
  );
  const longestRunAboveEngagementGateTicks = longestRunAtOrAbove(
    buildProxies,
    NEATRACING_DRIFT_PROXY_ENGAGEMENT_MIN_RAD,
  );

  // The entry attack: the first build tick at which the proxy reaches 63
  // percent of the hold proxy.
  const holdProxyRad = holdSample.proxyRad;
  let ticksTo63PctOfHold = -1;
  for (let index = 0; index < buildSamples.length; index += 1) {
    if (buildSamples[index].proxyRad >= 0.63 * holdProxyRad) {
      ticksTo63PctOfHold = index;
      break;
    }
  }
  if (ticksTo63PctOfHold < 0) {
    ticksTo63PctOfHold = buildSamples.length;
  }

  // The anti-mirror observable: the largest SIGNED slip over the counter
  // phase. A mirrored drift drives it above zero.
  const maxSlipAfterCounterOnset = counterSamples.reduce(
    (max, sample) => Math.max(max, sample.committed.slip),
    Number.NEGATIVE_INFINITY,
  );
  const maxAbsSlipDuringCounter = counterSamples.reduce(
    (max, sample) => Math.max(max, Math.abs(sample.committed.slip)),
    0,
  );
  const counterSample20 = counterSamples[19];

  return {
    point,
    popDeg,
    settleTicks,
    maxStepDeg,
    maxArticulationStepDeg,
    maxAbsPoseRad,
    maxProxyRad,
    longestRunAboveFloorTicks,
    longestRunAboveEngagementGateTicks,
    poseAtHold: holdSample.pose.pose,
    slipAtHold: holdSample.committed.slip,
    ticksTo63PctOfHold,
    maxSlipAfterCounterOnset,
    poseFinal,
    turnSignAtHold: holdSample.committed.turnSign,
    turnSignAfterCounter: lastCounterSample.committed.turnSign,
    slipAfter20Counter: counterSample20.committed.slip,
    lateralAfter20Counter: counterSample20.lateralVelocity,
    maxAbsSlipDuringCounter,
    lateralVelocityAtHold: holdSample.lateralVelocity,
    finalChannel: channel,
    samples,
  };
};

/* ------------------------------------------------------------------------- *
 * STEP 14Q (slice `P1-drift-law-red`): the PROPORTIONAL traction law.
 * ------------------------------------------------------------------------- */

/**
 * The 14Q LOW-SPEED lane (m/s).
 *
 * 25 m/s is 28 percent of the arcade top speed and ABOVE the 15 m/s
 * MIN_SPEED gate, so the lane is a REAL drifting lane today. A lane
 * below the gate would satisfy the contracts VACUOUSLY (the model
 * produces exact zeros there), so the low-speed lane is chosen where the
 * slide really exists.
 */
const RED_LOW_SPEED_MPS = 25;

/**
 * The RE-PINNED top-speed committed depth (rad) - the MOMENTUM demand of
 * the 89 m/s reference lane.
 *
 * BASIS (the AC-089 re-baseline ledger row P-M-3 (the depth), Step 14S,
 * slice `P1-momentum-repin`): the OLD baseline was the FULL `0.45` rad
 * depth budget. The MEASURED committed slide of the 89 m/s full-steer
 * lane is `0.44745539952559543` rad - the momentum law can NEVER REACH
 * the budget, because the momentum excess share is strictly below 1
 * wherever the tires hold ANY of the demanded lateral acceleration. The
 * literal is DELETED and REPLACED one-for-one by the MEASURED demand:
 * the tolerance, the assertion form and the band are UNCHANGED and the
 * pin is NEVER relaxed.
 */
const RED_TOP_SPEED_MOMENTUM_DEPTH_RAD = 0.44745539952559543;

/** The shallow slide the low-speed lane must stay at or below (rad). */
const RED_SHALLOW_SLIDE_RAD = 0.2;

/** The committed slip the 14Q chase-response contracts measure (rad). */
const RED_CHASE_COMMITTED_SLIP_RAD = -0.3;

/** The share of the chase target the 14Q chase contract measures. */
const RED_CHASE_TARGET_FRACTION = 0.5;

/** The 14Q equilibrium lane length (ticks at 60 Hz = 60 s). */
const RED_EQUILIBRIUM_TICKS = 3600;

/** The 14Q equilibrium convergence window (ticks at 60 Hz = 10 s). */
const RED_EQUILIBRIUM_WINDOW_TICKS = 600;

/**
 * The 14Q equilibrium convergence tolerance (km/h over the window).
 *
 * The MEASURED current-model change over the last 600 ticks of the
 * 3600-tick lane is 3.527265590719253e-9 km/h, so the tolerance clears
 * the measured convergence by a factor of 28.
 */
const RED_EQUILIBRIUM_CONVERGENCE_KMH = 1e-7;

/**
 * The RE-PINNED EQUILIBRIUM speed deficit (km/h) - the AC-087 number.
 *
 * BASIS (the AC-089 re-baseline ledger row P-T-2 (the equilibrium
 * deficit), Step 14T, slice `P1-turnrate-repin`): the OLD baseline was
 * `31.878446370992663` km/h (the settled deficit of the Step 14S
 * momentum model). The MEASURED settled deficit of the post-14T
 * turn-rate model is `31.65286497383213` km/h (the 3000-tick reading is
 * `31.652864970304865` km/h, so the CONVERGENCE guard holds; the 600-tick
 * transient reads `31.322144698677302` km/h, so the ANTI-TRANSIENT guard
 * holds). The deficit SHRINKS AGAIN because the 14T mild speed factor
 * softens the USER turn rate at the top speed, so the lane commits LESS
 * slide and loses LESS speed. The literal is DELETED and REPLACED
 * one-for-one: the BAND below is NOT widened.
 */
const RED_EQUILIBRIUM_DEFICIT_KMH = 31.65286497383213;

/**
 * The 14Q equilibrium band (km/h) - UNCHANGED by the Step 14S and the
 * Step 14T re-pins.
 *
 * The band absorbs the float64 noise of a 3600-tick lane; it is never
 * wide enough to admit the transient reading (0.33072027515482816 km/h
 * away on the re-pinned turn-rate model).
 */
const RED_EQUILIBRIUM_DEFICIT_BAND_KMH = 0.002;

/**
 * The one-tick chase RESPONSE FRACTION of the chase target.
 *
 * The chase target is `tan(committedSlip) * speed` (the lateral velocity
 * the committed slide implies); the fraction is the share of that target
 * the lateral velocity reaches in ONE tick. The MEASURED current-model
 * fraction is 0.26555632807026885 at EVERY speed (the speed-blind
 * response tau).
 */
const chaseResponseFraction = (
  speedMps: number,
  committedSlipRad: number,
): number => {
  const input = makeStepInput({
    steer: 1,
    speed: speedMps,
    speedPercent: speedMps / SCN_DRIFT_01_SPEED_MPS,
    lateralVelocity: 0,
    slipAngle: committedSlipRad,
  });
  const lateralVelocity = computeNextLateralVelocity(
    makeCommittedState(committedSlipRad, 1),
    input,
  );
  const target = Math.abs(Math.tan(committedSlipRad) * speedMps);
  return Math.abs(lateralVelocity) / target;
};

/**
 * The number of ticks the chase needs to reach a share of its target.
 *
 * The chase is driven from a standstill lateral velocity, so the count is
 * the RESPONSE TIME of the committed chase at that speed.
 */
const chaseTicksToFraction = (
  speedMps: number,
  committedSlipRad: number,
  fraction: number,
): number => {
  const target = Math.abs(Math.tan(committedSlipRad) * speedMps);
  let lateralVelocity = 0;
  for (let tick = 1; tick <= 600; tick += 1) {
    const input = makeStepInput({
      steer: 1,
      speed: speedMps,
      speedPercent: speedMps / SCN_DRIFT_01_SPEED_MPS,
      lateralVelocity,
      slipAngle: committedSlipRad,
    });
    lateralVelocity = computeNextLateralVelocity(
      makeCommittedState(committedSlipRad, 1),
      input,
    );
    if (Math.abs(lateralVelocity) >= fraction * target) {
      return tick;
    }
  }
  return Number.POSITIVE_INFINITY;
};

/**
 * The 14Q SERVICE lane helper: the pinned ROAD-ONLY drift lane.
 *
 * The equilibrium deficit is a SERVICE-level quantity (the speed scrub is
 * applied by the service), so the 14Q contract drives the real step
 * service: the road-only resolver, the full throttle and the full LEFT
 * steer, from the arcade top speed.
 */
const runServiceLane = (
  ticks: number,
  input: PhysicsStepInput,
): PhysicsCarState => {
  const surfaceResolver = (): PhysicsSurfaceType => 'road';
  let state: PhysicsCarState = {
    ...createInitialPhysicsCarState(42),
    speed: NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
  };
  for (let tick = 0; tick < ticks; tick += 1) {
    state = stepPhysics(state, input, surfaceResolver);
  }
  return state;
};

/**
 * The 14Q speed deficit of the pinned lane (km/h).
 *
 * The deficit is the STRAIGHT-LINE speed MINUS the DRIFTING speed: the
 * forward speed the sliding tires cannot hold. Both lanes run the same
 * number of ticks from the same entry speed, so the difference is the
 * drift scrub and nothing else.
 */
const equilibriumDeficitKmh = (ticks: number): number => {
  const straight = runServiceLane(ticks, { throttle: 1, brake: 0, steer: 0 });
  const drifted = runServiceLane(ticks, { throttle: 1, brake: 0, steer: 1 });
  return (straight.speed - drifted.speed) * 3.6;
};

/**
 * Counts the occurrences of the dead hold-ticks constant in the
 * NON-TEST physics sources.
 *
 * The scan covers the model home directory
 * (`examples/neatRacing/physics`) and EXCLUDES the `*.test.ts` files, so
 * the constant string in THIS file can never inflate the count. A count
 * of ONE is the DEAD constant (the definition with no reader); a count of
 * ZERO is the DELETED constant; a count of TWO or more is the WIRED
 * constant.
 */
const countHoldConstantOccurrences = (): number => {
  const directory = join(process.cwd(), 'examples', 'neatRacing', 'physics');
  const files = readdirSync(directory).filter(
    (name) => name.endsWith('.ts') && !name.endsWith('.test.ts'),
  );
  let occurrences = 0;
  for (const name of files) {
    const source = readFileSync(join(directory, name), 'utf8');
    occurrences +=
      source.split('NEATRACING_DRIFT_COUNTERSTEER_HOLD_TICKS').length - 1;
  }
  return occurrences;
};

describe('drift model constants (14N re-pinned envelope)', () => {
  // [RED-G1]
  it('AC-087: the committed slide is bounded by the re-pinned 0.45 rad envelope (G-1)', () => {
    // The re-pin: the old bound was pi/2 (1.5708 rad). The 14N slide is a
    // BUDGET of 0.45 rad (25.8 deg), not a quarter turn.
    expect(NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD).toBe(0.45);

    // The 7c COUPLING: the proxy saturates at MAX_SLIP * |GAIN - 1|, so the
    // 0.20 rad prominence floor is only reachable when the bound is at least
    // 0.40 rad at GAIN = 0.45.
    expect(NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD).toBeGreaterThanOrEqual(0.4);
  });

  // [RED-G1-POSE-CONSTANTS]
  it('AC-087: the pose spring carries the pinned 14N coefficients (merge 7a)', () => {
    // omega = 18 rad/s (stiffness 324), zeta = 0.35 (damping 12.6) and the
    // in-band articulation slew of 20 per s.
    expect(NEATRACING_DRIFT_POSE_STIFFNESS_RAD_PER_S2).toBe(324);
    expect(NEATRACING_DRIFT_POSE_DAMPING_RAD_PER_S).toBe(12.6);
    expect(NEATRACING_DRIFT_POSE_ARTICULATION_SLEW_PER_S).toBe(20);
  });

  // [RED-G1-POSE-CONSTANTS-2]
  it('AC-087: the articulation coefficients carry the pinned 14N values (merge 7a)', () => {
    // The arm slip gates the articulation term; the steer yaw is the term's
    // magnitude.
    expect(NEATRACING_DRIFT_POSE_STEER_YAW_FULL_SLIP_RAD).toBe(0.01);
    expect(NEATRACING_DRIFT_POSE_STEER_YAW_RAD).toBe(0.08);
  });

  // [RED-G1-GRIP-THRESHOLD]
  it('AC-087: the grip falloff threshold moves with the re-pinned bound (G-3)', () => {
    // At the new max slip 0.45 the slip demand is 0.45 - 0.20 = 0.25 and the
    // grip budget is 1/(1 + 2 * 0.25) = 0.667 < 0.7. The OLD threshold 0.25
    // would give 1/(1 + 2 * 0.20) = 0.714 and would BREAK that pin.
    expect(NEATRACING_DRIFT_GRIP_FALLOFF_THRESHOLD_RAD).toBe(0.2);
    expect(NEATRACING_DRIFT_GRIP_FALLOFF_RATE_PER_RAD).toBe(2);
  });

  // [RED-D1-MODEL-GAP-MIN-SPEED-CONSTANT]
  it('AC-079: the in-model speed gate keeps its named threshold', () => {
    expect(NEATRACING_DRIFT_MIN_SPEED_MPS).toBe(15);
  });

  // [RED-D1-MODEL-GAP-STEER-THRESHOLD]
  it('AC-087: the steer engagement threshold stays inside its ratified band', () => {
    // The engagement gate needs a named threshold; the plan ratifies the
    // band 0.4 - 0.6.
    expect(NEATRACING_DRIFT_STEER_ENGAGE_THRESHOLD).toBeGreaterThanOrEqual(0.4);
    expect(NEATRACING_DRIFT_STEER_ENGAGE_THRESHOLD).toBeLessThanOrEqual(0.6);
  });
});

describe('drift model published-slip ceilings (14N W3b owner contracts)', () => {
  // [W3B-PUBLISHED-SLIP-ON-ROAD]
  it('AC-087: the on-road published-slip ceiling is the DERIVED sum of the committed budget and the overshoot allowance', () => {
    // The DERIVED form, asserted against the NAMED constants - never a
    // literal. A re-pin of EITHER export moves the ceiling with it, so the
    // contract dies with the re-pin instead of silently drifting.
    expect(NEATRACING_DRIFT_PUBLISHED_SLIP_MAX_RAD).toBe(
      NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD +
        NEATRACING_DRIFT_PUBLISHED_SLIP_OVERSHOOT_RAD,
    );

    // The ceiling is an ASSERTION bound, NEVER a clip: it stays strictly
    // looser than the COMMITTED slide budget it is derived from.
    expect(NEATRACING_DRIFT_PUBLISHED_SLIP_MAX_RAD).toBeGreaterThan(
      NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD,
    );
  });

  // [W3B-PUBLISHED-SLIP-OFFROAD]
  it('AC-087: the off-road published-slip ceiling is the DECLARED value the declaration rule reproduces', () => {
    // The DECLARATION RULE: the ceiling is the smallest grid value at or
    // above the measured raw off-road peak plus the declaration margin.
    const declaredOffRoadCeiling =
      Math.ceil(
        (MEASURED_OFF_ROAD_RAW_PEAK_RAD + OFF_ROAD_DECLARATION_MARGIN_RAD) *
          OFF_ROAD_DECLARATION_STEPS_PER_RAD,
      ) / OFF_ROAD_DECLARATION_STEPS_PER_RAD;

    expect(NEATRACING_DRIFT_PUBLISHED_SLIP_OFFROAD_MAX_RAD).toBe(
      declaredOffRoadCeiling,
    );

    // The rule is NON-VACUOUS: the declared value clears the margin'd peak
    // and the grid step BELOW it does not - so the declared value is the
    // SMALLEST grid value that clears the peak, not a loose bound.
    expect(declaredOffRoadCeiling).toBeGreaterThanOrEqual(
      MEASURED_OFF_ROAD_RAW_PEAK_RAD + OFF_ROAD_DECLARATION_MARGIN_RAD,
    );
    expect(declaredOffRoadCeiling - OFF_ROAD_DECLARATION_GRID_RAD).toBeLessThan(
      MEASURED_OFF_ROAD_RAW_PEAK_RAD + OFF_ROAD_DECLARATION_MARGIN_RAD,
    );
  });
});

describe('drift model retired surface (14N GONE pins)', () => {
  // [RED-GONE-CONSTANTS]
  it('AC-079: the retired drift-rate constants are gone', () => {
    expect('NEATRACING_DRIFT_LATERAL_GRIP_RATE_PER_S' in driftModel).toBe(
      false,
    );
    expect(
      'NEATRACING_DRIFT_COUNTERSTEER_RECOVERY_RATE_PER_S' in driftModel,
    ).toBe(false);
    expect(
      'NEATRACING_DRIFT_LONGITUDINAL_TRACTION_LOSS_RATE_PER_S' in driftModel,
    ).toBe(false);
  });

  // [RED-GONE-HELPERS]
  it('AC-079: the retired drift helpers are gone (G-5 / G-6 / G-7)', () => {
    expect('computeDriftYawOffset' in driftModel).toBe(false);
    expect('computeOutwardPush' in driftModel).toBe(false);
    expect('computeLongitudinalTractionLoss' in driftModel).toBe(false);
  });

  // [RED-GONE-HELPERS-2]
  it('AC-047: the retired state helpers are gone', () => {
    expect('stepDriftModel' in driftModel).toBe(false);
    expect('createZeroDriftState' in driftModel).toBe(false);
  });

  // [RED-GONE-HELPERS-3]
  it('AC-079: the renamed helpers do not keep their old names (G-2 / G-4)', () => {
    expect('computeTractionFactor' in driftModel).toBe(false);
    expect('computeCountersteerDelta' in driftModel).toBe(false);
  });

  // [RED-J18-PRESERVE]
  it('AC-079: the hard grip floor stays gone', () => {
    expect('NEATRACING_DRIFT_MIN_GRIP_FACTOR' in driftModel).toBe(false);
    expect('computeGripFalloff' in driftModel).toBe(false);
  });
});

describe('drift model grip budget and speed scrub (14N G-2 / G-3 / G-5 / G-6)', () => {
  // [RED-14Q-1-GRIP-SPEED-DEPENDENT]
  it('AC-087: the traction law DEPENDS ON THE SPEED: the severity is strictly greater at a higher speed at the SAME slide (RED-1)', () => {
    // The RE-OPENED pin. The DELETED [RED-G2] pin asserted the OPPOSITE
    // (the speed-blind law):
    //     expect(computeGripBudgetFactor(0.9, 1)).toBe(computeGripBudgetFactor(0.9, 0.2));
    // The MEASURED current-model value of the deleted pin is
    // 0.4166666666666667 at BOTH 0.9 rad and 1 / 0.2 speed, so the
    // speed-blind budget is exactly what the model produces today.
    //
    // The 14Q law is PROPORTIONAL to the speed: the same slide at a higher
    // speed is a HARDER traction state (AC-088: the severity is 0 on a
    // straight line and 1 at the full slide, so a higher speed at the same
    // slide is a larger share of the traction that is lost). The severity
    // is the ONE shared scalar the service publishes, so this contract and
    // the RED-5 identity in the service test file read the SAME law.
    //
    // The NON-VACUITY GUARD comes FIRST: the severity must exist before it
    // can be compared, and it must be a finite number.
    expect(typeof computeTractionSeverity).toBe('function');

    const low = computeTractionSeverity(RED_SHALLOW_SLIDE_RAD, 20);
    const high = computeTractionSeverity(RED_SHALLOW_SLIDE_RAD, 80);

    expect(Number.isFinite(low)).toBe(true);
    expect(high).toBeGreaterThan(low);
  });

  // [RED-G3]
  it('AC-079: a sliding tire still gives up more grip than the old 30% cap allowed (G-3)', () => {
    // The argument is re-based to the RE-PINNED max slip (0.45 rad): 0.8 rad
    // is no longer a legal slip. The 5 percent margin of the old pin is kept.
    expect(
      computeGripBudgetFactor(NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD, 1),
    ).toBeLessThan(0.7);
    expect(
      computeGripBudgetFactor(NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD, 1),
    ).toBeGreaterThan(0);
  });

  // [RED-G5-SPEED-SCRUB]
  it('AC-079: the longitudinal speed scrub is zero on a straight line and grows with the slide (G-5)', () => {
    // The re-authored replacement of the retired `computeLongitudinalTractionLoss`
    // pin: the law is re-homed onto the saturating SPEED SCRUB.
    const straight = computeLongitudinalSpeedScrub(0, 60, 0.67);
    const slightSlide = computeLongitudinalSpeedScrub(0.1, 60, 0.67);
    const bigSlide = computeLongitudinalSpeedScrub(0.6, 60, 0.67);

    expect(straight).toBe(0);
    expect(slightSlide).toBeGreaterThan(0);
    expect(bigSlide).toBeGreaterThan(slightSlide);
  });

  // [RED-G5-SPEED-SCRUB-2]
  it('AC-079: the longitudinal speed scrub scales with the speed (G-5)', () => {
    const slow = computeLongitudinalSpeedScrub(0.4, 20, 0.22);
    const fast = computeLongitudinalSpeedScrub(0.4, 80, 0.9);

    expect(fast).toBeGreaterThan(slow);
  });

  // [RED-G6-COMMITTED-CHASE]
  it('AC-047: the committed chase replaces the retired outward push (G-6)', () => {
    // The re-authored replacement of the retired `computeOutwardPush` pin:
    // the 14N model has NO outward push - stage 3 is the committed chase, and
    // the chase tracks the COMMITTED slip, not the yaw rate.
    const input = makeStepInput({
      steer: 1,
      speed: SCN_DRIFT_01_SPEED_MPS,
      speedPercent: 1,
      lateralVelocity: 0,
      slipAngle: -0.3,
    });
    const chased = computeNextLateralVelocity(
      makeCommittedState(-0.3, 1),
      input,
    );
    const uncommitted = computeNextLateralVelocity(
      makeCommittedState(0, 0),
      input,
    );

    // A LEFT commitment chases to the LEFT, bounded by tan(slip) * speed.
    expect(chased).toBeLessThan(0);
    expect(Math.abs(chased)).toBeLessThanOrEqual(
      Math.abs(Math.tan(-0.3)) * SCN_DRIFT_01_SPEED_MPS + 1e-9,
    );
    // With no committed slip there is no lateral velocity at all.
    expect(uncommitted).toBe(0);
  });
});

describe('drift model commitment and countersteer (14N G-4 / D1 model gaps)', () => {
  // [RED-G4]
  it('AC-047: the countersteer authority unwinds the slide against the latched turn sign (G-4)', () => {
    const run = runDriftScenario();

    // The build committed a LEFT slide (a negative slip under the production
    // convention) and latched the turn sign.
    expect(run.slipAtHold).toBeLessThan(0);
    expect(run.turnSignAtHold).toBe(1);

    // The countersteer UNWINDS the slide - it never deepens it and it never
    // mirrors it (the old C5 defect drove the slip to +0.96 rad).
    expect(Math.abs(run.slipAfter20Counter)).toBeLessThan(
      Math.abs(run.slipAtHold),
    );
    expect(run.maxSlipAfterCounterOnset).toBeLessThanOrEqual(0);
  });

  // [RED-G4-HELPER-RENAME]
  it('AC-079: the countersteer authority is exported under its new name (G-4)', () => {
    // The old `computeCountersteerDelta` REMOVED lateral velocity against a
    // `steer * sign(slip)` gate - the exact C5 defect. The new helper ADDS
    // authority toward zero with a turn-sign gate, in m/s^2.
    expect(typeof computeCountersteerAuthority).toBe('function');
  });

  // [RED-14Q-3B-MIN-SPEED-SOFTENED]
  it('AC-089: below the in-model speed gate the commitment is SOFTENED into a proportional ramp, not frozen by a hard cliff (RED-3)', () => {
    // The RE-OPENED pin. The DELETED [RED-D1-MODEL-GAP-MIN-SPEED] pin
    // asserted the HARD CLIFF:
    //     expect(run.committed.turnSign).toBe(0);
    //     expect(run.committed.slip).toBe(0);
    //     expect(run.lateralVelocity).toBe(0);
    // The MEASURED current-model behaviour below the 15 m/s gate is that
    // exact zero triple (the commitment never latches and the lateral
    // velocity is EXACTLY 0 at 5, 10 and 12 m/s).
    //
    // The 14Q form (a RED decision): the gate is SOFTENED, never removed.
    // Below the gate the COMMITMENT is no longer frozen - the proportional
    // ramp produces a non-zero committed slip - while the VISIBLE slide
    // stays exactly zero (the lateral velocity and the published slip
    // remain exactly 0, so the service-side MIN_SPEED gate pin keeps
    // holding). The turn sign is NOT asserted here: it is an
    // implementation detail of the ramp, and the latch-release law
    // (RED-4) owns it.
    const belowGate = NEATRACING_DRIFT_MIN_SPEED_MPS - 5;
    const run = runCommittedTicks(SCN_DRIFT_01_BUILD_TICKS, 1, belowGate);
    const topSpeedRun = runCommittedTicks(
      SCN_DRIFT_01_BUILD_TICKS,
      1,
      SCN_DRIFT_01_SPEED_MPS,
    );

    // The NON-VACUITY GUARD comes FIRST: the softened gate must produce a
    // REAL slide, not a different flavour of zero.
    expect(Math.abs(run.committed.slip)).toBeGreaterThan(0);
    // The visible slide stays exactly zero below the gate.
    expect(run.lateralVelocity).toBe(0);
    // The gate is SOFTENED, not removed: the below-gate slide stays
    // shallower than the top-speed slide.
    expect(Math.abs(run.committed.slip)).toBeLessThan(
      Math.abs(topSpeedRun.committed.slip),
    );
  });

  // [RED-D1-MODEL-GAP-RE-ARM]
  it('AC-047: a released latch needs the re-arm token, so an opposite steer alone can never re-latch (RB-1)', () => {
    // Phase 1: commit a LEFT drift.
    const build = runCommittedTicks(
      SCN_DRIFT_01_BUILD_TICKS,
      1,
      SCN_DRIFT_01_SPEED_MPS,
    );
    expect(build.committed.turnSign).toBe(1);

    // Phase 2: the countersteer unwinds the slide to the hard zero, which
    // RELEASES the latch and CLEARS the re-arm token.
    const unwound = runCommittedTicks(
      SCN_DRIFT_01_COUNTER_TICKS,
      -1,
      SCN_DRIFT_01_SPEED_MPS,
      {
        start: {
          committed: build.committed,
          pose: createZeroDriftChannelState().pose,
        },
        startLateralVelocity: build.lateralVelocity,
      },
    );
    expect(unwound.committed.slip).toBe(0);
    expect(unwound.committed.turnSign).not.toBe(-1);

    // Phase 3: the opposite steer CONTINUES - and because the re-arm token
    // was cleared at the release, it can never latch a mirrored drift.
    const continued = runCommittedTicks(
      SCN_DRIFT_01_COUNTER_TICKS,
      -1,
      SCN_DRIFT_01_SPEED_MPS,
      {
        start: {
          committed: unwound.committed,
          pose: createZeroDriftChannelState().pose,
        },
        startLateralVelocity: unwound.lateralVelocity,
      },
    );
    expect(continued.committed.turnSign).not.toBe(-1);
    expect(continued.committed.slip).toBe(0);
  });

  // [RED-D1-MODEL-GAP-TURN-SIGN]
  it('AC-047: the latched turn sign is the TURN direction, never sign(slip) (D1 model gap)', () => {
    const run = runDriftScenario();

    // A LEFT turn (steer +1) latches +1 while the slip is NEGATIVE, so the
    // turn sign can never be sign(slip).
    expect(run.turnSignAtHold).toBe(1);
    expect(run.slipAtHold).toBeLessThan(0);

    // The countersteer authority NEVER flips the latched turn sign.
    expect(run.turnSignAfterCounter).not.toBe(-1);
  });

  // [RED-D1-MODEL-GAP-POSE-SIGN]
  it('AC-087: the pose target follows +committedSlip, not -committedSlip (D1 model gap)', () => {
    const run = runDriftScenario();

    // The upstream model used `-committedSlip` in the pose target, which is a
    // SIGN ERROR: the pose and the committed slip must share a sign, so a
    // LEFT turn yaws the body LEFT.
    expect(run.poseAtHold * run.slipAtHold).toBeGreaterThan(0);
    expect(run.poseAtHold).toBeLessThan(0);
  });
});

describe('drift pose spring feel contract (AC-087, merge 7a)', () => {
  // [RED-7A-POP]
  it('AC-087: the far-side pop is preserved at the pinned pose point (merge 7a)', () => {
    const run = runDriftScenario();

    // The pop is a CORPUS contract with a stated MINIMUM (D6.7a): a future
    // "tidy-up" that raises zeta to 0.45 or omega to 24 must not silently
    // halve it. Measured 2.22 deg at slew 20; the floor is 2.0 deg.
    expect(run.popDeg).toBeGreaterThanOrEqual(2.0);
    expect(Number.isFinite(run.popDeg)).toBe(true);
  });

  // [RED-7A-SETTLE]
  it('AC-087: the pose settles within the named window after the countersteer (merge 7a)', () => {
    const run = runDriftScenario();

    // Measured 33 ticks (550.0 ms). The band is 30..34 ticks (500..566.7 ms):
    // the upper edge is the merge's 550 ms plus one tick of measurement
    // slack, the lower edge stops a "tidy-up" from killing the ring-out.
    expect(run.settleTicks).toBeGreaterThanOrEqual(30);
    expect(run.settleTicks).toBeLessThanOrEqual(34);
  });

  // [RED-7A-STEP]
  it('AC-087: the per-tick pose-target step stays inside the slew bound (merge 7a)', () => {
    const run = runDriftScenario();

    // Measured 2.13 deg/tick; the arcade's own raw-steer term steps
    // -4.65 deg/tick at 49.8 m/s and 9.17 deg/tick at 89 m/s, so the slew
    // bound is the whole point of the amendment. Ceiling 2.20 deg (3.3%
    // headroom).
    expect(run.maxStepDeg).toBeLessThanOrEqual(2.2);

    // The SLEW-INDEPENDENT half, exactly: the articulation term can never
    // step the target by more than ARTICULATION_RAD * SLEW * dt, at ANY
    // speed.
    expect(run.maxArticulationStepDeg).toBeLessThanOrEqual(
      NEATRACING_DRIFT_POSE_STEER_YAW_RAD *
        NEATRACING_DRIFT_POSE_ARTICULATION_SLEW_PER_S *
        SCN_DRIFT_01_DT_S *
        (180 / Math.PI),
    );
  });

  // [RED-7A-POINT]
  it('AC-087: the three numbers are pinned at ONE point, not independently (merge 7a)', () => {
    // The anti-teardown guard: the pop, the settle and the step are read from
    // a SINGLE run of the SINGLE pinned point. A green pop bought with a
    // different (omega, zeta, art, arm, slew) is a contract violation.
    const run = runDriftScenario(POSE_POINT);

    expect(run.point).toEqual(POSE_POINT);
    expect(run.popDeg).toBeGreaterThanOrEqual(2.0);
    expect(run.settleTicks).toBeLessThanOrEqual(34);
    expect(run.maxStepDeg).toBeLessThanOrEqual(2.2);
  });
});

describe('drift pose spring straight-line identity (AC-087, merge 7b)', () => {
  // [RED-7B-UNIT]
  it('AC-087: a straight line carries EXACTLY zero visible yaw (merge 7b)', () => {
    // The articulation is slip-gated, so on a straight line every factor is
    // exactly 0 and the float arithmetic is exact: the identity is EXACT,
    // not a tolerance (D6.7). A steering input must not create a yaw either.
    let spring: DriftPoseSpringState = {
      pose: 0,
      poseRate: 0,
      articulation: 0,
    };
    for (let tick = 0; tick < 600; tick += 1) {
      spring = stepDriftPoseSpring(
        spring,
        makePoseInput({
          slipAngle: 0,
          steer: 1,
          speedPercent: 1,
          dt: SCN_DRIFT_01_DT_S,
        }),
      );
    }

    expect(spring.pose).toBe(0);
    expect(spring.poseRate).toBe(0);
    expect(spring.articulation).toBe(0);
  });
});

describe('drift slide prominence (AC-087, merge 7c / 7f)', () => {
  // [RED-7C-FLOOR]
  it('AC-087: the drift slide prominence reaches the named floor at a sustained full-steer turn (merge 7c)', () => {
    // The floor is a NAMED EXPORTED CONSTANT (the AC-087 wording).
    expect(NEATRACING_DRIFT_MIN_SLIDE_PROXY_RAD).toBe(0.2);

    const run = runDriftScenario();

    // The MEASURED proxy (the deterministic twin of the browser proxy).
    expect(run.maxProxyRad).toBeGreaterThanOrEqual(
      NEATRACING_DRIFT_MIN_SLIDE_PROXY_RAD,
    );
  });

  // [RED-7C-SUSTAIN]
  it('AC-087: the slide is SUSTAINED, not a spike (merge 7c)', () => {
    const run = runDriftScenario();

    // The proxy must stay at or above the floor for a sustained run inside
    // the 180-tick build phase, not for a single tick.
    expect(run.longestRunAboveFloorTicks).toBeGreaterThanOrEqual(120);
  });

  // [RED-7C-PROXY]
  it('AC-087: the floor is a proxy floor, not a slip floor (merge 7c)', () => {
    // The proxy identity, pinned so the two quantities can never be confused:
    //   proxy = |slipAngle| * |GAIN - 1|
    expect(computeDriftSlideProxyRad(0.09877710114921623)).toBeCloseTo(
      0.05432740563206893,
      12,
    );
    expect(computeDriftSlideProxyRad(0)).toBe(0);
  });

  // [RED-7F-ENGAGEMENT]
  it('AC-087: the drift engagement gate survives three consecutive frames (merge 7f)', () => {
    // The engagement gate threshold is UNCHANGED (0.01 rad of proxy).
    expect(NEATRACING_DRIFT_PROXY_ENGAGEMENT_MIN_RAD).toBe(0.01);
    expect(NEATRACING_DRIFT_PROXY_MIN_FRAMES).toBe(3);

    const run = runDriftScenario();

    // The gate counts CONSECUTIVE frames, not total frames: a single-frame
    // spike can never satisfy it.
    expect(run.longestRunAboveEngagementGateTicks).toBeGreaterThanOrEqual(
      NEATRACING_DRIFT_PROXY_MIN_FRAMES,
    );
  });
});

describe('drift visual yaw gain (AC-087, merge 7g)', () => {
  // [RED-7G-ANTI-NULL]
  it('AC-087: the visual yaw gain stays on the monotone branch (merge 7g)', () => {
    // The gain is RE-PINNED at 0.45 (the RED-decided FEASIBLE value - the
    // re-baseline ledger ROW 5) - the body yaw shape is retained
    // (AC-087: "NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD keeps owning the body
    // yaw").
    expect(NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD).toBe(0.45);

    // The anti-null guard: the proxy scales with |GAIN - 1|, which has a NULL
    // at GAIN = 1.0. The gain must never approach that null.
    expect(
      Math.abs(NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD - 1),
    ).toBeGreaterThanOrEqual(0.5);

    const run = runDriftScenario();

    // And the sign: the visible yaw must NOT mirror the slide. Under the
    // production convention (LEFT = negative slip) the pose target is
    // +committedSlip * GAIN, so the pose and the slip share a sign.
    expect(Math.sign(run.poseAtHold)).toBe(Math.sign(run.slipAtHold));
    expect(run.poseAtHold).toBeLessThan(0);
  });
});

describe('drift model step behavior (14N P0 / AC-047 holding pins)', () => {
  // [RED-P0-1-UNIT]
  it('AC-047: the countersteer UNWINDS the slide to the hard zero and never mirrors it', () => {
    const run = runDriftScenario();

    // The hard zero is a HELD contract: the countersteer drives the committed
    // slip to EXACTLY zero and the slide is DEAD there. The old model
    // MIRRORED to +0.9645 rad after 120 counter ticks, so the signed maximum
    // over the phase-B ticks is positive there; it must never be.
    expect(run.maxSlipAfterCounterOnset).toBeLessThanOrEqual(0);
  });

  // [RED-P0-1-UNIT-SLIDE]
  it('AC-047: the slide is still VISIBLE while the countersteer unwinds it', () => {
    const run = runDriftScenario();

    // The unwind must not be a "no drift at all" outcome: the commitment is
    // latched for the whole build phase, so the proxy is still at the floor
    // when the countersteer starts.
    expect(run.maxProxyRad).toBeGreaterThanOrEqual(
      NEATRACING_DRIFT_MIN_SLIDE_PROXY_RAD,
    );
  });

  // [RED-P0-2-UNIT]
  it('AC-047: the sustained slide stays inside the re-pinned slip envelope', () => {
    // 1200 ticks of sustained full-steer at the arcade top speed. The
    // re-pinned MAX_SLIP_ANGLE_RAD (0.45) is the envelope: the old model
    // reached 1.4282 rad.
    const channel = createZeroDriftChannelState();
    let committed: DriftCommittedState = channel.committed;
    let maxAbsSlip = 0;
    for (let tick = 0; tick < 1200; tick += 1) {
      committed = stepDriftCommitment(
        committed,
        makeStepInput({
          steer: 1,
          speed: SCN_DRIFT_01_SPEED_MPS,
          speedPercent: 1,
          dt: SCN_DRIFT_01_DT_S,
        }),
      );
      maxAbsSlip = Math.max(maxAbsSlip, Math.abs(committed.slip));
    }

    expect(maxAbsSlip).toBeLessThanOrEqual(NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD);
    expect(maxAbsSlip).toBeGreaterThan(0.2);
  });

  // [RED-P0-4]
  it('AC-047: the drift engages within the named entry-attack window', () => {
    const run = runDriftScenario();

    // The entry attack: the slide must be 63% of its hold value within 9
    // ticks (150 ms) of the steer input. A model that ramps in over 20
    // ticks reads as "the car does nothing" and then suddenly snaps.
    expect(run.ticksTo63PctOfHold).toBeLessThanOrEqual(9);
  });

  // [RED-J12-DETERMINISM]
  it('produces identical drift state for the same input across runs', () => {
    const runA = runDriftScenario();
    const runB = runDriftScenario();

    expect(serializeDriftChannel(runA.finalChannel)).toBe(
      serializeDriftChannel(runB.finalChannel),
    );
  });

  // [RED-G16-AC047A]
  it('AC-047A: high-speed sustained steering produces visible lateral slide', () => {
    // G-16: the slip band is RE-AUTHORED against the 14N API, but the
    // ASSERTION is unchanged - the new hold (0.45) is above the band, so
    // the pin still holds and must not be re-baselined.
    const highSpeedMps = 56;
    const speedPercent = highSpeedMps / 89;
    const channel = createZeroDriftChannelState();
    let committed: DriftCommittedState = channel.committed;
    let lateralVelocity = 0;
    for (let step = 0; step < 120; step += 1) {
      const input = makeStepInput({
        speed: highSpeedMps,
        speedPercent,
        steer: 1,
        dt: SCN_DRIFT_01_DT_S,
        lateralVelocity,
        slipAngle: committed.slip,
      });
      committed = stepDriftCommitment(committed, input);
      lateralVelocity = computeNextLateralVelocity(committed, input);
    }

    expect(Math.abs(lateralVelocity)).toBeGreaterThan(10);
    expect(Math.abs(committed.slip)).toBeGreaterThan(0.2);
    expect(Math.abs(committed.slip)).toBeLessThanOrEqual(
      NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD,
    );
  });

  // [RED-G17-AC047D]
  it('AC-047D: low-speed straight-line handling is unchanged', () => {
    // G-17: the byte-identity pin is UNCHANGED - the new committed state
    // fields (`slip`, `turnSign`, `reArmed`) are all exactly 0 on a straight
    // line below MIN_SPEED_MPS, so the serialised shape is byte-identical to
    // the zero channel.
    //
    // CANONICAL ANCHOR: the production byte-identity for AC-047D is the
    // serializer in `physics.step.service.test.ts` (another agent owns that
    // file). This is the PURE-MODEL equivalent against the same
    // production-shape state.
    const lowSpeedMps = 10;
    const speedPercent = lowSpeedMps / 89;
    let channel = createZeroDriftChannelState();
    let lateralVelocity = 0;
    for (let step = 0; step < 120; step += 1) {
      const input = makeStepInput({
        speed: lowSpeedMps,
        speedPercent,
        steer: 0,
        dt: SCN_DRIFT_01_DT_S,
        lateralVelocity,
        slipAngle: channel.committed.slip,
      });
      const committed = stepDriftCommitment(channel.committed, input);
      const nextLateralVelocity = computeNextLateralVelocity(committed, input);
      channel = { committed, pose: channel.pose };
      lateralVelocity = nextLateralVelocity;
    }

    expect(serializeDriftChannel(channel)).toBe(
      serializeDriftChannel(createZeroDriftChannelState()),
    );
  });

  // [RED-J24-PRESERVE]
  it('computes the slip angle from the velocity difference', () => {
    // J-24 (PRESERVE): the slip-angle helper is NOT part of the 14N
    // retirement list, so its contract survives the tune. The law is the
    // `atan2` form: zero at standstill, 45 deg at a matched lateral and
    // longitudinal speed, and NEGATIVE for a LEFT slide.
    expect(computeSlipAngle(0, 10)).toBe(0);
    expect(computeSlipAngle(10, 0)).toBe(0);
    expect(computeSlipAngle(10, 10)).toBeCloseTo(Math.PI / 4, 10);
    expect(computeSlipAngle(-10, 10)).toBeCloseTo(-Math.PI / 4, 10);
  });
});

/**
 * The PROPORTIONAL traction-law contracts (Step 14Q, slice
 * `P1-drift-law-red`; the plan's RED-1, RED-2, RED-3, RED-4, RED-6 and
 * RED-7 acceptance criteria).
 *
 * RED-1 (the law is NOT speed-blind): the ratified speed-blind pin is
 * REPLACED IN PLACE above (the [RED-G2] block) and this block adds the
 * SPEED-AXIS contract: the law reads the ABSOLUTE speed (the
 * post-traction car.speed the service publishes), NEVER the speed
 * percent. The discriminating lane is a same-percent / different-speed
 * pair: a law that read the percent would produce the SAME committed
 * depth on both lanes.
 *
 * RED-2 (the slide DEPTH is speed-proportional): the committed slide is
 * deep at the arcade top speed and shallow at a low speed, so the OLD
 * "the same 0.45 rad slide at 5 m/s and at 89 m/s" can never come back.
 *
 * RED-3 (the entry RAMP is speed-proportional): the time to a given
 * committed slip is not a fixed tick count at every speed. The MIN_SPEED
 * 15 m/s cliff is REPLACED IN PLACE above (the
 * [RED-D1-MODEL-GAP-MIN-SPEED] block) by the softened form.
 *
 * RED-4 (the neutral wheel RELEASES the slide): with a non-zero committed
 * slip and the steer returned to 0, the model must not take the drifting
 * branch (the `Math.sign(0) === -turnSign` defect), the slide must unwind
 * and the re-arm token must become reachable again. The re-arm law is a
 * REAL law: the dead hold-ticks constant is either WIRED or DELETED,
 * never left unread.
 *
 * RED-6 (the slip RESPONSE is not a fixed time constant): the committed
 * chase response rate scales with the speed, so the lateral velocity
 * reaches a given fraction of its target sooner at a higher speed.
 *
 * RED-7 (the EQUILIBRIUM, never the transient): the speed deficit is
 * measured at the EQUILIBRIUM (behind a convergence criterion) and never
 * at the 600-tick transient reading.
 */
describe('the PROPORTIONAL traction law (Step 14Q, slice P1-drift-law-red)', () => {
  // [RED-14Q-1-LAW-SPEED-AXIS]
  it('AC-088: the traction law reads the ABSOLUTE speed, not the speed percent (RED-1)', () => {
    // The discriminating lane: the SAME speed percent (0.5) at two
    // different absolute speeds (80 m/s and 20 m/s). A law that read the
    // percent would produce the SAME committed depth on both lanes; a law
    // that reads the absolute speed produces a STRICTLY deeper slide in
    // the faster lane.
    //
    // The percent is forced through the `speedPercent` override, so the
    // lane really is a same-percent lane and the percent can never leak
    // in from the speed.
    const fastLane = runCommittedTicks(SCN_DRIFT_01_BUILD_TICKS, 1, 80, {
      speedPercent: 0.5,
    });
    const slowLane = runCommittedTicks(SCN_DRIFT_01_BUILD_TICKS, 1, 20, {
      speedPercent: 0.5,
    });

    // The NON-VACUITY GUARD comes FIRST: both lanes must really slide.
    expect(Math.abs(fastLane.committed.slip)).toBeGreaterThan(0);
    expect(Math.abs(slowLane.committed.slip)).toBeGreaterThan(0);
    // The law reads the ABSOLUTE speed.
    expect(Math.abs(fastLane.committed.slip)).toBeGreaterThan(
      Math.abs(slowLane.committed.slip),
    );
  });

  // [RED-14Q-2-SPEED-PROPORTIONAL-DEPTH]
  it('AC-087: the committed slide DEPTH is SPEED-PROPORTIONAL: deep at the top speed, shallow at a low speed (RED-2)', () => {
    // The MEASURED momentum-model depth at the top speed is the MOMENTUM
    // demand of that lane (0.44745539952559543 rad) - the demand can
    // NEVER reach the 0.45 rad depth budget, so the pin moves with the
    // law. The low-speed lane stays BELOW the shallow band, so the
    // SPEED-PROPORTIONAL relation this pin exists for is preserved.
    const topSpeedRun = runCommittedTicks(
      SCN_DRIFT_01_BUILD_TICKS,
      1,
      SCN_DRIFT_01_SPEED_MPS,
    );
    const lowSpeedRun = runCommittedTicks(
      SCN_DRIFT_01_BUILD_TICKS,
      1,
      RED_LOW_SPEED_MPS,
    );

    // The NON-VACUITY GUARD comes FIRST: the low-speed lane must really
    // slide (a zero slide would satisfy the ceiling vacuously).
    expect(Math.abs(lowSpeedRun.committed.slip)).toBeGreaterThan(0);
    // The top speed reaches the RE-PINNED momentum depth inside the
    // sustain window.
    expect(Math.abs(topSpeedRun.committed.slip)).toBeGreaterThanOrEqual(
      RED_TOP_SPEED_MOMENTUM_DEPTH_RAD,
    );
    // The low-speed lane stays at or below the shallow band.
    expect(Math.abs(lowSpeedRun.committed.slip)).toBeLessThanOrEqual(
      RED_SHALLOW_SLIDE_RAD,
    );
  });

  // [RED-14Q-4A-NEUTRAL-RELEASE]
  it('AC-087: the NEUTRAL wheel RELEASES the slide: the committed slip and the lateral velocity unwind instead of staying frozen (RED-4)', () => {
    // The MEASURED current-model behaviour: with the steer returned to 0
    // the slide stays FROZEN at -0.45 rad (the `Math.sign(0) === -turnSign`
    // defect routes the neutral wheel into the drifting branch) and the
    // lateral velocity stays at -42.99190083987546 m/s.
    const build = runCommittedTicks(
      SCN_DRIFT_01_BUILD_TICKS,
      1,
      SCN_DRIFT_01_SPEED_MPS,
    );
    const released = runCommittedTicks(
      SCN_DRIFT_01_COUNTER_TICKS,
      0,
      SCN_DRIFT_01_SPEED_MPS,
      {
        start: {
          committed: build.committed,
          pose: createZeroDriftChannelState().pose,
        },
        startLateralVelocity: build.lateralVelocity,
      },
    );

    // The NON-VACUITY GUARD comes FIRST: the build lane must really be a
    // slide.
    expect(Math.abs(build.committed.slip)).toBeGreaterThan(0);
    // The released slip is SHALLOWER than the built slip: the slide
    // unwinds.
    expect(Math.abs(released.committed.slip)).toBeLessThan(
      Math.abs(build.committed.slip),
    );
    // The lateral velocity unwinds with it.
    expect(Math.abs(released.lateralVelocity)).toBeLessThan(
      Math.abs(build.lateralVelocity),
    );
  });

  // [RED-14Q-4B-RELEASE-LATCH]
  it('AC-089: the released slide RELEASES the latch: the turn sign returns to 0 and the re-arm token becomes reachable again (RED-4)', () => {
    // The MEASURED current-model behaviour: the turn sign stays latched
    // at 1 and the slide never unwinds, so the release branch (the one
    // that clears the turn sign and re-arms the token) is never reached.
    const build = runCommittedTicks(
      SCN_DRIFT_01_BUILD_TICKS,
      1,
      SCN_DRIFT_01_SPEED_MPS,
    );
    const released = runCommittedTicks(
      SCN_DRIFT_01_COUNTER_TICKS,
      0,
      SCN_DRIFT_01_SPEED_MPS,
      {
        start: {
          committed: build.committed,
          pose: createZeroDriftChannelState().pose,
        },
        startLateralVelocity: build.lateralVelocity,
      },
    );

    // The NON-VACUITY GUARD comes FIRST: the build lane must really have
    // latched a LEFT drift.
    expect(build.committed.turnSign).toBe(1);
    // The latch is RELEASED: the turn sign returns to 0.
    expect(released.committed.turnSign).toBe(0);
    // The re-arm token is reachable again (a fresh engagement is allowed).
    expect(released.committed.reArmed).toBe(true);
  });

  // [RED-14Q-4C-HOLD-CONSTANT]
  it('AC-089: the re-arm law is a REAL law: the hold-ticks constant is WIRED or DELETED, never left unread (RED-4)', () => {
    // The MEASURED current-model count is EXACTLY ONE: the definition
    // (`NEATRACING_DRIFT_COUNTERSTEER_HOLD_TICKS = 8`, drift.model.ts:251)
    // with no reader anywhere in the physics sources - the dead constant.
    //
    // The scan covers the NON-TEST physics sources only, so the constant
    // string in this test file can never inflate the count. ZERO is the
    // DELETED constant, TWO or more is the WIRED constant.
    expect(countHoldConstantOccurrences()).not.toBe(1);
  });

  // [RED-14Q-6A-CHASE-RATE]
  it('AC-087: the committed chase RESPONSE rate scales with the SPEED: the one-tick share of the target is larger at a higher speed (RED-6)', () => {
    // The MEASURED current-model one-tick fraction is
    // 0.26555632807026885 at EVERY speed (20 / 25 / 30 / 45 / 89 m/s):
    // the response tau is a fixed 0.045 s, so only the chase TARGET is
    // speed-proportional.
    const slowFraction = chaseResponseFraction(
      RED_LOW_SPEED_MPS,
      RED_CHASE_COMMITTED_SLIP_RAD,
    );
    const fastFraction = chaseResponseFraction(
      SCN_DRIFT_01_SPEED_MPS,
      RED_CHASE_COMMITTED_SLIP_RAD,
    );

    // The NON-VACUITY GUARD comes FIRST: both fractions must be real
    // shares of the target.
    expect(slowFraction).toBeGreaterThan(0);
    expect(fastFraction).toBeGreaterThan(0);
    // The faster lane moves a LARGER share of the target in one tick.
    expect(fastFraction).toBeGreaterThan(slowFraction);
  });

  // [RED-14Q-6B-CHASE-TIME]
  it('AC-087: the committed chase reaches a given share of its target SOONER at a higher speed (RED-6)', () => {
    // The MEASURED current-model tick count is 3 at EVERY speed.
    const slowTicks = chaseTicksToFraction(
      RED_LOW_SPEED_MPS,
      RED_CHASE_COMMITTED_SLIP_RAD,
      RED_CHASE_TARGET_FRACTION,
    );
    const fastTicks = chaseTicksToFraction(
      SCN_DRIFT_01_SPEED_MPS,
      RED_CHASE_COMMITTED_SLIP_RAD,
      RED_CHASE_TARGET_FRACTION,
    );

    // The NON-VACUITY GUARD comes FIRST: both lanes must reach the share.
    expect(Number.isFinite(slowTicks)).toBe(true);
    expect(Number.isFinite(fastTicks)).toBe(true);
    // The faster lane reaches the share in FEWER ticks.
    expect(fastTicks).toBeLessThan(slowTicks);
  });

  // [RED-14Q-7-EQUILIBRIUM-DEFICIT]
  it('AC-087: the speed deficit is pinned at the EQUILIBRIUM, never at a transient reading (RED-7)', () => {
    // The pinned scenario: the road-only resolver, the full throttle, the
    // full LEFT steer, the arcade top speed as the entry speed, dt 1/60.
    //
    // The MEASURED post-14T turn-rate-model values: the 600-tick
    // TRANSIENT reads 31.322144698677302 km/h and the 3600-tick
    // asymptote reads 31.65286497383213 km/h, so the turn-rate model
    // sits ON the re-pinned equilibrium (the deficit SHRANK AGAIN
    // because the 14T mild speed factor softens the USER turn rate at
    // the top speed, so the lane commits LESS slide and loses LESS
    // speed).
    const deficit = equilibriumDeficitKmh(RED_EQUILIBRIUM_TICKS);
    const windowDeficit = equilibriumDeficitKmh(
      RED_EQUILIBRIUM_TICKS - RED_EQUILIBRIUM_WINDOW_TICKS,
    );
    const transientDeficit = equilibriumDeficitKmh(
      RED_EQUILIBRIUM_WINDOW_TICKS,
    );

    // The NON-VACUITY GUARD comes FIRST: the lane must really converge,
    // so a slow transient can never pass or fail this contract by
    // accident.
    expect(Math.abs(deficit - windowDeficit)).toBeLessThan(
      RED_EQUILIBRIUM_CONVERGENCE_KMH,
    );
    // The ANTI-TRANSIENT guard: the 600-tick reading is NOT the
    // equilibrium (it stays far outside the band).
    expect(
      Math.abs(transientDeficit - RED_EQUILIBRIUM_DEFICIT_KMH),
    ).toBeGreaterThan(RED_EQUILIBRIUM_DEFICIT_BAND_KMH);
    // The MAIN assertion: the settled deficit is inside the band around
    // the pinned equilibrium.
    expect(Math.abs(deficit - RED_EQUILIBRIUM_DEFICIT_KMH)).toBeLessThanOrEqual(
      RED_EQUILIBRIUM_DEFICIT_BAND_KMH,
    );
  });
});

/* ------------------------------------------------------------------------- *
 * STEP 14S (slice `P1-momentum-red`): the MOMENTUM law.
 * ------------------------------------------------------------------------- */

/**
 * The MOMENTUM law: the drift is CAUSED by the momentum, not COMMANDED by
 * the wheel (AC-091, verdict-6 S-5).
 *
 * THE NAMED EXPORT (a RED DECISION of this slice - the plan's RED-1: "the
 * exact export name and the numeric window are a RED DECISION of this
 * slice"): `computeMomentumSlideDemandRad(speedMps, demandedTurnRateRadPerS,
 * availableGripShare)` - it maps (the speed, the demanded turn rate, the
 * available grip) to the momentum SLIDE DEMAND in RADIANS.
 *
 * THE AVAILABLE GRIP is a SHARE in `(0, 1]`: the shape the model already
 * uses for the grip budget (`computeGripBudgetFactor` returns a factor in
 * `(0, 1]`), so the law reads the grip the way the rest of the model reads
 * it. The full grip is exactly 1.
 *
 * THE RED FORM (the WIDENED LOCAL VIEW - the documented deviation from the
 * plan's literal MISSING-IMPORT wording): the export does not exist yet. A
 * NAMED import of a missing export is a FILE-LEVEL compile error
 * (`TS2305: Module '"./drift.model"' has no exported member 'X'`) under
 * this repo's jest configuration (`diagnostics: true`), and it MASKS every
 * sibling contract of this file - including the TURN-RATE contracts of
 * RED-2, which the plan MANDATES to be authored through a widened local
 * view in THIS SAME file, and including the PRESERVATION GUARDS, which
 * must stay observably GREEN. The law is therefore read through the SAME
 * widened local view the plan mandates for the internal drift input. NO
 * placeholder export is created and NO production file is touched by this
 * slice: the law is ABSENT, and every contract below fails on its OWN
 * assertion (with the assertion name in the failure message) instead of on
 * a file-level compile error.
 */
type MomentumSlideDemandLaw = (
  speedMps: number,
  demandedTurnRateRadPerS: number,
  availableGripShare: number,
) => number;

const momentumSlideDemandLaw = (
  driftModel as unknown as {
    computeMomentumSlideDemandRad?: MomentumSlideDemandLaw;
  }
).computeMomentumSlideDemandRad;

/**
 * Reads the momentum slide demand (rad) through the widened view.
 *
 * While the law does not exist the read yields `NaN`: every contract then
 * fails on ITS OWN assertion (the failure message names the assertion)
 * instead of on a `TypeError`, so no contract is masked by a fixture
 * error.
 *
 * @param speedMps - The tick-start speed (m/s).
 * @param demandedTurnRateRadPerS - The demanded turn rate (rad/s).
 * @param availableGripShare - The available grip share in `(0, 1]`.
 * @returns The momentum slide demand (rad), or `NaN` while the law is
 *   absent.
 */
const momentumSlideDemandRad = (
  speedMps: number,
  demandedTurnRateRadPerS: number,
  availableGripShare: number,
): number =>
  typeof momentumSlideDemandLaw === 'function'
    ? momentumSlideDemandLaw(
        speedMps,
        demandedTurnRateRadPerS,
        availableGripShare,
      )
    : Number.NaN;

/** The arcade TOP-speed lane of the momentum contracts (m/s). */
const MOMENTUM_TOP_SPEED_MPS = 89;

/**
 * The LOW-speed lane of the momentum contracts (m/s).
 *
 * 25 m/s is a REAL drifting lane (above the 15 m/s MIN_SPEED gate), so the
 * lane is never satisfied vacuously.
 */
const MOMENTUM_LOW_SPEED_MPS = 25;

/**
 * The MID-speed lane of the momentum contracts (m/s).
 *
 * 56 m/s is the speed of the ratified [RED-G16-AC047A] pin, which REQUIRES
 * a committed slide above 0.2 rad at that speed. The momentum window is
 * measured against that lane, because the pin belongs to the PRESERVE SET
 * (AC-089) and this step never re-baselines it.
 */
const MOMENTUM_MID_SPEED_MPS = 56;

/**
 * The FULL-GRIP share the momentum lanes measure.
 *
 * The grip share is the shape the model already uses for the grip budget
 * (`computeGripBudgetFactor` returns a factor in `(0, 1]`); the full grip is
 * exactly 1.
 */
const MOMENTUM_FULL_GRIP_SHARE = 1;

/**
 * The demanded turn rate the momentum lanes feed for a full steer at the
 * arcade top speed (rad/s).
 *
 * THE PINNED UNIT (Step 14S, RED-4): the demanded turn rate is a RATE
 * (rad/s), NOT the per-tick heading delta (rad/tick) the yaw law resolves
 * today - a per-tick delta and a rate are NOT interchangeable (a unit
 * mismatch is a silent factor-of-sixty error).
 *
 * THE BASIS: the yaw law builds its steer-axis rate from `steer *
 * NEATRACING_PHYSICS_STEERING_RATE * speedPercent` (rad/s); the SAME
 * product is the rate the lanes feed. At the arcade top speed the speed
 * percent is exactly 1, so the full-steer rate is exactly the named
 * steering rate - the RAW steer-axis rate, which is NOT the rate the
 * SERVICE resolves there (the service softens it with the speed factor,
 * so it resolves 1.6 rad/s at that speed).
 */
const MOMENTUM_FULL_STEER_TURN_RATE_RAD_PER_S =
  NEATRACING_PHYSICS_STEERING_RATE;

/**
 * The demanded turn rate the momentum lanes feed for a full steer at the
 * LOW-speed lane (rad/s).
 *
 * The lane feeds the RAW steer-axis rate, which scales with the speed
 * percent, so the low-speed lane demands `2 * (25 / 89)` rad/s. The
 * SERVICE resolves a softened rate at that speed (0.5404636609301664
 * rad/s) - the lanes keep the raw rate, so the measured window of this
 * step stays the one these lanes record.
 */
const MOMENTUM_LOW_SPEED_TURN_RATE_RAD_PER_S =
  (MOMENTUM_FULL_STEER_TURN_RATE_RAD_PER_S * MOMENTUM_LOW_SPEED_MPS) /
  MOMENTUM_TOP_SPEED_MPS;

/**
 * The DOUBLE of the low-speed demanded turn rate (rad/s).
 *
 * This is the "STRICTLY LARGER demanded turn rate" point of the RED-2
 * relation: the SAME speed, the SAME steer axis, a strictly larger demanded
 * turn rate.
 */
const MOMENTUM_DOUBLE_LOW_SPEED_TURN_RATE_RAD_PER_S =
  2 * MOMENTUM_LOW_SPEED_TURN_RATE_RAD_PER_S;

/**
 * The HOLD window of the LIVE-LATCH HOLD lane (ticks).
 *
 * ONE SECOND of 60 Hz ticks at the LOW-speed lane: long enough that a
 * snap-to-zero under a live latch could not hide in the transient, and
 * short enough to keep the lane fast.
 */
const MOMENTUM_LATCH_HOLD_TICKS = 60;

/**
 * The RELEASE window of the LIVE-LATCH HOLD lane (ticks).
 *
 * The COUNTERSTEER unwinds the committed slide at
 * `NEATRACING_DRIFT_EXIT_RATE_RAD_PER_S` (2.5 rad/s) - about 0.0417 rad per
 * 60 Hz tick - so the ~0.12 rad slide of the lane needs a few ticks. The
 * window is generous on purpose: the lane asserts the END state, never the
 * transient.
 */
const MOMENTUM_RELEASE_TICKS = 10;

/**
 * THE MEASURED FEASIBLE WINDOW for the momentum slide demand (rad).
 *
 * MEASURED, never carried (the plan's RED-3): the window is the set of
 * demand values the law may take at the lanes this slice measures, and
 * every bound is DERIVED from a MEASURED observable of the current model or
 * from a ratified pin. The implementing slice calibrates INSIDE this
 * window; the window is never carried as a literal into the implementation.
 *
 * 1. THE CEILING (every lane): the demand can NEVER exceed the declared
 *    depth budget `NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD` (0.45 rad) - the
 *    committed slide is clamped there today (drift.model.ts:817), so a
 *    demand above it would be MASKED by the clamp. The MEASURED depth
 *    budgets: 0.45 rad at 89 m/s, 0.31555221637866265 rad at 56 m/s,
 *    0.2535687453042825 rad at 45 m/s, 0.169045830202855 rad at 30 m/s,
 *    0.14087152516904583 rad at 25 m/s, 0.11269722013523667 rad at 20 m/s,
 *    0.09015777610818933 rad at 16 m/s, 0.0845229151014275 rad at 15 m/s;
 *    the floor is 0.02 rad below about 3.6 m/s.
 *
 * 2. THE FLOOR (the standstill and the zero-rate lanes): the demand is
 *    EXACTLY 0 at a standstill (there is no momentum at rest) and EXACTLY 0
 *    for a ZERO demanded turn rate.
 *
 * 3. THE STRICT-DEEPER LANE (25 m/s, the lane's OWN rate points
 *    0.5617977528089888 rad/s and its double 1.1235955056179776 rad/s -
 *    the RAW steer-axis rates the lane feeds, NOT the service-resolved
 *    ones, which are 0.5404636609301664 rad/s and its double
 *    1.0809273218603328 rad/s at that speed): the demand must stay inside
 *    (0, 0.14087152516904583) rad at BOTH rate points, otherwise the depth
 *    clamp of that lane would MASK the strict-deeper relation.
 *
 * 4. THE AC-047A LANE (56 m/s, the lane's OWN fallback rate
 *    1.2584269662921348 rad/s - the RAW steer-axis rate, NOT the
 *    service-resolved 1.1118759778125444 rad/s at that speed): the demand
 *    must stay ABOVE 0.2 rad, or the ratified [RED-G16-AC047A] pin (a
 *    committed slide above 0.2 rad at 56 m/s) would break - that pin
 *    belongs to the PRESERVE SET (AC-089) and this step never re-baselines
 *    it.
 *
 * THE MEASURED TENSION the implementing slice must resolve: a demand that
 * is LINEAR in BOTH the speed and the demanded turn rate and that is
 * calibrated so the reference lane (89 m/s, the lane's OWN full-steer rate
 * 2 rad/s, the full grip) reaches the depth budget (0.45 rad) yields
 * 0.45 * (56 / 89) * (1.2584269662921348 / 2) = 0.178 rad at the AC-047A
 * lane - BELOW the ratified 0.2 rad floor of (4). The service-resolved
 * rates do not lift it above that floor either: the same construct on the
 * service-resolved rates (the 89 m/s reference lane at 1.6 rad/s) yields
 * 0.45 * (56 / 89) * (1.1118759778125444 / 1.6) = 0.197 rad - still BELOW
 * the ratified 0.2 rad floor of (4). The window therefore admits only a
 * law whose demand does not fall below 0.2 rad at 56 m/s (a law with a
 * constant grip floor, or a reference demand below the depth budget, or a
 * law that is not linear in the demanded turn rate).
 */

/**
 * The WIDENED drift input of Step 14S.
 *
 * The demanded turn rate is an INTERNAL TYPE MEMBER of `DriftStepInput`
 * (an OPTIONAL one), so the contract reads it through a WIDENED LOCAL VIEW
 * (the Step 14R red-slice residual (2) lesson): a MISSING IMPORT failure is
 * UNSATISFIABLE for an internal type member, and the plan PINS the field
 * name here, never through a placeholder export.
 *
 * THE PINNED FIELD NAME: `demandedTurnRateRadPerS` - the same name and the
 * same unit (rad/s) the service contract (RED-4) pins.
 */
type MomentumDriftStepInput = DriftStepInput & {
  demandedTurnRateRadPerS: number;
};

/**
 * ONE tick of the MOMENTUM plan.
 *
 * The demanded turn rate is OPTIONAL on the plan: `undefined` OMITS the
 * field from the drift input, so the law falls back to the MIRRORED
 * steer-times-speed-percent rate (the production fallback,
 * `resolveDemandedTurnRateRadPerS`). An EXPLICIT rate - including an
 * explicit 0 - is AUTHORITATIVE.
 */
type MomentumTickPlan = {
  /** The steer axis (`+1` = LEFT, `-1` = RIGHT). */
  steer: number;
  /** The tick-start speed (m/s). */
  speed: number;
  /** The demanded turn rate (rad/s); `undefined` OMITS the field. */
  demandedTurnRateRadPerS?: number;
};

/**
 * Builds a run of IDENTICAL momentum ticks.
 *
 * The demanded turn rate is OPTIONAL: `undefined` OMITS the field from the
 * plan entries, so the drift input really carries NO demanded turn rate and
 * the production fallback takes over (the field is genuinely ABSENT, not
 * `undefined`-valued).
 *
 * @param count - The number of identical ticks.
 * @param steer - The steer axis (`+1` = LEFT, `-1` = RIGHT).
 * @param speed - The tick-start speed (m/s).
 * @param demandedTurnRateRadPerS - The demanded turn rate (rad/s), or
 *   `undefined` to OMIT the field.
 * @returns The per-tick plan of the run.
 */
const buildMomentumTicks = (
  count: number,
  steer: number,
  speed: number,
  demandedTurnRateRadPerS?: number,
): MomentumTickPlan[] =>
  Array.from({ length: count }, () =>
    demandedTurnRateRadPerS === undefined
      ? { steer, speed }
      : { steer, speed, demandedTurnRateRadPerS },
  );

/**
 * Drives the pure drift commitment with a PER-TICK plan.
 *
 * The runner mirrors `runCommittedTicks` (the same lateral-velocity
 * threading, the same speed percent) and differs in TWO things: the tick
 * input may carry the demanded turn rate, and the plan may OMIT that field
 * so the law falls back to the mirrored steer-times-speed-percent rate. The
 * input is built as a VARIABLE (never as an inline object literal), so the
 * widened member never trips the excess-property check of the typed
 * parameter.
 *
 * @param plans - The per-tick plan (one entry per tick).
 * @returns The committed run of the lane.
 */
const runMomentumCommittedPlan = (
  plans: ReadonlyArray<MomentumTickPlan>,
): CommittedRun => {
  const channel = createZeroDriftChannelState();
  let committed: DriftCommittedState = channel.committed;
  let lateralVelocity = 0;
  const samples: CommittedTickSample[] = [];
  let maxAbsCommittedSlipRad = 0;
  let maxAbsLateralVelocityMps = 0;

  for (const plan of plans) {
    const baseInput: DriftStepInput = {
      steer: plan.steer,
      speed: plan.speed,
      speedPercent: plan.speed / SCN_DRIFT_01_SPEED_MPS,
      dt: SCN_DRIFT_01_DT_S,
      lateralVelocity,
      slipAngle: 0,
    };
    // The EXPLICIT rate is carried through the WIDENED LOCAL VIEW (the
    // contract pins the field name); an ABSENT rate leaves the field
    // OMITTED, so the production fallback resolves the mirrored rate.
    let input: DriftStepInput = baseInput;
    if (plan.demandedTurnRateRadPerS !== undefined) {
      const widened: MomentumDriftStepInput = {
        ...baseInput,
        demandedTurnRateRadPerS: plan.demandedTurnRateRadPerS,
      };
      input = widened;
    }
    committed = stepDriftCommitment(committed, input);
    lateralVelocity = computeNextLateralVelocity(committed, input);
    samples.push({
      committed,
      lateralVelocity,
      slipAngleRad: 0,
      lateralVelocityAtStart: input.lateralVelocity,
    });
    maxAbsCommittedSlipRad = Math.max(
      maxAbsCommittedSlipRad,
      Math.abs(committed.slip),
    );
    maxAbsLateralVelocityMps = Math.max(
      maxAbsLateralVelocityMps,
      Math.abs(lateralVelocity),
    );
  }

  return {
    committed,
    lateralVelocity,
    samples,
    maxAbsCommittedSlipRad,
    maxAbsLateralVelocityMps,
  };
};

/**
 * Drives the pure drift commitment with a WIDENED input that carries the
 * demanded turn rate on EVERY tick.
 *
 * The runner mirrors `runCommittedTicks` (the same tick count, the same
 * lateral-velocity threading, the same speed percent) and differs in ONE
 * thing: the tick input carries the demanded turn rate.
 *
 * @param ticks - The number of 60 Hz ticks to run.
 * @param steer - The steer axis (`+1` = LEFT, `-1` = RIGHT).
 * @param speed - The tick-start speed (m/s).
 * @param demandedTurnRateRadPerS - The demanded turn rate (rad/s).
 * @returns The committed run of the lane.
 */
const runMomentumCommittedTicks = (
  ticks: number,
  steer: number,
  speed: number,
  demandedTurnRateRadPerS: number,
): CommittedRun =>
  runMomentumCommittedPlan(
    buildMomentumTicks(ticks, steer, speed, demandedTurnRateRadPerS),
  );

/**
 * The MOMENTUM law contracts (Step 14S, slice `P1-momentum-red`; the
 * plan's RED-1, RED-2 and RED-3 acceptance criteria).
 *
 * AC-091: the drift is CAUSED by the MOMENTUM, not COMMANDED by the wheel.
 * The committed slide is produced by the MOMENTUM EXCESS - the lateral
 * acceleration the commanded turn DEMANDS minus the lateral acceleration
 * the tires can HOLD at the current speed - so the SAME wheel command
 * produces a STRICTLY DEEPER slide at a higher speed, the slide is EXACTLY
 * 0 at a standstill, and the slide is always bounded by the declared grip
 * and depth budgets. THE TURN-RATE DIMENSION IS PART OF THE CAUSE: for the
 * SAME speed and the SAME steer axis a STRICTLY LARGER demanded turn rate
 * produces a STRICTLY DEEPER committed slide, and a ZERO demanded turn rate
 * produces an EXACTLY 0 committed slide AT ENGAGEMENT (an UNCOMMITTED
 * latch - the latch is never spent on a zero slide), while under a LIVE
 * latch a ZERO momentum demand HOLDS the committed depth (the slide never
 * SNAPS to zero under a live latch; only the RELEASE - the countersteer or
 * the NEUTRAL wheel - unwinds it).
 */
describe('the MOMENTUM law (Step 14S, slice P1-momentum-red)', () => {
  describe('RED-1 (the momentum law IS a law)', () => {
    // [RED-14S-1-EXISTENCE]
    it('AC-091: the drift model exports a NAMED momentum law that maps (the speed, the demanded turn rate, the available grip) to the momentum slide demand (RED-1)', () => {
      // The law must EXIST and be callable. The MEASURED current-model
      // state: there is NO momentum term anywhere in the drift law - the
      // only speed terms are the depth budget, the entry rate, the chase
      // tau and the severity, and NONE of them is a function of the
      // demanded turn rate. There is no mass, no lateral acceleration and
      // no grip limit in the file at all.
      expect(typeof momentumSlideDemandLaw).toBe('function');
    });

    // [RED-14S-1-STANDSTILL]
    it('AC-091: the momentum demand is EXACTLY 0 at a standstill (RED-1)', () => {
      // There is NO momentum at rest: the demand of a standstill is
      // EXACTLY 0, even though the wheel demands the SAME full turn the
      // moving lanes demand - the slide of a standstill is EXACTLY 0
      // (AC-091).
      expect(
        momentumSlideDemandRad(
          0,
          MOMENTUM_FULL_STEER_TURN_RATE_RAD_PER_S,
          MOMENTUM_FULL_GRIP_SHARE,
        ),
      ).toBe(0);
    });

    // [RED-14S-1-FUNCTION-OF-TURN-RATE]
    it('AC-091: the momentum demand is a FUNCTION of the demanded turn rate (RED-1)', () => {
      // The NON-VACUITY GUARD comes FIRST: the non-zero demanded turn
      // rate lane must REALLY produce a demand above 0 (a zero demand
      // would satisfy the exact-zero clause VACUOUSLY).
      const demanded = momentumSlideDemandRad(
        MOMENTUM_TOP_SPEED_MPS,
        MOMENTUM_FULL_STEER_TURN_RATE_RAD_PER_S,
        MOMENTUM_FULL_GRIP_SHARE,
      );
      expect(demanded).toBeGreaterThan(0);
      // A law that ignores the demanded turn rate entirely can NEVER pass
      // this contract: a ZERO demanded turn rate demands NO slide.
      expect(
        momentumSlideDemandRad(
          MOMENTUM_TOP_SPEED_MPS,
          0,
          MOMENTUM_FULL_GRIP_SHARE,
        ),
      ).toBe(0);
    });

    // [RED-14S-1-MONOTONIC-SPEED]
    it('AC-091: the momentum demand is MONOTONIC in the speed (RED-1)', () => {
      // The SAME demanded turn rate and the SAME grip at two speeds.
      const highSpeedDemand = momentumSlideDemandRad(
        MOMENTUM_TOP_SPEED_MPS,
        MOMENTUM_FULL_STEER_TURN_RATE_RAD_PER_S,
        MOMENTUM_FULL_GRIP_SHARE,
      );
      const lowSpeedDemand = momentumSlideDemandRad(
        MOMENTUM_LOW_SPEED_MPS,
        MOMENTUM_FULL_STEER_TURN_RATE_RAD_PER_S,
        MOMENTUM_FULL_GRIP_SHARE,
      );
      // The NON-VACUITY GUARD comes FIRST: BOTH lanes must REALLY demand
      // a slide (a zero demand would satisfy the strict clause
      // VACUOUSLY).
      expect(highSpeedDemand).toBeGreaterThan(0);
      expect(lowSpeedDemand).toBeGreaterThan(0);
      // The weight at speed IS the cause: the same wheel command demands
      // a STRICTLY deeper slide at the higher speed.
      expect(highSpeedDemand).toBeGreaterThan(lowSpeedDemand);
    });
  });

  describe('RED-2 (the weight causes the drift, not the wheel)', () => {
    // [RED-14S-2-SPEED-DIMENSION]
    it('AC-091: the committed slide for the SAME full steer command is STRICTLY DEEPER at a high speed than at a low speed (RED-2) - PRESERVATION GUARD', () => {
      // PRESERVATION GUARD, NOT A RED: the MEASURED current-model depth
      // is the speed-proportional depth budget (0.45 rad at 89 m/s and
      // 0.14087152516904583 rad at 25 m/s), so the speed dimension
      // ALREADY holds today. It is the RAIL the new cause must stay on -
      // it is NOT counted as a red.
      const topSpeedRun = runMomentumCommittedTicks(
        SCN_DRIFT_01_BUILD_TICKS,
        1,
        MOMENTUM_TOP_SPEED_MPS,
        MOMENTUM_FULL_STEER_TURN_RATE_RAD_PER_S,
      );
      const lowSpeedRun = runMomentumCommittedTicks(
        SCN_DRIFT_01_BUILD_TICKS,
        1,
        MOMENTUM_LOW_SPEED_MPS,
        MOMENTUM_FULL_STEER_TURN_RATE_RAD_PER_S,
      );
      // The NON-VACUITY GUARD comes FIRST: BOTH lanes must REALLY slide
      // (a zero slide would satisfy the strict-deeper clause VACUOUSLY).
      expect(Math.abs(topSpeedRun.committed.slip)).toBeGreaterThan(0);
      expect(Math.abs(lowSpeedRun.committed.slip)).toBeGreaterThan(0);
      // The same wheel command slides deeper at the higher speed.
      expect(Math.abs(topSpeedRun.committed.slip)).toBeGreaterThan(
        Math.abs(lowSpeedRun.committed.slip),
      );
    });

    // [RED-14S-2-STANDSTILL]
    it('AC-091: the committed slide is EXACTLY 0 at a standstill (RED-2) - PRESERVATION GUARD', () => {
      // PRESERVATION GUARD, NOT A RED: the MEASURED current-model
      // commitment at a standstill is EXACTLY 0 (the entry rate is 0 at a
      // standstill), so the standstill clause ALREADY holds today. It is
      // the RAIL the new cause must stay on.
      const standstillRun = runMomentumCommittedTicks(
        SCN_DRIFT_01_BUILD_TICKS,
        1,
        0,
        MOMENTUM_FULL_STEER_TURN_RATE_RAD_PER_S,
      );
      expect(Math.abs(standstillRun.committed.slip)).toBe(0);
    });

    // [RED-14S-2-MIRRORED-CONSTANT-GUARD]
    it('AC-091: at the SAME speed and the SAME steer axis the ABSENT-rate lane reads a DIFFERENT committed slide from the STRICTLY-LARGER explicit-rate lane (RED-2)', () => {
      // THE MIRRORED-CONSTANT GUARD (FP-14S-03), asserted BEFORE the
      // relational clause.
      //
      // The ABSENT-rate lane OMITS the demanded turn rate field from the
      // drift input, so the law falls back to the MIRRORED
      // steer-times-speed-percent rate (the production fallback is
      // `steer * NEATRACING_DRIFT_FULL_STEER_TURN_RATE_RAD_PER_S *
      // speedPercent`). The SMALLER explicit rate of the strict-deeper lane
      // IS that mirror, so a lane the MIRROR alone can satisfy does NOT
      // discriminate the law: a law that IGNORED the explicit demanded turn
      // rate would still satisfy the smaller lane. Without this guard the
      // strict-deeper relation would rest on ONE lane only.
      //
      // The lanes share the SAME speed and the SAME steer axis; only the
      // demanded turn rate differs.
      const absentRateLane = runMomentumCommittedPlan(
        buildMomentumTicks(SCN_DRIFT_01_BUILD_TICKS, 1, MOMENTUM_LOW_SPEED_MPS),
      );
      const smallerRateLane = runMomentumCommittedTicks(
        SCN_DRIFT_01_BUILD_TICKS,
        1,
        MOMENTUM_LOW_SPEED_MPS,
        MOMENTUM_LOW_SPEED_TURN_RATE_RAD_PER_S,
      );
      const largerRateLane = runMomentumCommittedTicks(
        SCN_DRIFT_01_BUILD_TICKS,
        1,
        MOMENTUM_LOW_SPEED_MPS,
        MOMENTUM_DOUBLE_LOW_SPEED_TURN_RATE_RAD_PER_S,
      );
      // The NON-VACUITY GUARD comes FIRST: BOTH lanes must REALLY slide (a
      // zero slide would satisfy the difference clause VACUOUSLY).
      expect(Math.abs(absentRateLane.committed.slip)).toBeGreaterThan(0);
      expect(Math.abs(largerRateLane.committed.slip)).toBeGreaterThan(0);
      // THE GUARD: the absent lane reads a DIFFERENT committed slide from
      // the strictly-larger explicit-rate lane, so the fallback is NOT the
      // law - the explicit demanded turn rate is.
      expect(Math.abs(absentRateLane.committed.slip)).not.toBe(
        Math.abs(largerRateLane.committed.slip),
      );
      // The MECHANISM the guard rests on: the production fallback resolves
      // EXACTLY the smaller explicit rate, so the absent lane IS the mirror
      // lane and the two read the IDENTICAL committed slide.
      expect(Math.abs(absentRateLane.committed.slip)).toBe(
        Math.abs(smallerRateLane.committed.slip),
      );
    });

    // [RED-14S-2-TURN-RATE-STRICTLY-DEEPER]
    it('AC-091: for the SAME speed and the SAME steer axis a STRICTLY LARGER demanded turn rate produces a STRICTLY DEEPER committed slide (RED-2)', () => {
      // The two lanes differ ONLY in the demanded turn rate: the SAME
      // speed, the SAME steer axis (full LEFT) and the SAME tick count.
      //
      // The demanded turn rate is expressed through the WIDENED LOCAL VIEW
      // of the internal drift input (the field is an internal TYPE MEMBER,
      // so a MISSING-IMPORT failure is UNSATISFIABLE for it) and the field
      // name is pinned by this contract.
      const smallerRateLane = runMomentumCommittedTicks(
        SCN_DRIFT_01_BUILD_TICKS,
        1,
        MOMENTUM_LOW_SPEED_MPS,
        MOMENTUM_LOW_SPEED_TURN_RATE_RAD_PER_S,
      );
      const largerRateLane = runMomentumCommittedTicks(
        SCN_DRIFT_01_BUILD_TICKS,
        1,
        MOMENTUM_LOW_SPEED_MPS,
        MOMENTUM_DOUBLE_LOW_SPEED_TURN_RATE_RAD_PER_S,
      );

      // The NON-VACUITY GUARD comes FIRST: BOTH lanes must REALLY slide
      // (a zero slide would satisfy the strict-deeper clause VACUOUSLY).
      expect(Math.abs(smallerRateLane.committed.slip)).toBeGreaterThan(0);
      expect(Math.abs(largerRateLane.committed.slip)).toBeGreaterThan(0);
      // The demanded turn rate is a CAUSE, not a decoration: a strictly
      // larger demanded turn rate produces a strictly deeper slide.
      //
      // The MEASURED current-model behaviour: the commitment is a COMMAND
      // - the slide opens on the steer gate and ramps at the speed-
      // proportional ENTRY RATE - so the drift input carries NO demanded
      // turn rate today and BOTH lanes read the IDENTICAL committed slide
      // (the depth budget of the lane, 0.14087152516904583 rad).
      expect(Math.abs(largerRateLane.committed.slip)).toBeGreaterThan(
        Math.abs(smallerRateLane.committed.slip),
      );
    });

    // [RED-14S-2-ZERO-TURN-RATE-AT-ENGAGEMENT]
    it('AC-091: a ZERO demanded turn rate produces an EXACTLY 0 committed slide AT ENGAGEMENT (an UNCOMMITTED latch) even at the top speed (RED-2)', () => {
      // THE ZERO-RATE CLAUSE IS SCOPED TO ENGAGEMENT (FP-14S-01): the
      // ENGAGE branch needs a POSITIVE momentum demand, so a wheel command
      // the momentum does NOT carry opens NO slide and the latch is never
      // spent on a zero slide.
      //
      // The lane is asserted FROM A FRESH STATE (an UNCOMMITTED latch): the
      // LIVE-LATCH HOLD is a SEPARATE lane below, so the zero clause can
      // never be carried by a fresh-state-only lane.
      //
      // The SAME steer axis (full LEFT) and the SAME speed (the arcade top
      // speed) as the sliding lane, but NO demanded turn rate: the wheel
      // commands a turn the momentum does not carry, so the car does not
      // slide at all.
      const zeroRateLane = runMomentumCommittedTicks(
        SCN_DRIFT_01_BUILD_TICKS,
        1,
        MOMENTUM_TOP_SPEED_MPS,
        0,
      );
      const slidingLane = runMomentumCommittedTicks(
        SCN_DRIFT_01_BUILD_TICKS,
        1,
        MOMENTUM_TOP_SPEED_MPS,
        MOMENTUM_FULL_STEER_TURN_RATE_RAD_PER_S,
      );
      // The NON-VACUITY GUARD comes FIRST: the comparison lane must REALLY
      // slide, or the exact-zero clause would be VACUOUS.
      expect(Math.abs(slidingLane.committed.slip)).toBeGreaterThan(0);
      // A zero demanded turn rate produces NO slide AT ENGAGEMENT.
      expect(Math.abs(zeroRateLane.committed.slip)).toBe(0);
      // The latch stays UNCOMMITTED: the zero slide never spent the latch.
      expect(zeroRateLane.committed.turnSign).toBe(0);
    });

    // [RED-14S-2-LIVE-LATCH-HOLD]
    it('AC-091: under a LIVE latch a ZERO momentum demand HOLDS the committed depth and only the RELEASE unwinds it (RED-2)', () => {
      // THE LIVE-LATCH HOLD LANE (FP-14S-03) - a SEPARATE lane from the
      // ENGAGEMENT-scoped zero lane above, so the zero clause can never be
      // carried by a fresh-state-only lane.
      //
      // PHASE A - ENGAGE: a FRESH latch, the LOW-speed lane, the full LEFT
      // steer axis and the lane's own demanded turn rate, so the latch
      // goes LIVE and a REAL committed slide opens.
      const engageRun = runMomentumCommittedTicks(
        1,
        1,
        MOMENTUM_LOW_SPEED_MPS,
        MOMENTUM_LOW_SPEED_TURN_RATE_RAD_PER_S,
      );
      // The NON-VACUITY GUARD comes FIRST: the latch must REALLY be live
      // and the committed slide must REALLY be non-zero, or the HOLD clause
      // would be VACUOUS.
      expect(engageRun.committed.turnSign).not.toBe(0);
      expect(Math.abs(engageRun.committed.slip)).toBeGreaterThan(0);

      // PHASE B - HOLD: the SAME speed and the SAME steer axis, but a ZERO
      // demanded turn rate - the momentum no longer carries the commanded
      // turn. Under a LIVE latch the committed depth is HELD: the slide
      // NEVER SNAPS to zero.
      const holdRun = runMomentumCommittedPlan([
        ...buildMomentumTicks(
          1,
          1,
          MOMENTUM_LOW_SPEED_MPS,
          MOMENTUM_LOW_SPEED_TURN_RATE_RAD_PER_S,
        ),
        ...buildMomentumTicks(
          MOMENTUM_LATCH_HOLD_TICKS,
          1,
          MOMENTUM_LOW_SPEED_MPS,
          0,
        ),
      ]);
      // The latch is still LIVE after the hold window.
      expect(holdRun.committed.turnSign).not.toBe(0);
      // The committed depth is UNCHANGED: it is HELD, not unwound.
      expect(Math.abs(holdRun.committed.slip)).toBeGreaterThan(0);
      expect(holdRun.committed.slip).toBe(engageRun.committed.slip);

      // PHASE C - RELEASE: the COUNTERSTEER unwinds the committed slide.
      const releaseRun = runMomentumCommittedPlan([
        ...buildMomentumTicks(
          1,
          1,
          MOMENTUM_LOW_SPEED_MPS,
          MOMENTUM_LOW_SPEED_TURN_RATE_RAD_PER_S,
        ),
        ...buildMomentumTicks(
          MOMENTUM_LATCH_HOLD_TICKS,
          1,
          MOMENTUM_LOW_SPEED_MPS,
          0,
        ),
        ...buildMomentumTicks(
          MOMENTUM_RELEASE_TICKS,
          -1,
          MOMENTUM_LOW_SPEED_MPS,
          0,
        ),
      ]);
      // ONLY the RELEASE unwinds it: the countersteer brings the committed
      // slide back to EXACTLY 0 and releases the latch.
      expect(releaseRun.committed.slip).toBe(0);
      expect(releaseRun.committed.turnSign).toBe(0);
    });
  });

  describe('RED-3 (the momentum excess stays inside the grip and the depth budget)', () => {
    // [RED-14S-3-EXISTENCE-FUNCTION]
    it('AC-091: the momentum demand EXISTS and is a FUNCTION of the demanded turn rate (RED-3)', () => {
      // The NON-VACUITY GUARD comes FIRST: the non-zero demanded turn
      // rate lane must REALLY produce a demand above 0.
      const demanded = momentumSlideDemandRad(
        MOMENTUM_MID_SPEED_MPS,
        MOMENTUM_FULL_STEER_TURN_RATE_RAD_PER_S,
        MOMENTUM_FULL_GRIP_SHARE,
      );
      expect(demanded).toBeGreaterThan(0);
      // A law that is only a clamp or only a bound can NEVER pass this
      // contract: at a ZERO demanded turn rate the demand is EXACTLY 0 at
      // the SAME speed.
      expect(
        momentumSlideDemandRad(
          MOMENTUM_MID_SPEED_MPS,
          0,
          MOMENTUM_FULL_GRIP_SHARE,
        ),
      ).toBe(0);
    });

    // [RED-14S-3-DEPTH-BUDGET-RAIL]
    it('AC-091: the momentum demand can NEVER push the committed slide past the declared depth budget (RED-3) - PRESERVATION GUARD', () => {
      // PRESERVATION GUARD, NOT A RED: the committed slide is clamped to
      // the declared depth budget TODAY (drift.model.ts:817) and the
      // MEASURED severity clamp (drift.model.ts:624) keeps the severity
      // inside the closed unit interval. Both clauses ALREADY hold - they
      // are the RAILS the new cause must stay on.
      const topSpeedRun = runMomentumCommittedTicks(
        SCN_DRIFT_01_BUILD_TICKS,
        1,
        MOMENTUM_TOP_SPEED_MPS,
        MOMENTUM_FULL_STEER_TURN_RATE_RAD_PER_S,
      );
      expect(Math.abs(topSpeedRun.committed.slip)).toBeLessThanOrEqual(
        NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD,
      );
      // The MEASURED depth budget of the saturation lane IS the ceiling of
      // the momentum demand: a demand above it would be MASKED by the
      // clamp.
      expect(computeDriftSlideDepthBudgetRad(MOMENTUM_TOP_SPEED_MPS)).toBe(
        NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD,
      );
    });

    // [RED-14S-3-SEVERITY-RAIL]
    it('AC-091: the published traction severity stays inside the closed unit interval at the saturation speed (RED-3) - PRESERVATION GUARD', () => {
      // PRESERVATION GUARD, NOT A RED: the severity is clamped to
      // `[0, 1]` TODAY (drift.model.ts:624) and the MEASURED severity of
      // the saturation lane (the full slide at the arcade top speed) is
      // EXACTLY 1.
      const severity = computeTractionSeverity(
        NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD,
        MOMENTUM_TOP_SPEED_MPS,
      );
      expect(severity).toBeGreaterThanOrEqual(0);
      expect(severity).toBeLessThanOrEqual(1);
      expect(severity).toBe(1);
    });
  });
});
