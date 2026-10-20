/**
 * Red-phase contract tests for the Neat Racing arcade drift model
 * (slice P1-arcade-drift; AC-047; spec §5.2 drift amendment).
 *
 * Red phase: this suite must fail while './drift.model' does not exist
 * ("Cannot find module './drift.model'" under ts-jest diagnostics).
 * A green suite during the red phase would be a contract defect.
 *
 * Green contract for the implementer: './drift.model' exports the named
 * drift constants, the pure `stepDriftModel(prev, input)` function, and
 * the focused helpers `computeOutwardPush`, `computeGripFalloff`,
 * `computeCountersteerDelta`, and `computeSlipAngle`.
 *
 * The model is deterministic (spec §5.3): no `Math.random`, no side effects,
 * so the same inputs always produce byte-identical state.
 */

import { describe, expect, it } from '@jest/globals';
import {
  NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD,
  NEATRACING_DRIFT_GRIP_FALLOFF_THRESHOLD_RAD,
  NEATRACING_DRIFT_GRIP_FALLOFF_RATE_PER_RAD,
  NEATRACING_DRIFT_MIN_GRIP_FACTOR,
  NEATRACING_DRIFT_COUNTERSTEER_RECOVERY_RATE_PER_S,
  NEATRACING_DRIFT_LATERAL_GRIP_RATE_PER_S,
  NEATRACING_DRIFT_MIN_SPEED_MPS,
  NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD,
  createZeroDriftState,
  stepDriftModel,
  computeOutwardPush,
  computeGripFalloff,
  computeCountersteerDelta,
  computeSlipAngle,
  computeDriftYawOffset,
} from './drift.model';
import type { DriftModelState, DriftStepInput } from './drift.model';

/** Deterministic JSON serialization for byte-identity comparisons. */
const serializeDriftState = (state: DriftModelState): string =>
  JSON.stringify(state);

/** Default fixed-step dt used by the physics service (60 Hz). */
const DT_SECONDS = 1 / 60;

/** Half of the arcade top speed, a representative mid-speed case. */
const MID_SPEED_M_PER_S = 44.5;

/** A modest yaw rate per tick for the build-up fixtures. */
const YAW_RATE_PER_TICK = 0.02;

const makeStepInput = (
  overrides: Partial<DriftStepInput> = {},
): DriftStepInput => ({
  steer: 0,
  yawRatePerTick: 0,
  speed: MID_SPEED_M_PER_S,
  speedPercent: MID_SPEED_M_PER_S / 89,
  dt: DT_SECONDS,
  centrifugalMultiplier: 0.3,
  ...overrides,
});

describe('drift model constants (spec §5.2 drift amendment)', () => {
  it('exposes bounded, positive drift envelope constants', () => {
    expect(NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD).toBe(Math.PI / 2);
    expect(NEATRACING_DRIFT_GRIP_FALLOFF_THRESHOLD_RAD).toBeGreaterThan(0);
    expect(NEATRACING_DRIFT_GRIP_FALLOFF_RATE_PER_RAD).toBeGreaterThan(0);
    expect(NEATRACING_DRIFT_MIN_GRIP_FACTOR).toBeGreaterThan(0);
    expect(NEATRACING_DRIFT_MIN_GRIP_FACTOR).toBeLessThan(1);
    expect(NEATRACING_DRIFT_COUNTERSTEER_RECOVERY_RATE_PER_S).toBeGreaterThan(
      0,
    );
    expect(NEATRACING_DRIFT_LATERAL_GRIP_RATE_PER_S).toBeGreaterThan(0);
    expect(NEATRACING_DRIFT_MIN_SPEED_MPS).toBeGreaterThan(0);
    expect(NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD).toBeGreaterThan(0);
  });
});

describe('drift model helpers (AC-047)', () => {
  it('starts from a zero slip / zero lateral-velocity state', () => {
    const state = createZeroDriftState();

    expect(state.slipAngle).toBe(0);
    expect(state.lateralVelocity).toBe(0);
  });

  it('computes outward push proportional to yaw rate and speed, zero at standstill', () => {
    const moving = computeOutwardPush(YAW_RATE_PER_TICK, MID_SPEED_M_PER_S, 0.3);
    const standing = computeOutwardPush(YAW_RATE_PER_TICK, 0, 0.3);
    const belowThreshold = computeOutwardPush(
      YAW_RATE_PER_TICK,
      NEATRACING_DRIFT_MIN_SPEED_MPS,
      0.3,
    );

    expect(Math.abs(moving)).toBeGreaterThan(0);
    expect(standing).toBe(0);
    expect(belowThreshold).toBe(0);
  });

  it('computes grip falloff: full grip near zero slip, reduced grip at large slip', () => {
    const fullGrip = computeGripFalloff(0);
    const thresholdGrip = computeGripFalloff(
      NEATRACING_DRIFT_GRIP_FALLOFF_THRESHOLD_RAD,
    );
    const largeSlipGrip = computeGripFalloff(
      NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD * 0.75,
    );

    expect(fullGrip).toBe(1);
    expect(thresholdGrip).toBe(1);
    expect(largeSlipGrip).toBeLessThan(1);
    expect(largeSlipGrip).toBeGreaterThanOrEqual(
      NEATRACING_DRIFT_MIN_GRIP_FACTOR - 1e-12,
    );
  });

  it('computes countersteer delta only when steer matches the slip sign (countersteer into the skid)', () => {
    const positiveSlip = 0.5;
    const sameSignSteer = computeCountersteerDelta(
      1,
      positiveSlip,
      1,
      DT_SECONDS,
    );
    const oppositeSignSteer = computeCountersteerDelta(
      -1,
      positiveSlip,
      1,
      DT_SECONDS,
    );
    const noSlip = computeCountersteerDelta(1, 0, 1, DT_SECONDS);

    expect(sameSignSteer).toBeLessThan(0);
    expect(oppositeSignSteer).toBe(0);
    expect(noSlip).toBe(0);
  });

  it('computes slip angle from lateral velocity and speed, bounded for a moving car', () => {
    expect(computeSlipAngle(10, 10)).toBeCloseTo(Math.PI / 4, 10);
    expect(Math.abs(computeSlipAngle(100, 10))).toBeLessThan(Math.PI / 2);
    expect(computeSlipAngle(5, 0)).toBe(0);
  });
});

describe('drift model step behavior (AC-047)', () => {
  it('builds up a bounded slip angle under sustained steer and throttle', () => {
    let state = createZeroDriftState();
    const input = makeStepInput({
      steer: 1,
      yawRatePerTick: YAW_RATE_PER_TICK,
    });

    for (let step = 0; step < 120; step += 1) {
      state = stepDriftModel(state, input);
    }

    expect(Math.abs(state.slipAngle)).toBeGreaterThan(0);
    expect(Math.abs(state.slipAngle)).toBeLessThan(
      NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD,
    );
    expect(Number.isFinite(state.slipAngle)).toBe(true);
  });

  it('recovers slip when the driver countersteers', () => {
    let state = createZeroDriftState();
    const buildInput = makeStepInput({
      steer: 1,
      yawRatePerTick: YAW_RATE_PER_TICK,
    });

    for (let step = 0; step < 120; step += 1) {
      state = stepDriftModel(state, buildInput);
    }

    const peakSlip = state.slipAngle;
    expect(Math.abs(peakSlip)).toBeGreaterThan(0);

    const recoverInput = makeStepInput({
      steer: -1,
      yawRatePerTick: -YAW_RATE_PER_TICK,
    });
    for (let step = 0; step < 120; step += 1) {
      state = stepDriftModel(state, recoverInput);
    }

    expect(Math.abs(state.slipAngle)).toBeLessThan(Math.abs(peakSlip));
  });

  it('produces identical drift state for the same input across runs', () => {
    const input = makeStepInput({
      steer: 1,
      yawRatePerTick: YAW_RATE_PER_TICK,
    });

    let runA = createZeroDriftState();
    let runB = createZeroDriftState();
    for (let step = 0; step < 120; step += 1) {
      runA = stepDriftModel(runA, input);
      runB = stepDriftModel(runB, input);
    }

    expect(serializeDriftState(runA)).toBe(serializeDriftState(runB));
  });

  it('AC-047A: high-speed sustained steering produces visible lateral slide', () => {
    const highSpeedMps = 56;
    const speedPercent = highSpeedMps / 89;
    const yawRatePerTick = 2 * speedPercent * DT_SECONDS;
    let state = createZeroDriftState();
    const input = makeStepInput({
      speed: highSpeedMps,
      speedPercent,
      steer: 1,
      yawRatePerTick,
      centrifugalMultiplier: 1.5,
    });

    for (let step = 0; step < 120; step += 1) {
      state = stepDriftModel(state, input);
    }

    expect(Math.abs(state.lateralVelocity)).toBeGreaterThan(10);
    expect(Math.abs(state.slipAngle)).toBeGreaterThan(0.2);
    expect(Math.abs(state.slipAngle)).toBeLessThan(
      NEATRACING_DRIFT_MAX_SLIP_ANGLE_RAD,
    );
  });

  it('AC-047B: countersteer recovery arrests the slide within a bounded time', () => {
    const highSpeedMps = 56;
    const speedPercent = highSpeedMps / 89;
    const yawRatePerTick = 2 * speedPercent * DT_SECONDS;
    let state = createZeroDriftState();
    const buildInput = makeStepInput({
      speed: highSpeedMps,
      speedPercent,
      steer: 1,
      yawRatePerTick,
      centrifugalMultiplier: 1.5,
    });

    for (let step = 0; step < 120; step += 1) {
      state = stepDriftModel(state, buildInput);
    }

    const recoverInput = makeStepInput({
      speed: highSpeedMps,
      speedPercent,
      steer: -1,
      yawRatePerTick: -yawRatePerTick,
      centrifugalMultiplier: 1.5,
    });
    for (let step = 0; step < 15; step += 1) {
      state = stepDriftModel(state, recoverInput);
    }

    expect(Math.abs(state.lateralVelocity)).toBeLessThan(5);
  });

  it('AC-047D: low-speed straight-line handling is unchanged', () => {
    const lowSpeedMps = 10;
    const speedPercent = lowSpeedMps / 89;
    let state = createZeroDriftState();
    const input = makeStepInput({
      speed: lowSpeedMps,
      speedPercent,
      steer: 0,
      yawRatePerTick: 0,
      centrifugalMultiplier: 1.5,
    });

    for (let step = 0; step < 120; step += 1) {
      state = stepDriftModel(state, input);
    }

    expect(serializeDriftState(state)).toBe(
      serializeDriftState(createZeroDriftState()),
    );
  });

  it('computes a visual yaw offset proportional to slip angle', () => {
    expect(computeDriftYawOffset(10, 10)).toBeCloseTo(
      (Math.PI / 4) * NEATRACING_DRIFT_VISUAL_YAW_GAIN_RAD,
      10,
    );
    expect(computeDriftYawOffset(10, 0)).toBe(0);
  });
});
