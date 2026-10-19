/**
 * Red / green contract tests for the front-wheel steer animation helper.
 *
 * AC-044: the visual front-wheel yaw is computed from the steering input
 * axis and clamped to a named max lock angle; wheel spin direction is
 * untouched by the steer transform.
 *
 * @module
 */

import {
  computeWheelSteerAngleRad,
  NEATRACING_MAX_WHEEL_STEER_RAD,
} from './wheel.steer';

describe('wheel steer helper', () => {
  it('exposes a finite, positive max lock angle', () => {
    expect(NEATRACING_MAX_WHEEL_STEER_RAD).toBeGreaterThan(0);
    expect(Number.isFinite(NEATRACING_MAX_WHEEL_STEER_RAD)).toBe(true);
  });

  it('maps full left input to negative max lock', () => {
    expect(computeWheelSteerAngleRad(-1)).toBe(
      -NEATRACING_MAX_WHEEL_STEER_RAD,
    );
  });

  it('maps centered input to zero steer', () => {
    expect(computeWheelSteerAngleRad(0)).toBe(0);
  });

  it('maps full right input to positive max lock', () => {
    expect(computeWheelSteerAngleRad(1)).toBe(NEATRACING_MAX_WHEEL_STEER_RAD);
  });

  it('clamps over-range inputs without flipping sign', () => {
    expect(computeWheelSteerAngleRad(2)).toBe(NEATRACING_MAX_WHEEL_STEER_RAD);
    expect(computeWheelSteerAngleRad(-3)).toBe(
      -NEATRACING_MAX_WHEEL_STEER_RAD,
    );
  });

  it('preserves spin-direction invariance (sign mirrors input)', () => {
    expect(Math.sign(computeWheelSteerAngleRad(0.5))).toBe(1);
    expect(Math.sign(computeWheelSteerAngleRad(-0.5))).toBe(-1);
    expect(computeWheelSteerAngleRad(0)).toBe(0);
  });
});
