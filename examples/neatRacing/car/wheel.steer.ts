/**
 * Front-wheel steer animation helper for the Neat Racing voxel racer.
 *
 * AC-044: the visual yaw of both front wheels follows the steering input
 * axis up to a named max lock angle. Wheel spin speed and direction are
 * independent of this visual yaw: the steer transform is a separate rotation
 * around the car's up axis applied after the G9 mount yaw and before the
 * spin rotation.
 *
 * @module
 */

/** Visual max lock angle for front wheels in radians (~35°). */
export const NEATRACING_MAX_WHEEL_STEER_RAD = (35 * Math.PI) / 180;

/** Minimum normalized steering-axis value. */
const STEER_AXIS_MIN = -1;

/** Maximum normalized steering-axis value. */
const STEER_AXIS_MAX = 1;

/**
 * Clamp a steering-axis sample to the normalized [-1, 1] range.
 * Non-finite values are coerced to 0 so a bad input sample can never
 * produce an unbounded wheel angle.
 *
 * @param steerAxis - Raw steering axis; positive steers counterclockwise.
 * @returns A clamped axis value in [-1, 1].
 */
function clampSteerAxis(steerAxis: number): number {
  if (!Number.isFinite(steerAxis)) {
    return 0;
  }
  if (steerAxis > STEER_AXIS_MAX) {
    return STEER_AXIS_MAX;
  }
  if (steerAxis < STEER_AXIS_MIN) {
    return STEER_AXIS_MIN;
  }
  return steerAxis;
}

/**
 * Compute the visual front-wheel steer angle for a steering-axis input.
 *
 * Positive input (left / A key) produces a positive angle, which rotates
 * the front wheels counterclockwise around the car's up axis. Negative
 * input (right / D key) produces a negative angle. Over-range inputs are
 * clamped so the wheels never exceed {@link NEATRACING_MAX_WHEEL_STEER_RAD}.
 *
 * @param steerAxis - Steering axis in [-1, 1]; values outside this range
 *   are clamped.
 * @returns Steer angle in radians to apply to the front wheels.
 */
export function computeWheelSteerAngleRad(steerAxis: number): number {
  return clampSteerAxis(steerAxis) * NEATRACING_MAX_WHEEL_STEER_RAD;
}
