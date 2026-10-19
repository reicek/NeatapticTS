/**
 * Car render-pose generator for the live drive loop.
 *
 * Bridges the deterministic physics state to the car/glass pass pose bags:
 * it fills a stable mutable pose object in place (P9: no per-frame
 * allocation churn) and computes the front-wheel steer angle from the
 * current steering input.
 *
 * @module
 */

import { computeWheelSteerAngleRad } from './wheel.steer';

export {
  computeWheelSteerAngleRad,
  NEATRACING_MAX_WHEEL_STEER_RAD,
} from './wheel.steer';

/**
 * Mutable car pose bag shared by the opaque car pass and the glass pass.
 * Properties are intentionally writable so one object can be refreshed
 * in place every frame while the composer holds a stable array reference.
 */
export interface MutableNeatRacingCarRenderPose {
  /** Car world X in meters. */
  worldX: number;
  /** Car world Z in meters (scene forward maps to -Z). */
  worldZ: number;
  /** Car yaw around world up in radians. */
  yawRad: number;
  /** Wheel spin in radians shared by all four wheels. */
  wheelSpinRad: number;
  /** Team color id for the AC-014 draw-time remap. */
  teamColorId: number;
  /** Optional front-wheel visual steer angle; omitted means no steer. */
  steerAngleRad?: number;
}

/**
 * Fill a mutable car render pose from the already-transformed scene values.
 *
 * The caller (browser-entry) owns the top-down -> scene coordinate mapping
 * (worldZ and yawRad), wheel-spin accumulation, and team color selection.
 * This helper centralizes the pose-bag shape so both the opaque and glass
 * passes receive the same fields.
 *
 * @param pose - Mutable pose bag to fill in place.
 * @param worldX - Car world X in meters.
 * @param worldZ - Car world Z in meters.
 * @param yawRad - Car yaw around world up in radians.
 * @param wheelSpinRad - Accumulated visual wheel spin in radians.
 * @param teamColorId - Team color id for the AC-014 remap.
 * @param steerInput - Current steering-axis input; clamped to [-1, 1] by
 *   {@link computeWheelSteerAngleRad}. Defaults to 0 when absent.
 */
export function fillCarRenderPose(
  pose: MutableNeatRacingCarRenderPose,
  worldX: number,
  worldZ: number,
  yawRad: number,
  wheelSpinRad: number,
  teamColorId: number,
  steerInput?: number,
): void {
  pose.worldX = worldX;
  pose.worldZ = worldZ;
  pose.yawRad = yawRad;
  pose.wheelSpinRad = wheelSpinRad;
  pose.teamColorId = teamColorId;
  pose.steerAngleRad = computeWheelSteerAngleRad(steerInput ?? 0);
}
