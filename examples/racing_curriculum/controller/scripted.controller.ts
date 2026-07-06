/**
 * Deterministic scripted waypoint-following controller for the Tier 0
 * racing curriculum demo.
 *
 * The controller maintains a target waypoint index that advances around the
 * closed-loop track as the car approaches each endpoint. Steering is a
 * proportional heading-error term; throttle is held at a fixed constant.
 *
 * This controller is intentionally simple and deterministic. It remains useful
 * as a baseline reference lane for regression comparisons against NGE-backed
 * controller behavior while preserving the same `computeScriptedControl`
 * integration seam.
 *
 * @see {@link https://en.wikipedia.org/wiki/Proportional_control Proportional control (Wikipedia)}
 */

import type {
  EnvironmentState,
  CarControlOutput,
} from '../environment/environment.types';
import type { SplineSample, TrackSpec } from '../track/track.generator.types';
import {
  TRACK_SPLINE_SAMPLES_PER_SEGMENT,
  resolveInnerLaneCenterlinePoint,
  resolveSplineSampleFrame,
} from '../track/track.spline.utils';

// ── Constants ─────────────────────────────────────────────────────────────────

/**
 * Distance (world units) within which the car advances to the next waypoint.
 * Set to roughly 10% of a typical segment length for smooth progression.
 */
const WAYPOINT_ADVANCE_RADIUS_WORLD = 16;
/** Number of spline samples to look ahead within the active target segment. */
const SCRIPTED_TARGET_LOOKAHEAD_SAMPLE_COUNT = 4;
/** Forward tangent projection used to keep the scripted target lane-centered. */
const SCRIPTED_TARGET_PROJECTION_DISTANCE_WORLD = 14;

/**
 * Constant forward throttle applied every tick.
 * Range [0, 1] where 1 is maximum forward speed.
 */
const SCRIPTED_THROTTLE_CONSTANT = 0.82;

/**
 * Proportional gain applied to the heading-error term to compute steer.
 * A gain of 2.2 saturates full steer at approximately 26 degrees of heading error.
 */
const STEER_PROPORTIONAL_GAIN = 2.2;
/** Shared empty spline sample used for empty-track safety fallbacks. */
const EMPTY_SPLINE_SAMPLE: SplineSample = {
  x: 0,
  y: 0,
  width: 0,
  segmentIndex: 0,
  sampleIndexWithinSegment: 0,
  globalIndex: 0,
};

// ── Types ─────────────────────────────────────────────────────────────────────

/**
 * Persistent state for the scripted waypoint-following controller.
 *
 * The `targetSegmentIndex` advances monotonically (modulo segment count) as the
 * car reaches successive waypoints around the closed loop.
 */
export interface ScriptedControllerState {
  /** Index of the track segment whose lane-center spline strip is the active target. */
  targetSegmentIndex: number;
}

// ── Public API ────────────────────────────────────────────────────────────────

/**
 * Creates a fresh scripted controller state targeting the first segment.
 *
 * @returns Fresh controller state with `targetSegmentIndex` at 0.
 *
 * @example
 * ```ts
 * const ctrl = createScriptedControllerState();
 * const output = computeScriptedControl(envState, spec, ctrl);
 * ```
 */
export function createScriptedControllerState(): ScriptedControllerState {
  return { targetSegmentIndex: 0 };
}

/**
 * Produces a control output that steers the car toward its next track waypoint.
 *
 * Algorithm:
 * 1. Check whether the car is within `WAYPOINT_ADVANCE_RADIUS_WORLD` of the
 *    current lane-center spline target — if so, advance to the next segment.
 * 2. Compute the heading error from the car's current heading to the angle
 *    toward the lane-centered spline target.
 * 3. Apply a proportional gain and clamp to [-1, 1] for the steer output.
 * 4. Return constant throttle + computed steer.
 *
 * Mutates `controllerState.targetSegmentIndex` in place.
 *
 * @param envState - Current physics state.
 * @param trackSpec - Frozen track geometry.
 * @param controllerState - Mutable controller state (target index advances in place).
 * @returns Control signals for the next simulation step.
 *
 * @example
 * ```ts
 * const output = computeScriptedControl(envState, trackSpec, controllerState);
 * const nextState = stepEnvironment(envState, output);
 * ```
 */
export function computeScriptedControl(
  envState: EnvironmentState,
  trackSpec: TrackSpec,
  controllerState: ScriptedControllerState,
): CarControlOutput {
  const segmentCount = trackSpec.segments.length;
  if (segmentCount === 0 || trackSpec.splineSamples.length === 0) {
    return { throttle: SCRIPTED_THROTTLE_CONSTANT, steer: 0 };
  }

  const closestSplineSampleIndex = findClosestSplineSampleIndex(
    envState,
    trackSpec,
  );
  const closestSplineSample =
    trackSpec.splineSamples[closestSplineSampleIndex] ?? EMPTY_SPLINE_SAMPLE;
  const normalizedTargetSegmentIndex = resolveWrappedSegmentIndex(
    controllerState.targetSegmentIndex,
    segmentCount,
  );
  const currentTargetSplineSample = resolveTargetSplineSample(
    trackSpec,
    closestSplineSampleIndex,
    normalizedTargetSegmentIndex,
  );

  // Step 1: Advance waypoint when the car reaches the current target.
  const distanceToTargetSplineSample = Math.hypot(
    envState.carX - currentTargetSplineSample.x,
    envState.carY - currentTargetSplineSample.y,
  );
  if (distanceToTargetSplineSample < WAYPOINT_ADVANCE_RADIUS_WORLD) {
    controllerState.targetSegmentIndex =
      (normalizedTargetSegmentIndex + 1) % segmentCount;
  } else {
    controllerState.targetSegmentIndex = normalizedTargetSegmentIndex;
  }

  const laneCenteredTargetPoint = resolveProjectedTargetPoint(
    closestSplineSample,
    trackSpec,
  );

  // Step 2: Compute heading error toward the projected lane-centered target.
  const targetDirectionAngle = Math.atan2(
    laneCenteredTargetPoint.y - envState.carY,
    laneCenteredTargetPoint.x - envState.carX,
  );
  const rawHeadingError = targetDirectionAngle - envState.carHeading;
  const normalizedHeadingError = wrapAngleToMinusPiPi(rawHeadingError);

  // Step 3: Proportional steer, clamped to the controller output range.
  const steer = Math.max(
    -1,
    Math.min(1, normalizedHeadingError * STEER_PROPORTIONAL_GAIN),
  );

  return { throttle: SCRIPTED_THROTTLE_CONSTANT, steer };
}

// ── Private helpers ───────────────────────────────────────────────────────────

/**
 * Wraps an angle in radians to the range [-π, π].
 *
 * @param angleRadians - Raw angle in radians (any value).
 * @returns Equivalent angle in [-π, π].
 */
function wrapAngleToMinusPiPi(angleRadians: number): number {
  let wrapped = angleRadians;
  while (wrapped > Math.PI) wrapped -= Math.PI * 2;
  while (wrapped < -Math.PI) wrapped += Math.PI * 2;
  return wrapped;
}

function findClosestSplineSampleIndex(
  envState: EnvironmentState,
  trackSpec: TrackSpec,
): number {
  let bestSampleIndex = 0;
  let bestDistanceWorld = Number.POSITIVE_INFINITY;

  for (const splineSample of trackSpec.splineSamples) {
    const distanceToSampleWorld = Math.hypot(
      splineSample.x - envState.carX,
      splineSample.y - envState.carY,
    );

    if (distanceToSampleWorld < bestDistanceWorld) {
      bestDistanceWorld = distanceToSampleWorld;
      bestSampleIndex = splineSample.globalIndex;
    }
  }

  return bestSampleIndex;
}

function resolveTargetSplineSample(
  trackSpec: TrackSpec,
  closestSplineSampleIndex: number,
  targetSegmentIndex: number,
): SplineSample {
  const closestSplineSample =
    trackSpec.splineSamples[closestSplineSampleIndex] ?? EMPTY_SPLINE_SAMPLE;

  if (closestSplineSample.segmentIndex === targetSegmentIndex) {
    const segmentEndSampleIndex =
      resolveSegmentStartSampleIndex(targetSegmentIndex) +
      TRACK_SPLINE_SAMPLES_PER_SEGMENT -
      1;
    const targetSampleIndex = Math.min(
      segmentEndSampleIndex,
      closestSplineSampleIndex + SCRIPTED_TARGET_LOOKAHEAD_SAMPLE_COUNT,
    );

    return (
      trackSpec.splineSamples[targetSampleIndex] ??
      trackSpec.splineSamples[segmentEndSampleIndex] ??
      closestSplineSample
    );
  }

  return (
    trackSpec.splineSamples[
      resolveSegmentMidpointSampleIndex(targetSegmentIndex)
    ] ?? closestSplineSample
  );
}

function resolveProjectedTargetPoint(
  targetSplineSample: SplineSample,
  trackSpec: TrackSpec,
): { readonly x: number; readonly y: number } {
  const targetSplineSampleFrame = resolveSplineSampleFrame(
    trackSpec.splineSamples,
    targetSplineSample.globalIndex,
  );
  const innerLaneCenterlinePoint = resolveInnerLaneCenterlinePoint(
    targetSplineSample,
    targetSplineSampleFrame,
  );

  return {
    x:
      innerLaneCenterlinePoint.x +
      Math.cos(targetSplineSampleFrame.tangentHeadingRadians) *
        SCRIPTED_TARGET_PROJECTION_DISTANCE_WORLD,
    y:
      innerLaneCenterlinePoint.y +
      Math.sin(targetSplineSampleFrame.tangentHeadingRadians) *
        SCRIPTED_TARGET_PROJECTION_DISTANCE_WORLD,
  };
}

function resolveSegmentMidpointSampleIndex(segmentIndex: number): number {
  return (
    resolveSegmentStartSampleIndex(segmentIndex) +
    Math.floor(TRACK_SPLINE_SAMPLES_PER_SEGMENT / 2)
  );
}

function resolveSegmentStartSampleIndex(segmentIndex: number): number {
  return segmentIndex * TRACK_SPLINE_SAMPLES_PER_SEGMENT;
}

function resolveWrappedSegmentIndex(
  targetSegmentIndex: number,
  segmentCount: number,
): number {
  return (
    ((Math.trunc(targetSegmentIndex) % segmentCount) + segmentCount) %
    segmentCount
  );
}
