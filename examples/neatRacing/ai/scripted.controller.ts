/**
 * The deterministic scripted waypoint-rivals controller for the NeatRacing
 * voxel racer (`examples/neatRacing/ai` — Step 02, slice P2-S2-C).
 *
 * AC-028 (spec §"AC Register"): "a deterministic scripted controller with a
 * per-course speed/throttle table; throttle constant 0.82 + proportional
 * steer". The waypoint advancement, the proportional steer term and the
 * throttle constant are RECYCLED from
 * `examples/racing_curriculum/controller/scripted.controller.ts` with every
 * recycled scale mirrored owner-local VERBATIM (advance radius 16, lookahead
 * 4 samples, target projection 14, throttle constant 0.82, steer gain 2.2 —
 * the shared 2.2 rad⁻¹ heading-error-per-gain convention of
 * BOUNDARY-2). The NEW (E7, NOT recycled) functionality is the per-course
 * throttle table: a deterministic, segment-keyed throttle map derived once
 * per course layout from the published racing-line target speeds, floored by
 * a drivability bound. The controller holds throttle at the table entry for
 * the road segment the car is DRIVING and falls back to the recycled
 * constant for an empty course — this is not new intelligence, just
 * deterministic throttle bookkeeping derived from geometry this module does
 * not own. THE TURN CHANNEL (the round-2 physics finding): a PRESENT
 * `turnRatePercent` is AUTHORITATIVE in the NeatRacing physics (Step 14T/
 * AC-092-AC-093 — a present `0` commands EXACTLY zero turn), so this
 * controller emits NO turn-rate percent at all and the steer axis — the
 * recycled clamped demand — is the SOLE turn channel, resolved by the
 * physics steer-axis law (recycled parity: the recycled demand rides the
 * steer axis).
 *
 * THE ONE SOURCE OF TRUTH (AC-086, LD-30 "track-relative features only"):
 * the throttle table is derived from the PUBLISHED racing-line table
 * (`resolveAutopilotRacingLine`) — the same memoized table the autopilot
 * drives on and the observation assembler reads. There is NO second geometry
 * and NO duplicated curvature math anywhere in this module; the table is a
 * pure deterministic function of the course's spline samples (the same
 * samples always produce the identical table) keyed by each sample's owning
 * chord `segmentIndex` metadata — no spline-density assumption.
 *
 * THE REALM: pure headless seam — no self, no postMessage, no DOM, no RNG,
 * no clock reads. `computeNeatRacingScriptedControl` mutates ONLY the
 * caller-owned controller state (`targetSegmentIndex` advances in place, the
 * recycled mutation seam) and never the car snapshot or the track spec.
 *
 * @module
 */

import type {
  SplineSample,
  TrackSpec,
} from '../../racing_curriculum/track/track.generator.types';
import type {
  PhysicsCarState,
  PhysicsStepInput,
} from '../physics/physics.step.service';
import { NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S } from '../physics/physics.step.service';
import { resolveAutopilotRacingLine } from '../physics/ai.autopilot';

// ── Recycled scales (mirrored owner-local VERBATIM from the racing curriculum
// `scripted.controller.ts`) ───────────────────────────────────────────────────

/**
 * Distance (m) within which the car advances to the next waypoint. Set to
 * roughly 10% of a typical segment length for smooth progression.
 * Recycled value verbatim (`WAYPOINT_ADVANCE_RADIUS_WORLD`).
 */
export const NEATRACING_SCRIPTED_WAYPOINT_ADVANCE_RADIUS_M = 16;

/** Number of spline samples to look ahead within the active target segment.
 * Recycled value verbatim (`SCRIPTED_TARGET_LOOKAHEAD_SAMPLE_COUNT`).
 */
export const NEATRACING_SCRIPTED_TARGET_LOOKAHEAD_SAMPLE_COUNT = 4;

/** Forward tangent projection used to keep the scripted target lane-centered.
 * Recycled value verbatim (`SCRIPTED_TARGET_PROJECTION_DISTANCE_WORLD`).
 */
export const NEATRACING_SCRIPTED_TARGET_PROJECTION_DISTANCE_M = 14;

/**
 * Constant forward throttle scale of the scripted baseline — the throttle
 * ceiling and the empty-course fallback. Recycled value verbatim
 * (`SCRIPTED_THROTTLE_CONSTANT`, AC-028 pins the value 0.82).
 */
export const NEATRACING_SCRIPTED_THROTTLE_CONSTANT = 0.82;

/**
 * Proportional gain applied to the wrapped heading-error term to compute
 * steer. A gain of 2.2 saturates full steer at approximately 26 degrees of
 * heading error. Recycled value verbatim (`STEER_PROPORTIONAL_GAIN`) and the
 * SAME rad⁻¹ convention as `NEATRACING_AUTOPILOT_STEER_GAIN_PER_RAD`
 * (BOUNDARY-2: one gain convention across NeatRacing controllers).
 */
export const NEATRACING_SCRIPTED_STEER_PROPORTIONAL_GAIN = 2.2;

// ── NEW per-course throttle-profile scales (E7 — NOT recycled) ───────────────

/**
 * Lower bound on the per-course throttle factor. The throttle axis is the
 * baseline rivals' ONLY forward-acceleration scale (the lane never brakes),
 * so the floor keeps the weakest corner segment away from a near-zero
 * command and the scripted plan keeps forward progress everywhere.
 */
export const NEATRACING_SCRIPTED_PROFILE_MIN_FACTOR = 0.35;

/** Shared empty spline sample used for empty-course safety fallbacks. */
const EMPTY_SPLINE_SAMPLE: SplineSample = {
  x: 0,
  y: 0,
  width: 0,
  segmentIndex: 0,
  sampleIndexWithinSegment: 0,
  globalIndex: 0,
};

/** Shared empty throttle profile for empty-course fallbacks. */
const EMPTY_THROTTLE_PROFILE: ReadonlyMap<number, number> = new Map<
  number,
  number
>();

/**
 * Per-course throttle-profile memo keyed by the course's sample-list
 * identity. The table is DERIVED once per layout; identical course bytes
 * derive identical entries through distinct memo slots — the profile is a
 * pure deterministic function of the spline samples.
 */
const SCRIPTED_THROTTLE_PROFILE_MEMO = new WeakMap<
  readonly SplineSample[],
  ReadonlyMap<number, number>
>();

/**
 * Per-course segment-strip memo (`segmentIndex` → ordered sample
 * `globalIndex` list) keyed by the course's sample-list identity. Replaces
 * the recycled density-18 segment indexing with the samples' OWN ownership
 * metadata so the waypoint walk never assumes a fixed sample density.
 */
const SCRIPTED_SEGMENT_STRIP_MEMO = new WeakMap<
  readonly SplineSample[],
  ReadonlyMap<number, readonly number[]>
>();

// ── Types ─────────────────────────────────────────────────────────────────────

/**
 * The car snapshot the scripted controller reads — the same Pick shape as the
 * autopilot's `AutopilotCarSnapshot` minus speed (the recycled waypoint
 * baseline does not consume speed).
 */
export type NeatRacingScriptedCarSnapshot = Pick<
  PhysicsCarState,
  'x' | 'y' | 'heading'
>;

/**
 * Persistent state for the scripted waypoint-following controller.
 *
 * The `targetSegmentIndex` advances monotonically (modulo segment count) as
 * the car reaches successive waypoints around the closed loop — the recycled
 * mutation seam, preserved.
 */
export interface NeatRacingScriptedControllerState {
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
 * const state = createNeatRacingScriptedControllerState();
 * const output = computeNeatRacingScriptedControl(car, trackSpec, state);
 * ```
 */
export function createNeatRacingScriptedControllerState(): NeatRacingScriptedControllerState {
  return { targetSegmentIndex: 0 };
}

/**
 * THE PER-COURSE THROTTLE TABLE (AC-028, E7): resolves the deterministic,
 * segment-keyed throttle map for one course layout.
 *
 * For every segment the table entry is the recycled throttle constant scaled
 * by the segment's BINDING speed factor — the minimum published racing-line
 * target speed over the segment's spline samples (the tightest corner within
 * the segment is the constraint) divided by the physics top speed, clamped to
 * `[NEATRACING_SCRIPTED_PROFILE_MIN_FACTOR, 1]`. Straight segments therefore
 * publish exactly the recycled constant; corner segments publish proportionally
 * less, floored for drivability.
 *
 * THE DETERMINISM CONTRACT (drivability gate, AC-031): the table is a pure
 * deterministic function of the course's spline samples — the same sample
 * layout always derives the identical entries — so a scripted-baseline probe
 * episode on the same track tuple reproduces the same throttle plan every
 * run. The result is memoized per sample-list identity; distinct layout bytes
 * get distinct tables (different courses behave differently).
 *
 * @param splineSamples - The course's frozen lane-center spline samples.
 * @returns Readonly map of `segmentIndex` → throttle in
 *   `[0.35 × 0.82, 0.82]` (empty map for an empty course).
 *
 * @example
 * ```ts
 * const profile = resolveNeatRacingScriptedThrottleProfile(trackSpec.splineSamples);
 * const segmentThrottle = profile.get(0) ?? NEATRACING_SCRIPTED_THROTTLE_CONSTANT;
 * ```
 */
export function resolveNeatRacingScriptedThrottleProfile(
  splineSamples: readonly SplineSample[],
): ReadonlyMap<number, number> {
  if (splineSamples.length === 0) {
    return EMPTY_THROTTLE_PROFILE;
  }

  const memoized = SCRIPTED_THROTTLE_PROFILE_MEMO.get(splineSamples);
  if (memoized) {
    return memoized;
  }

  const derived = deriveThrottleProfile(splineSamples);
  SCRIPTED_THROTTLE_PROFILE_MEMO.set(splineSamples, derived);
  return derived;
}

/**
 * Produces the control output that steers a scripted rival toward its next
 * track waypoint — the NeatRacing port of the recycled
 * `computeScriptedControl`.
 *
 * Algorithm (recycled waypoint plan preserved step-for-step):
 * 1. Check whether the car is within `WAYPOINT_ADVANCE_RADIUS_M` of the
 *    current lane-center spline target — if so, advance to the next segment
 *    (modulo segment count); otherwise hold the (wrap-normalized) target.
 * 2. Compute the wrapped heading error from the car's heading to the angle
 *    toward the lane-centered spline target — the recycled aim point of the
 *    CLOSEST spline sample (its inner-lane centerline projected forward
 *    along its own tangent).
 * 3. Apply the proportional gain and clamp to [-1, 1] for the steer output.
 * 4. Throttle reads the per-course table (AC-028) for the road segment the
 *    car is DRIVING — the closest sample's owning `segmentIndex` — falling
 *    back to the recycled constant for an empty course.
 *
 * Mutates ONLY `controllerState.targetSegmentIndex` in place (the recycled
 * mutation seam); the car snapshot and the frozen track spec are never
 * touched.
 *
 * @param car - Current scripted-rival car snapshot (x, y, heading).
 * @param trackSpec - Frozen track geometry (segment count + spline samples).
 * @param controllerState - Mutable controller state (target index advances in place).
 * @returns Physics step input for the next simulation step (no brake —
 *   the scripted baseline is a throttle-only lane; NO turn-rate percent is
 *   emitted because a PRESENT percent is authoritative in the physics —
 *   `0` would command exactly zero turn — so the steer axis is the sole
 *   turn channel).
 *
 * @example
 * ```ts
 * const state = createNeatRacingScriptedControllerState();
 * const input = computeNeatRacingScriptedControl(car, trackSpec, state);
 * ```
 */
export function computeNeatRacingScriptedControl(
  car: NeatRacingScriptedCarSnapshot,
  trackSpec: TrackSpec,
  controllerState: NeatRacingScriptedControllerState,
): PhysicsStepInput {
  const segmentCount = trackSpec.segments.length;
  const splineSamples = trackSpec.splineSamples;
  if (segmentCount === 0 || splineSamples.length === 0) {
    return {
      throttle: NEATRACING_SCRIPTED_THROTTLE_CONSTANT,
      brake: 0,
      steer: 0,
    };
  }

  const throttleProfile = resolveNeatRacingScriptedThrottleProfile(splineSamples);
  const stripMap = resolveSegmentStripMap(splineSamples);
  const closestSplineSampleIndex = findClosestSplineSampleIndex(
    car,
    splineSamples,
  );
  const closestSplineSample =
    resolveSampleAt(splineSamples, closestSplineSampleIndex) ??
    EMPTY_SPLINE_SAMPLE;
  const normalizedTargetSegmentIndex = resolveWrappedSegmentIndex(
    controllerState.targetSegmentIndex,
    segmentCount,
  );
  const currentTargetSplineSample = resolveTargetSplineSample(
    splineSamples,
    stripMap,
    closestSplineSample,
    normalizedTargetSegmentIndex,
  );

  // Step 1: Advance waypoint when the car reaches the current target.
  const distanceToTargetSplineSample = Math.hypot(
    car.x - currentTargetSplineSample.x,
    car.y - currentTargetSplineSample.y,
  );
  if (distanceToTargetSplineSample < NEATRACING_SCRIPTED_WAYPOINT_ADVANCE_RADIUS_M) {
    controllerState.targetSegmentIndex =
      (normalizedTargetSegmentIndex + 1) % segmentCount;
  } else {
    controllerState.targetSegmentIndex = normalizedTargetSegmentIndex;
  }

  const laneCenteredTargetPoint = resolveProjectedTargetPoint(
    closestSplineSample,
    splineSamples,
  );

  // Step 2: Compute heading error toward the projected lane-centered target.
  const targetDirectionAngle = Math.atan2(
    laneCenteredTargetPoint.y - car.y,
    laneCenteredTargetPoint.x - car.x,
  );
  const normalizedHeadingError = wrapAngleToMinusPiPi(
    targetDirectionAngle - car.heading,
  );

  // Step 3: Proportional steer, clamped to the controller output range.
  const steer = Math.max(
    -1,
    Math.min(1, normalizedHeadingError * NEATRACING_SCRIPTED_STEER_PROPORTIONAL_GAIN),
  );

  // Step 4: Per-course throttle for the road segment under the car (AC-028).
  const throttle =
    throttleProfile.get(closestSplineSample.segmentIndex) ??
    NEATRACING_SCRIPTED_THROTTLE_CONSTANT;

  // THE STEER AXIS IS THE SOLE TURN CHANNEL: a PRESENT turnRatePercent is
  // authoritative in the physics (AC-093: a present 0 commands EXACTLY zero
  // turn), so NO percent is emitted and the recycled demand rides the steer
  // axis alone.
  return { throttle, brake: 0, steer };
}

// ── Private helpers ───────────────────────────────────────────────────────────

/** Wraps an angle in radians to the range [-π, π] (recycled helper verbatim). */
function wrapAngleToMinusPiPi(angleRadians: number): number {
  let wrapped = angleRadians;
  while (wrapped > Math.PI) wrapped -= Math.PI * 2;
  while (wrapped < -Math.PI) wrapped += Math.PI * 2;
  return wrapped;
}

/** Wraps a sample index into the sample list (recycled `wrapSplineSampleIndex`). */
function wrapSampleIndex(sampleCount: number, sampleIndex: number): number {
  return ((Math.trunc(sampleIndex) % sampleCount) + sampleCount) % sampleCount;
}

/** Wraps a target segment index into `[0, segmentCount)` (recycled verbatim). */
function resolveWrappedSegmentIndex(
  targetSegmentIndex: number,
  segmentCount: number,
): number {
  return (
    ((Math.trunc(targetSegmentIndex) % segmentCount) + segmentCount) %
    segmentCount
  );
}

/**
 * Finds the closest lane-center spline sample for a car snapshot.
 *
 * Mirrors the recycled nearest-scan exactly: linear scan over the sample
 * list, strict `<` distance comparisons (the FIRST best sample wins ties),
 * returning the sample's `globalIndex`.
 */
function findClosestSplineSampleIndex(
  car: NeatRacingScriptedCarSnapshot,
  splineSamples: readonly SplineSample[],
): number {
  let bestSampleIndex = 0;
  let bestDistanceM = Number.POSITIVE_INFINITY;

  for (const splineSample of splineSamples) {
    const distanceToSampleM = Math.hypot(
      splineSample.x - car.x,
      splineSample.y - car.y,
    );

    if (distanceToSampleM < bestDistanceM) {
      bestDistanceM = distanceToSampleM;
      bestSampleIndex = splineSample.globalIndex;
    }
  }

  return bestSampleIndex;
}

/** Resolves the spline sample at a wrapped `globalIndex`, empty-safe. */
function resolveSampleAt(
  splineSamples: readonly SplineSample[],
  sampleIndex: number,
): SplineSample | undefined {
  if (splineSamples.length === 0) {
    return undefined;
  }
  return splineSamples[wrapSampleIndex(splineSamples.length, sampleIndex)];
}

/**
 * Resolves the per-segment ordered sample strip map for a course, memoized by
 * the sample-list identity. Each strip carries the `globalIndex` values of
 * the samples owned by one chord `segmentIndex`, in list (sample) order —
 * the same order the published racing-line table uses.
 *
 * THE INDEX-SPACE INVARIANT (recycled parity): the recycled controller reads
 * `trackSpec.splineSamples[globalIndex]` directly, i.e. globalIndex doubles
 * as the array position for the ordered production layouts the repo freezes
 * (`resolveSampleAt` applies exactly that recycled wrap). The strips carry
 * `globalIndex` so the waypoint walk resolves through the same index space
 * the recycled plan walks.
 */
function resolveSegmentStripMap(
  splineSamples: readonly SplineSample[],
): ReadonlyMap<number, readonly number[]> {
  const memoized = SCRIPTED_SEGMENT_STRIP_MEMO.get(splineSamples);
  if (memoized) {
    return memoized;
  }

  const strips = new Map<number, number[]>();
  for (const splineSample of splineSamples) {
    const strip = strips.get(splineSample.segmentIndex);
    if (strip) {
      strip.push(splineSample.globalIndex);
    } else {
      strips.set(splineSample.segmentIndex, [splineSample.globalIndex]);
    }
  }

  const stripMap: ReadonlyMap<number, readonly number[]> = strips;
  SCRIPTED_SEGMENT_STRIP_MEMO.set(splineSamples, stripMap);
  return stripMap;
}

/**
 * Resolves the target spline sample for the waypoint plan — the owner-local
 * port of the recycled `resolveTargetSplineSample` without the density-18
 * assumption: when the closest sample already sits in the target segment the
 * target walks the segment's OWN strip `LOOKAHEAD` samples ahead (capped at
 * the strip end); otherwise the target is the midpoint sample of the target
 * segment's strip.
 */
function resolveTargetSplineSample(
  splineSamples: readonly SplineSample[],
  stripMap: ReadonlyMap<number, readonly number[]>,
  closestSplineSample: SplineSample,
  targetSegmentIndex: number,
): SplineSample {
  const targetStrip = stripMap.get(targetSegmentIndex);
  if (closestSplineSample.segmentIndex === targetSegmentIndex && targetStrip) {
    const closestPosition = targetStrip.indexOf(closestSplineSample.globalIndex);
    if (closestPosition >= 0) {
      const targetPosition = Math.min(
        targetStrip.length - 1,
        closestPosition + NEATRACING_SCRIPTED_TARGET_LOOKAHEAD_SAMPLE_COUNT,
      );
      const targetSampleIndex = targetStrip[targetPosition];
      const targetSample =
        targetSampleIndex !== undefined
          ? resolveSampleAt(splineSamples, targetSampleIndex)
          : undefined;
      if (targetSample) {
        return targetSample;
      }
    }
    // The recycled fallback chain never resolves an unrelated sample — when
    // the strip walk cannot yield the lookahead sample the plan holds at the
    // closest sample.
    return closestSplineSample;
  }

  const midpointSampleIndex = targetStrip?.[Math.floor(targetStrip.length / 2)];
  if (midpointSampleIndex !== undefined) {
    return (
      resolveSampleAt(splineSamples, midpointSampleIndex) ?? closestSplineSample
    );
  }
  return closestSplineSample;
}

/** Owner-local mirror of the recycled `SplineSampleFrame` (tangent + left normal). */
function resolveSplineSampleFrame(
  splineSamples: readonly SplineSample[],
  sampleIndex: number,
): {
  readonly tangentHeadingRadians: number;
  readonly normalX: number;
  readonly normalY: number;
} {
  if (splineSamples.length === 0) {
    return { tangentHeadingRadians: 0, normalX: 0, normalY: 1 };
  }

  const wrappedSampleIndex = wrapSampleIndex(splineSamples.length, sampleIndex);
  const previousSample = splineSamples[
    wrapSampleIndex(splineSamples.length, wrappedSampleIndex - 1)
  ]!;
  const nextSample = splineSamples[
    wrapSampleIndex(splineSamples.length, wrappedSampleIndex + 1)
  ]!;
  const tangentDeltaX = nextSample.x - previousSample.x;
  const tangentDeltaY = nextSample.y - previousSample.y;
  const tangentLength = Math.hypot(tangentDeltaX, tangentDeltaY) || 1;
  const tangentUnitX = tangentDeltaX / tangentLength;
  const tangentUnitY = tangentDeltaY / tangentLength;

  return {
    tangentHeadingRadians: Math.atan2(tangentDeltaY, tangentDeltaX),
    normalX: -tangentUnitY,
    normalY: tangentUnitX,
  };
}

/**
 * Resolves the lateral distance from the road centerline to the inner-lane
 * centerline for one sample (recycled helper semantics verbatim).
 */
function resolveInnerLaneCenterlineOffsetWorld(
  splineSample: SplineSample,
): number {
  const laneCount = splineSample.laneCount ?? 2;
  const laneWidthWorld =
    splineSample.laneWidthWorld ?? splineSample.width / laneCount;

  return splineSample.width / 2 - laneWidthWorld / 2;
}

/** Resolves the inner-lane centerline point for one sample (recycled verbatim). */
function resolveInnerLaneCenterlinePoint(
  splineSample: SplineSample,
  splineSampleFrame: ReturnType<typeof resolveSplineSampleFrame>,
): { readonly x: number; readonly y: number } {
  const innerOffsetWorld = resolveInnerLaneCenterlineOffsetWorld(splineSample);

  return {
    x: splineSample.x + splineSampleFrame.normalX * innerOffsetWorld,
    y: splineSample.y + splineSampleFrame.normalY * innerOffsetWorld,
  };
}

/**
 * Projects the lane-centered target forward along the target sample's tangent
 * by the recycled projection distance (recycled `resolveProjectedTargetPoint`).
 */
function resolveProjectedTargetPoint(
  targetSplineSample: SplineSample,
  splineSamples: readonly SplineSample[],
): { readonly x: number; readonly y: number } {
  const targetSplineSampleFrame = resolveSplineSampleFrame(
    splineSamples,
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
        NEATRACING_SCRIPTED_TARGET_PROJECTION_DISTANCE_M,
    y:
      innerLaneCenterlinePoint.y +
      Math.sin(targetSplineSampleFrame.tangentHeadingRadians) *
        NEATRACING_SCRIPTED_TARGET_PROJECTION_DISTANCE_M,
  };
}

/**
 * Derives the per-course throttle table from the PUBLISHED racing line.
 *
 * THE ONE SOURCE OF TRUTH: the published racing-line table
 * (`resolveAutopilotRacingLine`) is read per sample `globalIndex`; each
 * segment's binding speed is the minimum published target speed over the
 * segment's samples; the entry scales the recycled constant with the
 * top-speed-normalized factor clamped to the drivability floor. A sample
 * missing from the published table defaults to the physics top speed (the
 * constant-throttle behavior of a straight).
 */
function deriveThrottleProfile(
  splineSamples: readonly SplineSample[],
): ReadonlyMap<number, number> {
  const racingLine = resolveAutopilotRacingLine(splineSamples);
  const bindingSpeedPerSegment = new Map<number, number>();

  for (const splineSample of splineSamples) {
    const wrappedSampleIndex = wrapSampleIndex(
      splineSamples.length,
      splineSample.globalIndex,
    );
    // Read through the recycled index-space invariant documented on
    // resolveSegmentStripMap: the published racing-line table is
    // array-position-indexed for the frozen production layouts.
    const racingLinePoint = racingLine[wrappedSampleIndex];
    const candidateSpeed =
      racingLinePoint?.targetSpeedUnitsPerS ??
      NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S;
    const currentBindingSpeed = bindingSpeedPerSegment.get(
      splineSample.segmentIndex,
    );
    if (
      currentBindingSpeed === undefined ||
      candidateSpeed < currentBindingSpeed
    ) {
      bindingSpeedPerSegment.set(splineSample.segmentIndex, candidateSpeed);
    }
  }

  const profile = new Map<number, number>();
  for (const [segmentIndex, bindingSpeed] of bindingSpeedPerSegment) {
    const speedFactor = Math.min(
      1,
      Math.max(
        NEATRACING_SCRIPTED_PROFILE_MIN_FACTOR,
        bindingSpeed / NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
      ),
    );
    profile.set(
      segmentIndex,
      NEATRACING_SCRIPTED_THROTTLE_CONSTANT * speedFactor,
    );
  }
  return profile;
}
