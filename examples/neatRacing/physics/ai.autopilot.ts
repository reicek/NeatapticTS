/**
 * NeatRacing AI autopilot (slice P1-physics-drive, user-gap patch batch 108;
 * Step 14M, slice `P1-ai-centerline`: AC-083..AC-086).
 *
 * The step-14B user gap: the served `/` page must open as a running race,
 * not a parked scene. Phase 1 ships one deterministic arcade autopilot that
 * drives the human car around the generated closed loop while the race-mode
 * selector is on the `ai` option; switching to `manual` hands control back
 * to the keyboard sample seam with no other contract change.
 *
 * Pure seam: the function reads only the car's kinematic snapshot and the
 * frozen track spline samples, and returns one analog step input. No
 * mutation, no RNG, no wall-clock reads, so the same (car, samples) pair
 * yields byte-identical inputs for the batch-83 replay contract.
 *
 * Steering law: chase a lookahead sample on the lane centerline ahead of
 * the nearest spline sample, steer proportional to the wrapped heading
 * error (positive error steers counterclockwise per the step-input
 * contract), and brake when the error exceeds a named arcade threshold.
 *
 * THE RACING LINE (Step 14M, AC-083..AC-086): the autopilot now CONSUMES
 * the deterministic racing-line table of
 * `examples/neatRacing/track/track.racing-line.ts` - the SAME table the
 * Phase 2 observation assembler and the scripted controller read, so there
 * is NO second geometry in the repo:
 *
 * - the STEER TARGET is displaced by the table's SIGNED LATERAL OFFSET
 *   (positive to the LEFT of travel, exactly zero on a straight), so the
 *   AI drives the racing line instead of the raw spline point;
 * - a CENTERLINE-CORRECTING steer term is added, proportional to the
 *   car's OWN signed lateral offset to the road centerline and therefore
 *   EXACTLY zero on the centerline (the DECIDED R1 reading of AC-083);
 * - a CURVATURE FEED-FORWARD term is added, proportional to the table's
 *   published signed curvature of the road ahead and therefore exactly
 *   zero on a straight, so the AI commands the steer the corner ahead
 *   actually needs (the arcade model turns at a speed-proportional rate,
 *   so the pure-pursuit chase alone under-steers every corner);
 * - the THROTTLE/BRAKE follow the table's CURVATURE-DERIVED TARGET SPEED,
 *   so the AI slows for the corners ahead instead of entering them at the
 *   top speed (that published target is a BRAKING INTENT - the 12 m
 *   window is shorter than the braking distance, so the brake is issued
 *   before the corner and the car enters it above the target);
 *
 * THE RETAINED LAW: the 12 m lookahead and the 2.2 rad^-1 heading-error
 * gain are UNCHANGED, the named `NEATRACING_AUTOPILOT_*` constants keep
 * their names, and the emergent speed behaviour on a straight is the same
 * as before (full throttle, no brake, at the top speed).
 *
 * Turn-rate channel (Step 14T, AC-093): the autopilot drives the physics
 * through the DEDICATED turn-rate-percent channel, never through the user
 * steer axis. The percent it sets is the SIGNED, BEHAVIOUR-PRESERVING
 * re-expression of the rate the AI already asked for
 * (`100 x steerAI x speedPercent`), so the physics resolves
 * `rate = STEERING_RATE x (percent / 100)` - the AI keeps the FULL range
 * (a percent above 100 commands a rate above the nominal rate) and is
 * EXEMPT from the user steer-axis softening factor.
 *
 * THE COMMAND SUFFICIENCY (Step 14Y, AC-098): the arcade law scales the
 * commanded rate WITH the speed, so the model realizes a CONSTANT 44.5 m
 * minimum turn radius - every corner tighter than that is UNHOLDABLE at
 * ANY speed, and the AI used to saturate its steer axis, brake to a
 * STANDSTILL at the corner entry and never complete a lap. The AI's turn
 * DEMAND is now the UNCLAMPED sum of the road-under-the-car curvature
 * feed-forward and the tracking chase, expressed through the SAME percent
 * channel: the demanded rate is `v x curvature` (the kinematic
 * requirement of the corner the car is ON) plus the tracking correction,
 * so a percent above 100 commands the rate the corner actually needs.
 * The published steer axis stays CLAMPED (the drift and the replay rails
 * read it), so the steer law itself is UNCHANGED - only the RATE it
 * commands grows, and only where the steer axis can no longer express
 * the road.
 *
 * THE PURSUIT EXPECTATION (Step 14Y, AC-098): the road the AI follows
 * turns, so the aim point it chases sits OFF the tangent the car drives
 * on - that offset is EXPECTED geometry and not a driving mistake. The
 * expectation is the racing line's OWN chord angle at the car (the angle
 * between the line's chord to the aim point and the line's own tangent at
 * the car), READ from the same published line the AI aims at - never a
 * formula, never a tuned half-angle. The chase term, the throttle cut and
 * the error brake therefore read the EXCESS error (the heading error
 * minus that expectation), so a car that tracks the line reads ZERO error
 * in a corner and the AI can hold a corner at speed instead of coasting
 * and braking for the road it is following.
 *
 * THE NEVER-STOP FLOOR (Step 14Y, AC-098(c)): at or below the physics'
 * own off-road cruise floor the AI never brakes, and on a road that turns
 * it never coasts either - so the AI can never buy its way out of a
 * corner by stopping in it.
 *
 * THE ROAD-BOUND LINE (Step 14Y, AC-098(d)): the AI drives the RACING
 * LINE the table publishes, and that line is built on the track's SPLINE
 * - but the spline is NOT the road: the road is the CONTROL POLYGON the
 * physics classifies and contains against, and the Catmull-Rom spline
 * that smooths the polygon OVERSHOOTS it at every corner (MEASURED on
 * the served track: 42 of 144 samples, 32.3% of the arc, up to 6.33 m
 * BEYOND the road's own edge). An AI that drives the raw spline therefore
 * drives OFF the road for a third of the lap - onto the grass, where the
 * off-road floor can be lost and never regained - and no lap is ever
 * completed. The line the AI drives is therefore the spline HELD ON THE
 * ROAD: every sample is clamped LATERALLY into the road core the edge
 * seam publishes (the very core the physics classifies as road), so the
 * aim point, the lateral offset, the pursuit expectation and the
 * lookahead walk all read ONE line the car can really drive - while the
 * racing-line TABLE (the curvature, the target speed and the apex
 * offset) keeps being the published one, consumed through the same seam.
 *
 * @module
 */

import {
  NEATRACING_PHYSICS_BRAKING_UNITS_PER_S2,
  NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS,
  NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
  NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S,
  NEATRACING_PHYSICS_STEERING_RATE,
  type PhysicsCarState,
  type PhysicsStepInput,
} from './physics.step.service';
import {
  buildTrackRacingLine,
  NEATRACING_RACING_LINE_STRAIGHT_CURVATURE_PER_M,
  type TrackRacingLinePoint,
} from '../track/track.racing-line';
import { resolveTrackEdgeProfile } from '../track/track.edge';
import { NEATRACING_TRACK_CURB_WIDTH_M } from '../track/track.slabs';
import type {
  SplineSample,
  TrackSegment,
} from '../../racing_curriculum/track/track.generator.types';

/** Lookahead distance along the loop from the nearest spline sample (m). */
export const NEATRACING_AUTOPILOT_LOOKAHEAD_M = 12;

/** Proportional steering gain from wrapped heading error (rad^-1). */
export const NEATRACING_AUTOPILOT_STEER_GAIN_PER_RAD = 2.2;

/** Full-throttle heading-error threshold below which the car holds throttle. */
export const NEATRACING_AUTOPILOT_FULL_THROTTLE_ERROR_RAD = 0.35;

/** Full-brake heading-error threshold above which the car brakes. */
export const NEATRACING_AUTOPILOT_FULL_BRAKE_ERROR_RAD = 0.7;

/**
 * The centerline-correcting steer gain from the signed lateral offset
 * (1/m).
 *
 * THE CENTERLINE-CORRECTING TERM (Step 14M, AC-083): the term is
 * `-gain x the signed lateral offset to the road centerline` (the offset
 * is POSITIVE to the LEFT of travel), so it is EXACTLY zero on the
 * centerline and always steers the car BACK toward the centerline.
 *
 * THE GAIN IS DERIVED, NOT TUNED: it is the RETAINED 2.2 rad^-1 heading
 * gain divided by the RETAINED 12 m lookahead, so the correction is the
 * lateral offset expressed as the heading error the retained lookahead
 * would produce (`offset / lookahead`) and scaled by the retained gain.
 * The classic Stanley cross-track term therefore reuses the two
 * constants the slice RETAINS, and the term stays exactly zero on the
 * centerline because the offset is.
 */
export const NEATRACING_AUTOPILOT_CENTERLINE_STEER_GAIN_PER_M =
  NEATRACING_AUTOPILOT_STEER_GAIN_PER_RAD / NEATRACING_AUTOPILOT_LOOKAHEAD_M;

/**
 * The curvature feed-forward gain of the racing line (m).
 *
 * THE FEED-FORWARD (Step 14M, AC-083..AC-085): the racing line publishes
 * the SIGNED curvature of the road ahead, and the physics realizes a turn
 * radius of `max speed / (steering rate x steer)`, so the steer command
 * that HOLDS a corner of radius `R` is `steer = (max speed / steering
 * rate) x curvature`. The gain is therefore the model's own minimum turn
 * radius, DERIVED from the physics constants - never a tuned number.
 *
 * WHY THE CHASE ALONE IS NOT ENOUGH (MEASURED, not a formula): the
 * arcade model turns at a rate proportional to the speed, so the chase's
 * CONTROL SCALE is `max speed / (steering rate x the 2.2 rad^-1 gain)` =
 * `89 / (2 x 2.2)` = 20.227 m - wider than the retained 12 m lookahead.
 * That scale is a CONTROL magnitude, NOT a curvature law: the
 * pure-pursuit `L^2 / (2R)` offset formula does NOT describe this
 * proportional-heading law, so NO offset is derived from it. The offset
 * was MEASURED instead: on the R = 60 m lap fixture the retained chase
 * law alone settles at a STEADY state of 2.6368 m (inside the 3 m
 * AC-085 bound) but peaks at 3.3230 m on the corner-entry TRANSIENT
 * (outside it), and the PRE-14M model on the same fixture read 6.000 m -
 * the pre-14W containment's drivable edge (the containment is RETIRED
 * since AC-090 VERDICT 8). The feed-forward supplies the corner's own
 * steer command, so the AI holds the corner instead of under-steering
 * toward the band.
 *
 * THE ZERO: the published curvature is EXACTLY zero on a straight, so the
 * term adds nothing on a straight and the emergent straight-line
 * behaviour is unchanged.
 */
export const NEATRACING_AUTOPILOT_CURVATURE_FEEDFORWARD_GAIN_M =
  NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S / NEATRACING_PHYSICS_STEERING_RATE;

/**
 * The brake gain from the relative speed excess over the curvature-derived
 * target speed (1).
 *
 * THE PROPORTIONAL BRAKE (Step 14M, AC-084): the brake is the speed
 * excess over the target speed as a RATIO of that target, so the AI
 * brakes in proportion to how far it is over the target and the brake
 * saturates at the full axis when the car is twice the target speed.
 */
export const NEATRACING_AUTOPILOT_BRAKE_GAIN_PER_SPEED_RATIO = 1;

/** Autopilot throttle axis under steering control (`[0, 1]`). */
export const NEATRACING_AUTOPILOT_THROTTLE = 1;

/** Autopilot brake axis under steering control (`[0, 1]`). */
export const NEATRACING_AUTOPILOT_BRAKE = 1;

/**
 * The percent scale of the AI turn-rate channel (percent per unit of the
 * signed AI turn command).
 *
 * A percent of `100` commands the NOMINAL turn rate (the steering rate),
 * a percent of `0` commands no turn at all and a percent above `100`
 * commands a rate above the nominal one - the AI keeps the full rate
 * range, because the AI channel is never softened.
 */
export const NEATRACING_AUTOPILOT_TURN_RATE_PERCENT_SCALE = 100;

/**
 * The command-sufficiency margin of the AI's turn demand (1).
 *
 * THE MARGIN (Step 14Y, AC-098): the curvature feed-forward supplies the
 * EXACT kinematic requirement of the corner the car is on (`v x
 * curvature`), so a demand built from it alone MEETS the requirement with
 * NO slack - and a demand that merely meets it leaves the corner on the
 * boundary, where any tracking error and any float noise lose the
 * corner. The margin is the small slack that keeps the demanded rate
 * STRICTLY above the requirement, so the corner is HOLDABLE rather than
 * borderline.
 *
 * THE VALUE: a tenth of slack - small enough that the AI still drives the
 * published racing line (the curvature it realizes stays within a tenth
 * of the road's own) and large enough that the demanded rate is never the
 * requirement minus a rounding error.
 */
export const NEATRACING_AUTOPILOT_COMMAND_SUFFICIENCY_MARGIN = 1.1;

/**
 * The AI's NEVER-STOP speed floor (m/s).
 *
 * THE NEVER-STOP FLOOR (Step 14Y, AC-098(c)): the AI never BRAKES a car
 * at or below this speed, and on a road that turns it never COASTS one
 * either, so the AI can never buy its way out of a corner - or out of an
 * off-road excursion - by STOPPING in it.
 *
 * WHY THE FLOOR IS THE PHYSICS' OWN, PLUS ONE BRAKING TICK (MEASURED,
 * not a formula): the physics floors an off-road car only while the car
 * ENTERS the tick at or above `NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S`
 * and is not braking - a coasting tick below that floor lets the car fall through
 * the floor's gate, and the off-road drag is stronger than the engine
 * (17.8 m/s^2 against 44.5 m/s^2), so the car then decays to a
 * standstill on the grass and CANNOT recover.
 *
 * THE AI'S OWN BRAKE IS WHAT STRANDED IT THERE: the brake is NOT
 * floor-protected, so a brake applied AT the floor drops the car BELOW
 * it within a single tick and the floor's gate is lost for good - on the
 * served track the brake measured the car decaying 21.21 -> 18.10 ->
 * 7.42 -> 0.30 -> 0.00 m/s and stopping for good, with the throttle
 * pinned at 1. The floor therefore sits ONE FULL-BRAKING TICK ABOVE the
 * physics' own floor: the model sheds
 * `|NEATRACING_PHYSICS_BRAKING_UNITS_PER_S2| x NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS`
 * = 1.4833 m/s in a single tick, so the AI RELEASES the brake while the
 * car is still above the floor and the physics' own floor holds it there
 * - the AI can never brake a car through the floor it must not lose.
 *
 * THE VALUE IS DERIVED, NOT TUNED: it IS the physics' own off-road
 * cruise floor plus the ONE tick of braking the model can apply inside
 * it, so the AI never drives a car slower than the speed the physics
 * itself refuses to let a car fall below - and never brakes one into
 * that speed either.
 */
export const NEATRACING_AUTOPILOT_MINIMUM_DRIVING_SPEED_UNITS_PER_S =
  NEATRACING_PHYSICS_OFF_ROAD_SPEED_FLOOR_UNITS_PER_S +
  Math.abs(NEATRACING_PHYSICS_BRAKING_UNITS_PER_S2) *
    NEATRACING_PHYSICS_FIXED_TIMESTEP_SECONDS;

/**
 * Neutral step input for degenerate (empty or zero-width) sample lists.
 *
 * The neutral input OWNS the turn-rate channel too: the AI percent is
 * `0`, so the degenerate lane commands NO turn through the AI channel
 * (and never falls back to the user channel by accident).
 */
const NEUTRAL_STEP_INPUT: PhysicsStepInput = {
  throttle: 0,
  brake: 0,
  steer: 0,
  turnRatePercent: 0,
};

/** Minimal kinematic snapshot consumed by the autopilot step (no RNG fields). */
export type AutopilotCarSnapshot = Pick<
  PhysicsCarState,
  'x' | 'y' | 'heading' | 'speed'
>;

/**
 * The per-layout racing-line MEMO (the zero per-frame allocation seam).
 *
 * THE MEMO: the racing-line table is a PURE function of the frozen sample
 * list, so it is built ONCE per track layout and reused for every frame -
 * the per-frame AI path reads the published table and allocates NO TABLE
 * (the per-frame step still builds its own small step-input object and
 * rescans the sample list).
 * The memo is a WeakMap, so a layout is released together with its sample
 * list and the module never keeps a track alive.
 */
const RACING_LINE_MEMO = new WeakMap<
  readonly SplineSample[],
  readonly TrackRacingLinePoint[]
>();

/**
 * Resolves the racing-line table of a track layout (AC-086).
 *
 * THE TRACK-RELATIVE SURFACE: the table is the ONE published surface of
 * the signed lateral offset, the signed curvature profile and the
 * curvature-derived target speed, so the Phase 2 observation assembler
 * (AC-027) and the scripted controller (AC-028) read the SAME table the
 * autopilot drives on - there is NO second geometry and NO second source
 * of truth anywhere in the repo.
 *
 * THE MEMO: the table is built once per layout and cached against the
 * frozen sample list, so the hot path never rebuilds it and never
 * allocates a TABLE (the per-frame step still builds its own small
 * step-input object and rescans the sample list).
 *
 * @param splineSamples - Frozen track spline samples (ordered closed loop).
 * @returns The racing-line points, one per sample, in the sample order.
 * @example
 * const line = resolveAutopilotRacingLine(trackSpec.splineSamples);
 * // line[0].targetSpeedUnitsPerS -> the curvature target speed (m/s)
 */
export function resolveAutopilotRacingLine(
  splineSamples: readonly SplineSample[],
): readonly TrackRacingLinePoint[] {
  const memoized = RACING_LINE_MEMO.get(splineSamples);
  if (memoized !== undefined) {
    return memoized;
  }
  const built = buildTrackRacingLine(splineSamples);
  RACING_LINE_MEMO.set(splineSamples, built);
  return built;
}

/**
 * The ROAD-BOUND LINE memo (the zero per-frame allocation seam).
 *
 * THE MEMO: the road-bound line is a PURE function of the frozen sample
 * list - the control chord ring is reconstructed from the samples' own
 * metadata and every sample is clamped into the road core the edge seam
 * publishes - so it is built ONCE per track layout and reused for every
 * frame, exactly like the racing-line table it is read beside.
 * The memo is a WeakMap, so a layout is released together with its
 * sample list and the module never keeps a track alive.
 */
const ROAD_BOUND_LINE_MEMO = new WeakMap<
  readonly SplineSample[],
  readonly SplineSample[]
>();

/**
 * The CONTROL chord ring the frozen samples describe (the road's own
 * chords).
 *
 * THE RECONSTRUCTION IS BIT-IDENTICAL to the spec's own chord list: the
 * first sample of a control segment carries an interpolation factor of
 * ZERO, so it sits EXACTLY on that segment's control point, and the
 * control chord of segment `k` runs from control point `k` to control
 * point `k + 1` with the width that segment publishes.
 */
interface ControlChordRing {
  /** The reconstructed control chords (the road's own chords). */
  readonly segments: readonly TrackSegment[];
  /** The chord index every ORIGINAL segment index names. */
  readonly chordIndexBySegmentIndex: ReadonlyMap<number, number>;
}

/**
 * Resolves the CONTROL chord ring the frozen samples describe.
 *
 * THE RING (Step 14Y, AC-098(d)): the road the physics classifies and
 * contains against is the CONTROL POLYGON, and the samples name that
 * polygon's chords through their own metadata - so the AI rebuilds the
 * very chords the spec published instead of measuring the road a second
 * time.
 *
 * THE FALLBACK: a sample list that names a SINGLE control segment
 * describes no control ring of its own (the hand-authored fixture rings
 * do), so the samples' OWN ring is the only ring it describes - and the
 * clamp is INACTIVE there (a sample sits ON the chord that starts at
 * it), so those layouts drive the very line they always drove.
 *
 * @param samples - Frozen track spline samples.
 * @returns The control chord ring, or `null` when the samples name no
 *   control ring of their own.
 */
function resolveControlChordRing(
  samples: readonly SplineSample[],
): ControlChordRing | null {
  const controlPointBySegmentIndex = new Map<number, SplineSample>();
  for (const sample of samples) {
    if (!controlPointBySegmentIndex.has(sample.segmentIndex)) {
      controlPointBySegmentIndex.set(sample.segmentIndex, sample);
    }
  }
  const segmentIndices = [...controlPointBySegmentIndex.keys()].sort(
    (left, right) => left - right,
  );
  if (segmentIndices.length < 2) {
    return null;
  }
  const chordIndexBySegmentIndex = new Map<number, number>();
  const controlPoints: SplineSample[] = [];
  for (const segmentIndex of segmentIndices) {
    const controlPoint = controlPointBySegmentIndex.get(segmentIndex);
    if (controlPoint === undefined) {
      continue;
    }
    chordIndexBySegmentIndex.set(segmentIndex, controlPoints.length);
    controlPoints.push(controlPoint);
  }
  const count = controlPoints.length;
  const segments: TrackSegment[] = [];
  for (let index = 0; index < count; index += 1) {
    const controlPoint = controlPoints[index];
    const nextControlPoint = controlPoints[(index + 1) % count];
    segments.push({
      startX: controlPoint.x,
      startY: controlPoint.y,
      endX: nextControlPoint.x,
      endY: nextControlPoint.y,
      width: controlPoint.width,
    });
  }
  return { segments, chordIndexBySegmentIndex };
}

/**
 * Builds the ROAD-BOUND LINE of a layout: the frozen samples with every
 * sample clamped LATERALLY into the road core the edge seam publishes.
 *
 * THE CLAMP (Step 14Y, AC-098(d)): the sample's signed offset from its
 * OWN control chord (the road's centreline, positive to the LEFT of
 * travel) is clamped into the road core of the side the sample sits on -
 * the published half-width of that side MINUS the curb band, because the
 * physics classifies a point as ROAD only inside that core. The sample
 * is then moved LATERALLY onto its clamped offset, so the line keeps the
 * spline's shape and its parametrisation and only the off-road
 * overshoot is shaved.
 *
 * THE SEAM, NOT A SECOND GEOMETRY (AC-082 / AC-086): the bound is the
 * one the edge seam publishes - the loop-global half-width PAIR of the
 * chord the sample owns - so the line the AI drives can never be off the
 * road the physics reads.
 *
 * THE INACTIVE CLAMP IS BIT-IDENTICAL: a sample that is already inside
 * the road core keeps its EXACT published coordinates (the very same
 * sample object), so a layout whose spline already lies on its road
 * drives the very line it always drove.
 *
 * @param samples - Frozen track spline samples.
 * @returns The road-bound line, one sample per published sample.
 */
function buildRoadBoundLine(
  samples: readonly SplineSample[],
): readonly SplineSample[] {
  const ring = resolveControlChordRing(samples);
  if (ring === null) {
    return samples;
  }
  const profile = resolveTrackEdgeProfile(ring.segments);
  const outerSideSign = profile.outerSideSign;
  return samples.map((sample) => {
    const chordIndex = ring.chordIndexBySegmentIndex.get(sample.segmentIndex);
    const chord =
      chordIndex === undefined ? undefined : ring.segments[chordIndex];
    const edge =
      chordIndex === undefined ? undefined : profile.chords[chordIndex];
    if (chord === undefined || edge === undefined) {
      return sample;
    }
    const chordX = chord.endX - chord.startX;
    const chordY = chord.endY - chord.startY;
    const chordLength = Math.hypot(chordX, chordY);
    if (chordLength === 0) {
      return sample;
    }
    // THE CHORD'S LEFT NORMAL: the same convention the racing line's
    // published offset uses (positive to the LEFT of travel).
    const normalX = -chordY / chordLength;
    const normalY = chordX / chordLength;
    const offsetM =
      (sample.x - chord.startX) * normalX + (sample.y - chord.startY) * normalY;
    // THE PUBLISHED HALF-WIDTH OF THE SIDE THE SAMPLE SITS ON: the
    // loop-global OUTER side is the widened one (the published drift
    // room) and the inner side keeps the plain half-width.
    const halfWidthM =
      Math.sign(offsetM) === outerSideSign
        ? edge.outerHalfWidthM
        : edge.innerHalfWidthM;
    // THE ROAD CORE: the physics calls a point ROAD only inside the
    // drivable edge MINUS the curb band, so the line is held inside the
    // road core and the curb band is left as the tracking buffer.
    const boundM = halfWidthM - NEATRACING_TRACK_CURB_WIDTH_M;
    const clampedOffsetM = Math.max(-boundM, Math.min(boundM, offsetM));
    if (clampedOffsetM === offsetM) {
      return sample;
    }
    return {
      ...sample,
      x: sample.x + normalX * (clampedOffsetM - offsetM),
      y: sample.y + normalY * (clampedOffsetM - offsetM),
    };
  });
}

/**
 * Resolves the ROAD-BOUND LINE of a track layout (the line the AI drives).
 *
 * @param splineSamples - Frozen track spline samples (ordered closed loop).
 * @returns The road-bound line, one sample per published sample.
 * @example
 * const line = resolveRoadBoundLine(trackSpec.splineSamples);
 * // line[i] is the published sample, held inside the published road core
 */
function resolveRoadBoundLine(
  splineSamples: readonly SplineSample[],
): readonly SplineSample[] {
  const memoized = ROAD_BOUND_LINE_MEMO.get(splineSamples);
  if (memoized !== undefined) {
    return memoized;
  }
  const built = buildRoadBoundLine(splineSamples);
  ROAD_BOUND_LINE_MEMO.set(splineSamples, built);
  return built;
}

/**
 * Wraps an angle into `[-π, π)` via the atan2 half-plane projection so
 * steering errors stay continuous across the ±π discontinuity.
 *
 * @param radians - Raw angle difference in radians.
 * @returns Angle wrapped into `[-π, π)`.
 */
function wrapSignedAngle(radians: number): number {
  return Math.atan2(Math.sin(radians), Math.cos(radians));
}

/**
 * Clamps a steering command into the analog axis range.
 *
 * @param value - Raw steering command.
 * @returns Value clamped into `[-1, 1]`.
 */
function clampSteer(value: number): number {
  return Math.max(-1, Math.min(1, value));
}

/**
 * Clamps a control ratio into the closed unit interval.
 *
 * @param value - Raw ratio.
 * @returns Value clamped into `[0, 1]`.
 */
function clampRatio(value: number): number {
  return Math.max(0, Math.min(1, value));
}

/**
 * Resolves the normalized speed percentile of a snapshot speed.
 *
 * THE SAME PERCENTILE (Step 14T, AC-093): the AI reads the speed
 * percentile the physics service resolves for its own laws - the same
 * ratio, clamped the same way - so the AI's behaviour-preserving
 * re-expression resolves to exactly the turn rate the AI commanded
 * before the channel was adopted.
 *
 * @param speed - The car speed in units per second.
 * @returns The normalized speed in `[0, 1]`.
 */
function resolveSpeedPercent(speed: number): number {
  return clampRatio(speed / NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S);
}

/**
 * Resolves the SIGNED turn-rate percent the autopilot commands.
 *
 * THE DEMAND (Step 14Y, AC-098): the percent is the signed re-expression
 * of the AI's UNCLAMPED turn DEMAND - the pursuit chase plus the road's
 * own curvature feed-forward - so the physics service resolves
 * `rate = STEERING_RATE x (percent / 100)`.
 *
 * THE UNCLAMPED DEMAND: the demand is NEVER clamped to the steer axis,
 * because that clamp is what made every corner tighter than the model's
 * 44.5 m minimum turn radius unholdable (a clamped steer axis caps the
 * rate at `v / 44.5` whatever the road asks for). The demand is the SUM
 * of the two terms the steer law already reads, so wherever the steer
 * axis can express the demand the two numbers are the SAME and the
 * percent stays BIT-IDENTICAL to the pre-14Y percent - the demand only
 * exceeds the axis where the road asks for MORE than the axis can
 * express.
 *
 * @param demand - The unclamped turn demand (the pursuit chase plus the
 *   curvature feed-forward of the road under the car).
 * @param speedPercent - The normalized speed in `[0, 1]`.
 * @returns The signed turn-rate percent.
 */
function resolveTurnRatePercent(demand: number, speedPercent: number): number {
  return NEATRACING_AUTOPILOT_TURN_RATE_PERCENT_SCALE * demand * speedPercent;
}

/**
 * Resolves the index of the spline sample nearest to the car (squared
 * distance scan over the frozen list; the sample count is small enough
 * that a linear scan beats a maintained spatial index for this arcade
 * autopilot).
 *
 * @param samples - Frozen track spline samples.
 * @param x - Car world x.
 * @param y - Car world y.
 * @returns The nearest sample index (0 when the list is empty).
 */
function nearestSampleIndex(
  samples: readonly SplineSample[],
  x: number,
  y: number,
): number {
  let best = 0;
  let bestDistance = Infinity;
  for (let index = 0; index < samples.length; index += 1) {
    const sample = samples[index];
    const dx = sample.x - x;
    const dy = sample.y - y;
    const distance = dx * dx + dy * dy;
    if (distance < bestDistance) {
      bestDistance = distance;
      best = index;
    }
  }
  return best;
}

/**
 * Resolves the lookahead spline sample index a named distance ahead of the
 * nearest sample along the closed loop (wrapped walk accumulating the
 * centerline chord lengths).
 *
 * @param samples - Frozen track spline samples.
 * @param nearestIndex - Index of the nearest sample (the walk origin).
 * @returns The index of the lookahead sample.
 */
function lookaheadSampleIndex(
  samples: readonly SplineSample[],
  nearestIndex: number,
): number {
  const total = samples.length;
  let remaining = NEATRACING_AUTOPILOT_LOOKAHEAD_M;
  let cursor = nearestIndex;
  for (let steps = 0; steps < total; steps += 1) {
    const from = samples[cursor];
    const to = samples[(cursor + 1) % total];
    const dx = to.x - from.x;
    const dy = to.y - from.y;
    remaining -= Math.hypot(dx, dy);
    cursor = (cursor + 1) % total;
    if (remaining <= 0) {
      break;
    }
  }
  return cursor;
}

/**
 * Measures the SIGNED lateral offset of a point from the road centerline.
 *
 * THE MEASUREMENT: the offset is the signed perpendicular distance to the
 * centerline chord at the nearest spline sample, POSITIVE to the LEFT of
 * the travel direction (the repo's left-normal convention). It is
 * measured DIRECTLY from the samples - never inferred from any clamp or
 * seam (the containment is RETIRED: AC-090 VERDICT 8) - so an off-center
 * car always reads off-center.
 *
 * THE ZERO: a car that sits exactly on the centerline chord reads EXACTLY
 * zero, which is what makes the centerline-correcting steer term zero on
 * the centerline.
 *
 * @param samples - Frozen track spline samples.
 * @param nearestIndex - Index of the nearest sample (the chord origin).
 * @param x - Car world x.
 * @param y - Car world y.
 * @returns The signed lateral offset (m); positive to the LEFT.
 */
function signedLateralOffsetM(
  samples: readonly SplineSample[],
  nearestIndex: number,
  x: number,
  y: number,
): number {
  const from = samples[nearestIndex];
  const to = samples[(nearestIndex + 1) % samples.length];
  const dx = to.x - from.x;
  const dy = to.y - from.y;
  const lengthSquared = dx * dx + dy * dy;
  if (lengthSquared === 0) {
    return 0;
  }
  const length = Math.sqrt(lengthSquared);
  const projection = Math.max(
    0,
    Math.min(1, ((x - from.x) * dx + (y - from.y) * dy) / lengthSquared),
  );
  const ex = x - (from.x + projection * dx);
  const ey = y - (from.y + projection * dy);
  return (ex * -dy + ey * dx) / length;
}

/**
 * The lookahead WINDOW reading of the racing line.
 */
interface LookaheadWindow {
  /** The curvature-derived TARGET SPEED of the window (m/s). */
  readonly targetSpeedUnitsPerS: number;
  /**
   * The SIGNED curvature of the TIGHTEST corner in the window (1/m).
   *
   * THE TIGHTEST CORNER (Step 14Y, AC-098): the window is the road the
   * AI is ABOUT TO drive, so the TIGHTEST curvature in it is the corner
   * the AI has to be ready for - the lookahead POINT's own curvature is
   * the corner the aim point happens to sit on, which is already BEHIND
   * the tightest corner the window spans.
   */
  readonly tightestCurvaturePerM: number;
  /**
   * The distance from the car to the CONSTRAINED corner of the window (m).
   *
   * THE CONSTRAINED CORNER (Step 14Y, AC-098): the corner the window's
   * target speed comes from - the sample whose curvature is the tightest
   * one the window spans, and therefore the sample that asks for the
   * lowest target speed. The distance is the ARC LENGTH from the car's
   * own sample to that sample along the samples the AI drives, so the
   * brake can be sized against the distance the AI still has to cover
   * before it has to be down to that speed.
   */
  readonly constrainedDistanceM: number;
}

/**
 * Resolves the lookahead WINDOW reading of the racing line.
 *
 * THE WINDOW: the samples from the nearest one up to (and including) the
 * lookahead one, so the TIGHTEST curvature in the window ahead of the car
 * sets the target speed - the AI slows for a corner while it is still
 * straight, not once it is already in the corner.
 *
 * THE TARGET IS A BRAKING INTENT, AND THE BRAKE IS SIZED FOR IT (Step
 * 14Y, AC-098): the window is 12 m long, while the arcade brake rate
 * (`NEATRACING_PHYSICS_BRAKING_UNITS_PER_S2` = -89 m/s^2) needs
 * `v^2 / (2 x 89)` of distance to shed the excess - 31.1 m to drop from
 * 89 m/s to the R = 60 m fixture's 48.75 m/s target, and about 18.5 m to
 * drop to the demo track's ~68 m/s target. Both are LONGER than the
 * 12 m window, so a WEAK brake reaches the corner far above the
 * published target - and above the target the model's grip can no longer
 * hold, so the AI runs wide. The brake is therefore sized against the
 * distance the AI still has (the arc length to the constrained corner),
 * so it sheds the speed as fast as the model allows and arrives at the
 * published target. The window stays 12 m long (the Step 14M lanes pin
 * it): the DISTANCE to the constrained corner inside that window is what
 * the brake is sized against, and the surplus beyond it is made up on
 * the way in.
 *
 * THE ONE SCAN (Step 14Y, AC-098): the window is scanned ONCE for all
 * three readings - the target speed, the tightest corner it spans and the
 * distance to that corner.
 *
 * @param samples - Frozen track spline samples.
 * @param racingLine - The published racing-line table of the layout.
 * @param nearestIndex - Index of the nearest sample (the window origin).
 * @param lookaheadIndex - Index of the lookahead sample (the window end).
 * @returns The window readings (the target speed, the tightest corner
 *   and the distance to that corner).
 */
function resolveLookaheadWindow(
  samples: readonly SplineSample[],
  racingLine: readonly TrackRacingLinePoint[],
  nearestIndex: number,
  lookaheadIndex: number,
): LookaheadWindow {
  const count = racingLine.length;
  if (count === 0) {
    return {
      targetSpeedUnitsPerS: NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S,
      tightestCurvaturePerM: 0,
      constrainedDistanceM: 0,
    };
  }
  let targetSpeedUnitsPerS = NEATRACING_PHYSICS_MAX_SPEED_UNITS_PER_S;
  let tightestCurvaturePerM = 0;
  let constrainedDistanceM = 0;
  let distanceM = 0;
  let index = nearestIndex;
  for (let step = 0; step <= count; step += 1) {
    const point = racingLine[index];
    if (
      point !== undefined &&
      point.targetSpeedUnitsPerS < targetSpeedUnitsPerS
    ) {
      targetSpeedUnitsPerS = point.targetSpeedUnitsPerS;
      // THE CONSTRAINED CORNER: the sample the lowest target speed comes
      // from is the tightest corner the window spans (the target speed is
      // monotonic in the curvature magnitude), so the distance to it is
      // the distance the AI still has to cover before it has to be down
      // to that speed.
      constrainedDistanceM = distanceM;
    }
    // THE TIGHTEST CORNER: the window's own scan already reads every
    // published curvature it spans, so the tightest one comes from the
    // SAME scan - the corner the AI has to be READY for is the tightest
    // corner the window spans, not the corner the aim point happens to
    // sit on (the aim point is the LAST sample of the window, so its
    // curvature is the corner the AI is ALREADY leaving).
    if (
      point !== undefined &&
      Math.abs(point.curvaturePerM) > Math.abs(tightestCurvaturePerM)
    ) {
      tightestCurvaturePerM = point.curvaturePerM;
    }
    if (index === lookaheadIndex) {
      break;
    }
    // THE ARC LENGTH: the distance to the NEXT sample is the chord
    // between the two centreline samples, so the distance to the
    // constrained corner is the road the AI still has to cover.
    const next = samples[(index + 1) % count];
    const current = samples[index];
    if (next !== undefined && current !== undefined) {
      distanceM += Math.hypot(next.x - current.x, next.y - current.y);
    }
    index = (index + 1) % count;
  }
  return {
    targetSpeedUnitsPerS,
    tightestCurvaturePerM,
    constrainedDistanceM,
  };
}

/**
 * A world point of the racing line (an offset-displaced sample).
 */
interface RacingLinePointXY {
  /** The world x. */
  readonly x: number;
  /** The world y. */
  readonly y: number;
}

/**
 * Resolves the racing line's OWN point at a sample index.
 *
 * THE POINT: the centerline sample displaced by the racing line's
 * published SIGNED LATERAL OFFSET along the chord's left normal - the
 * very point the AI aims at, so the pursuit expectation is measured
 * along the SAME line the AI follows.
 *
 * @param samples - Frozen track spline samples.
 * @param racingLine - The published racing-line table of the layout.
 * @param index - The sample index.
 * @returns The racing line's own point at the index.
 */
function racingLinePointOf(
  samples: readonly SplineSample[],
  racingLine: readonly TrackRacingLinePoint[],
  index: number,
): RacingLinePointXY {
  const count = samples.length;
  const sample = samples[index];
  const next = samples[(index + 1) % count];
  const dx = next.x - sample.x;
  const dy = next.y - sample.y;
  const length = Math.hypot(dx, dy);
  const offsetM = racingLine[index]?.offsetM ?? 0;
  if (length === 0) {
    return { x: sample.x, y: sample.y };
  }
  return {
    x: sample.x + (-dy / length) * offsetM,
    y: sample.y + (dx / length) * offsetM,
  };
}

/**
 * Resolves the PURSUIT EXPECTATION of the racing line (rad).
 *
 * THE EXPECTATION (Step 14Y, AC-098): the road the AI follows turns, so
 * the aim point it chases sits OFF the tangent the car drives on - that
 * offset is EXPECTED geometry and not a driving mistake. The expectation
 * is that EXPECTED angle, read from the racing line itself: the angle
 * between the line's own chord (from the line's point at the car to the
 * aim point) and the line's own tangent at the car.
 *
 * WHY IT IS READ, NOT FORMULATED (Step 14Y, AC-098): the tangent-chord
 * theorem holds for a CONSTANT curvature, while the published line's
 * curvature VARIES along the window - so the expectation is measured on
 * the line the AI actually follows, and a car that tracks the line reads
 * EXACTLY zero error in every corner (the half-angle of the window's own
 * turn over-read the corner and drove the AI into it).
 *
 * @param samples - Frozen track spline samples.
 * @param racingLine - The published racing-line table of the layout.
 * @param nearestIndex - Index of the nearest sample (the car's sample).
 * @param aimX - The world x of the aim point.
 * @param aimY - The world y of the aim point.
 * @returns The signed pursuit expectation (rad).
 */
function resolvePursuitExpectationRad(
  samples: readonly SplineSample[],
  racingLine: readonly TrackRacingLinePoint[],
  nearestIndex: number,
  aimX: number,
  aimY: number,
): number {
  const count = samples.length;
  const own = racingLinePointOf(samples, racingLine, nearestIndex);
  const next = racingLinePointOf(
    samples,
    racingLine,
    (nearestIndex + 1) % count,
  );
  return wrapSignedAngle(
    Math.atan2(aimY - own.y, aimX - own.x) -
      Math.atan2(next.y - own.y, next.x - own.x),
  );
}

/**
 * Computes one analog physics step input that chases the RACING LINE:
 * steer proportional to the wrapped heading error toward the racing-line
 * steer target (displaced by the signed lateral offset) MINUS the
 * centerline-correcting term, and throttle or brake from the
 * curvature-derived target speed and the named error thresholds.
 *
 * THE TURN-RATE CHANNEL (Step 14T, AC-093): the input ALWAYS carries a
 * DEFINED, signed turn-rate percent - the behaviour-preserving
 * re-expression of the AI's own turn (`100 x steerAI x speedPercent`) -
 * so the physics drives the AI through the AI channel and the AI is
 * exempt from the user steer-axis softening factor.
 *
 * THE RACING LINE (Step 14M, AC-083..AC-086): the seam reads the ONE
 * published racing-line table of the layout (memoized, so the hot path
 * allocates NO TABLE - it still builds its own small step-input object
 * and rescans the sample list) - the steer target is displaced by the
 * table's SIGNED LATERAL OFFSET, the steer command gains the CURVATURE
 * FEED-FORWARD of the road ahead, the throttle and the brake follow its
 * CURVATURE-DERIVED TARGET SPEED, and the car's own signed lateral
 * offset adds the centerline-correcting steer term (EXACTLY zero on the
 * centerline).
 *
 * THE COMMAND SUFFICIENCY (Step 14Y, AC-098): the STEER AXIS keeps the
 * RETAINED clamped law (the published steer rail the drift and the
 * replay read), while the TURN DEMAND the percent channel carries is the
 * UNCLAMPED sum of the same two terms with the road-under-the-car
 * curvature - so the commanded rate is `v x curvature` (the kinematic
 * requirement of the corner the car is ON) plus the pursuit correction,
 * and a corner tighter than the model's 44.5 m minimum turn radius is
 * holdable at the speed the AI drives it at.
 *
 * THE EXCESS ERROR (Step 14Y, AC-098): the throttle cut and the error
 * brake read the EXCESS heading error - the heading error minus the
 * pursuit expectation, the racing line's OWN chord angle at the car - so
 * the AI never coasts and brakes for the corner it is following.
 *
 * THE NEVER-STOP FLOOR (Step 14Y, AC-098(c)): at or below the physics'
 * own off-road cruise floor the AI never brakes, and on a road that turns
 * it never coasts either - so the AI can never stop in a corner.
 *
 * Deterministic pure seam — no RNG, no clock reads, no mutation.
 *
 * @param car - Kinematic car snapshot (x, y, heading, speed).
 * @param splineSamples - Frozen track spline samples (ordered closed loop).
 * @returns One step input for the 60 Hz fixed step.
 * @example
 * const input = stepAutopilotInput(car, trackSpec.splineSamples);
 * // { throttle: 1, brake: 0, steer: 0.21, turnRatePercent: 12.3 }
 * // chasing the racing line of the loop
 */
export function stepAutopilotInput(
  car: AutopilotCarSnapshot,
  splineSamples: readonly SplineSample[],
): PhysicsStepInput {
  if (splineSamples.length === 0) {
    return NEUTRAL_STEP_INPUT;
  }
  const racingLine = resolveAutopilotRacingLine(splineSamples);
  // THE ROAD-BOUND LINE (Step 14Y, AC-098(d)): the line the AI drives is
  // the spline HELD ON THE ROAD, so every geometry read below (the
  // nearest sample, the lookahead walk, the aim point, the lateral
  // offset, the window and the pursuit expectation) reads ONE line the
  // car can really stay on - while the racing-line TABLE stays the
  // published one.
  const line = resolveRoadBoundLine(splineSamples);
  const nearest = nearestSampleIndex(line, car.x, car.y);
  const lookaheadIndex = lookaheadSampleIndex(line, nearest);
  const target = line[lookaheadIndex];
  const offsetM = racingLine[lookaheadIndex]?.offsetM ?? 0;
  const chordTo = line[(lookaheadIndex + 1) % line.length];
  const chordX = chordTo.x - target.x;
  const chordY = chordTo.y - target.y;
  const chordLength = Math.sqrt(chordX * chordX + chordY * chordY);
  let aimX = target.x;
  let aimY = target.y;
  if (chordLength > 0) {
    // THE LEFT NORMAL of the travel direction: the racing-line offset is
    // positive to the LEFT, so the steer target moves with it.
    aimX += (-chordY / chordLength) * offsetM;
    aimY += (chordX / chordLength) * offsetM;
  }
  const desiredHeading = Math.atan2(aimY - car.y, aimX - car.x);
  const headingError = wrapSignedAngle(desiredHeading - car.heading);
  const lateralOffsetM = signedLateralOffsetM(line, nearest, car.x, car.y);
  const window = resolveLookaheadWindow(
    line,
    racingLine,
    nearest,
    lookaheadIndex,
  );
  // THE PURSUIT EXPECTATION (Step 14Y, AC-098): the racing line's OWN
  // chord angle at the car - the geometry the line itself contributes to
  // the heading error the AI reads.
  const pursuitExpectationRad = resolvePursuitExpectationRad(
    line,
    racingLine,
    nearest,
    aimX,
    aimY,
  );
  // THE EXCESS ERROR: the heading error minus the expectation - the part
  // of the error the AI really has to correct.
  const trackingErrorRad = wrapSignedAngle(
    headingError - pursuitExpectationRad,
  );
  const absoluteTrackingErrorRad = Math.abs(trackingErrorRad);
  // THE ROAD UNDER THE CAR (Step 14Y, AC-098): the published curvature at
  // the car's OWN sample - the corner the car is ON, not the corner it is
  // about to reach.
  //
  // WHY THE ROAD UNDER THE CAR (MEASURED, not a formula): the demand used
  // to read the TIGHTEST curvature of the 12 m window ahead, so the AI
  // pre-turned the whole window and wove across the road (the R = 8 m
  // fixture measured 9 m of lateral offset, outside the 3 m AC-085
  // bound). The road under the car is the road the AI is DRIVING, and the
  // racing line's published curvature is the smoothed profile of exactly
  // that road, so the feed-forward answers the corner the AI is in.
  const curvatureUnderPerM = racingLine[nearest]?.curvaturePerM ?? 0;
  // THE CHASE (the RETAINED seam): the heading-error term and the
  // centerline-correcting term, read exactly as before - except that the
  // heading-error term reads the EXCESS error, so the corner the road
  // itself contributes is never read as a driving mistake.
  const chaseDemand =
    trackingErrorRad * NEATRACING_AUTOPILOT_STEER_GAIN_PER_RAD -
    lateralOffsetM * NEATRACING_AUTOPILOT_CENTERLINE_STEER_GAIN_PER_M;
  // THE FEED-FORWARD: the road under the car, scaled by the model's own
  // minimum turn radius and by the command-sufficiency margin - so the
  // demanded rate is the kinematic requirement of the corner the car is
  // ON (`v x curvature`), strictly met.
  const feedForwardDemand =
    curvatureUnderPerM *
    NEATRACING_AUTOPILOT_CURVATURE_FEEDFORWARD_GAIN_M *
    NEATRACING_AUTOPILOT_COMMAND_SUFFICIENCY_MARGIN;
  const steer = clampSteer(chaseDemand + feedForwardDemand);
  // THE TURN DEMAND (Step 14Y, AC-098): the SAME two terms the steer
  // axis reads - the tracking chase plus the curvature feed-forward of
  // the road UNDER the car - but the demand is NEVER clamped, so the rate
  // the AI commands can hold a corner the steer axis alone can no longer
  // express.
  //
  // WHEREVER THE AXIS CAN EXPRESS THE DEMAND THE TWO NUMBERS ARE THE
  // SAME, and where the axis saturates the demand still carries the full
  // corner - so the published steer rail (the drift and the replay read
  // it) stays the clamped law and only the RATE grows.
  const turnDemand = chaseDemand + feedForwardDemand;
  const speedExcessUnitsPerS = car.speed - window.targetSpeedUnitsPerS;
  // THE PROPORTIONAL SPEED BRAKE (the RETAINED law): the brake asks for
  // the excess over the window's target speed as a ratio of that target.
  const proportionalSpeedBrake =
    speedExcessUnitsPerS > 0
      ? clampRatio(
          (speedExcessUnitsPerS / window.targetSpeedUnitsPerS) *
            NEATRACING_AUTOPILOT_BRAKE_GAIN_PER_SPEED_RATIO,
        )
      : 0;
  // THE BRAKE SUFFICIENCY (Step 14Y, AC-098): the deceleration the AI
  // actually NEEDS to be at the window's target speed by the time it
  // reaches the constrained corner - `a = (v^2 - v_target^2) / (2 x d)` -
  // expressed as a ratio of the model's OWN braking rate, so the brake
  // asks for exactly as much deceleration as the distance allows.
  //
  // WHY (MEASURED, not a formula): the proportional brake only ever asked
  // for a fraction of the model's braking - the R = 60 m fixture measured
  // 0.27 of the axis - so the AI entered that corner at 64 m/s against a
  // published target of 48.7 m/s. Above the target the model's grip can
  // no longer hold the corner (the path direction lagged the heading by
  // 0.31 rad there), so the AI ran 3.13 m wide - outside the 3 m AC-085
  // bound. The brake is now sized against the distance the AI still has,
  // so it sheds the speed as fast as the model allows and ARRIVES at the
  // published target: the AI's target speed stays REACHABLE (AC-098).
  const requiredDecelerationUnitsPerS2 =
    window.constrainedDistanceM > 0 && speedExcessUnitsPerS > 0
      ? (car.speed * car.speed -
          window.targetSpeedUnitsPerS * window.targetSpeedUnitsPerS) /
        (2 * window.constrainedDistanceM)
      : 0;
  const sufficiencySpeedBrake = clampRatio(
    requiredDecelerationUnitsPerS2 /
      Math.abs(NEATRACING_PHYSICS_BRAKING_UNITS_PER_S2),
  );
  // THE BRAKE IS THE STRICTER OF THE TWO: the AI brakes for the excess it
  // can see (the retained law) AND for the deceleration it needs to reach
  // the target speed in time (the sufficiency law).
  const speedBrake = Math.max(proportionalSpeedBrake, sufficiencySpeedBrake);
  const errorBrake =
    absoluteTrackingErrorRad > NEATRACING_AUTOPILOT_FULL_BRAKE_ERROR_RAD
      ? NEATRACING_AUTOPILOT_BRAKE
      : 0;
  // THE NEVER-STOP FLOOR (Step 14Y, AC-098(c)): at or below the physics'
  // own off-road cruise floor the AI never brakes, so the car can never be
  // left below the floor it needs to keep moving - and the off-road floor
  // can never be lost to the AI's own brake.
  const keepsMoving =
    car.speed <= NEATRACING_AUTOPILOT_MINIMUM_DRIVING_SPEED_UNITS_PER_S;
  // THE CORNER GATE: the road AHEAD turns, so the AI is in (or entering)
  // a corner - a place the AI may never coast, because a stopped AI in a
  // corner never restarts on its own.
  const inCorner =
    Math.abs(window.tightestCurvaturePerM) >
    NEATRACING_RACING_LINE_STRAIGHT_CURVATURE_PER_M;
  const brake = keepsMoving ? 0 : Math.max(speedBrake, errorBrake);
  const slowsForTheRoad =
    speedExcessUnitsPerS > 0 ||
    absoluteTrackingErrorRad > NEATRACING_AUTOPILOT_FULL_THROTTLE_ERROR_RAD;
  // THE THROTTLE: the RETAINED coast law - the AI coasts while a heading
  // error it has not shaken off yet is still on the table, and it drives
  // otherwise - with the CORNER gate in front of it.
  //
  // THE CORNER GATE (Step 14Y, AC-098(c)): a car in (or entering) a
  // corner is never coasted, so the AI can never buy its way out of a
  // corner by STOPPING in it.
  //
  // THE NEVER-STOP FLOOR GATES THE BRAKE, NOT THE THROTTLE (the fix
  // packet `fix-packet-P1-turnfeel-ai-iteration-1`, requested change 1):
  // the floor is the BRAKE gate, and the BRAKE was the strand - on the
  // served track the brake measured the car decaying 21.21 -> 18.10 ->
  // 7.42 -> 0.30 -> 0.00 m/s and stopping for good - while the physics'
  // own off-road floor reads the BRAKE and the tick-start speed and
  // NEVER the throttle (it floors the coasting grass car at the named
  // off-road speed floor - the USER-named 20 km/h sand cap re-derived by
  // fix-packet-P1-sand-red-iteration-3 - through `max(nextSpeed, floor)`),
  // so a coast can never lose the floor the way a brake can.
  //
  // WHY THE FLOOR-GATED THROTTLE CAME OUT (MEASURED, not a formula): the
  // gated throttle bought nothing on the grass and cost the coast law on
  // the road - the legacy lane (a car at 10 m/s with a 0.3927 rad
  // heading error on a straight) read throttle 1 where the legacy law
  // reads 0, because the car sits below the floor and the gate drove
  // the throttle through the heading-correction coast.
  const throttle =
    inCorner || !slowsForTheRoad ? NEATRACING_AUTOPILOT_THROTTLE : 0;
  const speedPercent = resolveSpeedPercent(car.speed);
  return {
    throttle,
    brake,
    steer,
    turnRatePercent: resolveTurnRatePercent(turnDemand, speedPercent),
  };
}
