/**
 * The NAMED track edge seam for the neatRacing example (plan slice
 * P1-outer-edge-geometry, Step 14U, AC-095).
 *
 * THE SEAM: the track edge stops being a SYMMETRIC half-width and becomes a
 * NAMED, CURVATURE-DERIVED PAIR - an INNER half-width and an OUTER
 * half-width, both measured from the road centreline. The road is WIDER on
 * the OUTER side of a corner (the drift room the 2026-10-30 user directive
 * asks for) and UNCHANGED on the INNER side, so the road never narrows and
 * the road centreline never moves.
 *
 * WHY IT EXISTS: the edge is read by SEVERAL readers over TWO different
 * models - the chord-capsule distance (the road rasterizer, the tri-colour
 * band paint and the containment bound) and the spline-normal distance (the
 * prop placement and the racing line) - and two readers that derive the
 * edge themselves can disagree about where the edge IS. This module is the
 * ONE seam every reader resolves the edge through, so the rasterizer, the
 * band paint and the containment bound can NEVER disagree about how WIDE the
 * road is: the WIDTH PROFILE (the inner/outer half-width pair per chord and
 * per spline sample), the SIDE ASSIGNMENT (loop-global, never flipping at a
 * straight), the CHORD-DISTANCE MEASURE and the NEAREST-CHORD RULE are all
 * published here. The readers MEASURE the width through the seam, and each
 * reader keeps its OWN ratified band policy - the rasterizer and the band
 * paint keep their min-past-edge UNION over all chords, the containment
 * bound keeps its arg-min nearest-chord rule - so this seam does NOT change
 * which chord a reader binds a point to. The corner measure is derived ONCE
 * here and PUBLISHED - as the ONE corner-measure function of the loop
 * position, sampled at each publication granularity - so no reader ever
 * derives its own.
 *
 * THE TWO FRAMES: the geometry lives in TWO frames and this seam names
 * BOTH of them. The GENERATOR PLANE is the plane of the segment endpoints
 * (the generator's own x/y); the WORLD frame MIRRORS that plane as
 * `worldZ = -generatorY`. ALL THREE READERS MEASURE IN THE GENERATOR PLANE
 * - the road rasterizer and the tri-colour band paint negate their lattice
 * z BEFORE they measure, and the containment bound measures the car state
 * directly - so what differs between them is only the frame their INPUT
 * arrives in: the rasterizer and the band paint are handed WORLD
 * coordinates, the containment bound is handed PHYSICS coordinates (which
 * ARE the generator plane). The side resolution is therefore published in
 * the GENERATOR PLANE ({@link resolveTrackEdgeSideAt}) and the WORLD-frame
 * entry point ({@link resolveTrackEdgeSideAtWorldZ}) mirrors the coordinate
 * for the rasterizer and the band paint, so a world-frame reader can never
 * invert the side of the road; the chord distance follows the SAME
 * convention through {@link distanceToTrackChordMAtWorldZ}, so a world-frame
 * reader can never measure the wrong chord either.
 *
 * THE ASYMMETRY: the OUTER half-width grows with the local corner measure by
 * the MEASURED drift space ({@link NEATRACING_TRACK_OUTER_DRIFT_SPACE_M});
 * the INNER half-width is the EXISTING half-width, UNCHANGED. The widening
 * is applied on the OUTER side ONLY - the inner side of a corner never
 * widens and never narrows.
 *
 * THE CONTINUITY GUARANTEE: the outer/inner side assignment is a property
 * of the WHOLE loop ({@link TrackEdgeProfile.outerSideSign}), NOT of the
 * local turn, so it NEVER flips at a straight (where the local turn is zero
 * and its sign would be meaningless). THAT is the guarantee that holds
 * EXACTLY, and it is the one the raster, the paint and the bound need:
 * because the side never flips, the published width profile has NO
 * discontinuity AT A SIDE FLIP, so no two readers can disagree about which
 * side of the road a point sits on. The published pair is CONSISTENT at
 * every publication: the outer half-width is never narrower than the inner
 * one, the widening is never negative and never exceeds the measured drift
 * space.
 *
 * THE ONE CORNER-MEASURE FUNCTION: the corner measure is ONE function of
 * the loop position - the linear interpolation of the published vertex
 * measures ({@link TrackEdgeProfile.vertexCornerMeasuresDeg}) along the
 * chord they join - and the seam publishes that function at TWO
 * granularities, both of them SAMPLES of it, so the two granularities can
 * never contradict each other: the per-chord publication is the function's
 * MIDPOINT sample, HELD over the chord (the chord-capsule model carries ONE
 * width per chord), and the per-spline-sample publication is the function's
 * value AT the sample. Where a sample sits at its chord's midpoint, the two
 * publications carry the SAME corner measure and the SAME drift space, to
 * the published double precision. The INNER half-width of each publication
 * is its OWN existing half-width (the chord's own width, the sample's own
 * interpolated width): the seam never moves an inner edge, it only adds
 * the OUTER drift room.
 *
 * THE MEASURED DRIFT SPACE: the numeric value was MEASURED by the RED slice
 * `P1-outer-edge-red` against the production readers - never decided by an
 * author - and it is recorded on {@link NEATRACING_TRACK_OUTER_DRIFT_SPACE_M}.
 *
 * Determinism: every export is a pure function of the track geometry; the
 * same segments and the same samples always resolve to byte-identical
 * results on the same runtime (no runtime-random state, no mutation of the
 * inputs, no side effects).
 */

import type {
  SplineSample,
  TrackSegment,
} from '../../racing_curriculum/track/track.generator.types';

/**
 * The MEASURED outer drift space of the track edge, in METERS: the extra
 * half-width the road gains on the OUTER side of a corner. It is exactly the
 * width of the named curb band beside the road.
 *
 * MEASURED, NEVER AUTHORED: the RED slice `P1-outer-edge-red` measured the
 * minimum admissible outer extension against the PRODUCTION readers (the
 * surface classification, the band paint and the containment bound) and
 * recorded D >= 1.0 m EXACTLY. The derivation: RED-1's outer probe sits AT
 * the retired symmetric half-width (EXACTLY 4.0 m from the corner chord)
 * and MUST classify as ROAD; the production rule is "road iff distance <=
 * half-width - curb width", so the OUTER half-width at that corner must be
 * >= 4.0 m + 1.0 m = 5.0 m, hence the outer extension must be >= 1.0 m.
 * The implemented value is that measured bound.
 *
 * THE SIDE IT APPLIES TO: the OUTER side of a corner ONLY. The inner side
 * never widens (and never narrows), and the road centreline never moves.
 * The value equals the named curb width, so the drift room the outer side
 * gains is exactly one curb band.
 *
 * THE CURB-WIDTH COUPLING: the extension is EXACTLY one named curb band -
 * the drift space EQUALS the named curb width. That curb-width constant is
 * owned by the READERS (the rasterizer's private curb band and the
 * physics/containment bound's `NEATRACING_TRACK_CURB_WIDTH_M`) and NOT by
 * this module: this module owns the WIDENING, the readers own the BAND. If
 * a reader's curb width ever changes, the drift space must be RE-PINNED in
 * the SAME slice that changes the curb width, so the two can never drift
 * apart silently.
 *
 * @example
 * ```ts
 * expect(NEATRACING_TRACK_OUTER_DRIFT_SPACE_M).toBe(1);
 * ```
 */
export const NEATRACING_TRACK_OUTER_DRIFT_SPACE_M = 1.0;

/**
 * The reference corner measure, in DEGREES, at which the FULL measured drift
 * space applies: a quarter turn (90 degrees) is the reference corner the
 * measurement was taken at, a straight (0 degrees) gets no drift space at
 * all, and the widening never exceeds the measured drift space.
 */
const TRACK_EDGE_REFERENCE_CORNER_DEG = 90;

/**
 * The tolerance, in DEGREES, that absorbs a SHORTFALL of the reference
 * corner measure of AT MOST one nanodegree.
 *
 * WHY IT EXISTS: the reference corner (90 degrees) publishes the FULL
 * measured drift space, so a corner measure that lands a SHORTFALL BELOW 90
 * - a shortfall AT OR BELOW THIS TOLERANCE (one nanodegree, INCLUSIVE at
 * exactly one nanodegree) - must never shrink the published drift room
 * below the MEASURED bound. The tolerance is a FLOOR, never a widening.
 *
 * WHAT IT DOES NOT GUARD: ONLY the `>= 90 degrees` SATURATION is
 * tolerance-guarded. The INTERIOR ramp is exact only UP TO the double
 * precision of the PUBLISHED corner measure, which inherits the
 * double-precision error of the measured turn: a STRAIGHT publishes 0 and a
 * 45-degree corner publishes 0.5 to that precision, not necessarily the
 * EXACTLY representable 0.5.
 */
const TRACK_EDGE_CORNER_MEASURE_TOLERANCE_DEG = 1e-9;

/**
 * The factor that converts a turn measured in radians into a corner measure
 * in degrees (the published corner measure unit).
 */
const TRACK_EDGE_DEGREES_PER_RADIAN = 180 / Math.PI;

/** A full turn in radians: the wrap period of the signed turn normalisation. */
const TRACK_EDGE_FULL_TURN_RADIANS = Math.PI * 2;

/**
 * Which side of the road centreline a point or a half-width belongs to:
 * `'outer'` is the OUTER side of the loop (the drift side, where the drift
 * room is added) and `'inner'` is the other side.
 *
 * THE SIDE IS LOOP-GLOBAL, NEVER PER CORNER: the side is resolved from
 * {@link TrackEdgeProfile.outerSideSign} - the winding of the WHOLE loop -
 * and NEVER from the local turn. `'outer'` is the side the loop's winding
 * turns toward (the drift side) and `'inner'` is the side it turns away
 * from. A corner whose OWN local turn opposes the loop winding therefore
 * has NO separately published local outer side: it carries the loop's outer
 * side, exactly like every other place on the loop. The assignment NEVER
 * flips at a straight.
 */
export type TrackEdgeSide = 'inner' | 'outer';

/**
 * The winding of the closed loop the edge was resolved from, read from the
 * sign of the shoelace signed area: `'counter-clockwise'` for a positive
 * signed area, `'clockwise'` for a negative one.
 *
 * THE DEGENERATE TIE-BREAK: an EMPTY or ZERO-AREA loop has no winding at
 * all, so its orientation is undetermined and the seam reports it as
 * `'counter-clockwise'` (the `signedArea >= 0` branch). The seam never
 * throws for a degenerate loop.
 */
export type TrackEdgeLoopOrientation = 'counter-clockwise' | 'clockwise';

/**
 * The NAMED edge pair of one place on the track: the INNER half-width (the
 * existing half-width, UNCHANGED - the road never narrows) and the OUTER
 * half-width (the inner half-width plus the corner-measured drift space).
 *
 * Both half-widths are measured from the road CENTRELINE, so widening the
 * outer side never moves the centreline.
 */
export interface TrackEdgeHalfWidths {
  /** The inner half-width of the road at this place, in meters. */
  readonly innerHalfWidthM: number;
  /** The outer half-width of the road at this place, in meters. */
  readonly outerHalfWidthM: number;
}

/**
 * The published edge of ONE raw chord segment: the named half-width pair,
 * the corner measure the widening was derived from, and the widening itself.
 */
export interface TrackChordEdge extends TrackEdgeHalfWidths {
  /** Index of the owning raw chord segment in the track segment list. */
  readonly segmentIndex: number;
  /**
   * The published local corner measure, in degrees: the MEAN of the turn
   * magnitudes at the two vertices the chord joins - the MIDPOINT sample of
   * the ONE corner-measure function
   * ({@link TrackEdgeProfile.vertexCornerMeasuresDeg}), HELD over the chord
   * (the chord-capsule model carries ONE width per chord). It is published
   * AS MEASURED - the raw mean, with NO tolerance applied to it - so a
   * consumer that pins a literal against it pins the MEASURED value. The
   * one-nanodegree tolerance lives in the DRIFT FACTOR (see
   * {@link TrackChordEdge.outerWideningM}), never in this number.
   */
  readonly cornerMeasureDeg: number;
  /**
   * The drift space added on the OUTER side, in meters: the measured drift
   * space scaled by the corner measure. The 90-degree reference corner
   * publishes the FULL measured drift space - a shortfall of the corner
   * measure AT OR BELOW THE TOLERANCE (one nanodegree, inclusive at exactly
   * one nanodegree) is absorbed by it, so the published drift room can
   * never fall below the measured bound. The INTERIOR ramp is exact only
   * UP TO the double precision of the published corner measure, so a
   * STRAIGHT publishes 0 and a 45-degree corner publishes 0.5 to that
   * precision.
   */
  readonly outerWideningM: number;
}

/**
 * The resolved edge profile of a whole closed loop: the raw segments, the
 * published per-chord edges, the published corner-measure NODES, and the
 * loop-global side assignment.
 *
 * The side assignment lives on the PROFILE (not on the chord) because the
 * outer side of a corner is a property of the WHOLE loop: a chord-local
 * turn sign would flip at a straight, where the local turn is zero and its
 * sign is meaningless.
 */
export interface TrackEdgeProfile {
  /**
   * A SNAPSHOT of the raw closed-loop segments the profile was resolved from:
   * every segment object is COPIED, never aliased, so a caller that mutates
   * its own segment objects can never desynchronise the published chords
   * from the segments they were resolved from. The elements are published as
   * DEEPLY READONLY types, so mutating a published segment is a COMPILE-TIME
   * error rather than a silent desync from the published chords.
   */
  readonly segments: readonly Readonly<TrackSegment>[];
  /** The published edge of every chord, in segment order. */
  readonly chords: readonly TrackChordEdge[];
  /**
   * The corner measure AT every vertex of the loop, in degrees, in segment
   * order: index `i` is the START vertex of chord `i`, and the value is the
   * MAGNITUDE of the signed turn measured at that vertex.
   *
   * THE NODES OF THE ONE CORNER-MEASURE FUNCTION: the seam's corner measure
   * is ONE function of the loop position - the linear interpolation of
   * THESE nodes along each chord they join - and these nodes are its
   * published anchor points. The per-chord publication is that function's
   * MIDPOINT sample (see {@link TrackChordEdge.cornerMeasureDeg}) and the
   * per-spline-sample publication is that function's value AT the sample
   * (see {@link TrackSplineEdge.cornerMeasureDeg}), so the two
   * granularities are samples of the SAME function and can never contradict
   * each other.
   */
  readonly vertexCornerMeasuresDeg: readonly number[];
  /**
   * The winding of the closed loop, read in the GENERATOR PLANE (the plane
   * of the segment endpoints).
   */
  readonly loopOrientation: TrackEdgeLoopOrientation;
  /**
   * The sign of the chord cross product that points to the OUTER side: `-1`
   * for a counter-clockwise loop, `+1` for a clockwise one. It is read in
   * the GENERATOR PLANE (the same plane as the segment endpoints), so a
   * world-frame reader must mirror its coordinate first - see
   * {@link resolveTrackEdgeSideAtWorldZ}.
   */
  readonly outerSideSign: 1 | -1;
}

/**
 * The chord a point binds to, as published by the NEAREST-CHORD RULE: the
 * index of the owning chord and the distance from the point to that chord.
 *
 * The rule is measured in the GENERATOR PLANE with the canonical chord
 * distance ({@link distanceToTrackChordM}).
 */
export interface TrackNearestChord {
  /** Index of the nearest chord in the profile's segment order. */
  readonly segmentIndex: number;
  /** Distance from the point to the nearest chord, in meters. */
  readonly distanceM: number;
}

/**
 * The published edge of ONE spline sample: the named half-width pair at the
 * sample's own interpolated width, plus the LOCAL corner measure at the
 * sample and the drift space that measure earns.
 *
 * THE LOCAL MEASURE: the corner measure is the ONE corner-measure function
 * of the loop position - the linear interpolation of the published vertex
 * measures ({@link TrackEdgeProfile.vertexCornerMeasuresDeg}) along the
 * owning chord - sampled AT this sample, so the sampled ribbon carries the
 * SAME function the chords do. The owning chord still VALIDATES the sample:
 * a sample that names a segment the profile does not own throws.
 */
export interface TrackSplineEdge extends TrackEdgeHalfWidths {
  /** Stable global sample index around the closed loop. */
  readonly globalIndex: number;
  /** Index of the owning raw chord segment. */
  readonly segmentIndex: number;
  /**
   * The published local corner measure, in degrees: the ONE
   * corner-measure function sampled AT this sample.
   */
  readonly cornerMeasureDeg: number;
  /** The drift space added on the OUTER side, in meters. */
  readonly outerWideningM: number;
}

/**
 * Resolves the NAMED edge profile of a closed track loop: the published
 * per-chord edge (the inner/outer half-width pair, the corner measure and
 * the drift space), the published corner-measure NODES and the loop-global
 * outer-side assignment.
 *
 * THE ONE CORNER-MEASURE FUNCTION: the corner measure is ONE function of
 * the loop position - the linear interpolation of the published vertex
 * measures ({@link TrackEdgeProfile.vertexCornerMeasuresDeg}) along each
 * chord they join - and the profile publishes its NODES. The per-chord
 * publication is that function's MIDPOINT sample, HELD over the chord (the
 * chord-capsule model carries ONE width per chord), so the widening is
 * CENTRED on the corner: the STEP in the widening lands SYMMETRICALLY ON
 * the corner vertex instead of entirely on one side of it. The published
 * per-chord half-widths are PIECEWISE CONSTANT and STEP between adjacent
 * chords - the CONTINUITY guarantee that holds EXACTLY is the loop-global
 * SIDE ASSIGNMENT (it never flips at a straight), together with a
 * non-negative widening that never exceeds the measured drift space. The
 * outer half-width grows with the corner measure by the MEASURED drift
 * space ({@link NEATRACING_TRACK_OUTER_DRIFT_SPACE_M}); the inner
 * half-width is the existing half-width, UNCHANGED.
 *
 * THE DEGENERATE TIE-BREAK: an EMPTY or ZERO-AREA loop has no winding, so
 * its orientation is undetermined and the profile reports it as
 * `'counter-clockwise'` (the `signedArea >= 0` branch) with the outer-side
 * sign `-1`; the seam never throws for it.
 *
 * @param segments - Ordered closed-loop chord segments (read only, never
 *   mutated; the profile SNAPSHOTS them, so the caller's segment objects
 *   are copied rather than aliased - see {@link TrackEdgeProfile.segments}).
 * @returns The resolved edge profile: the segments, the per-chord edges, the
 *   vertex corner measures, the loop orientation and the outer-side sign.
 *
 * @example
 * ```ts
 * const profile = resolveTrackEdgeProfile(spec.segments);
 * profile.outerSideSign; // -1 for a counter-clockwise loop
 * ```
 */
export function resolveTrackEdgeProfile(
  segments: readonly TrackSegment[],
): TrackEdgeProfile {
  const turnRadians = measureTurnRadians(segments);
  const vertexCornerMeasuresDeg = turnRadians.map(
    (turnRadiansAtVertex) =>
      Math.abs(turnRadiansAtVertex) * TRACK_EDGE_DEGREES_PER_RADIAN,
  );
  const chords = segments.map((segment, segmentIndex) =>
    buildChordEdge(segment, segmentIndex, turnRadians),
  );
  const signedArea = measureSignedArea(segments);
  const loopOrientation = signedArea >= 0 ? 'counter-clockwise' : 'clockwise';

  return {
    segments: segments.map((segment) => ({ ...segment })),
    chords,
    vertexCornerMeasuresDeg,
    loopOrientation,
    outerSideSign: loopOrientation === 'counter-clockwise' ? -1 : 1,
  };
}

/**
 * Resolves the published edge of every spline sample from the SAME edge
 * profile the chords use, so the sampled ribbon and the raw chords can
 * never disagree about the drift room.
 *
 * The base half-width of a sample is the sample's OWN interpolated width;
 * the corner measure is the ONE corner-measure function sampled AT the
 * sample (the linear interpolation of the profile's published vertex
 * measures along the owning chord), and the drift space is the drift that
 * measure earns. The owning chord still VALIDATES the sample: a sample
 * that names a segment the profile does not own throws.
 *
 * @param samples - Ordered spline samples of the closed loop (read only,
 *   never mutated).
 * @param profile - The edge profile resolved from the same track geometry.
 * @returns One published edge per sample, in sample order.
 * @throws Error when a sample names a segment the profile does not own.
 *
 * @example
 * ```ts
 * const edges = resolveTrackSplineEdges(spec.splineSamples, profile);
 * edges[0].outerHalfWidthM; // the outer half-width at the first sample
 * ```
 */
export function resolveTrackSplineEdges(
  samples: readonly SplineSample[],
  profile: TrackEdgeProfile,
): readonly TrackSplineEdge[] {
  return samples.map((sample) => buildSplineEdge(sample, profile));
}

/**
 * Resolves the half-width of one named edge for one named side.
 *
 * @param edge - The named edge (the inner/outer half-width pair).
 * @param side - The side to resolve: `'inner'` or `'outer'`.
 * @returns The half-width of the requested side, in meters.
 *
 * @example
 * ```ts
 * resolveTrackEdgeHalfWidthM(profile.chords[3], 'outer'); // 5 at a 90 degree corner
 * ```
 */
export function resolveTrackEdgeHalfWidthM(
  edge: TrackEdgeHalfWidths,
  side: TrackEdgeSide,
): number {
  switch (true) {
    case side === 'outer':
      return edge.outerHalfWidthM;
    default:
      return edge.innerHalfWidthM;
  }
}

/**
 * Resolves which side of the road centreline a ground-plane point sits on,
 * relative to the chord it is measured against.
 *
 * THE COORDINATE FRAME: the point is read in the GENERATOR PLANE - the SAME
 * plane as the segment endpoints (`startX`/`startY`). The WORLD frame
 * MIRRORS that plane as `worldZ = -generatorY`, so a world-frame reader must
 * NOT pass its z straight in here: an un-mirrored z INVERTS the side of the
 * road on every chord with a non-zero x component. The world-frame entry
 * point is {@link resolveTrackEdgeSideAtWorldZ}.
 *
 * The side is resolved from the chord cross product and the PROFILE's
 * loop-global outer-side sign, so it never flips at a straight and never
 * depends on the local turn. A point that sits EXACTLY on the chord (a zero
 * cross product) resolves to `'inner'`: the centreline is road under both
 * half-widths, and the inner half-width is the conservative one.
 *
 * @param profile - The edge profile resolved from the track geometry.
 * @param segmentIndex - Index of the chord the point is measured against.
 * @param pointX - Generator-plane x of the point, in meters.
 * @param pointY - Generator-plane y of the point, in meters (NOT a world z).
 * @returns The side of the road centreline the point sits on: `'outer'` or
 *   `'inner'`.
 * @throws Error when the segment index is outside the profile.
 *
 * @example
 * ```ts
 * resolveTrackEdgeSideAt(profile, 3, 94, -26); // 'outer' on a counter-clockwise loop
 * ```
 */
export function resolveTrackEdgeSideAt(
  profile: TrackEdgeProfile,
  segmentIndex: number,
  pointX: number,
  pointY: number,
): TrackEdgeSide {
  const segment = profile.segments[segmentIndex];
  if (segment === undefined) {
    throw new Error(
      `resolveTrackEdgeSideAt: segment index ${segmentIndex} is outside the edge profile (${profile.segments.length} segments)`,
    );
  }

  const offsetX = pointX - segment.startX;
  const offsetY = pointY - segment.startY;
  const cross =
    (segment.endX - segment.startX) * offsetY -
    (segment.endY - segment.startY) * offsetX;

  return Math.sign(cross) === profile.outerSideSign ? 'outer' : 'inner';
}

/**
 * Resolves which side of the road centreline a WORLD-frame point sits on,
 * relative to the chord it is measured against.
 *
 * THE COORDINATE FRAME: the point is read in the WORLD frame, the frame the
 * road rasterizer and the tri-colour band paint live in. The world frame
 * MIRRORS the generator plane (the plane of the segment endpoints) as
 * `worldZ = -generatorY`, so this entry point mirrors the coordinate back
 * (`-worldZ`) before it resolves the side.
 *
 * WHY IT EXISTS: a world-frame reader that passed its z straight into the
 * generator-plane resolver ({@link resolveTrackEdgeSideAt}) would get an
 * INVERTED side on every chord with a non-zero x component - the same
 * physical point would read `'outer'` in one frame and `'inner'` in the
 * other. Mirroring HERE, at the seam, is what keeps the two frames from
 * disagreeing.
 *
 * @param profile - The edge profile resolved from the track geometry.
 * @param segmentIndex - Index of the chord the point is measured against.
 * @param worldX - World-frame x of the point, in meters.
 * @param worldZ - World-frame z of the point, in meters (the MIRROR of the
 *   generator-plane y).
 * @returns The side of the road centreline the point sits on: `'outer'` or
 *   `'inner'`.
 * @throws Error when the segment index is outside the profile - the same
 *   out-of-range error the generator-plane resolver throws.
 *
 * @example
 * ```ts
 * resolveTrackEdgeSideAtWorldZ(profile, 1, 0, 34); // 'outer' on a counter-clockwise loop
 * ```
 */
export function resolveTrackEdgeSideAtWorldZ(
  profile: TrackEdgeProfile,
  segmentIndex: number,
  worldX: number,
  worldZ: number,
): TrackEdgeSide {
  return resolveTrackEdgeSideAt(profile, segmentIndex, worldX, -worldZ);
}

/**
 * Measures the distance from a point to a chord in the GENERATOR PLANE: the
 * Euclidean distance to the CLAMPED nearest point of the chord (the same
 * clamped-projection measure the production readers use).
 *
 * A degenerate (zero-length) chord has no projection direction, so its
 * whole geometry is its start point and the distance is measured to that
 * point.
 *
 * THE CANONICAL CHORD DISTANCE: this is the ONE chord-distance measure the
 * readers should use. The rasterizer, the band paint and the containment
 * bound keep their own private copies for now; those copies are DELETED in
 * `P1-outer-edge-paint` when the readers join this seam, so every reader
 * measures the chord the same way.
 *
 * @param segment - The chord segment the distance is measured to (read
 *   only, never mutated; the profile's DEEPLY READONLY segment snapshot is
 *   accepted).
 * @param pointX - Generator-plane x of the point, in meters.
 * @param pointY - Generator-plane y of the point, in meters.
 * @returns The distance from the point to the chord, in meters.
 *
 * @example
 * ```ts
 * distanceToTrackChordM(profile.segments[3], 94, -26); // 4 at the fixture corner
 * ```
 */
export function distanceToTrackChordM(
  segment: Readonly<TrackSegment>,
  pointX: number,
  pointY: number,
): number {
  const chordX = segment.endX - segment.startX;
  const chordY = segment.endY - segment.startY;
  const chordLengthSquared = chordX * chordX + chordY * chordY;

  if (chordLengthSquared <= 0) {
    return Math.hypot(pointX - segment.startX, pointY - segment.startY);
  }

  const projection =
    ((pointX - segment.startX) * chordX + (pointY - segment.startY) * chordY) /
    chordLengthSquared;
  const clampedProjection = Math.min(1, Math.max(0, projection));
  const nearestX = segment.startX + clampedProjection * chordX;
  const nearestY = segment.startY + clampedProjection * chordY;

  return Math.hypot(pointX - nearestX, pointY - nearestY);
}

/**
 * Measures the distance from a WORLD-frame point to a chord, in meters: the
 * same clamped-projection measure as {@link distanceToTrackChordM}, with the
 * world z mirrored back into the generator plane.
 *
 * WHY IT EXISTS: the world frame MIRRORS the generator plane (the plane of
 * the segment endpoints) as `worldZ = -generatorY`, so a world-frame reader
 * that measures against the generator-plane chord has to mirror its
 * coordinate. Mirroring the POINT is enough (the mirror is an isometry, so
 * the distance is unchanged), but a reader that forgets it would measure the
 * wrong chord and could read the wrong side of the road. This entry point
 * performs the mirror for the two world-frame readers (the road rasterizer
 * and the tri-colour band paint), so the frame convention lives in ONE
 * place.
 *
 * @param segment - The chord segment the distance is measured to (read
 *   only, never mutated; the profile's DEEPLY READONLY segment snapshot is
 *   accepted).
 * @param worldX - World-frame x of the point, in meters.
 * @param worldZ - World-frame z of the point, in meters (the MIRROR of the
 *   generator-plane y).
 * @returns The distance from the point to the chord, in meters.
 *
 * @example
 * ```ts
 * distanceToTrackChordMAtWorldZ(profile.segments[3], 94, 26); // 4 at the fixture corner
 * ```
 */
export function distanceToTrackChordMAtWorldZ(
  segment: Readonly<TrackSegment>,
  worldX: number,
  worldZ: number,
): number {
  return distanceToTrackChordM(segment, worldX, -worldZ);
}

/**
 * Resolves the chord a point binds to by the NEAREST-CHORD RULE: the chord
 * with the smallest {@link distanceToTrackChordM}, measured in the GENERATOR
 * PLANE.
 *
 * THE TIE-BREAK: two chords at the SAME distance are a genuine tie, and the
 * FIRST chord in segment order wins it (the comparison is a strict `<`, so
 * a later chord never displaces an earlier one). That is the same policy the
 * containment bound already uses, so the bound and this rule can never
 * disagree about which chord a point belongs to.
 *
 * WHAT THIS RULE IS NOT: the rasterizer and the band paint keep their own
 * ratified min-past-edge UNION band policy (a point is inside a band when
 * ANY chord's band holds it). That policy is NOT the nearest-chord rule, and
 * this seam does NOT change it: the two readers may still bind a point to
 * different chords, because the track widths vary per chord. What the seam
 * guarantees is that every reader measures the WIDTH through the same
 * published profile.
 *
 * @param profile - The edge profile resolved from the track geometry.
 * @param pointX - Generator-plane x of the point, in meters.
 * @param pointY - Generator-plane y of the point, in meters.
 * @returns The nearest chord: its index in the profile's segment order and
 *   the distance from the point to that chord, in meters.
 * @throws Error when the profile owns no segments (there is no chord to
 *   bind the point to).
 *
 * @example
 * ```ts
 * resolveNearestTrackChord(profile, 94, -26); // { segmentIndex: 3, distanceM: 4 }
 * ```
 */
export function resolveNearestTrackChord(
  profile: TrackEdgeProfile,
  pointX: number,
  pointY: number,
): TrackNearestChord {
  let nearestSegmentIndex = -1;
  let nearestDistanceM = Number.POSITIVE_INFINITY;

  for (
    let segmentIndex = 0;
    segmentIndex < profile.segments.length;
    segmentIndex += 1
  ) {
    const segment = profile.segments[segmentIndex];
    if (segment === undefined) {
      continue;
    }

    const distanceM = distanceToTrackChordM(segment, pointX, pointY);
    if (distanceM < nearestDistanceM) {
      nearestDistanceM = distanceM;
      nearestSegmentIndex = segmentIndex;
    }
  }

  if (nearestSegmentIndex < 0) {
    throw new Error(
      'resolveNearestTrackChord: the edge profile owns no segments',
    );
  }

  return { segmentIndex: nearestSegmentIndex, distanceM: nearestDistanceM };
}

/**
 * Measures the signed turn at the START vertex of every chord: the signed
 * difference between the incoming chord direction and the outgoing chord
 * direction, normalised into `(-PI, PI]`. A positive turn is a LEFT turn.
 *
 * The chord directions are read from the raw chord endpoints (the segment
 * start and end points), never from a smoothed spline, so the measure is
 * the geometry the readers actually drive on.
 *
 * THE INCOMING DIRECTION is the PREVIOUS CHORD'S OWN HEADING (its start
 * point to its end point), NOT the connector from the previous chord's start
 * point to this chord's start point. For a closed loop of CONTIGUOUS chords
 * (the closed-loop invariant `segments.at(-1).end === segments[0].start`,
 * and every interior chord's end equal to the next chord's start) the two
 * are identical, and the previous chord's own heading stays well-defined
 * even when the loop is NOT contiguous.
 *
 * @param segments - Ordered closed-loop chord segments (read only).
 * @returns One signed turn in radians per chord, in segment order.
 * @throws Error when the segment list holds an UNDEFINED (HOLED) element at
 *   the measured index - a SPARSE array: the turn array is indexed by
 *   SEGMENT INDEX, so a silent skip would shorten the turn array and
 *   MISALIGN every published corner measure after the hole. The throw is
 *   loud on purpose. A merely NON-CONTIGUOUS loop is NOT rejected: the
 *   previous chord's OWN heading stays well-defined, so the turn is
 *   measured against it. An EMPTY segment list resolves without throwing:
 *   the loop body never runs.
 */
function measureTurnRadians(
  segments: readonly TrackSegment[],
): readonly number[] {
  const turnRadians: number[] = [];

  for (
    let segmentIndex = 0;
    segmentIndex < segments.length;
    segmentIndex += 1
  ) {
    const previous =
      segments[(segmentIndex - 1 + segments.length) % segments.length];
    const current = segments[segmentIndex];
    if (previous === undefined || current === undefined) {
      throw new Error(
        `measureTurnRadians: the segment list has a hole (a sparse array) at index ${segmentIndex} - the turn array is indexed by segment index, so a hole would misalign the published corner measures`,
      );
    }

    const incoming = Math.atan2(
      previous.endY - previous.startY,
      previous.endX - previous.startX,
    );
    const outgoing = Math.atan2(
      current.endY - current.startY,
      current.endX - current.startX,
    );
    turnRadians.push(normalizeTurnRadians(outgoing - incoming));
  }

  return turnRadians;
}

/**
 * Normalises a signed turn into `(-PI, PI]`: the wrap keeps the turn of a
 * corner whose incoming and outgoing directions straddle the atan2 branch
 * cut on the correct side.
 *
 * @param radians - The raw signed turn difference, in radians.
 * @returns The normalised signed turn, in radians.
 */
function normalizeTurnRadians(radians: number): number {
  let normalized = radians;

  while (normalized > Math.PI) {
    normalized -= TRACK_EDGE_FULL_TURN_RADIANS;
  }
  while (normalized <= -Math.PI) {
    normalized += TRACK_EDGE_FULL_TURN_RADIANS;
  }

  return normalized;
}

/**
 * Resolves the OUTER drift space a corner measure earns, in meters: the
 * MEASURED drift space scaled by the corner measure.
 *
 * THE SATURATION: the reference corner (90 degrees) and every measure above
 * it earn the FULL measured drift space, so the published drift room can
 * never fall below the MEASURED bound - a shortfall AT OR BELOW the
 * one-nanodegree tolerance
 * ({@link TRACK_EDGE_CORNER_MEASURE_TOLERANCE_DEG}) is absorbed by the
 * saturation instead of shrinking the drift room. ONLY the saturation is
 * tolerance-guarded: the INTERIOR ramp is exact only UP TO the double
 * precision of the published corner measure, which inherits the
 * double-precision error of the measured turn.
 *
 * ONE HELPER FOR BOTH PUBLICATIONS: the per-chord publication and the
 * per-spline-sample publication both derive their drift space from THIS
 * function, so the corner measure is scaled ONCE and the two granularities
 * can never disagree about the drift room.
 *
 * @param cornerMeasureDeg - The corner measure the drift space is derived
 *   from, in degrees.
 * @returns The drift space added on the OUTER side, in meters.
 */
function resolveOuterWideningM(cornerMeasureDeg: number): number {
  const driftFactor =
    cornerMeasureDeg >=
    TRACK_EDGE_REFERENCE_CORNER_DEG - TRACK_EDGE_CORNER_MEASURE_TOLERANCE_DEG
      ? 1
      : cornerMeasureDeg / TRACK_EDGE_REFERENCE_CORNER_DEG;

  return NEATRACING_TRACK_OUTER_DRIFT_SPACE_M * driftFactor;
}

/**
 * Builds the published edge of one chord: the corner measure (the mean of
 * the turn magnitudes at the two vertices the chord joins), the drift space
 * that measure earns, and the named inner/outer half-width pair.
 *
 * THE PUBLICATION IS A MIDPOINT SAMPLE: the corner measure is the MIDPOINT
 * sample of the ONE corner-measure function
 * ({@link TrackEdgeProfile.vertexCornerMeasuresDeg}), HELD over the chord
 * (the chord-capsule model carries ONE width per chord).
 *
 * THE REFERENCE-CORNER TOLERANCE: the drift factor is EXACTLY 1 at the
 * 90-degree reference corner and at every measure above it, so the published
 * drift room can never fall below the MEASURED bound - a shortfall AT OR
 * BELOW the one-nanodegree tolerance
 * ({@link TRACK_EDGE_CORNER_MEASURE_TOLERANCE_DEG}) is absorbed by it
 * instead of shrinking the drift room. ONLY the saturation is
 * tolerance-guarded: the INTERIOR ramp is exact only UP TO the double
 * precision of the published corner measure, so a STRAIGHT (a measure of 0)
 * earns 0 and a 45-degree corner earns 0.5 to that precision.
 *
 * @param segment - The owning raw chord segment.
 * @param segmentIndex - Index of the owning chord in the segment list.
 * @param turnRadians - The signed turn at every chord's start vertex.
 * @returns The published chord edge.
 */
function buildChordEdge(
  segment: TrackSegment,
  segmentIndex: number,
  turnRadians: readonly number[],
): TrackChordEdge {
  const nextTurn = turnRadians[(segmentIndex + 1) % turnRadians.length] ?? 0;
  const turnMagnitude = Math.abs(turnRadians[segmentIndex] ?? 0);
  const cornerMeasureDeg =
    ((turnMagnitude + Math.abs(nextTurn)) / 2) * TRACK_EDGE_DEGREES_PER_RADIAN;
  const outerWideningM = resolveOuterWideningM(cornerMeasureDeg);
  const innerHalfWidthM = segment.width / 2;

  return {
    segmentIndex,
    cornerMeasureDeg,
    outerWideningM,
    innerHalfWidthM,
    outerHalfWidthM: innerHalfWidthM + outerWideningM,
  };
}

/**
 * Resolves the NORMALISED parameter of a point along a chord: the clamped
 * projection of the point onto the chord, in the range `[0, 1]`.
 *
 * A DEGENERATE (zero-length) chord has no projection direction, so it
 * resolves to `0` (the chord's own start vertex).
 *
 * @param segment - The chord the point is projected onto (read only).
 * @param pointX - Generator-plane x of the point, in meters.
 * @param pointY - Generator-plane y of the point, in meters.
 * @returns The parameter of the point along the chord, in `[0, 1]`.
 */
function resolveChordParameter(
  segment: Readonly<TrackSegment>,
  pointX: number,
  pointY: number,
): number {
  const chordX = segment.endX - segment.startX;
  const chordY = segment.endY - segment.startY;
  const chordLengthSquared = chordX * chordX + chordY * chordY;

  if (chordLengthSquared <= 0) {
    return 0;
  }

  const projection =
    ((pointX - segment.startX) * chordX + (pointY - segment.startY) * chordY) /
    chordLengthSquared;

  return Math.min(1, Math.max(0, projection));
}

/**
 * Resolves the corner measure at a parameter along a chord: the linear
 * interpolation of the ONE corner-measure function between the two vertex
 * measures the chord joins.
 *
 * @param vertexCornerMeasuresDeg - The published vertex corner measures, in
 *   segment order.
 * @param segmentIndex - Index of the owning chord.
 * @param parameter - The parameter along the chord, in `[0, 1]`.
 * @returns The corner measure at the parameter, in degrees.
 */
function resolveCornerMeasureDeg(
  vertexCornerMeasuresDeg: readonly number[],
  segmentIndex: number,
  parameter: number,
): number {
  const startMeasureDeg = vertexCornerMeasuresDeg[segmentIndex] ?? 0;
  const endMeasureDeg =
    vertexCornerMeasuresDeg[
      (segmentIndex + 1) % vertexCornerMeasuresDeg.length
    ] ?? 0;

  return startMeasureDeg * (1 - parameter) + endMeasureDeg * parameter;
}

/**
 * Builds the published edge of one spline sample from its own interpolated
 * width and the LOCAL corner measure at the sample.
 *
 * THE LOCAL MEASURE: the sample's parameter along its owning chord is the
 * clamped normalised projection of the sample position onto that chord, so
 * the sample carries the ONE corner-measure function - the linear
 * interpolation of the profile's published vertex measures along the owning
 * chord - sampled AT the sample. The owning chord still VALIDATES the
 * sample: a sample that names a segment the profile does not own throws.
 *
 * @param sample - The spline sample (read only).
 * @param profile - The edge profile resolved from the same track geometry.
 * @returns The published spline edge.
 */
function buildSplineEdge(
  sample: SplineSample,
  profile: TrackEdgeProfile,
): TrackSplineEdge {
  // The owning chord VALIDATES the sample: an out-of-range segment index
  // throws, exactly as it did before the local-measure publication.
  resolveChordEdge(profile, sample.segmentIndex);
  const owningSegment = profile.segments[sample.segmentIndex];
  const parameter = resolveChordParameter(owningSegment, sample.x, sample.y);
  const cornerMeasureDeg = resolveCornerMeasureDeg(
    profile.vertexCornerMeasuresDeg,
    sample.segmentIndex,
    parameter,
  );
  const outerWideningM = resolveOuterWideningM(cornerMeasureDeg);
  const innerHalfWidthM = sample.width / 2;

  return {
    globalIndex: sample.globalIndex,
    segmentIndex: sample.segmentIndex,
    cornerMeasureDeg,
    outerWideningM,
    innerHalfWidthM,
    outerHalfWidthM: innerHalfWidthM + outerWideningM,
  };
}

/**
 * Reads the published chord edge a sample or a probe names, and fails loudly
 * when the profile does not own that segment - a silent fallback would let a
 * reader measure the edge against the WRONG chord.
 *
 * @param profile - The edge profile resolved from the track geometry.
 * @param segmentIndex - Index of the named chord.
 * @returns The published chord edge.
 * @throws Error when the segment index is outside the profile.
 */
function resolveChordEdge(
  profile: TrackEdgeProfile,
  segmentIndex: number,
): TrackChordEdge {
  const chordEdge = profile.chords[segmentIndex];
  if (chordEdge === undefined) {
    throw new Error(
      `resolveTrackSplineEdges: segment index ${segmentIndex} is outside the edge profile (${profile.chords.length} chords)`,
    );
  }

  return chordEdge;
}

/**
 * Measures the shoelace signed area of the closed loop: a positive area
 * means the loop winds counter-clockwise.
 *
 * An EMPTY or ZERO-AREA loop measures exactly `0`, so its winding is
 * UNDETERMINED; the caller resolves that degenerate case to
 * `'counter-clockwise'` (see {@link TrackEdgeLoopOrientation}) rather than
 * throwing.
 *
 * @param segments - Ordered closed-loop chord segments (read only).
 * @returns The signed area of the loop, in square meters (`0` for an empty
 *   or zero-area loop).
 */
function measureSignedArea(segments: readonly TrackSegment[]): number {
  let doubledArea = 0;

  for (const segment of segments) {
    doubledArea +=
      segment.startX * segment.endY - segment.endX * segment.startY;
  }

  return doubledArea / 2;
}
