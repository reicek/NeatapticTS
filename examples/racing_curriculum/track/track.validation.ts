import type { TrackAabb, TrackSpec } from './track.generator.types';

const TRACK_CLOSED_LOOP_TOLERANCE = 0.001;

/**
 * Validates that a `TrackSpec` satisfies all generation invariants.
 *
 * Invariants checked:
 * - `segments.length >= 3` (minimum closed-loop polygon)
 * - All segments have `width > 0`
 * - Closed-loop: last segment end equals first segment start (within tolerance)
 * - No self-intersecting segment pairs
 * - Tier 4 pit corridors do not overlap
 * - Tier 4 pit corridors remain reachable from the driveable ribbon
 *
 * @param spec - The generated track to validate.
 * @returns `true` when all invariants pass.
 * @throws {RangeError} When any invariant is violated.
 *
 * @example
 * ```ts
 * validateTrackSpec(spec); // throws if spec has zero segments
 * ```
 */
export function validateTrackSpec(spec: TrackSpec): true {
  if (spec.segments.length < 3) {
    throw new RangeError('TrackSpec must contain at least 3 segments.');
  }

  if (spec.segments.some((segment) => segment.width <= 0)) {
    throw new RangeError('TrackSpec segments must all have positive width.');
  }

  const firstSegment = spec.segments[0];
  const lastSegment = spec.segments.at(-1);

  if (
    lastSegment === undefined ||
    !areCloseEnough(lastSegment.endX, firstSegment.startX) ||
    !areCloseEnough(lastSegment.endY, firstSegment.startY)
  ) {
    throw new RangeError('TrackSpec must form a closed loop.');
  }

  if (!hasNoSelfIntersection(spec)) {
    throw new RangeError('TrackSpec contains a self-intersection.');
  }

  validatePitCorridorNonOverlap(spec);
  validatePitCorridorReachability(spec);
  return true;
}

/**
 * Returns `true` when no two segments in the spec geometrically intersect.
 *
 * Adjacent segments may share an endpoint; non-adjacent segments must not
 * cross.
 *
 * @param spec - The track geometry to inspect.
 * @returns `true` when no self-intersection is found.
 *
 * @example
 * ```ts
 * if (!hasNoSelfIntersection(spec)) throw new Error('track is self-intersecting');
 * ```
 */
export function hasNoSelfIntersection(spec: TrackSpec): boolean {
  const lastSegmentIndex = spec.segments.length - 1;

  for (
    let firstSegmentIndex = 0;
    firstSegmentIndex < spec.segments.length;
    firstSegmentIndex++
  ) {
    for (
      let secondSegmentIndex = firstSegmentIndex + 1;
      secondSegmentIndex < spec.segments.length;
      secondSegmentIndex++
    ) {
      if (
        isAdjacentSegmentPair(
          firstSegmentIndex,
          secondSegmentIndex,
          lastSegmentIndex,
        )
      ) {
        continue;
      }

      if (
        doLineSegmentsIntersect(
          spec.segments[firstSegmentIndex],
          spec.segments[secondSegmentIndex],
        )
      ) {
        return false;
      }
    }
  }

  return true;
}

/**
 * Validates that the two Tier 4 pit corridors do not overlap.
 *
 * Pit entry uses each corridor's axis-aligned bounding box directly, so overlap
 * would make ownership ambiguous and would collapse the one-pit-per-team rule.
 *
 * @param spec - Track specification containing optional pit metadata.
 * @returns `true` when the pit-corridor layout is valid.
 */
export function validatePitCorridorNonOverlap(spec: TrackSpec): true {
  const pitBoxes = spec.pitBoxes ?? [];

  for (
    let firstPitIndex = 0;
    firstPitIndex < pitBoxes.length;
    firstPitIndex++
  ) {
    for (
      let secondPitIndex = firstPitIndex + 1;
      secondPitIndex < pitBoxes.length;
      secondPitIndex++
    ) {
      if (
        doAxisAlignedBoxesOverlap(
          pitBoxes[firstPitIndex].entranceCorridor,
          pitBoxes[secondPitIndex].entranceCorridor,
        )
      ) {
        throw new RangeError(
          'TrackSpec pit entrance corridors must not overlap.',
        );
      }
    }
  }

  return true;
}

/**
 * Validates that each Tier 4 pit corridor stays reachable from the track.
 *
 * Reachability is approximated by checking whether the corridor center lies
 * within the segment half-width plus the corridor radius of any centerline
 * segment. This keeps the corridor AABB honest: cars must be able to enter it
 * from the driveable ribbon instead of teleporting into a detached pit zone.
 *
 * @param spec - Track specification containing optional pit metadata.
 * @returns `true` when every pit corridor is reachable.
 */
export function validatePitCorridorReachability(spec: TrackSpec): true {
  const pitBoxes = spec.pitBoxes ?? [];

  for (const pitBox of pitBoxes) {
    if (!isPitCorridorReachable(spec, pitBox.entranceCorridor)) {
      throw new RangeError(
        'TrackSpec pit entrance corridor must be reachable from the track.',
      );
    }
  }

  return true;
}

/**
 * Compares two coordinates using the track closed-loop tolerance.
 *
 * @param leftValue - First coordinate.
 * @param rightValue - Second coordinate.
 * @returns True when the coordinates are effectively equal.
 */
function areCloseEnough(leftValue: number, rightValue: number): boolean {
  return Math.abs(leftValue - rightValue) <= TRACK_CLOSED_LOOP_TOLERANCE;
}

/**
 * Returns true when a segment pair is adjacent in the closed-loop ordering.
 *
 * @param firstSegmentIndex - First segment index.
 * @param secondSegmentIndex - Second segment index.
 * @param lastSegmentIndex - Final segment index in the loop.
 * @returns True when the segments share the loop adjacency exemption.
 */
function isAdjacentSegmentPair(
  firstSegmentIndex: number,
  secondSegmentIndex: number,
  lastSegmentIndex: number,
): boolean {
  return (
    Math.abs(firstSegmentIndex - secondSegmentIndex) === 1 ||
    (firstSegmentIndex === 0 && secondSegmentIndex === lastSegmentIndex)
  );
}

/**
 * Returns true when two 2D line segments intersect or overlap.
 *
 * @param firstSegment - First line segment.
 * @param secondSegment - Second line segment.
 * @returns True when the two segments cross.
 */
function doLineSegmentsIntersect(
  firstSegment: TrackSpec['segments'][number],
  secondSegment: TrackSpec['segments'][number],
): boolean {
  const firstStart = { x: firstSegment.startX, y: firstSegment.startY };
  const firstEnd = { x: firstSegment.endX, y: firstSegment.endY };
  const secondStart = { x: secondSegment.startX, y: secondSegment.startY };
  const secondEnd = { x: secondSegment.endX, y: secondSegment.endY };
  const firstOrientation = resolveOrientation(
    firstStart,
    firstEnd,
    secondStart,
  );
  const secondOrientation = resolveOrientation(firstStart, firstEnd, secondEnd);
  const thirdOrientation = resolveOrientation(
    secondStart,
    secondEnd,
    firstStart,
  );
  const fourthOrientation = resolveOrientation(
    secondStart,
    secondEnd,
    firstEnd,
  );

  if (
    firstOrientation !== secondOrientation &&
    thirdOrientation !== fourthOrientation
  ) {
    return true;
  }

  return (
    (firstOrientation === 0 &&
      isPointOnSegment(firstStart, secondStart, firstEnd)) ||
    (secondOrientation === 0 &&
      isPointOnSegment(firstStart, secondEnd, firstEnd)) ||
    (thirdOrientation === 0 &&
      isPointOnSegment(secondStart, firstStart, secondEnd)) ||
    (fourthOrientation === 0 &&
      isPointOnSegment(secondStart, firstEnd, secondEnd))
  );
}

/**
 * Resolves the orientation of three points.
 *
 * @param startPoint - First point.
 * @param middlePoint - Second point.
 * @param endPoint - Third point.
 * @returns 0 for collinear, 1 for clockwise, 2 for counterclockwise.
 */
function resolveOrientation(
  startPoint: { x: number; y: number },
  middlePoint: { x: number; y: number },
  endPoint: { x: number; y: number },
): 0 | 1 | 2 {
  const crossProduct =
    (middlePoint.y - startPoint.y) * (endPoint.x - middlePoint.x) -
    (middlePoint.x - startPoint.x) * (endPoint.y - middlePoint.y);

  if (Math.abs(crossProduct) <= TRACK_CLOSED_LOOP_TOLERANCE) {
    return 0;
  }

  return crossProduct > 0 ? 1 : 2;
}

/**
 * Returns true when a collinear point falls within a segment's bounds.
 *
 * @param startPoint - Segment start.
 * @param point - Candidate point.
 * @param endPoint - Segment end.
 * @returns True when the point lies on the segment.
 */
function isPointOnSegment(
  startPoint: { x: number; y: number },
  point: { x: number; y: number },
  endPoint: { x: number; y: number },
): boolean {
  return (
    point.x <=
      Math.max(startPoint.x, endPoint.x) + TRACK_CLOSED_LOOP_TOLERANCE &&
    point.x >=
      Math.min(startPoint.x, endPoint.x) - TRACK_CLOSED_LOOP_TOLERANCE &&
    point.y <=
      Math.max(startPoint.y, endPoint.y) + TRACK_CLOSED_LOOP_TOLERANCE &&
    point.y >= Math.min(startPoint.y, endPoint.y) - TRACK_CLOSED_LOOP_TOLERANCE
  );
}

/**
 * Returns whether two axis-aligned rectangles overlap.
 *
 * @param firstBox - First rectangle.
 * @param secondBox - Second rectangle.
 * @returns True when the rectangles overlap with positive area.
 */
function doAxisAlignedBoxesOverlap(
  firstBox: TrackAabb,
  secondBox: TrackAabb,
): boolean {
  return !(
    firstBox.x + firstBox.width <= secondBox.x ||
    secondBox.x + secondBox.width <= firstBox.x ||
    firstBox.y + firstBox.height <= secondBox.y ||
    secondBox.y + secondBox.height <= firstBox.y
  );
}

/**
 * Returns whether a pit entrance corridor can be reached from the track ribbon.
 *
 * When spline samples are present, reachability is checked against the smooth
 * sampled lane center (accurate for corridors placed via spline-normal offsets).
 * When no spline samples exist (e.g., unit-test fixtures), the check falls back
 * to the polygon segment approximation.
 *
 * @param spec - Track specification.
 * @param corridor - Candidate pit entrance corridor.
 * @returns True when the corridor lies within the reachable ribbon distance.
 */
function isPitCorridorReachable(spec: TrackSpec, corridor: TrackAabb): boolean {
  const corridorCenterX = corridor.x + corridor.width / 2;
  const corridorCenterY = corridor.y + corridor.height / 2;
  const corridorRadius = Math.max(corridor.width, corridor.height) / 2;

  // Prefer spline samples when available — they represent the actual smooth
  // lane center that corridors are offset from.
  if (spec.splineSamples.length > 0) {
    return spec.splineSamples.some((sample) => {
      const distanceToSample = Math.hypot(
        corridorCenterX - sample.x,
        corridorCenterY - sample.y,
      );
      return distanceToSample <= sample.width / 2 + corridorRadius;
    });
  }

  // Fallback: use polygon segment approximation (for unit-test fixtures with
  // no spline samples).
  return spec.segments.some((segment) => {
    const distanceToSegment = resolvePointToSegmentDistance(
      { x: corridorCenterX, y: corridorCenterY },
      { x: segment.startX, y: segment.startY },
      { x: segment.endX, y: segment.endY },
    );
    return distanceToSegment <= segment.width / 2 + corridorRadius;
  });
}

/**
 * Resolves the shortest distance from a point to a line segment.
 *
 * @param point - Query point.
 * @param segmentStart - Segment start point.
 * @param segmentEnd - Segment end point.
 * @returns Euclidean point-to-segment distance.
 */
function resolvePointToSegmentDistance(
  point: { x: number; y: number },
  segmentStart: { x: number; y: number },
  segmentEnd: { x: number; y: number },
): number {
  const deltaX = segmentEnd.x - segmentStart.x;
  const deltaY = segmentEnd.y - segmentStart.y;
  const segmentLengthSquared = deltaX * deltaX + deltaY * deltaY;

  if (segmentLengthSquared <= TRACK_CLOSED_LOOP_TOLERANCE) {
    return Math.hypot(point.x - segmentStart.x, point.y - segmentStart.y);
  }

  const projection =
    ((point.x - segmentStart.x) * deltaX +
      (point.y - segmentStart.y) * deltaY) /
    segmentLengthSquared;
  const clampedProjection = Math.max(0, Math.min(1, projection));
  const closestPoint = {
    x: segmentStart.x + deltaX * clampedProjection,
    y: segmentStart.y + deltaY * clampedProjection,
  };

  return Math.hypot(point.x - closestPoint.x, point.y - closestPoint.y);
}
