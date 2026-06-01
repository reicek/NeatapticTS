import type {
  SplineSample,
  TrackAabb,
  TrackGeneratorInput,
  TrackPitBox,
  TrackSpec,
} from './track.generator.types';
import {
  buildTrackSplineSamples,
  resolveSplineSampleFrame,
} from './track.spline.utils';
import { validateTrackSpec } from './track.validation';

const TRACK_BUCKET_RADIUS_BY_SIZE: Record<string, number> = {
  small: 120,
  medium: 180,
  large: 240,
};

const TRACK_BUCKET_WIDTH_BY_SIZE: Record<string, number> = {
  small: 18,
  medium: 24,
  large: 30,
};

const DEFAULT_TRACK_RADIUS = 180;
const DEFAULT_TRACK_WIDTH = 24;
const TRACK_MIN_VERTEX_COUNT = 6;
const TRACK_VERTEX_VARIATION_COUNT = 3;
const TRACK_RADIUS_VARIATION_RATIO = 0.18;
const TRACK_WIDTH_VARIATION_RATIO = 0.1;
const TRACK_GEOMETRY_PRECISION_FACTOR = 1_000;
const DEFAULT_TRACK_VIEWPORT_EDGE_PADDING_RATIO = 0.08;
const MIN_TRACK_VIEWPORT_EDGE_PADDING_RATIO = 0;
const MAX_TRACK_VIEWPORT_EDGE_PADDING_RATIO = 0.3;
const ALTERNATING_PIT_PROGRESS_SAMPLES = [
  0.083333,
  0.25,
  0.416667,
  0.583333,
  0.75,
  0.916667,
] as const;
const PIT_BOX_WIDTH = 18;
const PIT_BOX_HEIGHT = 12;
const PIT_CORRIDOR_WIDTH = 12;
const PIT_CORRIDOR_HEIGHT = 12;
const PIT_BOX_OFFSET_MULTIPLIER = 1.25;
/** Fraction of track width used to push the corridor center away from the racing line. */
const PIT_CORRIDOR_CENTERLINE_OFFSET_MULTIPLIER = 0.5;

/**
 * Generates a deterministic closed-loop `TrackSpec` from the given seed,
 * layout version, and size bucket.
 *
 * The same `(seed, layoutVersion, sizeBucket, viewport)` tuple always produces
 * the same `TrackSpec`. The spec is frozen into the race pack at episode reset;
 * no regeneration occurs during a live viewport resize. Tier 4+ also derives
 * three pit boxes per team (six total) at stable lap-progress anchors, with
 * on-ribbon entrance-corridor AABBs and off-line rendered stall rectangles.
 *
 * @param input - Determinism key for the generation algorithm.
 * @returns A frozen `TrackSpec` whose segments form a valid closed loop.
 *
 * @example
 * ```ts
 * const spec = generateTrack({ seed: 42, layoutVersion: 1, sizeBucket: 'medium' });
 * spec.segments; // ordered closed-loop centerline segments
 * spec.pitBoxes?.length; // 6
 * ```
 */
export function generateTrack(input: TrackGeneratorInput): TrackSpec {
  const combinedSeed = resolveCombinedSeed(input);
  const random = createDeterministicRandom(combinedSeed);
  const baseRadius =
    TRACK_BUCKET_RADIUS_BY_SIZE[input.sizeBucket] ?? DEFAULT_TRACK_RADIUS;
  const baseWidth =
    TRACK_BUCKET_WIDTH_BY_SIZE[input.sizeBucket] ?? DEFAULT_TRACK_WIDTH;
  const radiusProfile = resolveTrackRadiusProfile(input, baseRadius, baseWidth);
  const vertexCount =
    TRACK_MIN_VERTEX_COUNT +
    Math.floor(random() * TRACK_VERTEX_VARIATION_COUNT);
  const rotationRadians = random() * Math.PI * 2;
  const vertices = Array.from({ length: vertexCount }, (_, vertexIndex) => {
    const angleRadians =
      rotationRadians + (Math.PI * 2 * vertexIndex) / vertexCount;
    const radiusMultiplier =
      1 -
      TRACK_RADIUS_VARIATION_RATIO +
      random() * TRACK_RADIUS_VARIATION_RATIO * 2;
    const radius = baseRadius * radiusMultiplier;

    return {
      x: roundTrackGeometry(Math.cos(angleRadians) * radius * radiusProfile.x),
      y: roundTrackGeometry(Math.sin(angleRadians) * radius * radiusProfile.y),
    };
  });
  const segments = vertices.map((startVertex, segmentIndex) => {
    const endVertex = vertices[(segmentIndex + 1) % vertices.length];
    const widthMultiplier =
      1 -
      TRACK_WIDTH_VARIATION_RATIO +
      random() * TRACK_WIDTH_VARIATION_RATIO * 2;

    return {
      startX: startVertex.x,
      startY: startVertex.y,
      endX: endVertex.x,
      endY: endVertex.y,
      width: roundTrackGeometry(baseWidth * widthMultiplier),
    };
  });
  const splineSamples = buildTrackSplineSamples(segments);
  const pitBoxes = buildPitBoxes(splineSamples);
  const spec = {
    seed: input.seed,
    layoutVersion: input.layoutVersion,
    sizeBucket: input.sizeBucket,
    segments,
    splineSamples,
    pitBoxes,
  };

  validateTrackSpec(spec);
  return spec;
}

/**
 * Deeply freezes a `TrackSpec` so that any mutation attempt throws a
 * `TypeError` in strict mode.
 *
 * Call this immediately after generation to satisfy the frozen-at-reset
 * contract: once the spec is locked into the race pack it must be immutable.
 *
 * @param spec - The `TrackSpec` to freeze.
 * @returns A `Readonly<TrackSpec>` that rejects mutation.
 *
 * @example
 * ```ts
 * const spec = freezeTrackSpec(generateTrack(input));
 * spec.segments.push(segment); // throws TypeError in strict mode
 * ```
 */
export function freezeTrackSpec(spec: TrackSpec): Readonly<TrackSpec> {
  const frozenSegments = spec.segments.map((segment) =>
    Object.freeze({ ...segment }),
  );
  const frozenSplineSamples = spec.splineSamples.map((splineSample) =>
    Object.freeze({ ...splineSample }),
  );
  const frozenPitBoxes = spec.pitBoxes?.map((pitBox) =>
    Object.freeze({
      ...pitBox,
      boxCenter:
        pitBox.boxCenter === undefined
          ? undefined
          : Object.freeze({ ...pitBox.boxCenter }),
      entranceCorridor: Object.freeze({ ...pitBox.entranceCorridor }),
      pitBox:
        pitBox.pitBox === undefined
          ? undefined
          : Object.freeze({ ...pitBox.pitBox }),
    }),
  ) as Readonly<TrackSpec['pitBoxes']>;

  return Object.freeze({
    ...spec,
    segments: Object.freeze(frozenSegments),
    splineSamples: Object.freeze(frozenSplineSamples),
    pitBoxes:
      frozenPitBoxes === undefined ? undefined : Object.freeze(frozenPitBoxes),
  });
}

/**
 * Combines the generator determinism tuple into one 32-bit seed.
 *
 * @param input - Track generator determinism key.
 * @returns Unsigned 32-bit seed for the local PRNG.
 */
function resolveCombinedSeed(input: TrackGeneratorInput): number {
  let combinedSeed = input.seed ^ (input.layoutVersion * 0x9e3779b9);

  for (const character of input.sizeBucket) {
    combinedSeed = Math.imul(combinedSeed ^ character.charCodeAt(0), 16777619);
  }

  return combinedSeed >>> 0;
}

/**
 * Creates a deterministic xorshift32 PRNG.
 *
 * @param initialSeed - Unsigned 32-bit seed.
 * @returns Stable pseudo-random number generator in the range [0, 1).
 */
function createDeterministicRandom(initialSeed: number): () => number {
  let state = initialSeed === 0 ? 0x6d2b79f5 : initialSeed >>> 0;

  return () => {
    state ^= state << 13;
    state ^= state >>> 17;
    state ^= state << 5;

    return (state >>> 0) / 0x100000000;
  };
}

/**
 * Builds deterministic alternating-team pit metadata from spline progress.
 *
 * @param splineSamples - Shared lane-center samples for the generated track.
 * @returns Six frozen pit-box descriptors in `[0, 1, 0, 1, 0, 1]` ownership order.
 */
function buildPitBoxes(
  splineSamples: readonly SplineSample[],
): readonly TrackPitBox[] {
  return ALTERNATING_PIT_PROGRESS_SAMPLES.map((pitProgress, pitIndex) => {
    const teamIndex = (pitIndex % 2) as 0 | 1;
    const normalDirection = teamIndex === 0 ? 1 : -1;

    return buildPitBoxForTeam(
      teamIndex,
      resolvePitAnchorSample(splineSamples, pitProgress),
      splineSamples,
      normalDirection,
    );
  });
}

/**
 * Resolves the spline sample nearest the requested lap progress.
 *
 * @param splineSamples - Shared lane-center samples for the generated track.
 * @param progress01 - Closed `[0, 1]` lap-progress target.
 * @returns Stable sampled anchor for the requested progress bucket.
 */
function resolvePitAnchorSample(
  splineSamples: readonly SplineSample[],
  progress01: number,
): SplineSample {
  const sampleIndex = Math.min(
    splineSamples.length - 1,
    Math.max(0, Math.floor(splineSamples.length * progress01)),
  );

  return (
    splineSamples[sampleIndex] ?? {
      x: 0,
      y: 0,
      width: DEFAULT_TRACK_WIDTH,
      segmentIndex: 0,
      sampleIndexWithinSegment: 0,
      globalIndex: 0,
    }
  );
}

/**
 * Builds one team's pit metadata from a spline anchor and normal direction.
 *
 * The entrance corridor is offset half a track width away from the centerline
 * so cars on the normal racing line do not accidentally trigger pit stops.
 *
 * @param teamIndex - Owning team index.
 * @param anchorSample - Spline sample anchoring the pit location.
 * @param normalDirection - Signed side selector (`1` or `-1`).
 * @returns Deterministic pit metadata for one team.
 */
function buildPitBoxForTeam(
  teamIndex: 0 | 1,
  anchorSample: SplineSample,
  splineSamples: readonly SplineSample[],
  normalDirection: 1 | -1,
): TrackPitBox {
  const sampleFrame = resolveSplineSampleFrame(
    splineSamples,
    anchorSample.globalIndex,
  );
  // Offset the corridor center toward the pit side of the track so cars
  // on the racing line (centerline) do not accidentally trigger pit stops.
  const corridorCenterPoint = resolveOffsetPoint(
    anchorSample.x,
    anchorSample.y,
    sampleFrame.normalX,
    sampleFrame.normalY,
    anchorSample.width *
      PIT_CORRIDOR_CENTERLINE_OFFSET_MULTIPLIER *
      normalDirection,
  );
  const corridorCenter = {
    x: roundTrackGeometry(corridorCenterPoint.x),
    y: roundTrackGeometry(corridorCenterPoint.y),
  };
  const boxCenter = resolveOffsetPoint(
    anchorSample.x,
    anchorSample.y,
    sampleFrame.normalX,
    sampleFrame.normalY,
    anchorSample.width * PIT_BOX_OFFSET_MULTIPLIER * normalDirection,
  );

  return {
    teamIndex,
    boxCenter: {
      x: roundTrackGeometry(boxCenter.x),
      y: roundTrackGeometry(boxCenter.y),
    },
    entranceCorridor: createAxisAlignedBox(
      corridorCenter.x,
      corridorCenter.y,
      PIT_CORRIDOR_WIDTH,
      PIT_CORRIDOR_HEIGHT,
    ),
    pitBox: createAxisAlignedBox(
      boxCenter.x,
      boxCenter.y,
      PIT_BOX_WIDTH,
      PIT_BOX_HEIGHT,
    ),
  };
}

/**
 * Resolves an offset point along the supplied local normal vector.
 *
 * @param x - Anchor X coordinate.
 * @param y - Anchor Y coordinate.
 * @param normalX - Unit normal X component.
 * @param normalY - Unit normal Y component.
 * @param offsetDistance - Signed offset distance.
 * @returns Offset point in world coordinates.
 */
function resolveOffsetPoint(
  x: number,
  y: number,
  normalX: number,
  normalY: number,
  offsetDistance: number,
): { readonly x: number; readonly y: number } {
  return {
    x: roundTrackGeometry(x + normalX * offsetDistance),
    y: roundTrackGeometry(y + normalY * offsetDistance),
  };
}

/**
 * Creates an axis-aligned rectangle from center-point inputs.
 *
 * @param centerX - Rectangle center X coordinate.
 * @param centerY - Rectangle center Y coordinate.
 * @param width - Rectangle width.
 * @param height - Rectangle height.
 * @returns Rounded axis-aligned box descriptor.
 */
function createAxisAlignedBox(
  centerX: number,
  centerY: number,
  width: number,
  height: number,
): TrackAabb {
  return {
    x: roundTrackGeometry(centerX - width / 2),
    y: roundTrackGeometry(centerY - height / 2),
    width: roundTrackGeometry(width),
    height: roundTrackGeometry(height),
  };
}

/**
 * Resolves radius scale factors that adapt the loop to viewport aspect ratio.
 *
 * The profile scales each axis from the available viewport half-size after
 * edge padding and lane-width safety margins are reserved, so generated loops
 * fill the visible area while preserving rounded circle/oval geometry.
 *
 * @param input - Generator input possibly carrying viewport metadata.
 * @param baseRadius - Size-bucket baseline radius before viewport scaling.
 * @param baseWidth - Size-bucket baseline lane width before viewport scaling.
 * @returns Radius multipliers for X and Y axes.
 */
function resolveTrackRadiusProfile(
  input: TrackGeneratorInput,
  baseRadius: number,
  baseWidth: number,
): { readonly x: number; readonly y: number } {
  const viewport = input.viewport;
  if (
    viewport === undefined ||
    !Number.isFinite(viewport.width) ||
    !Number.isFinite(viewport.height) ||
    viewport.width <= 0 ||
    viewport.height <= 0
  ) {
    return { x: 1, y: 1 };
  }

  const safeEdgePaddingRatio = clampNumber(
    viewport.edgePaddingRatio ?? DEFAULT_TRACK_VIEWPORT_EDGE_PADDING_RATIO,
    MIN_TRACK_VIEWPORT_EDGE_PADDING_RATIO,
    MAX_TRACK_VIEWPORT_EDGE_PADDING_RATIO,
  );
  const usableWidth = Math.max(
    1,
    viewport.width * (1 - safeEdgePaddingRatio * 2),
  );
  const usableHeight = Math.max(
    1,
    viewport.height * (1 - safeEdgePaddingRatio * 2),
  );
  const maxRadiusMultiplier = 1 + TRACK_RADIUS_VARIATION_RATIO;
  const maxWidthMultiplier = 1 + TRACK_WIDTH_VARIATION_RATIO;
  const laneHalfWidth = (baseWidth * maxWidthMultiplier) / 2;
  const availableRadiusX = Math.max(1, usableWidth / 2 - laneHalfWidth);
  const availableRadiusY = Math.max(1, usableHeight / 2 - laneHalfWidth);
  const horizontalScale = clampNumber(
    availableRadiusX / (baseRadius * maxRadiusMultiplier),
    0.1,
    Number.POSITIVE_INFINITY,
  );
  const verticalScale = clampNumber(
    availableRadiusY / (baseRadius * maxRadiusMultiplier),
    0.1,
    Number.POSITIVE_INFINITY,
  );

  return {
    x: roundTrackGeometry(horizontalScale),
    y: roundTrackGeometry(verticalScale),
  };
}

/**
 * Rounds geometry values so serialized specs stay byte-stable.
 *
 * @param value - Floating-point geometry value.
 * @returns Rounded geometry value.
 */
function roundTrackGeometry(value: number): number {
  return (
    Math.round(value * TRACK_GEOMETRY_PRECISION_FACTOR) /
    TRACK_GEOMETRY_PRECISION_FACTOR
  );
}

/**
 * Clamps a number to the closed range `[minValue, maxValue]`.
 *
 * @param value - Input value.
 * @param minValue - Lower bound.
 * @param maxValue - Upper bound.
 * @returns Clamped value.
 */
function clampNumber(value: number, minValue: number, maxValue: number): number {
  return Math.max(minValue, Math.min(maxValue, value));
}
