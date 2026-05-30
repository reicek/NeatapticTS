/**
 * A single 2D line segment forming part of the track centerline.
 *
 * Segments are ordered from start to finish of the closed loop.
 * The closed-loop invariant requires that `segments.at(-1).endX === segments[0].startX`
 * and `segments.at(-1).endY === segments[0].startY`.
 */
export type TrackSegment = {
  /** X coordinate of the segment start in logical world units. */
  startX: number;
  /** Y coordinate of the segment start in logical world units. */
  startY: number;
  /** X coordinate of the segment end in logical world units. */
  endX: number;
  /** Y coordinate of the segment end in logical world units. */
  endY: number;
  /** Driveable width of the track at this segment in logical world units. */
  width: number;
};

/**
 * One precomputed Catmull-Rom lane-center sample shared across racing systems.
 *
 * The ownership metadata keeps each sample attached to its source chord while
 * preserving the stable closed-loop ordering of the sampled ribbon.
 */
export type SplineSample = {
  /** X coordinate of the sampled lane center in logical world units. */
  readonly x: number;
  /** Y coordinate of the sampled lane center in logical world units. */
  readonly y: number;
  /** Interpolated driveable width at this sample in logical world units. */
  readonly width: number;
  /** Owning raw chord segment index for this sample. */
  readonly segmentIndex: number;
  /** Sample position within the owning segment strip. */
  readonly sampleIndexWithinSegment: number;
  /** Stable global sample index around the closed loop. */
  readonly globalIndex: number;
};

/** Axis-aligned pit geometry used for occupancy checks and renderer overlays. */
export type TrackAabb = {
  /** Left edge of the rectangle in world units. */
  readonly x: number;
  /** Top edge of the rectangle in world units. */
  readonly y: number;
  /** Rectangle width in world units. */
  readonly width: number;
  /** Rectangle height in world units. */
  readonly height: number;
};

/**
 * Per-team pit metadata frozen into the track specification.
 *
 * Tier 4 generates exactly one pit descriptor per team. `entranceCorridor` is
 * the axis-aligned world-space box used by the environment and validators to
 * decide whether a car has legally reached pit entry, while `pitBox` is the
 * off-line stall rectangle drawn by the renderer.
 */
export type TrackPitBox = {
  /** Team index owning this pit box (`0 = Team A`, `1 = Team B`). */
  readonly teamIndex: 0 | 1;
  /** Optional center point of the team's pit box. */
  readonly boxCenter?: {
    /** Pit-box center X position in world units. */
    readonly x: number;
    /** Pit-box center Y position in world units. */
    readonly y: number;
  };
  /** Axis-aligned entrance corridor AABB used for pit-entry checks and reachability validation. */
  readonly entranceCorridor: TrackAabb;
  /** Optional rendered pit-box rectangle derived from `boxCenter`. */
  readonly pitBox?: TrackAabb;
};

/**
 * Frozen specification for one procedurally generated track layout.
 *
 * A `TrackSpec` is frozen into the race pack at episode reset and never
 * regenerated during a live resize. The determinism key is
 * `seed + layoutVersion + sizeBucket`. Generated racing tracks may also carry a
 * fixed `[teamA, teamB]` `pitBoxes` tuple so every team gets exactly one pit
 * entrance corridor and one rendered stall.
 */
export type TrackSpec = {
  /** Seed used to drive the PRNG during generation. */
  readonly seed: number;
  /** Generation algorithm version — increment to invalidate cached specs. */
  readonly layoutVersion: number;
  /** Quantized canvas size bucket used as part of the determinism key. */
  readonly sizeBucket: string;
  /** Ordered closed-loop centerline segments. Read-only after freeze. */
  readonly segments: readonly TrackSegment[];
  /** Shared sampled lane-center geometry used by the renderer and controllers. */
  readonly splineSamples: readonly SplineSample[];
  /** Optional `[Team A, Team B]` pit-box tuple. Present on generated racing tracks. */
  readonly pitBoxes?: readonly [TrackPitBox, TrackPitBox];
};

/**
 * Inputs consumed by the deterministic track generator.
 *
 * Together they form the determinism key: same values always produce the same
 * `TrackSpec`.
 */
export type TrackGeneratorInput = {
  /** PRNG seed — determines the geometry of the generated layout. */
  seed: number;
  /** Algorithm version tag — determines which generation procedure to use. */
  layoutVersion: number;
  /** Coarse canvas size bucket — determines the target world extent. */
  sizeBucket: string;
};
