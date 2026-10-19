/**
 * Recycled Neatenstein public/agent billboard sprites for the neatRacing
 * right-world render pass (slice P1-neatenstein-public, Step 14D, AC-049).
 *
 * Loads a 2D sprite atlas derived from the examples/neatenstein public sprites
 * and builds screen-facing billboard quads placed on the right-hand side of
 * the track. Billboards keep a 1:1 world-space pixel scale; the right-world
 * render pass applies the same `focal / distance` perspective projection used
 * for 3D voxels, so distant crowds shrink while nearby ones stay readable.
 *
 * GREEN TARGET (04-implementing must make exactly these pass):
 * - Exports {@link loadPublicSpriteAtlas} returning a non-empty atlas whose
 *   source is 'neatenstein-public'.
 * - Exports {@link selectPublicSpriteFrame} returning a deterministic frame
 *   for a given (atlas, seed, index) tuple.
 * - Exports {@link buildPublicBillboards} returning right-side track billboards
 *   with positive screen-space quads and perspective-correct scale.
 * - Exports {@link computeBillboardScale} returning a constant 1.0 world scale
 *   so sprites share the same perspective anchor as voxels (AC-054).
 * - The billboard representation is 2D; it carries no voxel fallback fields.
 */

import { generateTrack } from '../../racing_curriculum/track/track.generator';
import { resolveSplineSampleFrame } from '../../racing_curriculum/track/track.spline.utils';
import {
  ROBOT_SPRITE_FRAMES,
  ROBOT_SPRITE_PALETTE,
  ROBOT_SPRITE_SCALE,
} from '../../neatenstein/robot-sprite-data';
import { buildRightWorldProps } from './right-world.builder';
import type { RightWorldProp } from './right-world.builder';
import { resolvePropHalfSize } from './right-world.prop-sizes';
import type { TrackSlabBuildInput } from './track.slabs';

/** One recycled Neatenstein public/agent sprite frame. */
export interface PublicSpriteFrame {
  /** Stable frame identifier. */
  readonly id: string;
  /** Frame width in pixels. */
  readonly width: number;
  /** Frame height in pixels. */
  readonly height: number;
  /** Raw RGBA pixel data. */
  readonly rgba: Uint8ClampedArray;
}

/** Sprite atlas loaded from the recycled Neatenstein public sprites. */
export interface PublicSpriteAtlas {
  readonly source: 'neatenstein-public';
  readonly frames: readonly PublicSpriteFrame[];
}

/** Minimal camera contract used to project world billboards into screen space. */
export interface PublicCamera {
  readonly position: readonly [number, number, number];
  readonly forward: readonly [number, number, number];
  readonly right: readonly [number, number, number];
  readonly near: number;
  readonly far: number;
  readonly fovDegrees: number;
}

/** A 2D billboard instance placed in world space and projected to screen space. */
export interface PublicBillboard {
  /**
   * World-space foot/base anchor in meters.
   *
   * The y coordinate is the surface the billboard stands on (ground or prop top),
   * not the geometric center. The render pass expands the quad upward from this
   * point so the sprite stays anchored to the surface regardless of runtime
   * camera distance.
   */
  readonly position: readonly [number, number, number];
  /** Selected atlas frame id (build-time hint; the pass recomputes it live). */
  readonly frameId: string;
  /** Runtime role used by the pass to recompute direction/walk/gaze each frame. */
  readonly runtimeRole?: 'seated' | 'walker';
  /** Deterministic seed carried through from the builder. */
  readonly seed?: number;
  /** Billboard index carried through from the builder. */
  readonly index?: number;
  /** Preselected walk direction index (0..7) for walkers. */
  readonly walkDirectionIndex?: number;
  /**
   * World-space scale factor relative to the atlas frame's pixel size.
   *
   * Kept at 1:1 with voxel world size so perspective shrink is handled by the
   * shared `focal / distance` projection in the right-world pass rather than
   * by an extra per-distance scale factor.
   */
  readonly scale: number;
  /** Distance from the camera in meters. */
  readonly distance: number;
  /** Projection-space quad as [left, top, right, bottom] in clip/projection units. */
  readonly quad: readonly [number, number, number, number];
}

/** Density budget for public/agent billboards per track. */
const NEATRACING_PUBLIC_BILLBOARD_DENSITY_BUDGET = 32;

/** Meters per sprite pixel used to turn frame pixels into world-space size. */
const NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M = 0.05;

/** Small vertical clearance so a surface-aligned billboard is strictly above scenery. */
const NEATRACING_BILLBOARD_SURFACE_EPSILON_M = 0.001;

/** Minimum distance used to avoid division by zero in projection math. */
const BILLBOARD_MIN_DISTANCE_M = 0.1;

/** World-space scale that preserves a 1:1 pixel-to-meter ratio.
 *
 * With this value, the billboard's world width equals
 * `frame.width * NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M`, which is then
 * perspective-projected by the right-world pass exactly like a voxel of the
 * same world size. Distant sprites shrink only because of the shared
 * `focal / distance` projection, not because of an extra per-distance scale
 * factor.
 */
const BILLBOARD_WORLD_SCALE = 1.0;

/** RGBA channel count used when decoding Neatenstein palette data. */
const RGBA_CHANNELS = 4;

/** Canonical 8-direction order used by the Neatenstein robot sprite data. */
export const DIRECTION_ORDER = [
  'front',
  'frontRight',
  'right',
  'backRight',
  'back',
  'backLeft',
  'left',
  'frontLeft',
] as const;

/** Diverse neon accent palette used for spectator public sprites (AC-062). */
const PUBLIC_NEON_ACCENT_PALETTE: readonly (readonly [number, number, number, number])[] = [
  [0, 255, 255, 255],
  [255, 0, 255, 255],
  [0, 255, 128, 255],
  [255, 255, 0, 255],
  [255, 0, 128, 255],
  [128, 0, 255, 255],
];

/** Encoded robot sprite frame shape reused from Neatenstein. */
type EncodedFrame = readonly (readonly number[])[];

/**
 * Loads the recycled Neatenstein public/agent sprite atlas.
 *
 * @returns A non-empty 2D sprite atlas whose source is 'neatenstein-public'.
 *
 * @example
 * ```ts
 * const atlas = loadPublicSpriteAtlas();
 * console.log(atlas.source); // 'neatenstein-public'
 * console.log(atlas.frames.length > 0); // true
 * ```
 */
export function loadPublicSpriteAtlas(): PublicSpriteAtlas {
  return {
    source: 'neatenstein-public',
    frames: buildPublicSpriteAtlasFrames(),
  };
}

/**
 * Selects a deterministic sprite frame from the atlas.
 *
 * @param atlas - Public sprite atlas.
 * @param seed - Deterministic seed for frame selection.
 * @param index - Billboard index used to vary the selected frame.
 * @returns A frame with positive dimensions.
 *
 * @example
 * ```ts
 * const atlas = loadPublicSpriteAtlas();
 * const frame = selectPublicSpriteFrame(atlas, 42, 0);
 * console.log(frame.width > 0); // true
 * console.log(frame.height > 0); // true
 * ```
 */
export function selectPublicSpriteFrame(
  atlas: PublicSpriteAtlas,
  seed: number,
  index: number,
): PublicSpriteFrame {
  const random = createDeterministicRandom(seed ^ (index * 0x9e3779b9));
  const frameIndex = Math.floor(random() * atlas.frames.length);

  return atlas.frames[frameIndex]!;
}

/**
 * Returns the billboard world-space scale relative to its raw sprite pixel size.
 *
 * The returned value is a constant 1.0 so the billboard keeps a 1:1
 * pixel-to-meter proportion with the voxel world. The right-world pass then
 * applies the same `focal / distance` perspective projection used for 3D
 * voxels, so distant sprites shrink at the same rate as geometry of the same
 * world size.
 *
 * @param distance - Distance from the camera in meters (kept for API
 *   stability; the caller is responsible for perspective projection).
 * @returns World-space scale of 1.0 for all distances.
 *
 * @example
 * ```ts
 * const nearScale = computeBillboardScale(5);
 * const farScale = computeBillboardScale(50);
 * console.log(nearScale).toBeCloseTo(1, 6); // 1:1 proportion
 * console.log(farScale).toBeCloseTo(1, 6);  // perspective handles shrink
 * ```
 */
export function computeBillboardScale(distance: number): number {
  // Distance is intentionally ignored: the billboard world size stays at 1:1
  // and the right-world render pass applies perspective projection.
  void distance;

  return BILLBOARD_WORLD_SCALE;
}

/**
 * Resolves a world yaw angle in degrees to the nearest 8-direction sprite label.
 *
 * @param yawDegrees - World yaw in degrees. 0 faces +Z (front), positive
 *   values rotate toward +X (right).
 * @returns One of the eight Neatenstein direction labels.
 */
export function resolveDirectionId(yawDegrees: number): string {
  const step = 360 / DIRECTION_ORDER.length;
  const normalized = ((yawDegrees % 360) + 360 + step / 2) % 360;
  const index = Math.floor(normalized / step) % DIRECTION_ORDER.length;

  return DIRECTION_ORDER[index]!;
}

/**
 * Resolves a robot-to-target heading to the nearest 8-direction index.
 *
 * @param robotPosition - Robot world-space foot position.
 * @param targetPosition - Target world-space position to face.
 * @returns Index in {@link DIRECTION_ORDER} (0..7).
 */
export function resolveDirectionIndexFromTarget(
  robotPosition: readonly [number, number, number],
  targetPosition: readonly [number, number, number],
): number {
  const dx = targetPosition[0] - robotPosition[0];
  const dz = targetPosition[2] - robotPosition[2];
  const yawDegrees = (Math.atan2(dx, dz) * 180) / Math.PI;
  const step = 360 / DIRECTION_ORDER.length;
  const normalized = ((yawDegrees % 360) + 360 + step / 2) % 360;

  return Math.floor(normalized / step) % DIRECTION_ORDER.length;
}

/**
 * Returns the 8-direction stand frame id for a robot looking at a target point.
 */
function selectPublicLookAtCarFrameId(
  atlas: PublicSpriteAtlas,
  robotPosition: readonly [number, number, number],
  targetPosition: readonly [number, number, number],
): string {
  void atlas;
  const directionIndex = resolveDirectionIndexFromTarget(robotPosition, targetPosition);

  return `${DIRECTION_ORDER[directionIndex]!}-stand`;
}

/** Numeric index of the stand pose inside a direction×pose lookup table. */
export const PUBLIC_POSE_INDEX_STAND = 0;
/** Numeric index of the first walk pose inside a direction×pose lookup table. */
export const PUBLIC_POSE_INDEX_WALK1 = 1;
/** Numeric index of the second walk pose inside a direction×pose lookup table. */
export const PUBLIC_POSE_INDEX_WALK2 = 2;
/** Number of poses stored per direction in the runtime lookup table. */
export const PUBLIC_POSE_COUNT = 3;

/** Slow walk cycle period in seconds (must be >= 2.4s per directed fix). */
const PUBLIC_WALK_CYCLE_PERIOD_SECONDS = 2.4;

/**
 * Selects the current walk pose index for a walker at a given clock time.
 *
 * The cycle is slow (2.4s per pose) and deterministic for a fixed clock.
 *
 * @param timeSeconds - Monotonic clock time in seconds.
 * @returns 0 for walk1, 1 for walk2.
 */
function selectPublicWalkPoseIndex(timeSeconds: number): number {
  return Math.floor(timeSeconds / PUBLIC_WALK_CYCLE_PERIOD_SECONDS) % 2;
}

/**
 * Deterministic slow look-around pause for a subset of walkers.
 *
 * A walker glances at the car only during a short window every 16 seconds,
 * and only for ~20% of walkers chosen deterministically from seed+index.
 * The math is branch-free aside from the boolean result and allocates no
 * objects, keeping the per-frame draw path allocation-free.
 */
function isLookAroundPause(
  seed: number,
  index: number,
  timeSeconds: number,
): boolean {
  const period = 16.0;
  const inWindow = (timeSeconds % period) < PUBLIC_WALK_CYCLE_PERIOD_SECONDS;
  if (!inWindow) {
    return false;
  }
  const phase = Math.floor(timeSeconds / period);

  return (((seed ^ index ^ (phase * 0x9e3779b9)) >>> 0) % 5) === 0;
}

/**
 * Computes the runtime atlas frame index for a public/agent billboard.
 *
 * No strings or objects are allocated: the result is read directly from the
 * prebuilt direction×pose index table.
 *
 * @param billboard - Billboard with runtime role metadata.
 * @param cameraWorld - Current camera world position.
 * @param carPosition - Live car world position for gaze.
 * @param timeSeconds - Monotonic frame time in seconds.
 * @param indexTable - Prebuilt direction×pose atlas index table.
 * @returns Atlas frame index, or -1 if the frame is missing.
 */
export function computePublicRuntimeFrameIndex(
  billboard: PublicBillboard,
  cameraWorld: readonly [number, number, number],
  carPosition: readonly [number, number, number],
  timeSeconds: number,
  indexTable: Int16Array,
): number {
  void cameraWorld;
  const role = billboard.runtimeRole ?? 'walker';
  const seed = billboard.seed ?? 0;
  const index = billboard.index ?? 0;
  let directionIndex: number;
  let poseIndex: number;

  if (role === 'seated' || isLookAroundPause(seed, index, timeSeconds)) {
    directionIndex = resolveDirectionIndexFromTarget(billboard.position, carPosition);
    poseIndex = PUBLIC_POSE_INDEX_STAND;
  } else {
    directionIndex = billboard.walkDirectionIndex ?? 0;
    poseIndex = selectPublicWalkPoseIndex(timeSeconds) === 0
      ? PUBLIC_POSE_INDEX_WALK1
      : PUBLIC_POSE_INDEX_WALK2;
  }

  const tableIndex = directionIndex * PUBLIC_POSE_COUNT + poseIndex;
  const atlasIndex = indexTable[tableIndex];

  return atlasIndex !== undefined && atlasIndex >= 0 ? atlasIndex : -1;
}

/**
 * Selects a deterministic slow two-step walk frame.
 *
 * @returns Walk frame id, e.g. `'front-walk1'`.
 */
export function selectPublicWalkFrame(
  atlas: PublicSpriteAtlas,
  seed: number,
  index: number,
  timeSeconds: number,
): string {
  void atlas;
  const random = createDeterministicRandom(seed ^ (index * 0x9e3779b9));
  const directionIndex = Math.floor(random() * DIRECTION_ORDER.length) % DIRECTION_ORDER.length;
  const pose = selectPublicWalkPoseIndex(timeSeconds) === 0 ? 'walk1' : 'walk2';

  return `${DIRECTION_ORDER[directionIndex]!}-${pose}`;
}

/**
 * Builds deterministic seat positions on top of every outer-ring bleacher row.
 * Seats are spaced across the bleacher top so spectator billboards sit in the
 * stands instead of floating on the inner ground band (OBS-1).
 *
 * @param bleachers - Bleacher props emitted by the right-world builder.
 * @returns Seat anchors in world-space [x, topY, z] order, one per seat.
 */
function buildBleacherSeatPositions(
  bleachers: readonly RightWorldProp[],
): readonly [number, number, number][] {
  const seats: [number, number, number][] = [];
  const seatXOffsets = [-1.5, -0.5, 0.5, 1.5];
  const seatZOffsets = [-0.5, 0.5];

  for (const bleacher of bleachers) {
    const [px, py, pz] = bleacher.position;
    const [, pHy] = resolvePropHalfSize(bleacher.kind);
    const topY = py + 2 * pHy;

    for (const zOffset of seatZOffsets) {
      for (const xOffset of seatXOffsets) {
        seats.push([px + xOffset, topY + NEATRACING_BILLBOARD_SURFACE_EPSILON_M, pz + zOffset]);
      }
    }
  }

  return seats;
}

/**
 * Builds public/agent billboards for the right-world pass.
 *
 * Bleacher spectators face the camera with a stand frame so they look at the
 * car, while non-seated robots use a slow deterministic walk frame. All
 * billboards keep the foot-anchor convention and 1:1 world scale.
 *
 * @param input - Determinism tuple for the track layout.
 * @param camera - Camera used for projection and 8-direction frame selection.
 * @param atlas - Public/agent sprite atlas.
 * @returns Ordered list of 2D billboards on the right side of the track.
 */
export function buildPublicBillboards(
  input: TrackSlabBuildInput,
  camera: PublicCamera,
  atlas: PublicSpriteAtlas,
): readonly PublicBillboard[] {
  if (atlas.frames.length === 0) {
    return [];
  }

  const spec = generateTrack(input);
  const samples = spec.splineSamples;
  if (samples.length === 0) {
    return [];
  }

  const random = createDeterministicRandom(resolvePublicSpriteSeed(input, 0));
  const cameraUp = computeCameraUp(camera);
  const props = buildRightWorldProps(input);
  const targetCount = Math.min(
    NEATRACING_PUBLIC_BILLBOARD_DENSITY_BUDGET,
    Math.max(1, Math.floor(samples.length / 10)),
  );
  const bleacherSeats = buildBleacherSeatPositions(
    props.filter((prop) => prop.kind === 'bleacher'),
  );
  const billboards: PublicBillboard[] = [];

  for (let index = 0; index < targetCount; index += 1) {
    const sampleIndex = Math.floor((index * samples.length) / targetCount);
    const sample = samples[sampleIndex]!;
    const frame = resolveSplineSampleFrame(samples, sample.globalIndex);

    if (index < bleacherSeats.length) {
      const [seatX, , seatZ] = bleacherSeats[index]!;
      const seatPosition: readonly [number, number, number] = [seatX, 0, seatZ];
      const standFrameId = selectPublicLookAtCarFrameId(atlas, seatPosition, camera.position);
      const selectedFrame = atlas.frames.find((candidate) => candidate.id === standFrameId) ?? atlas.frames[0]!;
      const placed = placeBillboardOnSurface(
        seatX,
        seatZ,
        selectedFrame,
        camera,
        cameraUp,
        props,
      );
      billboards.push({
        ...placed,
        runtimeRole: 'seated',
        seed: input.seed,
        index,
        walkDirectionIndex: 0,
      });
    } else {
      const offset = sample.width / 2 + 6 + random() * 4;
      const generatorX = sample.x - frame.normalX * offset;
      const generatorY = sample.y - frame.normalY * offset;
      const walkDirectionIndex = Math.floor(random() * DIRECTION_ORDER.length) % DIRECTION_ORDER.length;
      const walkPose = selectPublicWalkPoseIndex(0) === 0 ? 'walk1' : 'walk2';
      const walkFrameId = `${DIRECTION_ORDER[walkDirectionIndex]!}-${walkPose}`;
      const selectedFrame = atlas.frames.find((candidate) => candidate.id === walkFrameId) ?? atlas.frames[0]!;
      const placed = placeBillboardOnSurface(
        generatorX,
        -generatorY,
        selectedFrame,
        camera,
        cameraUp,
        props,
      );

      billboards.push({
        ...placed,
        runtimeRole: 'walker',
        seed: input.seed,
        index,
        walkDirectionIndex,
      });
    }
  }

  return billboards;
}

/**
 * Builds a stable, sorted frame list from the Neatenstein robot sprite atlas.
 *
 * Sorting by id makes the frame order independent of object-key iteration
 * order, which keeps {@link selectPublicSpriteFrame} deterministic across
 * runtimes.
 *
 * @returns Decoded 2D RGBA frames recycled from `ROBOT_SPRITE_FRAMES`.
 */
function buildPublicSpriteAtlasFrames(): readonly PublicSpriteFrame[] {
  const entries: {
    readonly id: string;
    readonly frame: EncodedFrame;
    readonly accent: readonly [number, number, number, number];
  }[] = [];
  let entryIndex = 0;

  for (const [direction, poseMap] of Object.entries(ROBOT_SPRITE_FRAMES)) {
    for (const [pose, frame] of Object.entries(poseMap)) {
      const accent =
        PUBLIC_NEON_ACCENT_PALETTE[entryIndex % PUBLIC_NEON_ACCENT_PALETTE.length]!;
      entries.push({ id: `${direction}-${pose}`, frame: frame as EncodedFrame, accent });
      entryIndex += 1;
    }
  }

  return entries
    .toSorted((a, b) => a.id.localeCompare(b.id))
    .map((entry) => recolorFrameRgba(decodePublicFrame(entry.frame, entry.id), entry.accent));
}

/**
 * Decodes one encoded Neatenstein robot frame into a 2D RGBA sprite.
 *
 * @param frame - Encoded palette-index rows.
 * @param id - Stable frame identifier.
 * @returns Decoded RGBA sprite frame with positive dimensions.
 */
function decodePublicFrame(frame: EncodedFrame, id: string): PublicSpriteFrame {
  const logicalHeight = frame.length;
  const logicalWidth = logicalHeight > 0 ? frame[0]!.length : 0;
  const width = logicalWidth * ROBOT_SPRITE_SCALE;
  const height = logicalHeight * ROBOT_SPRITE_SCALE;
  const rgba = new Uint8ClampedArray(width * height * RGBA_CHANNELS);

  for (let y = 0; y < height; y += 1) {
    const logicalY = Math.floor(y / ROBOT_SPRITE_SCALE);
    const row = frame[logicalY];
    if (row === undefined) {
      continue;
    }

    for (let x = 0; x < width; x += 1) {
      const logicalX = Math.floor(x / ROBOT_SPRITE_SCALE);
      const paletteIndex = row[logicalX] ?? 0;
      const color = ROBOT_SPRITE_PALETTE[paletteIndex] ?? [0, 0, 0, 0];
      const offset = (y * width + x) * RGBA_CHANNELS;
      rgba[offset] = color[0]!;
      rgba[offset + 1] = color[1]!;
      rgba[offset + 2] = color[2]!;
      rgba[offset + 3] = color[3]!;
    }
  }

  return { id, width, height, rgba };
}

/**
 * Recolors the saturated red/orange accent pixels of a decoded sprite to a neon
 * accent. This is applied once at atlas build time, so the render path stays
 * allocation-free.
 *
 * @param frame - Decoded sprite frame.
 * @param accent - RGBA accent color to apply.
 * @returns The same frame reference, with accent pixels recolored.
 */
function recolorFrameRgba(
  frame: PublicSpriteFrame,
  accent: readonly [number, number, number, number],
): PublicSpriteFrame {
  const rgba = frame.rgba;

  for (let offset = 0; offset < rgba.length; offset += RGBA_CHANNELS) {
    const alpha = rgba[offset + 3]!;
    if (alpha < 32) {
      continue;
    }

    const red = rgba[offset]!;
    const green = rgba[offset + 1]!;
    const blue = rgba[offset + 2]!;

    // Detect the original Neatenstein red/orange accent family (palette
    // indices 5/6/7). Body grays and whites fall outside this threshold.
    if (red > 150 && green < 120 && blue < 80) {
      rgba[offset] = accent[0];
      rgba[offset + 1] = accent[1];
      rgba[offset + 2] = accent[2];
      // Preserve the original alpha so transparent accents stay transparent.
    }
  }

  return frame;
}

/**
 * Resolves a deterministic unsigned 32-bit seed for the public billboard builder.
 *
 * @param input - Determinism tuple `{ seed, layoutVersion, sizeBucket }`.
 * @param index - Index mixed into the seed so parallel slices do not collide.
 * @returns Unsigned 32-bit seed for the local PRNG.
 */
function resolvePublicSpriteSeed(input: TrackSlabBuildInput, index: number): number {
  let combinedSeed = input.seed ^ (input.layoutVersion * 0x9e3779b9);

  for (const character of input.sizeBucket) {
    combinedSeed = Math.imul(combinedSeed ^ character.charCodeAt(0), 16777619);
  }

  return ((combinedSeed >>> 0) ^ index ^ 0x7075_626c) >>> 0;
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

    return (state >>> 0) / 0x1_0000_0000;
  };
}

/**
 * Computes the camera-up vector from the forward and right basis vectors.
 *
 * @param camera - Camera whose forward/right basis is known.
 * @returns Normalized world-up vector perpendicular to forward and right.
 */
function computeCameraUp(camera: PublicCamera): readonly [number, number, number] {
  const upX = camera.forward[1] * camera.right[2] - camera.forward[2] * camera.right[1];
  const upY = camera.forward[2] * camera.right[0] - camera.forward[0] * camera.right[2];
  const upZ = camera.forward[0] * camera.right[1] - camera.forward[1] * camera.right[0];
  const length = Math.hypot(upX, upY, upZ) || 1;

  return [upX / length, upY / length, upZ / length];
}

/**
 * Projects one world-space billboard into screen space.
 *
 * @param position - World-space position in meters.
 * @param frame - Selected 2D sprite frame.
 * @param camera - Camera used for projection and distance scaling.
 * @param cameraUp - Normalized camera-up vector.
 * @returns Billboard with positive screen-space quad and perspective-correct scale.
 */
function buildBillboard(
  position: readonly [number, number, number],
  frame: PublicSpriteFrame,
  camera: PublicCamera,
  cameraUp: readonly [number, number, number],
): PublicBillboard {
  const relX = position[0] - camera.position[0];
  const relY = position[1] - camera.position[1];
  const relZ = position[2] - camera.position[2];

  const distance = Math.max(BILLBOARD_MIN_DISTANCE_M, Math.hypot(relX, relY, relZ));
  const scale = computeBillboardScale(distance);
  const focal = 1 / Math.tan((camera.fovDegrees * Math.PI / 180) / 2);
  const rightDot = relX * camera.right[0] + relY * camera.right[1] + relZ * camera.right[2];
  const upDot = relX * cameraUp[0] + relY * cameraUp[1] + relZ * cameraUp[2];
  const screenX = (focal * rightDot) / distance;
  const screenY = (focal * upDot) / distance;
  const halfWidth = (frame.width * scale) / 2;
  const halfHeight = (frame.height * scale) / 2;

  return {
    position,
    frameId: frame.id,
    scale,
    distance,
    quad: [screenX - halfWidth, screenY - halfHeight, screenX + halfWidth, screenY + halfHeight],
  };
}

/**
 * Returns the highest surface a billboard footprint must rest on.
 *
 * @param x - World-space x center of the billboard.
 * @param z - World-space z center of the billboard.
 * @param halfWidth - World-space half-width of the billboard footprint.
 * @param halfDepth - World-space half-depth of the billboard footprint.
 * @param props - Scenery props to test against.
 * @returns Y value of the highest overlapping prop top, or 0 for the ground plane.
 */
function resolveBillboardBaseY(
  x: number,
  z: number,
  halfWidth: number,
  halfDepth: number,
  props: readonly RightWorldProp[],
): number {
  let base = 0;

  for (const prop of props) {
    const [px, py, pz] = prop.position;
    const [pHx, pHy, pHz] = resolvePropHalfSize(prop.kind);
    const propTop = py + 2 * pHy;

    if (
      Math.abs(x - px) < halfWidth + pHx &&
      Math.abs(z - pz) < halfDepth + pHz &&
      propTop > base
    ) {
      base = propTop;
    }
  }

  return base;
}

/**
 * Computes the world-space half-width and half-height of a billboard at a given scale.
 *
 * @param frame - Sprite frame whose pixel dimensions are known.
 * @param scale - Distance-aware scale applied to the sprite.
 * @returns Half-width and half-height in meters.
 */
function billboardWorldHalfSize(
  frame: PublicSpriteFrame,
  scale: number,
): readonly [number, number] {
  const worldScale = scale * NEATRACING_BILLBOARD_WORLD_PIXEL_SCALE_M;

  return [(frame.width * worldScale) / 2, (frame.height * worldScale) / 2];
}

/**
 * Places a billboard so its base rests on the ground or on top of any scenery
 * prop its footprint overlaps. The returned position is the foot/base anchor
 * (y = surface + a small clearance), not the geometric center. The render pass
 * expands the quad upward from this anchor so the sprite remains surface-aligned
 * at any runtime camera distance.
 *
 * @param x - World-space x center.
 * @param z - World-space z center.
 * @param frame - Selected sprite frame.
 * @param camera - Camera used for projection.
 * @param cameraUp - Normalized camera-up vector.
 * @param props - Scenery props that can elevate the billboard.
 * @returns Billboard with a surface-aligned base anchor and consistent distance/scale.
 */
function placeBillboardOnSurface(
  x: number,
  z: number,
  frame: PublicSpriteFrame,
  camera: PublicCamera,
  cameraUp: readonly [number, number, number],
  props: readonly RightWorldProp[],
): PublicBillboard {
  let positionY = 0;

  for (let iteration = 0; iteration < 8; iteration += 1) {
    const candidate: readonly [number, number, number] = [x, positionY, z];
    const preview = buildBillboard(candidate, frame, camera, cameraUp);
    const [halfWidth, halfDepth] = billboardWorldHalfSize(frame, preview.scale);
    const baseY = resolveBillboardBaseY(x, z, halfWidth, halfDepth, props);
    const targetY = baseY + NEATRACING_BILLBOARD_SURFACE_EPSILON_M;

    if (Math.abs(positionY - targetY) <= 0.01) {
      return buildBillboard([x, targetY, z], frame, camera, cameraUp);
    }

    positionY = targetY;
  }

  return buildBillboard([x, positionY, z], frame, camera, cameraUp);
}

