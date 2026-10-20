/**
 * Recycled Neatenstein mechanic-crew billboard sprites for the neatRacing
 * right-world render pass (slice P1-mechanic-crew, Step 14D, AC-050).
 *
 * Loads a 2D sprite atlas derived from the examples/neatenstein mechanic-crew
 * sprites and builds screen-facing billboard quads placed near the start line
 * and pit area on the right-hand side of the track. Billboard scale respects
 * camera distance so that distant crew members shrink while nearby ones stay
 * readable.
 *
 * GREEN TARGET (04-implementing must make exactly these pass):
 * - Exports {@link loadMechanicSpriteAtlas} returning a non-empty atlas whose
 *   source is 'neatenstein-mechanic-crew'.
 * - Exports {@link selectMechanicSpriteFrame} returning a deterministic frame
 *   for a given (atlas, seed, index) tuple.
 * - Exports {@link buildMechanicBillboards} returning right-side track
 *   billboards near the start line with positive screen-space quads and
 *   distance-aware scale.
 * - Exports {@link computeMechanicBillboardScale} returning a constant 1.0 world scale
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
import {
  buildRightWorldProps,
  filterRobotsByPersonalSpace,
  getBleacherAxisAlignedHalfSize,
  getBleacherSeatTopY,
} from './right-world.builder';
import type { RightWorldProp } from './right-world.builder';
import { resolvePropHalfSize } from './right-world.prop-sizes';
import {
  DIRECTION_ORDER,
  PUBLIC_POSE_COUNT,
  PUBLIC_POSE_INDEX_STAND,
  resolveDirectionIndexFromTarget,
} from './public.sprite';
import type { TrackSlabBuildInput } from './track.slabs';

/** One recycled Neatenstein mechanic-crew sprite frame. */
export interface MechanicSpriteFrame {
  /** Stable frame identifier. */
  readonly id: string;
  /** Frame width in pixels. */
  readonly width: number;
  /** Frame height in pixels. */
  readonly height: number;
  /** Raw RGBA pixel data. */
  readonly rgba: Uint8ClampedArray;
}

/** Sprite atlas loaded from the recycled Neatenstein mechanic-crew sprites. */
export interface MechanicSpriteAtlas {
  readonly source: 'neatenstein-mechanic-crew';
  readonly frames: readonly MechanicSpriteFrame[];
}

/** Minimal camera contract used to project world billboards into screen space. */
export interface MechanicCamera {
  readonly position: readonly [number, number, number];
  readonly forward: readonly [number, number, number];
  readonly right: readonly [number, number, number];
  readonly near: number;
  readonly far: number;
  readonly fovDegrees: number;
}

/** A 2D billboard instance placed in world space and projected to screen space. */
export interface MechanicBillboard {
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
  /** Runtime role used by the pass to recompute gaze each frame. */
  readonly runtimeRole?: 'mechanic';
  /** Deterministic seed carried through from the builder. */
  readonly seed?: number;
  /** Billboard index carried through from the builder. */
  readonly index?: number;
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

/** Density budget for mechanic-crew billboards per track. */
const NEATRACING_MECHANIC_BILLBOARD_DENSITY_BUDGET = 6;

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

/** Team-color accent palette for mechanic-crew sprites (AC-062). */
const MECHANIC_TEAM_COLORS: readonly (readonly [number, number, number, number])[] = [
  [221, 34, 0, 255],
  [0, 255, 255, 255],
  [255, 255, 0, 255],
  [255, 0, 255, 255],
  [0, 255, 128, 255],
  [128, 0, 255, 255],
];

/** Number of palette colors available for mechanic robots. */
const MECHANIC_COLOR_COUNT = MECHANIC_TEAM_COLORS.length;

/**
 * Stable per-robot accent color index derived from deterministic identity.
 *
 * The same hash is used for public sprites so that both sprite systems share
 * a consistent robot-identity color seed, but each system indexes its own
 * palette size.
 */
function computeMechanicRobotColorId(seed: number, index: number): number {
  let hash = seed >>> 0;
  hash ^= (index * 0x9e3779b9) >>> 0;
  hash = Math.imul(hash ^ 0x85ebca6b, 0xc2b2ae35);
  hash ^= hash >>> 16;
  hash = Math.imul(hash, 0x27d4eb2f);
  hash ^= hash >>> 15;

  return (hash >>> 0) % MECHANIC_COLOR_COUNT;
}

/** Spline sample shape used by the generated track specification. */
type TrackSplineSample = ReturnType<typeof generateTrack>['splineSamples'][number];

/** Encoded robot sprite frame shape reused from Neatenstein. */
type EncodedFrame = readonly (readonly number[])[];

/**
 * Loads the recycled Neatenstein mechanic-crew sprite atlas.
 *
 * @returns A non-empty 2D sprite atlas whose source is 'neatenstein-mechanic-crew'.
 *
 * @example
 * ```ts
 * const atlas = loadMechanicSpriteAtlas();
 * console.log(atlas.source); // 'neatenstein-mechanic-crew'
 * console.log(atlas.frames.length > 0); // true
 * ```
 */
export function loadMechanicSpriteAtlas(): MechanicSpriteAtlas {
  return {
    source: 'neatenstein-mechanic-crew',
    frames: buildMechanicSpriteAtlasFrames(),
  };
}

/**
 * Selects a deterministic sprite frame from the mechanic-crew atlas.
 *
 * @param atlas - Mechanic-crew sprite atlas.
 * @param seed - Deterministic seed for frame selection.
 * @param index - Billboard index used to vary the selected frame.
 * @returns A frame with positive dimensions.
 *
 * @example
 * ```ts
 * const atlas = loadMechanicSpriteAtlas();
 * const frame = selectMechanicSpriteFrame(atlas, 42, 0);
 * console.log(frame.width > 0); // true
 * console.log(frame.height > 0); // true
 * ```
 */
export function selectMechanicSpriteFrame(
  atlas: MechanicSpriteAtlas,
  seed: number,
  index: number,
): MechanicSpriteFrame {
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
 * voxels, so distant sprites shrink at the same rate as geometry of the
 * same world size.
 *
 * @param distance - Distance from the camera in meters (kept for API
 *   stability; the caller is responsible for perspective projection).
 * @returns World-space scale of 1.0 for all distances.
 *
 * @example
 * ```ts
 * const nearScale = computeMechanicBillboardScale(5);
 * const farScale = computeMechanicBillboardScale(50);
 * console.log(nearScale).toBeCloseTo(1, 6); // 1:1 proportion
 * console.log(farScale).toBeCloseTo(1, 6);  // perspective handles shrink
 * ```
 */
export function computeMechanicBillboardScale(distance: number): number {
  // Distance is intentionally ignored: the billboard world size stays at 1:1
  // and the right-world render pass applies perspective projection.
  void distance;

  return BILLBOARD_WORLD_SCALE;
}

/**
 * Selects the stand frame that makes a mechanic-crew robot face the car.
 *
 * @param atlas - Mechanic-crew sprite atlas.
 * @param robotPosition - Robot world-space foot position.
 * @param carPosition - Car/camera world-space position to face.
 * @returns Mechanic frame id, e.g. `'mechanic-front-stand'`.
 */
export function selectMechanicLookAtCarFrame(
  atlas: MechanicSpriteAtlas,
  robotPosition: readonly [number, number, number],
  carPosition: readonly [number, number, number],
): string {
  void atlas;
  const directionIndex = resolveDirectionIndexFromTarget(robotPosition, carPosition);

  return `mechanic-${DIRECTION_ORDER[directionIndex]!}-stand`;
}

/**
 * Computes the runtime atlas frame index for a mechanic-crew billboard.
 *
 * The mechanic always faces the live car position and uses the stand pose.
 * No strings or objects are allocated on the draw path.
 *
 * @param billboard - Mechanic billboard with world-space position.
 * @param carPosition - Live car world position for gaze.
 * @param indexTable - Prebuilt direction×pose atlas index table.
 * @returns Atlas frame index, or -1 if the frame is missing.
 */
export function computeMechanicRuntimeFrameIndex(
  billboard: MechanicBillboard,
  carPosition: readonly [number, number, number],
  indexTable: Int16Array,
): number {
  const directionIndex = resolveDirectionIndexFromTarget(billboard.position, carPosition);
  const tableIndex = directionIndex * PUBLIC_POSE_COUNT + PUBLIC_POSE_INDEX_STAND;
  const defaultAtlasIndex = indexTable[tableIndex];

  if (defaultAtlasIndex === undefined || defaultAtlasIndex < 0) {
    return -1;
  }

  // The atlas is laid out color-major and the simple id-split table points at
  // the LAST color block. Offset back to this robot's stable identity color.
  // Each mechanic color block contains one frame per direction, so the stride
  // between color blocks is the direction count (not PUBLIC_POSE_COUNT).
  const defaultColorId = MECHANIC_COLOR_COUNT - 1;
  const colorId = computeMechanicRobotColorId(
    billboard.seed ?? 0,
    billboard.index ?? 0,
  );
  const frameBlockSize = DIRECTION_ORDER.length;
  const atlasIndex = defaultAtlasIndex + (colorId - defaultColorId) * frameBlockSize;

  return atlasIndex;
}

/**
 * Builds the mechanic-crew billboards near the start line and pit area.
 *
 * @param input - Determinism tuple for the track layout.
 * @param camera - Camera used for projection and distance scaling.
 * @param atlas - Mechanic-crew sprite atlas.
 * @returns Ordered list of 2D billboards placed on the right side near the start line.
 *
 * @example
 * ```ts
 * const atlas = loadMechanicSpriteAtlas();
 * const camera = {
 *   position: [0, 2, 20] as const,
 *   forward: [0, -0.1, -1] as const,
 *   right: [1, 0, 0] as const,
 *   near: 0.1,
 *   far: 200,
 *   fovDegrees: 60,
 * };
 * const billboards = buildMechanicBillboards(
 *   { seed: 42, layoutVersion: 1, sizeBucket: 'large' },
 *   camera,
 *   atlas,
 * );
 * console.log(billboards.length > 0); // true
 * ```
 */
export function buildMechanicBillboards(
  input: TrackSlabBuildInput,
  camera: MechanicCamera,
  atlas: MechanicSpriteAtlas,
): readonly MechanicBillboard[] {
  if (atlas.frames.length === 0) {
    return [];
  }

  const spec = generateTrack(input);
  const samples = spec.splineSamples;
  if (samples.length === 0) {
    return [];
  }

  const cameraUp = computeCameraUp(camera);
  const props = buildRightWorldProps(input);
  const billboards: MechanicBillboard[] = [];

  // Inner-ring mechanic/crew props are pit pedestals. Anchor every mechanic
  // billboard to a pedestal surface so the sprite never shares volume with the
  // rendered voxel cube (OBS-2). Pedestals are short (≤0.5m tall) so the robot
  // stands beside/behind a low block, not inside a full-height one.
  const pedestals = props.filter(
    (prop) => prop.kind === 'mechanic' || prop.kind === 'crew',
  );
  const targetCount = Math.min(
    NEATRACING_MECHANIC_BILLBOARD_DENSITY_BUDGET,
    Math.max(1, pedestals.length),
  );

  if (pedestals.length === 0) {
    // Last-resort fallback: place a single start-line mechanic on the ground if
    // the track emitted no inner-ring pedestals.
    const startLine = buildStartLineMechanic(
      samples[0]!,
      samples,
      atlas,
      input.seed,
      camera,
      cameraUp,
      props,
    );
    billboards.push({
      ...startLine,
      runtimeRole: 'mechanic',
      seed: input.seed,
      index: 0,
    });

    return billboards;
  }

  for (let index = 0; index < targetCount; index += 1) {
    const pedestal = pedestals[index % pedestals.length]!;
    const [px, py, pz] = pedestal.position;
    const [, pHy] = resolvePropHalfSize(pedestal.kind);
    const pedestalTop = py + 2 * pHy;
    // Deterministic lateral offset on the pedestal top so multiple sprites on
    // the same pedestal do not z-fight or overlap each other.
    const xOffset = (index % 2 === 0 ? -1 : 1) * 0.2;
    const zOffset = (Math.floor(index / 2) % 2 === 0 ? -1 : 1) * 0.2;
    const robotX = px + xOffset;
    const robotZ = pz + zOffset;
    const robotPosition: readonly [number, number, number] = [robotX, pedestalTop, robotZ];
    const frameId = selectMechanicLookAtCarFrame(atlas, robotPosition, camera.position);
    const selectedFrame = atlas.frames.find((candidate) => candidate.id === frameId) ?? atlas.frames[0]!;

    const placed = placeBillboardOnSurface(
      robotX,
      robotZ,
      pedestalTop + NEATRACING_BILLBOARD_SURFACE_EPSILON_M,
      selectedFrame,
      camera,
      cameraUp,
      props,
    );
    billboards.push({
      ...placed,
      runtimeRole: 'mechanic',
      seed: input.seed,
      index,
    });
  }

  // AC-065: keep only the mechanic crew robots that fit the 1 m XZ personal-space
  // band. Filtering is deterministic and happens at build time, leaving the draw
  // path allocation-free.
  return filterRobotsByPersonalSpace(billboards);
}

/**
 * Builds a stable frame list from the Neatenstein robot sprite atlas.
 *
 * Only the standing pose is reused for the pit crew so the sprites read as
 * stationary mechanics rather than animated walkers. Frames are emitted in
 * COLOR-major order: for each palette color, every direction frame is recolored
 * and appended. The frame id stays in the base `mechanic-direction-stand` form
 * so existing id-split lookup tables keep working. The last color block is
 * therefore the one a simple direction table maps to; runtime helpers offset
 * back to each robot's stable identity color (fix-packet-P1-bleacher-shape-
 * iteration-3, OBS-2).
 *
 * @returns Decoded 2D RGBA frames recycled from `ROBOT_SPRITE_FRAMES`.
 */
function buildMechanicSpriteAtlasFrames(): readonly MechanicSpriteFrame[] {
  const frames: MechanicSpriteFrame[] = [];

  for (const accent of MECHANIC_TEAM_COLORS) {
    for (const direction of DIRECTION_ORDER) {
      const poseMap = ROBOT_SPRITE_FRAMES[direction];
      if (!poseMap) {
        continue;
      }
      const standFrame = poseMap.stand;
      if (standFrame === undefined) {
        continue;
      }
      const id = `mechanic-${direction}-stand`;
      frames.push(
        recolorFrameRgba(
          decodeMechanicFrame(standFrame as EncodedFrame, id),
          accent,
        ),
      );
    }
  }

  return frames;
}

/**
 * Decodes one encoded Neatenstein robot frame into a 2D RGBA sprite.
 *
 * @param frame - Encoded palette-index rows.
 * @param id - Stable frame identifier.
 * @returns Decoded RGBA sprite frame with positive dimensions.
 */
function decodeMechanicFrame(frame: EncodedFrame, id: string): MechanicSpriteFrame {
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
 * Recolors the saturated red/orange accent pixels of a decoded sprite to a
 * team accent. This is applied once at atlas build time, so the render path
 * stays allocation-free.
 *
 * @param frame - Decoded sprite frame.
 * @param accent - RGBA accent color to apply.
 * @returns The same frame reference, with accent pixels recolored.
 */
function recolorFrameRgba(
  frame: MechanicSpriteFrame,
  accent: readonly [number, number, number, number],
): MechanicSpriteFrame {
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
function computeCameraUp(
  camera: MechanicCamera,
): readonly [number, number, number] {
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
 * @returns Billboard with positive screen-space quad and distance-aware scale.
 */
function buildBillboard(
  position: readonly [number, number, number],
  frame: MechanicSpriteFrame,
  camera: MechanicCamera,
  cameraUp: readonly [number, number, number],
): MechanicBillboard {
  const relX = position[0] - camera.position[0];
  const relY = position[1] - camera.position[1];
  const relZ = position[2] - camera.position[2];

  const distance = Math.max(BILLBOARD_MIN_DISTANCE_M, Math.hypot(relX, relY, relZ));
  const scale = computeMechanicBillboardScale(distance);
  const focal = 1 / Math.tan((camera.fovDegrees * Math.PI) / 180 / 2);
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

    let pHx: number;
    let pHz: number;
    let propTop: number;
    if (prop.kind === 'bleacher' && prop.bleacherGeometry) {
      const [bleacherHalfWidth, bleacherHalfDepth] = getBleacherAxisAlignedHalfSize(prop);
      pHx = bleacherHalfWidth;
      pHz = bleacherHalfDepth;
      propTop = getBleacherSeatTopY(prop, x, z);
    } else {
      const [phx, phy, phz] = resolvePropHalfSize(prop.kind);
      pHx = phx;
      pHz = phz;
      propTop = py + 2 * phy;
    }

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
  frame: MechanicSpriteFrame,
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
 * @param startY - Initial vertical guess for the foot anchor (e.g. a pedestal top).
 * @param frame - Selected sprite frame.
 * @param camera - Camera used for projection.
 * @param cameraUp - Normalized camera-up vector.
 * @param props - Scenery props that can elevate the billboard.
 * @returns Billboard with a surface-aligned base anchor and consistent distance/scale.
 */
function placeBillboardOnSurface(
  x: number,
  z: number,
  startY: number,
  frame: MechanicSpriteFrame,
  camera: MechanicCamera,
  cameraUp: readonly [number, number, number],
  props: readonly RightWorldProp[],
): MechanicBillboard {
  let positionY = startY;

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

/**
 * Builds a mechanic billboard at the start line on the right-hand side.
 *
 * @param startSample - First spline sample at the start line.
 * @param samples - Full spline sample list.
 * @param atlas - Mechanic-crew sprite atlas.
 * @param seed - Track seed used for frame selection.
 * @param camera - Camera used for projection and distance scaling.
 * @param cameraUp - Normalized camera-up vector.
 * @param props - Scenery props that can elevate the billboard.
 * @returns Start-line mechanic billboard.
 */
function buildStartLineMechanic(
  startSample: TrackSplineSample,
  samples: readonly TrackSplineSample[],
  atlas: MechanicSpriteAtlas,
  seed: number,
  camera: MechanicCamera,
  cameraUp: readonly [number, number, number],
  props: readonly RightWorldProp[],
): MechanicBillboard {
  const frame = resolveSplineSampleFrame(samples, startSample.globalIndex);
  const offset = startSample.width / 2 + 2;
  const generatorX = startSample.x - frame.normalX * offset;
  const generatorY = startSample.y - frame.normalY * offset;
  const robotPosition: readonly [number, number, number] = [generatorX, 0, -generatorY];
  const frameId = selectMechanicLookAtCarFrame(atlas, robotPosition, camera.position);
  const selectedFrame = atlas.frames.find((candidate) => candidate.id === frameId) ?? atlas.frames[0]!;

  return placeBillboardOnSurface(
    generatorX,
    -generatorY,
    0,
    selectedFrame,
    camera,
    cameraUp,
    props,
  );
}

