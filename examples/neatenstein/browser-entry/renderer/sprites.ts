/**
 * Enemy wireframe sprite rendering for the Neatenstein neon raycaster.
 *
 * Each sprite is projected from world space into screen space using the inverse
 * camera matrix, clipped against the per-column z-buffer, and drawn as a
 * vertical neon bar for every visible screen column.
 *
 * This module is intentionally CPU-friendly:
 *
 * - projection is scalar and deterministic
 * - clipping is delegated to the z-buffer helper
 * - color parsing is cached
 * - framebuffer writes are direct RGBA byte writes
 * - the full framebuffer is flushed once per rendered sprite
 *
 * The sprite renderer uses the same camera matrix convention as classic
 * Wolfenstein-style raycasters: `dir` is the forward vector and `plane` is the
 * camera plane vector.
 *
 * @module
 */

import { clampInt, NEATENSTEIN_RENDER_DISTANCE_CAP } from './framebuffer';
import {
  clipNeatensteinSpriteSpan,
  type NeatensteinSpriteClip,
} from './zbuffer';
import { NEATENSTEIN_FLOOR_FOV_RADIANS } from './floor';
import {
  ROBOT_SPRITE_FRAMES,
  ROBOT_SPRITE_PALETTE,
  ROBOT_SPRITE_SCALE,
} from '../../robot-sprite-data.js';
import { type VoxelSnapshot } from '../../../neatenstein/scripts/snapshot-renderer';
import { type EnemyAnimationState } from '../../../neatenstein/scripts/enemy-animator';

/**
 * Number of RGBA channels per framebuffer pixel.
 */
const NEATENSTEIN_RGBA_CHANNELS = 4;

/**
 * Minimum absolute determinant accepted for the inverse camera matrix.
 *
 * A determinant close to zero means the camera direction and plane are
 * degenerate, which would make projection unstable.
 */
const NEATENSTEIN_CAMERA_DETERMINANT_EPSILON = 1e-9;

/**
 * World-space size of a sprite in grid cells.
 *
 * The projected screen size is:
 *
 * ```ts
 * focalLength / perpDist * NEATENSTEIN_SPRITE_WORLD_SIZE
 * ```
 *
 * where `focalLength` is derived from {@link NEATENSTEIN_FLOOR_FOV_RADIANS}.
 * Keeping this value in world units makes enemies scale consistently as they
 * move toward or away from the camera.
 */
export const NEATENSTEIN_SPRITE_WORLD_SIZE = 1.0;

/**
 * Minimum perpendicular distance at which a sprite is drawn.
 *
 * Sprites closer than this are considered too close to the camera plane and are
 * culled to avoid division-by-zero and unstable projection.
 */
export const NEATENSTEIN_SPRITE_NEAR_CLIP = 0.1;

/**
 * Fraction of projected sprite scale used as horizontal sprite thickness.
 *
 * A value of 1.0 makes the projected sprite width match its height, which
 * matches the square aspect of the 192×192 reference robot silhouettes.
 */
export const NEATENSTEIN_SPRITE_THICKNESS_RATIO = 1.0;

/**
 * Minimal canvas-like context consumed by the CPU sprite renderer.
 *
 * Only `putImageData` is required, mirroring the narrow context contract used
 * by the CPU wall renderer.
 */
export interface NeatensteinSpriteRenderContext {
  /**
   * Flush an ImageData-like payload to the canvas.
   *
   * @param imageData - Object with `data`, `width`, and `height`.
   * @param dx - Destination X coordinate.
   * @param dy - Destination Y coordinate.
   */
  putImageData(
    imageData: { data: Uint8ClampedArray; width: number; height: number },
    dx: number,
    dy: number,
  ): void;
}

/**
 * Camera transform consumed by the sprite projector.
 */
export interface NeatensteinCamera {
  /** Camera X position in world cells. */
  posX: number;
  /** Camera Y position in world cells. */
  posY: number;
  /** View direction X component. */
  dirX: number;
  /** View direction Y component. */
  dirY: number;
  /** Camera plane X component. */
  planeX: number;
  /** Camera plane Y component. */
  planeY: number;
}

/**
 * World-space sprite to project.
 */
export interface NeatensteinSprite {
  /** Sprite X position in world cells. */
  worldX: number;
  /** Sprite Y position in world cells. */
  worldY: number;
  /** Sprite yaw in radians. 0 points toward +X; a {@link resolveNeatensteinEnemyFrame} helper uses this together with the camera position to select a pre-rendered voxel frame. */
  facing?: number;
  /** Current animation state used to pick a frame from the voxel atlas. */
  animationState?: EnemyAnimationState;
  /** Animation frame index. Currently only frame 0 is pre-rendered in the runtime atlas. */
  frameIndex?: number;
  /** Optional enemy type index for hue selection by higher-level callers. */
  type?: number;
  /** Sim tick counter driving the walk cycle (stand → walk1 → stand → walk2). When provided, overrides position-based walk alternation. */
  walkTick?: number;
  /** Remaining shoot-blink ticks; when > 0 the renderer composites the shoot upper body over the walk lower body. */
  shootBlinkTicks?: number;
  /** Optional team color [r, g, b] to swap palette indices 5/6/7 at runtime, preserving alpha. */
  teamColor?: readonly [number, number, number];
}

/**
 * Screen-space projection result for a single sprite.
 */
export interface NeatensteinSpriteProjection {
  /** Horizontal screen coordinate of the sprite center. */
  screenX: number;
  /** Projected screen size in pixels. */
  scale: number;
  /** Perpendicular distance from the camera plane to the sprite. */
  perpDist: number;
  /** Left edge of the projected sprite in screen pixels. */
  left: number;
  /** Right edge of the projected sprite in screen pixels. */
  right: number;
  /** `false` when the sprite is behind the camera or otherwise invisible. */
  visible: boolean;
}

/**
 * Resolved framebuffer dimensions.
 */
interface ResolvedSpriteFramebufferSize {
  /** Framebuffer width in pixels. */
  width: number;
  /** Framebuffer height in pixels. */
  height: number;
}

/**
 * Return whether a number is a positive finite integer dimension.
 *
 * @param value - Candidate dimension.
 * @returns Whether the value is usable as a framebuffer/canvas dimension.
 */
function isPositiveIntegerDimension(value: number): boolean {
  return Number.isInteger(value) && value > 0;
}

/**
 * Return whether a number is finite and greater than zero.
 *
 * @param value - Candidate scalar.
 * @returns Whether the value is positive and finite.
 */
function isPositiveFinite(value: number): boolean {
  return Number.isFinite(value) && value > 0;
}

/**
 * Return a reusable invisible projection result.
 *
 * @param perpDist - Perpendicular distance associated with the rejected sprite.
 * @returns Invisible sprite projection.
 */
function createInvisibleSpriteProjection(
  perpDist = Number.POSITIVE_INFINITY,
): NeatensteinSpriteProjection {
  return {
    screenX: -1,
    scale: 0,
    perpDist,
    left: -1,
    right: -1,
    visible: false,
  };
}

/**
 * Resolve framebuffer dimensions from explicit dimensions or available buffers.
 *
 * A flat RGBA buffer length alone cannot uniquely identify rectangular
 * dimensions. Therefore:
 *
 * - explicit width/height are preferred when provided
 * - the z-buffer length is used as width when available
 * - legacy square inference is used only as a final fallback
 *
 * @param framebuffer - Flat RGBA framebuffer.
 * @param preferredWidth - Optional known framebuffer width.
 * @param preferredHeight - Optional known framebuffer height.
 * @returns Resolved framebuffer dimensions.
 */
function resolveFramebufferSize(
  framebuffer: Uint8ClampedArray,
  preferredWidth?: number,
  preferredHeight?: number,
): ResolvedSpriteFramebufferSize {
  const pixelCount = Math.floor(framebuffer.length / NEATENSTEIN_RGBA_CHANNELS);

  // Best case: caller supplied exact dimensions.
  if (
    preferredWidth !== undefined &&
    preferredHeight !== undefined &&
    isPositiveIntegerDimension(preferredWidth) &&
    isPositiveIntegerDimension(preferredHeight) &&
    preferredWidth * preferredHeight <= pixelCount
  ) {
    return {
      width: preferredWidth,
      height: preferredHeight,
    };
  }

  // Common sprite-render path: z-buffer length gives the canvas width.
  if (
    preferredWidth !== undefined &&
    isPositiveIntegerDimension(preferredWidth) &&
    pixelCount >= preferredWidth
  ) {
    return {
      width: preferredWidth,
      height: Math.floor(pixelCount / preferredWidth),
    };
  }

  // Less common case: height is known but width is not.
  if (
    preferredHeight !== undefined &&
    isPositiveIntegerDimension(preferredHeight) &&
    pixelCount >= preferredHeight
  ) {
    return {
      width: Math.floor(pixelCount / preferredHeight),
      height: preferredHeight,
    };
  }

  // Legacy fallback for older square-framebuffer callers.
  const side = Math.floor(Math.sqrt(pixelCount));
  return {
    width: side,
    height: side,
  };
}

/**
 * Return whether a framebuffer size can be drawn into safely.
 *
 * @param size - Resolved framebuffer dimensions.
 * @returns Whether width and height are drawable.
 */
function isDrawableFramebufferSize(
  size: ResolvedSpriteFramebufferSize,
): boolean {
  return (
    isPositiveIntegerDimension(size.width) &&
    isPositiveIntegerDimension(size.height)
  );
}

/**
 * Return whether a projection has usable finite screen-space values.
 *
 * @param projection - Projection to validate.
 * @returns Whether the projection can be used for clipping/rendering.
 */
function isDrawableSpriteProjection(
  projection: NeatensteinSpriteProjection,
): boolean {
  return (
    projection.visible &&
    Number.isFinite(projection.screenX) &&
    isPositiveFinite(projection.scale) &&
    isPositiveFinite(projection.perpDist) &&
    Number.isFinite(projection.left) &&
    Number.isFinite(projection.right) &&
    projection.right >= projection.left
  );
}

/**
 * Discrete yaw directions stored in the encoded robot sprite atlas.
 */
const NEATENSTEIN_VOXEL_ATLAS_YAW_STEPS = 8;

/**
 * Full rotation in radians.
 */
const NEATENSTEIN_FULL_ROTATION_RADIANS = Math.PI * 2;

/**
 * Angular step between adjacent yaw atlas entries, in radians.
 */
const NEATENSTEIN_VOXEL_ATLAS_YAW_STEP_RADIANS =
  NEATENSTEIN_FULL_ROTATION_RADIANS / NEATENSTEIN_VOXEL_ATLAS_YAW_STEPS;

/**
 * Encoded robot sprite frame: rows of palette indices into
 * {@link ROBOT_SPRITE_PALETTE}.
 */
export type EncodedRobotSpriteFrame = readonly (readonly number[])[];

/**
 * Renderable sprite source: a pre-rendered voxel snapshot, an encoded robot
 * frame, or a legacy color string (ignored).
 */
export type NeatensteinSpriteSource =
  VoxelSnapshot | EncodedRobotSpriteFrame | string;

/** Distance traveled in world cells between walk-cycle pose swaps. */
const NEATENSTEIN_WALK_CYCLE_HALF_STEP_CELLS = 0.5;

/** Row index where the upper body (shoot) and lower body (walk) split. Rows 0–34 are upper body; rows 35–47 are lower body. */
const NEATENSTEIN_SPRITE_UPPER_BODY_SPLIT_ROW = 35;

/** Walk cycle pose sequence indexed by `floor(walkTick / 4) % 4`: stand → walk1 → stand → walk2 (4x slowed). */
const NEATENSTEIN_WALK_CYCLE_POSES = [
  'stand',
  'walk1',
  'stand',
  'walk2',
] as const;

/**
 * Map canonical animation states to encoded pose names.
 *
 * The runtime only carries one idle pose (`stand`) plus a two-frame walk cycle
 * and a single shoot pose. Unknown or unmapped states fall back to `stand`.
 *
 * The `move` state is resolved dynamically by
 * {@link resolveNeatensteinEnemyFrame} so walking enemies alternate between
 * `walk1` and `walk2` as they travel.
 */
const NEATENSTEIN_ANIMATION_TO_POSE: Record<
  EnemyAnimationState,
  'stand' | 'walk1' | 'walk2' | 'shoot'
> = {
  idle: 'stand',
  move: 'walk1',
  fire: 'shoot',
  death: 'stand',
  damage: 'stand',
};

/**
 * Camera-relative yaw indices mapped to encoded direction names.
 *
 * Index 0 means the sprite faces the camera; index 4 means it faces away.
 */
const NEATENSTEIN_ENCODED_DIRECTIONS = [
  'front',
  'frontRight',
  'right',
  'backRight',
  'back',
  'backLeft',
  'left',
  'frontLeft',
] as const;

/**
 * Decode an encoded robot sprite frame into a pre-rendered RGBA snapshot.
 *
 * Each palette index is mapped through {@link ROBOT_SPRITE_PALETTE}, preserving
 * semitransparent muzzle-blast colors (indices 7 and 8). The decoded frame is
 * scaled up by {@link ROBOT_SPRITE_SCALE} using nearest-neighbor sampling so
 * the renderer can sample it directly.
 *
 * @param frame - Encoded rows of palette indices.
 * @returns Decoded RGBA {@link VoxelSnapshot}.
 */
function decodeRobotSpriteFrame(
  frame: EncodedRobotSpriteFrame,
  palette: readonly (readonly [
    number,
    number,
    number,
    number,
  ])[] = ROBOT_SPRITE_PALETTE,
): VoxelSnapshot {
  const logicalHeight = frame.length;
  const logicalWidth = (frame[0] as number[]).length;
  const width = logicalWidth * ROBOT_SPRITE_SCALE;
  const height = logicalHeight * ROBOT_SPRITE_SCALE;
  const data = new Uint8ClampedArray(
    width * height * NEATENSTEIN_RGBA_CHANNELS,
  );

  for (let y = 0; y < height; y += 1) {
    const logicalY = Math.floor(y / ROBOT_SPRITE_SCALE);
    const row = frame[logicalY] as number[];
    for (let x = 0; x < width; x += 1) {
      const logicalX = Math.floor(x / ROBOT_SPRITE_SCALE);
      const color = palette[row[logicalX]] as [number, number, number, number];
      const offset = (y * width + x) * NEATENSTEIN_RGBA_CHANNELS;
      data[offset] = color[0];
      data[offset + 1] = color[1];
      data[offset + 2] = color[2];
      data[offset + 3] = color[3];
    }
  }

  return { width, height, data };
}

/**
 * Lazily decoded frame cache.
 *
 * Encoded frames are immutable, so reference identity is a stable cache key.
 */
const decodedRobotSpriteCache = new Map<
  EncodedRobotSpriteFrame,
  VoxelSnapshot
>();

/**
 * Return a decoded RGBA snapshot for an encoded robot frame, caching the
 * result so repeated renders of the same direction/pose are allocation-free.
 *
 * @param frame - Encoded robot sprite frame.
 * @returns Decoded RGBA {@link VoxelSnapshot}.
 */
function resolveDecodedRobotSpriteFrame(
  frame: EncodedRobotSpriteFrame,
): VoxelSnapshot {
  const cached = decodedRobotSpriteCache.get(frame);
  if (cached !== undefined) {
    return cached;
  }
  const decoded = decodeRobotSpriteFrame(frame);
  decodedRobotSpriteCache.set(frame, decoded);
  return decoded;
}

/**
 * Build a modified palette with a team color applied to indices 5/6/7.
 *
 * The RGB channels of palette entries 5, 6, and 7 are replaced with the
 * team color while their original alpha values are preserved. All other
 * palette entries remain unchanged. When no team color is specified the
 * default {@link ROBOT_SPRITE_PALETTE} is used.
 *
 * @param teamColor - [r, g, b] team color to apply.
 * @returns Modified palette array.
 */
function buildTeamColorPalette(
  teamColor: readonly [number, number, number],
): readonly (readonly [number, number, number, number])[] {
  return ROBOT_SPRITE_PALETTE.map((color, i) => {
    if (i === 5 || i === 6 || i === 7) {
      return [teamColor[0], teamColor[1], teamColor[2], color[3]] as const;
    }
    return color;
  });
}

/**
 * Lazily decoded team-color frame cache, keyed by encoded frame reference
 * then by color tuple string.
 */
const teamColorDecodedCache = new Map<
  EncodedRobotSpriteFrame,
  Map<string, VoxelSnapshot>
>();

/**
 * Return a decoded RGBA snapshot for an encoded robot frame with a team
 * color applied to palette indices 5/6/7, caching the result.
 *
 * @param frame - Encoded robot sprite frame.
 * @param teamColor - [r, g, b] team color.
 * @returns Decoded RGBA {@link VoxelSnapshot} with team color applied.
 */
function resolveDecodedRobotSpriteFrameWithTeamColor(
  frame: EncodedRobotSpriteFrame,
  teamColor: readonly [number, number, number],
): VoxelSnapshot {
  let colorMap = teamColorDecodedCache.get(frame);
  if (colorMap === undefined) {
    colorMap = new Map();
    teamColorDecodedCache.set(frame, colorMap);
  }
  const colorKey = `${teamColor[0]},${teamColor[1]},${teamColor[2]}`;
  const cached = colorMap.get(colorKey);
  if (cached !== undefined) {
    return cached;
  }
  const modifiedPalette = buildTeamColorPalette(teamColor);
  const decoded = decodeRobotSpriteFrame(frame, modifiedPalette);
  colorMap.set(colorKey, decoded);
  return decoded;
}

/**
 * Cache for composite shoot+walk frames keyed by direction index and walk pose.
 *
 * The composite frame takes the upper body (rows 0 to split row − 1) from
 * the shoot pose and the lower body (split row and below) from the walk
 * pose, allowing the enemy to shoot while walking.
 */
const compositeShootWalkCache = new Map<string, EncodedRobotSpriteFrame>();

/**
 * Build a composite encoded frame: upper body from the shoot pose and
 * lower body from the walk pose.
 *
 * @param directionIndex - Yaw atlas direction index 0–7.
 * @param walkPoseName - Lower-body walk pose ('stand', 'walk1', or 'walk2').
 * @returns Composite encoded frame.
 */
function resolveCompositeShootWalkFrame(
  directionIndex: number,
  walkPoseName: 'stand' | 'walk1' | 'walk2',
): EncodedRobotSpriteFrame {
  const key = `${directionIndex}:${walkPoseName}`;
  const cached = compositeShootWalkCache.get(key);
  if (cached !== undefined) {
    return cached;
  }

  const direction = NEATENSTEIN_ENCODED_DIRECTIONS[directionIndex];
  const directionFrames = ROBOT_SPRITE_FRAMES[direction];
  const shootFrame = directionFrames.shoot as EncodedRobotSpriteFrame;
  const walkFrame = directionFrames[walkPoseName] as EncodedRobotSpriteFrame;

  const upperRows = shootFrame.slice(
    0,
    NEATENSTEIN_SPRITE_UPPER_BODY_SPLIT_ROW,
  );
  const lowerRows = walkFrame.slice(NEATENSTEIN_SPRITE_UPPER_BODY_SPLIT_ROW);
  const composite = [...upperRows, ...lowerRows] as EncodedRobotSpriteFrame;

  compositeShootWalkCache.set(key, composite);
  return composite;
}

/**
 * Determine whether a source value is an encoded robot sprite frame.
 *
 * @param source - Candidate renderable source.
 * @returns Whether the candidate is an encoded frame.
 */
function isEncodedRobotSpriteFrame(
  source: NeatensteinSpriteSource,
): source is EncodedRobotSpriteFrame {
  if (!Array.isArray(source)) {
    return false;
  }
  return source.length > 0 && Array.isArray(source[0]);
}

/**
 * Convert a relative yaw angle to the nearest atlas yaw index.
 *
 * @param relativeYaw - Camera-relative yaw in radians, already normalized to
 *   the sprite's facing direction.
 * @returns Integer index in `[0, NEATENSTEIN_VOXEL_ATLAS_YAW_STEPS - 1]`.
 */
function yawIndexFromRelativeYaw(relativeYaw: number): number {
  const normalized =
    ((relativeYaw % NEATENSTEIN_FULL_ROTATION_RADIANS) +
      NEATENSTEIN_FULL_ROTATION_RADIANS) %
    NEATENSTEIN_FULL_ROTATION_RADIANS;

  const rawIndex = Math.round(
    normalized / NEATENSTEIN_VOXEL_ATLAS_YAW_STEP_RADIANS,
  );
  return rawIndex % NEATENSTEIN_VOXEL_ATLAS_YAW_STEPS;
}

/**
 * Resolve the encoded robot sprite frame for an enemy sprite facing a camera.
 *
 * Returns `null` when the sprite does not carry the animation or facing data
 * required by the runtime atlas.
 *
 * @param sprite - Enemy sprite with optional facing and animation fields.
 * @param camera - Camera whose position determines the relative yaw.
 * @returns The matching {@link EncodedRobotSpriteFrame}, or `null` if not resolvable.
 */
export function resolveNeatensteinEnemyFrame(
  sprite: NeatensteinSprite,
  camera: NeatensteinCamera,
): EncodedRobotSpriteFrame | null {
  if (
    sprite.animationState === undefined ||
    sprite.facing === undefined ||
    !Number.isFinite(sprite.facing)
  ) {
    return null;
  }

  if (
    !Number.isFinite(sprite.worldX) ||
    !Number.isFinite(sprite.worldY) ||
    !Number.isFinite(camera.posX) ||
    !Number.isFinite(camera.posY)
  ) {
    return null;
  }

  const cameraYaw = Math.atan2(camera.dirY, camera.dirX);
  const relativeYaw = cameraYaw + Math.PI - sprite.facing;
  const yawIndex = yawIndexFromRelativeYaw(relativeYaw);

  const poseName = NEATENSTEIN_ANIMATION_TO_POSE[sprite.animationState];
  if (poseName === undefined) {
    return null;
  }

  // Determine the walk pose for the lower body.
  // When walkTick is provided, use the sim-tick-based cycle
  // (stand → walk1 → stand → walk2). Otherwise fall back to the
  // position-based alternation for backward compatibility.
  let walkPoseName: 'stand' | 'walk1' | 'walk2';
  if (sprite.walkTick !== undefined) {
    walkPoseName =
      NEATENSTEIN_WALK_CYCLE_POSES[Math.floor(sprite.walkTick / 4) % 4];
  } else if (sprite.animationState === 'move') {
    const walkPhase =
      Math.floor(
        (sprite.worldX + sprite.worldY) /
          NEATENSTEIN_WALK_CYCLE_HALF_STEP_CELLS,
      ) % 2;
    walkPoseName = walkPhase === 0 ? 'walk1' : 'walk2';
  } else {
    walkPoseName = poseName === 'shoot' ? 'stand' : poseName;
  }

  // Determine if the shoot blink is active.
  // When shootBlinkTicks is provided, the blink is active when > 0 and the
  // upper body shows the shoot frame; when it expires the upper body reverts
  // to the walk frame even if the enemy is still firing (AC-10f-005).
  // When shootBlinkTicks is not provided, fall back to showing the shoot
  // composite for the 'fire' animation state (backward compatibility).
  const isShootBlinkActive =
    sprite.shootBlinkTicks !== undefined
      ? sprite.shootBlinkTicks > 0
      : sprite.animationState === 'fire';

  // When the shoot blink is active, composite the upper body from the
  // shoot frame and the lower body from the walk pose (AC-10f-002).
  if (isShootBlinkActive) {
    return resolveCompositeShootWalkFrame(yawIndex, walkPoseName);
  }

  return ROBOT_SPRITE_FRAMES[NEATENSTEIN_ENCODED_DIRECTIONS[yawIndex]][
    walkPoseName
  ] as EncodedRobotSpriteFrame;
}

/**
 * Resolve the decoded sprite for an enemy, applying team color if specified.
 *
 * This is a convenience function that combines frame resolution with
 * team-color palette decoding. When no team color is specified, the
 * standard palette is used and the result is cached like the normal decode
 * path.
 *
 * @param sprite - Enemy sprite with optional walk tick, shoot blink, and team color.
 * @param camera - Camera whose position determines the relative yaw.
 * @returns Decoded RGBA {@link VoxelSnapshot}, or `null` if not resolvable.
 */
export function resolveNeatensteinEnemySprite(
  sprite: NeatensteinSprite,
  camera: NeatensteinCamera,
): VoxelSnapshot | null {
  const frame = resolveNeatensteinEnemyFrame(sprite, camera);
  if (frame === null) {
    return null;
  }
  if (sprite.teamColor !== undefined) {
    return resolveDecodedRobotSpriteFrameWithTeamColor(frame, sprite.teamColor);
  }
  return resolveDecodedRobotSpriteFrame(frame);
}

/**
 * Project a world-space sprite into screen coordinates.
 *
 * The projection uses the inverse camera matrix, matching the classic
 * raycasting sprite transform. Sprites behind the camera plane, too close to
 * the camera, or produced by an invalid camera matrix are marked invisible.
 *
 * @param sprite - World-space sprite position.
 * @param camera - Current camera transform.
 * @param canvasWidth - Canvas width in backing-store pixels.
 * @param canvasHeight - Canvas height in backing-store pixels.
 * @returns The screen-space projection, including the span `[left, right]`.
 *
 * @example
 * ```ts
 * const projection = projectNeatensteinSprite(
 *   { worldX: 5, worldY: 5, type: 0 },
 *   { posX: 1, posY: 1, dirX: 1, dirY: 0, planeX: 0, planeY: 0.66 },
 *   640,
 *   480,
 * );
 * ```
 */
export function projectNeatensteinSprite(
  sprite: NeatensteinSprite,
  camera: NeatensteinCamera,
  canvasWidth: number,
  canvasHeight: number,
): NeatensteinSpriteProjection {
  // Invalid render dimensions cannot produce a meaningful projection.
  if (!isPositiveFinite(canvasWidth) || !isPositiveFinite(canvasHeight)) {
    return createInvisibleSpriteProjection();
  }

  // Reject malformed camera/sprite values before they can produce NaN paths.
  if (
    !Number.isFinite(sprite.worldX) ||
    !Number.isFinite(sprite.worldY) ||
    !Number.isFinite(camera.posX) ||
    !Number.isFinite(camera.posY) ||
    !Number.isFinite(camera.dirX) ||
    !Number.isFinite(camera.dirY) ||
    !Number.isFinite(camera.planeX) ||
    !Number.isFinite(camera.planeY)
  ) {
    return createInvisibleSpriteProjection();
  }

  // Translate sprite into camera-relative world space.
  const relativeX = sprite.worldX - camera.posX;
  const relativeY = sprite.worldY - camera.posY;

  // Invert the 2x2 camera matrix. A near-zero determinant means the camera
  // direction and plane are degenerate, so projection would be unstable.
  const determinant = camera.planeX * camera.dirY - camera.dirX * camera.planeY;
  if (
    !Number.isFinite(determinant) ||
    Math.abs(determinant) < NEATENSTEIN_CAMERA_DETERMINANT_EPSILON
  ) {
    return createInvisibleSpriteProjection();
  }

  const invDet = 1 / determinant;

  const transformX =
    invDet * (camera.dirY * relativeX - camera.dirX * relativeY);
  const transformY =
    invDet * (-camera.planeY * relativeX + camera.planeX * relativeY);

  // transformY is the perpendicular camera-space depth.
  const perpDist = transformY;
  if (!isPositiveFinite(perpDist) || perpDist <= NEATENSTEIN_SPRITE_NEAR_CLIP) {
    return createInvisibleSpriteProjection(perpDist);
  }

  // Cull sprites beyond the shared render-distance cap.
  if (perpDist >= NEATENSTEIN_RENDER_DISTANCE_CAP) {
    return createInvisibleSpriteProjection(perpDist);
  }

  const focalLength =
    canvasHeight / 2 / Math.tan(NEATENSTEIN_FLOOR_FOV_RADIANS / 2);
  const screenX = (canvasWidth / 2) * (1 + transformX / transformY);
  const scale =
    Math.abs(focalLength / transformY) * NEATENSTEIN_SPRITE_WORLD_SIZE;
  const halfThickness = (scale * NEATENSTEIN_SPRITE_THICKNESS_RATIO) / 2;
  const left = screenX - halfThickness;
  const right = screenX + halfThickness;

  return {
    screenX,
    scale,
    perpDist,
    left,
    right,
    visible: true,
  };
}

/**
 * Build a {@link NeatensteinSpriteClip} for a projected sprite.
 *
 * This helper combines projection and z-buffer clipping into one call so the
 * renderer can iterate only the columns that are actually visible.
 *
 * @param sprite - World-space sprite.
 * @param camera - Current camera transform.
 * @param canvasWidth - Canvas width in backing-store pixels.
 * @param canvasHeight - Canvas height in backing-store pixels.
 * @param zBuffer - Per-column wall-depth buffer.
 * @returns The projection plus clipped span and visible column indices.
 */
export function clipNeatensteinSprite(
  sprite: NeatensteinSprite,
  camera: NeatensteinCamera,
  canvasWidth: number,
  canvasHeight: number,
  zBuffer: Readonly<Float32Array>,
): NeatensteinSpriteProjection & NeatensteinSpriteClip {
  const projection = projectNeatensteinSprite(
    sprite,
    camera,
    canvasWidth,
    canvasHeight,
  );

  if (!isDrawableSpriteProjection(projection) || zBuffer.length === 0) {
    return {
      ...projection,
      left: 0,
      right: -1,
      visibleColumns: [],
    };
  }

  const clip = clipNeatensteinSpriteSpan(
    zBuffer,
    projection.left,
    projection.right,
    projection.perpDist,
  );

  return { ...projection, ...clip };
}

/**
 * Return precomputed visible columns from a projection if present.
 *
 * This lets callers pass the result of {@link clipNeatensteinSprite} directly
 * into {@link renderNeatensteinSprite} without recomputing clipping.
 *
 * @param projection - Projection that may also contain clip data.
 * @returns Visible columns if already available, otherwise `null`.
 */
function getPrecomputedVisibleColumns(
  projection: NeatensteinSpriteProjection,
): readonly number[] | null {
  const maybeClip = projection as NeatensteinSpriteProjection &
    Partial<NeatensteinSpriteClip>;

  return Array.isArray(maybeClip.visibleColumns)
    ? maybeClip.visibleColumns
    : null;
}

/**
 * Copy one column of a pre-rendered voxel frame onto a screen column.
 *
 * `visibleColumns` already guarantees that the screen column is closer than
 * the wall at that X, so this helper does not re-check the z-buffer. It skips
 * transparent frame pixels (alpha == 0) and clamps vertical writes to the
 * framebuffer bounds.
 *
 * @param framebuffer - Flat RGBA framebuffer to write into.
 * @param width - Framebuffer width in pixels.
 * @param height - Framebuffer height in pixels.
 * @param screenColumn - Horizontal framebuffer column to write.
 * @param drawStart - Top screen row of the projected sprite, inclusive.
 * @param drawEnd - Bottom screen row of the projected sprite, exclusive.
 * @param frame - Pre-rendered voxel snapshot.
 * @param frameX - Column of the voxel frame to sample.
 */
function renderNeatensteinVoxelSpriteColumn(
  framebuffer: Uint8ClampedArray,
  width: number,
  height: number,
  screenColumn: number,
  drawStart: number,
  drawEnd: number,
  frame: VoxelSnapshot,
  frameX: number,
): void {
  const clampedStart = clampInt(drawStart, 0, height);
  const clampedEnd = clampInt(drawEnd, 0, height);
  if (clampedStart >= clampedEnd || screenColumn < 0 || screenColumn >= width) {
    return;
  }

  const frameHeight = frame.height;
  const frameWidth = frame.width;
  const frameData = frame.data;
  if (frameHeight === 0 || frameWidth === 0) {
    return;
  }

  const safeFrameX = clampInt(frameX, 0, frameWidth - 1);
  const spriteHeightPixels = clampedEnd - clampedStart;

  for (let rowOffset = 0; rowOffset < spriteHeightPixels; rowOffset += 1) {
    const screenY = clampedStart + rowOffset;
    const v = rowOffset / (drawEnd - drawStart);
    const frameY = Math.floor(v * (frameHeight - 1));
    const safeFrameY = clampInt(frameY, 0, frameHeight - 1);

    const frameOffset =
      (safeFrameY * frameWidth + safeFrameX) * NEATENSTEIN_RGBA_CHANNELS;
    const alpha = frameData[frameOffset + 3];
    if (alpha === 0) {
      continue;
    }

    const screenOffset =
      (screenY * width + screenColumn) * NEATENSTEIN_RGBA_CHANNELS;
    framebuffer[screenOffset] = frameData[frameOffset];
    framebuffer[screenOffset + 1] = frameData[frameOffset + 1];
    framebuffer[screenOffset + 2] = frameData[frameOffset + 2];
    framebuffer[screenOffset + 3] = alpha;
  }
}

/**
 * Render a projected sprite into the CPU ImageData framebuffer with z-buffer
 * occlusion.
 *
 * Only sprite columns closer than the wall distance at the same screen column
 * are drawn. The framebuffer is flushed with a single `putImageData` call.
 *
 * The framebuffer width is inferred from `zBuffer.length`, which correctly
 * supports rectangular render targets such as `640x480`.
 *
 * @param framebuffer - Flat RGBA framebuffer.
 * @param zBuffer - Per-column wall-depth buffer filled by the wall pass.
 * @param projection - Screen-space projection from
 *   {@link projectNeatensteinSprite} or {@link clipNeatensteinSprite}.
 * @param source - Pre-rendered {@link VoxelSnapshot}, an {@link EncodedRobotSpriteFrame},
 *   or a legacy color string. Color strings are ignored.
 * @param ctx - Canvas-like context with `putImageData`.
 * @param teamColor - Optional [r, g, b] team color to swap palette indices
 *   5/6/7 when decoding an encoded frame source. Ignored for VoxelSnapshot sources.
 */
export function renderNeatensteinSprite(
  framebuffer: Uint8ClampedArray,
  zBuffer: Readonly<Float32Array>,
  projection: NeatensteinSpriteProjection,
  source: NeatensteinSpriteSource,
  ctx: NeatensteinSpriteRenderContext,
  teamColor?: readonly [number, number, number],
): void {
  if (!isDrawableSpriteProjection(projection) || zBuffer.length === 0) {
    return;
  }

  if (typeof source === 'string') {
    return;
  }

  const frame = isEncodedRobotSpriteFrame(source)
    ? teamColor !== undefined
      ? resolveDecodedRobotSpriteFrameWithTeamColor(source, teamColor)
      : resolveDecodedRobotSpriteFrame(source)
    : source;

  // In the normal renderer path, zBuffer.length is the framebuffer width.
  const { width, height } = resolveFramebufferSize(
    framebuffer,
    zBuffer.length,
    undefined,
  );

  if (!isDrawableFramebufferSize({ width, height })) {
    return;
  }

  // Use precomputed clipping if the caller passed clipNeatensteinSprite(...).
  const visibleColumns =
    getPrecomputedVisibleColumns(projection) ??
    clipNeatensteinSpriteSpan(
      zBuffer,
      projection.left,
      projection.right,
      projection.perpDist,
    ).visibleColumns;

  if (visibleColumns.length === 0) {
    return;
  }

  const spanPixels = projection.right - projection.left;

  // Center sprite vertically on the horizon/midline.
  const halfScale = projection.scale / 2;
  const centerY = height / 2;
  const drawStart = Math.floor(centerY - halfScale);
  const drawEnd = Math.floor(centerY + halfScale);

  for (const column of visibleColumns) {
    const u = spanPixels > 0 ? (column - projection.left) / spanPixels : 0;
    const frameX = Math.floor(u * (frame.width - 1));

    renderNeatensteinVoxelSpriteColumn(
      framebuffer,
      width,
      height,
      column,
      drawStart,
      drawEnd,
      frame,
      frameX,
    );
  }

  ctx.putImageData({ data: framebuffer, width, height }, 0, 0);
}

/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyResolveFramebufferSize = resolveFramebufferSize;

/* istanbul ignore next -- test-only introspection hook */
export const __testOnlyRenderNeatensteinVoxelSpriteColumn =
  renderNeatensteinVoxelSpriteColumn;
