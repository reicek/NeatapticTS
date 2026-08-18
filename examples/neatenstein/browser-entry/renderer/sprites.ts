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

import { resolveNeatensteinFogFactor } from './framebuffer';
import { clipNeatensteinSpriteSpan } from './zbuffer';
import { ROBOT_SPRITE_FRAMES } from '../../robot-sprite-data.js';
import { type VoxelSnapshot } from '../shared/snapshot-renderer';
import {
  resolveCompositeShootWalkFrame,
  resolveDecodedRobotSpriteFrame,
  resolveDecodedRobotSpriteFrameWithTeamColor,
} from './sprites.atlas.utils';
import {
  getPrecomputedVisibleColumns,
  isDrawableFramebufferSize,
  isDrawableSpriteProjection,
  resolveFramebufferSize,
} from './sprites.projection.utils';
import { renderNeatensteinVoxelSpriteColumn } from './sprites.column.utils';
import {
  NEATENSTEIN_VOXEL_ATLAS_YAW_STEPS,
  NEATENSTEIN_FULL_ROTATION_RADIANS,
  NEATENSTEIN_VOXEL_ATLAS_YAW_STEP_RADIANS,
  NEATENSTEIN_WALK_CYCLE_HALF_STEP_CELLS,
  NEATENSTEIN_WALK_CYCLE_POSES,
  NEATENSTEIN_ANIMATION_TO_POSE,
  NEATENSTEIN_ENCODED_DIRECTIONS,
} from './renderer.sprite.constants';
import type {
  NeatensteinSpriteRenderContext,
  NeatensteinCamera,
  NeatensteinSprite,
  NeatensteinDerezState,
  NeatensteinSpriteSource,
  NeatensteinSpriteProjection,
  EncodedRobotSpriteFrame,
} from './renderer.sprite.types';

// Re-export previously-public symbols that moved to dedicated files.
export {
  NEATENSTEIN_VOXEL_ATLAS_YAW_STEPS,
  NEATENSTEIN_FULL_ROTATION_RADIANS,
  NEATENSTEIN_VOXEL_ATLAS_YAW_STEP_RADIANS,
  NEATENSTEIN_ENCODED_DIRECTIONS,
  NEATENSTEIN_SPRITE_UPPER_BODY_SPLIT_ROW,
  NEATENSTEIN_SPRITE_WORLD_SIZE,
  NEATENSTEIN_SPRITE_NEAR_CLIP,
  NEATENSTEIN_SPRITE_THICKNESS_RATIO,
  NEATENSTEIN_CAMERA_DETERMINANT_EPSILON,
  NEATENSTEIN_WALK_CYCLE_HALF_STEP_CELLS,
  NEATENSTEIN_WALK_CYCLE_POSES,
  NEATENSTEIN_ANIMATION_TO_POSE,
  RGBA_CHANNELS,
  NEATENSTEIN_INVISIBLE_SENTINEL,
} from './renderer.sprite.constants';
export type {
  NeatensteinSpriteRenderContext,
  NeatensteinCamera,
  NeatensteinSprite,
  NeatensteinDerezState,
  NeatensteinSpriteSource,
  NeatensteinSpriteProjection,
  EncodedRobotSpriteFrame,
} from './renderer.sprite.types';
export {
  projectNeatensteinSprite,
  clipNeatensteinSprite,
} from './sprites.projection.utils';

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
 * @param derezState - Optional de-rez death animation state. When provided,
 *   the column renderer dissolves pixels and tints survivors toward
 *   {@link NEATENSTEIN_ENEMY_DEATH_COLOR}.
 */
export function renderNeatensteinSprite(
  framebuffer: Uint8ClampedArray,
  zBuffer: Readonly<Float32Array>,
  projection: NeatensteinSpriteProjection,
  source: NeatensteinSpriteSource,
  ctx: NeatensteinSpriteRenderContext,
  teamColor?: readonly [number, number, number],
  derezState?: NeatensteinDerezState,
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

  // Compute distance fog factor so sprites fade into the background as they
  // approach the 30-cell render distance cap.
  const fogFactor = resolveNeatensteinFogFactor(projection.perpDist);

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
      fogFactor,
      derezState,
    );
  }

  // Per-sprite putImageData removed (B3.5): callers flush the framebuffer
  // once after all sprites are drawn. The ctx parameter is retained for API
  // compatibility.
}

/**
 * Test-only hook: expose the framebuffer-size resolver for direct testing.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the framebuffer-size resolver. */
export const __testOnlyResolveFramebufferSize = resolveFramebufferSize;

/**
 * Test-only hook: expose the per-column voxel sprite renderer for direct
 * pixel-level testing.
 *
 * @internal
 */
/* istanbul ignore next -- test-only introspection hook */
/** Test-only accessor for the per-column voxel sprite renderer. */
export const __testOnlyRenderNeatensteinVoxelSpriteColumn =
  renderNeatensteinVoxelSpriteColumn;
