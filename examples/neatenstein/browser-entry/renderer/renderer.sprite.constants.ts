/**
 * Shared sprite constants extracted from the sprite renderer modules.
 *
 * Centralises pose names, direction names, atlas yaw parameters, walk-cycle
 * constants, and projection geometry values so they are defined once and
 * reused across {@link module:./sprites}, {@link module:./sprites.atlas.utils},
 * {@link module:./sprites.projection.utils}, and
 * {@link module:./sprites.column.utils}.
 *
 * Shared constants (`RGBA_CHANNELS`, `NEATENSTEIN_INVISIBLE_SENTINEL`,
 * `FULL_CIRCLE_RADIANS`, `NEATENSTEIN_EPSILON_1E9`) are re-exported from
 * `browser-entry/constants.ts` to maintain a single source of truth.
 *
 * @module
 */

import {
  RGBA_CHANNELS,
  NEATENSTEIN_INVISIBLE_SENTINEL,
  FULL_CIRCLE_RADIANS,
  NEATENSTEIN_EPSILON_1E9,
} from '../constants';
import { type EnemyAnimationState } from '../../../neatenstein/scripts/enemy-animator';

// Re-export shared constants for renderer-level consumers.
export { RGBA_CHANNELS, NEATENSTEIN_INVISIBLE_SENTINEL };

/**
 * Number of discrete yaw directions stored in the encoded robot sprite atlas.
 */
export const NEATENSTEIN_VOXEL_ATLAS_YAW_STEPS = 8;

/**
 * Full rotation in radians (re-exported as the sprite-specific alias).
 */
export const NEATENSTEIN_FULL_ROTATION_RADIANS = FULL_CIRCLE_RADIANS;

/**
 * Angular step between adjacent yaw atlas entries, in radians.
 */
export const NEATENSTEIN_VOXEL_ATLAS_YAW_STEP_RADIANS =
  FULL_CIRCLE_RADIANS / NEATENSTEIN_VOXEL_ATLAS_YAW_STEPS;

/**
 * Camera-relative yaw indices mapped to encoded direction names.
 *
 * Index 0 means the sprite faces the camera; index 4 means it faces away.
 */
export const NEATENSTEIN_ENCODED_DIRECTIONS = [
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
 * Row index where the upper body (shoot) and lower body (walk) split.
 *
 * Rows 0–34 are upper body; rows 35–47 are lower body.
 */
export const NEATENSTEIN_SPRITE_UPPER_BODY_SPLIT_ROW = 35;

/**
 * World-space size of a sprite in grid cells.
 *
 * The projected screen size is:
 *
 * ```ts
 * focalLength / perpDist * NEATENSTEIN_SPRITE_WORLD_SIZE
 * ```
 *
 * where `focalLength` is derived from the floor FOV. Keeping this value in
 * world units makes enemies scale consistently as they move toward or away
 * from the camera.
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
 * Minimum absolute determinant accepted for the inverse camera matrix.
 *
 * A determinant close to zero means the camera direction and plane are
 * degenerate, which would make projection unstable. Uses the shared
 * {@link NEATENSTEIN_EPSILON_1E9} from `browser-entry/constants.ts`.
 */
export const NEATENSTEIN_CAMERA_DETERMINANT_EPSILON = NEATENSTEIN_EPSILON_1E9;

/**
 * Distance traveled in world cells between walk-cycle pose swaps.
 */
export const NEATENSTEIN_WALK_CYCLE_HALF_STEP_CELLS = 0.5;

/**
 * Walk cycle pose sequence indexed by `floor(walkTick / 4) % 4`:
 * stand → walk1 → stand → walk2 (4x slowed).
 */
export const NEATENSTEIN_WALK_CYCLE_POSES = [
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
 * The `move` state is resolved dynamically by `resolveNeatensteinEnemyFrame`
 * so walking enemies alternate between `walk1` and `walk2` as they travel.
 */
export const NEATENSTEIN_ANIMATION_TO_POSE: Record<
  EnemyAnimationState,
  'stand' | 'walk1' | 'walk2' | 'shoot'
> = {
  idle: 'stand',
  move: 'walk1',
  fire: 'shoot',
  death: 'stand',
  damage: 'stand',
};
