/**
 * @module enemy-animator.constants
 *
 * Extracted constants for the enemy animator module.
 *
 * Re-exports shared animation state string constants from
 * `browser-entry/constants.ts` and defines the canonical frame/reference
 * pixel sizes and animation timing constants consumed across the sprite
 * pipeline.
 */

export {
  ANIM_STATE_IDLE,
  ANIM_STATE_MOVE,
  ANIM_STATE_FIRE,
  ANIM_STATE_DEATH,
  ANIM_STATE_DAMAGE,
  ANIMATION_STATES,
} from '../browser-entry/constants';

/**
 * Pixel size of each runtime sprite frame in the generated atlas.
 *
 * Single source of truth for the 128-pixel frame size used by the sprite
 * generator, snapshot renderer, and sprite constants.
 */
export const ENEMY_FRAME_SIZE_PX = 128;

/**
 * Width and height in pixels of each generated reference snapshot.
 *
 * Single source of truth for the 192-pixel reference size used by the
 * sprite generator, snapshot renderer, and sprite constants.
 */
export const ENEMY_REFERENCE_SIZE_PX = 192;

/**
 * Default milliseconds per animation frame.
 *
 * Controls the speed at which the deterministic animator cycles through
 * frames for a given state.
 */
export const MS_PER_FRAME = 100;
