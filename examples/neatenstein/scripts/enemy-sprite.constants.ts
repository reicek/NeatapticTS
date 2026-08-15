/**
 * @module enemy-sprite.constants
 *
 * Extracted constants for the enemy sprite rendering module.
 *
 * Single source of truth for the shared `ENEMY_SPRITE_STATES` and
 * `ENEMY_SPRITE_DIRECTIONS` arrays, sprite timing/size constants, bolt
 * colors, and the atlas/manifest filenames. Re-exports the shared frame
 * and reference sizes from `enemy-animator.constants`.
 */

import type { EnemyAnimationState } from './enemy-animator.types';

// Re-export shared sizes from the animator constants for convenience.
export {
  ENEMY_FRAME_SIZE_PX,
  ENEMY_REFERENCE_SIZE_PX,
} from './enemy-animator.constants';

/**
 * Duration of the spawn force-field in milliseconds, controlling how long new
 * enemies remain wrapped in the pulsing teal energy shell.
 *
 * A newly spawned enemy is wrapped in a pulsing teal energy shell for this
 * length of time. The shell is purely visual and does not block damage.
 */
export const ENEMY_SPRITE_SPAWN_FORCE_FIELD_DURATION_MS = 3000;

/**
 * Duration of the death de-rez animation in milliseconds, matching the
 * controller-side timing so the visual and AI states stay locked.
 *
 * The 700 ms window delivers a fast Tron-style pixel scatter rather than the
 * original 4-second fade, keeping combat feedback snappy.
 */
export const ENEMY_SPRITE_DEATH_DE_REZ_DURATION_MS = 700;

/**
 * World-space size of a billboard sprite in grid cells, defining both width
 * and height so the projected sprite stays square in camera space.
 */
export const ENEMY_SPRITE_WORLD_SIZE = 0.5;

/**
 * Minimum perpendicular distance from the camera plane at which a billboard
 * is drawn, used to avoid division-by-zero and extreme near-plane stretching.
 */
export const ENEMY_SPRITE_NEAR_CLIP = 0.1;

/**
 * Number of RGBA channels per framebuffer pixel.
 *
 * Shared by the orchestrator in the main module and the executor functions
 * in `enemy-sprite.utils`.
 */
export const ENEMY_SPRITE_RGBA_CHANNELS = 4;

/**
 * Packed teal RGB color (`0x00dcdc`) used for the pulsing spawn force-field
 * bolt effect that wraps newly spawned enemies.
 */
export const ENEMY_BOLT_LIGHT_TEAL = 0x00dcdc;

/**
 * Packed orange RGB color (`0xff8c00`) used for the death de-rez bolt effect
 * that accompanies an enemy's ammo or health depletion.
 */
export const ENEMY_BOLT_LIGHT_ORANGE = 0xff8c00;

/**
 * Minimum absolute determinant accepted for the inverse camera matrix.
 *
 * Values below this threshold are treated as a singular (degenerate) camera
 * matrix and the sprite is marked invisible.
 */
export const ENEMY_SPRITE_DETERMINANT_EPSILON = 1e-9;

/**
 * Default ambient light added to directional diffuse.
 *
 * Ensures the enemy sprite is visible even when the diffuse term is zero
 * (light coming from behind the enemy).
 */
export const ENEMY_SPRITE_AMBIENT_LIGHT = 0.25;

/**
 * Maximum light multiplier; used to clamp the final lighting value.
 *
 * Prevents over-bright sprites when the diffuse and ambient terms sum above
 * 1.0.
 */
export const ENEMY_SPRITE_MAX_LIGHT = 1.0;

/**
 * Period of the bolt flicker in milliseconds.
 *
 * Controls the high-frequency sine pulse overlaid on the spawn/death bolt
 * fade to give the energy shell a "bolt" flicker appearance.
 */
export const ENEMY_BOLT_PULSE_PERIOD_MS = 200;

/**
 * Animation states stored in the generated runtime atlas, in the same order
 * as `generate-enemy-sprites.ts` blits them.
 *
 * Single source of truth — `generate-enemy-sprites.ts` and
 * `enemy-sprite.utils.ts` both import from here.
 */
export const ENEMY_SPRITE_STATES: readonly EnemyAnimationState[] = [
  'idle',
  'move',
  'fire',
  'death',
];

/**
 * Number of yaw directions stored in the generated runtime atlas.
 *
 * Single source of truth — used by the sprite system, snapshot renderer,
 * and reference snapshot generator.
 */
export const ENEMY_SPRITE_DIRECTIONS = 8;

/** Filename of the generated enemy sprite atlas PNG. */
export const ENEMY_SPRITE_ATLAS_FILENAME = 'enemy-sprite-atlas.png';

/** Filename of the generated enemy sprite manifest JSON. */
export const ENEMY_SPRITE_MANIFEST_FILENAME = 'enemy-sprite-manifest.json';