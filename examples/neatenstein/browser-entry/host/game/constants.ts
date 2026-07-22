/**
 * Gameplay constants for the Neatenstein host-side simulation.
 *
 * These values are locked by the Phase 2 design consensus and are the
 * authoritative source for tick cadence, player limits, enemy concurrency,
 * dash timing, and episode bounds. They are kept in a single file so that
 * red-phase contract tests can assert their exact values before the rest of
 * the game logic consumes them.
 *
 * @module
 */

import { NEATENSTEIN_MAP_SIZE } from '../../constants';

/** Fixed simulation timestep in milliseconds (≈60 Hz). */
export const NEATENSTEIN_FIXED_TIMESTEP_MS = 16;

/** Maximum player health at the start of an episode. */
export const NEATENSTEIN_PLAYER_MAX_HEALTH = 100;

/** Maximum ammo the player can carry at the start of an episode. */
export const NEATENSTEIN_PLAYER_MAX_AMMO = 30;

/** Maximum number of enemies that can be active at the same time. */
export const NEATENSTEIN_ENEMY_MAX_CONCURRENT = 8;

/** Milliseconds of invulnerability granted by a single dash. */
export const NEATENSTEIN_DASH_INVULNERABILITY_MS = 200;

/**
 * Minimum milliseconds between consecutive dashes.
 *
 * Must remain strictly greater than {@link NEATENSTEIN_DASH_INVULNERABILITY_MS}
 * so the cooldown is observable after the i-frame window ends.
 */
export const NEATENSTEIN_DASH_COOLDOWN_MS = 500;

/** Minimum duration of a default episode in milliseconds. */
export const NEATENSTEIN_EPISODE_MIN_DURATION_MS = 15_000;

/** Maximum duration of a default episode in milliseconds. */
export const NEATENSTEIN_EPISODE_MAX_DURATION_MS = 25_000;

/**
 * Default duration of a Neatenstein episode in milliseconds.
 *
 * Chosen to land in the middle of the allowed 15–25 second band so the
 * replay window is long enough to express behavior but short enough to keep
 * the evolution harness above the minimum generation cadence.
 */
export const NEATENSTEIN_EPISODE_DEFAULT_DURATION_MS = 20_000;

/**
 * Minimum number of evolutionary generations the harness must support per
 * minute of wall-clock time. Used to sanity-check episode length plus
 * evaluation overhead.
 */
export const NEATENSTEIN_MIN_GENERATIONS_PER_MINUTE = 2;

/**
 * Player movement speed in cells per second.
 *
 * Tuned so a keyboard-only player can cross the central arena in about one
 * second while remaining controllable in narrow corridors.
 */
export const NEATENSTEIN_PLAYER_SPEED_CELLS_PER_SECOND = 6;

/**
 * Player collision radius in cells.
 *
 * Slightly smaller than half a cell so the player can slide through a
 * one-cell-wide opening while still colliding with walls that come within the
 * body radius.
 */
export const NEATENSTEIN_PLAYER_RADIUS_CELLS = 0.25;

/**
 * Cell distance within which an enemy triggers contact damage.
 *
 * Measured from the player center to the enemy center; chosen to feel fair
 * without letting enemies damage the player through thin walls.
 */
export const NEATENSTEIN_CONTACT_RANGE_CELLS = 0.5;

/**
 * Hit points removed from the player on each contact-damage tick.
 *
 * Smaller than the beam damage so melee pressure is threatening but not
 * instantly lethal.
 */
export const NEATENSTEIN_CONTACT_DAMAGE = 10;

/**
 * Milliseconds of invulnerability granted after taking contact damage.
 *
 * Prevents a single enemy from draining health every frame while the player
 * is standing in a doorway.
 */
export const NEATENSTEIN_CONTACT_IFRAME_MS = 500;

/**
 * Spawn X coordinate for the player at episode start.
 *
 * Placed in the center of the central open arena so the camera never starts
 * inside a wall cell.
 */
export const NEATENSTEIN_SPAWN_CENTER_X =
  Math.floor(NEATENSTEIN_MAP_SIZE / 2) + 0.5;

/**
 * Spawn Y coordinate for the player at episode start.
 *
 * Paired with {@link NEATENSTEIN_SPAWN_CENTER_X} to form a guaranteed-open
 * starting position.
 */
export const NEATENSTEIN_SPAWN_CENTER_Y =
  Math.floor(NEATENSTEIN_MAP_SIZE / 2) + 0.5;

/**
 * Mouse sensitivity used for pointer-lock, touch and worker-tier look.
 *
 * Expressed in radians of yaw/pitch per pixel of movement. A value of 0.0022
 * rad/px is a common FPS default and keeps a 1920px horizontal sweep close to
 * a half rotation.
 */
export const NEATENSTEIN_MOUSE_SENSITIVITY = 0.0022;

/**
 * Pointer-lock options used when requesting lock on the game canvas.
 *
 * `unadjustedMovement: true` disables OS-level mouse acceleration so that the
 * same physical mouse motion always produces the same in-game rotation. This
 * is feature-detected at runtime; older browsers fall back to plain
 * `requestPointerLock()`.
 */
export const NEATENSTEIN_POINTER_LOCK_OPTIONS: PointerLockOptions = {
  unadjustedMovement: true,
};

/**
 * Mapping from movement intent to the keyboard `code` values that drive it.
 *
 * The router reads these codes during `keydown`/`keyup` events so the game
 * responds to physical key positions rather than localized characters.
 */
export const NEATENSTEIN_KEY_MAP_MOVEMENT = {
  forward: 'KeyW',
  backward: 'KeyS',
  left: 'KeyA',
  right: 'KeyD',
} as const;

/**
 * Mapping from look intent to the keyboard `code` values used as a mouse
 * fallback.
 */
export const NEATENSTEIN_KEY_MAP_LOOK = {
  left: 'ArrowLeft',
  right: 'ArrowRight',
  up: 'ArrowUp',
  down: 'ArrowDown',
} as const;

/**
 * Keyboard `code` for the dash action.
 *
 * The router tracks this key in the input snapshot; the state machine applies
 * a short invulnerability window and movement boost when the key is held.
 */
export const NEATENSTEIN_DASH_KEY = 'Space' as const;

/**
 * Keyboard `code` for the fire action.
 *
 * The router tracks this key as a keyboard fallback; the primary fire input is
 * the left mouse button (see {@link ../controls.ts#bindMouseFire}). The combat
 * system decides when a press becomes an actual shot based on ammo and cooldown.
 */
export const NEATENSTEIN_FIRE_KEY = 'KeyF' as const;

/**
 * Maximum cell distance a neon beam can travel before it is forced to end.
 *
 * Set generously larger than the map diagonal so the beam always reaches
 * the far wall from any valid player position.
 */
export const NEATENSTEIN_BEAM_MAX_RANGE_CELLS = 48;

/**
 * Hit points removed from an enemy by a single neon beam hit.
 *
 * Chosen so a freshly spawned enemy with moderate health is destroyed by
 * one or two well-placed shots.
 */
export const NEATENSTEIN_BEAM_DAMAGE = 50;

/**
 * Milliseconds a fired tracer remains visible in the world.
 *
 * Short enough to read as a transient laser flash rather than a lingering
 * beam, but long enough to be clearly visible at 60 FPS.
 */
export const NEATENSTEIN_TRACER_DURATION_MS = 80;

/**
 * CSS color applied to all neon beam tracers.
 *
 * A bright cyan/blue that reads as "neon" against the dark cell-shaded walls.
 */
export const NEATENSTEIN_BEAM_COLOR = '#00bfff';

/**
 * Perpendicular distance within which an enemy center is considered hit by
 * the beam.
 *
 * Tuned to feel generous without making thin grazing shots count as hits.
 */
export const NEATENSTEIN_ENEMY_HIT_RADIUS_CELLS = 0.4;

/**
 * Minimum touch drag distance in CSS pixels before a touch-move is treated
 * as intentional look input.
 *
 * Keeps small accidental taps from jerking the camera on mobile.
 */
export const NEATENSTEIN_TOUCH_DRAG_THRESHOLD_PX = 8;

/**
 * Keyboard look rotation applied for each arrow-key event.
 *
 * Used by {@link ./controls.ts#bindKeyboardLook} as a discrete fallback when
 * pointer lock is unavailable.
 */
export const NEATENSTEIN_KEYBOARD_LOOK_RAD_PER_EVENT = 0.05;
