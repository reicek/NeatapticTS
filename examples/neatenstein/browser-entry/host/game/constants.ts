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

import {
  NEATENSTEIN_FIXED_TIMESTEP_MS,
  NEATENSTEIN_LIGHT_TOGGLE_KEY,
  NEATENSTEIN_MAP_SIZE,
} from '../../constants';

export {
  NEATENSTEIN_FIXED_TIMESTEP_MS,
  NEATENSTEIN_LIGHT_TOGGLE_KEY,
  NEATENSTEIN_MAP_SIZE,
};

/** Number of milliseconds in one second. */
export const NEATENSTEIN_MS_PER_SECOND = 1000;

/** Maximum player health at the start of an episode. */
export const NEATENSTEIN_PLAYER_MAX_HEALTH = 100;

/** Maximum ammo the player can carry at the start of an episode. */
export const NEATENSTEIN_PLAYER_MAX_AMMO = 50;

/** Maximum number of enemies that can be active at the same time. */
export const NEATENSTEIN_ENEMY_MAX_CONCURRENT = 8;

/**
 * Radius in world cells around the map center where enemies may spawn.
 *
 * Enemies are placed at a random angle and a random distance between
 * {@link NEATENSTEIN_ENEMY_SPAWN_MIN_DISTANCE_CELLS} and this value, so the
 * effective spawn region is an annulus centered on
 * {@link NEATENSTEIN_SPAWN_CENTER_X} / {@link NEATENSTEIN_SPAWN_CENTER_Y}.
 */
export const NEATENSTEIN_ENEMY_SPAWN_RADIUS = 8;

/**
 * Minimum distance in world cells between an enemy spawn and the player spawn
 * center.
 *
 * Keeps enemies from appearing inside {@link NEATENSTEIN_CONTACT_RANGE_CELLS},
 * which would deal immediate contact damage and end the episode far earlier
 * than the intended 15–25 second duration band. The value is chosen to be
 * larger than the contact range so the player has a brief reaction window
 * even before movement begins.
 */
export const NEATENSTEIN_ENEMY_SPAWN_MIN_DISTANCE_CELLS = 1;

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
 * Enemy collision radius in world cells.
 *
 * Enemies are authored on a 192×192 block footprint. With a floor cell size of
 * 252 blocks, the enemy radius is half the footprint: 96 blocks, or 96/252 cells.
 */
export const NEATENSTEIN_ENEMY_COLLISION_RADIUS_CELLS = 96 / 252;

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
 * Smaller than the bolt damage so melee pressure is threatening but not
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
 * Distance in cells to offset the bolt origin forward from the player center.
 *
 * Keeps the projectile origin in front of the camera near-plane so the
 * renderer's depth projection accepts it instead of rejecting it as depth
 * less than or equal to the near-clip epsilon.
 */
export const NEATENSTEIN_MUZZLE_OFFSET_CELLS = 0.2;

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

/**
 * Button identifier for the primary (left) mouse button.
 *
 * Used by {@link ./controls.ts#bindMouseFire} so fire activates on left click
 * and ignores other buttons.
 */
export const NEATENSTEIN_PRIMARY_MOUSE_BUTTON = 0;

/**
 * Button identifier for the secondary (right) mouse button.
 *
 * Used by tests to verify that non-primary mouse buttons are ignored for fire
 * input.
 */
export const NEATENSTEIN_SECONDARY_MOUSE_BUTTON = 2;

/**
 * Deterministic seed used by red-phase combat tests.
 *
 * A fixed seed keeps test expectations stable across RNG-driven map and
 * player-angle generation.
 */
export const NEATENSTEIN_TEST_SEED = 1;

/**
 * Default enemy health used in combat tests.
 *
 * Chosen so that a single {@link NEATENSTEIN_BOLT_DAMAGE} hit reduces health
 * to a non-zero value, while a second hit (or health equal to the bolt damage)
 * produces a confirmed kill.
 */
export const NEATENSTEIN_TEST_ENEMY_HEALTH = 100;

/**
 * Health value representing a dead enemy in contact-damage tests.
 *
 * Used to verify that enemies with zero health do not apply contact damage
 * even when their position overlaps the player.
 */
export const NEATENSTEIN_TEST_ENEMY_DEAD_HEALTH = 0;

/**
 * Hit points removed from an enemy by a single traveling plasma bolt hit.
 *
 * Chosen so a freshly spawned enemy with moderate health is destroyed by
 * one or two well-placed shots.
 */
export const NEATENSTEIN_BOLT_DAMAGE = 50;

/**
 * Travel speed of a plasma bolt in world cells per second.
 *
 * Tuned so a bolt reaches the arena edge quickly, giving it a snappy
 * projectile feel while remaining fast enough to compete with the original
 * hitscan weapon.
 */
export const NEATENSTEIN_BOLT_SPEED_CELLS_PER_SECOND = 36;

/**
 * Fixed visual travel duration of a plasma bolt in milliseconds.
 *
 * Using a constant duration instead of distance/speed makes every bolt travel
 * at the same screen-space rate regardless of target distance.
 */
export const NEATENSTEIN_BOLT_TRAVEL_DURATION_MS = 300;

/**
 * Maximum lifetime of a plasma bolt in milliseconds.
 *
 * Caps the distance a bolt can travel and prevents deactivated bolts from
 * lingering in the simulation.
 */
export const NEATENSTEIN_BOLT_LIFETIME_MS = 2000;

/**
 * Radius in world cells used for bolt/enemy collision.
 *
 * Tuned to feel generous without making thin grazing shots count as hits.
 */
export const NEATENSTEIN_BOLT_HIT_RADIUS_CELLS = 0.4;

/**
 * Maximum distance a plasma bolt can travel in world cells.
 *
 * Bolts are deleted once they travel this far. They stop existing, cannot hit
 * anything, and fade out visually over the same range.
 */
export const NEATENSTEIN_BOLT_MAX_RANGE_CELLS = 30;

/**
 * Maximum screen-space recoil offset applied to the gun overlay after firing.
 *
 * Expressed in pixels relative to the bottom-center HUD coordinate. A larger
 * value makes each shot feel punchier; a smaller value keeps the overlay stable.
 */
export const NEATENSTEIN_GUN_RECOIL_MAX_OFFSET_PX = 8;

/**
 * Recoil decay rate in pixels per second.
 *
 * Determines how quickly the gun overlay returns to its rest position after a
 * shot. The value is chosen so the recoil settles within roughly one tick at
 * 60 FPS while still producing a visible kick.
 */
export const NEATENSTEIN_GUN_RECOIL_DECAY_PX_PER_SECOND = 480;

/**
 * CSS color applied to the dynamic light overlay.
 *
 * A bright teal that matches the gun accent color and reads as a neon muzzle
 * flash or carried light source.
 */
export const NEATENSTEIN_DYNAMIC_LIGHT_COLOR = '#00f0ff';

/**
 * World-cell radius of the dynamic light overlay.
 *
 * Defines how far the teal light illuminates nearby floor/wall geometry from
 * the player position when enabled.
 */
export const NEATENSTEIN_DYNAMIC_LIGHT_RADIUS_CELLS = 6;

/**
 * Distance in cells just beyond {@link NEATENSTEIN_CONTACT_RANGE_CELLS} used
 * to place an enemy outside contact-damage range for out-of-range testing.
 */
export const NEATENSTEIN_TEST_ENEMY_BEYOND_CONTACT_RANGE_CELLS =
  NEATENSTEIN_CONTACT_RANGE_CELLS + 1;

/**
 * Distance in cells from the player to a test enemy placed directly on the
 * bolt path.
 *
 * Small enough to sit before the nearest wall from the central spawn point
 * when firing along the +X axis, so the bolt reliably hits the enemy first.
 */
export const NEATENSTEIN_TEST_ENEMY_NEAR_DISTANCE_CELLS = 2;

/**
 * Perpendicular offset in cells used to place a test enemy just outside the
 * bolt hit radius.
 *
 * The value is larger than {@link NEATENSTEIN_BOLT_HIT_RADIUS_CELLS} so the
 * enemy is missed even though it shares the same forward distance as a hit
 * enemy.
 */
export const NEATENSTEIN_TEST_ENEMY_OFF_BOLT_OFFSET_CELLS = 0.5;
