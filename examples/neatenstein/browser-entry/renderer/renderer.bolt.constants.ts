/**
 * Bolt rendering constants extracted from the bolt renderer modules.
 *
 * Centralises composite operations, color literals, bolt radii, alpha values,
 * and projection ratios so they are defined once and reused across
 * {@link module:./bolt-render} and {@link module:./bolt.utils}.
 *
 * @module
 */

/**
 * Canvas composite operation used for additive blending of neon bolts and
 * impact spots.
 *
 * The `'lighter'` composite operation adds source and destination pixel
 * values, producing the characteristic neon glow overlap effect.
 */
export const COMPOSITE_OP_LIGHTER = 'lighter' as const;

/** CSS hex color for the bright white inner core of a player plasma bolt. */
export const COLOR_WHITE_HEX = '#ffffff';

/** Empty CSS color string used to disable glow shadow on inner-core draws. */
export const COLOR_EMPTY_STRING = '';

/**
 * Camera-height offset used when projecting plasma bolts to screen space.
 *
 * A value of zero places the bolt on the horizon, so airborne projectiles
 * read as center-screen shots from the DOOM-style gun overlay rather than
 * floor-plane decals.
 */
export const BOLT_PROJECTED_CAMERA_HEIGHT_WORLD = 0;

/**
 * Screen-space ratio for the plasma-cannon muzzle anchor X coordinate.
 *
 * Bolts are interpolated from a point just below the gun barrel tip so the
 * projectile visibly leaves the weapon and travels toward the projected
 * target.
 */
export const BOLT_MUZZLE_SCREEN_X_RATIO = 0.5;

/**
 * Screen-space ratio for the plasma-cannon muzzle anchor Y coordinate.
 *
 * The 0.82 ratio places the muzzle near the bottom of the screen, matching
 * the gun overlay position.
 */
export const BOLT_MUZZLE_SCREEN_Y_RATIO = 0.82;

/**
 * Screen-space plasma bolt radius at the muzzle.
 *
 * Twice the previous 4.5 px bolt radius so the shot reads as a chunky
 * plasma projectile as it leaves the gun.
 */
export const NEATENSTEIN_BOLT_MUZZLE_SCREEN_RADIUS_PX = 9;

/**
 * Screen-space plasma bolt radius at maximum range.
 *
 * At 30 cells the bolt must shrink to a single screen pixel in diameter.
 */
export const NEATENSTEIN_BOLT_MIN_SCREEN_RADIUS_PX = 0.5;

/** Maximum opacity of a freshly spawned plasma bolt. */
export const NEATENSTEIN_BOLT_MAX_SCREEN_ALPHA = 0.95;

/** Fraction of the outer bolt radius occupied by the bright inner core. */
export const NEATENSTEIN_BOLT_CORE_RADIUS_RATIO = 0.6;

/** CSS color for enemy bolt glow (red-orange, distinct from player teal). */
export const NEATENSTEIN_ENEMY_BOLT_COLOR = '#ff4400';

/** CSS color for the bright inner core of an enemy bolt. */
export const NEATENSTEIN_ENEMY_BOLT_CORE_COLOR = '#ffaa00';

/** Screen-space enemy bolt radius at the muzzle (enemy position). */
export const NEATENSTEIN_ENEMY_BOLT_MUZZLE_SCREEN_RADIUS_PX = 7;

/** Screen-space enemy bolt radius at maximum range. */
export const NEATENSTEIN_ENEMY_BOLT_MIN_SCREEN_RADIUS_PX = 0.5;

/** Maximum opacity of a freshly spawned enemy bolt. */
export const NEATENSTEIN_ENEMY_BOLT_MAX_SCREEN_ALPHA = 0.9;

/** Fraction of the outer enemy bolt radius occupied by the bright inner core. */
export const NEATENSTEIN_ENEMY_BOLT_CORE_RADIUS_RATIO = 0.55;

/** CSS color for the enemy bolt explosion flash on player hit. */
export const NEATENSTEIN_BOLT_EXPLOSION_FLASH_COLOR = '#ff8800';

/** CSS glow color for the enemy bolt explosion flash on player hit. */
export const NEATENSTEIN_BOLT_EXPLOSION_FLASH_GLOW_COLOR = '#ff6600';

/** Radius multiplier for the explosion flash relative to the bolt radius. */
export const NEATENSTEIN_BOLT_EXPLOSION_FLASH_RADIUS_MULTIPLIER = 2.5;

/** Alpha multiplier for the explosion flash relative to the bolt alpha. */
export const NEATENSTEIN_BOLT_EXPLOSION_FLASH_ALPHA_MULTIPLIER = 0.6;
