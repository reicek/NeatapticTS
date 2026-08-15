/**
 * Type declarations for the palette-indexed gun sprite asset used by the
 * Neatenstein on-screen cannon renderer.
 *
 * @module
 */

/** Pixel scale factor applied to the logical sprite grid on decode. */
export const GUN_SPRITE_SCALE: number;

/** Palette entries as `[r, g, b, a]` tuples indexed by palette index. */
export const GUN_SPRITE_PALETTE: readonly (readonly number[])[];

/** Encoded gun sprite frames keyed by animation state. */
export const GUN_SPRITE_FRAMES: {
  /** Idle (resting) frame grid of palette indices. */
  readonly idle: readonly (readonly number[])[];
  /** Firing frame grid of palette indices (includes muzzle flash). */
  readonly fire: readonly (readonly number[])[];
};
