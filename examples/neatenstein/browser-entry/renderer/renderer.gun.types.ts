/**
 * Type definitions for the gun sprite module extracted from
 * {@link module:./gun-sprite-decode}.
 *
 * @module
 */

/**
 * Encoded gun sprite frame: rows of palette indices into
 * `GUN_SPRITE_PALETTE`.
 */
export type EncodedGunSpriteFrame = readonly (readonly number[])[];
