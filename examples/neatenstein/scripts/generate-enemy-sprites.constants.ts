/**
 * @module generate-enemy-sprites.constants
 *
 * Shared constants for the enemy sprite-sheet and reference-snapshot
 * generator. Re-exports frame-size and sprite-state constants from their
 * canonical sources to provide a single import surface for the generator
 * orchestrator and its sibling utility modules.
 */

// Re-export frame-size constants so the generator uses the same source as the
// runtime animator (DRY consolidation: ENEMY_FRAME_SIZE_PX / ENEMY_REFERENCE_SIZE_PX).
export {
  ENEMY_FRAME_SIZE_PX,
  ENEMY_REFERENCE_SIZE_PX,
} from './enemy-animator.constants';

// Re-export sprite-state constants from their canonical source.
export {
  ENEMY_SPRITE_STATES,
  ENEMY_SPRITE_DIRECTIONS,
} from './enemy-sprite.constants';

// Re-export direction count from the controller constants.
export { NUM_DIRECTIONS } from './enemy-controller.constants';

// Re-export RGBA channel count from the shared browser-entry constants.
export { RGBA_CHANNELS } from '../browser-entry/constants';

/**
 * Default filesystem directory where generated enemy sprite assets are written.
 */
export const DEFAULT_GENERATED_DIR = 'examples/neatenstein/generated';

/**
 * PNG IHDR chunk type identifier.
 */
export const PNG_CHUNK_IHDR = 'IHDR';

/**
 * PNG IDAT chunk type identifier.
 */
export const PNG_CHUNK_IDAT = 'IDAT';

/**
 * PNG IEND chunk type identifier.
 */
export const PNG_CHUNK_IEND = 'IEND';
