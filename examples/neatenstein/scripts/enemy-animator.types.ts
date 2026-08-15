/**
 * @module enemy-animator.types
 *
 * Extracted type definitions for the enemy animator module.
 */

/**
 * Valid animation states used by the Neatenstein enemy sprite animator.
 *
 * The four primary states drive the enemy sprite sheet. The optional
 * `damage` state is a two-frame overlay used for hit-flash feedback.
 */
export type EnemyAnimationState = 'idle' | 'move' | 'fire' | 'death' | 'damage';

/**
 * Frame index and total frame count returned by the enemy animator.
 */
export interface EnemyAnimationFrame {
  /** Zero-based frame index for the current state. */
  frameIndex: number;
  /** Total number of frames available for the current state. */
  frameCount: number;
}