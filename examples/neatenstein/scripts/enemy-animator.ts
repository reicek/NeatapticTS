/**
 * @module enemy-animator
 *
 * State-based enemy animator for the Neatenstein voxel-sprite pipeline.
 *
 * Exports the approved per-state frame counts and a deterministic
 * `getEnemyAnimationFrame` helper used by the sprite-sheet generator to pick
 * the correct frame for a given enemy animation state and elapsed time.
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

/**
 * Approved per-state frame counts used by the deterministic enemy animator.
 *
 * These counts are locked by the Step 06 art pipeline:
 * - `idle`: 6 frames of breathing/bob.
 * - `move`: 12 frames of stride cycle.
 * - `fire`: 3 frames of cannon recoil + muzzle flash bloom.
 * - `death`: 12 frames of collapse, disk flicker, and darkening.
 * - `damage`: 2 optional frames for red/white hit-flash overlay.
 */
export const ENEMY_ANIMATION_FRAME_COUNTS: Record<EnemyAnimationState, number> =
  {
    idle: 6,
    move: 12,
    fire: 3,
    death: 12,
    damage: 2,
  };

/** Default milliseconds per animation frame. */
const MS_PER_FRAME = 100;

/**
 * Clamp negative elapsed time to zero so frame math stays well-defined.
 *
 * @param elapsedMs - Raw elapsed time in milliseconds.
 * @returns Non-negative elapsed time.
 */
const clampElapsed = (elapsedMs: number): number => Math.max(0, elapsedMs);

/**
 * Return the current enemy animation frame for a given state.
 *
 * Frame selection is deterministic: the same `(state, elapsedMs, seed)` tuple
 * always returns the same frame index. The optional `seed` acts as a phase
 * offset so individual enemies can start at different points in a looping
 * cycle while remaining fully reproducible.
 *
 * @param state - Animation state.
 * @param elapsedMs - Elapsed time in milliseconds.
 * @param seed - Optional determinism seed; used as a phase offset.
 * @returns The frame index and total frame count for the state.
 * @throws Error when `state` is not a known animation state.
 *
 * @example
 * ```ts
 * const frame = getEnemyAnimationFrame('move', 250, 42);
 * console.log(frame.frameIndex, frame.frameCount); // 0..11, 12
 * ```
 */
export function getEnemyAnimationFrame(
  state: EnemyAnimationState,
  elapsedMs: number,
  seed?: number,
): EnemyAnimationFrame {
  const frameCount = ENEMY_ANIMATION_FRAME_COUNTS[state];
  if (frameCount === undefined) {
    throw new Error(`Unknown enemy animation state: ${String(state)}`);
  }

  const safeElapsed = clampElapsed(elapsedMs);
  const offset = seed ?? 0;
  const frameIndex =
    Math.floor((safeElapsed + offset) / MS_PER_FRAME) % frameCount;

  return { frameIndex, frameCount };
}
