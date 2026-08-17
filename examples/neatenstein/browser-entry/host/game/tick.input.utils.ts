/**
 * Input normalization and movement-intent executors for the Neatenstein tick
 * pipeline.
 *
 * @module
 */

import type { Vector2 } from './types';
import type {
  GameTickInputSnapshot,
  NormalizedGameTickInputSnapshot,
} from '../types';

// Re-export consolidated types so existing imports from this module remain valid.
export type { GameTickInputSnapshot } from '../types';

/**
 * Convert an unknown or partial movement vector into finite numeric components.
 *
 * Missing and non-finite components are treated as zero.
 *
 * @param move - Optional movement vector from the input snapshot.
 * @returns Sanitized movement vector.
 */
export function normalizeMoveVector(move?: Partial<Vector2>): Vector2 {
  return {
    x: typeof move?.x === 'number' && Number.isFinite(move.x) ? move.x : 0,
    y: typeof move?.y === 'number' && Number.isFinite(move.y) ? move.y : 0,
  };
}

/**
 * Normalize a partial tick input snapshot.
 *
 * Missing fields default to neutral input. Non-finite numeric fields are
 * ignored so malformed input cannot poison the deterministic game state.
 *
 * @param snapshot - Partial input snapshot supplied by host or worker.
 * @returns Fully normalized tick input.
 */
export function normalizeGameTickInput(
  snapshot: Partial<GameTickInputSnapshot>,
): NormalizedGameTickInputSnapshot {
  return {
    move: normalizeMoveVector(snapshot.move),
    lookDelta:
      typeof snapshot.lookDelta === 'number' &&
      Number.isFinite(snapshot.lookDelta)
        ? snapshot.lookDelta
        : 0,
    fire: snapshot.fire === true,
    dash: snapshot.dash === true,
  };
}

/**
 * Convert a movement vector into the directional boolean intent used by the
 * keyboard-oriented movement helper.
 *
 * A non-zero axis value is treated as pressed on that side. Diagonal vectors
 * preserve both components so {@link updatePlayerMovement} can normalize the
 * final world-space step.
 *
 * @param move - Sanitized movement vector.
 * @returns Directional movement intent.
 */
export function snapshotToMovement(move?: Vector2): {
  forward: boolean;
  backward: boolean;
  left: boolean;
  right: boolean;
} {
  const normalizedMove = normalizeMoveVector(move);

  return {
    forward: normalizedMove.y > 0,
    backward: normalizedMove.y < 0,
    left: normalizedMove.x < 0,
    right: normalizedMove.x > 0,
  };
}
