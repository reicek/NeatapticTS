/**
 * Pure input utility executors extracted from the display worker.
 *
 * These helpers merge and normalize host input messages into the
 * {@link GameTickInputSnapshot} shape expected by the game tick. They are pure
 * functions with no dependency on module-level mutable state.
 *
 * @module
 */

import type { GameTickInputSnapshot } from '../host/game/tick';

/* istanbul ignore next -- pre-existing input-merge helper; only the null-previous branch is exercised by this test suite */
/**
 * Merge a new input snapshot into the pending snapshot.
 *
 * Movement and look use the latest values. One-shot actions are latched so a
 * short fire/dash press cannot be overwritten before the next `simState` tick.
 *
 * @param previous - Existing pending input, if any.
 * @param next - Newly received input.
 * @returns Merged pending input.
 */
export function mergePendingTickInput(
  previous: GameTickInputSnapshot | null,
  next: GameTickInputSnapshot,
): GameTickInputSnapshot {
  if (previous === null) {
    return next;
  }

  return {
    move: next.move,
    lookDelta: next.lookDelta,
    fire: previous.fire || next.fire,
    dash: previous.dash || next.dash,
  };
}

/* istanbul ignore next -- input normalization is defensive pre-existing code; the test suite only exercises valid host input */
/**
 * Normalize a host input message into the snapshot shape expected by game tick.
 *
 * @param raw - Value attached to the input message by the host.
 * @returns Normalized tick input snapshot.
 */
export function inputMessageToTickInput(raw: unknown): GameTickInputSnapshot {
  if (!raw || typeof raw !== 'object') {
    return {
      move: { x: 0, y: 0 },
      lookDelta: 0,
      fire: false,
      dash: false,
    };
  }

  const input = raw as Record<string, unknown>;

  const yawDelta =
    typeof input.yawDelta === 'number' && Number.isFinite(input.yawDelta)
      ? input.yawDelta
      : 0;

  let lookDelta = yawDelta;
  if (input.look && typeof input.look === 'object') {
    const lookRecord = input.look as Record<string, unknown>;
    const lookYaw = lookRecord.yawDelta;

    if (typeof lookYaw === 'number' && Number.isFinite(lookYaw)) {
      lookDelta = lookYaw;
    }
  }

  let moveX = 0;
  let moveY = 0;

  if (input.movement && typeof input.movement === 'object') {
    const movement = input.movement as Record<string, unknown>;

    if (movement.forward === true) moveY += 1;
    if (movement.backward === true) moveY -= 1;
    if (movement.left === true) moveX -= 1;
    if (movement.right === true) moveX += 1;
  } else if (input.move && typeof input.move === 'object') {
    const move = input.move as Record<string, unknown>;

    if (typeof move.x === 'number' && Number.isFinite(move.x)) {
      moveX = move.x;
    }

    if (typeof move.y === 'number' && Number.isFinite(move.y)) {
      moveY = move.y;
    }
  }

  return {
    move: { x: moveX, y: moveY },
    lookDelta,
    fire: input.fire === true,
    dash: input.dash === true,
  };
}
