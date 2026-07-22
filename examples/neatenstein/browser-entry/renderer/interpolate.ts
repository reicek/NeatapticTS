/**
 * Render-side scalar state interpolation for the Neatenstein demo.
 *
 * The simulation runs at a fixed tick rate, but the renderer may draw between
 * ticks. This module blends two simulation snapshots so that camera and entity
 * motion stay smooth even when the frame rate is higher than the simulation rate.
 * Every own enumerable numeric field is interpolated between the two snapshots.
 * The generic bound `T extends Record<string, number>` guarantees that every
 * field is a number, so interpolation is always defined at the type level.
 *
 * @module
 */

/**
 * Linearly interpolate every numeric field between two state snapshots.
 *
 * @param previous - Snapshot from the previous simulation tick.
 * @param current - Snapshot from the current simulation tick.
 * @param alpha - Blend factor in the closed interval `[0, 1]`. `0` returns the
 *   previous snapshot, `1` returns the current snapshot, and intermediate values
 *   produce a proportional blend. Must be a finite number.
 * @returns A new state object with interpolated scalar fields.
 * @throws {Error} When `alpha` is not a finite number.
 *
 * @example
 * ```ts
 * const prev = { posX: 1, posY: 2, yaw: 0.5 };
 * const curr = { posX: 3, posY: 4, yaw: 1.5 };
 * const mid = lerpNeatensteinState(prev, curr, 0.25);
 * // mid === { posX: 1.5, posY: 2.5, yaw: 0.75 }
 * ```
 */
export function lerpNeatensteinState<T extends Record<string, number>>(
  previous: T,
  current: T,
  alpha: number,
): T {
  if (!Number.isFinite(alpha)) {
    throw new Error(`alpha must be a finite number, got ${String(alpha)}`);
  }

  const clamped = Math.max(0, Math.min(1, alpha));

  if (clamped <= 0) {
    return { ...previous };
  }

  if (clamped >= 1) {
    return { ...current };
  }

  const result = {} as Record<string, number>;
  const keys = new Set([...Object.keys(previous), ...Object.keys(current)]);

  for (const key of keys) {
    const from = previous[key];
    const to = current[key];
    if (typeof from === 'number' && typeof to === 'number') {
      result[key] = from + (to - from) * clamped;
    } else if (typeof to === 'number') {
      // Key was added in the current snapshot; skip interpolation and use
      // the current value so the result stays finite.
      result[key] = to;
    } else {
      // Key was removed in the current snapshot; carry over the previous value.
      result[key] = from;
    }
  }

  return result as T;
}
