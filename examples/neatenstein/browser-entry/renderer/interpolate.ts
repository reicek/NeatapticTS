/**
 * Render-side scalar state interpolation for the Neatenstein demo.
 *
 * The simulation may update at a fixed tick rate while the renderer draws
 * between simulation ticks. This module blends two scalar simulation snapshots
 * so camera and entity motion remain visually smooth.
 *
 * The interpolation is intentionally generic: every own enumerable string-keyed
 * numeric field is copied or blended. This makes the helper useful for simple
 * render-state objects such as positions, camera offsets, timers, and other
 * scalar values.
 *
 * Important: this function performs plain linear interpolation. Values that
 * wrap around a range, such as normalized angles, should either be stored as
 * unwrapped accumulated values or interpolated with a dedicated angle-aware
 * helper before being placed into this scalar state.
 *
 * @module
 */

import {
  INTERP_CURRENT,
  INTERP_PREVIOUS,
  MAX_INTERPOLATION_ALPHA,
  MIN_INTERPOLATION_ALPHA,
} from './renderer.interpolate.constants';
import type { NeatensteinNumericState } from './renderer.interpolate.types';

// Re-export constants and types for external consumers.
export {
  INTERP_CURRENT,
  INTERP_PREVIOUS,
  MAX_INTERPOLATION_ALPHA,
  MIN_INTERPOLATION_ALPHA,
} from './renderer.interpolate.constants';
export type { NeatensteinNumericState } from './renderer.interpolate.types';

/**
 * Clamp an interpolation alpha into the supported blend range.
 *
 * @param alpha - Finite interpolation alpha.
 * @returns `alpha` clamped into `[0, 1]`.
 */
function clampInterpolationAlpha(alpha: number): number {
  return Math.max(
    MIN_INTERPOLATION_ALPHA,
    Math.min(MAX_INTERPOLATION_ALPHA, alpha),
  );
}

/**
 * Full-circle constant in radians (`2 * Math.PI`).
 */
const TWO_PI = 2 * Math.PI;

/**
 * Threshold below which a normalised angle close to `2π` is snapped to `0`.
 *
 * Floating-point arithmetic on approximate radian inputs can produce a result
 * just shy of `2π` (e.g. `6.28309` instead of `6.28319`). After normalisation
 * to `[0, 2π)` such a value remains near `2π` instead of wrapping to `0`.
 * Snapping within this epsilon (≈ 0.000 1 rad ≈ 0.006°) avoids that artifact.
 */
const ANGLE_WRAP_EPSILON = 1e-4;

/**
 * Shortest-arc angle interpolation in radians.
 *
 * Interpolates between two angles along the shortest rotational path,
 * correctly handling the wrap-around at `2π`. The result is normalised to
 * `[0, 2π)`.
 *
 * Sibling to {@link lerpNeatensteinState} for scalar fields that represent
 * angular quantities (e.g. `yaw`, `heading`).
 *
 * @param from - Starting angle in radians (any range).
 * @param to - Ending angle in radians (any range).
 * @param alpha - Finite blend factor. Values outside `[0, 1]` are clamped.
 * @returns Interpolated angle in `[0, 2π)`.
 * @throws {TypeError} When `alpha` is not a finite number.
 *
 * @example
 * ```ts
 * // 350° → 10° (shortest arc crosses 0°/2π, midpoint ≈ 0°)
 * const mid = lerpNeatensteinAngle(6.108, 0.175, 0.5);
 * // mid ≈ 0
 * ```
 */
export function lerpNeatensteinAngle(
  from: number,
  to: number,
  alpha: number,
): number {
  if (!Number.isFinite(alpha)) {
    throw new TypeError(`alpha must be a finite number, got ${String(alpha)}`);
  }

  const clampedAlpha = clampInterpolationAlpha(alpha);

  // Shortest signed angular difference in [-π, π].
  let delta = ((to - from) % TWO_PI + TWO_PI) % TWO_PI;
  if (delta > Math.PI) {
    delta -= TWO_PI;
  }

  let result = from + delta * clampedAlpha;

  // Normalise to [0, 2π).
  result = ((result % TWO_PI) + TWO_PI) % TWO_PI;

  // Snap values just below 2π to 0 to avoid floating-point wrap artifacts.
  if (result > TWO_PI - ANGLE_WRAP_EPSILON) {
    result = 0;
  }

  return result;
}

/**
 * Return whether an object has an own enumerable property for the given key.
 *
 * This mirrors the key set used by `Object.keys(...)`, keeping interpolation
 * limited to the same field category described by the module documentation.
 *
 * @param value - Object to inspect.
 * @param key - String property key to test.
 * @returns Whether `key` is an own enumerable property of `value`.
 */
function hasOwnEnumerableKey(
  value: NeatensteinNumericState,
  key: string,
): boolean {
  return Object.prototype.propertyIsEnumerable.call(value, key);
}

/**
 * Clamp a non-finite or non-numeric value to a finite number.
 *
 * `NaN` and non-number values clamp to `0`. `Infinity` clamps to
 * `Number.MAX_SAFE_INTEGER`; `-Infinity` clamps to `-Number.MAX_SAFE_INTEGER`.
 *
 * @param value - Runtime field value (may be any type).
 * @returns A finite number clamped from the non-finite input.
 */
function clampNonFiniteValue(value: unknown): number {
  if (typeof value !== 'number' || Number.isNaN(value)) {
    return 0;
  }
  // value is Infinity or -Infinity — clamp to the largest safe integer.
  return value > 0 ? Number.MAX_SAFE_INTEGER : -Number.MAX_SAFE_INTEGER;
}

/**
 * Read a numeric field from a snapshot, clamping non-finite values.
 *
 * TypeScript enforces numeric fields at compile time, but renderer code still
 * runs against JavaScript values at runtime. Rather than throwing — which
 * would abort the entire render frame — this guard logs a warning and clamps
 * `NaN`, infinities, missing fields, and malformed state to finite values so
 * the interpolated render state remains usable.
 *
 * @param snapshotName - Human-readable snapshot name for warning messages.
 * @param key - Field key being read.
 * @param value - Runtime field value.
 * @returns A finite number — the original value or a clamped replacement.
 */
function readFiniteSnapshotNumber(
  snapshotName: string,
  key: string,
  value: unknown,
): number {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }

  const clamped = clampNonFiniteValue(value);
  if (typeof console !== 'undefined' && typeof console.warn === 'function') {
    console.warn(
      `${snapshotName}.${key} was not finite (got ${String(value)}), clamped to ${clamped}`,
    );
  }

  return clamped;
}

/**
 * Copy all own enumerable numeric fields from a snapshot into a new object.
 *
 * This is used for alpha endpoints so callers still receive a fresh object
 * rather than the original snapshot reference.
 *
 * @param snapshot - Snapshot to copy.
 * @param snapshotName - Human-readable snapshot name for validation errors.
 * @returns A new object containing the snapshot's finite numeric fields.
 */
function copyFiniteSnapshot(
  snapshot: NeatensteinNumericState,
  snapshotName: string,
): NeatensteinNumericState {
  const result: NeatensteinNumericState = {};

  for (const key of Object.keys(snapshot)) {
    result[key] = readFiniteSnapshotNumber(snapshotName, key, snapshot[key]);
  }

  return result;
}

/**
 * Collect interpolation keys from both snapshots in deterministic order.
 *
 * Keys from the previous snapshot are emitted first. Keys that only exist on
 * the current snapshot are appended afterward. This preserves stable output
 * ordering while still supporting additive state evolution between snapshots.
 *
 * @param previous - Previous simulation snapshot.
 * @param current - Current simulation snapshot.
 * @returns Ordered list of own enumerable string keys to process.
 */
function collectInterpolationKeys(
  previous: NeatensteinNumericState,
  current: NeatensteinNumericState,
): string[] {
  const keys = Object.keys(previous);
  const seen = new Set(keys);

  for (const key of Object.keys(current)) {
    if (!seen.has(key)) {
      seen.add(key);
      keys.push(key);
    }
  }

  return keys;
}

/**
 * Linearly interpolate every own enumerable numeric field between two state
 * snapshots.
 *
 * The returned object is always a new object. It never mutates `previous` or
 * `current`.
 *
 * Field behavior:
 *
 * - Fields present in both snapshots are linearly interpolated.
 * - Fields only present in `current` use the current value.
 * - Fields only present in `previous` carry over the previous value.
 *
 * Alpha behavior:
 *
 * - `0` returns a finite numeric copy of `previous`.
 * - `1` returns a finite numeric copy of `current`.
 * - Values between `0` and `1` produce proportional interpolation.
 * - Finite values outside `[0, 1]` are clamped.
 * - Non-finite alpha values throw.
 * - Non-finite snapshot fields are logged and clamped to finite values.
 *
 * @param previous - Snapshot from the previous simulation tick.
 * @param current - Snapshot from the current simulation tick.
 * @param alpha - Finite blend factor. Values outside `[0, 1]` are clamped.
 * @returns A new state object with copied or interpolated scalar fields.
 * @throws {TypeError} When `alpha` is not finite. Non-finite snapshot
 *   fields are logged and clamped to finite values instead of throwing.
 *
 * @example
 * ```ts
 * const prev = { posX: 1, posY: 2, yaw: 0.5 };
 * const curr = { posX: 3, posY: 4, yaw: 1.5 };
 *
 * const mid = lerpNeatensteinState(prev, curr, 0.25);
 *
 * console.log(mid);
 * // { posX: 1.5, posY: 2.5, yaw: 0.75 }
 * ```
 */
export function lerpNeatensteinState<T extends Record<string, number>>(
  previous: T,
  current: T,
  alpha: number,
): T {
  if (!Number.isFinite(alpha)) {
    throw new TypeError(`alpha must be a finite number, got ${String(alpha)}`);
  }

  // Clamp instead of rejecting out-of-range alpha. This keeps render timing
  // tolerant of small accumulator overshoots while preserving predictable bounds.
  const clampedAlpha = clampInterpolationAlpha(alpha);

  // Endpoint fast paths still return fresh objects and validate copied fields.
  if (clampedAlpha <= MIN_INTERPOLATION_ALPHA) {
    return copyFiniteSnapshot(previous, INTERP_PREVIOUS) as T;
  }

  if (clampedAlpha >= MAX_INTERPOLATION_ALPHA) {
    return copyFiniteSnapshot(current, INTERP_CURRENT) as T;
  }

  const result: NeatensteinNumericState = {};
  const keys = collectInterpolationKeys(previous, current);

  for (const key of keys) {
    const hasPreviousValue = hasOwnEnumerableKey(previous, key);
    const hasCurrentValue = hasOwnEnumerableKey(current, key);

    if (hasPreviousValue && hasCurrentValue) {
      // Normal path: both snapshots have the field, so blend linearly.
      const from = readFiniteSnapshotNumber(
        INTERP_PREVIOUS,
        key,
        previous[key],
      );
      const to = readFiniteSnapshotNumber(INTERP_CURRENT, key, current[key]);

      result[key] = from + (to - from) * clampedAlpha;
      continue;
    }

    if (hasCurrentValue) {
      // The field was added in the current snapshot. There is no previous value
      // to interpolate from, so use the current finite value directly.
      result[key] = readFiniteSnapshotNumber(INTERP_CURRENT, key, current[key]);
      continue;
    }

    // The field was removed in the current snapshot. Preserve the previous
    // finite value so the result remains stable for this render frame.
    result[key] = readFiniteSnapshotNumber(INTERP_PREVIOUS, key, previous[key]);
  }

  return result as T;
}

/**
 * Input color shape for wall-column interpolation.
 */
interface NeatensteinWallColumn {
  screenX: number;
  r: number;
  g: number;
  b: number;
}

/**
 * Interpolate wall color between two adjacent cast columns (C1.4).
 *
 * The screen X position is NOT interpolated via `alpha` — it is always the
 * arithmetic midpoint of the two neighbor columns, rounded to the nearest
 * integer pixel. This preserves the true column pixel position required by
 * Invariant §1 (half-res decimation must never shift a column's screen X).
 *
 * Only the RGB color channels are interpolated using `alpha`.
 *
 * @param left - Left neighbor wall column (lower screen X).
 * @param right - Right neighbor wall column (higher screen X).
 * @param alpha - Interpolation factor in `[0, 1]` (`0` = left, `1` = right).
 * @returns Interpolated wall column with midpoint screenX and blended color.
 *
 * @example
 * ```ts
 * const mid = interpolateNeatensteinWallColumn(
 *   { screenX: 80, r: 10, g: 142, b: 160 },
 *   { screenX: 82, r: 0, g: 183, b: 255 },
 *   0.5,
 * );
 * // mid.screenX === 81, mid.r === 5
 * ```
 */
export function interpolateNeatensteinWallColumn(
  left: NeatensteinWallColumn,
  right: NeatensteinWallColumn,
  alpha: number,
): NeatensteinWallColumn {
  const clampedAlpha = clampInterpolationAlpha(
    Number.isFinite(alpha) ? alpha : 0,
  );

  return {
    screenX: Math.round((left.screenX + right.screenX) / 2),
    r: left.r + (right.r - left.r) * clampedAlpha,
    g: left.g + (right.g - left.g) * clampedAlpha,
    b: left.b + (right.b - left.b) * clampedAlpha,
  };
}
