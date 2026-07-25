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

/**
 * Object shape accepted by {@link lerpNeatensteinState}.
 *
 * The generic state is expected to expose own enumerable string keys whose
 * values are finite numbers at runtime.
 */
type NeatensteinNumericState = Record<string, number>;

/** Lower bound for interpolation alpha. */
const MIN_INTERPOLATION_ALPHA = 0;

/** Upper bound for interpolation alpha. */
const MAX_INTERPOLATION_ALPHA = 1;

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
 * Read and validate a numeric field from a snapshot.
 *
 * TypeScript enforces numeric fields at compile time, but renderer code still
 * runs against JavaScript values at runtime. This guard prevents `NaN`,
 * infinities, missing fields, and malformed state from silently contaminating
 * the interpolated render state.
 *
 * @param snapshotName - Human-readable snapshot name for error messages.
 * @param key - Field key being read.
 * @param value - Runtime field value.
 * @returns The validated finite number.
 * @throws {TypeError} When the value is not a finite number.
 */
function readFiniteSnapshotNumber(
  snapshotName: string,
  key: string,
  value: unknown,
): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    throw new TypeError(
      `${snapshotName}.${key} must be a finite number, got ${String(value)}`,
    );
  }

  return value;
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
 *
 * @param previous - Snapshot from the previous simulation tick.
 * @param current - Snapshot from the current simulation tick.
 * @param alpha - Finite blend factor. Values outside `[0, 1]` are clamped.
 * @returns A new state object with copied or interpolated scalar fields.
 * @throws {TypeError} When `alpha` or a processed snapshot field is not finite.
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
    return copyFiniteSnapshot(previous, 'previous') as T;
  }

  if (clampedAlpha >= MAX_INTERPOLATION_ALPHA) {
    return copyFiniteSnapshot(current, 'current') as T;
  }

  const result: NeatensteinNumericState = {};
  const keys = collectInterpolationKeys(previous, current);

  for (const key of keys) {
    const hasPreviousValue = hasOwnEnumerableKey(previous, key);
    const hasCurrentValue = hasOwnEnumerableKey(current, key);

    if (hasPreviousValue && hasCurrentValue) {
      // Normal path: both snapshots have the field, so blend linearly.
      const from = readFiniteSnapshotNumber('previous', key, previous[key]);
      const to = readFiniteSnapshotNumber('current', key, current[key]);

      result[key] = from + (to - from) * clampedAlpha;
      continue;
    }

    if (hasCurrentValue) {
      // The field was added in the current snapshot. There is no previous value
      // to interpolate from, so use the current finite value directly.
      result[key] = readFiniteSnapshotNumber('current', key, current[key]);
      continue;
    }

    if (hasPreviousValue) {
      // The field was removed in the current snapshot. Preserve the previous
      // finite value so the result remains stable for this render frame.
      result[key] = readFiniteSnapshotNumber('previous', key, previous[key]);
    }
  }

  return result as T;
}
