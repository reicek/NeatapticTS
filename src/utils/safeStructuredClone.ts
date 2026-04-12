/**
 * Deep-clone a value with a native structuredClone fast path and a JSON fallback.
 *
 * Use this helper when the runtime baseline may expose `structuredClone`, but a
 * plain-data fallback is still required for older or constrained environments.
 *
 * Behavior of the JSON fallback path:
 * - **Throws** on circular references and `BigInt` values (JSON serialization error).
 * - **Lossy** for `undefined`, functions, and symbols — these are silently
 *   dropped or coerced to `null` by `JSON.stringify`, so callers should not
 *   rely on strict value preservation through the fallback path.
 *
 * @param value Value to clone.
 * @returns Deep-cloned value.
 *
 * @example
 * ```ts
 * const cloned = safeStructuredClone({ nested: { count: 1 } });
 * ```
 */
export function safeStructuredClone<T>(value: T): T {
  try {
    return typeof globalThis.structuredClone === 'function'
      ? globalThis.structuredClone(value)
      : JSON.parse(JSON.stringify(value));
  } catch {
    return JSON.parse(JSON.stringify(value));
  }
}
