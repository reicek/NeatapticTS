/**
 * Deep-clone a value with a native structuredClone fast path and a JSON fallback.
 *
 * Use this helper when the runtime baseline may expose `structuredClone`, but a
 * plain-data fallback is still required for older or constrained environments.
 * The JSON path intentionally preserves the project's previous fail-fast
 * behavior for non-serializable values.
 *
 * @param value - Value to clone.
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
