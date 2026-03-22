import type Group from '../group/group';

const FUNCTION_TYPE_NAME = 'function';

/**
 * Checks whether an unknown value is group-like.
 *
 * This is a structural runtime guard used by layer helpers that must safely
 * operate on mixed node/group collections.
 *
 * @param candidate - The value to inspect.
 * @returns True when the value exposes group-like members.
 *
 * Example:
 *
 * ```ts
 * if (isGroup(value)) {
 *   value.set({ bias: 0 });
 * }
 * ```
 */
export function isGroup(candidate: unknown): candidate is Group {
  const typedCandidate = candidate as { set?: unknown; nodes?: unknown };
  return (
    !!candidate &&
    typeof typedCandidate.set === FUNCTION_TYPE_NAME &&
    Array.isArray(typedCandidate.nodes)
  );
}
