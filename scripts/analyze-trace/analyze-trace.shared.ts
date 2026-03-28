import type { Primitive } from './analyze-trace.types.js';

/**
 * Safely resolves a nested primitive value from a trace event args object.
 *
 * @param value - Candidate object tree.
 * @param pathSegments - Nested path segments to follow.
 * @returns Primitive value when the path resolves cleanly.
 */
export function readNestedPrimitive(
  value: unknown,
  pathSegments: readonly string[],
): Primitive {
  let currentValue: unknown = value;

  for (const pathSegment of pathSegments) {
    if (!currentValue || typeof currentValue !== 'object') {
      return undefined;
    }

    currentValue = (currentValue as Record<string, unknown>)[pathSegment];
  }

  if (
    currentValue == null ||
    typeof currentValue === 'string' ||
    typeof currentValue === 'number' ||
    typeof currentValue === 'boolean'
  ) {
    return currentValue;
  }

  return undefined;
}

/**
 * Formats a duration in milliseconds for stable textual output.
 *
 * @param durationMs - Duration in milliseconds.
 * @returns Fixed two-decimal duration string.
 */
export function formatMs(durationMs: number): string {
  return `${durationMs.toFixed(2)}ms`;
}

/**
 * Computes a nearest-rank percentile from an already sorted numeric array.
 *
 * @param sortedDurations - Sorted numeric sample set.
 * @param rank - Percentile rank in the inclusive range `[0, 1]`.
 * @returns Percentile value or `0` when the list is empty.
 */
export function percentile(
  sortedDurations: readonly number[],
  rank: number,
): number {
  if (sortedDurations.length === 0) {
    return 0;
  }

  const clampedRank = Math.max(0, Math.min(1, rank));
  const index = Math.min(
    sortedDurations.length - 1,
    Math.floor(clampedRank * (sortedDurations.length - 1)),
  );

  return sortedDurations[index];
}

/**
 * Formats a compact percentile summary for one ascending-sorted duration collection.
 *
 * @param sortedDurationsMs - Ascending-sorted duration samples in milliseconds.
 * @returns Human-readable summary string.
 */
export function formatDistribution(
  sortedDurationsMs: readonly number[],
): string {
  if (sortedDurationsMs.length === 0) {
    return 'count=0';
  }

  return [
    `count=${sortedDurationsMs.length.toLocaleString()}`,
    `p50=${formatMs(percentile(sortedDurationsMs, 0.5))}`,
    `p90=${formatMs(percentile(sortedDurationsMs, 0.9))}`,
    `p99=${formatMs(percentile(sortedDurationsMs, 0.99))}`,
    `max=${formatMs(sortedDurationsMs.at(-1) ?? 0)}`,
  ].join(', ');
}

/**
 * Creates a stable lookup key for one process id.
 *
 * @param pid - Process identifier from the trace.
 * @returns Stable process key when the id exists.
 */
export function createProcessKey(pid: number | undefined): string | undefined {
  return typeof pid === 'number' ? `pid:${pid}` : undefined;
}

/**
 * Creates a stable lookup key for one process/thread pair.
 *
 * @param pid - Process identifier from the trace.
 * @param tid - Thread identifier from the trace.
 * @returns Stable thread key when both ids exist.
 */
export function createThreadKey(
  pid: number | undefined,
  tid: number | undefined,
): string | undefined {
  return typeof pid === 'number' && typeof tid === 'number'
    ? `pid:${pid}:tid:${tid}`
    : undefined;
}
