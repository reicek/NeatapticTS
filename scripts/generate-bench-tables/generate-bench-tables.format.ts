/**
 * generate-bench-tables.format.ts
 *
 * Formatting utilities for the benchmark table generator.
 *
 * All functions here are pure — they take scalar inputs and return strings or
 * numbers with no side effects, making them straightforward to test in isolation.
 */

import type { NormalizedAggregatedResults } from './generate-bench-tables.types.js';

/**
 * Pads or truncates a value to fixed width for the fenced monospace tables.
 *
 * @param v - Cell value to render.
 * @param w - Target cell width.
 * @returns Fixed-width cell content.
 */
export function cell(v: string | number | undefined | null, w: number): string {
  const s = String(v ?? '');
  return s.length >= w ? s.slice(0, w) : s + ' '.repeat(w - s.length);
}

/**
 * Formats a numeric value with trimmed trailing zeros.
 *
 * This keeps markdown tables compact without losing the ability to request
 * higher precision for smaller benchmark values.
 *
 * @param n - Candidate numeric value.
 * @param digits - Maximum fixed decimal precision.
 * @returns Formatted number or an empty string when the input is not finite.
 */
export function fmtNum(n: number | undefined, digits = 4): string {
  if (typeof n !== 'number' || !Number.isFinite(n)) return '';
  return n.toFixed(digits).replace(/0+$/, '').replace(/\.$/, '');
}

/**
 * Compares size keys numerically, falling back to lexical comparison.
 *
 * The fallback ensures a stable sort order even when size keys contain
 * non-numeric suffixes or are entirely non-numeric strings.
 *
 * @param left - Left size key.
 * @param right - Right size key.
 * @returns Standard comparator delta.
 */
export function compareNumericSizeKeys(left: string, right: string): number {
  const leftValue = Number.parseInt(left, 10);
  const rightValue = Number.parseInt(right, 10);

  if (Number.isNaN(leftValue) || Number.isNaN(rightValue)) {
    return left.localeCompare(right);
  }

  return leftValue - rightValue || left.localeCompare(right);
}

/**
 * Resolves size keys in ascending numeric order.
 *
 * Deduplication via `Set` guards against a size key appearing in `src` but
 * not `dist` (or vice versa) without double-counting.
 *
 * @param aggregatedResults - Normalized aggregated result buckets.
 * @returns Ordered size keys.
 */
export function resolveOrderedSizes(
  aggregatedResults: NormalizedAggregatedResults,
): string[] {
  return [...new Set(Object.keys(aggregatedResults.src))].toSorted(
    compareNumericSizeKeys,
  );
}
