/**
 * generate-bench-tables.constants.ts
 *
 * Named constants for the benchmark table generator.
 * Centralizing these avoids magic strings scattered across normalizers and
 * table builders, and lets TypeScript derive the metric union types from the
 * arrays directly via `(typeof X)[number]`.
 */

import path from 'node:path';

/**
 * Absolute path to the canonical benchmark artifact consumed by this script.
 *
 * Resolved relative to the process working directory so the script works
 * regardless of which directory it is launched from, as long as the CWD is
 * the repository root.
 */
export const BENCHMARK_ARTIFACT_PATH = path.resolve(
  'benchmarks/benchmark.results.json',
);

/** Ordered tuple of metric names rendered in the src-vs-dist variant delta regression comparison table. */
export const DELTA_METRICS = ['buildMs', 'fwdAvgMs', 'bytesPerConn'] as const;

/** Ordered tuple of metric names rendered in the heap and resident-set memory comparison table. */
export const HEAP_METRICS = ['heapUsed', 'rss'] as const;

/** Full metric set recognized across both current and legacy artifact schemas. */
export const AGGREGATED_METRICS = [...DELTA_METRICS, ...HEAP_METRICS] as const;
