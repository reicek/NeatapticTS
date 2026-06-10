/**
 * generate-bench-tables.types.ts
 *
 * All shared type aliases and interfaces for the benchmark table generator.
 * These types model both the current nested artifact schema and the older
 * legacy flat array schema so that the normalizer and table builders can
 * remain schema-agnostic.
 */

import type {
  AGGREGATED_METRICS,
  DELTA_METRICS,
  HEAP_METRICS,
} from './generate-bench-tables.constants.js';

/** String literal union of the two build variant identifiers emitted by the benchmark suite. */
export type BenchmarkMode = 'src' | 'dist';

/** String literal union of timing and memory metrics used in the variant delta regression comparison table. */
export type DeltaMetricName = (typeof DELTA_METRICS)[number];

/** String literal union of heap and RSS metric names shown in the heap memory comparison table. */
export type HeapMetricName = (typeof HEAP_METRICS)[number];

/** Union of all metric names recognized by this script, spanning both the delta and heap table buckets. */
export type BenchmarkMetricName = (typeof AGGREGATED_METRICS)[number];

/** Template literal type producing legacy flat-artifact key names by appending `Mean` to each benchmark metric name. */
export type LegacyMeanMetricKey = `${BenchmarkMetricName}Mean`;

/**
 * Aggregate statistics for one benchmark metric.
 *
 * The current table generator only consumes `mean`, but the wider schema can
 * also carry percentile and spread values that may be used by later reports.
 */
export interface AggregatedMetric {
  mean?: number;
  p50?: number;
  p95?: number;
  std?: number;
}

/**
 * Scenario-level metrics under one size bucket.
 *
 * The script currently reads the `all` scenario, but the normalized shape keeps
 * the scenario layer so future reporting can distinguish specialized runs.
 */
export type AggregatedScenarioBucket = Partial<
  Record<BenchmarkMetricName, AggregatedMetric>
>;

/**
 * Size bucket keyed by scenario name.
 *
 * Example: `1024 -> all -> buildMs -> { mean: ... }`.
 */
export type AggregatedSizeBucket = Record<
  string,
  AggregatedScenarioBucket | undefined
> & {
  all?: AggregatedScenarioBucket;
};

/** Map from benchmark size string to a size bucket containing scenario-level aggregated metrics per variant. */
export type AggregatedModeBucket = Record<string, AggregatedSizeBucket>;

/** Root container holding the fully-normalized `src` and `dist` mode buckets produced by the normalizer. */
export interface NormalizedAggregatedResults {
  src: AggregatedModeBucket;
  dist: AggregatedModeBucket;
}

/**
 * Minimal structural contract for the benchmark results artifact consumed from disk.
 *
 * The `aggregated` field is typed as `unknown` so the normalizer can safely
 * narrow it to either the current nested schema or the legacy flat array schema.
 */
export interface BenchmarkArtifact {
  aggregated?: unknown;
}

/**
 * Legacy array-based aggregated entry shape.
 *
 * Older benchmark artifacts stored one flat record per `(mode, size)` pair
 * using `fooMean` properties instead of the nested scenario bucket structure.
 */
export interface LegacyAggregatedEntry extends Partial<
  Record<LegacyMeanMetricKey, number>
> {
  mode: BenchmarkMode;
  size: number | string;
}
