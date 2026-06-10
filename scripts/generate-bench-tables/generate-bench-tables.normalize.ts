/**
 * generate-bench-tables.normalize.ts
 *
 * Schema normalization for the benchmark table generator.
 *
 * The benchmark pipeline has emitted two distinct artifact schemas over time:
 *  - Current: nested `mode → size → scenario → metric → { mean, ... }` object.
 *  - Legacy:  flat array of `{ mode, size, fooMean, ... }` records.
 *
 * Both are folded here into one stable `NormalizedAggregatedResults` shape so
 * that the table builders remain schema-agnostic and easy to extend.
 */

import { AGGREGATED_METRICS } from './generate-bench-tables.constants.js';
import type {
  AggregatedModeBucket,
  AggregatedScenarioBucket,
  AggregatedSizeBucket,
  BenchmarkArtifact,
  LegacyAggregatedEntry,
  NormalizedAggregatedResults,
} from './generate-bench-tables.types.js';

/**
 * Normalizes the artifact's aggregated payload into one stable nested shape.
 *
 * The benchmark pipeline has emitted both nested and legacy flat array schemas.
 * Folding both into the same mode/size/scenario/metric structure keeps the
 * table builders simple and resilient.
 *
 * @param artifact - Parsed benchmark artifact.
 * @returns Normalized aggregated result buckets.
 */
export function normalizeAggregated(
  artifact: BenchmarkArtifact,
): NormalizedAggregatedResults {
  if (Array.isArray(artifact.aggregated)) {
    return normalizeLegacyAggregated(artifact.aggregated);
  }

  if (!isRecord(artifact.aggregated)) {
    return createEmptyAggregatedResults();
  }

  return {
    src: normalizeModeBucket(artifact.aggregated.src),
    dist: normalizeModeBucket(artifact.aggregated.dist),
  };
}

/**
 * Converts the older flat array schema into the normalized nested shape.
 *
 * Iterates the legacy entries, skipping malformed records, and delegates
 * per-metric extraction to {@link applyLegacyEntryMetrics} to keep the outer
 * loop body below the complexity threshold.
 *
 * @param rawEntries - Legacy aggregated entries from the old flat array schema.
 * @returns Normalized aggregated result buckets matching the current shape.
 */
export function normalizeLegacyAggregated(
  rawEntries: readonly unknown[],
): NormalizedAggregatedResults {
  const aggregatedResults = createEmptyAggregatedResults();

  for (const rawEntry of rawEntries) {
    if (!isLegacyAggregatedEntry(rawEntry)) {
      continue;
    }

    const sizeKey = String(rawEntry.size);
    const scenarioBucket = ensureAllScenarioBucket(
      aggregatedResults[rawEntry.mode],
      sizeKey,
    );

    applyLegacyEntryMetrics(rawEntry, scenarioBucket);
  }

  return aggregatedResults;
}

/**
 * Copies finite mean values from a legacy flat entry into a scenario bucket.
 *
 * Iterates every recognized metric name, reads the corresponding `fooMean`
 * property from the legacy entry, and writes a normalized `{ mean }` object
 * when the value is a finite number. Non-finite and absent values are skipped.
 *
 * @param rawEntry - Validated legacy aggregated entry.
 * @param scenarioBucket - Mutable scenario bucket to populate.
 * @returns Nothing; the bucket is mutated in-place.
 */
function applyLegacyEntryMetrics(
  rawEntry: LegacyAggregatedEntry,
  scenarioBucket: AggregatedScenarioBucket,
): void {
  for (const metricName of AGGREGATED_METRICS) {
    const meanKey = `${metricName}Mean` as const;
    const meanValue = rawEntry[meanKey];
    if (typeof meanValue === 'number' && Number.isFinite(meanValue)) {
      scenarioBucket[metricName] = { mean: meanValue };
    }
  }
}

/**
 * Normalizes one top-level mode bucket such as `src` or `dist`.
 *
 * @param rawModeBucket - Unknown raw mode payload.
 * @returns Normalized size buckets.
 */
export function normalizeModeBucket(
  rawModeBucket: unknown,
): AggregatedModeBucket {
  if (!isRecord(rawModeBucket)) {
    return {};
  }

  return Object.fromEntries(
    Object.entries(rawModeBucket).map(([sizeKey, rawSizeBucket]) => [
      sizeKey,
      normalizeSizeBucket(rawSizeBucket),
    ]),
  );
}

/**
 * Converts a raw unknown size payload into a map of scenario names to
 * normalized scenario metric buckets, dropping non-object entries.
 *
 * @param rawSizeBucket - Unknown raw size payload from the artifact.
 * @returns Normalized map of scenario names to scenario metric buckets.
 */
export function normalizeSizeBucket(
  rawSizeBucket: unknown,
): AggregatedSizeBucket {
  if (!isRecord(rawSizeBucket)) {
    return {};
  }

  return Object.fromEntries(
    Object.entries(rawSizeBucket).map(([scenarioName, rawScenarioBucket]) => [
      scenarioName,
      normalizeScenarioBucket(rawScenarioBucket),
    ]),
  ) as AggregatedSizeBucket;
}

/**
 * Normalizes one scenario bucket keyed by metric name.
 *
 * Invalid or incomplete metric objects are dropped rather than guessed.
 *
 * @param rawScenarioBucket - Unknown raw scenario payload.
 * @returns Normalized metric bucket.
 */
export function normalizeScenarioBucket(
  rawScenarioBucket: unknown,
): AggregatedScenarioBucket {
  if (!isRecord(rawScenarioBucket)) {
    return {};
  }

  const normalizedEntries = AGGREGATED_METRICS.flatMap((metricName) => {
    const rawMetric = rawScenarioBucket[metricName];
    if (!isRecord(rawMetric)) {
      return [];
    }

    const meanValue = rawMetric.mean;
    if (typeof meanValue !== 'number' || !Number.isFinite(meanValue)) {
      return [];
    }

    return [[metricName, { mean: meanValue }] as const];
  });

  return Object.fromEntries(normalizedEntries) as AggregatedScenarioBucket;
}

/**
 * Ensures the `all` scenario exists for one `(mode, size)` pair.
 *
 * The `??=` assignment is intentional: it mutates the shared mode bucket
 * in-place during legacy array normalization, which is the only context where
 * the bucket may not yet have an entry for the given size key.
 *
 * @param modeBucket - Normalized mode bucket.
 * @param sizeKey - Size identifier.
 * @returns Mutable `all` scenario bucket.
 */
export function ensureAllScenarioBucket(
  modeBucket: AggregatedModeBucket,
  sizeKey: string,
): AggregatedScenarioBucket {
  modeBucket[sizeKey] ??= {};
  modeBucket[sizeKey].all ??= {};
  return modeBucket[sizeKey].all!;
}

/**
 * Allocates an empty `NormalizedAggregatedResults` with zero-entry `src` and
 * `dist` mode buckets, used as the starting state for both normalization paths.
 *
 * @returns Fresh empty normalized results container.
 */
export function createEmptyAggregatedResults(): NormalizedAggregatedResults {
  return {
    src: {},
    dist: {},
  };
}

/**
 * Narrows an unknown value to the legacy flat aggregated entry shape.
 *
 * @param value - Candidate legacy entry.
 * @returns `true` when the value matches the legacy schema.
 */
export function isLegacyAggregatedEntry(
  value: unknown,
): value is LegacyAggregatedEntry {
  return (
    isRecord(value) &&
    (value.mode === 'src' || value.mode === 'dist') &&
    (typeof value.size === 'number' || typeof value.size === 'string')
  );
}

/**
 * Returns `true` when the value is a non-null, non-array object, narrowing it
 * to a plain key-value record safe for property access.
 *
 * @param value - Candidate value of unknown shape.
 * @returns `true` when the value qualifies as a plain object record.
 */
export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}
