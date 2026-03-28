/**
 * generate-bench-tables.ts
 *
 * Reads the unified benchmark artifact (benchmark.results.json) and emits two markdown table
 * snippets to STDOUT for easy inclusion in Memory_Optimization.md:
 *  1. Variant Delta Table (timings + bytes/conn)
 *  2. Node Heap Metrics Table (heapUsed/rss)
 *
 * Usage:
 *   npm run bench:tables > bench_tables.md
 *
 * Design notes:
 *  - No external dependencies (keep execution lightweight & CI friendly).
 *  - Gracefully degrades when fields missing (older artifact schema).
 *  - Extend later for variance/regression annotation summaries.
 */
import fs from 'node:fs';
import path from 'node:path';

/** Canonical benchmark artifact consumed by the table generator. */
const BENCHMARK_ARTIFACT_PATH = path.resolve(
  'test/benchmarks/benchmark.results.json',
);

/** Metrics shown in the variant delta table. */
const DELTA_METRICS = ['buildMs', 'fwdAvgMs', 'bytesPerConn'] as const;

/** Metrics shown in the heap and resident-set table. */
const HEAP_METRICS = ['heapUsed', 'rss'] as const;

/** Full metric set recognized across both current and legacy artifact schemas. */
const AGGREGATED_METRICS = [...DELTA_METRICS, ...HEAP_METRICS] as const;

/** Benchmark variant identifiers emitted by the build-variant suite. */
type BenchmarkMode = 'src' | 'dist';

/** Metric names that participate in src-vs-dist regression comparison. */
type DeltaMetricName = (typeof DELTA_METRICS)[number];

/** Metric names that participate in the heap summary table. */
type HeapMetricName = (typeof HEAP_METRICS)[number];

/** All metric names recognized by the script. */
type BenchmarkMetricName = (typeof AGGREGATED_METRICS)[number];

/** Legacy flat-artifact key names such as `buildMsMean`. */
type LegacyMeanMetricKey = `${BenchmarkMetricName}Mean`;

/**
 * Aggregate statistics for one benchmark metric.
 *
 * The current table generator only consumes `mean`, but the wider schema can
 * also carry percentile and spread values that may be used by later reports.
 */
interface AggregatedMetric {
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
type AggregatedScenarioBucket = Partial<
  Record<BenchmarkMetricName, AggregatedMetric>
>;

/**
 * Size bucket keyed by scenario name.
 *
 * Example: `1024 -> all -> buildMs -> { mean: ... }`.
 */
type AggregatedSizeBucket = Record<
  string,
  AggregatedScenarioBucket | undefined
> & {
  all?: AggregatedScenarioBucket;
};

/** Aggregated metrics keyed by benchmark size. */
type AggregatedModeBucket = Record<string, AggregatedSizeBucket>;

/** Fully normalized aggregated results keyed by build variant. */
interface NormalizedAggregatedResults {
  src: AggregatedModeBucket;
  dist: AggregatedModeBucket;
}

/** Minimal artifact contract needed by this script. */
interface BenchmarkArtifact {
  aggregated?: unknown;
}

/**
 * Legacy array-based aggregated entry shape.
 *
 * Older benchmark artifacts stored one flat record per `(mode, size)` pair
 * using `fooMean` properties instead of the nested scenario bucket structure.
 */
interface LegacyAggregatedEntry extends Partial<
  Record<LegacyMeanMetricKey, number>
> {
  mode: BenchmarkMode;
  size: number | string;
}

/**
 * Pads or truncates a value to fixed width for the fenced monospace tables.
 *
 * @param v - Cell value to render.
 * @param w - Target cell width.
 * @returns Fixed-width cell content.
 */
function cell(v: string | number | undefined | null, w: number): string {
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
function fmtNum(n: number | undefined, digits = 4): string {
  if (typeof n !== 'number' || !Number.isFinite(n)) return '';
  return n.toFixed(digits).replace(/0+$/, '').replace(/\.$/, '');
}

/**
 * Loads the benchmark artifact from disk.
 *
 * @returns Parsed artifact when present and readable, otherwise `null`.
 */
function loadArtifact(): BenchmarkArtifact | null {
  if (!fs.existsSync(BENCHMARK_ARTIFACT_PATH)) {
    console.error(
      '[bench:tables] Artifact not found:',
      BENCHMARK_ARTIFACT_PATH,
    );
    return null;
  }

  try {
    return JSON.parse(
      fs.readFileSync(BENCHMARK_ARTIFACT_PATH, 'utf8'),
    ) as BenchmarkArtifact;
  } catch (error) {
    console.error('[bench:tables] Parse error', error);
    return null;
  }
}

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
function normalizeAggregated(
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
 * @param rawEntries - Legacy aggregated entries.
 * @returns Normalized aggregated result buckets.
 */
function normalizeLegacyAggregated(
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

    for (const metricName of AGGREGATED_METRICS) {
      const meanKey = `${metricName}Mean` as const;
      const meanValue = rawEntry[meanKey];
      if (typeof meanValue === 'number' && Number.isFinite(meanValue)) {
        scenarioBucket[metricName] = { mean: meanValue };
      }
    }
  }

  return aggregatedResults;
}

/**
 * Normalizes one top-level mode bucket such as `src` or `dist`.
 *
 * @param rawModeBucket - Unknown raw mode payload.
 * @returns Normalized size buckets.
 */
function normalizeModeBucket(rawModeBucket: unknown): AggregatedModeBucket {
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
 * Normalizes one size bucket keyed by scenario names.
 *
 * @param rawSizeBucket - Unknown raw size payload.
 * @returns Normalized scenario buckets.
 */
function normalizeSizeBucket(rawSizeBucket: unknown): AggregatedSizeBucket {
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
function normalizeScenarioBucket(
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
 * @param modeBucket - Normalized mode bucket.
 * @param sizeKey - Size identifier.
 * @returns Mutable `all` scenario bucket.
 */
function ensureAllScenarioBucket(
  modeBucket: AggregatedModeBucket,
  sizeKey: string,
): AggregatedScenarioBucket {
  modeBucket[sizeKey] ??= {};
  modeBucket[sizeKey].all ??= {};
  return modeBucket[sizeKey].all!;
}

/**
 * Creates an empty normalized results shell.
 *
 * @returns Empty `src` and `dist` mode buckets.
 */
function createEmptyAggregatedResults(): NormalizedAggregatedResults {
  return {
    src: {},
    dist: {},
  };
}

/**
 * Resolves size keys in ascending numeric order.
 *
 * @param aggregatedResults - Normalized aggregated result buckets.
 * @returns Ordered size keys.
 */
function resolveOrderedSizes(
  aggregatedResults: NormalizedAggregatedResults,
): string[] {
  return [...new Set(Object.keys(aggregatedResults.src))].toSorted(
    compareNumericSizeKeys,
  );
}

/**
 * Builds the variant delta table comparing `src` and `dist` means.
 *
 * The result is tuned for markdown fenced blocks rather than GFM pipe tables,
 * which keeps wide benchmark rows easier to scan in docs and pull requests.
 *
 * @param artifact - Parsed benchmark artifact.
 * @returns Monospace markdown table body.
 */
function buildVariantDeltaTable(artifact: BenchmarkArtifact): string {
  const aggregatedResults = normalizeAggregated(artifact);
  const orderedSizes = resolveOrderedSizes(aggregatedResults);
  const rows: string[] = [];
  rows.push(
    'Size    Metric            Src Mean      Dist Mean     Δ Abs         Δ %        Flag          Result',
  );
  rows.push(
    '------  ----------------- ------------- ------------- ------------- ---------- ------------- ------------------------------------------------------------',
  );
  for (const size of orderedSizes) {
    for (const metric of DELTA_METRICS) {
      const srcAgg = aggregatedResults.src[size]?.all?.[metric];
      const distAgg = aggregatedResults.dist[size]?.all?.[metric];
      const sMean = srcAgg?.mean;
      const dMean = distAgg?.mean;
      let dAbs: number | undefined;
      let dPct: number | undefined;
      if (typeof sMean === 'number' && typeof dMean === 'number') {
        dAbs = dMean - sMean;
        dPct = sMean === 0 ? undefined : (dAbs / sMean) * 100;
      }
      const threshold = metric === 'bytesPerConn' ? 3 : 5;
      const flag =
        typeof dPct === 'number'
          ? Math.abs(dPct) > threshold
            ? '(REG)'
            : 'OK'
          : '';
      const note =
        metric === 'bytesPerConn'
          ? flag === '(REG)'
            ? 'Dist bytes divergence'
            : 'Parity'
          : flag === '(REG)'
            ? `Dist ${dAbs! > 0 ? 'slower' : 'faster'} ${metric}`
            : 'Parity';
      rows.push(
        cell(size, 6) +
          '  ' +
          cell(metric + 'Mean', 17) +
          ' ' +
          cell(typeof sMean === 'number' ? fmtNum(sMean, 4) : '', 13) +
          ' ' +
          cell(typeof dMean === 'number' ? fmtNum(dMean, 4) : '', 13) +
          ' ' +
          cell(
            typeof dAbs === 'number'
              ? (dAbs >= 0 ? '+' : '') + fmtNum(dAbs, 4)
              : '',
            13,
          ) +
          ' ' +
          cell(
            typeof dPct === 'number'
              ? (dPct >= 0 ? '+' : '') + fmtNum(dPct, 2)
              : '',
            10,
          ) +
          ' ' +
          cell(flag, 13) +
          ' ' +
          note,
      );
    }
  }
  return rows.join('\n');
}

/**
 * Builds the heap and RSS comparison table.
 *
 * @param artifact - Parsed benchmark artifact.
 * @returns Monospace markdown table body.
 */
function buildHeapTable(artifact: BenchmarkArtifact): string {
  const aggregatedResults = normalizeAggregated(artifact);
  const orderedSizes = resolveOrderedSizes(aggregatedResults);
  const rows: string[] = [];
  rows.push(
    'Size    Metric         Src Bytes    Dist Bytes    Src MB    Dist MB   Δ Bytes      Δ %     Note',
  );
  rows.push(
    '------  -------------- ------------ ------------ --------- --------- ----------- ------- -------------------------------------------------',
  );
  for (const size of orderedSizes) {
    for (const metric of HEAP_METRICS) {
      const s = aggregatedResults.src[size]?.all?.[metric]?.mean;
      const d = aggregatedResults.dist[size]?.all?.[metric]?.mean;
      if (typeof s !== 'number' || typeof d !== 'number') continue;
      const dBytes = d - s;
      const dPct = s === 0 ? 0 : (dBytes / s) * 100;
      rows.push(
        cell(size, 6) +
          '  ' +
          cell(metric + 'Mean', 14) +
          ' ' +
          cell(s, 12) +
          ' ' +
          cell(d, 12) +
          ' ' +
          cell((s / 1048576).toFixed(1), 9) +
          ' ' +
          cell((d / 1048576).toFixed(1), 9) +
          ' ' +
          cell((dBytes >= 0 ? '+' : '') + dBytes, 11) +
          ' ' +
          cell((dPct >= 0 ? '+' : '') + dPct.toFixed(2), 7) +
          ' ' +
          'Informational',
      );
    }
  }
  return rows.join('\n');
}

/**
 * Compares size keys numerically, falling back to lexical comparison.
 *
 * @param left - Left size key.
 * @param right - Right size key.
 * @returns Standard comparator delta.
 */
function compareNumericSizeKeys(left: string, right: string): number {
  const leftValue = Number.parseInt(left, 10);
  const rightValue = Number.parseInt(right, 10);

  if (Number.isNaN(leftValue) || Number.isNaN(rightValue)) {
    return left.localeCompare(right);
  }

  return leftValue - rightValue || left.localeCompare(right);
}

/**
 * Narrows an unknown value to the legacy flat aggregated entry shape.
 *
 * @param value - Candidate legacy entry.
 * @returns `true` when the value matches the legacy schema.
 */
function isLegacyAggregatedEntry(
  value: unknown,
): value is LegacyAggregatedEntry {
  return (
    isRecord(value) &&
    (value.mode === 'src' || value.mode === 'dist') &&
    (typeof value.size === 'number' || typeof value.size === 'string')
  );
}

/**
 * Narrows an unknown value to a plain object record.
 *
 * @param value - Candidate value.
 * @returns `true` when the value is a non-array object.
 */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

/**
 * Runs the artifact load and emits both markdown tables to standard output.
 *
 * @returns Nothing.
 */
function main() {
  const artifact = loadArtifact();
  if (!artifact) process.exit(1);
  const out = [
    '```',
    buildVariantDeltaTable(artifact),
    '```',
    '',
    '```',
    buildHeapTable(artifact),
    '```',
  ].join('\n');
  console.log(out);
}

main();
