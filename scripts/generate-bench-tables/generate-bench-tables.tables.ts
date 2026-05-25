/**
 * generate-bench-tables.tables.ts
 *
 * Markdown table builders for the benchmark table generator.
 *
 * Each builder owns one table's header, separator, and row-generation loop.
 * Tables use fenced monospace blocks rather than GFM pipe tables so that wide
 * benchmark rows remain easy to scan in docs and pull-request diffs.
 */

import { DELTA_METRICS, HEAP_METRICS } from './generate-bench-tables.constants.js';
import { cell, fmtNum, resolveOrderedSizes } from './generate-bench-tables.format.js';
import { normalizeAggregated } from './generate-bench-tables.normalize.js';
import type {
  BenchmarkArtifact,
  DeltaMetricName,
  NormalizedAggregatedResults,
} from './generate-bench-tables.types.js';

const DELTA_TABLE_HEADER =
  'Size    Metric            Src Mean      Dist Mean     Δ Abs         Δ %        Flag          Result';
const DELTA_TABLE_SEPARATOR =
  '------  ----------------- ------------- ------------- ------------- ---------- ------------- ------------------------------------------------------------';
const HEAP_TABLE_HEADER =
  'Size    Metric         Src Bytes    Dist Bytes    Src MB    Dist MB   Δ Bytes      Δ %     Note';
const HEAP_TABLE_SEPARATOR =
  '------  -------------- ------------ ------------ --------- --------- ----------- ------- -------------------------------------------------';

/**
 * Per-metric regression thresholds used when classifying src-vs-dist deltas.
 *
 * `bytesPerConn` uses a tighter 3 % gate because byte regressions compound
 * across connections; all other metrics use the looser 5 % gate.
 */
const DELTA_METRIC_THRESHOLDS: Readonly<Record<DeltaMetricName, number>> = {
  bytesPerConn: 3,
  buildMs: 5,
  fwdAvgMs: 5,
};

/** Computed src-vs-dist delta statistics for one metric cell. */
interface DeltaStatistics {
  dAbs: number | undefined;
  dPct: number | undefined;
}

/**
 * Derives absolute and percentage deltas between two optional mean values.
 *
 * Returns `undefined` for both fields when either mean is absent, preserving
 * the distinction between a missing measurement and a zero-delta measurement.
 *
 * @param sMean - Source build mean value.
 * @param dMean - Dist build mean value.
 * @returns Computed absolute and percentage delta statistics.
 */
function computeDeltaStatistics(
  sMean: number | undefined,
  dMean: number | undefined,
): DeltaStatistics {
  if (typeof sMean !== 'number' || typeof dMean !== 'number') {
    return { dAbs: undefined, dPct: undefined };
  }
  const dAbs = dMean - sMean;
  const dPct = sMean === 0 ? undefined : (dAbs / sMean) * 100;
  return { dAbs, dPct };
}

/**
 * Classifies a percentage delta as a regression flag or an OK label.
 *
 * Returns an empty string when the delta is absent, which happens when one or
 * both means are missing from the artifact for this size bucket.
 *
 * @param dPct - Percentage delta between src and dist means.
 * @param threshold - Regression percentage threshold for the metric.
 * @returns `'(REG)'`, `'OK'`, or `''` based on the delta magnitude.
 */
function computeDeltaFlag(dPct: number | undefined, threshold: number): string {
  if (typeof dPct !== 'number') return '';
  return Math.abs(dPct) > threshold ? '(REG)' : 'OK';
}

/**
 * Produces a human-readable regression note for one delta table row.
 *
 * The note distinguishes bytes-divergence regressions from timing regressions
 * and labels the direction (slower/faster) for timing metrics.
 *
 * @param metric - Benchmark metric name for the current row.
 * @param flag - Regression flag produced by `computeDeltaFlag`.
 * @param dAbs - Absolute delta used to determine direction for timing metrics.
 * @returns Human-readable note string for the Result column.
 */
function computeDeltaNote(
  metric: DeltaMetricName,
  flag: string,
  dAbs: number | undefined,
): string {
  if (metric === 'bytesPerConn') {
    return flag === '(REG)' ? 'Dist bytes divergence' : 'Parity';
  }
  if (flag === '(REG)') {
    return `Dist ${(dAbs ?? 0) > 0 ? 'slower' : 'faster'} ${metric}`;
  }
  return 'Parity';
}

/**
 * Assembles one formatted row for the variant delta table.
 *
 * All numeric fields are padded with {@link cell} so columns stay aligned in
 * the fenced monospace block regardless of value magnitude.
 *
 * @param size - Benchmark size bucket identifier.
 * @param metric - Metric name for the row.
 * @param sMean - Source build mean value.
 * @param dMean - Dist build mean value.
 * @param dAbs - Absolute delta between the two means.
 * @param dPct - Percentage delta between the two means.
 * @param flag - Regression flag string.
 * @param note - Result note string.
 * @returns Fixed-width table row.
 */
function buildDeltaTableRow(
  size: string,
  metric: DeltaMetricName,
  sMean: number | undefined,
  dMean: number | undefined,
  dAbs: number | undefined,
  dPct: number | undefined,
  flag: string,
  note: string,
): string {
  const sMeanCell = typeof sMean === 'number' ? fmtNum(sMean, 4) : '';
  const dMeanCell = typeof dMean === 'number' ? fmtNum(dMean, 4) : '';
  const dAbsCell =
    typeof dAbs === 'number' ? (dAbs >= 0 ? '+' : '') + fmtNum(dAbs, 4) : '';
  const dPctCell =
    typeof dPct === 'number' ? (dPct >= 0 ? '+' : '') + fmtNum(dPct, 2) : '';

  return (
    cell(size, 6) +
    '  ' +
    cell(metric + 'Mean', 17) +
    ' ' +
    cell(sMeanCell, 13) +
    ' ' +
    cell(dMeanCell, 13) +
    ' ' +
    cell(dAbsCell, 13) +
    ' ' +
    cell(dPctCell, 10) +
    ' ' +
    cell(flag, 13) +
    ' ' +
    note
  );
}

/**
 * Assembles one formatted row for the heap and RSS comparison table.
 *
 * Both raw byte values and megabyte conversions are rendered side-by-side so
 * the reader can spot proportional and absolute regressions simultaneously.
 *
 * @param size - Benchmark size bucket identifier.
 * @param metric - Heap metric name for the row.
 * @param srcMean - Source build mean byte value.
 * @param distMean - Dist build mean byte value.
 * @returns Fixed-width table row.
 */
function buildHeapTableRow(
  size: string,
  metric: string,
  srcMean: number,
  distMean: number,
): string {
  const dBytes = distMean - srcMean;
  const dPct = srcMean === 0 ? 0 : (dBytes / srcMean) * 100;

  return (
    cell(size, 6) +
    '  ' +
    cell(metric + 'Mean', 14) +
    ' ' +
    cell(srcMean, 12) +
    ' ' +
    cell(distMean, 12) +
    ' ' +
    cell((srcMean / 1048576).toFixed(1), 9) +
    ' ' +
    cell((distMean / 1048576).toFixed(1), 9) +
    ' ' +
    cell((dBytes >= 0 ? '+' : '') + dBytes, 11) +
    ' ' +
    cell((dPct >= 0 ? '+' : '') + dPct.toFixed(2), 7) +
    ' ' +
    'Informational'
  );
}

/**
 * Emits all delta rows for one size bucket into the row accumulator.
 *
 * Extracted to keep the outer size-loop body in {@link buildVariantDeltaTable}
 * free of per-metric conditional logic.
 *
 * @param size - Benchmark size bucket identifier.
 * @param aggregatedResults - Normalized results containing src and dist buckets.
 * @param rows - Mutable row accumulator.
 * @returns Nothing; rows are appended in-place.
 */
function appendDeltaRowsForSize(
  size: string,
  aggregatedResults: NormalizedAggregatedResults,
  rows: string[],
): void {
  for (const metric of DELTA_METRICS) {
    const srcMean = aggregatedResults.src[size]?.all?.[metric]?.mean;
    const distMean = aggregatedResults.dist[size]?.all?.[metric]?.mean;
    const threshold = DELTA_METRIC_THRESHOLDS[metric];
    const { dAbs, dPct } = computeDeltaStatistics(srcMean, distMean);
    const flag = computeDeltaFlag(dPct, threshold);
    const note = computeDeltaNote(metric, flag, dAbs);
    rows.push(buildDeltaTableRow(size, metric, srcMean, distMean, dAbs, dPct, flag, note));
  }
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
export function buildVariantDeltaTable(artifact: BenchmarkArtifact): string {
  const aggregatedResults = normalizeAggregated(artifact);
  const orderedSizes = resolveOrderedSizes(aggregatedResults);
  const rows = [DELTA_TABLE_HEADER, DELTA_TABLE_SEPARATOR];
  for (const size of orderedSizes) {
    appendDeltaRowsForSize(size, aggregatedResults, rows);
  }
  return rows.join('\n');
}

/**
 * Builds the heap and RSS comparison table for all sizes and heap metrics.
 *
 * Assembles the heap and resident-set memory comparison table comparing `src`
 * and `dist` means side-by-side as both raw bytes and megabytes.
 *
 * @param artifact - Parsed benchmark artifact.
 * @returns Monospace markdown table body.
 */
export function buildHeapTable(artifact: BenchmarkArtifact): string {
  const aggregatedResults = normalizeAggregated(artifact);
  const orderedSizes = resolveOrderedSizes(aggregatedResults);
  const rows = [HEAP_TABLE_HEADER, HEAP_TABLE_SEPARATOR];
  for (const size of orderedSizes) {
    for (const metric of HEAP_METRICS) {
      const srcMean = aggregatedResults.src[size]?.all?.[metric]?.mean;
      const distMean = aggregatedResults.dist[size]?.all?.[metric]?.mean;
      if (typeof srcMean !== 'number' || typeof distMean !== 'number') continue;
      rows.push(buildHeapTableRow(size, metric, srcMean, distMean));
    }
  }
  return rows.join('\n');
}
