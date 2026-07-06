/**
 * @module eval-baseline
 * @description Baseline storage and regression detection for the Repo Cortex
 * advanced RAG evaluation suite.
 *
 * Stores reference measurements under `data/eval-baselines/` and compares
 * current eval runs against them to produce FAIL/WARN/PASS gate status.
 *
 * @example
 * ```js
 * import { storeBaseline, compareToBaseline } from './eval-baseline.mjs';
 *
 * const baseline = await storeBaseline({
 *   baseline_id: 'baseline-2026-06-15',
 *   conditions: { hybrid: { mrr_at_5: 0.32 } },
 * });
 *
 * const report = compareToBaseline(
 *   { condition: 'hybrid', metrics: { mrr_at_5: 0.30 } },
 *   { condition: 'hybrid', metrics: { mrr_at_5: 0.32 } },
 *   { failThresholds: { mrr_at_5: 0.01 } },
 * );
 * ```
 */

import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { repoRoot } from './init-schema.mjs';

/** Default directory for baseline JSON files. */
const DEFAULT_BASELINE_DIR = path.join(repoRoot, 'data', 'eval-baselines');

/**
 * Resolve the file path for a baseline.
 *
 * @param {string} baselineId
 * @param {string} [customPath]
 * @returns {string}
 */
function resolveBaselinePath(baselineId, customPath) {
  if (customPath) {
    return path.isAbsolute(customPath)
      ? customPath
      : path.join(repoRoot, customPath);
  }
  return path.join(DEFAULT_BASELINE_DIR, `${baselineId}.json`);
}

/**
 * Store a baseline measurement to disk.
 *
 * @param {object} data - Baseline payload with `baseline_id` and `conditions`.
 * @param {object} [options={}]
 * @param {string} [options.path] - Optional file path override.
 * @returns {Promise<object>} The stored baseline.
 * @throws {Error} When `baseline_id` is missing.
 */
export async function storeBaseline(data, options = {}) {
  if (
    !data ||
    typeof data.baseline_id !== 'string' ||
    data.baseline_id === ''
  ) {
    throw new Error('baseline_id is required to store a baseline.');
  }

  const baseline = {
    ...data,
  };
  const filePath = resolveBaselinePath(baseline.baseline_id, options.path);
  await mkdir(path.dirname(filePath), { recursive: true });
  await writeFile(filePath, JSON.stringify(baseline, null, 2), 'utf8');
  return baseline;
}

/**
 * Load a stored baseline by its identifier.
 *
 * @param {string} baselineId
 * @param {object} [options={}]
 * @param {string} [options.path] - Optional file path override.
 * @returns {Promise<object>}
 * @throws {Error} When the baseline file cannot be read or parsed.
 */
export async function loadBaseline(baselineId, options = {}) {
  if (typeof baselineId !== 'string' || baselineId === '') {
    throw new Error('baseline_id is required to load a baseline.');
  }
  const filePath = resolveBaselinePath(baselineId, options.path);
  const text = await readFile(filePath, 'utf8');
  return JSON.parse(text);
}

/**
 * Format a numeric delta as a signed, fixed string.
 *
 * @param {number} delta
 * @returns {string}
 */
function formatDelta(delta) {
  const sign = delta >= 0 ? '+' : '';
  return `${sign}${delta.toFixed(3)}`;
}

/**
 * Extract a latency p95 value from a metrics object.
 *
 * @param {object} metrics
 * @returns {number}
 */
function getLatencyP95(metrics) {
  const latency = metrics?.latency_ms;
  if (latency && typeof latency === 'object' && 'p95' in latency) {
    return Number(latency.p95 ?? 0);
  }
  return Number(latency ?? 0);
}

/**
 * Compare current metrics against a baseline and return FAIL/WARN/PASS status.
 *
 * @param {object} current - { condition: string, metrics: object }
 * @param {object} baseline - { condition: string, metrics: object }
 * @param {object} [options={}]
 * @param {Record<string, number>} [options.failThresholds={}] - Absolute
 *   regressions that cause FAIL.
 * @param {Record<string, number>} [options.warnThresholds={}] - Absolute
 *   regressions that cause WARN.
 * @param {number} [options.latencyWarnMs=100] - Latency p95 increase that
 *   causes WARN.
 * @returns {object} Gate report with `pass`, `metrics`, `regressions`, `warnings`.
 * @throws {Error} When conditions mismatch or baseline metrics are missing.
 */
export function compareToBaseline(current, baseline, options = {}) {
  if (!current || typeof current !== 'object') {
    throw new Error('current run object is required.');
  }
  if (!baseline || typeof baseline !== 'object') {
    throw new Error('baseline object is required.');
  }
  if (current.condition !== baseline.condition) {
    throw new Error(
      `Condition mismatch: current ${current.condition} vs baseline ${baseline.condition}.`,
    );
  }
  if (!baseline.metrics || typeof baseline.metrics !== 'object') {
    throw new Error('baseline metrics are required.');
  }
  if (!current.metrics || typeof current.metrics !== 'object') {
    throw new Error('current metrics are required.');
  }

  const failThresholds = options.failThresholds ?? {};
  const warnThresholds = options.warnThresholds ?? {};
  const latencyWarnMs = Number(options.latencyWarnMs ?? 100);

  const metricsReport = {};
  const regressions = [];
  const warnings = [];
  let pass = true;

  const metricNames = new Set([
    ...Object.keys(failThresholds),
    ...Object.keys(warnThresholds),
  ]);

  for (const metricName of metricNames) {
    const currentValue = Number(current.metrics[metricName] ?? 0);
    const baselineValue = Number(baseline.metrics[metricName] ?? 0);
    const delta = currentValue - baselineValue;
    const failThreshold = Number(failThresholds[metricName] ?? Infinity);
    const warnThreshold = Number(warnThresholds[metricName] ?? Infinity);

    let status = 'PASS';
    if (metricName in failThresholds && delta < -failThreshold) {
      status = 'FAIL';
      pass = false;
      regressions.push(metricName);
    } else if (metricName in warnThresholds && delta < -warnThreshold) {
      status = 'WARN';
      warnings.push(metricName);
    }

    metricsReport[metricName] = {
      current: currentValue,
      baseline: baselineValue,
      delta: formatDelta(delta),
      status,
    };
  }

  // Latency p95 comparison is always reported as a warning when breached.
  const currentLatency = getLatencyP95(current.metrics);
  const baselineLatency = getLatencyP95(baseline.metrics);
  const latencyDelta = currentLatency - baselineLatency;
  const latencyStatus = latencyDelta > latencyWarnMs ? 'WARN' : 'PASS';
  if (latencyStatus === 'WARN') {
    warnings.push('latency_p95');
  }
  metricsReport.latency_p95 = {
    current: currentLatency,
    baseline: baselineLatency,
    delta: formatDelta(latencyDelta),
    status: latencyStatus,
  };

  return {
    condition: current.condition,
    pass,
    metrics: metricsReport,
    regressions,
    warnings,
  };
}
