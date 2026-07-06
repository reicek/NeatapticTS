#!/usr/bin/env node
/**
 * @module perf-step28
 * @description Warm micro-benchmark for Step 28 graph traversal latency.
 *
 * Measures in-memory graph traversal P50/P95/P99 after cache warmup and exits
 * non-zero if the P50 budget is exceeded.
 *
 * @example
 * node rag-index/perf-step28.mjs
 * node rag-index/perf-step28.mjs /path/to/turso-replica.sqlite
 */
import path from 'node:path';

import { traverseGraph } from '../scripts/mcp-semantic/tools/traverse-graph.mjs';
import { defaultDatabasePath } from './init-schema.mjs';

const GRAPH_BUDGET_MS = 20;
const WARMUP_RUNS = 3;
const MEASUREMENT_RUNS = 30;

const SEED_SETS = [
  ['src/architecture/activationArrayPool'],
  ['src/architecture/architect', 'src/architecture/connection'],
  [
    'src/architecture/group',
    'src/architecture/layer',
    'src/architecture/network',
  ],
  ['src/architecture/connection/connection'],
  ['src/neat/selection/selection'],
];

/**
 * Return the value at a given percentile from a sorted array.
 *
 * @param {number[]} sorted - Ascending sorted measurements.
 * @param {number} p - Percentile (0–100).
 * @returns {number} Percentile value.
 */
function percentile(sorted, p) {
  const index = Math.max(0, Math.ceil((sorted.length - 1) * (p / 100)));
  return sorted[index];
}

/**
 * Measure graph traversal latency over repeated warm seed sets.
 *
 * @param {string} databasePath - Resolved SQLite database path.
 * @returns {Promise<object>} Latency statistics.
 */
async function measureGraphLatency(databasePath) {
  for (let i = 0; i < WARMUP_RUNS; i++) {
    for (const seeds of SEED_SETS) {
      await traverseGraph({
        databasePath,
        seed_names: seeds,
        max_hops: 2,
        max_results: 20,
      });
    }
  }

  const times = [];
  for (let i = 0; i < MEASUREMENT_RUNS; i++) {
    for (const seeds of SEED_SETS) {
      const start = performance.now();
      await traverseGraph({
        databasePath,
        seed_names: seeds,
        max_hops: 2,
        max_results: 20,
      });
      times.push(performance.now() - start);
    }
  }

  times.sort((a, b) => a - b);
  return {
    p50: percentile(times, 50),
    p95: percentile(times, 95),
    p99: percentile(times, 99),
    min: times[0],
    max: times[times.length - 1],
    samples: times.length,
  };
}

/**
 * Run the benchmark and report results.
 *
 * @returns {Promise<void>}
 */
async function main() {
  const databasePath = process.argv[2] ?? defaultDatabasePath;
  const results = await measureGraphLatency(databasePath);
  const passes = results.p50 <= GRAPH_BUDGET_MS;

  console.log(
    JSON.stringify(
      {
        subsystem: 'graph_traversal',
        budget_ms: GRAPH_BUDGET_MS,
        passes,
        results,
      },
      null,
      2,
    ),
  );

  process.exit(passes ? 0 : 1);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
