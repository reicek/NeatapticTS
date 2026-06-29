#!/usr/bin/env node
/**
 * @module perf-benchmark
 * @description Benchmark all 18 registered repo-cortex-mcp tools for latency.
 *
 * Measures p50/p95/p99 latencies for every tool exposed by
 * `scripts/mcp-semantic/repo-cortex-mcp.mjs` against the live Turso-backed
 * corpus index, plus embedded-replica local-read latency, sync lag, and heap
 * stability during 100 sequential queries.
 *
 * @example
 * node rag-index/perf-benchmark.mjs --json
 * TURSO_SYNC_URL=libsql://... node rag-index/perf-benchmark.mjs --json
 */
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

import { createClient } from '@libsql/client';
import { createRepoCortexMcpServer } from '../mcp-semantic/repo-cortex-mcp.mjs';
import { invokeServerRequest } from '../agent-customization/mcp/mcp-utils.mjs';
import {
  closeTursoClient,
  getCachedClientCount,
  getTursoClient,
} from '../scripts/mcp-semantic/tools/cortex-db.mjs';
import { defaultDatabasePath } from '../semantic-index/init-schema.mjs';

const repoRoot = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  '..',
  '..',
);
const envPath = path.join(repoRoot, '.env');

/** Number of warmup iterations before measurement. */
const DEFAULT_WARMUP = 3;
/** Default measurement iterations for fast tools. */
const DEFAULT_ITERATIONS = 30;
/** Number of sequential queries for the heap-stability probe. */
const HEAP_PROBE_QUERIES = 100;
/** Acceptable heap growth during the sequential-query probe (MB). */
const HEAP_SPIKE_THRESHOLD_MB = 100;
/** p95 budget for search_corpus (ms). */
const SEARCH_CORPUS_BUDGET_MS = 100;
/** p95 budget for search_advanced (ms). */
const SEARCH_ADVANCED_BUDGET_MS = 200;
/** p95 budget for search_context (ms). */
const SEARCH_CONTEXT_BUDGET_MS = 150;
/** Average local-read budget (ms). */
const EMBEDDED_READ_BUDGET_MS = 1;
/** Sync-lag budget (ms). */
const SYNC_LAG_BUDGET_MS = 60_000;

/**
 * Best-effort load a `.env` file into `process.env` without adding a dependency.
 *
 * Only sets variables that are not already present, and never logs values.
 *
 * @returns {Promise<void>}
 */
async function loadEnvFile() {
  try {
    const text = await readFile(envPath, 'utf8');
    for (const line of text.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed || trimmed.startsWith('#')) continue;
      const separatorIndex = trimmed.indexOf('=');
      if (separatorIndex === -1) continue;
      const key = trimmed.slice(0, separatorIndex).trim();
      const value = trimmed.slice(separatorIndex + 1).trim();
      if (process.env[key] === undefined) process.env[key] = value;
    }
  } catch {
    // No .env file; rely on externally-set environment variables.
  }
}

/**
 * Compute the p-th percentile using linear interpolation.
 *
 * @param {number[]} sorted - Ascending sorted measurements.
 * @param {number} p - Percentile in the range [0, 100].
 * @returns {number} Percentile value.
 */
function percentile(sorted, p) {
  if (sorted.length === 0) return 0;
  if (sorted.length === 1) return sorted[0];
  const index = (p / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  const weight = index - lower;
  return sorted[lower] * (1 - weight) + sorted[upper] * weight;
}

/**
 * Aggregate latency samples into p50/p95/p99/min/max/avg statistics.
 *
 * @param {number[]} samples - Raw latency measurements in milliseconds.
 * @returns {{p50: number, p95: number, p99: number, min: number, max: number, avg: number, samples: number}} Summary.
 */
function aggregateLatency(samples) {
  const values = Array.isArray(samples) ? samples : [];
  const sorted = values.toSorted((a, b) => a - b);
  const sum = sorted.reduce((acc, value) => acc + value, 0);
  return {
    p50: percentile(sorted, 50),
    p95: percentile(sorted, 95),
    p99: percentile(sorted, 99),
    min: sorted[0] ?? 0,
    max: sorted[sorted.length - 1] ?? 0,
    avg: sorted.length > 0 ? sum / sorted.length : 0,
    samples: sorted.length,
  };
}

/**
 * Invoke a single MCP tool and return its latency.
 *
 * @param {object} server - Repo Cortex MCP server instance.
 * @param {string} toolName - Registered tool name.
 * @param {Record<string, unknown>} args - Tool arguments.
 * @returns {Promise<{latency_ms: number, ok: boolean, error?: string}>} Invocation result.
 */
async function invokeTimed(server, toolName, args) {
  const startMs = performance.now();
  try {
    const result = await invokeServerRequest(server, {
      method: 'tools/call',
      params: { name: toolName, arguments: args },
    });
    if (isErrorResult(result)) {
      return {
        latency_ms: performance.now() - startMs,
        ok: false,
        error: extractToolErrorMessage(result),
      };
    }
    return { latency_ms: performance.now() - startMs, ok: true };
  } catch (error) {
    return {
      latency_ms: performance.now() - startMs,
      ok: false,
      error: error instanceof Error ? error.message : String(error),
    };
  }
}

/**
 * Determine whether a tool response envelope represents a tool-level failure.
 *
 * `mcp-utils.mjs` catches handler errors and returns `{ isError: true, ... }`
 * rather than throwing, so callers must inspect the envelope explicitly.
 *
 * @param {unknown} result - Raw tool response envelope.
 * @returns {boolean} True when the envelope signals an error.
 */
function isErrorResult(result) {
  return (
    result !== null && typeof result === 'object' && result.isError === true
  );
}

/**
 * Extract a readable error message from a tool error envelope.
 *
 * Falls back through `structuredContent.error`, the first `text` content item,
 * and finally a generic stringification.
 *
 * @param {object} result - Tool error envelope.
 * @returns {string} Error message.
 */
function extractToolErrorMessage(result) {
  if (result.structuredContent?.error) {
    return String(result.structuredContent.error);
  }
  if (Array.isArray(result.content)) {
    const textItem = result.content.find((item) => item?.type === 'text');
    if (textItem?.text) return String(textItem.text);
  }
  return 'Tool returned isError: true';
}

/**
 * Benchmark a single MCP tool.
 *
 * @param {object} server - Repo Cortex MCP server instance.
 * @param {string} toolName - Registered tool name.
 * @param {Record<string, unknown>} args - Representative tool arguments.
 * @param {number} iterations - Number of measured invocations.
 * @param {number} warmup - Number of warmup invocations.
 * @returns {Promise<{p50: number, p95: number, p99: number, min: number, max: number, avg: number, samples: number, errors: number, error_sample?: string}>} Latency summary.
 */
async function benchmarkTool(server, toolName, args, iterations, warmup) {
  for (let i = 0; i < warmup; i++) {
    await invokeTimed(server, toolName, args);
  }

  const times = [];
  let errors = 0;
  let errorSample;
  for (let i = 0; i < iterations; i++) {
    const { latency_ms, ok, error } = await invokeTimed(server, toolName, args);
    if (ok) {
      times.push(latency_ms);
    } else {
      errors += 1;
      if (!errorSample) errorSample = error;
    }
  }

  return {
    ...aggregateLatency(times),
    errors,
    ...(errorSample ? { error_sample: errorSample } : {}),
  };
}

/**
 * Discover safe sample IDs from the live corpus for load/parent/document tools.
 *
 * @param {import('@libsql/client').Client} client - libSQL client.
 * @returns {Promise<{chunkId: number, parentChunkId: number, filePath: string}>} Sample identifiers.
 */
async function discoverSampleIds(client) {
  const chunkResult = await client.execute({
    sql: 'SELECT chunk_id FROM chunks ORDER BY chunk_id LIMIT 1',
    args: [],
  });
  const chunkId = Number(chunkResult.rows[0]?.chunk_id ?? 1);

  const parentChunkResult = await client.execute({
    sql: 'SELECT chunk_id FROM chunks WHERE depth = 1 ORDER BY chunk_id LIMIT 1',
    args: [],
  });
  const parentChunkId = Number(parentChunkResult.rows[0]?.chunk_id ?? chunkId);

  const fileResult = await client.execute({
    sql: 'SELECT file_path FROM documents ORDER BY doc_id LIMIT 1',
    args: [],
  });
  const filePath =
    fileResult.rows[0]?.file_path ?? 'src/architecture/network/network.ts';

  return { chunkId, parentChunkId, filePath };
}

/**
 * Capture the SQLite query plan for the BM25 FTS5 search path.
 *
 * Runs `EXPLAIN QUERY PLAN` against the same SQL shape used by
 * {@link runBm25Search} so the optimizer choices (FTS5 index, rowid joins,
 * document lookup) can be reviewed as part of the optimization slice.
 *
 * @param {import('@libsql/client').Client} client - Plain local libSQL client.
 * @returns {Promise<Array<{detail: string}>>} Query-plan rows.
 */
async function measureFtsPlan(client) {
  const query = 'network activate';
  const limit = 5;

  const sqlWithFeedback = `
    SELECT c.chunk_id
    FROM chunks_fts
    JOIN chunks c ON c.chunk_id = chunks_fts.rowid
    JOIN documents d ON d.doc_id = c.doc_id
    LEFT JOIN feedback_scores fs ON fs.chunk_id = c.chunk_id
    WHERE chunks_fts MATCH ?
    ORDER BY bm25(chunks_fts)
    LIMIT ?
  `;

  const sqlNoFeedback = `
    SELECT c.chunk_id
    FROM chunks_fts
    JOIN chunks c ON c.chunk_id = chunks_fts.rowid
    JOIN documents d ON d.doc_id = c.doc_id
    WHERE chunks_fts MATCH ?
    ORDER BY bm25(chunks_fts)
    LIMIT ?
  `;

  const args = [query, limit];
  try {
    const result = await client.execute({
      sql: `EXPLAIN QUERY PLAN ${sqlWithFeedback}`,
      args,
    });
    return result.rows;
  } catch {
    const result = await client.execute({
      sql: `EXPLAIN QUERY PLAN ${sqlNoFeedback}`,
      args,
    });
    return result.rows;
  }
}

/**
 * Measure raw local-read latency and sync behavior for the embedded replica.
 *
 * Uses the supplied plain local client for local reads, then attempts a
 * separate embedded-replica client (if a sync URL was configured) just to
 * measure the sync() latency. This keeps the main benchmark resilient to
 * sync-configuration errors while still reporting sync lag when possible.
 *
 * @param {import('@libsql/client').Client} client - Plain local libSQL client.
 * @param {string | undefined} syncUrl - Original TURSO_SYNC_URL value.
 * @param {string | undefined} authToken - Original TURSO_AUTH_TOKEN value.
 * @param {string} databaseUrl - Local database path or URL used by the benchmark.
 * @returns {Promise<{local_read_avg_ms: number, local_read_samples: number, sync_configured: boolean, sync_lag_ms: number | null, sync_error?: string}>} Embedded-replica metrics.
 */
async function measureEmbeddedReplica(client, syncUrl, authToken, databaseUrl) {
  const localReadTimes = [];
  for (let i = 0; i < 100; i++) {
    const startMs = performance.now();
    await client.execute({
      sql: 'SELECT chunk_id FROM chunks LIMIT 1',
      args: [],
    });
    localReadTimes.push(performance.now() - startMs);
  }

  const localAvg =
    localReadTimes.reduce((sum, value) => sum + value, 0) /
    localReadTimes.length;

  let syncConfigured = false;
  let syncLagMs = null;
  let syncError;
  if (syncUrl) {
    syncConfigured = true;
    const isUrl =
      /^(libsql|wss|ws|https|http|file):/i.test(databaseUrl) ||
      databaseUrl === ':memory:';
    const resolvedUrl = isUrl ? databaseUrl : pathToFileURL(databaseUrl).href;
    const syncIntervalEnv = process.env.TURSO_SYNC_INTERVAL;
    /** @type {import('@libsql/client').Client | undefined} */
    let syncClient;
    try {
      syncClient = createClient({
        url: resolvedUrl,
        syncUrl,
        authToken,
        syncInterval:
          syncIntervalEnv != null && syncIntervalEnv !== ''
            ? Number(syncIntervalEnv)
            : undefined,
      });
      const syncStart = performance.now();
      await syncClient.sync();
      syncLagMs = performance.now() - syncStart;
    } catch (error) {
      syncError = error instanceof Error ? error.message : String(error);
    } finally {
      if (syncClient) {
        try {
          await syncClient.close();
        } catch {
          // Ignore close errors; the sync error is the relevant signal.
        }
      }
    }
  }

  return {
    local_read_avg_ms: localAvg,
    local_read_samples: localReadTimes.length,
    sync_configured: syncConfigured,
    sync_lag_ms: syncLagMs,
    ...(syncError ? { sync_error: syncError } : {}),
  };
}

/**
 * Probe heap stability during 100 sequential search_corpus queries.
 *
 * Warms the embedding model first, then records the maximum heap growth.
 *
 * @param {object} server - Repo Cortex MCP server instance.
 * @param {Record<string, unknown>} queryArgs - Representative search_corpus arguments.
 * @returns {Promise<{heap_start_mb: number, heap_end_mb: number, heap_max_mb: number, delta_mb: number, max_delta_mb: number}>} Memory summary.
 */
async function measureHeapStability(server, queryArgs) {
  // Warm up any lazy-loaded models before measuring heap.
  for (let i = 0; i < 3; i++) {
    await invokeTimed(server, 'search_corpus', queryArgs);
  }
  if (global.gc) global.gc();

  const startHeapMb = process.memoryUsage().heapUsed / 1_048_576;
  let maxHeapMb = startHeapMb;

  for (let i = 0; i < HEAP_PROBE_QUERIES; i++) {
    await invokeTimed(server, 'search_corpus', queryArgs);
    if (i % 10 === 0) {
      const currentMb = process.memoryUsage().heapUsed / 1_048_576;
      if (currentMb > maxHeapMb) maxHeapMb = currentMb;
    }
  }

  const endHeapMb = process.memoryUsage().heapUsed / 1_048_576;
  return {
    heap_start_mb: startHeapMb,
    heap_end_mb: endHeapMb,
    heap_max_mb: maxHeapMb,
    delta_mb: endHeapMb - startHeapMb,
    max_delta_mb: maxHeapMb - startHeapMb,
  };
}

/**
 * Mask a sensitive URL for safe logging.
 *
 * @param {string | undefined} value - Raw env value.
 * @returns {string} "set", "unset", or a masked string.
 */
function maskEnv(value) {
  if (value == null || value === '') return 'unset';
  return 'set';
}

/**
 * Build the benchmark configuration for all 18 registered repo-cortex-mcp tools.
 *
 * @param {{chunkId: number, parentChunkId: number, filePath: string}} samples - Discovered sample IDs.
 * @returns {Array<{name: string, args: Record<string, unknown>, iterations: number, warmup: number}>} Tool configurations.
 */
function buildToolConfigs(samples) {
  const { chunkId, parentChunkId, filePath } = samples;
  return [
    {
      name: 'search_corpus',
      args: { query: 'network activate', limit: 5, compact: true },
      iterations: DEFAULT_ITERATIONS,
      warmup: DEFAULT_WARMUP,
    },
    {
      name: 'search_context',
      args: {
        query: 'network activate',
        limit: 5,
        budget: 1_024,
        compact: true,
      },
      iterations: DEFAULT_ITERATIONS,
      warmup: DEFAULT_WARMUP,
    },
    {
      name: 'search_advanced',
      args: { query: 'network activate', limit: 5, compact: true },
      iterations: DEFAULT_ITERATIONS,
      warmup: DEFAULT_WARMUP,
    },
    {
      name: 'load_chunk',
      args: { chunk_id: chunkId },
      iterations: DEFAULT_ITERATIONS,
      warmup: DEFAULT_WARMUP,
    },
    {
      name: 'load_parent_chunk',
      args: { chunk_id: parentChunkId },
      iterations: DEFAULT_ITERATIONS,
      warmup: DEFAULT_WARMUP,
    },
    {
      name: 'load_document',
      args: { file_path: filePath },
      iterations: DEFAULT_ITERATIONS,
      warmup: DEFAULT_WARMUP,
    },
    {
      name: 'freshness_check',
      args: { file_path: filePath },
      iterations: DEFAULT_ITERATIONS,
      warmup: DEFAULT_WARMUP,
    },
    {
      name: 'index_stats',
      args: { include_metadata_coverage: false },
      iterations: DEFAULT_ITERATIONS,
      warmup: DEFAULT_WARMUP,
    },
    {
      name: 'ann_build_index',
      args: { force: 'diskann' },
      iterations: 1,
      warmup: 0,
    },
    {
      name: 'list_families',
      args: {},
      iterations: DEFAULT_ITERATIONS,
      warmup: DEFAULT_WARMUP,
    },
    {
      name: 'scan_code_quality',
      args: {
        source_paths: [filePath],
        complexity_threshold: 15,
        min_jsdoc_words: 5,
      },
      iterations: 3,
      warmup: 1,
    },
    {
      name: 'traverse_graph',
      args: { seed_names: ['network'], max_hops: 2, max_results: 20 },
      iterations: DEFAULT_ITERATIONS,
      warmup: DEFAULT_WARMUP,
    },
    {
      name: 'expand_query',
      args: { query: 'network activate', expand_query: true },
      iterations: DEFAULT_ITERATIONS,
      warmup: DEFAULT_WARMUP,
    },
    {
      name: 'submit_feedback',
      args: { chunk_id: chunkId, signal_type: 'reference' },
      iterations: 10,
      warmup: 1,
    },
    {
      name: 'parallel_search',
      args: {
        queries: [{ sql: 'SELECT chunk_id FROM chunks LIMIT 5', args: [] }],
      },
      iterations: DEFAULT_ITERATIONS,
      warmup: DEFAULT_WARMUP,
    },
    {
      name: 'multi_hop_search',
      args: { query: 'network activate', max_hops: 2, limit: 5 },
      iterations: DEFAULT_ITERATIONS,
      warmup: DEFAULT_WARMUP,
    },
    {
      name: 'turso_branch',
      args: {
        branch_name: 'perf-benchmark-test-branch',
        action: 'delete',
      },
      iterations: 1,
      warmup: 0,
    },
    {
      name: 'turso_pitr',
      args: {
        database_name: 'perf-benchmark-test-pitr',
        timestamp: new Date().toISOString(),
      },
      iterations: 1,
      warmup: 0,
    },
  ];
}

/**
 * Print CLI help and exit.
 *
 * @returns {never}
 */
function printHelp() {
  console.log(`Benchmark all 18 repo-cortex-mcp tools for latency.

Usage:
  node rag-index/perf-benchmark.mjs [options]

Options:
  --database <path|url>  Override the local database path (default: rag-index/data/turso-replica.sqlite).
  --json                 Emit JSON results to stdout (default: human-readable table).
  --help                 Show this help message.

Environment:
  TURSO_DATABASE_URL    Turso database URL or local path (benchmark overrides placeholder .env values).
  TURSO_SYNC_URL        Sync URL for embedded-replica mode (optional).
  TURSO_SYNC_INTERVAL   Sync interval in seconds (optional).
  TURSO_AUTH_TOKEN      JWT auth token for cloud access (optional).`);
  process.exit(0);
}

/**
 * Format a latency number for human-readable output.
 *
 * @param {number} value - Latency in milliseconds.
 * @returns {string} Formatted value.
 */
function fmt(value) {
  return `${value.toFixed(2)}ms`;
}

/**
 * Run the full benchmark suite and report results.
 *
 * @returns {Promise<void>}
 */
async function main() {
  const argv = process.argv.slice(2);
  if (argv.includes('--help') || argv.includes('-h')) printHelp();
  const jsonOutput = argv.includes('--json');

  const databaseFlagIndex = argv.indexOf('--database');
  const databasePathOverride =
    databaseFlagIndex !== -1 ? argv[databaseFlagIndex + 1] : undefined;

  await loadEnvFile();

  // Default to the local embedded replica so the placeholder
  // TURSO_DATABASE_URL in .env does not route benchmarks to a 404 URL.
  // Exported MCP handlers also consult getTursoClient() without an argument,
  // so keep the environment consistent by overriding the env var.
  const databaseUrl = databasePathOverride ?? defaultDatabasePath;
  process.env.TURSO_DATABASE_URL = databaseUrl;

  // Preserve the user's sync configuration for a dedicated sync-lag probe,
  // but use a plain local libSQL client for the bulk of the benchmark.
  // Embedded-replica creation can fail on Windows path handling or with an
  // expired auth token; the benchmark should still report local latencies.
  const originalSyncUrl = process.env.TURSO_SYNC_URL;
  const originalAuthToken = process.env.TURSO_AUTH_TOKEN;
  delete process.env.TURSO_SYNC_URL;

  const server = createRepoCortexMcpServer();
  const client = await getTursoClient();
  const samples = await discoverSampleIds(client);
  const toolConfigs = buildToolConfigs(samples);
  const ftsPlan = await measureFtsPlan(client);
  const cachedClientCount = getCachedClientCount();

  // Run tool benchmarks.
  const toolResults = {};
  for (const config of toolConfigs) {
    toolResults[config.name] = await benchmarkTool(
      server,
      config.name,
      config.args,
      config.iterations,
      config.warmup,
    );
  }

  const embeddedReplica = await measureEmbeddedReplica(
    client,
    originalSyncUrl,
    originalAuthToken,
    databaseUrl,
  );
  const memory = await measureHeapStability(server, {
    query: 'network activate',
    limit: 5,
    compact: true,
  });

  // Evaluate acceptance criteria.
  const passes = {
    search_corpus: toolResults.search_corpus.p95 < SEARCH_CORPUS_BUDGET_MS,
    search_advanced:
      toolResults.search_advanced.p95 < SEARCH_ADVANCED_BUDGET_MS,
    search_context: toolResults.search_context.p95 < SEARCH_CONTEXT_BUDGET_MS,
    embedded_replica:
      embeddedReplica.local_read_avg_ms < EMBEDDED_READ_BUDGET_MS,
    sync_lag: embeddedReplica.sync_configured
      ? typeof embeddedReplica.sync_lag_ms === 'number' &&
        embeddedReplica.sync_lag_ms < SYNC_LAG_BUDGET_MS
      : null,
    memory: memory.max_delta_mb < HEAP_SPIKE_THRESHOLD_MB,
    connection_pool: cachedClientCount === 1,
  };

  const overallPasses =
    passes.search_corpus &&
    passes.search_advanced &&
    passes.search_context &&
    passes.embedded_replica &&
    (passes.sync_lag ?? true) &&
    passes.memory &&
    passes.connection_pool;

  const output = {
    benchmark: 'repo-cortex-mcp-latency',
    timestamp: new Date().toISOString(),
    environment: {
      database_path: databaseUrl,
      TURSO_DATABASE_URL: maskEnv(process.env.TURSO_DATABASE_URL),
      TURSO_SYNC_URL: maskEnv(process.env.TURSO_SYNC_URL),
      TURSO_AUTH_TOKEN: maskEnv(process.env.TURSO_AUTH_TOKEN),
      TURSO_SYNC_INTERVAL: process.env.TURSO_SYNC_INTERVAL ?? 'default(60)',
    },
    tools: toolResults,
    thresholds: {
      search_corpus_p95_ms: SEARCH_CORPUS_BUDGET_MS,
      search_advanced_p95_ms: SEARCH_ADVANCED_BUDGET_MS,
      search_context_p95_ms: SEARCH_CONTEXT_BUDGET_MS,
      embedded_read_avg_ms: EMBEDDED_READ_BUDGET_MS,
      sync_lag_ms: SYNC_LAG_BUDGET_MS,
      heap_max_delta_mb: HEAP_SPIKE_THRESHOLD_MB,
    },
    embedded_replica: embeddedReplica,
    connection_pool: {
      cached_clients: cachedClientCount,
      pooled: cachedClientCount === 1,
    },
    fts_plan: ftsPlan.map((row) => row.detail),
    memory,
    passes,
    overall_passes: overallPasses,
  };

  if (jsonOutput) {
    console.log(JSON.stringify(output, null, 2));
  } else {
    console.log('Repo Cortex MCP Tool Latency Benchmark');
    console.log('======================================');
    console.log(`Database path : ${databaseUrl}`);
    console.log(`TURSO_SYNC_URL: ${maskEnv(process.env.TURSO_SYNC_URL)}`);
    console.log(
      `Sync interval : ${process.env.TURSO_SYNC_INTERVAL ?? 'default(60)'}`,
    );
    console.log();
    console.log(
      '| Tool                | Samples | p50      | p95      | p99      | Avg      | Errors |',
    );
    console.log(
      '|---------------------|--------:|---------:|---------:|---------:|---------:|-------:|',
    );
    for (const [name, stats] of Object.entries(toolResults)) {
      const paddedName = name.padEnd(19, ' ');
      console.log(
        `| ${paddedName} | ${String(stats.samples).padStart(7, ' ')} | ${fmt(stats.p50).padStart(8, ' ')} | ${fmt(stats.p95).padStart(8, ' ')} | ${fmt(stats.p99).padStart(8, ' ')} | ${fmt(stats.avg).padStart(8, ' ')} | ${String(stats.errors).padStart(6, ' ')} |`,
      );
    }
    console.log();
    console.log('Acceptance criteria');
    console.log('-------------------');
    console.log(
      `search_corpus p95 < ${SEARCH_CORPUS_BUDGET_MS}ms   : ${passes.search_corpus ? 'PASS' : 'FAIL'} (${fmt(toolResults.search_corpus.p95)})`,
    );
    console.log(
      `search_advanced p95 < ${SEARCH_ADVANCED_BUDGET_MS}ms : ${passes.search_advanced ? 'PASS' : 'FAIL'} (${fmt(toolResults.search_advanced.p95)})`,
    );
    console.log(
      `search_context p95 < ${SEARCH_CONTEXT_BUDGET_MS}ms  : ${passes.search_context ? 'PASS' : 'FAIL'} (${fmt(toolResults.search_context.p95)})`,
    );
    console.log(
      `Embedded replica local read avg < ${EMBEDDED_READ_BUDGET_MS}ms : ${passes.embedded_replica ? 'PASS' : 'FAIL'} (${fmt(embeddedReplica.local_read_avg_ms)})`,
    );
    if (embeddedReplica.sync_configured) {
      const lagText =
        typeof embeddedReplica.sync_lag_ms === 'number'
          ? fmt(embeddedReplica.sync_lag_ms)
          : `error${embeddedReplica.sync_error ? `: ${embeddedReplica.sync_error}` : ''}`;
      console.log(
        `Sync lag < ${SYNC_LAG_BUDGET_MS}ms        : ${passes.sync_lag ? 'PASS' : 'FAIL'} (${lagText})`,
      );
    } else {
      console.log('Sync lag             : N/A (TURSO_SYNC_URL not configured)');
    }
    console.log(
      `Heap max delta < ${HEAP_SPIKE_THRESHOLD_MB}MB : ${passes.memory ? 'PASS' : 'FAIL'} (${memory.max_delta_mb.toFixed(2)}MB)`,
    );
    console.log();
    console.log(`Overall: ${overallPasses ? 'PASS' : 'FAIL'}`);
  }

  await closeTursoClient();
  process.exit(overallPasses ? 0 : 1);
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
