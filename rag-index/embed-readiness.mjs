/**
 * @module embed-readiness
 * @description Caching readiness probe for the embedding (dense search) subsystem.
 *
 * Wraps {@link checkDenseReadiness} with a process-lifetime cache and adds a
 * `latency_ms` measurement so callers can observe cold-start cost. The first
 * call probes the local ONNX model cache and embeddings database; subsequent
 * calls with the same options key return the cached report and avoid repeated
 * filesystem or database work.
 *
 * @param {boolean} [--json]                         - Emit JSON readiness output.
 * @param {string}  [--database <path>]                - Override the corpus database path.
 * @param {string}  [--model-directory <path>]           - Override the local model directory.
 * @param {string}  [--model-id <id>]                  - Override the embedding model identifier.
 * @param {boolean} [--help]                           - Show help and exit.
 *
 * @returns {void} Exits 0 after reporting the current readiness state.
 */
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { checkDenseReadiness } from './dense-readiness.mjs';
import {
  fail,
  parseCliArgs,
  printHelp,
  writeJsonOrText,
} from './cli-utils.mjs';

/** @type {object | null} */
let cachedReport = null;

/** @type {string | null} */
let cachedOptionsKey = null;

/**
 * Build a stable cache key from the subset of options that affect the readiness
 * probe result.
 *
 * @param {object} options - Probe options.
 * @returns {string} JSON-serialized normalized key.
 */
function buildOptionsKey(options) {
  return JSON.stringify({
    corpusDatabasePath:
      options.corpusDatabasePath ?? options.databasePath ?? null,
    modelDirectory: options.modelDirectory ?? null,
    modelId: options.modelId ?? null,
  });
}

/**
 * Probe dense-embedding readiness and cache the result per options key.
 *
 * The first call with a given options key performs a live probe and records
 * the elapsed time as `latency_ms`. Later calls return a cached report with
 * `cached: true` and a fresh latency measurement for the cache hit itself.
 *
 * @param {{
 *   corpusDatabasePath?: string,
 *   databasePath?: string,
 *   modelDirectory?: string,
 *   modelId?: string,
 * }} [options={}] - Probe configuration.
 * @returns {Promise<{
 *   state: 'cold' | 'model-only' | 'warm',
 *   ready: boolean,
 *   latency_ms: number,
 *   cached: boolean,
 *   reason: string,
 *   chunk_count: number | null,
 *   embedding_count: number | null,
 * }>} Readiness report with latency and cache metadata.
 */
export async function getEmbedReadiness(options = {}) {
  const startTime = performance.now();
  const key = buildOptionsKey(options);

  if (cachedReport !== null && key === cachedOptionsKey) {
    return {
      ...cachedReport,
      latency_ms: performance.now() - startTime,
      cached: true,
    };
  }

  const probeReport = await checkDenseReadiness(options);
  const latency_ms = performance.now() - startTime;
  const report = {
    cached: false,
    chunk_count: probeReport.chunk_count ?? null,
    embedding_count: probeReport.embedding_count ?? null,
    latency_ms,
    ready: Boolean(probeReport.ready),
    reason: probeReport.reason ?? '',
    state: probeReport.state,
  };

  cachedReport = report;
  cachedOptionsKey = key;
  return report;
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));

  if (args.help) {
    printHelp({
      title: 'Embedding readiness probe',
      usage: 'node rag-index/embed-readiness.mjs [--json]',
      options: [
        '--json                       Emit the readiness report as JSON.',
        '--database <path>            Override the semantic-index corpus database path.',
        '--model-directory <path>     Override the local dense model directory.',
        '--model-id <id>              Override the embedding model identifier.',
        '--help                       Show this help.',
      ],
    });
    return;
  }

  try {
    const report = await getEmbedReadiness({
      corpusDatabasePath: args.database,
      modelDirectory: args['model-directory'],
      modelId: args['model-id'],
    });
    writeJsonOrText(
      report,
      Boolean(args.json),
      (payload) => `${payload.state}: ${payload.reason ?? ''}`,
    );
  } catch (error) {
    fail(
      error instanceof Error ? error.message : String(error),
      Boolean(args.json),
    );
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();
