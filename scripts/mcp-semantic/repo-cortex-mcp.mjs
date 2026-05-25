/**
 * @module repo-cortex-mcp
 * @description Repo Cortex MCP server — exposes the NeatapticTS semantic corpus index as MCP tools.
 *
 * Wraps corpus search, chunk loading, document loading, freshness checking,
 * index statistics, family listing, and code-quality scanning in a
 * dependency-light stdio JSON-RPC server that VS Code's MCP host can invoke.
 *
 * @remarks
 * ### Cortex MCP Tool Map
 *
 * ```mermaid
 * graph LR
 *   MCP[neataptic-cortex-mcp] --> search_corpus
 *   MCP --> load_chunk
 *   MCP --> load_document
 *   MCP --> freshness_check
 *   MCP --> index_stats
 *   MCP --> list_families
 *   MCP --> scan_code_quality
 *   search_corpus --> searchCorpus
 *   load_chunk --> loadChunk
 *   load_document --> loadDocument
 *   freshness_check --> freshnessCheck
 *   index_stats --> indexStats
 *   list_families --> listFamilies
 *   scan_code_quality --> runDocsQualityMetrics
 * ```
 */
import { pathToFileURL } from 'node:url';

import {
  createMcpServer,
  createSelfCheckReport,
  createTool,
  emitSelfCheckReport,
  invokeServerRequest,
  parseMcpCliArgs,
  printMcpUsage,
  runStdioMcpServer,
  selfCheckError,
} from '../agent-customization/mcp/mcp-utils.mjs';
import { freshnessCheck } from './tools/freshness-check.mjs';
import { indexStats } from './tools/index-stats.mjs';
import { listFamilies } from './tools/list-families.mjs';
import { loadChunk } from './tools/load-chunk.mjs';
import { loadDocument } from './tools/load-document.mjs';
import { runDocsQualityMetrics } from '../semantic-index/docs-quality/docs-quality.metrics.mjs';
import { searchCorpus } from './tools/search-corpus.mjs';

const SERVER_VERSION = '0.1.0';
const ENTRYPOINT = 'scripts/mcp-semantic/repo-cortex-mcp.mjs';

/**
 * Create the Repo Cortex MCP server instance.
 *
 * @param {{ databasePath?: string }} [options={}] - Optional database path override.
 * @returns {{ serverInfo: { name: string, version: string }, tools: Array<Record<string, unknown>>, dispatch: (request: Record<string, unknown>) => Promise<unknown> }} MCP server.
 */
export function createRepoCortexMcpServer(options = {}) {
  const databasePath = options.databasePath;
  const tools = createRepoCortexTools(databasePath);
  return createMcpServer({
    serverName: 'neataptic-cortex-mcp',
    serverVersion: SERVER_VERSION,
    tools,
  });
}

/**
 * Build the full tool list for the Repo Cortex MCP server.
 *
 * @param {string | undefined} databasePath - Optional corpus database path override.
 * @returns {Array<{ name: string, description: string, inputSchema: Record<string, unknown>, handler: Function }>} Tool descriptors.
 */
export function createRepoCortexTools(databasePath) {
  return [
    createTool({
      name: 'search_corpus',
      description: 'BM25 full-text search over indexed repository chunks with default-on hybrid dense reranking. If embeddings are cold or model-only, the tool returns BM25 results with dense_state, dense_degraded, and dense_reason; run npm run index:prewarm to warm dense search.',
      inputSchema: {
        type: 'object',
        properties: {
          query: { type: 'string' },
          limit: { type: 'number' },
          family: { type: 'string' },
          use_dense: {
            type: 'boolean',
            default: true,
            description: 'Defaults to true. Set false for BM25-only search; run npm run index:prewarm when dense_state reports cold or model-only.',
          },
          alpha: { type: 'number' },
        },
        required: ['query'],
        additionalProperties: false,
      },
      outputSchema: {
        type: 'object',
        properties: {
          query: { type: 'string' },
          limit: { type: 'number' },
          family: { type: 'string' },
          alpha: { type: 'number' },
          use_dense: { type: 'boolean' },
          dense_degraded: {
            type: 'boolean',
            description: 'True when default-on or requested dense search degraded to BM25 because embeddings were cold or model-only.',
          },
          dense_state: {
            type: 'string',
            enum: ['cold', 'model-only', 'warm'],
            description: 'Dense-readiness provenance emitted on default-on dense responses: cold, model-only, or warm.',
          },
          dense_reason: {
            type: 'string',
            description: 'Human-readable degradation reason when dense search falls back to BM25; operators should run npm run index:prewarm.',
          },
          results: {
            type: 'array',
            items: { type: 'object' },
          },
        },
        required: ['query', 'limit', 'use_dense', 'results'],
        additionalProperties: true,
      },
      handler: (argumentsObject) => searchCorpus({ ...argumentsObject, databasePath }),
    }),
    createTool({
      name: 'load_chunk',
      description: 'Load one indexed corpus chunk by numeric chunk ID.',
      inputSchema: {
        type: 'object',
        properties: { chunk_id: { type: 'number' } },
        required: ['chunk_id'],
        additionalProperties: false,
      },
      handler: (argumentsObject) => loadChunk({ ...argumentsObject, databasePath }),
    }),
    createTool({
      name: 'load_document',
      description: 'Load all ordered chunks for one indexed repository path.',
      inputSchema: {
        type: 'object',
        properties: { file_path: { type: 'string' } },
        required: ['file_path'],
        additionalProperties: false,
      },
      handler: (argumentsObject) => loadDocument({ ...argumentsObject, databasePath }),
    }),
    createTool({
      name: 'freshness_check',
      description: 'Compare indexed document freshness proofs with current filesystem metadata.',
      inputSchema: {
        type: 'object',
        properties: {
          file_path: { type: 'string' },
          freshnessProof: { type: 'object' },
        },
        additionalProperties: false,
      },
      handler: (argumentsObject) => freshnessCheck({ ...argumentsObject, databasePath }),
    }),
    createTool({
      name: 'index_stats',
      description: 'Return corpus row counts, family counts, and last indexed timestamp.',
      handler: () => indexStats({ databasePath }),
    }),
    createTool({
      name: 'list_families',
      description: 'List indexed document families with document and chunk counts.',
      handler: () => listFamilies({ databasePath }),
    }),
    createTool({
      name: 'scan_code_quality',
      description: 'Scan exported TypeScript symbols for missing or weak JSDoc and high cyclomatic complexity.',
      inputSchema: {
        type: 'object',
        properties: {
          complexity_threshold: { type: 'number' },
          min_jsdoc_words: { type: 'number' },
          source_paths: {
            type: 'array',
            items: { type: 'string' },
          },
        },
        additionalProperties: false,
      },
      handler: (argumentsObject) => runDocsQualityMetrics({
        complexityThreshold: argumentsObject.complexity_threshold,
        minJsdocWords: argumentsObject.min_jsdoc_words,
        scope: Array.isArray(argumentsObject.source_paths) && argumentsObject.source_paths.length > 0 ? 'paths' : 'src',
        sourcePaths: Array.isArray(argumentsObject.source_paths) ? argumentsObject.source_paths : undefined,
      }),
    }),
  ];
}

/**
 * Run a self-check against the Repo Cortex MCP server to validate the semantic index.
 *
 * Invokes the `index_stats` tool internally and reports an error when the
 * index is empty or unreachable. Suitable for CI gate validation.
 *
 * @param {{ databasePath?: string }} [options={}] - Optional database path override.
 * @returns {Promise<Record<string, unknown>>} Self-check report in the standard `{ ok, issues, ... }` format.
 */
export async function runSelfCheck(options = {}) {
  const server = createRepoCortexMcpServer({ databasePath: options.databasePath });
  const issues = [];
  let stats = null;

  try {
    const statsResult = await invokeServerRequest(server, {
      method: 'tools/call',
      params: { name: 'index_stats', arguments: {} },
    });
    stats = statsResult.structuredContent;
    if (!stats || Number(stats.total_chunks) < 1) {
      issues.push(selfCheckError('data/semantic-index.sqlite', 'Semantic index has no chunks.'));
    }
  } catch (error) {
    issues.push(selfCheckError('data/semantic-index.sqlite', error instanceof Error ? error.message : String(error)));
  }

  return createSelfCheckReport('repo-cortex-mcp', issues, { stats });
}

/**
 * Extract an explicit database path from CLI arguments.
 *
 * Accepts `--databasePath=<path>` or `--database=<path>` for convenience.
 *
 * @param {string[]} argv - CLI argument list (excluding node executable and script path).
 * @returns {string | undefined} Database path string, or `undefined` if not provided.
 */
function parseDatabasePath(argv) {
  return argv.find((argument) => argument.startsWith('--databasePath='))?.slice('--databasePath='.length)
    ?? argv.find((argument) => argument.startsWith('--database='))?.slice('--database='.length);
}

/**
 * CLI entrypoint for the Repo Cortex MCP server.
 *
 * Supports `--help`, `--self-check`, `--json`, and `--databasePath=<path>`.
 * Without flags, starts the stdio MCP server.
 *
 * @returns {Promise<void>}
 */
async function main() {
  const argv = process.argv.slice(2);
  const options = parseMcpCliArgs(argv);
  const databasePath = parseDatabasePath(argv);
  const tools = createRepoCortexTools(databasePath);

  if (options.help) {
    printMcpUsage({
      title: 'Repo Cortex MCP server',
      entrypoint: ENTRYPOINT,
      summary: 'Expose the semantic repository corpus index as dependency-light stdio MCP tools.',
      tools,
    });
    return;
  }

  if (options.selfCheck) {
    emitSelfCheckReport(await runSelfCheck({ databasePath }), { json: options.json });
    return;
  }

  await runStdioMcpServer(createRepoCortexMcpServer({ databasePath }));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();