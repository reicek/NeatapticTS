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
import { scanCodeQuality } from '../semantic-index/code-quality-scanner.mjs';
import { searchCorpus } from './tools/search-corpus.mjs';

const SERVER_VERSION = '0.1.0';
const ENTRYPOINT = 'scripts/mcp-semantic/repo-cortex-mcp.mjs';

export function createRepoCortexMcpServer(options = {}) {
  const databasePath = options.databasePath;
  const tools = createRepoCortexTools(databasePath);
  return createMcpServer({
    serverName: 'neataptic-cortex-mcp',
    serverVersion: SERVER_VERSION,
    tools,
  });
}

export function createRepoCortexTools(databasePath) {
  return [
    createTool({
      name: 'search_corpus',
      description: 'BM25 full-text search over indexed repository chunks with optional hybrid dense reranking.',
      inputSchema: {
        type: 'object',
        properties: {
          query: { type: 'string' },
          limit: { type: 'number' },
          family: { type: 'string' },
          use_dense: { type: 'boolean' },
          alpha: { type: 'number' },
        },
        required: ['query'],
        additionalProperties: false,
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
      handler: (argumentsObject) => scanCodeQuality({
        complexityThreshold: argumentsObject.complexity_threshold,
        minJsdocWords: argumentsObject.min_jsdoc_words,
        sourcePaths: Array.isArray(argumentsObject.source_paths) ? argumentsObject.source_paths : undefined,
      }),
    }),
  ];
}

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

function parseDatabasePath(argv) {
  return argv.find((argument) => argument.startsWith('--databasePath='))?.slice('--databasePath='.length)
    ?? argv.find((argument) => argument.startsWith('--database='))?.slice('--database='.length);
}

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