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
 *   MCP --> load_parent_chunk
 *   MCP --> load_document
 *   MCP --> freshness_check
 *   MCP --> index_stats
 *   MCP --> list_families
 *   MCP --> scan_code_quality
 *   MCP --> expand_query
 *   MCP --> submit_feedback
 *   search_corpus --> searchCorpus
 *   load_chunk --> loadChunk
 *   load_parent_chunk --> loadParentChunk
 *   load_document --> loadDocument
 *   freshness_check --> freshnessCheck
 *   index_stats --> indexStats
 *   list_families --> listFamilies
 *   scan_code_quality --> runDocsQualityMetrics
 *   expand_query --> expandQuery
 *   submit_feedback --> submitFeedback
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
import { expandQueryHandler } from './tools/expand-query.mjs';
import { freshnessCheck } from './tools/freshness-check.mjs';
import { indexStats } from './tools/index-stats.mjs';
import { listFamilies } from './tools/list-families.mjs';
import { loadChunk } from './tools/load-chunk.mjs';
import { loadDocument } from './tools/load-document.mjs';
import { loadParentChunk } from './tools/load-parent-chunk.mjs';
import { runDocsQualityMetrics } from '../semantic-index/docs-quality/docs-quality.metrics.mjs';
import { searchCorpus } from './tools/search-corpus.mjs';
import { submitFeedback } from './tools/submit-feedback.mjs';
import { traverseGraphHandler } from './tools/traverse-graph.mjs';

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
      description:
        'BM25 full-text search over indexed repository chunks with default-on hybrid dense reranking. Supports optional query classification for classification-aware alpha and family filter selection. If embeddings are cold or model-only, the tool returns BM25 results with dense_state, dense_degraded, and dense_reason; run npm run index:prewarm to warm dense search. When use_rerank is true, cross-encoder re-ranking is applied to the top hybrid candidates; if the reranker is cold or model-only, the response includes rerank_degraded, rerank_state, and rerank_reason.',
      inputSchema: {
        type: 'object',
        properties: {
          query: { type: 'string' },
          limit: { type: 'number' },
          family: { type: 'string' },
          use_dense: {
            type: 'boolean',
            default: true,
            description:
              'Defaults to true. Set false for BM25-only search; run npm run index:prewarm when dense_state reports cold or model-only.',
          },
          alpha: {
            type: 'number',
            description:
              'BM25/dense blend weight (0 = BM25 only, 1 = dense only). When omitted and query_class is not specified, classification determines alpha automatically.',
          },
          query_class: {
            type: 'string',
            enum: [
              'simple_lookup',
              'cross_boundary',
              'multi_hop',
              'exploratory',
              'code_specific',
              'plan_specific',
            ],
            description:
              'Override query classification for routing. When provided, alpha and family are set from the routing table unless explicitly overridden.',
          },
          classification_hints: {
            type: 'object',
            description:
              'Optional overrides for classification-derived alpha and family. When provided, these take precedence over automatic classification.',
            properties: {
              alpha: {
                type: 'number',
                description: 'Override alpha from classification routing.',
              },
              family: {
                type: 'string',
                description:
                  'Override family filter from classification routing.',
              },
            },
          },
          use_rerank: {
            type: 'boolean',
            default: false,
            description:
              'Defaults to false. Set true to apply cross-encoder re-ranking to the top hybrid candidates after dense retrieval; if the reranker is cold or model-only, the response includes rerank_degraded=true.',
          },
          rerank_candidates_count: {
            type: 'number',
            description:
              'Number of hybrid candidates to re-rank with the cross-encoder (default: 50). Ignored when use_rerank is false.',
          },
          expand_query: {
            description:
              'Enable query expansion before search. true for full expansion (domain associations + embedding synonyms), "domain-only" for domain associations only, false (default) for no expansion.',
            oneOf: [
              { type: 'boolean' },
              { type: 'string', enum: ['domain-only'] },
            ],
          },
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
          use_rerank: {
            type: 'boolean',
            description:
              'Whether cross-encoder re-ranking was applied to the results.',
          },
          rerank_candidates_count: {
            type: 'number',
            description:
              'Number of hybrid candidates sent to the cross-encoder for re-ranking.',
          },
          rerank_degraded: {
            type: 'boolean',
            description:
              'True when use_rerank was requested but the reranker was cold or model-only, so results are not re-ranked.',
          },
          rerank_state: {
            type: 'string',
            enum: ['cold', 'model-only', 'warm'],
            description: 'Reranker readiness state: cold, model-only, or warm.',
          },
          rerank_reason: {
            type: 'string',
            description:
              'Human-readable degradation reason when cross-encoder re-ranking fell back; run npm run index:prewarm:reranker.',
          },
          expansion: {
            type: 'object',
            description:
              'Query expansion metadata. Present when expand_query is enabled.',
            properties: {
              applied: {
                type: 'boolean',
                description: 'Whether query expansion was applied.',
              },
              degraded: {
                type: 'boolean',
                description:
                  'True when expansion degraded (e.g., ONNX model unavailable).',
              },
              reason: {
                type: 'string',
                description:
                  'Reason when expansion was not applied or degraded.',
              },
            },
          },
          query_class: {
            type: 'string',
            description:
              'Detected or specified query classification: simple_lookup, cross_boundary, multi_hop, exploratory, code_specific, or plan_specific.',
            enum: [
              'simple_lookup',
              'cross_boundary',
              'multi_hop',
              'exploratory',
              'code_specific',
              'plan_specific',
            ],
          },
          confidence: {
            type: 'number',
            description:
              'Classification confidence (0–1). Higher is more confident.',
          },
          classification_fallback: {
            type: 'boolean',
            description:
              'True when classification confidence was below the degradation threshold and the system fell back to simple_lookup with alpha 0.50.',
          },
          dense_degraded: {
            type: 'boolean',
            description:
              'True when default-on or requested dense search degraded to BM25 because embeddings were cold or model-only.',
          },
          dense_state: {
            type: 'string',
            enum: ['cold', 'model-only', 'warm'],
            description:
              'Dense-readiness provenance emitted on default-on dense responses: cold, model-only, or warm.',
          },
          dense_reason: {
            type: 'string',
            description:
              'Human-readable degradation reason when dense search falls back to BM25; operators should run npm run index:prewarm.',
          },
          results: {
            type: 'array',
            items: { type: 'object' },
          },
        },
        required: ['query', 'limit', 'use_dense', 'results'],
        additionalProperties: true,
      },
      handler: (argumentsObject) =>
        searchCorpus({ ...argumentsObject, databasePath }),
    }),
    createTool({
      name: 'load_chunk',
      description:
        'Load one indexed corpus chunk by numeric chunk ID. Returns v2 semantic metadata including depth, parent_chunk_id, context_header, symbol_name, signature_text, jsdoc_text, export_type, and module_path.',
      inputSchema: {
        type: 'object',
        properties: {
          chunk_id: { type: 'number' },
          query: {
            type: 'string',
            description:
              'Optional originating query for click correlation and deduplication.',
          },
        },
        required: ['chunk_id'],
        additionalProperties: false,
      },
      outputSchema: {
        type: 'object',
        properties: {
          chunk: {
            type: 'object',
            properties: {
              chunk_id: { type: 'number' },
              chunk_index: { type: 'number' },
              file_path: { type: 'string' },
              family: { type: 'string' },
              heading_path: { type: 'string' },
              text: { type: 'string' },
              char_start: { type: 'number' },
              char_end: { type: 'number' },
              depth: {
                type: 'number',
                description:
                  'Chunk depth: 0 for top-level, 1 for sub-chunks within a parent.',
              },
              parent_chunk_id: {
                type: 'number',
                description:
                  'Database ID of the parent chunk for depth-1 sub-chunks, or null for depth-0.',
              },
              context_header: {
                type: 'string',
                description:
                  'Agent-facing context header, e.g. "[src/file.ts > ClassName > methodName]"',
              },
              symbol_name: {
                type: 'string',
                description: 'Exported symbol name for TypeScript chunks.',
              },
              signature_text: {
                type: 'string',
                description: 'Full signature text for TypeScript symbols.',
              },
              jsdoc_text: {
                type: 'string',
                description: 'JSDoc summary text for the symbol.',
              },
              export_type: {
                type: 'string',
                description:
                  'Export kind: "export", "default", "reexport", or null.',
              },
              module_path: {
                type: 'string',
                description: 'Re-export target module path, or null.',
              },
            },
          },
        },
      },
      handler: (argumentsObject) =>
        loadChunk({ ...argumentsObject, databasePath }),
    }),
    createTool({
      name: 'load_parent_chunk',
      description:
        'Load the parent chunk for a given depth-1 sub-chunk by its numeric chunk ID. Returns the full parent chunk descriptor with v2 semantic metadata.',
      inputSchema: {
        type: 'object',
        properties: {
          chunk_id: {
            type: 'number',
            description:
              'Chunk ID of a depth-1 sub-chunk whose parent to load.',
          },
        },
        required: ['chunk_id'],
        additionalProperties: false,
      },
      outputSchema: {
        type: 'object',
        properties: {
          parent_chunk: {
            type: 'object',
            properties: {
              chunk_id: { type: 'number' },
              chunk_index: { type: 'number' },
              file_path: { type: 'string' },
              family: { type: 'string' },
              heading_path: { type: 'string' },
              text: { type: 'string' },
              char_start: { type: 'number' },
              char_end: { type: 'number' },
              depth: { type: 'number' },
              parent_chunk_id: { type: 'number' },
              context_header: { type: 'string' },
              symbol_name: { type: 'string' },
              signature_text: { type: 'string' },
              jsdoc_text: { type: 'string' },
              export_type: { type: 'string' },
              module_path: { type: 'string' },
            },
          },
        },
      },
      handler: (argumentsObject) =>
        loadParentChunk({ ...argumentsObject, databasePath }),
    }),
    createTool({
      name: 'load_document',
      description:
        'Load all ordered chunks for one indexed repository path. Returns v2 semantic metadata per chunk plus a hierarchy summary (depth_0_count, depth_1_count, has_sub_chunks).',
      inputSchema: {
        type: 'object',
        properties: { file_path: { type: 'string' } },
        required: ['file_path'],
        additionalProperties: false,
      },
      outputSchema: {
        type: 'object',
        properties: {
          file_path: { type: 'string' },
          chunks: { type: 'array', items: { type: 'object' } },
          hierarchy: {
            type: 'object',
            description:
              'Summary of chunk depth distribution within this document.',
            properties: {
              depth_0_count: {
                type: 'number',
                description: 'Number of top-level (depth-0) chunks.',
              },
              depth_1_count: {
                type: 'number',
                description:
                  'Number of sub-chunks (depth-1) within parent chunks.',
              },
              has_sub_chunks: {
                type: 'boolean',
                description:
                  'True when the document contains depth-1 sub-chunks.',
              },
            },
          },
        },
      },
      handler: (argumentsObject) =>
        loadDocument({ ...argumentsObject, databasePath }),
    }),
    createTool({
      name: 'freshness_check',
      description:
        'Compare indexed document freshness proofs with current filesystem metadata.',
      inputSchema: {
        type: 'object',
        properties: {
          file_path: { type: 'string' },
          freshnessProof: { type: 'object' },
        },
        additionalProperties: false,
      },
      handler: (argumentsObject) =>
        freshnessCheck({ ...argumentsObject, databasePath }),
    }),
    createTool({
      name: 'index_stats',
      description:
        'Return corpus row counts, family counts, and last indexed timestamp.',
      handler: () => indexStats({ databasePath }),
    }),
    createTool({
      name: 'list_families',
      description:
        'List indexed document families with document and chunk counts.',
      handler: () => listFamilies({ databasePath }),
    }),
    createTool({
      name: 'scan_code_quality',
      description:
        'Scan exported TypeScript symbols for missing or weak JSDoc and high cyclomatic complexity.',
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
      handler: (argumentsObject) =>
        runDocsQualityMetrics({
          complexityThreshold: argumentsObject.complexity_threshold,
          minJsdocWords: argumentsObject.min_jsdoc_words,
          scope:
            Array.isArray(argumentsObject.source_paths) &&
            argumentsObject.source_paths.length > 0
              ? 'paths'
              : 'src',
          sourcePaths: Array.isArray(argumentsObject.source_paths)
            ? argumentsObject.source_paths
            : undefined,
        }),
    }),
    createTool({
      name: 'traverse_graph',
      description:
        'Traverse the entity/relationship graph from seed entities, following specified relationship types for a configurable number of hops. Returns discovered entities, relationships, and associated chunk_ids for context expansion.',
      inputSchema: {
        type: 'object',
        properties: {
          seed_names: {
            type: 'array',
            items: { type: 'string' },
            description:
              'Qualified names or partial names of seed entities to start traversal from.',
          },
          seed_query: {
            type: 'string',
            description:
              'Free-text query to discover seed entities via name/qualified_name search.',
          },
          relationship_types: {
            type: 'array',
            items: {
              type: 'string',
              enum: [
                'imports',
                'exports',
                'depends-on',
                'implements',
                'references',
                'owns',
                'part-of',
                'contains',
              ],
            },
            default: [
              'imports',
              'exports',
              'depends-on',
              'implements',
              'references',
              'owns',
              'part-of',
              'contains',
            ],
            description: 'Relationship types to follow during traversal.',
          },
          entity_types: {
            type: 'array',
            items: {
              type: 'string',
              enum: [
                'module',
                'class',
                'function',
                'interface',
                'type-alias',
                'variable',
                'error-class',
                'plan',
                'skill',
                'agent',
                'demo',
                'benchmark',
              ],
            },
            default: [
              'module',
              'class',
              'function',
              'interface',
              'type-alias',
              'variable',
              'error-class',
              'plan',
              'skill',
              'agent',
              'demo',
              'benchmark',
            ],
            description: 'Entity types to include in results.',
          },
          max_hops: {
            type: 'number',
            default: 2,
            description:
              'Maximum number of hops from seed entities. Default: 2, max: 3.',
          },
          max_results: {
            type: 'number',
            default: 20,
            description:
              'Maximum number of entities to return. Default: 20, max: 50.',
          },
          confidence_filter: {
            type: 'array',
            items: { type: 'string', enum: ['high', 'medium', 'low'] },
            default: ['high', 'medium'],
            description: 'Minimum confidence levels to follow.',
          },
        },
        required: [],
        additionalProperties: false,
      },
      outputSchema: {
        type: 'object',
        properties: {
          seed_entities: { type: 'array', items: { type: 'object' } },
          entities: { type: 'array', items: { type: 'object' } },
          relationships: { type: 'array', items: { type: 'object' } },
          chunk_ids: { type: 'array', items: { type: 'number' } },
          doc_ids: { type: 'array', items: { type: 'number' } },
          hop_count: { type: 'number' },
          total_discovered: { type: 'number' },
          returned_count: { type: 'number' },
          graph_available: { type: 'boolean' },
        },
        required: [
          'seed_entities',
          'entities',
          'relationships',
          'chunk_ids',
          'doc_ids',
          'hop_count',
          'total_discovered',
          'returned_count',
          'graph_available',
        ],
      },
      handler: (argumentsObject) => traverseGraphHandler(argumentsObject),
    }),
    createTool({
      name: 'expand_query',
      description:
        'Expand a search query using domain associations and embedding-based synonym discovery. Returns expanded terms, an OR-expanded BM25 query, and expansion metadata. Supports classification-aware expansion behavior (simple_lookup=false, code_specific/plan_specific=domain-only, cross_boundary/multi_hop/exploratory=full).',
      inputSchema: {
        type: 'object',
        properties: {
          query: {
            type: 'string',
            description: 'Free-text query string to expand.',
          },
          expand_query: {
            description:
              'Enable query expansion. true for full expansion (domain associations + embedding synonyms), "domain-only" for domain associations only, false (default) for no expansion.',
            oneOf: [
              { type: 'boolean' },
              { type: 'string', enum: ['domain-only'] },
            ],
          },
          query_class: {
            type: 'string',
            enum: [
              'simple_lookup',
              'cross_boundary',
              'multi_hop',
              'exploratory',
              'code_specific',
              'plan_specific',
            ],
            description:
              'Override query classification for expansion behavior. When provided, expansion behavior is determined by expansionBehaviorForClass().',
          },
        },
        required: ['query'],
        additionalProperties: false,
      },
      outputSchema: {
        type: 'object',
        properties: {
          original_query: { type: 'string' },
          expanded_terms: {
            type: 'array',
            items: { type: 'object' },
            description:
              'Array of expanded term objects with original, expanded, source, confidence, similarity, relevanceScore, and type fields.',
          },
          bm25_query: {
            type: 'string',
            description:
              'OR-expanded FTS5 query string, or null if no expansion applied.',
          },
          expansion: {
            type: 'object',
            properties: {
              applied: {
                type: 'boolean',
                description: 'Whether expansion was applied.',
              },
              degraded: {
                type: 'boolean',
                description: 'True when expansion degraded.',
              },
              reason: {
                type: 'string',
                description:
                  'Reason when expansion was not applied or degraded.',
              },
            },
          },
        },
        required: ['original_query', 'expanded_terms', 'expansion'],
        additionalProperties: true,
      },
      handler: (argumentsObject) => expandQueryHandler({ ...argumentsObject }),
    }),
    createTool({
      name: 'submit_feedback',
      description:
        'Submit an explicit feedback signal for a corpus chunk. Records reference, positive, or negative events and recomputes the chunk feedback boost score.',
      inputSchema: {
        type: 'object',
        properties: {
          chunk_id: {
            type: 'number',
            description: 'Numeric chunk identifier that received the feedback.',
          },
          signal_type: {
            type: 'string',
            enum: ['reference', 'positive', 'negative'],
            description:
              'Feedback signal type: reference (cited), positive (helpful), or negative (not helpful).',
          },
          context: {
            type: 'string',
            description:
              'Optional free-text context explaining the feedback (truncated to 500 characters).',
          },
          query: {
            type: 'string',
            description: 'Optional originating query for correlation.',
          },
          agent_id: {
            type: 'string',
            description: 'Optional agent identifier that submitted the feedback.',
          },
        },
        required: ['chunk_id', 'signal_type'],
        additionalProperties: false,
      },
      outputSchema: {
        type: 'object',
        properties: {
          chunk_id: { type: 'number' },
          signal_type: { type: 'string' },
          recorded: {
            type: 'boolean',
            description: 'True when the event was persisted.',
          },
          feedback_boost_after: {
            type: 'number',
            description: 'Recomputed feedback boost score after recording the event.',
          },
        },
        required: ['chunk_id', 'signal_type', 'recorded', 'feedback_boost_after'],
        additionalProperties: false,
      },
      handler: (argumentsObject) =>
        submitFeedback({ ...argumentsObject, databasePath }),
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
  const server = createRepoCortexMcpServer({
    databasePath: options.databasePath,
  });
  const issues = [];
  let stats = null;

  try {
    const statsResult = await invokeServerRequest(server, {
      method: 'tools/call',
      params: { name: 'index_stats', arguments: {} },
    });
    stats = statsResult.structuredContent;
    if (!stats || Number(stats.total_chunks) < 1) {
      issues.push(
        selfCheckError(
          'data/semantic-index.sqlite',
          'Semantic index has no chunks.',
        ),
      );
    }
  } catch (error) {
    issues.push(
      selfCheckError(
        'data/semantic-index.sqlite',
        error instanceof Error ? error.message : String(error),
      ),
    );
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
  return (
    argv
      .find((argument) => argument.startsWith('--databasePath='))
      ?.slice('--databasePath='.length) ??
    argv
      .find((argument) => argument.startsWith('--database='))
      ?.slice('--database='.length)
  );
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
      summary:
        'Expose the semantic repository corpus index as dependency-light stdio MCP tools.',
      tools,
    });
    return;
  }

  if (options.selfCheck) {
    emitSelfCheckReport(await runSelfCheck({ databasePath }), {
      json: options.json,
    });
    return;
  }

  await runStdioMcpServer(createRepoCortexMcpServer({ databasePath }));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();
