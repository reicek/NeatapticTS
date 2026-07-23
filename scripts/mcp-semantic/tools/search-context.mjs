/**
 * @module search-context
 * @description Search the Repo Cortex corpus and assemble a token-bounded context window.
 *
 * Combines hybrid corpus retrieval from {@link searchCorpus} with the
 * five-stage {@link assembleContext} pipeline (enrich → deduplicate → order →
 * budget → stitch). The result is a ready-to-send context payload for an LLM,
 * annotated with provenance metadata and dense-search health.
 *
 * @example
 * ```js
 * import { searchContext } from './search-context.mjs';
 *
 * const result = await searchContext({
 *   query: 'NEAT activation',
 *   limit: 5,
 *   budget: 2048,
 *   context_format: 'markdown',
 * });
 * console.log(result.context, result.token_count, result.dense_state);
 * ```
 */

import { requireString } from '../../agent-customization/mcp/mcp-utils.mjs';
import { normalizeLimit } from './cortex-db.mjs';
import { buildResponseFreshness, searchCorpus } from './search-corpus.mjs';
import { assembleContext } from '../../../rag-index/assemble-context.mjs';

/**
 * Default token budget for an assembled context window.
 * Kept small (800-1200 tokens) so the response fits in an LLM context window.
 * @type {number}
 */
const DEFAULT_BUDGET = 1024;

/**
 * Default result limit passed to corpus retrieval.
 * @type {number}
 */
const DEFAULT_LIMIT = 5;

/**
 * Maximum characters for the assembled context string in compact mode.
 * @type {number}
 */
const COMPACT_CONTEXT_THRESHOLD = 2000;

/**
 * Default context serialization format.
 * @type {'markdown'|'json'}
 */
const DEFAULT_FORMAT = 'markdown';

/**
 * @typedef {Object} SearchContextOptions
 * @property {string} query - Free-text query (required).
 * @property {number} [limit=10] - Maximum number of corpus results to retrieve.
 * @property {number} [budget=4096] - Token budget for the assembled context.
 * @property {'markdown'|'json'} [context_format='markdown'] - Output serialization format.
 * @property {boolean} [use_dense=true] - Enable hybrid dense reranking when the index is warm.
 * @property {boolean} [use_rerank=false] - Enable cross-encoder re-ranking after dense retrieval.
 * @property {number} [alpha] - BM25/dense blend weight (0 = BM25 only, 1 = dense only).
 * @property {boolean|'domain-only'} [expand_query=false] - Enable query expansion.
 * @property {number} [rerank_candidates_count=50] - Number of hybrid candidates to re-rank.
 * @property {string} [slice_id] - Optional slice identifier filter forwarded to corpus search.
 * @property {number} [step_number] - Optional step number filter forwarded to corpus search.
 * @property {string} [databasePath] - Optional corpus database path override.
 * @property {Function} [assembleContextFn] - Optional assembler override for testing.
 */

/**
 * @typedef {Object} SearchContextResult
 * @property {string|Object} context - Assembled context (Markdown string or JSON object).
 * @property {number} token_count - Estimated tokens in the assembled context.
 * @property {{essential: number, supporting: number, supplementary: number}} tier_counts - Tier counts.
 * @property {string} dense_state - Dense-readiness state: 'cold', 'model-only', 'warm', or 'none'.
 * @property {boolean} [dense_degraded] - Present and true when dense search fell back to BM25.
 * @property {'markdown'|'json'} context_format - Effective output format.
 * @property {Object} metadata - Provenance metadata for MCP consumers.
 */

/**
 * Validate and normalize the requested context serialization format.
 *
 * @param {unknown} format
 * @returns {'markdown'|'json'}
 * @throws {Error} When the format is not supported.
 */
function validateContextFormat(format) {
  const normalizedFormat = format ?? DEFAULT_FORMAT;
  if (normalizedFormat !== 'markdown' && normalizedFormat !== 'json') {
    throw new Error(`context_format must be 'markdown' or 'json'.`);
  }
  return normalizedFormat;
}

/**
 * Validate and normalize the token budget.
 *
 * @param {unknown} budget
 * @returns {number}
 * @throws {Error} When the budget is not a positive finite number.
 */
function validateBudget(budget) {
  const numericBudget = Number(budget ?? DEFAULT_BUDGET);
  if (!Number.isFinite(numericBudget) || numericBudget <= 0) {
    throw new Error('budget must be a positive finite number.');
  }
  return numericBudget;
}

/**
 * Convert a {@link searchCorpus} result row into an {@link assembleContext} chunk.
 *
 * searchCorpus exposes the chunk body as `text`, while assembleContext expects
 * `body_text` (or `content`). This helper bridges the two naming conventions
 * without mutating the original result.
 *
 * @param {Object} result - Raw corpus search result.
 * @returns {Object} Chunk compatible with assembleContext.
 */
export function normalizeSearchResultToChunk(result) {
  return {
    ...result,
    body_text: result.text ?? result.body_text ?? '',
  };
}

/**
 * Build the inline top_result descriptor from the first raw corpus result.
 *
 * @param {object | undefined} result - Top raw corpus result.
 * @returns {{ chunk_id: number, file_path: string, family: string, text: string } | null} Top result descriptor.
 */
export function buildTopResult(result) {
  if (!result) {
    return null;
  }
  return {
    chunk_id: result.chunk_id,
    file_path: result.file_path,
    family: result.family,
    text: result.body_text ?? result.text ?? '',
  };
}

/**
 * Build suggested follow-up tool calls from the top corpus results.
 *
 * @param {Array<object>} results - Raw corpus search results.
 * @param {string} query - Original query string.
 * @returns {Array<{ tool: string, args: object, reason: string }>} Follow-up refs.
 */
export function buildFollowUpRefs(results, query) {
  const refs = [];
  if (results[0]) {
    refs.push({
      tool: 'load_chunk',
      args: { chunk_id: results[0].chunk_id, query },
      reason: 'Full text of the top-ranked result',
    });
  }
  if (results[1]) {
    refs.push({
      tool: 'load_chunk',
      args: { chunk_id: results[1].chunk_id, query },
      reason: 'Next sequential chunk in the same document',
    });
  }
  refs.push({
    tool: 'search_context',
    args: { query: `Related: ${query}` },
    reason: 'Explore related context',
  });
  return refs;
}

/**
 * Search the corpus and assemble a bounded context window.
 *
 * @param {SearchContextOptions} [options={}]
 * @returns {Promise<SearchContextResult>}
 * @throws {Error} When required inputs are missing or invalid.
 */
export async function searchContext(options = {}) {
  const query = requireString(options.query, 'query');
  const limit = normalizeLimit(options.limit, DEFAULT_LIMIT);
  const budget = validateBudget(options.budget);
  const contextFormat = validateContextFormat(options.context_format);
  const dedupStrategy = options.dedup_strategy ?? 'exact';
  const fetchLimit = dedupStrategy === 'exact' ? limit * 2 : limit;
  const includeMetadata = options.include_metadata === true;
  const compact = options.compact === true;
  const readTopResult = options.read_top_result === true;

  const searchFn = options.searchCorpusFn ?? searchCorpus;
  const searchResponse = await searchFn({
    query,
    limit: fetchLimit,
    use_dense: options.use_dense !== false,
    use_rerank: options.use_rerank === true,
    alpha: options.alpha,
    expand_query: options.expand_query,
    rerank_candidates_count: options.rerank_candidates_count,
    slice_id: options.slice_id,
    step_number: options.step_number,
    metadata: options.metadata,
    databasePath: options.databasePath,
    client: options.client,
  });

  const chunks = (searchResponse.results ?? []).map(
    normalizeSearchResultToChunk,
  );
  const topResult = readTopResult ? buildTopResult(chunks[0]) : null;
  const followUpRefs = buildFollowUpRefs(chunks, query);
  const assembleFn = options.assembleContextFn ?? assembleContext;
  const assembled = await assembleFn(chunks, {
    budget,
    context_format: contextFormat,
    query_class: searchResponse.query_class,
    client: options.client,
  });

  const denseState =
    typeof searchResponse.dense_state === 'string'
      ? searchResponse.dense_state
      : 'none';
  const rerankState =
    typeof searchResponse.rerank_state === 'string'
      ? searchResponse.rerank_state
      : 'not_requested';

  const tierCounts = assembled.tierCounts ?? {
    essential: 0,
    supporting: 0,
    supplementary: 0,
  };

  const selectedChunks = (assembled.selectedChunks ?? []).slice(0, limit);
  const tokenCount = assembled.tokenCount ?? 0;
  const budgetRemaining = Math.max(0, budget - tokenCount);

  const results = compact
    ? selectedChunks.map((chunk) => ({
        chunk_id: chunk.chunk_id,
        truncated: chunk.truncated === true,
      }))
    : selectedChunks.map((chunk) => {
        const result = {
          chunk_id: chunk.chunk_id,
          feedback_boost: chunk.feedback_boost ?? 0,
          truncated: chunk.truncated === true,
        };
        if (includeMetadata) {
          result.metadata = {
            file_path: chunk.file_path ?? null,
            family: chunk.family ?? null,
            chunk_index: chunk.chunk_index ?? null,
            heading_path: chunk.heading_path ?? null,
            depth: chunk.depth ?? null,
            parent_chunk_id: chunk.parent_chunk_id ?? null,
            context_header: chunk.context_header ?? null,
            symbol_name: chunk.symbol_name ?? null,
            signature_text: chunk.signature_text ?? null,
            jsdoc_text: chunk.jsdoc_text ?? null,
            export_type: chunk.export_type ?? null,
            module_path: chunk.module_path ?? null,
            arch_layer: chunk.arch_layer ?? null,
            jsdoc_quality: chunk.jsdoc_quality ?? null,
            jsdoc_word_count: chunk.jsdoc_word_count ?? null,
            cyclomatic_complexity: chunk.cyclomatic_complexity ?? null,
            test_coverage: chunk.test_coverage ?? null,
            source_path_pattern: chunk.source_path_pattern ?? null,
          };
        }
        return result;
      });

  let context = assembled.context;
  const isJsonContext = contextFormat === 'json';
  if (isJsonContext) {
    context = {
      context: assembled.context,
      chunks: selectedChunks.map((chunk) => ({
        chunk_id: chunk.chunk_id ?? null,
        file_path: chunk.file_path,
        heading_path: chunk.heading_path ?? null,
        char_start: chunk.char_start ?? null,
        char_end: chunk.char_end ?? null,
        content: String(chunk.body_text ?? chunk.content ?? chunk.text ?? ''),
        score: chunk.score ?? null,
        tier: chunk.tier ?? null,
      })),
      tokenCount,
    };
  } else if (
    compact &&
    typeof context === 'string' &&
    context.length > COMPACT_CONTEXT_THRESHOLD
  ) {
    context = `${context.slice(0, COMPACT_CONTEXT_THRESHOLD - 1)}…`;
  }

  return {
    context,
    compact,
    token_count: tokenCount,
    tier_counts: tierCounts,
    dense_state: denseState,
    rerank_state: rerankState,
    context_format: contextFormat,
    total_chunks_retrieved: chunks.length,
    chunks_in_context: selectedChunks.length,
    tokens_used: tokenCount,
    context_budget_consumed: tokenCount,
    budget_remaining: budgetRemaining,
    dedup_strategy: dedupStrategy,
    results,
    ...(searchResponse.dense_degraded === true ? { dense_degraded: true } : {}),
    freshness:
      searchResponse.freshness ??
      (await buildResponseFreshness(options.databasePath, options.client)),
    metadata: {
      chunkCount: chunks.length,
      totalTokens: tokenCount,
      budget,
      tiersIncluded: tierCounts,
      truncated: selectedChunks.some((chunk) => chunk.truncated === true),
      format: contextFormat,
    },
    ...(topResult ? { top_result: topResult } : {}),
    follow_up_refs: followUpRefs,
  };
}

/**
 * Alias exported for callers that name the handler `searchContextTool`.
 *
 * Points to the same implementation as {@link searchContext} so server
 * registration and direct imports can use either name.
 */
export const searchContextTool = searchContext;

export { validateBudget, validateContextFormat };
