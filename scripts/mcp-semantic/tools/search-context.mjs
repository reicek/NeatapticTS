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
import { searchCorpus } from './search-corpus.mjs';
import { assembleContext } from '../../semantic-index/assemble-context.mjs';

/**
 * Default token budget for an assembled context window.
 * @type {number}
 */
const DEFAULT_BUDGET = 4096;

/**
 * Default result limit passed to corpus retrieval.
 * @type {number}
 */
const DEFAULT_LIMIT = 10;

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
 * @property {string} [databasePath] - Optional corpus database path override.
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
function normalizeSearchResultToChunk(result) {
  return {
    ...result,
    body_text: result.text ?? result.body_text ?? '',
  };
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

  const searchResponse = await searchCorpus({
    query,
    limit,
    use_dense: options.use_dense !== false,
    use_rerank: options.use_rerank === true,
    alpha: options.alpha,
    expand_query: options.expand_query,
    rerank_candidates_count: options.rerank_candidates_count,
    databasePath: options.databasePath,
  });

  const chunks = (searchResponse.results ?? []).map(
    normalizeSearchResultToChunk,
  );
  const assembled = await assembleContext(chunks, {
    budget,
    context_format: contextFormat,
    query_class: searchResponse.query_class,
  });

  const denseState =
    typeof searchResponse.dense_state === 'string'
      ? searchResponse.dense_state
      : 'none';

  const tierCounts = assembled.tierCounts ?? {
    essential: 0,
    supporting: 0,
    supplementary: 0,
  };

  const truncated = assembled.selectedChunks?.some(
    (chunk) => chunk.truncated === true,
  );

  return {
    context: assembled.context,
    token_count: assembled.tokenCount,
    tier_counts: tierCounts,
    dense_state: denseState,
    context_format: contextFormat,
    ...(searchResponse.dense_degraded === true
      ? { dense_degraded: true }
      : {}),
    metadata: {
      chunkCount: chunks.length,
      totalTokens: assembled.tokenCount,
      budget,
      tiersIncluded: tierCounts,
      truncated,
      format: contextFormat,
    },
  };
}

/**
 * Alias exported for callers that name the handler `searchContextTool`.
 *
 * Points to the same implementation as {@link searchContext} so server
 * registration and direct imports can use either name.
 */
export const searchContextTool = searchContext;
