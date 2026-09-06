/**
 * @module search-corpus
 * @description Hybrid corpus search tool for the Repo Cortex MCP server.
 *
 * Routes queries through BM25 full-text search or dense-vector reranking
 * depending on dense-index readiness. When the embeddings model and database
 * are both present the tool returns semantically ranked results; when either
 * is absent it degrades gracefully to BM25 and reports the degradation state
 * in `dense_state`, `dense_degraded`, and `dense_reason` response fields.
 *
 * @remarks
 * ### Dense Retrieval Strategy (ANN-first)
 *
 * Dense retrieval is delegated to `queryDenseIndex` (from `query-dense.mjs`),
 * which uses an ANN-first strategy:
 *
 * - **ANN path (primary):** `vector_top_k(chunks_embedding_idx, vector8(?), ?)`
 *   joined with chunks — DiskANN-backed approximate nearest neighbor search.
 *   The k limit for `vector_top_k(idx, ?, ?)` controls candidate pool size.
 *   Full SQL: `SELECT ... FROM vector_top_k(chunks_embedding_idx, vector8(?), ?) AS v
 *   JOIN chunks c ON c.rowid = v.rowid JOIN documents d ON d.doc_id = c.doc_id`
 *
 * - **Brute-force fallback:** `vector_distance_cos(c.embedding, vector8(?))`
 *   ordered by distance — used when the DiskANN index is cold or missing.
 *
 * ### Hybrid Ranking: Reciprocal Ranked Fusion (RRF)
 *
 * When both BM25 and dense results are available, candidates are merged using
 * Reciprocal Ranked Fusion (RRF) with the standard formula
 * `score = sum(1/(k + rank_i))` and default k=60. RRF depends only on rank
 * positions, not raw scores, avoiding the normalization issues of the former
 * alpha-blend weighting approach.
 *
 * ### BM25 vs Dense/Hybrid Decision Flow
 *
 * ```mermaid
 * flowchart TD
 *   A[searchCorpus] --> B{query empty?}
 *   B -- yes --> C{use_dense?}
 *   C -- no  --> D[Empty BM25 response]
 *   C -- yes --> E{dense warm?}
 *   E -- no  --> F[Degraded BM25 response<br/>dense_degraded=true]
 *   E -- yes --> G[Empty dense response<br/>dense_state=warm]
 *   B -- no  --> H{use_dense?}
 *   H -- no  --> I[runBm25Search<br/>BM25-only results]
 *   H -- yes --> J{dense warm?}
 *   J -- no  --> K[Degraded BM25 results<br/>dense_degraded=true]
 *   J -- yes --> L[queryDenseIndex<br/>RRF hybrid results]
 * ```
 */
import { requireString } from '../../agent-customization/mcp/mcp-utils.mjs';
import { queryDenseIndex } from '../../../rag-index/query-dense.mjs';
import { checkDenseReadiness } from '../../../rag-index/dense-readiness.mjs';
import { checkRerankerReadiness } from '../../../rag-index/reranker-readiness.mjs';
import {
  rerankCandidates,
  normalizeRerankCandidates,
} from '../../../rag-index/rerank-index.mjs';
import { getTursoClient, normalizeLimit, readChunkRow } from './cortex-db.mjs';
import { sanitizeFtsQuery } from '../../../rag-index/tokenizer.mjs';
import { classifyForSearchCorpus } from '../../../rag-index/classify-query.mjs';
import { classifyAndRoute } from '../../../rag-index/routing-table.mjs';
import {
  validateFilter,
  compileFilterToSqlAliased,
} from '../../../rag-index/metadata-filter.mjs';
import { expandQuery } from '../../../rag-index/expand-query.mjs';
import { runParallelQueries } from '../../../rag-index/parallel-search.mjs';
import { createHash, randomUUID } from 'node:crypto';
import { ErrorCodes, cortexError } from './cortex-error.mjs';
import {
  DEFAULT_ANN_THRESHOLD,
  resolveDenseStrategy,
} from './ann-strategy.mjs';
import { evaluateSelfHeal } from '../../agent-customization/cortex/cortex-health-guard.mjs';

/**
 * Default maximum number of corpus results returned by searchCorpus.
 * Kept small (3-5) so responses fit comfortably in an LLM context window.
 * @type {number}
 */
const DEFAULT_LIMIT = 5;

/**
 * Maximum characters for a result text snippet in compact mode.
 * @type {number}
 */
const COMPACT_TEXT_THRESHOLD = 300;

/**
 * Rough characters-per-token estimate for response token budgeting.
 * @type {number}
 */
const RESPONSE_CHARS_PER_TOKEN = 4;

/**
 * Process-lifetime cache for the dense-readiness probe result.
 *
 * Set to the readiness report on the first warm response so subsequent
 * calls skip the probe entirely. Reset to `null` when `DENSE_FORCE_STATE`
 * is `cold` or `model-only` so integration tests can override the state.
 *
 * @type {object | null}
 */
let cachedDenseReadiness = null;
let cachedRerankerReadiness = null;

/**
 * Reset the process-lifetime readiness caches.
 *
 * Exported for tests so each test starts with a clean cache and cannot be
 * influenced by a previous probe result. Affected caches:
 * - {@link cachedDenseReadiness}
 * - {@link cachedRerankerReadiness}
 */
export function resetReadinessCaches() {
  cachedDenseReadiness = null;
  cachedRerankerReadiness = null;
}

/**
 * Invalidate only the dense-readiness process-lifetime cache.
 *
 * Exported so a completed self-heal repair (or a caller that knows the
 * dense state has changed) can force the next search call to re-probe the
 * embeddings store without restarting the server process.
 */
export function invalidateDenseReadinessCache() {
  cachedDenseReadiness = null;
}

/**
 * Determine the chunk count to use for dense-strategy selection.
 *
 * Prefers the count from a warm readiness report, then falls back to a direct
 * SQL count so callers that only mock `readinessProbe` still receive the right
 * strategy for their fixture database.
 *
 * @param {string | undefined} databasePath - Corpus database path override.
 * @param {object} readinessReport - Dense readiness report.
 * @returns {number} Chunk count.
 */
async function resolveChunkCountForStrategy(
  databasePath,
  readinessReport,
  client,
) {
  if (
    readinessReport &&
    typeof readinessReport.chunk_count === 'number' &&
    Number.isFinite(readinessReport.chunk_count)
  ) {
    return readinessReport.chunk_count;
  }

  const resolvedClient = client ?? (await getTursoClient(databasePath));
  try {
    const result = await resolvedClient.execute({
      sql: 'SELECT COUNT(*) AS count FROM chunks',
      args: [],
    });
    return Number(result.rows[0]?.count ?? 0);
  } catch {
    return 0;
  }
}

/**
 * Resolve the ANN strategy for a dense search response.
 *
 * @param {object} options - Strategy inputs.
 * @param {string | undefined} options.databasePath - Corpus database path override.
 * @param {object} options.readinessReport - Dense readiness report.
 * @returns {'diskann'} Selected strategy.
 */
async function resolveDenseStrategyForSearch({
  databasePath,
  readinessReport,
  client,
}) {
  const chunkCount = await resolveChunkCountForStrategy(
    databasePath,
    readinessReport,
    client,
  );
  const indexStatus =
    readinessReport?.ann_index_status ??
    (readinessReport?.state === 'warm' ? 'ready' : 'stale');

  return resolveDenseStrategy({
    chunkCount,
    annThreshold: DEFAULT_ANN_THRESHOLD,
    indexStatus,
  });
}

async function searchCorpusImpl(options = {}) {
  const rawQuery = requireString(options.query, 'query');
  const query = sanitizeFtsQuery(rawQuery);
  const limit = normalizeLimit(options.limit, DEFAULT_LIMIT);
  const explicitFamily =
    typeof options.family === 'string' && options.family.trim()
      ? options.family.trim()
      : null;
  const skipFamilyClassification = options.skip_family_classification === true;
  const useDense = options.use_dense !== false;

  // Validate and compile metadata filter if provided.
  // When slice_id or step_number are supplied as top-level options, they are
  // combined with any metadata.filter using AND so both paths use a single
  // compiled SQL fragment.
  const metadataFilter = options.metadata?.filter;
  const extraPredicates = [];
  if (
    typeof options.slice_id === 'string' &&
    options.slice_id.trim().length > 0
  ) {
    extraPredicates.push({
      op: 'eq',
      field: 'slice_id',
      value: options.slice_id,
    });
  }
  if (
    options.step_number !== undefined &&
    options.step_number !== null &&
    String(options.step_number).trim() !== '' &&
    Number.isFinite(Number(options.step_number))
  ) {
    extraPredicates.push({
      op: 'eq',
      field: 'step_number',
      value: Number(options.step_number),
    });
  }

  let combinedFilter = null;
  if (metadataFilter !== undefined && metadataFilter !== null) {
    if (extraPredicates.length > 0) {
      combinedFilter = {
        op: 'and',
        predicates: [metadataFilter, ...extraPredicates],
      };
    } else {
      combinedFilter = metadataFilter;
    }
  } else if (extraPredicates.length > 0) {
    combinedFilter =
      extraPredicates.length === 1
        ? extraPredicates[0]
        : { op: 'and', predicates: extraPredicates };
  }

  const compileFilterFn = options.compileFilterFn ?? compileFilterToSqlAliased;
  let compiledFilter = null;
  if (combinedFilter !== null) {
    try {
      validateFilter(combinedFilter);
      compiledFilter = compileFilterFn(combinedFilter);
    } catch (error) {
      throw cortexError(
        ErrorCodes.INVALID_METADATA_FILTER,
        error instanceof Error ? error.message : String(error),
      );
    }
  }

  // Classification-aware alpha and family selection
  const explicitAlpha = options.alpha;
  const queryClass = options.query_class;
  const classificationHints = options.classification_hints;
  const hintAlpha = classificationHints?.alpha;
  const hintFamily = classificationHints?.family;

  let classificationMetadata = null;

  // Determine alpha and family from classification
  let effectiveAlpha = explicitAlpha;
  let effectiveFamily = explicitFamily;

  if (queryClass) {
    // Explicit class from caller — use full routing
    const routing = classifyAndRoute(rawQuery, classificationHints);
    effectiveAlpha = hintAlpha ?? explicitAlpha ?? routing.alpha;
    effectiveFamily = skipFamilyClassification
      ? explicitFamily
      : (hintFamily ?? explicitFamily ?? routing.strategy.family);
    classificationMetadata = {
      query_class: routing.query_class,
      confidence: routing.confidence,
      classification_fallback: skipFamilyClassification,
      family_fallback: skipFamilyClassification,
    };
  } else if (explicitAlpha === undefined && hintAlpha === undefined) {
    // No explicit alpha and no explicit class → lightweight classification
    const classification = classifyForSearchCorpus(rawQuery);
    effectiveAlpha = classification.alpha;
    effectiveFamily = skipFamilyClassification
      ? explicitFamily
      : (hintFamily ?? explicitFamily ?? classification.family);
    classificationMetadata = {
      query_class: classification.query_class,
      confidence: classification.confidence,
      classification_fallback:
        skipFamilyClassification || classification.classification_fallback,
      family_fallback: skipFamilyClassification,
    };
  } else {
    // Caller supplied explicit alpha/hints without a query_class; still
    // emit classification metadata from lightweight classification.
    const classification = classifyForSearchCorpus(rawQuery);
    effectiveAlpha = hintAlpha ?? explicitAlpha ?? classification.alpha;
    effectiveFamily = skipFamilyClassification
      ? explicitFamily
      : (hintFamily ?? explicitFamily ?? classification.family);
    classificationMetadata = {
      query_class: classification.query_class,
      confidence: classification.confidence,
      classification_fallback:
        skipFamilyClassification || classification.classification_fallback,
      family_fallback: skipFamilyClassification,
    };
  }
  // else: caller provided explicit alpha → respect it, no classification

  const alpha = normalizeAlpha(effectiveAlpha);
  const classifiedFamily = effectiveFamily;

  const useRerank = options.use_rerank === true;
  const rerankCandidatesCount = normalizeRerankCandidates(
    options.rerank_candidates_count,
  );

  // Query expansion pipeline: when expand_query is truthy, expand the query
  // with domain associations and/or embedding-based synonyms before search.
  let expansionMetadata = null;
  let expandedBm25Query = null;

  if (options.expand_query) {
    const expandQueryFn = options.expandQueryFn ?? expandQuery;
    try {
      const expansionResult = await expandQueryFn({
        query: rawQuery,
        expandQuery: options.expand_query,
        modelDirectory: options.modelDirectory,
        modelId: options.modelId,
        associationsPath: options.associationsPath,
      });
      expansionMetadata = expansionResult.expansion;
      if (expansionResult.bm25Query) {
        expandedBm25Query = expansionResult.bm25Query;
      }
    } catch {
      // Expansion failed — fall back to unexpanded search
      expansionMetadata = {
        applied: false,
        degraded: true,
        reason: 'Query expansion failed',
      };
    }
  }

  // Exact symbol fast path: when the raw query names a known symbol, return
  // the matching chunk(s) directly without running BM25 or dense ranking.
  const exactSymbolResponse = await tryExactSymbolLookup({
    classificationMetadata,
    databasePath: options.databasePath,
    family: classifiedFamily,
    limit,
    rawQuery,
    client: options.client,
    compiledFilter,
  });
  if (exactSymbolResponse) {
    return exactSymbolResponse;
  }

  if (!query) {
    if (!useDense)
      return {
        ...createEmptyBm25Response({
          classificationMetadata,
          family: classifiedFamily,
          limit,
          rawQuery,
        }),
        ...(expansionMetadata ? { expansion: expansionMetadata } : {}),
      };

    const readinessReport = await getDenseReadiness(options);
    const denseStrategy = await resolveDenseStrategyForSearch({
      databasePath: options.databasePath,
      readinessReport,
      client: options.client,
    });

    if (readinessReport.state !== 'warm') {
      return {
        ...(await createDegradedBm25Response({
          alpha,
          classificationMetadata,
          compiledFilter,
          family: classifiedFamily,
          limit,
          query,
          readinessReport,
          client: options.client,
        })),
        ...(expansionMetadata ? { expansion: expansionMetadata } : {}),
        dense_strategy: denseStrategy,
      };
    }

    return {
      alpha,
      dense_state: 'warm',
      dense_strategy: denseStrategy,
      limit,
      diskann_used: false,
      rrf_used: false,
      ...(classifiedFamily ? { family: classifiedFamily } : {}),
      query_class: classificationMetadata.query_class,
      confidence: classificationMetadata.confidence,
      classification_fallback: classificationMetadata.classification_fallback,
      family_fallback: classificationMetadata.family_fallback,
      ...(expansionMetadata ? { expansion: expansionMetadata } : {}),
      query: rawQuery,
      results: [],
      use_dense: true,
      ...(useRerank
        ? {
            use_rerank: false,
            rerank_degraded: true,
            rerank_state: 'cold',
            rerank_reason: 'Reranker not available for empty query.',
          }
        : {}),
    };
  }

  if (useDense) {
    const readinessReport = await getDenseReadiness(options);
    const denseStrategy = await resolveDenseStrategyForSearch({
      databasePath: options.databasePath,
      readinessReport,
      client: options.client,
    });

    if (readinessReport.state !== 'warm') {
      const degradedResponse = await createDegradedBm25Response({
        alpha,
        classificationMetadata,
        compiledFilter,
        databasePath: options.databasePath,
        family: classifiedFamily,
        limit,
        query: expandedBm25Query ?? query,
        readinessReport,
        client: options.client,
      });
      const degradedBase = {
        ...degradedResponse,
        ...(expansionMetadata ? { expansion: expansionMetadata } : {}),
        dense_strategy: denseStrategy,
      };

      if (!useRerank) {
        return degradedBase;
      }

      const rerankerReadiness = await getRerankerReadiness(options);
      if (rerankerReadiness.state !== 'warm') {
        return {
          ...degradedBase,
          rerank_degraded: true,
          rerank_reason: normalizeRerankReason(
            rerankerReadiness.reason,
            rerankerReadiness.state,
          ),
          rerank_state: rerankerReadiness.state,
          use_rerank: false,
        };
      }

      const rerankerFn = options.rerankerFn ?? rerankCandidates;
      const candidatesForRerank = degradedResponse.results.slice(
        0,
        rerankCandidatesCount,
      );
      const rerankedResults = await rerankerFn(rawQuery, candidatesForRerank, {
        rerankerModelDirectory: options.rerankerModelDirectory,
        rerankerModelId: options.rerankerModelId,
        rerankCandidatesCount,
      });

      return {
        ...degradedBase,
        rerank_candidates_count: rerankCandidatesCount,
        results: rerankedResults.slice(0, limit),
        rerank_state: 'warm',
        use_rerank: true,
      };
    }

    const denseQueryFn = options.denseQuery ?? queryDenseIndex;
    const denseResult = await denseQueryFn({
      alpha,
      compiledFilter,
      corpusDatabasePath: options.databasePath,
      dense: true,
      family: classifiedFamily,
      limit,
      modelDirectory: options.modelDirectory,
      modelId: options.modelId,
      parallelRunner: runParallelQueries,
      query: rawQuery,
    });

    const denseResponse = {
      ...denseResult,
      results: denseResult.results,
      query_class: classificationMetadata.query_class,
      confidence: classificationMetadata.confidence,
      classification_fallback: classificationMetadata.classification_fallback,
      family_fallback: classificationMetadata.family_fallback,
      ...(expansionMetadata ? { expansion: expansionMetadata } : {}),
      dense_state: 'warm',
      dense_strategy: denseStrategy,
      diskann_used: true,
      rrf_used: true,
      use_dense: true,
    };

    // Cross-encoder re-ranking: when use_rerank is requested, check reranker
    // readiness and re-rank the top rerank_candidates_count candidates.
    if (!useRerank) {
      return { ...denseResponse, use_rerank: false };
    }

    const rerankerReadiness = await getRerankerReadiness(options);
    if (rerankerReadiness.state !== 'warm') {
      return {
        ...denseResponse,
        rerank_degraded: true,
        rerank_reason: normalizeRerankReason(
          rerankerReadiness.reason,
          rerankerReadiness.state,
        ),
        rerank_state: rerankerReadiness.state,
        use_rerank: false,
      };
    }

    const rerankerFn = options.rerankerFn ?? rerankCandidates;
    const candidatesForRerank = denseResult.results.slice(
      0,
      rerankCandidatesCount,
    );
    const rerankedResults = await rerankerFn(rawQuery, candidatesForRerank, {
      rerankerModelDirectory: options.rerankerModelDirectory,
      rerankerModelId: options.rerankerModelId,
      rerankCandidatesCount,
    });

    return {
      ...denseResponse,
      rerank_candidates_count: rerankCandidatesCount,
      results: rerankedResults.slice(0, limit),
      rerank_state: 'warm',
      use_rerank: true,
    };
  }

  return {
    ...(await runBm25Search({
      alpha,
      classificationMetadata,
      compiledFilter,
      databasePath: options.databasePath,
      family: classifiedFamily,
      limit,
      query: expandedBm25Query ?? query,
      client: options.client,
    })),
    ...(expansionMetadata ? { expansion: expansionMetadata } : {}),
  };
}

/**
 * Build a concise ranking explanation for a single corpus result.
 *
 * Exposes the per-stage scores that contributed to the final ranking so callers
 * can debug why a chunk was selected. Unavailable stages (for example dense
 * similarity when `use_dense` is false) are reported as zero with a short reason.
 *
 * @param {object} result - Raw corpus search result.
 * @param {boolean} useDense - Whether dense similarity was requested.
 * @param {boolean} useRerank - Whether cross-encoder reranking was requested.
 * @returns {{ bm25_score: number, dense_score: number, rerank_score: number, final_score: number, reason: string }} Ranking explanation descriptor.
 */
export function buildRankingExplanation(result, useDense, useRerank) {
  const bm25Score =
    typeof result.bm25_score === 'number'
      ? result.bm25_score
      : typeof result.score === 'number'
        ? result.score
        : 0;
  const denseScore = useDense
    ? typeof result.cosine_score === 'number'
      ? result.cosine_score
      : 0
    : 0;
  const rerankScore = useRerank
    ? typeof result.rerank_score === 'number'
      ? result.rerank_score
      : 0
    : 0;
  const finalScore = typeof result.score === 'number' ? result.score : 0;

  const stages = ['BM25 lexical match'];
  if (useDense) {
    stages.push('dense cosine similarity');
  }
  if (useRerank) {
    stages.push('cross-encoder rerank');
  }

  return {
    bm25_score: bm25Score,
    dense_score: denseScore,
    rerank_score: rerankScore,
    final_score: finalScore,
    reason: `Final score blends ${stages.join(', ')}.`,
  };
}

/**
 * Strip non-essential metadata from a single corpus result and truncate its text
 * snippet to {@link COMPACT_TEXT_THRESHOLD} characters.
 *
 * Compact mode keeps only the fields an LLM caller typically needs for
 * relevance decisions: identity, provenance, score, and a short text preview.
 * When a ranking explanation is present it is preserved so compact callers still
 * receive transparency into how each result was ranked.
 *
 * @param {object} result - Raw corpus search result.
 * @returns {object} Compact result descriptor.
 */
function compactSearchResult(result) {
  const text = typeof result.text === 'string' ? result.text : '';
  const truncated =
    text.length > COMPACT_TEXT_THRESHOLD
      ? `${text.slice(0, COMPACT_TEXT_THRESHOLD - 1)}…`
      : text;
  const compacted = {
    chunk_id: result.chunk_id,
    file_path: result.file_path,
    family: result.family,
    chunk_index: result.chunk_index,
    heading_path: result.heading_path ?? null,
    context_header: result.context_header ?? null,
    symbol_name: result.symbol_name ?? null,
    text: truncated,
    score: result.score,
    feedback_boost: result.feedback_boost ?? 0,
    feedback_signals: result.feedback_signals,
  };

  if (
    typeof result.ranking_explanation === 'object' &&
    result.ranking_explanation !== null
  ) {
    compacted.ranking_explanation = result.ranking_explanation;
  }

  return compacted;
}

/**
 * Estimate the token cost of a list of compact corpus results.
 *
 * Uses a conservative character-per-token ratio so callers can compare the
 * response size against their own context budget.
 *
 * @param {Array<object>} results - Corpus results (after optional compact filtering).
 * @returns {number} Estimated token count.
 */
function estimateResponseTokens(results) {
  const charCount = results.reduce(
    (sum, result) => sum + (result.text?.length ?? 0),
    0,
  );
  return Math.ceil(charCount / RESPONSE_CHARS_PER_TOKEN);
}

/**
 * Fire-and-forget impression recording for every chunk returned by a search.
 *
 * Uses a fresh writable SQLite connection so the read-only search database
 * connection can close immediately. Errors are caught and silently dropped
 * so feedback writes never fail the search response or add response latency.
 *
 * @param {object} response - Search response payload.
 * @param {string} response.query - Query string correlated with the impressions.
 * @param {Array<{chunk_id?: number}>} [response.results] - Search result chunks.
 * @param {string | undefined} databasePath - Optional corpus database path override.
 */
async function recordSearchImpressions(response, databasePath, client) {
  const results = response?.results;
  const query = response?.query;
  if (
    !Array.isArray(results) ||
    results.length === 0 ||
    typeof query !== 'string'
  ) {
    return;
  }

  const resolvedClient = client ?? (await getTursoClient(databasePath));
  try {
    const createdAt = new Date().toISOString();
    const queryHash = createHash('sha256').update(query).digest('hex');
    const statements = [];
    for (const result of results) {
      const chunkId = result?.chunk_id;
      if (typeof chunkId === 'number') {
        statements.push({
          sql: `INSERT INTO feedback_events
            (event_id, chunk_id, signal_type, signal_strength, query_hash, agent_id, context, created_at)
          VALUES
            (?, ?, 'impression', 0.1, ?, NULL, NULL, ?)`,
          args: [randomUUID(), chunkId, queryHash, createdAt],
        });
      }
    }
    if (statements.length > 0) {
      await resolvedClient.batch(statements, 'write');
    }
  } catch {
    // Best-effort: silently drop feedback write failures.
  }
}

/**
 * Default feedback signal counters used when no score row exists for a chunk.
 */
const DEFAULT_FEEDBACK_SIGNALS = {
  total_positive: 0,
  total_negative: 0,
  total_impressions: 0,
  total_clicks: 0,
  total_references: 0,
};

/**
 * Attach feedback boost and signal counters to each search result.
 *
 * Queries the `feedback_scores` table read-only and mutates each result object
 * in place. When no score row exists, `feedback_boost` is set to `0` and the
 * signal counters are zeroed. Errors are swallowed so feedback enrichment never
 * fails a search response.
 *
 * @param {Array<object>} results - Search result list.
 * @param {string | undefined} databasePath - Optional corpus database path override.
 */
export async function attachFeedbackToResults(results, databasePath, client) {
  if (!Array.isArray(results) || results.length === 0) {
    return;
  }

  const chunkIds = results
    .map((result) => result?.chunk_id)
    .filter((chunkId) => typeof chunkId === 'number');

  if (chunkIds.length === 0) {
    for (const result of results) {
      result.feedback_boost = 0;
      result.feedback_signals = { ...DEFAULT_FEEDBACK_SIGNALS };
    }
    return;
  }

  /** @type {Map<number, object>} */
  const scoresByChunkId = new Map();

  const resolvedClient = client ?? (await getTursoClient(databasePath));
  try {
    const placeholders = chunkIds.map(() => '?').join(',');
    const result = await resolvedClient.execute({
      sql: `SELECT chunk_id, feedback_boost, total_positive, total_negative, total_impressions, total_clicks, total_references FROM feedback_scores WHERE chunk_id IN (${placeholders})`,
      args: chunkIds,
    });
    for (const row of result.rows) {
      scoresByChunkId.set(Number(row.chunk_id), row);
    }
  } catch {
    // Best-effort: silently skip feedback enrichment when the table is missing.
  }

  for (const result of results) {
    const score = scoresByChunkId.get(result.chunk_id);
    if (score) {
      result.feedback_boost = Number(score.feedback_boost);
      result.feedback_signals = {
        total_positive: Number(score.total_positive),
        total_negative: Number(score.total_negative),
        total_impressions: Number(score.total_impressions),
        total_clicks: Number(score.total_clicks),
        total_references: Number(score.total_references),
      };
    } else {
      result.feedback_boost = 0;
      result.feedback_signals = { ...DEFAULT_FEEDBACK_SIGNALS };
    }
  }
}

/**
 * Build a lightweight freshness stanza for search responses.
 *
 * Reads the corpus documents table to report the last known index update and
 * whether the corpus appears empty. This avoids filesystem access so it adds
 * minimal latency to search responses while still providing the transparency
 * contract (timestamp, stale flag, last_update_source).
 *
 * @param {string | undefined} databasePath - Optional corpus database path override.
 * @returns {{ timestamp: number, stale: boolean, last_update_source: string, last_indexed_at?: number, freshness_proof?: object }} Freshness stanza.
 */
export async function buildResponseFreshness(databasePath, client) {
  const timestamp = Date.now();

  const resolvedClient = client ?? (await getTursoClient(databasePath));
  try {
    const result = await resolvedClient.execute(
      'SELECT COUNT(*) AS count, MAX(indexed_at) AS last_indexed_at FROM documents',
    );
    const row = result.rows[0];
    const count = Number(row?.count ?? 0);
    const lastIndexedAt = row?.last_indexed_at
      ? Number(row.last_indexed_at)
      : null;
    const freshness = {
      timestamp,
      stale: count === 0,
      last_update_source: count === 0 ? 'empty_corpus_index' : 'corpus_index',
    };
    if (lastIndexedAt) {
      freshness.last_indexed_at = lastIndexedAt;
      const proofResult = await resolvedClient.execute(
        'SELECT mtime_ms, file_size, sha256 FROM documents ORDER BY indexed_at DESC LIMIT 1',
      );
      const proofRow = proofResult.rows[0];
      if (proofRow) {
        freshness.freshness_proof = {
          mtime_ms:
            proofRow.mtime_ms === null ? null : Number(proofRow.mtime_ms),
          size: proofRow.file_size === null ? null : Number(proofRow.file_size),
          sha256: proofRow.sha256 ?? null,
        };
      }
    }
    return freshness;
  } catch {
    return {
      timestamp,
      stale: true,
      last_update_source: 'unknown',
    };
  }
}

/**
 * Search the indexed corpus using BM25 or hybrid dense reranking.
 *
 * This is the public entry point. It delegates to the internal search
 * implementation, attaches per-result feedback boost and signal counters,
 * and then schedules best-effort impression feedback recording for every
 * returned chunk without blocking the response.
 *
 * Selects the search strategy based on dense-index readiness and the
 * `use_dense` option. Sanitizes the raw query with {@link sanitizeFtsQuery}
 * before executing any SQL to prevent FTS5 operator injection.
 *
 * @param {object} [options={}] - Search options.
 * @param {string} options.query - Free-text query string (required for BM25; optional for dense).
 * @param {number} [options.limit=10] - Maximum result count, clamped to [1, 50].
 * @param {string} [options.family] - Optional document family filter.
 * @param {boolean} [options.use_dense=true] - Enable hybrid dense reranking when the index is warm.
 * @param {number} [options.alpha] - Deprecated legacy blend weight; retained for API compatibility but RRF ranking ignores it.
 * @param {string} [options.query_class] - Override query classification for routing. One of: simple_lookup, cross_boundary, multi_hop, exploratory, code_specific, plan_specific.
 * @param {object} [options.classification_hints] - Optional overrides for classification-derived alpha and family.
 * @param {number} [options.classification_hints.alpha] - Override alpha from classification routing.
 * @param {string} [options.classification_hints.family] - Override family filter from classification routing.
 * @param {object} [options.metadata] - Optional metadata filter with a `filter` predicate tree.
 * @param {object} [options.metadata.filter] - Structured filter predicate tree (14 ops: eq, neq, in, not_in, gt, gte, lt, lte, like, is_null, is_not_null, and, or, not).
 * @param {string} [options.slice_id] - Optional slice identifier filter; only chunks with `slice_id = ?` are returned.
 * @param {number} [options.step_number] - Optional step number filter; only chunks with `step_number = ?` are returned.
 * @param {string} [options.databasePath] - Override corpus SQLite database path.
 * @param {string} [options.modelDirectory] - Override ONNX model directory path.
 * @param {string} [options.modelId] - Override ONNX model identifier.
 * @param {Function} [options.denseQuery] - Override dense-query implementation (for testing).
 * @param {Function} [options.readinessProbe] - Override readiness probe (for testing).
 * @param {boolean} [options.use_rerank=false] - Enable cross-encoder re-ranking on hybrid search results.
 * @param {number} [options.rerank_candidates_count=50] - Number of hybrid candidates to re-rank (default: 50).
 * @param {string} [options.rerankerModelDirectory] - Override reranker model directory (for testing).
 * @param {string} [options.rerankerModelId] - Override reranker model identifier (for testing).
 * @param {Function} [options.rerankerReadinessProbe] - Override reranker readiness probe (for testing).
 * @param {Function} [options.rerankerFn] - Override rerankCandidates implementation (for testing).
 * @param {boolean | string} [options.expand_query=false] - Enable query expansion: `true` for full expansion, `'domain-only'` for domain associations only, `false` (default) for no expansion.
 * @param {string} [options.associationsPath] - Override domain associations file path (for testing).
 * @param {Function} [options.expandQueryFn] - Override expandQuery implementation (for testing).
 * @returns {Promise<object>} Search result payload including `query`, `limit`, `use_dense`, `results`, `latency_ms`, `response_tokens`, `freshness`, `dense_state`, `rerank_state`, `diskann_used`, `rrf_used`, and, when dense retrieval is degraded, `dense_degraded`, `dense_reason`, and `self_heal`. Each result carries `feedback_boost` and `feedback_signals`.
 */
export async function searchCorpus(options = {}) {
  const startTime = performance.now();
  try {
    let response = await searchCorpusImpl(options);

    // If the classifier-derived family filter produced zero results and the
    // caller did not request a specific family, broaden the search to the
    // full corpus once. This prevents rare plan/code misclassifications from
    // returning empty result sets while still respecting explicit family
    // filters when the user supplies them.
    const familyWasDerived =
      !('family' in options) &&
      Array.isArray(response.results) &&
      response.results.length === 0 &&
      response.family != null;
    if (familyWasDerived) {
      const broadResponse = await searchCorpusImpl({
        ...options,
        family: null,
        skip_family_classification: true,
      });
      if (
        Array.isArray(broadResponse.results) &&
        broadResponse.results.length > 0
      ) {
        response = broadResponse;
      }
    }

    // BM25 search results already include feedback_boost via the SQL LEFT
    // JOIN in runBm25Search. Only call attachFeedbackToResults for search
    // paths that don't (dense, exact-symbol).
    if (
      Array.isArray(response.results) &&
      response.results.length > 0 &&
      typeof response.results[0]?.feedback_boost !== 'number'
    ) {
      await attachFeedbackToResults(
        response.results,
        options?.databasePath,
        options?.client,
      );
    }
    await recordSearchImpressions(
      response,
      options?.databasePath,
      options?.client,
    );

    if (options.compact === true) {
      response.results = response.results.map(compactSearchResult);
    }
    response.compact = options.compact === true;
    response.response_tokens = estimateResponseTokens(response.results);

    response.latency_ms = performance.now() - startTime;
    response.freshness = await buildResponseFreshness(
      options?.databasePath,
      options?.client,
    );
    return response;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    if (
      message.includes('Semantic index not found') ||
      message.includes('unable to open database') ||
      message.includes('SQLITE_CANTOPEN') ||
      message.includes('no such table')
    ) {
      throw cortexError(
        ErrorCodes.CORPUS_NOT_FOUND,
        `Corpus database not found: ${options?.databasePath ?? 'default path'}.`,
      );
    }
    throw error;
  }
}

/**
 * Return the dense-index readiness report, using a process-lifetime cache when possible.
 *
 * The cache is bypassed when `DENSE_FORCE_STATE` is `cold` or `model-only` in the
 * environment so integration tests and manual probes can override the state without
 * restarting the process. A successful warm result is stored in `cachedDenseReadiness`
 * to avoid repeated filesystem or model probes within the same process lifetime.
 *
 * @param {object} options - Options forwarded from {@link searchCorpus}.
 * @param {string} [options.databasePath] - Corpus database path.
 * @param {string} [options.modelDirectory] - ONNX model directory.
 * @param {string} [options.modelId] - ONNX model identifier.
 * @param {Function} [options.readinessProbe] - Override probe implementation.
 * @returns {Promise<object>} Dense readiness report with at least a `state` field.
 */
async function getDenseReadiness(options) {
  const readinessProbe = options.readinessProbe ?? checkDenseReadiness;
  const readinessOptions = {
    corpusDatabasePath: options.databasePath,
    modelDirectory: options.modelDirectory,
    modelId: options.modelId,
  };

  if (shouldBypassDenseReadinessCache()) {
    return readinessProbe(readinessOptions);
  }

  if (cachedDenseReadiness !== null) {
    return cachedDenseReadiness;
  }

  const readinessReport = await readinessProbe(readinessOptions);
  cachedDenseReadiness =
    readinessReport.state === 'warm' ? readinessReport : null;
  return readinessReport;
}

/**
 * Whether to skip the dense-readiness cache for the current invocation.
 *
 * Returns `true` when `DENSE_FORCE_STATE` is set to `cold` or `model-only`,
 * allowing integration tests to force a live probe without restarting the server.
 *
 * @returns {boolean} `true` if the cache should be bypassed.
 */
function shouldBypassDenseReadinessCache() {
  const forcedState =
    typeof process.env.DENSE_FORCE_STATE === 'string'
      ? process.env.DENSE_FORCE_STATE.trim()
      : '';
  return forcedState === 'cold' || forcedState === 'model-only';
}

/**
 * Build an empty BM25-only response for a blank query.
 *
 * @param {{ classificationMetadata?: object | null, family: string | null, limit: number, rawQuery: string }} params - Response parameters.
 * @returns {object} Empty search response with `use_dense: false` and an empty `results` array.
 */
function createEmptyBm25Response({
  classificationMetadata,
  family,
  limit,
  rawQuery,
}) {
  return {
    limit,
    ...(family ? { family } : {}),
    ...(classificationMetadata
      ? {
          query_class: classificationMetadata.query_class,
          confidence: classificationMetadata.confidence,
          classification_fallback:
            classificationMetadata.classification_fallback,
          family_fallback: classificationMetadata.family_fallback ?? false,
        }
      : {}),
    query: rawQuery,
    results: [],
    use_dense: false,
    diskann_used: false,
    rrf_used: false,
  };
}

/**
 * Build a BM25 response annotated with dense-degradation metadata.
 *
 * Called when dense search is requested but the embeddings index is cold or
 * model-only. Adds `dense_degraded: true`, `dense_reason`, `dense_state`, and
 * a structured `self_heal` block (from {@link evaluateSelfHeal}) to the BM25
 * result so callers can surface the degradation to the user and know whether
 * a background repair has been started.
 *
 * @param {{ alpha?: number, classificationMetadata?: object | null, client?: import('@libsql/client').Client, compiledFilter?: { sql: string, params: Array<string | number | null> } | null, databasePath?: string, family: string | null, limit: number, query: string, readinessReport: object }} params - Response parameters.
 * @returns {object} BM25 results with `dense_degraded: true`, degradation details, and `self_heal` guidance.
 */
async function createDegradedBm25Response({
  alpha,
  classificationMetadata,
  compiledFilter,
  databasePath,
  family,
  limit,
  query,
  readinessReport,
  client,
}) {
  const bm25Response = query
    ? await runBm25Search({
        alpha,
        classificationMetadata,
        compiledFilter,
        databasePath,
        family,
        limit,
        query,
        client,
      })
    : createEmptyBm25Response({
        classificationMetadata,
        family,
        limit,
        rawQuery: query,
      });

  const response = {
    ...bm25Response,
    dense_degraded: true,
    dense_reason: normalizeDenseReason(
      readinessReport.reason,
      readinessReport.state,
    ),
    dense_state: readinessReport.state,
    use_dense: false,
  };

  try {
    const decision = await evaluateSelfHeal({
      probe: () => Promise.resolve(readinessReport),
    });
    if (decision && typeof decision === 'object' && decision.guidanceFields) {
      response.self_heal = decision.guidanceFields;
      if (decision.action === 'started') {
        invalidateDenseReadinessCache();
      }
    }
  } catch {
    // The guard must never break search: if evaluating self-heal fails,
    // return the plain BM25 response so the caller still gets results.
  }

  return response;
}

/**
 * Produce a human-readable dense degradation reason.
 *
 * Falls back to a canonical message when the readiness probe did not supply one,
 * distinguishing between a cold (no model assets) and a model-only (model present
 * but embeddings database absent or incomplete) state.
 *
 * @param {string | undefined} reason - Reason string from the readiness probe.
 * @param {string} state - Dense readiness state (`cold` or `model-only`).
 * @returns {string} Non-empty reason string.
 */
function normalizeDenseReason(reason, state) {
  const trimmedReason = typeof reason === 'string' ? reason.trim() : '';
  if (trimmedReason) return trimmedReason;
  return state === 'cold'
    ? 'Dense embeddings are unavailable because the model assets are absent.'
    : 'Dense embeddings are unavailable because the embeddings database is missing or incomplete.';
}

/**
 * Clamp and coerce the BM25/dense blend weight to a valid finite float.
 *
 * @param {unknown} alpha - Raw alpha value from the caller.
 * @returns {number} Normalized alpha, defaulting to 0.5 when absent or non-finite.
 */
function normalizeAlpha(alpha) {
  const numericAlpha = Number(alpha ?? 0.5);
  return Number.isFinite(numericAlpha) ? numericAlpha : 0.5;
}

/**
 * Return the reranker readiness report, using a process-lifetime cache when possible.
 *
 * Mirrors the dense-readiness caching pattern. The cache is bypassed when
 * `RERANKER_FORCE_STATE` is `cold` or `model-only` so integration tests can
 * override the state. A successful warm result is cached in
 * `cachedRerankerReadiness` for the rest of the process lifetime.
 *
 * @param {object} options - Options forwarded from {@link searchCorpus}.
 * @param {string} [options.rerankerModelDirectory] - Override reranker model directory.
 * @param {string} [options.rerankerModelId] - Override reranker model identifier.
 * @param {Function} [options.rerankerReadinessProbe] - Override probe implementation.
 * @returns {Promise<object>} Reranker readiness report with at least a `state` field.
 */
async function getRerankerReadiness(options) {
  const readinessProbe =
    options.rerankerReadinessProbe ?? checkRerankerReadiness;
  const readinessOptions = {
    modelDirectory: options.rerankerModelDirectory,
    modelId: options.rerankerModelId,
  };

  if (shouldBypassRerankerReadinessCache()) {
    return readinessProbe(readinessOptions);
  }

  if (cachedRerankerReadiness !== null) {
    return cachedRerankerReadiness;
  }

  const readinessReport = await readinessProbe(readinessOptions);
  cachedRerankerReadiness =
    readinessReport.state === 'warm' ? readinessReport : null;
  return readinessReport;
}

/**
 * Whether to skip the reranker-readiness cache for the current invocation.
 *
 * Returns `true` when `RERANKER_FORCE_STATE` is set to `cold` or `model-only`,
 * allowing integration tests to force a live probe without restarting the server.
 *
 * @returns {boolean} `true` if the cache should be bypassed.
 */
function shouldBypassRerankerReadinessCache() {
  const forcedState =
    typeof process.env.RERANKER_FORCE_STATE === 'string'
      ? process.env.RERANKER_FORCE_STATE.trim()
      : '';
  return forcedState === 'cold' || forcedState === 'model-only';
}

/**
 * Produce a human-readable reranker degradation reason.
 *
 * Falls back to a canonical message when the readiness probe did not supply one,
 * distinguishing between a cold (no model assets) and a model-only (model present
 * but session creation failed) state.
 *
 * @param {string | undefined} reason - Reason string from the readiness probe.
 * @param {string} state - Reranker readiness state (`cold` or `model-only`).
 * @returns {string} Non-empty reason string.
 */
function normalizeRerankReason(reason, state) {
  const trimmedReason = typeof reason === 'string' ? reason.trim() : '';
  if (trimmedReason) return trimmedReason;
  return state === 'cold'
    ? 'Cross-encoder reranker is unavailable because the model assets are absent.'
    : 'Cross-encoder reranker is unavailable because the ONNX session could not be created.';
}

/**
 * Try an exact `symbol_name` lookup for the raw query.
 *
 * Returns a response object when the corpus contains one or more chunks whose
 * `symbol_name` equals the raw query, otherwise `null` so the normal BM25 or
 * dense search path can proceed.
 *
 * @param {{ classificationMetadata?: object | null, compiledFilter?: { sql: string, params: Array<unknown> } | null, databasePath?: string, family: string | null, limit: number, rawQuery: string }} params - Lookup parameters.
 * @returns {object | null} Exact-symbol response, or `null` when no symbol matches.
 */
async function tryExactSymbolLookup({
  classificationMetadata,
  compiledFilter,
  databasePath,
  family,
  limit,
  rawQuery,
  client,
}) {
  const rows = await runExactSymbolLookup({
    compiledFilter,
    databasePath,
    family,
    limit,
    rawQuery,
    client,
  });
  if (!Array.isArray(rows) || rows.length === 0) {
    return null;
  }

  return {
    exact_symbol_match: true,
    limit,
    query: rawQuery,
    results: rows,
    use_dense: false,
    diskann_used: false,
    rrf_used: false,
    ...(family ? { family } : {}),
    ...(classificationMetadata
      ? {
          query_class: classificationMetadata.query_class,
          confidence: classificationMetadata.confidence,
          classification_fallback:
            classificationMetadata.classification_fallback,
          family_fallback: classificationMetadata.family_fallback ?? false,
        }
      : {}),
  };
}

/**
 * Query the corpus for chunks whose `symbol_name` exactly matches the query.
 *
 * Uses the dedicated `chunks_symbol_idx` index on `symbol_name`. When the
 * corpus database is missing or the lookup fails, returns an empty array so
 * the caller can fall back to the normal search path.
 *
 * @param {{ compiledFilter?: { sql: string, params: Array<unknown> } | null, databasePath?: string, family: string | null, limit: number, rawQuery: string }} params - Lookup parameters.
 * @returns {Array<object>} Matching chunk rows, or an empty array when no symbol matches.
 */
async function runExactSymbolLookup({
  compiledFilter,
  databasePath,
  family,
  limit,
  rawQuery,
  client,
}) {
  const resolvedClient = client ?? (await getTursoClient(databasePath));
  try {
    const familyFilter = family ? 'AND d.doc_family = ?' : '';
    const filterSql = compiledFilter ? `AND ${compiledFilter.sql}` : '';
    const filterParams = compiledFilter ? compiledFilter.params : [];
    const sql = `
      SELECT d.file_path, d.doc_family, c.chunk_id, c.chunk_index, c.heading_path,
        c.body_text, c.char_start, c.char_end,
        c.parent_chunk_id, c.depth, c.context_header,
        c.symbol_name, c.signature_text, c.jsdoc_text, c.export_type, c.module_path,
        c.arch_layer, c.jsdoc_quality, c.jsdoc_word_count,
        c.cyclomatic_complexity, c.test_coverage, c.source_path_pattern
      FROM chunks c
      JOIN documents d ON d.doc_id = c.doc_id
      WHERE c.symbol_name = ? ${familyFilter} ${filterSql}
      ORDER BY c.chunk_id
      LIMIT ?
    `;
    const args = family
      ? [rawQuery, family, ...filterParams, limit]
      : [rawQuery, ...filterParams, limit];

    const result = await resolvedClient.execute({ sql, args });
    return result.rows.map((row) => ({
      ...readChunkRow(row),
      body_text: row.body_text,
    }));
  } catch {
    return [];
  }
}

/**
 * Execute a BM25 full-text search against the corpus SQLite database.
 *
 * Queries the FTS5 virtual table (`chunks_fts`) joined with `chunks` and
 * `documents`. Applies an optional family filter via a safe parameterized
 * clause, and an optional compiled metadata filter via AND, then returns
 * the top `limit` rows ordered by descending BM25 score.
 *
 * When a compiled metadata filter is provided, its `?` placeholders are
 * bound as positional parameters alongside the FTS query and family
 * parameters.
 *
 * @param {{ classificationMetadata?: object | null, compiledFilter?: { sql: string, params: Array<string | number | null> } | null, databasePath?: string, family: string | null, limit: number, query: string }} params - Query parameters.
 * @returns {object} BM25 results payload with `query`, `limit`, `use_dense: false`, and `results`.
 */
async function runBm25Search({
  alpha,
  classificationMetadata,
  compiledFilter,
  databasePath,
  family,
  limit,
  query,
  client,
}) {
  const resolvedClient = client ?? (await getTursoClient(databasePath));

  const familyFilter = family ? 'AND d.doc_family = ?' : '';

  let metadataFilterSql = '';
  const metadataArgs = [];
  if (compiledFilter) {
    metadataFilterSql = `AND ${compiledFilter.sql}`;
    metadataArgs.push(...compiledFilter.params);
  }

  // POWER(0.95, days) time-decay is applied in buildChunkAggregateAsync when
  // computing and storing feedback_boost.  The search query retrieves the
  // already-decayed stored value directly from feedback_scores.
  const sqlWithFeedback = `
    SELECT d.file_path, d.doc_family, c.chunk_id, c.chunk_index, c.heading_path,
      c.body_text, c.char_start, c.char_end,
      c.parent_chunk_id, c.depth, c.context_header,
      c.symbol_name, c.signature_text, c.jsdoc_text, c.export_type, c.module_path,
      c.arch_layer, c.jsdoc_quality, c.jsdoc_word_count,
      c.cyclomatic_complexity, c.test_coverage, c.source_path_pattern,
      bm25(chunks_fts) AS score,
      COALESCE(fs.feedback_boost, 0) AS feedback_boost,
      COALESCE(fs.total_positive, 0) AS fb_total_positive,
      COALESCE(fs.total_negative, 0) AS fb_total_negative,
      COALESCE(fs.total_impressions, 0) AS fb_total_impressions,
      COALESCE(fs.total_clicks, 0) AS fb_total_clicks,
      COALESCE(fs.total_references, 0) AS fb_total_references
    FROM chunks_fts
    JOIN chunks c ON c.chunk_id = chunks_fts.rowid
    JOIN documents d ON d.doc_id = c.doc_id
    LEFT JOIN feedback_scores fs ON fs.chunk_id = c.chunk_id
    WHERE chunks_fts MATCH ? ${familyFilter} ${metadataFilterSql}
    ORDER BY score
    LIMIT ?
  `;

  const sqlNoFeedback = `
    SELECT d.file_path, d.doc_family, c.chunk_id, c.chunk_index, c.heading_path,
      c.body_text, c.char_start, c.char_end,
      c.parent_chunk_id, c.depth, c.context_header,
      c.symbol_name, c.signature_text, c.jsdoc_text, c.export_type, c.module_path,
      c.arch_layer, c.jsdoc_quality, c.jsdoc_word_count,
      c.cyclomatic_complexity, c.test_coverage, c.source_path_pattern,
      bm25(chunks_fts) AS score,
      0 AS feedback_boost,
      0 AS fb_total_positive,
      0 AS fb_total_negative,
      0 AS fb_total_impressions,
      0 AS fb_total_clicks,
      0 AS fb_total_references
    FROM chunks_fts
    JOIN chunks c ON c.chunk_id = chunks_fts.rowid
    JOIN documents d ON d.doc_id = c.doc_id
    WHERE chunks_fts MATCH ? ${familyFilter} ${metadataFilterSql}
    ORDER BY score
    LIMIT ?
  `;

  const args = [query];
  if (family) args.push(family);
  args.push(...metadataArgs);
  args.push(limit);

  let result;
  try {
    result = await resolvedClient.execute({ sql: sqlWithFeedback, args });
  } catch (feedbackErr) {
    // feedback_scores table may be missing — fall back to no-join query
    result = await resolvedClient.execute({ sql: sqlNoFeedback, args });
  }

  return {
    query,
    limit,
    alpha,
    ...(family ? { family } : {}),
    ...(classificationMetadata
      ? {
          query_class: classificationMetadata.query_class,
          confidence: classificationMetadata.confidence,
          classification_fallback:
            classificationMetadata.classification_fallback,
          family_fallback: classificationMetadata.family_fallback ?? false,
        }
      : {}),
    use_dense: false,
    diskann_used: false,
    rrf_used: false,
    results: result.rows.map((row) => ({
      ...readChunkRow(row),
      body_text: row.body_text,
      score: Number(row.score),
      feedback_boost: Number(row.feedback_boost ?? 0),
      feedback_signals: {
        total_positive: Number(row.fb_total_positive ?? 0),
        total_negative: Number(row.fb_total_negative ?? 0),
        total_impressions: Number(row.fb_total_impressions ?? 0),
        total_clicks: Number(row.fb_total_clicks ?? 0),
        total_references: Number(row.fb_total_references ?? 0),
      },
    })),
  };
}

export {
  compactSearchResult,
  createDegradedBm25Response,
  createEmptyBm25Response,
  estimateResponseTokens,
  getDenseReadiness,
  getRerankerReadiness,
  normalizeAlpha,
  normalizeDenseReason,
  normalizeRerankReason,
  recordSearchImpressions,
  resolveChunkCountForStrategy,
  resolveDenseStrategyForSearch,
  runBm25Search,
  runExactSymbolLookup,
  searchCorpusImpl,
  shouldBypassDenseReadinessCache,
  shouldBypassRerankerReadinessCache,
  tryExactSymbolLookup,
};
