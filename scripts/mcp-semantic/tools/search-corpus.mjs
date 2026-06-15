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
 *   J -- yes --> L[queryDenseIndex<br/>Hybrid dense results]
 * ```
 */
import Database from 'better-sqlite3';
import { requireString } from '../../agent-customization/mcp/mcp-utils.mjs';
import { queryDenseIndex } from '../../semantic-index/query-dense.mjs';
import { checkDenseReadiness } from '../../semantic-index/dense-readiness.mjs';
import { checkRerankerReadiness } from '../../semantic-index/reranker-readiness.mjs';
import {
  rerankCandidates,
  normalizeRerankCandidates,
} from '../../semantic-index/rerank-index.mjs';
import {
  normalizeLimit,
  openCortexDatabase,
  readChunkRow,
  resolveDatabasePath,
  sanitizeFtsQuery,
} from './cortex-db.mjs';
import { classifyForSearchCorpus } from '../../semantic-index/classify-query.mjs';
import { classifyAndRoute } from '../../semantic-index/routing-table.mjs';
import {
  validateFilter,
  compileFilterToSqlAliased,
  applyPostRetrievalFilter,
} from '../../semantic-index/metadata-filter.mjs';
import { expandQuery } from '../../semantic-index/expand-query.mjs';
import { recordFeedbackEvent } from './feedback-core.mjs';

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
 * Search the indexed corpus using BM25 or hybrid dense reranking.
 *
 * Selects the search strategy based on dense-index readiness and the
 * `use_dense` option. Sanitizes the raw query with {@link sanitizeFtsQuery}
 * before executing any SQL to prevent FTS5 operator injection.
 *
 * When `metadata.filter` is provided, the filter is compiled to SQL for
 * BM25-only searches (applied as WHERE conditions on the chunks/documents
 * tables) and applied as a post-retrieval filter for dense searches. When
 * both `family` and `metadata.filter` are provided, they are combined with AND.
 *
 * @param {object} [options={}] - Search options.
 * @param {string} options.query - Free-text query string (required for BM25; optional for dense).
 * @param {number} [options.limit=10] - Maximum result count, clamped to [1, 50].
 * @param {string} [options.family] - Optional document family filter.
 * @param {boolean} [options.use_dense=true] - Enable hybrid dense reranking when the index is warm.
 * @param {number} [options.alpha] - BM25/dense blend weight (0 = BM25 only, 1 = dense only); default 0.5.
 * @param {string} [options.query_class] - Override query classification for routing. One of: simple_lookup, cross_boundary, multi_hop, exploratory, code_specific, plan_specific.
 * @param {object} [options.classification_hints] - Optional overrides for classification-derived alpha and family.
 * @param {number} [options.classification_hints.alpha] - Override alpha from classification routing.
 * @param {string} [options.classification_hints.family] - Override family filter from classification routing.
 * @param {object} [options.metadata] - Optional metadata filter with a `filter` predicate tree.
 * @param {object} [options.metadata.filter] - Structured filter predicate tree (14 ops: eq, neq, in, not_in, gt, gte, lt, lte, like, is_null, is_not_null, and, or, not).
 * @param {string} [options.databasePath] - Override corpus SQLite database path.
 * @param {string} [options.embeddingsDatabasePath] - Override embeddings SQLite database path.
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
 * @returns {Promise<object>} Search result payload including `query`, `limit`, `use_dense`, and `results`.
 */
async function searchCorpusImpl(options = {}) {
  const rawQuery = requireString(options.query, 'query');
  const query = sanitizeFtsQuery(rawQuery);
  const limit = normalizeLimit(options.limit, 10);
  const explicitFamily =
    typeof options.family === 'string' && options.family.trim()
      ? options.family.trim()
      : null;
  const useDense = options.use_dense !== false;

  // Validate and compile metadata filter if provided
  const metadataFilter = options.metadata?.filter;
  let compiledFilter = null;
  if (metadataFilter) {
    validateFilter(metadataFilter);
    compiledFilter = compileFilterToSqlAliased(metadataFilter);
  }

  // Classification-aware alpha and family selection
  const explicitAlpha = options.alpha;
  const queryClass = options.query_class;
  const classificationHints = options.classification_hints;

  let classificationMetadata = null;

  // Determine alpha and family from classification
  let effectiveAlpha = explicitAlpha;
  let effectiveFamily = explicitFamily;

  if (queryClass) {
    // Explicit class from caller — use full routing
    const routing = classifyAndRoute(rawQuery, classificationHints);
    effectiveAlpha = explicitAlpha ?? routing.alpha;
    effectiveFamily = explicitFamily ?? routing.strategy.family;
    classificationMetadata = {
      query_class: routing.query_class,
      confidence: routing.confidence,
      classification_fallback: false,
    };
  } else if (explicitAlpha === undefined) {
    // No explicit alpha and no explicit class → lightweight classification
    const classification = classifyForSearchCorpus(rawQuery);
    effectiveAlpha = classification.alpha;
    effectiveFamily = explicitFamily ?? classification.family;
    classificationMetadata = {
      query_class: classification.query_class,
      confidence: classification.confidence,
      classification_fallback: classification.classification_fallback,
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
        embeddingsDatabasePath: options.embeddingsDatabasePath,
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
    if (readinessReport.state !== 'warm') {
      return {
        ...createDegradedBm25Response({
          classificationMetadata,
          compiledFilter,
          family: classifiedFamily,
          limit,
          query: rawQuery,
          readinessReport,
        }),
        ...(expansionMetadata ? { expansion: expansionMetadata } : {}),
      };
    }

    return {
      alpha,
      dense_state: 'warm',
      limit,
      ...(classifiedFamily ? { family: classifiedFamily } : {}),
      ...(classificationMetadata
        ? {
            query_class: classificationMetadata.query_class,
            confidence: classificationMetadata.confidence,
            classification_fallback:
              classificationMetadata.classification_fallback,
          }
        : {}),
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
    if (readinessReport.state !== 'warm') {
      return {
        ...createDegradedBm25Response({
          classificationMetadata,
          compiledFilter,
          databasePath: options.databasePath,
          family: classifiedFamily,
          limit,
          query: expandedBm25Query ?? query,
          readinessReport,
        }),
        ...(expansionMetadata ? { expansion: expansionMetadata } : {}),
      };
    }

    const denseQuery = options.denseQuery ?? queryDenseIndex;
    const denseResult = await denseQuery({
      alpha,
      corpusDatabasePath: options.databasePath,
      dense: true,
      embeddingsDatabasePath: options.embeddingsDatabasePath,
      family: classifiedFamily,
      limit,
      modelDirectory: options.modelDirectory,
      modelId: options.modelId,
      query: rawQuery,
    });

    // Apply metadata filter as post-retrieval filter on dense candidates
    let filteredResults = denseResult.results;
    if (metadataFilter) {
      filteredResults = applyPostRetrievalFilter(
        denseResult.results,
        metadataFilter,
      );
    }

    const denseResponse = {
      ...denseResult,
      results: filteredResults,
      ...(classificationMetadata
        ? {
            query_class: classificationMetadata.query_class,
            confidence: classificationMetadata.confidence,
            classification_fallback:
              classificationMetadata.classification_fallback,
          }
        : {}),
      ...(expansionMetadata ? { expansion: expansionMetadata } : {}),
      dense_state: 'warm',
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
    const candidatesForRerank = filteredResults.slice(0, rerankCandidatesCount);
    const rerankedResults = await rerankerFn(rawQuery, candidatesForRerank, {
      rerankerModelDirectory: options.rerankerModelDirectory,
      rerankerModelId: options.rerankerModelId,
      rerankCandidatesCount,
    });

    return {
      ...denseResponse,
      rerank_candidates_count: rerankCandidatesCount,
      results: rerankedResults.slice(0, limit),
      use_rerank: true,
    };
  }

  return {
    ...runBm25Search({
      classificationMetadata,
      compiledFilter,
      databasePath: options.databasePath,
      family: classifiedFamily,
      limit,
      query: expandedBm25Query ?? query,
    }),
    ...(expansionMetadata ? { expansion: expansionMetadata } : {}),
  };
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
function recordSearchImpressions(response, databasePath) {
  const results = response?.results;
  const query = response?.query;
  if (!Array.isArray(results) || results.length === 0 || typeof query !== 'string') {
    return;
  }

  Promise.resolve().then(() => {
    try {
      const feedbackDb = new Database(resolveDatabasePath(databasePath));
      try {
        for (const result of results) {
          const chunkId = result?.chunk_id;
          if (typeof chunkId === 'number') {
            recordFeedbackEvent(feedbackDb, {
              chunk_id: chunkId,
              signal_type: 'impression',
              query,
            });
          }
        }
      } finally {
        feedbackDb.close();
      }
    } catch {
      // Best-effort: silently drop feedback write failures.
    }
  });
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
export function attachFeedbackToResults(results, databasePath) {
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
  try {
    const database = openCortexDatabase(databasePath);
    try {
      const placeholders = chunkIds.map(() => '?').join(',');
      const rows = database
        .prepare(
          `SELECT chunk_id, feedback_boost, total_positive, total_negative, total_impressions, total_clicks, total_references FROM feedback_scores WHERE chunk_id IN (${placeholders})`,
        )
        .all(...chunkIds);

      for (const row of rows) {
        scoresByChunkId.set(row.chunk_id, row);
      }
    } finally {
      database.close();
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
 * Search the indexed corpus using BM25 or hybrid dense reranking.
 *
 * This is the public entry point. It delegates to the internal search
 * implementation, attaches per-result feedback boost and signal counters,
 * and then schedules best-effort impression feedback recording for every
 * returned chunk without blocking the response.
 *
 * @param {object} [options={}] - Search options (same as {@link searchCorpusImpl}).
 * @returns {Promise<object>} Search result payload with `feedback_boost` and `feedback_signals` on each result.
 */
export async function searchCorpus(options = {}) {
  const response = await searchCorpusImpl(options);
  attachFeedbackToResults(response.results, options?.databasePath);
  recordSearchImpressions(response, options?.databasePath);
  return response;
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
 * @param {string} [options.embeddingsDatabasePath] - Embeddings database path.
 * @param {string} [options.modelDirectory] - ONNX model directory.
 * @param {string} [options.modelId] - ONNX model identifier.
 * @param {Function} [options.readinessProbe] - Override probe implementation.
 * @returns {Promise<object>} Dense readiness report with at least a `state` field.
 */
async function getDenseReadiness(options) {
  const readinessProbe = options.readinessProbe ?? checkDenseReadiness;
  const readinessOptions = {
    corpusDatabasePath: options.databasePath,
    embeddingsDatabasePath: options.embeddingsDatabasePath,
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
        }
      : {}),
    query: rawQuery,
    results: [],
    use_dense: false,
  };
}

/**
 * Build a BM25 response annotated with dense-degradation metadata.
 *
 * Called when dense search is requested but the embeddings index is cold or
 * model-only. Adds `dense_degraded: true`, `dense_reason`, and `dense_state`
 * to the BM25 result so callers can surface the degradation to the user and
 * know to run `npm run index:prewarm`.
 *
 * @param {{ classificationMetadata?: object | null, compiledFilter?: { sql: string, params: Array<string | number | null> } | null, databasePath?: string, family: string | null, limit: number, query: string, readinessReport: object }} params - Response parameters.
 * @returns {object} BM25 results with `dense_degraded: true` and degradation details.
 */
function createDegradedBm25Response({
  classificationMetadata,
  compiledFilter,
  databasePath,
  family,
  limit,
  query,
  readinessReport,
}) {
  const bm25Response = query
    ? runBm25Search({
        classificationMetadata,
        compiledFilter,
        databasePath,
        family,
        limit,
        query,
      })
    : createEmptyBm25Response({
        classificationMetadata,
        family,
        limit,
        rawQuery: query,
      });

  return {
    ...bm25Response,
    dense_degraded: true,
    dense_reason: normalizeDenseReason(
      readinessReport.reason,
      readinessReport.state,
    ),
    dense_state: readinessReport.state,
    use_dense: false,
  };
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
 * Execute a BM25 full-text search against the corpus SQLite database.
 *
 * Queries the FTS5 virtual table (`chunks_fts`) joined with `chunks` and
 * `documents`. Applies an optional family filter via a safe parameterized
 * clause, and an optional compiled metadata filter via AND, then returns
 * the top `limit` rows ordered by descending BM25 score.
 *
 * When a compiled metadata filter is provided, its `?` placeholders are
 * converted to named parameters (`@mf0`, `@mf1`, …) compatible with
 * `better-sqlite3` named-parameter binding, and bound alongside the FTS
 * query and family parameters.
 *
 * @param {{ classificationMetadata?: object | null, compiledFilter?: { sql: string, params: Array<string | number | null> } | null, databasePath?: string, family: string | null, limit: number, query: string }} params - Query parameters.
 * @returns {object} BM25 results payload with `query`, `limit`, `use_dense: false`, and `results`.
 */
function runBm25Search({
  classificationMetadata,
  compiledFilter,
  databasePath,
  family,
  limit,
  query,
}) {
  const database = openCortexDatabase(databasePath);

  try {
    const familyFilter = family ? 'AND d.doc_family = @family' : '';

    // Convert compiled filter ? placeholders to named @mfN parameters for better-sqlite3
    let metadataFilterSql = '';
    const namedParams = {};
    if (compiledFilter) {
      let namedSql = compiledFilter.sql;
      for (
        let paramIndex = 0;
        paramIndex < compiledFilter.params.length;
        paramIndex++
      ) {
        namedSql = namedSql.replace('?', `@mf${paramIndex}`);
        namedParams[`mf${paramIndex}`] = compiledFilter.params[paramIndex];
      }
      metadataFilterSql = `AND ${namedSql}`;
    }

    const rows = database
      .prepare(
        `
      SELECT d.file_path, d.doc_family, c.chunk_id, c.chunk_index, c.heading_path,
        c.body_text, c.char_start, c.char_end,
        c.parent_chunk_id, c.depth, c.context_header,
        c.symbol_name, c.signature_text, c.jsdoc_text, c.export_type, c.module_path,
        c.arch_layer, c.jsdoc_quality, c.jsdoc_word_count,
        c.cyclomatic_complexity, c.test_coverage, c.source_path_pattern,
        bm25(chunks_fts) AS score
      FROM chunks_fts
      JOIN chunks c ON c.chunk_id = chunks_fts.rowid
      JOIN documents d ON d.doc_id = c.doc_id
      WHERE chunks_fts MATCH @query ${familyFilter} ${metadataFilterSql}
      ORDER BY score
      LIMIT @limit
    `,
      )
      .all({ query, family, limit, ...namedParams });

    return {
      query,
      limit,
      ...(family ? { family } : {}),
      ...(classificationMetadata
        ? {
            query_class: classificationMetadata.query_class,
            confidence: classificationMetadata.confidence,
            classification_fallback:
              classificationMetadata.classification_fallback,
          }
        : {}),
      use_dense: false,
      results: rows.map((row) => ({
        ...readChunkRow(row),
        score: Number(row.score),
      })),
    };
  } finally {
    database.close();
  }
}
