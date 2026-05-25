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
import { requireString } from '../../agent-customization/mcp/mcp-utils.mjs';
import { queryDenseIndex } from '../../semantic-index/query-dense.mjs';
import { checkDenseReadiness } from '../../semantic-index/dense-readiness.mjs';
import { normalizeLimit, openCortexDatabase, readChunkRow, sanitizeFtsQuery } from './cortex-db.mjs';

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

/**
 * Search the indexed corpus using BM25 or hybrid dense reranking.
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
 * @param {number} [options.alpha] - BM25/dense blend weight (0 = BM25 only, 1 = dense only); default 0.5.
 * @param {string} [options.databasePath] - Override corpus SQLite database path.
 * @param {string} [options.embeddingsDatabasePath] - Override embeddings SQLite database path.
 * @param {string} [options.modelDirectory] - Override ONNX model directory path.
 * @param {string} [options.modelId] - Override ONNX model identifier.
 * @param {Function} [options.denseQuery] - Override dense-query implementation (for testing).
 * @param {Function} [options.readinessProbe] - Override readiness probe (for testing).
 * @returns {Promise<object>} Search result payload including `query`, `limit`, `use_dense`, and `results`.
 */
export async function searchCorpus(options = {}) {
  const rawQuery = requireString(options.query, 'query');
  const query = sanitizeFtsQuery(rawQuery);
  const limit = normalizeLimit(options.limit, 10);
  const family = typeof options.family === 'string' && options.family.trim() ? options.family.trim() : null;
  const useDense = options.use_dense !== false;

  if (!query) {
    if (!useDense) return createEmptyBm25Response({ family, limit, rawQuery });

    const readinessReport = await getDenseReadiness(options);
    if (readinessReport.state !== 'warm') {
      return createDegradedBm25Response({
        family,
        limit,
        query: rawQuery,
        readinessReport,
      });
    }

    return {
      alpha: normalizeAlpha(options.alpha),
      dense_state: 'warm',
      limit,
      ...(family ? { family } : {}),
      query: rawQuery,
      results: [],
      use_dense: true,
    };
  }

  if (useDense) {
    const readinessReport = await getDenseReadiness(options);
    if (readinessReport.state !== 'warm') {
      return createDegradedBm25Response({
        databasePath: options.databasePath,
        family,
        limit,
        query,
        readinessReport,
      });
    }

    const denseQuery = options.denseQuery ?? queryDenseIndex;
    const denseResult = await denseQuery({
      alpha: options.alpha,
      corpusDatabasePath: options.databasePath,
      dense: true,
      embeddingsDatabasePath: options.embeddingsDatabasePath,
      family,
      limit,
      modelDirectory: options.modelDirectory,
      modelId: options.modelId,
      query: rawQuery,
    });

    return {
      ...denseResult,
      dense_state: 'warm',
    };
  }

  return runBm25Search({ databasePath: options.databasePath, family, limit, query });
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
  cachedDenseReadiness = readinessReport.state === 'warm' ? readinessReport : null;
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
  const forcedState = typeof process.env.DENSE_FORCE_STATE === 'string' ? process.env.DENSE_FORCE_STATE.trim() : '';
  return forcedState === 'cold' || forcedState === 'model-only';
}

/**
 * Build an empty BM25-only response for a blank query.
 *
 * @param {{ family: string | null, limit: number, rawQuery: string }} params - Response parameters.
 * @returns {object} Empty search response with `use_dense: false` and an empty `results` array.
 */
function createEmptyBm25Response({ family, limit, rawQuery }) {
  return {
    limit,
    ...(family ? { family } : {}),
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
 * @param {{ databasePath?: string, family: string | null, limit: number, query: string, readinessReport: object }} params - Response parameters.
 * @returns {object} BM25 results with `dense_degraded: true` and degradation details.
 */
function createDegradedBm25Response({ databasePath, family, limit, query, readinessReport }) {
  const bm25Response = query
    ? runBm25Search({ databasePath, family, limit, query })
    : createEmptyBm25Response({ family, limit, rawQuery: query });

  return {
    ...bm25Response,
    dense_degraded: true,
    dense_reason: normalizeDenseReason(readinessReport.reason, readinessReport.state),
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
 * Execute a BM25 full-text search against the corpus SQLite database.
 *
 * Queries the FTS5 virtual table (`chunks_fts`) joined with `chunks` and
 * `documents`. Applies an optional family filter via a safe parameterized
 * clause and returns the top `limit` rows ordered by descending BM25 score.
 *
 * @param {{ databasePath?: string, family: string | null, limit: number, query: string }} params - Query parameters.
 * @returns {object} BM25 results payload with `query`, `limit`, `use_dense: false`, and `results`.
 */
function runBm25Search({ databasePath, family, limit, query }) {
  const database = openCortexDatabase(databasePath);

  try {
    const familyFilter = family ? 'AND d.doc_family = @family' : '';
    const rows = database.prepare(`
      SELECT d.file_path, d.doc_family, c.chunk_id, c.chunk_index, c.heading_path,
        c.body_text, c.char_start, c.char_end, bm25(chunks_fts) AS score
      FROM chunks_fts
      JOIN chunks c ON c.chunk_id = chunks_fts.rowid
      JOIN documents d ON d.doc_id = c.doc_id
      WHERE chunks_fts MATCH @query ${familyFilter}
      ORDER BY score
      LIMIT @limit
    `).all({ query, family, limit });

    return {
      query,
      limit,
      ...(family ? { family } : {}),
      use_dense: false,
      results: rows.map((row) => ({ ...readChunkRow(row), score: Number(row.score) })),
    };
  } finally {
    database.close();
  }
}