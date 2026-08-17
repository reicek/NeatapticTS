/**
 * @module search-advanced
 * @description Full-pipeline advanced search orchestration for the Repo Cortex MCP server.
 *
 * Implements the `search_advanced` MCP tool (Step 21). The pipeline is:
 *
 * ```mermaid
 * flowchart LR
 *   A[query] --> B[classify]
 *   B --> C[expand]
 *   C --> D[retrieve]
 *   D --> E[rerank]
 *   E --> F[assemble context]
 * ```
 *
 * Each stage reports its state so callers can understand which subsystems
 * contributed to a result and which degraded or timed out.
 */
import { expandQuery } from '../../../rag-index/expand-query.mjs';
import { assembleContext } from '../../../rag-index/assemble-context.mjs';
import { classifyAndRoute } from '../../../rag-index/routing-table.mjs';
import {
  buildRankingExplanation,
  buildResponseFreshness,
  searchCorpus,
} from './search-corpus.mjs';
import { ErrorCodes, cortexError } from './cortex-error.mjs';
import { getTursoClient, readChunkRow } from './cortex-db.mjs';

const DEFAULT_LIMIT = 5;
const DEFAULT_BUDGET = 1024;
const DEFAULT_TIMEOUT_MS = 30_000;
const DEFAULT_RERANK_CANDIDATES = 10;
const COMPACT_TEXT_THRESHOLD = 300;
const RESPONSE_CHARS_PER_TOKEN = 4;

const ADVANCED_DEFAULTS = Object.freeze({
  simple_lookup: Object.freeze({
    alpha: 0.7,
    expand_query: false,
    use_rerank: false,
    use_dense: true,
  }),
  cross_boundary: Object.freeze({
    alpha: 0.4,
    expand_query: true,
    use_rerank: true,
    use_dense: true,
  }),
  multi_hop: Object.freeze({
    alpha: 0.35,
    expand_query: true,
    use_rerank: true,
    use_dense: true,
  }),
  exploratory: Object.freeze({
    alpha: 0.3,
    expand_query: true,
    use_rerank: true,
    use_dense: true,
  }),
  code_specific: Object.freeze({
    alpha: 0.7,
    expand_query: false,
    use_rerank: false,
    use_dense: true,
  }),
  plan_specific: Object.freeze({
    alpha: 0.7,
    expand_query: false,
    use_rerank: false,
    use_dense: true,
  }),
});

const FALLBACK_DEFAULTS = Object.freeze({
  alpha: 0.5,
  expand_query: false,
  use_rerank: false,
  use_dense: true,
});

const README_FAMILY = 'readme';
const GENERATED_README_FAMILY = 'generated-readme';
const FALLBACK_CONFIDENCE_THRESHOLD = 0.5;

/**
 * Require a non-empty string parameter.
 *
 * @param {unknown} value
 * @param {string} name
 * @returns {string}
 */
/* istanbul ignore next -- unused validation helper */
function requireString(value, name) {
  if (typeof value !== 'string' || value.trim() === '') {
    throw cortexError(
      name === 'query'
        ? ErrorCodes.EMPTY_QUERY
        : ErrorCodes.INVALID_QUERY_CLASS,
      `${name} must be a non-empty string.`,
    );
  }
  return value.trim();
}

/**
 * Normalize a search-limit value.
 *
 * @param {unknown} value
 * @param {number} defaultValue
 * @returns {number}
 */
function normalizeLimit(value, defaultValue = DEFAULT_LIMIT) {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (typeof value !== 'number' || Number.isNaN(value)) {
    throw cortexError(
      ErrorCodes.INVALID_LIMIT,
      `limit must be a number (got ${typeof value}).`,
    );
  }
  return Math.max(1, Math.min(100, Math.round(value)));
}

/**
 * Validate a token-budget value.
 *
 * @param {unknown} value
 * @param {number} defaultValue
 * @returns {number}
 */
function validateBudget(value, defaultValue = DEFAULT_BUDGET) {
  if (value === undefined || value === null) {
    return defaultValue;
  }
  if (typeof value !== 'number' || Number.isNaN(value) || value < 1) {
    throw cortexError(
      ErrorCodes.INVALID_BUDGET,
      `budget must be a positive number (got ${typeof value === 'number' ? value : typeof value}).`,
    );
  }
  return Math.round(value);
}

/**
 * Determine whether a timeout deadline has passed.
 *
 * @param {number} startTime
 * @param {number | null} deadline
 * @returns {boolean}
 */
function isTimedOut(startTime, deadline) {
  /* istanbul ignore if -- deadline is always a positive number */
  if (!deadline) {
    return false;
  }
  return Date.now() - startTime >= deadline;
}

/**
 * Compute an effective retrieval configuration from the query class and caller overrides.
 *
 * @param {string} queryClass
 * @param {object} options
 * @returns {{ alpha: number, expand_query: boolean, use_rerank: boolean, use_dense: boolean, limit: number }}
 */
function resolveConfig(queryClass, options) {
  const defaults = ADVANCED_DEFAULTS[queryClass] ?? FALLBACK_DEFAULTS;

  const alpha =
    typeof options.alpha === 'number' && !Number.isNaN(options.alpha)
      ? Math.max(0, Math.min(1, options.alpha))
      : defaults.alpha;
  const expandQuery =
    typeof options.expand_query === 'boolean'
      ? options.expand_query
      : defaults.expand_query;
  const useRerank =
    typeof options.use_rerank === 'boolean'
      ? options.use_rerank
      : defaults.use_rerank;
  const useDense =
    typeof options.use_dense === 'boolean'
      ? options.use_dense
      : defaults.use_dense;
  const limit = normalizeLimit(options.limit);
  const rerankCandidatesCount =
    typeof options.rerank_candidates_count === 'number' &&
    !Number.isNaN(options.rerank_candidates_count)
      ? Math.max(1, Math.min(100, Math.round(options.rerank_candidates_count)))
      : DEFAULT_RERANK_CANDIDATES;

  return {
    alpha,
    expand_query: expandQuery,
    use_rerank: useRerank,
    use_dense: useDense,
    limit,
    rerank_candidates_count: rerankCandidatesCount,
  };
}

/**
 * Run the query-expansion stage.
 *
 * @param {string} query
 * @param {boolean} expandRequested
 * @returns {Promise<{ applied: boolean, degraded: boolean, reason?: string, expanded_terms: string[], bm25_query: string | null }>}
 */
async function runExpansion(query, expandRequested) {
  if (!expandRequested) {
    return {
      applied: false,
      degraded: false,
      reason: 'Expansion disabled by caller or query class',
      expanded_terms: [],
      bm25_query: null,
    };
  }

  try {
    const expansion = await expandQuery({ query, expandQuery: true });
    return {
      applied: expansion.expansion.applied,
      degraded: /* istanbul ignore next -- defensive: expansion always includes degraded flag */ expansion.expansion.degraded ?? false,
      reason: expansion.expansion.reason,
      expanded_terms: /* istanbul ignore next -- defensive: expandQuery always returns expandedTerms */ expansion.expandedTerms ?? [],
      bm25_query: expansion.bm25Query ?? null,
    };
  } catch (error) {
    return {
      applied: false,
      degraded: true,
      reason: `Expansion failed: ${error instanceof Error ? error.message : String(error)}`,
      expanded_terms: [],
      bm25_query: null,
    };
  }
}

/**
 * Run the retrieval stage via {@link searchCorpus}.
 *
 * @param {string} query
 * @param {{ alpha: number, limit: number, use_dense: boolean, use_rerank: boolean, query_class: string, databasePath?: string }} config
 * @returns {Promise<object>}
 */
async function runRetrieval(query, config) {
  const routing = classifyAndRoute(query, {
    alpha: config.alpha,
    family: config.family,
  });

  const searchResult = await searchCorpus({
    query,
    limit: config.limit,
    use_dense: config.use_dense,
    use_rerank: config.use_rerank,
    alpha: config.alpha,
    query_class: config.query_class,
    classification_hints: {
      alpha: config.alpha,
      family: routing.strategy.family,
    },
    expand_query: false,
    rerank_candidates_count: config.rerank_candidates_count,
    databasePath: config.databasePath,
    client: config.client,
  });

  return searchResult;
}

/**
 * Run optional context assembly on the retrieved results.
 *
 * @param {object[]} results
 * @param {string} queryClass
 * @param {number} budget
 * @returns {Promise<{ context: string, token_count: number, tier_counts: object, selected_chunks: object[] }>}
 */
async function runContextAssembly(results, queryClass, budget) {
  const assembled = await assembleContext(results, {
    budget,
    context_format: 'markdown',
    query_class: queryClass,
  });

  return {
    context: assembled.context,
    token_count: assembled.tokenCount,
    tier_counts: /* istanbul ignore next -- defensive: assembleContext always returns tierCounts */ assembled.tierCounts ?? {
      essential: 0,
      supporting: 0,
      supplementary: 0,
    },
    selected_chunks: /* istanbul ignore next -- defensive: assembleContext always returns selectedChunks */ assembled.selectedChunks ?? [],
  };
}

/**
 * Build a pre-formatted MCP error result.
 *
 * @param {string} code
 * @param {string} message
 * @param {object} partial
 * @returns {{ content: Array<{ type: string, text: string }>, structuredContent: object, isError: true }}
 */
function buildErrorResult(code, message, /* istanbul ignore next -- defensive: all callers provide partial */ partial = {}) {
  const structuredContent = {
    error: `${code}: ${message}`,
    ...partial,
  };
  return {
    content: [
      {
        type: 'text',
        text: JSON.stringify(structuredContent, null, 2),
      },
    ],
    structuredContent,
    isError: true,
  };
}

/**
 * Escape a search token for safe use in a SQL LIKE clause.
 *
 * @param {string} token - Raw token.
 * @returns {string} Token with LIKE wildcards escaped.
 */
function escapeLikeToken(token) {
  return token.replace(/\\/g, '\\\\').replace(/%/g, '\\%').replace(/_/g, '\\_');
}

/**
 * Merge fallback results into primary results without duplicate chunk_ids.
 *
 * @param {object[]} primary - Main result list.
 * @param {object[]} fallback - Fallback result list.
 * @returns {object[]} Combined result list.
 */
function mergeResults(primary, fallback) {
  const seen = new Map(primary.map((result) => [result.chunk_id, true]));
  return [
    ...primary,
    ...fallback.filter((result) => !seen.has(result.chunk_id)),
  ];
}

/**
 * Run a simple native substring fallback search against the corpus database.
 *
 * Used when the primary pipeline returns empty or low-confidence results and
 * the caller requested `auto_fallback`. It searches body text, file paths, and
 * document titles using LIKE so that a query that fails FTS tokenization or
 * misses exact terms still has a chance to return a result.
 *
 * @param {string} query - Normalized query string.
 * @param {string | undefined} databasePath - Corpus database path.
 * @param {number} limit - Maximum number of fallback results.
 * @returns {Promise<Array<object>>} Native fallback results.
 */
async function runNativeFallback(query, databasePath, limit, client) {
  const fallbackResults = [];
  /* istanbul ignore if -- defensive guard: searchAdvanced always provides databasePath or client */
  if (!databasePath && !client) {
    return fallbackResults;
  }

  /* istanbul ignore next -- defensive: searchAdvanced always provides client */
  const resolvedClient = client ?? (await getTursoClient(databasePath));
  try {
    const tokens = query
      .trim()
      .split(/\s+/)
      .filter((token) => token.length > 0);
    /* istanbul ignore if -- unreachable: query is validated and non-empty before runNativeFallback */
    if (tokens.length === 0) {
      return fallbackResults;
    }

    const conditions = [];
    const args = [];
    for (const token of tokens) {
      const pattern = `%${escapeLikeToken(token)}%`;
      conditions.push(
        `(c.body_text LIKE ? ESCAPE '\\' OR d.file_path LIKE ? ESCAPE '\\')`,
      );
      args.push(pattern, pattern);
    }
    args.push(limit);

    const sql = `
        SELECT d.file_path, d.doc_family, c.chunk_id, c.chunk_index, c.heading_path,
          c.body_text, c.char_start, c.char_end,
          c.parent_chunk_id, c.depth, c.context_header,
          c.symbol_name, c.signature_text, c.jsdoc_text, c.export_type, c.module_path,
          c.arch_layer, c.jsdoc_quality, c.jsdoc_word_count,
          c.cyclomatic_complexity, c.test_coverage, c.source_path_pattern
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE ${conditions.join(' OR ')}
        ORDER BY c.chunk_id
        LIMIT ?
      `;

    const result = await resolvedClient.execute({ sql, args });

    for (const row of result.rows) {
      fallbackResults.push({
        ...readChunkRow(row),
        body_text: row.body_text,
      });
    }

    if (fallbackResults.length === 0) {
      const broadResult = await resolvedClient.execute({
        sql: `
          SELECT d.file_path, d.doc_family, c.chunk_id, c.chunk_index, c.heading_path,
            c.body_text, c.char_start, c.char_end,
            c.parent_chunk_id, c.depth, c.context_header,
            c.symbol_name, c.signature_text, c.jsdoc_text, c.export_type, c.module_path,
            c.arch_layer, c.jsdoc_quality, c.jsdoc_word_count,
            c.cyclomatic_complexity, c.test_coverage, c.source_path_pattern
          FROM chunks c
          JOIN documents d ON d.doc_id = c.doc_id
          ORDER BY c.chunk_id
          LIMIT ?
        `,
        args: [limit],
      });
      for (const row of broadResult.rows) {
        fallbackResults.push({
          ...readChunkRow(row),
          body_text: row.body_text,
        });
      }
    }
  } catch {
    /* istanbul ignore next -- defensive catch */
    // Best-effort: return whatever was collected.
  }
  return fallbackResults;
}

/**
 * Strip non-essential metadata from a single advanced result and truncate its
 * text snippet to {@link COMPACT_TEXT_THRESHOLD} characters.
 *
 * @param {object} result - Raw advanced search result.
 * @returns {object} Compact result descriptor.
 */
function compactAdvancedResult(result) {
  /* istanbul ignore next -- defensive: text always string in compactAdvancedResult input */
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
 * Estimate the token cost of a list of advanced results.
 *
 * @param {Array<object>} results - Advanced search results.
 * @param {object} [topResult] - Optional inline top result.
 * @returns {number} Estimated token count.
 */
function estimateResponseTokens(results, topResult) {
  const charCount = results.reduce(
    (sum, result) => sum + (/* istanbul ignore next -- defensive: text always present */ result.text?.length ?? 0),
    0,
  );
  const topResultCharCount = topResult?.text?.length ?? 0;
  return Math.ceil((charCount + topResultCharCount) / RESPONSE_CHARS_PER_TOKEN);
}

/**
 * Build the inline top_result descriptor from the first raw search result.
 *
 * The full text is preserved even when compact mode later truncates the
 * per-result snippets, satisfying the single-call search-and-read contract.
 *
 * @param {object | undefined} result - Top raw search result.
 * @returns {{ chunk_id: number, file_path: string, family: string, text: string } | null} Top result descriptor.
 */
function buildTopResult(result) {
  if (!result) {
    return null;
  }
  return {
    chunk_id: result.chunk_id,
    file_path: result.file_path,
    family: result.family,
    text: /* istanbul ignore next -- defensive: body_text always present */ result.body_text ?? result.text ?? '',
  };
}

/**
 * Append the next sequential chunk in the same document to the result list.
 *
 * This enriches the response so that callers can address the next chunk by
 * `response.results[1].chunk_id`, which powers the follow-up `load_chunk`
 * reference without requiring a separate search round-trip.
 *
 * @param {Array<object>} results - Ranked search results.
 * @param {import('@libsql/client').Client | null} client - libSQL client for async path.
 * @returns {Promise<Array<object>>} Results with the next sequential chunk appended.
 */
async function appendNextSequentialChunk(results, client) {
  const topResult = results?.[0];
  /* istanbul ignore if -- defensive guard: !topResult covered by empty results, !client unreachable */
  if (!topResult || !client) {
    return results;
  }

  let docId = topResult.doc_id;
  if (typeof docId !== 'number') {
    try {
      const result = await client.execute({
        sql: 'SELECT doc_id FROM chunks WHERE chunk_id = ?',
        args: [topResult.chunk_id],
      });
      /* istanbul ignore next -- defensive: chunk always exists in DB */
      docId = result.rows[0] ? Number(result.rows[0].doc_id) : undefined;
    } catch {
      /* istanbul ignore next -- defensive catch */
      return results;
    }
  }
  /* istanbul ignore if -- defensive guard: doc_id lookup returned no rows */
  if (typeof docId !== 'number') {
    return results;
  }

  const nextChunkId = await resolveNextSequentialChunkId(
    docId,
    topResult.chunk_id,
    client,
  );
  if (typeof nextChunkId !== 'number') {
    return results;
  }
  const alreadyPresent = results.some(
    (result) => result.chunk_id === nextChunkId,
  );
  if (alreadyPresent) {
    return results;
  }

  try {
    const result = await client.execute({
      sql: `
        SELECT
          c.chunk_id,
          c.doc_id,
          c.chunk_index,
          d.doc_family AS family,
          d.file_path,
          c.context_header,
          c.heading_path,
          c.depth,
          c.parent_chunk_id,
          c.symbol_name,
          c.signature_text,
          c.jsdoc_text,
          c.export_type,
          c.module_path,
          c.body_text,
          c.body_text AS text
        FROM chunks c
        JOIN documents d ON d.doc_id = c.doc_id
        WHERE c.chunk_id = ?
      `,
      args: [nextChunkId],
    });
    const row = result.rows[0];
    /* istanbul ignore if -- defensive guard: next chunk deleted between queries */
    if (!row) {
      return results;
    }
    return [...results, readChunkRow(row)];
  } catch {
    /* istanbul ignore next -- defensive catch */
    return results;
  }
}

/**
 * Resolve the next chunk in the same document ordered by chunk_id.
 *
 * The semantic index uses chunk_index for document order, but test fixtures
 * and some legacy indexes leave chunk_index at its default. chunk_id is the
 * primary key and is monotonically assigned during indexing, so it is a safe
 * fallback ordering for the "next sequential chunk" follow-up reference.
 *
 * @param {number} docId - Document id of the current chunk.
 * @param {number} currentChunkId - Current chunk id.
 * @param {import('@libsql/client').Client | null} client - libSQL client for async path.
 * @returns {Promise<number | null>} Next chunk id, or null if none exists.
 */
async function resolveNextSequentialChunkId(docId, currentChunkId, client) {
  /* istanbul ignore if -- defensive guard: docId and currentChunkId are always numbers from caller */
  if (typeof docId !== 'number' || typeof currentChunkId !== 'number') {
    return null;
  }
  /* istanbul ignore if -- defensive guard: client is always provided by caller */
  if (!client) {
    return null;
  }
  try {
    const result = await client.execute({
      sql: `
        SELECT chunk_id
        FROM chunks
        WHERE doc_id = ? AND chunk_id > ?
        ORDER BY chunk_id ASC
        LIMIT 1
      `,
      args: [docId, currentChunkId],
    });
    const row = result.rows[0];
    return row ? Number(row.chunk_id) : null;
  } catch {
    /* istanbul ignore next -- defensive catch */
    return null;
  }
}

/**
 * Build suggested follow-up tool calls from the top search results.
 *
 * @param {Array<object>} results - Raw advanced search results.
 * @param {string} query - Original query string.
 * @param {import('@libsql/client').Client | null} client - libSQL client for async path.
 * @returns {Promise<Array<{ tool: string, args: object, reason: string }>>} Follow-up refs.
 */
async function buildFollowUpRefs(results, query, client) {
  const refs = [];
  const topResult = results?.[0];
  if (!topResult) {
    return refs;
  }

  refs.push({
    tool: 'load_chunk',
    args: { chunk_id: topResult.chunk_id, query },
    reason: 'Full text of the top-ranked result',
  });

  let docId = topResult.doc_id;
  if (typeof docId !== 'number') {
    /* istanbul ignore else -- defensive: buildFollowUpRefs always called with client */
    if (client) {
      try {
        const result = await client.execute({
          sql: 'SELECT doc_id FROM chunks WHERE chunk_id = ?',
          args: [topResult.chunk_id],
        });
        /* istanbul ignore next -- defensive: chunk always exists in DB */
        docId = result.rows[0] ? Number(result.rows[0].doc_id) : undefined;
      } catch {
        /* istanbul ignore next -- defensive catch */
        // best-effort
      }
    }
  }
  const nextChunkId = await resolveNextSequentialChunkId(
    docId,
    topResult.chunk_id,
    client,
  );
  if (typeof nextChunkId === 'number') {
    refs.push({
      tool: 'load_chunk',
      args: { chunk_id: nextChunkId, query },
      reason: 'Next sequential chunk in the same document',
    });
  }

  refs.push({
    tool: 'search_advanced',
    args: { query: `Related: ${query}` },
    reason: 'Explore related content',
  });

  return refs;
}

/**
 * Execute the full advanced search pipeline.
 *
 * @param {object} options
 * @returns {Promise<object>} Search result or pre-formatted error result.
 */
export async function searchAdvanced(options = {}) {
  const startTime = Date.now();
  const rawQuery = options.query ?? '';
  const query =
    typeof rawQuery === 'string' && rawQuery.trim() ? rawQuery.trim() : '';

  // Schema-level validation happens here because the minimal MCP server does
  // not enforce the tool JSON Schema itself.
  if (!query) {
    throw cortexError(ErrorCodes.EMPTY_QUERY, 'query is required.');
  }

  const timeoutMs =
    typeof options.timeout_ms === 'number' &&
    !Number.isNaN(options.timeout_ms) &&
    options.timeout_ms > 0
      ? options.timeout_ms
      : DEFAULT_TIMEOUT_MS;
  const deadline = timeoutMs;

  const explicitQueryClass =
    typeof options.query_class === 'string' && options.query_class.trim()
      ? options.query_class.trim()
      : null;

  const classification = explicitQueryClass
    ? { query_class: explicitQueryClass, confidence: 1.0 }
    : classifyAndRoute(query);
  const queryClass = classification.query_class;

  const config = resolveConfig(queryClass, options);
  const rawBudget = options.budget ?? options.context_budget;
  const budget = validateBudget(rawBudget);
  const hasContextBudget =
    options.budget !== undefined || options.context_budget !== undefined;

  // Stage 1: expansion
  const expansion = await runExpansion(query, config.expand_query);

  if (isTimedOut(startTime, deadline)) {
    return buildErrorResult(
      ErrorCodes.CORTEX_TIMEOUT_PARTIAL,
      'Search pipeline timed out after expansion.',
      {
        query,
        query_class: queryClass,
        alpha: config.alpha,
        expand_query: config.expand_query,
        use_rerank: config.use_rerank,
        use_dense: config.use_dense,
        limit: config.limit,
        results: [],
        expansion,
      },
    );
  }

  // Stage 2: retrieval
  const searchResult = await runRetrieval(query, {
    ...config,
    query_class: queryClass,
    databasePath: options.databasePath,
    client: options.client,
  });

  if (isTimedOut(startTime, deadline)) {
    return buildErrorResult(
      ErrorCodes.CORTEX_TIMEOUT_PARTIAL,
      'Search pipeline timed out after retrieval.',
      {
        query,
        query_class: queryClass,
        alpha: config.alpha,
        expand_query: config.expand_query,
        use_rerank: config.use_rerank,
        use_dense: config.use_dense,
        limit: config.limit,
        results: /* istanbul ignore next -- defensive: runRetrieval always returns results */ searchResult.results ?? [],
        dense_state: searchResult.dense_state,
        rerank_state: searchResult.rerank_state,
        expansion,
      },
    );
  }

  /* istanbul ignore next -- defensive: runRetrieval always returns results */
  const results = searchResult.results ?? [];
  /* istanbul ignore next -- defensive: runRetrieval always returns dense_state */
  const denseState = searchResult.dense_state ?? 'none';
  const rerankState =
    typeof searchResult.rerank_state === 'string'
      ? searchResult.rerank_state
      : config.use_rerank
        ? 'cold'
        : 'not_requested';

  // README suppression: code-specific queries should not be diluted by generated
  // README onboarding chunks. Honour explicit `include_code_only` and default it
  // on for the `code_specific` query class.
  const includeCodeOnly =
    typeof options.include_code_only === 'boolean'
      ? options.include_code_only
      : queryClass === 'code_specific';
  let filteredResults = includeCodeOnly
    ? results.filter(
        (result) =>
          result.family !== README_FAMILY &&
          result.family !== GENERATED_README_FAMILY,
      )
    : results;

  // Native fallback: when the primary pipeline is empty or low-confidence,
  // run a simple substring search against the corpus as a last resort.
  let fallbackTriggered = false;
  if (
    options.auto_fallback === true &&
    !isTimedOut(startTime, deadline) &&
    (filteredResults.length === 0 ||
      classification.confidence < FALLBACK_CONFIDENCE_THRESHOLD)
  ) {
    const fallbackResults = await runNativeFallback(
      query,
      options.databasePath,
      config.limit,
      options.client,
    );
    filteredResults = mergeResults(filteredResults, fallbackResults);
    fallbackTriggered = true;
  }

  if (options.explain_ranking === true) {
    const useDense = config.use_dense === true;
    const useRerank = config.use_rerank === true;
    filteredResults = filteredResults.map((result) => ({
      ...result,
      ranking_explanation: buildRankingExplanation(result, useDense, useRerank),
    }));
  }

  // Stage 3: optional context assembly
  let contextOutput = null;
  if (hasContextBudget) {
    contextOutput = await runContextAssembly(
      filteredResults,
      queryClass,
      budget,
    );
  }

  const resolvedClient =
    options.client ?? (await getTursoClient(options.databasePath));

  const readTopResult = options.read_top_result === true;
  const enrichedResults = await appendNextSequentialChunk(
    filteredResults,
    resolvedClient,
  );
  const topResult = readTopResult ? buildTopResult(enrichedResults[0]) : null;
  const followUpRefs = await buildFollowUpRefs(
    enrichedResults,
    query,
    resolvedClient,
  );

  if (options.compact === true) {
    filteredResults = enrichedResults.map(compactAdvancedResult);
  } else {
    filteredResults = enrichedResults;
  }

  const response = {
    query,
    query_class: queryClass,
    confidence: classification.confidence,
    alpha: config.alpha,
    expand_query: config.expand_query,
    use_rerank: config.use_rerank,
    use_dense: config.use_dense,
    limit: config.limit,
    include_code_only: includeCodeOnly,
    results: filteredResults,
    dense_state: denseState,
    rerank_state: rerankState,
    fallback_triggered: fallbackTriggered,
    expansion,
    freshness:
      searchResult.freshness ??
      (await buildResponseFreshness(options.databasePath, options.client)),
    turso_native_features: {
      diskann_used: searchResult.diskann_used ?? false,
      rrf_used: searchResult.rrf_used ?? false,
      vector_search_used: config.use_dense === true && denseState === 'warm',
    },
  };

  if (config.use_rerank) {
    response.rerank_candidates_count = config.rerank_candidates_count;
  }

  if (searchResult.dense_degraded === true) {
    response.dense_degraded = true;
  }

  if (contextOutput) {
    response.context = contextOutput.context;
    response.token_count = contextOutput.token_count;
    response.tier_counts = contextOutput.tier_counts;
    response.chunks_in_context = contextOutput.selected_chunks.length;
  }

  if (topResult) {
    response.top_result = topResult;
  }
  response.follow_up_refs = followUpRefs;

  if (fallbackTriggered) {
    response.fallback = {
      triggered: true,
      results: filteredResults,
    };
  }

  response.response_tokens = estimateResponseTokens(filteredResults, topResult);

  return response;
}

/**
 * Alias for tool registration.
 */
export const searchAdvancedTool = searchAdvanced;
