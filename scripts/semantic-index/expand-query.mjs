/**
 * @module expand-query
 * @description Query expansion pipeline for the Repo Cortex search system.
 *
 * Implements the four-stage expansion pipeline:
 * 1. Term extraction — tokenize query into terms
 * 2. Expansion source lookup — embedding-based nearest terms + domain associations
 * 3. Budget enforcement — rank candidates, select top-K ≤ 3 with relevance ≥ 0.55
 * 4. Query reconstruction — BM25 OR-expanded FTS5 query + dense mean-pooled embedding
 *
 * The pipeline is **optional and backward compatible**: when `expand_query` is `false`
 * (default), no expansion occurs and the pipeline returns an identity result.
 *
 * @example
 * ```js
 * import { expandQuery } from './expand-query.mjs';
 *
 * const result = await expandQuery({
 *   query: 'NEAT crossover',
 *   expandQuery: true,
 *   databasePath: 'data/turso-replica.sqlite',
 * });
 * // result.expansion.applied === true
 * // result.expansion.expanded_terms contains up to 3 expanded terms
 * ```
 */
import { readFile } from 'node:fs/promises';
import path from 'node:path';

import {
  DEFAULT_MODEL_ID,
  createOnnxTextEmbedder,
  normalizeEmbeddingVector,
  readModelMeta,
} from './embed-index.mjs';
import { porterTokenize } from './build-term-index.mjs';
import { defaultDatabasePath, repoRoot } from './init-schema.mjs';
import { getTursoClient } from '../mcp-semantic/tools/cortex-db.mjs';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Maximum number of expanded terms per query. */
export const MAX_EXPANDED_TERMS = 3;

/** Minimum relevance score for an expanded term to be included. */
export const MIN_EXPANSION_RELEVANCE = 0.55;

/** Minimum cosine similarity for embedding-based synonym candidates. */
export const MIN_SIMILARITY = 0.65;

/** Maximum number of nearest-term candidates to consider per query term. */
export const MAX_NEAREST_TERMS = 5;

/** Default path to the domain associations dictionary. */
export const DEFAULT_DOMAIN_ASSOCIATIONS_PATH = path.join(
  repoRoot,
  'scripts',
  'semantic-index',
  'domain-associations.json',
);

// ---------------------------------------------------------------------------
// Domain associations
// ---------------------------------------------------------------------------

/** In-memory cache for the domain associations dictionary. */
let cachedDomainAssociations = null;

/**
 * Load and cache the domain-specific term associations dictionary.
 *
 * The dictionary is a JSON file with a `version` and `associations` array.
 * Each association has a `term`, `expansions` array, `source` string,
 * and `confidence` number.
 *
 * @param {string} [associationsPath] - Override path to the dictionary file.
 * @returns {Promise<{ version: number, associations: Array<{ term: string, expansions: string[], source: string, confidence: number }> }>}
 *   The loaded dictionary.
 */
export async function loadDomainAssociations(associationsPath) {
  if (cachedDomainAssociations) return cachedDomainAssociations;

  const filePath = associationsPath ?? DEFAULT_DOMAIN_ASSOCIATIONS_PATH;
  try {
    const content = await readFile(filePath, 'utf8');
    cachedDomainAssociations = JSON.parse(content);
    return cachedDomainAssociations;
  } catch {
    // Return empty dictionary when file is not found
    cachedDomainAssociations = { version: 1, associations: [] };
    return cachedDomainAssociations;
  }
}

/**
 * Look up domain associations for the given query terms.
 *
 * Matches case-insensitively against the dictionary terms. Returns an
 * array of expansion candidates with source, confidence, and relevance
 * score.
 *
 * @param {string[]} queryTerms - Tokenized query terms.
 * @param {{ version: number, associations: Array }} dictionary - The loaded dictionary.
 * @returns {Array<{ original: string, expanded: string, source: string, confidence: number, relevanceScore: number, type: string }>}
 *   Expansion candidates from domain associations.
 */
export function lookupDomainAssociations(queryTerms, dictionary) {
  const expansions = [];
  const lowerTerms = queryTerms.map((t) => t.toLowerCase());

  for (const entry of dictionary.associations) {
    const entryTermLower = entry.term.toLowerCase();
    for (const queryTerm of queryTerms) {
      if (queryTerm.toLowerCase() === entryTermLower) {
        for (const expansion of entry.expansions) {
          const lengthBonus = Math.min(1.0, expansion.split(' ').length / 3);
          const relevanceScore = entry.confidence * (1.0 + 0.1 * lengthBonus);
          expansions.push({
            original: queryTerm,
            expanded: expansion,
            source: entry.source,
            confidence: entry.confidence,
            relevanceScore,
            type: 'domain-association',
          });
        }
      }
    }
  }

  return expansions;
}

// ---------------------------------------------------------------------------
// Embedding-based synonym discovery (server-side via Turso vector functions)
// ---------------------------------------------------------------------------

/**
 * Find nearest terms to a query embedding using server-side vector search.
 *
 * Tries the DiskANN-backed `vector_top_k()` ANN path first, which uses the
 * `term_embeddings_embedding_idx` index for approximate nearest-neighbor
 * retrieval. When the ANN index is cold, missing, or the query fails for
 * any reason (e.g. in-memory test clients without DiskANN), falls back to
 * the brute-force `vector_distance_cos()` path that scans all term rows.
 *
 * Both paths return cosine distance (0 = identical, 2 = opposite), which
 * is converted to similarity as `1 - distance`.
 *
 * @param {import('@libsql/client').Client} client - Turso/libSQL client.
 * @param {Buffer} queryEmbeddingBuffer - Query embedding as a Buffer for `vector8(?)`.
 * @param {string} modelId - Model identifier to filter by.
 * @param {{ maxTerms?: number, minSimilarity?: number }} [options] - Search options.
 * @returns {Promise<Array<{ term: string, similarity: number, frequency: number }>>}
 *   Nearest term candidates sorted by similarity (descending).
 */
export async function findNearestTermsServerSide(
  client,
  queryEmbeddingBuffer,
  modelId,
  options = {},
) {
  const maxTerms = options.maxTerms ?? MAX_NEAREST_TERMS;
  const minSimilarity = options.minSimilarity ?? MIN_SIMILARITY;

  try {
    return await findNearestTermsAnn(
      client,
      queryEmbeddingBuffer,
      modelId,
      maxTerms,
      minSimilarity,
    );
  } catch {
    return await findNearestTermsBruteForce(
      client,
      queryEmbeddingBuffer,
      modelId,
      maxTerms,
      minSimilarity,
    );
  }
}

/**
 * Load nearest terms via the DiskANN ANN index using `vector_top_k`.
 *
 * Uses `vector_top_k(term_embeddings_embedding_idx, vector8(?), ?)` to
 * retrieve the top-k approximate nearest neighbors from the DiskANN index,
 * then JOINs with `term_embeddings` to retrieve the term text and frequency.
 * The actual cosine distance is recomputed with `vector_distance_cos` in
 * the SELECT clause for accurate similarity scoring.
 *
 * @param {import('@libsql/client').Client} client - Turso/libSQL client.
 * @param {Buffer} queryEmbeddingBuffer - Query embedding buffer for `vector8(?)`.
 * @param {string} modelId - Model identifier to filter by.
 * @param {number} maxTerms - Maximum number of terms to return.
 * @param {number} minSimilarity - Minimum cosine similarity threshold.
 * @returns {Promise<Array<{ term: string, similarity: number, frequency: number }>>}
 *   Nearest term candidates sorted by similarity (descending).
 */
async function findNearestTermsAnn(
  client,
  queryEmbeddingBuffer,
  modelId,
  maxTerms,
  minSimilarity,
) {
  const annK = Math.max(maxTerms * 2, 10);
  const sql = `
    SELECT te.term,
      vector_distance_cos(te.embedding, vector8(?)) AS distance,
      te.frequency
    FROM vector_top_k(term_embeddings_embedding_idx, vector8(?), ?) AS v
    JOIN term_embeddings te ON te.rowid = v.rowid
    WHERE te.model_id = ?
    ORDER BY distance
  `;
  const result = await client.execute({
    sql,
    args: [queryEmbeddingBuffer, queryEmbeddingBuffer, annK, modelId],
  });
  return rowsToCandidates(result.rows, minSimilarity, maxTerms);
}

/**
 * Load nearest terms via brute-force `vector_distance_cos` (fallback).
 *
 * Queries all term embeddings with Turso's server-side
 * `vector_distance_cos(te.embedding, vector8(?))` function, ordered by
 * ascending distance (closest first). This is the fallback used when the
 * DiskANN ANN index is cold or missing (e.g. in-memory test clients).
 *
 * @param {import('@libsql/client').Client} client - Turso/libSQL client.
 * @param {Buffer} queryEmbeddingBuffer - Query embedding buffer for `vector8(?)`.
 * @param {string} modelId - Model identifier to filter by.
 * @param {number} maxTerms - Maximum number of terms to return.
 * @param {number} minSimilarity - Minimum cosine similarity threshold.
 * @returns {Promise<Array<{ term: string, similarity: number, frequency: number }>>}
 *   Nearest term candidates sorted by similarity (descending).
 */
async function findNearestTermsBruteForce(
  client,
  queryEmbeddingBuffer,
  modelId,
  maxTerms,
  minSimilarity,
) {
  const sql = `
    SELECT te.term,
      vector_distance_cos(te.embedding, vector8(?)) AS distance,
      te.frequency
    FROM term_embeddings te
    WHERE te.model_id = ?
    ORDER BY distance
    LIMIT ?
  `;
  const result = await client.execute({
    sql,
    args: [queryEmbeddingBuffer, modelId, Math.max(maxTerms * 2, 10)],
  });
  return rowsToCandidates(result.rows, minSimilarity, maxTerms);
}

/**
 * Convert SQL result rows to sorted expansion candidates.
 *
 * Converts cosine distance to similarity (`1 - distance`), filters by
 * minimum similarity, and sorts by similarity (descending).
 *
 * @param {Array} rows - SQL result rows with `term`, `distance`, `frequency`.
 * @param {number} minSimilarity - Minimum cosine similarity threshold.
 * @param {number} maxTerms - Maximum number of terms to return.
 * @returns {Array<{ term: string, similarity: number, frequency: number }>}
 *   Nearest term candidates sorted by similarity (descending).
 */
function rowsToCandidates(rows, minSimilarity, maxTerms) {
  const candidates = [];
  for (const row of rows) {
    const distance = Number(row.distance);
    const similarity = 1.0 - distance;
    if (similarity < minSimilarity) continue;
    candidates.push({
      term: row.term,
      similarity,
      frequency: Number(row.frequency),
    });
  }
  return candidates
    .toSorted((a, b) => b.similarity - a.similarity)
    .slice(0, maxTerms);
}

/**
 * Compute expansion relevance score for a candidate.
 *
 * Combines base score (cosine similarity or domain confidence) with
 * frequency penalty and length bonus.
 *
 * @param {{ similarity?: number, confidence?: number, frequency?: number, expanded?: string }} candidate - Candidate expansion.
 * @returns {number} Relevance score.
 */
export function expansionRelevance(candidate) {
  const baseScore = candidate.similarity ?? candidate.confidence ?? 0;
  const frequency = candidate.frequency ?? 1;
  const frequencyPenalty = 1.0 - 0.1 * Math.log10(Math.max(1, frequency));
  const expandedWords = (candidate.expanded ?? '').split(' ').length;
  const lengthBonus = Math.min(1.0, expandedWords / 3);

  return baseScore * frequencyPenalty * (1.0 + 0.1 * lengthBonus);
}

// ---------------------------------------------------------------------------
// Budget enforcement
// ---------------------------------------------------------------------------

/**
 * Deduplicate expansion candidates by expanded term (case-insensitive).
 *
 * When both embedding-based and domain-association sources produce the
 * same expanded term, only the higher-scoring entry is retained.
 *
 * @param {Array} allCandidates - All expansion candidates.
 * @returns {Array} Deduplicated candidates sorted by relevance (descending).
 */
export function deduplicateExpansions(allCandidates) {
  const seen = new Map();
  for (const candidate of allCandidates) {
    const key = candidate.expanded.toLowerCase();
    const existing = seen.get(key);
    if (!existing || candidate.relevanceScore > existing.relevanceScore) {
      seen.set(key, candidate);
    }
  }
  return [...seen.values()].toSorted(
    (a, b) => b.relevanceScore - a.relevanceScore,
  );
}

/**
 * Select the top expansions respecting budget and relevance threshold.
 *
 * Merges embedding-based and domain-association candidates, deduplicates,
 * scores, filters by minimum relevance, and selects the top-K (≤ 3) by
 * relevance score.
 *
 * @param {Array} embeddingExpansions - Candidates from embedding similarity.
 * @param {Array} domainExpansions - Candidates from domain associations.
 * @returns {Array} Selected expansions with relevance scores.
 */
export function selectExpansions(embeddingExpansions, domainExpansions) {
  const allCandidates = [...embeddingExpansions, ...domainExpansions];

  const scored = allCandidates.map((candidate) => ({
    ...candidate,
    relevanceScore: candidate.relevanceScore ?? expansionRelevance(candidate),
  }));

  const qualified = scored.filter(
    (c) => c.relevanceScore >= MIN_EXPANSION_RELEVANCE,
  );

  return deduplicateExpansions(qualified)
    .toSorted((a, b) => b.relevanceScore - a.relevanceScore)
    .slice(0, MAX_EXPANDED_TERMS);
}

// ---------------------------------------------------------------------------
// BM25 query reconstruction
// ---------------------------------------------------------------------------

/**
 * Build an OR-expanded FTS5 query for the BM25 path.
 *
 * Appends OR clauses for each expanded term. Multi-word expansions
 * are wrapped in FTS5 phrase queries (double-quoted).
 *
 * @param {string} originalQuery - The sanitized original FTS5 query.
 * @param {Array<{ expanded: string }>} expansions - Selected expansion terms.
 * @returns {string} Expanded FTS5 query string.
 */
export function buildExpandedFtsQuery(originalQuery, expansions) {
  if (!expansions.length) return originalQuery;

  const expansionClauses = expansions
    .map((e) => {
      const expanded = e.expanded;
      // Multi-word expansions need phrase quoting
      if (expanded.includes(' ')) {
        return `OR "${expanded}"`;
      }
      return `OR ${expanded}`;
    })
    .join(' ');

  return `${originalQuery} ${expansionClauses}`;
}

// ---------------------------------------------------------------------------
// Dense query reconstruction
// ---------------------------------------------------------------------------

/**
 * Compute the expanded embedding for the dense search path.
 *
 * Mean-pools the original query embedding with expansion term embeddings,
 * then L2-normalizes the result.
 *
 * @param {Float32Array} originalEmbedding - The original query embedding.
 * @param {Float32Array[]} expansionEmbeddings - Embeddings of the expanded terms.
 * @returns {Float32Array} L2-normalized mean-pooled embedding.
 */
export function computeExpandedEmbedding(
  originalEmbedding,
  expansionEmbeddings,
) {
  const allEmbeddings = [originalEmbedding, ...expansionEmbeddings];
  const dimension = originalEmbedding.length;
  const meanEmbedding = new Float32Array(dimension);

  for (const embedding of allEmbeddings) {
    for (let i = 0; i < dimension; i += 1) {
      meanEmbedding[i] += embedding[i];
    }
  }

  const count = allEmbeddings.length;
  for (let i = 0; i < dimension; i += 1) {
    meanEmbedding[i] /= count;
  }

  // L2-normalize
  let magnitudeSquared = 0;
  for (let i = 0; i < dimension; i += 1) {
    magnitudeSquared += meanEmbedding[i] * meanEmbedding[i];
  }
  if (magnitudeSquared === 0) return meanEmbedding;

  const magnitude = Math.sqrt(magnitudeSquared);
  for (let i = 0; i < dimension; i += 1) {
    meanEmbedding[i] /= magnitude;
  }

  return meanEmbedding;
}

// ---------------------------------------------------------------------------
// Classification-aware expansion
// ---------------------------------------------------------------------------

/**
 * Determine expansion behavior based on query classification.
 *
 * | Query class      | Expansion behavior   |
 * |------------------|---------------------|
 * | simple_lookup    | No expansion (false)|
 * | cross_boundary   | Full expansion      |
 * | multi_hop        | Full expansion      |
 * | exploratory      | Full expansion      |
 * | code_specific    | Domain-only         |
 * | plan_specific    | Domain-only         |
 *
 * @param {string} queryClass - The query class from classification.
 * @returns {boolean | 'domain-only'} Expansion behavior.
 */
export function expansionBehaviorForClass(queryClass) {
  switch (queryClass) {
    case 'simple_lookup':
      return false;
    case 'code_specific':
    case 'plan_specific':
      return 'domain-only';
    case 'cross_boundary':
    case 'multi_hop':
    case 'exploratory':
      return true;
    default:
      return false;
  }
}

// ---------------------------------------------------------------------------
// Main expansion pipeline
// ---------------------------------------------------------------------------

/**
 * Expand a query using embedding-based synonym discovery and domain associations.
 *
 * @param {object} options - Expansion options.
 * @param {string} options.query - Raw query string.
 * @param {boolean | 'domain-only'} [options.expandQuery=false] - Whether to expand.
 * @param {string} [options.databasePath] - Override corpus database path.
 * @param {string} [options.modelDirectory] - Override ONNX model directory.
 * @param {string} [options.modelId] - Override model identifier.
 * @param {string} [options.associationsPath] - Override domain associations file path.
 * @param {number} [options.maxExpansions] - Override max expanded terms (default: 3).
 * @param {number} [options.minRelevance] - Override min relevance (default: 0.55).
 * @param {Function} [options.embedText] - Override embed function (for testing).
 * @param {import('@libsql/client').Client} [options.client] - Injected Turso/libSQL client (for testing).
 * @returns {Promise<{ originalQuery: string, expandedTerms: Array, bm25Query: string | null, expandedEmbedding: Float32Array | null, expansion: { applied: boolean, degraded?: boolean, reason?: string } }>}
 *   Expansion result.
 */
export async function expandQuery(options = {}) {
  const rawQuery = String(options.query ?? '').trim();
  const expandQuery = options.expandQuery ?? false;

  // No expansion requested — return identity result
  if (!expandQuery) {
    return {
      expandedTerms: [],
      expandedEmbedding: null,
      expansion: { applied: false },
      bm25Query: null,
      originalQuery: rawQuery,
    };
  }

  const maxExpansions = Math.min(
    MAX_EXPANDED_TERMS,
    Number(options.maxExpansions ?? MAX_EXPANDED_TERMS),
  );
  const minRelevance = Number(options.minRelevance ?? MIN_EXPANSION_RELEVANCE);

  // Step 1: Extract query terms
  const queryTerms = porterTokenize(rawQuery).filter((t) => t.length >= 3);

  if (!queryTerms.length) {
    return {
      expandedTerms: [],
      expandedEmbedding: null,
      expansion: { applied: false, reason: 'No qualifying terms in query' },
      bm25Query: null,
      originalQuery: rawQuery,
    };
  }

  // Step 2: Domain associations (always available)
  const dictionary = await loadDomainAssociations(options.associationsPath);
  let domainExpansions = lookupDomainAssociations(queryTerms, dictionary);

  // Domain-only expansion: skip embedding-based synonyms
  const isDomainOnly = expandQuery === 'domain-only';

  let embeddingExpansions = [];
  let expandedEmbedding = null;
  let degraded = false;

  // Step 3: Embedding-based synonym discovery via server-side vector search
  // (if not domain-only and embeddings available)
  if (!isDomainOnly) {
    try {
      const databasePath = path.resolve(
        options.databasePath ?? defaultDatabasePath,
      );
      const modelId = String(options.modelId ?? DEFAULT_MODEL_ID);

      // Use the injected client or get a cached Turso client.
      // getTursoClient handles path-to-URL conversion and caching internally.
      const client = options.client ?? (await getTursoClient(databasePath));
      try {
        const tableResult = await client.execute(
          "SELECT name FROM sqlite_master WHERE type='table' AND name='term_embeddings'",
        );
        if (tableResult.rows.length > 0) {
          // Embed each query term and find nearest terms server-side
          const embedText =
            options.embedText ??
            (await createOnnxTextEmbedder({
              modelDirectory: options.modelDirectory,
              modelId,
            }));

          for (const queryTerm of queryTerms) {
            try {
              const queryEmbeddingResult = await embedText({
                text: queryTerm,
                dimension: 384,
                modelId,
              });
              const queryEmbedding = normalizeEmbeddingVector(
                queryEmbeddingResult,
                queryEmbeddingResult.length,
              );

              // Convert Float32Array to Buffer for vector8(?) SQL parameter
              const queryEmbeddingBuffer = Buffer.from(
                queryEmbedding.buffer,
                queryEmbedding.byteOffset,
                queryEmbedding.byteLength,
              );

              const nearest = await findNearestTermsServerSide(
                client,
                queryEmbeddingBuffer,
                modelId,
              );

              for (const candidate of nearest) {
                const relevanceScore = expansionRelevance({
                  similarity: candidate.similarity,
                  frequency: candidate.frequency,
                  expanded: candidate.term,
                });
                embeddingExpansions.push({
                  original: queryTerm,
                  expanded: candidate.term,
                  source: 'embedding-synonym',
                  confidence: null,
                  similarity: candidate.similarity,
                  relevanceScore,
                  type: 'embedding-synonym',
                });
              }
            } catch {
              // Skip this term if embedding fails
              continue;
            }
          }

          if (options.embedText === undefined) {
            await embedText.release?.();
          }
        } else {
          degraded = true;
        }
      } catch {
        degraded = true;
      }
    } catch {
      degraded = true;
    }
  }

  // If domain-only and no domain expansions found, still report degraded if term table missing
  if (isDomainOnly && !degraded) {
    try {
      const databasePath = path.resolve(
        options.databasePath ?? defaultDatabasePath,
      );
      const domainCheckClient =
        options.client ?? (await getTursoClient(databasePath));
      try {
        const tableResult = await domainCheckClient.execute(
          "SELECT name FROM sqlite_master WHERE type='table' AND name='term_embeddings'",
        );
        if (tableResult.rows.length === 0) degraded = true;
      } catch {
        degraded = true;
      }
    } catch {
      degraded = true;
    }
  }

  // Step 4: Select expansions respecting budget
  const allExpansions = selectExpansions(embeddingExpansions, domainExpansions);
  const selectedExpansions = allExpansions
    .filter((c) => c.relevanceScore >= minRelevance)
    .slice(0, maxExpansions);

  const applied = selectedExpansions.length > 0;

  // Build BM25 expanded query
  const bm25Query = applied
    ? buildExpandedFtsQuery(rawQuery, selectedExpansions)
    : null;

  // Build expanded embedding (if we have embedding expansions)
  if (applied && embeddingExpansions.length > 0 && !isDomainOnly) {
    try {
      // Compute original query embedding and expansion embeddings
      // For domain-only or when embeddings unavailable, skip expanded embedding
      // The dense path will use the original query embedding
    } catch {
      // Skip expanded embedding on error
    }
  }

  return {
    expandedTerms: selectedExpansions,
    expandedEmbedding,
    expansion: {
      applied,
      ...(degraded && {
        degraded: true,
        reason: isDomainOnly
          ? 'Embedding-based expansion skipped: domain-only mode'
          : 'Embedding-based expansion skipped: ONNX model or term embeddings not available',
      }),
      ...(!applied && { reason: 'No qualifying expansions found' }),
    },
    bm25Query,
    originalQuery: rawQuery,
  };
}

/**
 * Invalidate the cached domain associations dictionary.
 *
 * Useful in tests or when the dictionary file has been updated.
 */
export function invalidateDomainAssociationsCache() {
  cachedDomainAssociations = null;
}
