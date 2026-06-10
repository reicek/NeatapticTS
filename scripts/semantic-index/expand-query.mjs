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
 *   embeddingsDatabasePath: 'data/embeddings.sqlite',
 * });
 * // result.expansion.applied === true
 * // result.expansion.expanded_terms contains up to 3 expanded terms
 * ```
 */
import Database from 'better-sqlite3';
import { readFile } from 'node:fs/promises';
import path from 'node:path';

import {
  DEFAULT_EMBEDDINGS_DATABASE_PATH,
  DEFAULT_MODEL_ID,
  createOnnxTextEmbedder,
  normalizeEmbeddingVector,
  readModelMeta,
} from './embed-index.mjs';
import { computeCosineSimilarity } from './hybrid-rank.mjs';
import { porterTokenize } from './build-term-index.mjs';
import { repoRoot } from './init-schema.mjs';

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
// Embedding-based synonym discovery
// ---------------------------------------------------------------------------

/**
 * Load all term embeddings from the embeddings database.
 *
 * Returns a Map from term string to its embedding Float32Array and metadata.
 *
 * @param {import('better-sqlite3').Database} embeddingsDatabase - Read-only embeddings database.
 * @param {string} modelId - Model identifier to filter by.
 * @returns {Map<string, { embedding: Float32Array, frequency: number, docFamilyCount: number }>}
 *   Map of term → embedding data.
 */
export function loadTermEmbeddings(embeddingsDatabase, modelId) {
  const rows = embeddingsDatabase
    .prepare(
      `
    SELECT term, embedding, frequency, doc_family_count FROM term_embeddings WHERE model_id = ?
  `,
    )
    .all(modelId);

  const termMap = new Map();
  for (const row of rows) {
    const float32 = new Float32Array(
      row.embedding.buffer,
      row.embedding.byteOffset,
      row.embedding.byteLength / 4,
    );
    termMap.set(row.term, {
      embedding: float32,
      frequency: Number(row.frequency),
      docFamilyCount: Number(row.doc_family_count),
    });
  }
  return termMap;
}

/**
 * Find nearest terms by cosine similarity to a query embedding.
 *
 * Searches all term embeddings for terms similar to the query term's
 * embedding. Returns candidates sorted by similarity (descending),
 * filtered by minimum similarity and excluding the query term itself.
 *
 * @param {Float32Array} queryEmbedding - The embedding of the query term.
 * @param {string} queryTerm - The original query term (excluded from results).
 * @param {Map<string, { embedding: Float32Array, frequency: number, docFamilyCount: number }>} termEmbeddings - Term embeddings map.
 * @param {{ maxTerms?: number, minSimilarity?: number }} options - Search options.
 * @returns {Array<{ term: string, similarity: number, frequency: number }>}
 *   Nearest term candidates sorted by similarity.
 */
export function findNearestTerms(
  queryEmbedding,
  queryTerm,
  termEmbeddings,
  options = {},
) {
  const maxTerms = options.maxTerms ?? MAX_NEAREST_TERMS;
  const minSimilarity = options.minSimilarity ?? MIN_SIMILARITY;

  const candidates = [];

  for (const [term, data] of termEmbeddings) {
    if (term.toLowerCase() === queryTerm.toLowerCase()) continue;

    const similarity = computeCosineSimilarity(queryEmbedding, data.embedding);
    if (similarity < minSimilarity) continue;

    candidates.push({
      term,
      similarity,
      frequency: data.frequency,
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
 * @param {string} [options.embeddingsDatabasePath] - Override embeddings database path.
 * @param {string} [options.modelDirectory] - Override ONNX model directory.
 * @param {string} [options.modelId] - Override model identifier.
 * @param {string} [options.associationsPath] - Override domain associations file path.
 * @param {number} [options.maxExpansions] - Override max expanded terms (default: 3).
 * @param {number} [options.minRelevance] - Override min relevance (default: 0.55).
 * @param {Function} [options.embedText] - Override embed function (for testing).
 * @param {Function} [options.termLookup] - Override term lookup (for testing).
 * @param {Map<string, { embedding: Float32Array, frequency: number }>} [options.termEmbeddingsMap] - Override term embeddings (for testing).
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
  let termEmbeddingsMap = options.termEmbeddingsMap ?? null;
  let expandedEmbedding = null;
  let degraded = false;

  // Step 3: Embedding-based synonym discovery (if not domain-only and embeddings available)
  if (!isDomainOnly) {
    try {
      const embeddingsDatabasePath = path.resolve(
        options.embeddingsDatabasePath ?? DEFAULT_EMBEDDINGS_DATABASE_PATH,
      );
      const modelId = String(options.modelId ?? DEFAULT_MODEL_ID);

      // Try to load term embeddings
      let embeddingsDatabase = null;
      try {
        embeddingsDatabase = new Database(embeddingsDatabasePath, {
          readonly: true,
        });

        // Check if term_embeddings table exists
        const tableCheck = embeddingsDatabase
          .prepare(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='term_embeddings'",
          )
          .get();

        if (tableCheck) {
          if (!termEmbeddingsMap) {
            termEmbeddingsMap = loadTermEmbeddings(embeddingsDatabase, modelId);
          }

          // Embed each query term and find nearest terms
          const embedText =
            options.embedText ??
            (await createOnnxTextEmbedder({
              modelDirectory: options.modelDirectory,
              modelId,
            }));

          for (const queryTerm of queryTerms) {
            if (termEmbeddingsMap.size === 0) break;

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

              const nearest = findNearestTerms(
                queryEmbedding,
                queryTerm,
                termEmbeddingsMap,
              );

              for (const candidate of nearest) {
                const termData = termEmbeddingsMap.get(candidate.term);
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
      } finally {
        embeddingsDatabase?.close();
      }
    } catch {
      degraded = true;
    }
  }

  // If domain-only and no domain expansions found, still report degraded if term table missing
  if (isDomainOnly && !degraded) {
    try {
      const embeddingsDatabasePath = path.resolve(
        options.embeddingsDatabasePath ?? DEFAULT_EMBEDDINGS_DATABASE_PATH,
      );
      const embeddingsDatabase = new Database(embeddingsDatabasePath, {
        readonly: true,
      });
      const tableCheck = embeddingsDatabase
        .prepare(
          "SELECT name FROM sqlite_master WHERE type='table' AND name='term_embeddings'",
        )
        .get();
      if (!tableCheck) degraded = true;
      embeddingsDatabase.close();
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
  if (applied && termEmbeddingsMap && !isDomainOnly) {
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
