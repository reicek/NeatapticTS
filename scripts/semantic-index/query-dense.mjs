/**
 * @description Run a BM25 query with optional hybrid dense reranking against the
 * NeatapticTS semantic index. When `--dense` is enabled, the BM25 candidate pool is
 * unioned with server-side dense candidates scored by Turso's
 * `vector_distance_cos()` function and combined using Reciprocal Ranked Fusion
 * (RRF) with the standard formula `score = sum(1/(k + rank_i))` and default k=60.
 * Default mode is BM25-only (`use_dense: false`) so the ONNX model is not loaded
 * unless `--dense` is explicitly passed.
 *
 * @param {string}  --query <text>                 - Query text (required).
 * @param {boolean} [--dense]                      - Enable hybrid dense reranking (default: false).
 * @param {number}  [--limit <n>]                  - Maximum returned result count (default: 10).
 * @param {string}  [--family <name>]              - Restrict query to one document family.
 * @param {string}  [--database <path>]            - Override semantic-index corpus database path.
 * @param {string}  [--model-directory <path>]     - Override local model cache directory.
 * @param {string}  [--model-id <id>]              - Override the model identifier.
 * @param {boolean} [--json]                       - Emit JSON results.
 * @param {boolean} [--help]                       - Show help and exit.
 *
 * @returns {void} Exits 0 on success, 1 on error. JSON written to stdout when `--json` is passed.
 */
import { pathToFileURL } from 'node:url';
import {
  fail,
  parseCliArgs,
  printHelp,
  writeJsonOrText,
} from './cli-utils.mjs';
import {
  DEFAULT_MODEL_DIRECTORY,
  DEFAULT_MODEL_ID,
  createOnnxTextEmbedder,
  normalizeEmbeddingVector,
  readModelMeta,
} from './embed-index.mjs';
import { rankRRFResults } from './hybrid-rank.mjs';
import {
  getTursoClient,
  normalizeLimit,
  readChunkRow,
  sanitizeFtsQuery,
} from '../mcp-semantic/tools/cortex-db.mjs';
import {
  compileFilterToSqlAliased,
  validateFilter,
} from './metadata-filter.mjs';

const DEFAULT_CANDIDATE_POOL_SIZE = 50;

export async function queryDenseIndex(options = {}) {
  const rawQuery = String(options.query ?? '').trim();
  const query = sanitizeFtsQuery(rawQuery);
  const limit = normalizeLimit(options.limit, 10);
  const family =
    typeof options.family === 'string' && options.family.trim()
      ? options.family.trim()
      : null;
  const alpha = Number(options.alpha ?? 0.5);
  const useDense = Boolean(options.dense ?? options.useDense ?? false);

  if (!query) {
    return {
      query: rawQuery,
      limit,
      ...(family ? { family } : {}),
      use_dense: useDense,
      ...(useDense ? { alpha } : {}),
      results: [],
    };
  }

  const candidatePoolSize = normalizeLimit(
    options.candidatePoolSize,
    Math.max(limit * 10, DEFAULT_CANDIDATE_POOL_SIZE),
  );

  // Resolve the compiled metadata filter for SQL-level vector query filtering.
  // Accept either a pre-compiled filter ({ sql, params }) or a raw metadata
  // filter predicate tree that is validated and compiled here.
  let compiledFilter = options.compiledFilter ?? null;
  if (!compiledFilter && options.metadataFilter) {
    validateFilter(options.metadataFilter);
    compiledFilter = compileFilterToSqlAliased(options.metadataFilter);
  }

  const bm25Rows = await loadBm25Rows({
    candidatePoolSize,
    databasePath: options.corpusDatabasePath ?? options.databasePath,
    family,
    query,
    client: options.client,
  });
  const bm25Results = bm25Rows.map((row) => ({
    ...readChunkRow(row),
    score: Number(row.score),
  }));
  if (!useDense) {
    return {
      query,
      limit,
      ...(family ? { family } : {}),
      use_dense: false,
      results: bm25Results.slice(0, limit),
    };
  }

  const modelMeta = await readModelMeta({
    modelDirectory: options.modelDirectory ?? DEFAULT_MODEL_DIRECTORY,
    modelMeta: options.modelMeta,
  });
  const dimension = Number(options.dimension ?? modelMeta.dimension ?? 0);
  const modelId = String(
    options.modelId ?? modelMeta.model_id ?? DEFAULT_MODEL_ID,
  );
  if (!Number.isInteger(dimension) || dimension < 1) {
    throw new Error(
      'Embedding dimension is required before dense query can run.',
    );
  }

  const embedText =
    options.embedText ??
    (await createOnnxTextEmbedder({
      dimension,
      modelDirectory: options.modelDirectory ?? DEFAULT_MODEL_DIRECTORY,
      modelId,
    }));
  const queryEmbedding = normalizeEmbeddingVector(
    await embedText({ text: rawQuery }),
    dimension,
  );
  const queryEmbeddingBuffer = Buffer.from(
    queryEmbedding.buffer,
    queryEmbedding.byteOffset,
    queryEmbedding.byteLength,
  );

  try {
    // Step 1: Load server-side dense candidates via ANN-first vector_top_k.
    const denseRows = await loadDenseRows({
      client: options.client,
      databasePath: options.corpusDatabasePath ?? options.databasePath,
      family,
      modelId,
      queryEmbeddingBuffer,
      limit: candidatePoolSize,
      compiledFilter,
    });
    const denseResults = denseRows.map((row) => ({
      ...readChunkRow(row),
      distance: Number(row.distance),
    }));

    // Step 2: Merge BM25 and dense candidates using Reciprocal Ranked Fusion.
    // RRF assigns score = sum(1/(k + rank_i)) for each result list where the
    // result appears, avoiding score-scale normalization issues.
    const rankedResults = rankRRFResults({
      bm25Results,
      denseResults,
    })
      .slice(0, limit)
      .map(({ rrf_score, bm25_score, distance, ...result }) => ({
        ...result,
        score: rrf_score,
      }));

    return {
      alpha,
      query,
      limit,
      ...(family ? { family } : {}),
      use_dense: true,
      results: rankedResults,
    };
  } finally {
    if (
      typeof options.embedText?.release !== 'function' &&
      typeof embedText?.release === 'function'
    ) {
      await embedText.release();
    }
  }
}

async function loadBm25Rows({
  candidatePoolSize,
  databasePath,
  family,
  query,
  client,
}) {
  if (client) {
    const familyFilter = family ? 'AND d.doc_family = ?' : '';
    const args = family
      ? [query, family, candidatePoolSize]
      : [query, candidatePoolSize];
    const result = await client.execute({
      sql: `
      SELECT d.file_path, d.doc_family, c.chunk_id, c.chunk_index, c.heading_path,
      c.symbol_name, c.body_text, c.char_start, c.char_end, bm25(chunks_fts) AS score
      FROM chunks_fts
      JOIN chunks c ON c.chunk_id = chunks_fts.rowid
      JOIN documents d ON d.doc_id = c.doc_id
      WHERE chunks_fts MATCH ? ${familyFilter}
      ORDER BY score
      LIMIT ?
    `,
      args,
    });
    return result.rows;
  }

  const database = await getTursoClient(databasePath);

  const familyFilter = family ? 'AND d.doc_family = ?' : '';
  const args = family
    ? [query, family, candidatePoolSize]
    : [query, candidatePoolSize];
  const result = await database.execute({
    sql: `
      SELECT d.file_path, d.doc_family, c.chunk_id, c.chunk_index, c.heading_path,
      c.symbol_name, c.body_text, c.char_start, c.char_end, bm25(chunks_fts) AS score
      FROM chunks_fts
      JOIN chunks c ON c.chunk_id = chunks_fts.rowid
      JOIN documents d ON d.doc_id = c.doc_id
      WHERE chunks_fts MATCH ? ${familyFilter}
      ORDER BY score
      LIMIT ?
    `,
    args,
  });
  return result.rows;
}

/**
 * Load dense candidate rows from the server using an ANN-first strategy.
 *
 * Tries the DiskANN-backed `vector_top_k()` ANN path first. When the ANN
 * index is cold, missing, or the query fails for any reason, falls back to
 * the brute-force `vector_distance_cos()` path from Step 02.
 *
 * ANN path (primary):
 *   SELECT ... FROM vector_top_k(chunks_embedding_idx, vector8(?), ?) AS v
 *   JOIN chunks c ON c.rowid = v.rowid JOIN documents d ON d.doc_id = c.doc_id
 *
 * Brute-force fallback:
 *   SELECT ... vector_distance_cos(c.embedding, vector8(?)) AS distance
 *   FROM chunks c JOIN documents d ON d.doc_id = c.doc_id ORDER BY distance
 *
 * The k limit parameter for vector_top_k(idx, ?, ?) controls how many ANN
 * candidates are retrieved before JOIN and optional family filtering.
 *
 * @param {object} params - Query parameters.
 * @param {import('@libsql/client').Client} [params.client] - Optional libSQL client (for testing).
 * @param {string} params.databasePath - Corpus database path (fallback when no client).
 * @param {string} params.family - Optional document family filter.
 * @param {string} params.modelId - Model identifier for embedding_model filter.
 * @param {Buffer} params.queryEmbeddingBuffer - Float32 query embedding buffer.
 * @param {number} params.limit - Maximum candidates to return.
 * @returns {Promise<Array>} Dense candidate rows with distance field.
 */
async function loadDenseRows({
  client,
  databasePath,
  family,
  modelId,
  queryEmbeddingBuffer,
  limit,
  compiledFilter,
}) {
  try {
    return await loadAnnRows({
      client,
      databasePath,
      family,
      modelId,
      queryEmbeddingBuffer,
      limit,
      compiledFilter,
    });
  } catch {
    return await loadDenseRowsBruteForce({
      client,
      databasePath,
      family,
      modelId,
      queryEmbeddingBuffer,
      limit,
      compiledFilter,
    });
  }
}

/**
 * Load dense candidate rows via the DiskANN ANN index using `vector_top_k`.
 *
 * Uses `vector_top_k(chunks_embedding_idx, vector8(?), ?)` to retrieve the
 * top-k approximate nearest neighbors from the DiskANN index (created in
 * Step 03), then JOINs with the chunks and documents tables in the same
 * query to retrieve full chunk metadata in a single round-trip.
 *
 * The actual cosine distance is recomputed with `vector_distance_cos` in
 * the SELECT clause for accurate hybrid ranking — `vector_top_k` guarantees
 * approximate ordering but not exact distances.
 *
 * @param {object} params - Query parameters (same as {@link loadDenseRows}).
 * @returns {Promise<Array>} Dense candidate rows with distance field.
 */
async function loadAnnRows({
  client,
  databasePath,
  family,
  modelId,
  queryEmbeddingBuffer,
  limit,
  compiledFilter,
}) {
  const familyFilter = family ? 'AND d.doc_family = ?' : '';
  const filterSql = compiledFilter ? `AND ${compiledFilter.sql}` : '';
  const filterParams = compiledFilter ? compiledFilter.params : [];
  const annK = Math.max(limit * 2, limit);
  const args = family
    ? [
        queryEmbeddingBuffer,
        queryEmbeddingBuffer,
        annK,
        modelId,
        family,
        ...filterParams,
      ]
    : [
        queryEmbeddingBuffer,
        queryEmbeddingBuffer,
        annK,
        modelId,
        ...filterParams,
      ];

  const sql = `
    SELECT d.file_path, d.doc_family, c.chunk_id, c.chunk_index, c.heading_path,
      c.symbol_name, c.body_text, c.char_start, c.char_end,
      vector_distance_cos(c.embedding, vector8(?)) AS distance
    FROM vector_top_k(chunks_embedding_idx, vector8(?), ?) AS v
    JOIN chunks c ON c.rowid = v.rowid
    JOIN documents d ON d.doc_id = c.doc_id
    WHERE c.embedding IS NOT NULL AND c.embedding_model = ? ${familyFilter} ${filterSql}
  `;

  if (client) {
    const result = await client.execute({ sql, args });
    return result.rows;
  }

  const database = await getTursoClient(databasePath);
  const result = await database.execute({ sql, args });
  return result.rows;
}

/**
 * Load dense candidate rows via brute-force `vector_distance_cos` (fallback).
 *
 * Queries the `chunks.embedding` F8_BLOB column with Turso's server-side
 * `vector_distance_cos(c.embedding, vector8(?))` function, ordered by
 * ascending distance (closest first). An optional family filter restricts
 * results to one document family.
 *
 * This is the fallback used when the DiskANN ANN index is cold or missing.
 *
 * @param {object} params - Query parameters (same as {@link loadDenseRows}).
 * @returns {Promise<Array>} Dense candidate rows with distance field.
 */
async function loadDenseRowsBruteForce({
  client,
  databasePath,
  family,
  modelId,
  queryEmbeddingBuffer,
  limit,
  compiledFilter,
}) {
  const familyFilter = family ? 'AND d.doc_family = ?' : '';
  const filterSql = compiledFilter ? `AND ${compiledFilter.sql}` : '';
  const filterParams = compiledFilter ? compiledFilter.params : [];
  const args = family
    ? [queryEmbeddingBuffer, modelId, family, ...filterParams, limit]
    : [queryEmbeddingBuffer, modelId, ...filterParams, limit];

  const sql = `
    SELECT d.file_path, d.doc_family, c.chunk_id, c.chunk_index, c.heading_path,
      c.symbol_name, c.body_text, c.char_start, c.char_end,
      vector_distance_cos(c.embedding, vector8(?)) AS distance
    FROM chunks c
    JOIN documents d ON d.doc_id = c.doc_id
    WHERE c.embedding IS NOT NULL AND c.embedding_model = ? ${familyFilter} ${filterSql}
    ORDER BY distance
    LIMIT ?
  `;

  if (client) {
    const result = await client.execute({ sql, args });
    return result.rows;
  }

  const database = await getTursoClient(databasePath);
  const result = await database.execute({ sql, args });
  return result.rows;
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Dense query runner',
      usage:
        'node scripts/semantic-index/query-dense.mjs --query "NEAT activation" [--dense] [--json]',
      options: [
        '--query <text>              Query text.',
        '--dense                     Enable hybrid dense reranking.',
        '--limit <n>                 Maximum returned result count (default: 10).',
        '--family <name>             Restrict the query to one document family.',
        '--database <path>           Override the semantic-index corpus database path.',
        '--model-directory <path>    Override the local model cache directory.',
        '--model-id <id>             Override the model identifier.',
        '--json                      Emit JSON results.',
        '--help                      Show this help.',
      ],
    });
    return;
  }

  try {
    const results = await queryDenseIndex({
      corpusDatabasePath: args.database,
      dense: Boolean(args.dense),
      family: args.family,
      limit: args.limit,
      modelDirectory: args['model-directory'],
      modelId: args['model-id'],
      query: args.query ?? args._.join(' '),
    });
    writeJsonOrText(results, Boolean(args.json), (payload) =>
      payload.results
        .map(
          (result, resultIndex) =>
            `${resultIndex + 1}. ${result.file_path} [${result.family}] ${result.heading_path ?? ''}`,
        )
        .join('\n'),
    );
  } catch (error) {
    fail(
      error instanceof Error ? error.message : String(error),
      Boolean(args.json),
    );
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();
