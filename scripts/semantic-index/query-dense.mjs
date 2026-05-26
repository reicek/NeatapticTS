/**
 * @description Run a BM25 query with optional hybrid dense reranking against the
 * NeatapticTS semantic index. When `--dense` is enabled, the BM25 candidate pool is
 * unioned with the nearest-neighbour dense candidates and scores are combined using the
 * formula `score = alpha * bm25_norm + (1 - alpha) * cosine_sim`. Default mode is
 * BM25-only (`use_dense: false`) so the ONNX model is not loaded unless `--dense` is
 * explicitly passed.
 *
 * @param {string}  --query <text>                 - Query text (required).
 * @param {boolean} [--dense]                      - Enable hybrid dense reranking (default: false).
 * @param {number}  [--alpha <n>]                  - BM25 weight in hybrid formula (0–1, default: 0.5).
 * @param {number}  [--limit <n>]                  - Maximum returned result count (default: 10).
 * @param {string}  [--family <name>]              - Restrict query to one document family.
 * @param {string}  [--database <path>]            - Override semantic-index corpus database path.
 * @param {string}  [--embeddings-database <p>]    - Override embeddings database path.
 * @param {string}  [--model-directory <path>]     - Override local model cache directory.
 * @param {string}  [--model-id <id>]              - Override the model identifier.
 * @param {boolean} [--json]                       - Emit JSON results.
 * @param {boolean} [--help]                       - Show help and exit.
 *
 * @returns {void} Exits 0 on success, 1 on error. JSON written to stdout when `--json` is passed.
 */
import Database from 'better-sqlite3';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { fail, parseCliArgs, printHelp, writeJsonOrText } from './cli-utils.mjs';
import {
  DEFAULT_EMBEDDINGS_DATABASE_PATH,
  DEFAULT_MODEL_DIRECTORY,
  DEFAULT_MODEL_ID,
  createOnnxTextEmbedder,
  normalizeEmbeddingVector,
  readModelMeta,
} from './embed-index.mjs';
import { DEFAULT_HYBRID_ALPHA, computeCosineSimilarity, rankHybridResults } from './hybrid-rank.mjs';
import { normalizeLimit, openCortexDatabase, readChunkRow, sanitizeFtsQuery } from '../mcp-semantic/tools/cortex-db.mjs';

const DEFAULT_CANDIDATE_POOL_SIZE = 50;

export async function queryDenseIndex(options = {}) {
  const rawQuery = String(options.query ?? '').trim();
  const query = sanitizeFtsQuery(rawQuery);
  const limit = normalizeLimit(options.limit, 10);
  const family = typeof options.family === 'string' && options.family.trim() ? options.family.trim() : null;
  const alpha = normalizeAlpha(options.alpha);
  const useDense = Boolean(options.dense ?? options.useDense ?? false);

  if (!query) {
    return { query: rawQuery, limit, ...(family ? { family } : {}), use_dense: useDense, ...(useDense ? { alpha } : {}), results: [] };
  }

  const candidatePoolSize = normalizeLimit(options.candidatePoolSize, Math.max(limit * 10, DEFAULT_CANDIDATE_POOL_SIZE));
  const bm25Rows = loadBm25Rows({ candidatePoolSize, databasePath: options.corpusDatabasePath ?? options.databasePath, family, query });
  const bm25Results = bm25Rows.map((row) => ({ ...readChunkRow(row), score: Number(row.score) }));
  if (!useDense) {
    return {
      query,
      limit,
      ...(family ? { family } : {}),
      use_dense: false,
      results: bm25Results.slice(0, limit),
    };
  }

  const embeddingsDatabasePath = path.resolve(options.embeddingsDatabasePath ?? DEFAULT_EMBEDDINGS_DATABASE_PATH);
  const modelMeta = await readModelMeta({ modelDirectory: options.modelDirectory ?? DEFAULT_MODEL_DIRECTORY, modelMeta: options.modelMeta });
  const dimension = Number(options.dimension ?? modelMeta.dimension ?? 0);
  const modelId = String(options.modelId ?? modelMeta.model_id ?? DEFAULT_MODEL_ID);
  if (!Number.isInteger(dimension) || dimension < 1) {
    throw new Error('Embedding dimension is required before dense query can run.');
  }

  const embedText = options.embedText ?? await createOnnxTextEmbedder({
    dimension,
    modelDirectory: options.modelDirectory ?? DEFAULT_MODEL_DIRECTORY,
    modelId,
  });
  const queryEmbedding = normalizeEmbeddingVector(await embedText({ text: rawQuery }), dimension);
  const embeddingsDatabase = new Database(embeddingsDatabasePath, { readonly: true, fileMustExist: true });

  try {
    const bm25EmbeddingByChunkId = loadEmbeddingsForChunks({
      chunkIds: bm25Results.map(({ chunk_id: chunkId }) => chunkId),
      database: embeddingsDatabase,
      modelId,
    });
    const denseCandidateRows = loadDenseCandidateRows({
      corpusDatabasePath: options.corpusDatabasePath ?? options.databasePath,
      database: embeddingsDatabase,
      dimension,
      family,
      limit: candidatePoolSize,
      modelId,
      queryEmbedding,
    });
    const candidateRowsByChunkId = mergeCandidateRows(bm25Results, denseCandidateRows);
    const bm25ScoreByChunkId = new Map(bm25Results.map((result) => [result.chunk_id, -Number(result.score)]));
    const rankedResults = rankHybridResults({
      alpha,
      candidates: [...candidateRowsByChunkId.values()].map((result) => ({
        ...result,
        bm25_score: bm25ScoreByChunkId.get(result.chunk_id) ?? 0,
        embedding: result.embedding ?? bm25EmbeddingByChunkId.get(result.chunk_id) ?? new Float32Array(dimension),
      })),
      queryEmbedding,
    }).slice(0, limit).map(({ embedding, ...result }) => result);

    return {
      alpha,
      query,
      limit,
      ...(family ? { family } : {}),
      use_dense: true,
      results: rankedResults,
    };
  } finally {
    embeddingsDatabase.close();
    if (typeof options.embedText?.release !== 'function' && typeof embedText?.release === 'function') {
      await embedText.release();
    }
  }
}

function loadBm25Rows({ candidatePoolSize, databasePath, family, query }) {
  const database = openCortexDatabase(databasePath);

  try {
    const familyFilter = family ? 'AND d.doc_family = @family' : '';
    return database.prepare(`
      SELECT d.file_path, d.doc_family, c.chunk_id, c.chunk_index, c.heading_path,
        c.body_text, c.char_start, c.char_end, bm25(chunks_fts) AS score
      FROM chunks_fts
      JOIN chunks c ON c.chunk_id = chunks_fts.rowid
      JOIN documents d ON d.doc_id = c.doc_id
      WHERE chunks_fts MATCH @query ${familyFilter}
      ORDER BY score
      LIMIT @limit
    `).all({ family, limit: candidatePoolSize, query });
  } finally {
    database.close();
  }
}

function loadEmbeddingsForChunks({ chunkIds, database, modelId }) {
  if (chunkIds.length === 0) return new Map();

  const placeholders = chunkIds.map(() => '?').join(', ');
  const rows = database.prepare(`
    SELECT chunk_id, embedding
    FROM chunk_embeddings
    WHERE model_id = ? AND chunk_id IN (${placeholders})
  `).all(modelId, ...chunkIds);

  return new Map(rows.map((row) => [Number(row.chunk_id), decodeEmbeddingBlob(row.embedding)]));
}

function loadDenseCandidateRows({ corpusDatabasePath, database, dimension, family, limit, modelId, queryEmbedding }) {
  const allowedChunkIds = family ? loadFamilyChunkIds({ databasePath: corpusDatabasePath, family }) : null;
  const rankedEmbeddingRows = database.prepare(`
    SELECT chunk_id, embedding
    FROM chunk_embeddings
    WHERE model_id = ? AND dimension = ?
  `).all(modelId, dimension)
    .map((row) => ({
      chunk_id: Number(row.chunk_id),
      embedding: decodeEmbeddingBlob(row.embedding),
    }))
    .filter((row) => !allowedChunkIds || allowedChunkIds.has(row.chunk_id))
    .map((row) => ({
      ...row,
      cosine_score: computeCosineSimilarity(queryEmbedding, row.embedding),
    }))
    .toSorted((leftRow, rightRow) => rightRow.cosine_score - leftRow.cosine_score)
    .slice(0, limit);

  const chunkRowsById = loadChunkRowsByIds({
    chunkIds: rankedEmbeddingRows.map(({ chunk_id: chunkId }) => chunkId),
    databasePath: corpusDatabasePath,
    family,
  });

  return rankedEmbeddingRows
    .map((embeddingRow) => {
      const chunkRow = chunkRowsById.get(embeddingRow.chunk_id);
      return chunkRow ? { ...chunkRow, embedding: embeddingRow.embedding } : null;
    })
    .filter(Boolean);
}

function loadFamilyChunkIds({ databasePath, family }) {
  const database = openCortexDatabase(databasePath);

  try {
    const rows = database.prepare(`
      SELECT c.chunk_id
      FROM chunks c
      JOIN documents d ON d.doc_id = c.doc_id
      WHERE d.doc_family = ?
    `).all(family);
    return new Set(rows.map((row) => Number(row.chunk_id)));
  } finally {
    database.close();
  }
}

function loadChunkRowsByIds({ chunkIds, databasePath, family }) {
  if (chunkIds.length === 0) return new Map();

  const database = openCortexDatabase(databasePath);
  const placeholders = chunkIds.map(() => '?').join(', ');
  const familyFilter = family ? 'AND d.doc_family = ?' : '';
  const parameters = family ? [...chunkIds, family] : chunkIds;

  try {
    const rows = database.prepare(`
      SELECT d.file_path, d.doc_family, c.chunk_id, c.chunk_index, c.heading_path,
        c.body_text, c.char_start, c.char_end
      FROM chunks c
      JOIN documents d ON d.doc_id = c.doc_id
      WHERE c.chunk_id IN (${placeholders}) ${familyFilter}
    `).all(...parameters);
    return new Map(rows.map((row) => {
      const chunkRow = readChunkRow(row);
      return [chunkRow.chunk_id, chunkRow];
    }));
  } finally {
    database.close();
  }
}

function mergeCandidateRows(bm25Results, denseCandidateRows) {
  const candidateRowsByChunkId = new Map(bm25Results.map((result) => [result.chunk_id, result]));

  for (const denseCandidateRow of denseCandidateRows) {
    const existingCandidateRow = candidateRowsByChunkId.get(denseCandidateRow.chunk_id);
    candidateRowsByChunkId.set(
      denseCandidateRow.chunk_id,
      existingCandidateRow ? { ...existingCandidateRow, embedding: denseCandidateRow.embedding } : denseCandidateRow
    );
  }

  return candidateRowsByChunkId;
}

function decodeEmbeddingBlob(blob) {
  const float32Length = Math.trunc(blob.byteLength / Float32Array.BYTES_PER_ELEMENT);
  return Float32Array.from(new Float32Array(blob.buffer, blob.byteOffset, float32Length));
}

function normalizeAlpha(alpha) {
  const numericAlpha = Number(alpha ?? DEFAULT_HYBRID_ALPHA);
  if (!Number.isFinite(numericAlpha)) return DEFAULT_HYBRID_ALPHA;
  return Math.min(Math.max(numericAlpha, 0), 1);
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Dense query runner',
      usage: 'node scripts/semantic-index/query-dense.mjs --query "NEAT activation" [--dense] [--json]',
      options: [
        '--query <text>              Query text.',
        '--dense                     Enable hybrid dense reranking.',
        '--alpha <n>                 Hybrid BM25 weight (default: 0.5).',
        '--limit <n>                 Maximum returned result count (default: 10).',
        '--family <name>             Restrict the query to one document family.',
        '--database <path>           Override the semantic-index corpus database path.',
        '--embeddings-database <p>   Override the embeddings database path.',
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
      alpha: args.alpha,
      corpusDatabasePath: args.database,
      dense: Boolean(args.dense),
      embeddingsDatabasePath: args['embeddings-database'],
      family: args.family,
      limit: args.limit,
      modelDirectory: args['model-directory'],
      modelId: args['model-id'],
      query: args.query ?? args._.join(' '),
    });
    writeJsonOrText(results, Boolean(args.json), (payload) => payload.results
      .map((result, resultIndex) => `${resultIndex + 1}. ${result.file_path} [${result.family}] ${result.heading_path ?? ''}`)
      .join('\n'));
  } catch (error) {
    fail(error instanceof Error ? error.message : String(error), Boolean(args.json));
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();