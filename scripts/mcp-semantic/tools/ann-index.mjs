/**
 * @module ann-index
 * @description Approximate-nearest-neighbor index builder and query helper for
 * the Repo Cortex dense retrieval layer.
 *
 * Wraps the optional `hnswlib-node` dependency so the server starts and serves
 * queries even when HNSW is not installed. When HNSW is unavailable or the
 * corpus is below the 50K-chunk threshold, the module falls back to a
 * metadata-only `brute_force_cached` index entry and maintains the external-id
 * to chunk-id mapping table for compatibility with HNSW-based callers.
 */
import Database from 'better-sqlite3';
import { mkdirSync } from 'node:fs';
import path from 'node:path';
import {
  DEFAULT_ANN_THRESHOLD,
  DEFAULT_HNSW_EF_CONSTRUCTION,
  DEFAULT_HNSW_EF_SEARCH,
  DEFAULT_HNSW_M,
  __hnswTestSeam,
  isHnswAvailable,
  resolveDenseStrategy,
} from './ann-strategy.mjs';

/**
 * Ensure the ANN metadata and chunk-map tables exist in the embeddings database.
 *
 * @param {import('better-sqlite3').Database} db - Open embeddings database connection.
 */
function ensureAnnTables(db) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS ann_index_meta (
      index_id TEXT PRIMARY KEY,
      index_type TEXT NOT NULL,
      model_id TEXT NOT NULL,
      model_sha256 TEXT,
      dimension INTEGER NOT NULL,
      metric TEXT NOT NULL DEFAULT 'cosine',
      max_elements INTEGER NOT NULL DEFAULT 0,
      current_elements INTEGER NOT NULL DEFAULT 0,
      m_param INTEGER,
      ef_construction_param INTEGER,
      ef_search_param INTEGER,
      build_status TEXT NOT NULL DEFAULT 'pending',
      build_started_at TEXT,
      build_completed_at TEXT,
      build_duration_ms INTEGER,
      build_error TEXT,
      last_incremental_update_at TEXT,
      index_file_path TEXT,
      created_at TEXT NOT NULL DEFAULT(datetime('now')),
      updated_at TEXT NOT NULL DEFAULT(datetime('now'))
    );

    CREATE TABLE IF NOT EXISTS ann_index_chunk_map (
      index_id TEXT NOT NULL,
      external_id INTEGER NOT NULL,
      chunk_id INTEGER NOT NULL,
      PRIMARY KEY(index_id, external_id)
    );
  `);
}

/**
 * Check whether a table exists in the database.
 *
 * @param {import('better-sqlite3').Database} db - Open database connection.
 * @param {string} tableName - Table name.
 * @returns {boolean} True if the table exists.
 */
function tableExists(db, tableName) {
  const row = db
    .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name=?")
    .get(tableName);
  return row !== undefined;
}

/**
 * Ensure the chunk_embeddings table exists before reading embeddings.
 *
 * @param {import('better-sqlite3').Database} db - Open embeddings database connection.
 */
function ensureChunkEmbeddingsTable(db) {
  db.exec(`
    CREATE TABLE IF NOT EXISTS chunk_embeddings (
      chunk_id INTEGER NOT NULL,
      model_id TEXT NOT NULL,
      dimension INTEGER NOT NULL,
      embedding BLOB NOT NULL,
      chunk_sha256 TEXT,
      created_at TEXT NOT NULL DEFAULT(datetime('now')),
      updated_at TEXT NOT NULL DEFAULT(datetime('now')),
      PRIMARY KEY(chunk_id, model_id)
    );
    CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_model
      ON chunk_embeddings(model_id, dimension);
  `);
}

/**
 * Populate the external-id to chunk-id mapping for an index.
 *
 * @param {import('better-sqlite3').Database} db - Open embeddings database connection.
 * @param {string} indexId - Index identifier.
 * @param {Array<{ chunk_id: number }>} chunkRows - Ordered chunk rows.
 */
function populateChunkMap(db, indexId, chunkRows) {
  const insert = db.prepare(
    'INSERT OR REPLACE INTO ann_index_chunk_map (index_id, external_id, chunk_id) VALUES (?, ?, ?)',
  );
  for (let externalId = 0; externalId < chunkRows.length; externalId++) {
    insert.run(indexId, externalId, chunkRows[externalId].chunk_id);
  }
}

/**
 * Clear an existing chunk-map for a given index.
 *
 * @param {import('better-sqlite3').Database} db - Open embeddings database connection.
 * @param {string} indexId - Index identifier.
 */
function clearChunkMap(db, indexId) {
  db.prepare('DELETE FROM ann_index_chunk_map WHERE index_id = ?').run(indexId);
}

/**
 * Decode a SQLite BLOB embedding back into a Float32Array.
 *
 * @param {Buffer} buffer - Raw SQLite BLOB buffer.
 * @param {number} dimension - Expected embedding dimension.
 * @returns {Float32Array} Decoded embedding vector.
 */
function decodeEmbeddingBuffer(buffer, dimension) {
  const expectedBytes = dimension * 4;
  const safeBuffer =
    buffer.length >= expectedBytes ? buffer.subarray(0, expectedBytes) : buffer;
  return new Float32Array(
    safeBuffer.buffer,
    safeBuffer.byteOffset,
    Math.floor(safeBuffer.byteLength / 4),
  );
}

/**
 * Build the ANN index for a given model and embedding dimension.
 *
 * Selects the strategy automatically and writes the index metadata into the
 * embeddings database. When the strategy is HNSW and `hnswlib-node` is available,
 * the HNSW graph is written to `indexFilePath` and the external-id mapping is
 * populated. When HNSW is unavailable the module transparently falls back to a
 * `brute_force_cached` metadata entry so downstream tools still receive a valid
 * `index_id` and `build_status`.
 *
 * @param {object} options - Build inputs.
 * @param {string} options.embeddingsDatabasePath - Path to the embeddings SQLite database.
 * @param {string} options.indexFilePath - Path where an HNSW index file should be written.
 * @param {string} [options.modelId='all-MiniLM-L6-v2'] - Embedding model identifier.
 * @param {number} [options.dimension=384] - Embedding dimension.
 * @param {boolean} [options.hnswAvailable] - Explicit HNSW availability override.
 * @param {string} [options.forceStrategy] - Force a specific strategy.
 * @returns {Promise<{ strategy: string, build_status: string, index_id: string | null, index_type: string | null, current_elements: number, build_error?: string | null }>} Build result summary.
 */
export async function buildAnnIndex(options = {}) {
  const embeddingsDatabasePath = options.embeddingsDatabasePath;
  if (!embeddingsDatabasePath) {
    throw new Error(
      'embeddingsDatabasePath is required to build an ANN index.',
    );
  }

  const indexFilePath =
    options.indexFilePath ??
    path.join(path.dirname(embeddingsDatabasePath), 'hnsw.index');
  const modelId = String(options.modelId ?? 'all-MiniLM-L6-v2');
  const dimension = Number(options.dimension ?? 384);

  const explicitHnswUnavailable = options.hnswAvailable === false;
  const effectiveHnswAvailable =
    typeof options.hnswAvailable === 'boolean'
      ? options.hnswAvailable
      : isHnswAvailable;

  const db = new Database(embeddingsDatabasePath);
  const startedAt = new Date().toISOString();

  try {
    ensureAnnTables(db);
    ensureChunkEmbeddingsTable(db);

    const chunkRows = db
      .prepare(
        'SELECT chunk_id FROM chunk_embeddings WHERE model_id = ? AND dimension = ? ORDER BY chunk_id',
      )
      .all(modelId, dimension);
    const chunkCount = chunkRows.length;

    const strategy = resolveDenseStrategy({
      chunkCount,
      annThreshold: DEFAULT_ANN_THRESHOLD,
      indexStatus: 'pending',
      forceStrategy: options.forceStrategy,
      hnswAvailable: effectiveHnswAvailable,
    });

    // When the caller explicitly disabled HNSW, report an error status instead
    // of silently serving a fallback. This keeps tooling aware of the mismatch.
    if (strategy !== 'hnsw' && explicitHnswUnavailable) {
      return {
        strategy,
        index_id: null,
        index_type: null,
        build_status: 'error',
        build_error: 'hnswlib-node is not available',
        current_elements: chunkCount,
      };
    }

    if (strategy !== 'hnsw') {
      const indexId = `brute_force_cached_${modelId}_${dimension}`;
      const now = new Date().toISOString();

      clearChunkMap(db, indexId);
      populateChunkMap(db, indexId, chunkRows);

      db.prepare(
        `INSERT OR REPLACE INTO ann_index_meta (
          index_id, index_type, model_id, model_sha256, dimension, metric, max_elements,
          current_elements, m_param, ef_construction_param, ef_search_param,
          build_status, build_started_at, build_completed_at, build_duration_ms,
          build_error, index_file_path, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
      ).run(
        indexId,
        'brute_force_cached',
        modelId,
        '',
        dimension,
        'cosine',
        chunkCount,
        chunkCount,
        null,
        null,
        null,
        'ready',
        startedAt,
        now,
        0,
        null,
        null,
        now,
      );

      return {
        strategy,
        index_id: indexId,
        index_type: 'brute_force_cached',
        build_status: 'ready',
        build_error: null,
        current_elements: chunkCount,
      };
    }

    // HNSW strategy.
    if (!effectiveHnswAvailable) {
      return {
        strategy,
        index_id: null,
        index_type: null,
        build_status: 'error',
        build_error: 'hnswlib-node is not available',
        current_elements: chunkCount,
      };
    }

    const { HierarchicalNSW } = await __hnswTestSeam.importFn();
    const maxElements = Math.max(chunkCount + 1, Math.ceil(chunkCount * 1.2));
    const index = new HierarchicalNSW('cosine', dimension);
    index.initIndex(
      maxElements,
      DEFAULT_HNSW_M,
      DEFAULT_HNSW_EF_CONSTRUCTION,
      100,
    );

    const embeddingRows = db
      .prepare(
        'SELECT chunk_id, embedding FROM chunk_embeddings WHERE model_id = ? AND dimension = ? ORDER BY chunk_id',
      )
      .all(modelId, dimension);

    for (let externalId = 0; externalId < embeddingRows.length; externalId++) {
      const vector = decodeEmbeddingBuffer(
        embeddingRows[externalId].embedding,
        dimension,
      );
      index.addPoint(vector, externalId);
    }

    index.setEf(DEFAULT_HNSW_EF_SEARCH);
    mkdirSync(path.dirname(indexFilePath), { recursive: true });
    index.writeIndexSync(indexFilePath);

    const indexId = `hnsw_${modelId}_${dimension}`;
    const now = new Date().toISOString();

    clearChunkMap(db, indexId);
    populateChunkMap(db, indexId, embeddingRows);

    db.prepare(
      `INSERT OR REPLACE INTO ann_index_meta (
        index_id, index_type, model_id, model_sha256, dimension, metric, max_elements,
        current_elements, m_param, ef_construction_param, ef_search_param,
        build_status, build_started_at, build_completed_at, build_duration_ms,
        build_error, index_file_path, updated_at
      ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
    ).run(
      indexId,
      'hnsw',
      modelId,
      '',
      dimension,
      'cosine',
      maxElements,
      embeddingRows.length,
      DEFAULT_HNSW_M,
      DEFAULT_HNSW_EF_CONSTRUCTION,
      DEFAULT_HNSW_EF_SEARCH,
      'ready',
      startedAt,
      now,
      0,
      null,
      indexFilePath,
      now,
    );

    return {
      strategy,
      index_id: indexId,
      index_type: 'hnsw',
      build_status: 'ready',
      build_error: null,
      current_elements: embeddingRows.length,
    };
  } finally {
    db.close();
  }
}

/**
 * Query an existing HNSW index for the k nearest neighbours of an embedding.
 *
 * The index is identified either by `indexId` (looked up in `ann_index_meta`) or,
 * when omitted, by the newest HNSW index for the default model/dimension pair.
 * If `hnswlib-node` is not installed this function throws because HNSW search
 * cannot be executed without the native library.
 *
 * @param {Float32Array | number[]} embedding - Query embedding.
 * @param {object} options - Query inputs.
 * @param {number} options.k - Number of neighbours to return.
 * @param {string} [options.indexId] - Index identifier to query.
 * @param {string} [options.embeddingsDatabasePath] - Path to the embeddings database.
 * @param {string} [options.modelId='all-MiniLM-L6-v2'] - Embedding model identifier.
 * @param {number} [options.dimension=384] - Embedding dimension.
 * @returns {Promise<Array<{ chunk_id: number, external_id: number, score: number }>>} Nearest neighbours in deterministic order (ascending score, tie-broken by chunk_id).
 */
export async function queryHnswIndex(embedding, options = {}) {
  if (!isHnswAvailable) {
    throw new Error('hnswlib-node is not available');
  }

  const k = Number(options.k ?? 10);
  const modelId = String(options.modelId ?? 'all-MiniLM-L6-v2');
  const dimension = Number(options.dimension ?? 384);
  const embeddingsDatabasePath = options.embeddingsDatabasePath;

  if (!embeddingsDatabasePath) {
    throw new Error(
      'embeddingsDatabasePath is required to query an HNSW index.',
    );
  }

  const db = new Database(embeddingsDatabasePath, {
    readonly: true,
    fileMustExist: true,
  });
  try {
    if (!tableExists(db, 'ann_index_meta')) {
      throw new Error('No HNSW index found for the given model/dimension.');
    }

    let indexFilePath = null;
    let resolvedIndexId = options.indexId ?? null;

    if (resolvedIndexId) {
      const row = db
        .prepare(
          'SELECT index_file_path FROM ann_index_meta WHERE index_id = ?',
        )
        .get(resolvedIndexId);
      indexFilePath = row?.index_file_path ?? null;
    } else {
      const row = db
        .prepare(
          `SELECT index_id, index_file_path FROM ann_index_meta
           WHERE index_type = 'hnsw' AND model_id = ? AND dimension = ?
           ORDER BY updated_at DESC LIMIT 1`,
        )
        .get(modelId, dimension);
      if (row) {
        resolvedIndexId = row.index_id;
        indexFilePath = row.index_file_path;
      }
    }

    if (!indexFilePath) {
      throw new Error('No HNSW index found for the given model/dimension.');
    }

    const { HierarchicalNSW } = await __hnswTestSeam.importFn();
    const index = new HierarchicalNSW('cosine', dimension);
    index.readIndexSync(indexFilePath);
    index.setEf(DEFAULT_HNSW_EF_SEARCH);

    const queryVector =
      embedding instanceof Float32Array
        ? embedding
        : new Float32Array(embedding);
    const result = index.searchKnn(queryVector, k);
    const neighbours = [];
    for (let i = 0; i < result.neighbors.length; i++) {
      const externalId = result.neighbors[i];
      const row = db
        .prepare(
          'SELECT chunk_id FROM ann_index_chunk_map WHERE index_id = ? AND external_id = ?',
        )
        .get(resolvedIndexId, externalId);
      if (row) {
        neighbours.push({
          chunk_id: row.chunk_id,
          external_id: externalId,
          score: Number(result.distances[i]),
        });
      }
    }

    // Deterministic tie-breaking: the same embedding against the same index must
    // always return the same order. HNSW can permute equal-distance neighbours,
    // so we stabilise by ascending score (cosine distance) then chunk_id.
    return neighbours.toSorted((a, b) => {
      if (a.score !== b.score) {
        return a.score - b.score;
      }
      return a.chunk_id - b.chunk_id;
    });
  } finally {
    db.close();
  }
}
