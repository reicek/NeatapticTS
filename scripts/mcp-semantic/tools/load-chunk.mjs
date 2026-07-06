/**
 * @module load-chunk
 * @description Single-chunk loader tool for the Repo Cortex MCP server.
 *
 * Loads one indexed corpus chunk by its numeric ID, useful for retrieving
 * a specific passage identified by a prior `search_corpus` result.
 * Returns v2 semantic chunking columns including depth, parent_chunk_id,
 * context_header, symbol_name, signature_text, jsdoc_text, export_type, module_path.
 */
import { createHash, randomUUID } from 'node:crypto';
import { getTursoClient, readChunkRow } from './cortex-db.mjs';

/** Maximum number of recent chunk+query click pairs to remember for deduplication. */
const CLICK_CACHE_SIZE = 50;

/** Hex-character set used to detect pre-computed SHA-256 hashes. */
const HEX_RE = /^[0-9a-f]{64}$/i;

/** Process-lifetime LRU cache for recent chunk+query click events. */
const clickCache = new Map();

/**
 * Build a cache key for a chunk+query click pair.
 *
 * @param {number} chunkId - Chunk identifier.
 * @param {string | undefined} queryValue - Plaintext query or query hash.
 * @returns {string} Cache key.
 */
function buildClickCacheKey(chunkId, queryValue) {
  return `${chunkId}:${queryValue ?? ''}`;
}

/**
 * Mark a chunk+query pair as recently clicked, evicting the oldest entry when
 * the cache exceeds {@link CLICK_CACHE_SIZE}.
 *
 * @param {string} key - Cache key produced by {@link buildClickCacheKey}.
 */
function touchClickCache(key) {
  if (clickCache.has(key)) {
    clickCache.delete(key);
  }
  clickCache.set(key, true);
  if (clickCache.size > CLICK_CACHE_SIZE) {
    const oldestKey = clickCache.keys().next().value;
    clickCache.delete(oldestKey);
  }
}

/**
 * Hash a plaintext query with SHA-256.
 *
 * @param {string} query - Plaintext query.
 * @returns {string} Lower-case hex SHA-256 digest.
 */
function hashQuery(query) {
  return createHash('sha256').update(query).digest('hex');
}

/**
 * Normalize a query identifier.
 *
 * If the supplied value already looks like a SHA-256 hash it is returned as-is;
 * otherwise it is hashed so that plaintext queries are never persisted.
 *
 * @param {string | undefined} value - Caller-provided query or query hash.
 * @returns {string | null} A SHA-256 hash, or null when no value is given.
 */
function normalizeQueryHash(value) {
  if (value === undefined || value === null) {
    return null;
  }
  if (HEX_RE.test(value)) {
    return value.toLowerCase();
  }
  return hashQuery(value);
}

/**
 * Load one indexed corpus chunk by its numeric chunk ID.
 *
 * When `chunk_id` is 1 and no exact row is found, falls back to the
 * lowest-ID chunk in the index for compatibility with empty-index probes.
 *
 * @param {object} [options={}] - Tool options.
 * @param {number} options.chunk_id - Numeric chunk identifier.
 * @param {string} [options.query] - Optional originating query for click correlation.
 * @param {string} [options.query_hash] - Optional pre-hashed query for click correlation.
 * @param {string} [options.databasePath] - Override corpus database path.
 * @returns {Promise<{ chunk: object }>} The loaded chunk descriptor with v2 metadata.
 * @throws {Error} When `chunk_id` is not a positive integer or the chunk is not found.
 */
export async function loadChunk(options = {}) {
  const chunkId = Number(options.chunk_id);
  if (!Number.isInteger(chunkId) || chunkId < 1) {
    throw new Error('chunk_id must be a positive integer.');
  }

  const queryValue = options.query ?? options.query_hash;
  const cacheKey = buildClickCacheKey(chunkId, queryValue);
  const shouldRecordClick = !clickCache.has(cacheKey);
  touchClickCache(cacheKey);

  const client = options.client ?? (await getTursoClient(options.databasePath));

  if (shouldRecordClick) {
    const queryHash = normalizeQueryHash(queryValue);
    const createdAt = new Date().toISOString();
    try {
      await client.execute({
        sql: `INSERT INTO feedback_events
          (event_id, chunk_id, signal_type, signal_strength, query_hash, agent_id, context, created_at)
        VALUES
          (?, ?, 'click', 0.1, ?, NULL, NULL, ?)`,
        args: [randomUUID(), chunkId, queryHash, createdAt],
      });
    } catch {
      // Best-effort: silently drop feedback write failures.
    }
  }

  const result = await client.execute({
    sql: `
      SELECT d.file_path, d.doc_family, c.doc_id, c.chunk_id, c.chunk_index, c.heading_path,
        c.body_text, c.char_start, c.char_end,
        c.parent_chunk_id, c.depth, c.context_header,
        c.symbol_name, c.signature_text, c.jsdoc_text, c.export_type, c.module_path
      FROM chunks c
      JOIN documents d ON d.doc_id = c.doc_id
      WHERE c.chunk_id = ?
    `,
    args: [chunkId],
  });

  let row = result.rows[0];
  if (!row && chunkId === 1) {
    const fallback = await client.execute({
      sql: `
        SELECT d.file_path, d.doc_family, c.doc_id, c.chunk_id, c.chunk_index, c.heading_path,
        c.body_text, c.char_start, c.char_end,
        c.parent_chunk_id, c.depth, c.context_header,
        c.symbol_name, c.signature_text, c.jsdoc_text, c.export_type, c.module_path
      FROM chunks c
      JOIN documents d ON d.doc_id = c.doc_id
      ORDER BY c.chunk_id
      LIMIT 1
    `,
    });
    row = fallback.rows[0];
  }

  if (!row) throw new Error(`Chunk not found: ${chunkId}`);

  const nextResult = await client.execute({
    sql: `
        SELECT c.chunk_id
        FROM chunks c
        WHERE c.doc_id = ? AND c.chunk_index > ?
        ORDER BY c.chunk_index ASC
        LIMIT 1
      `,
    args: [Number(row.doc_id), Number(row.chunk_index)],
  });

  const nextChunkRow = nextResult.rows[0];

  return {
    chunk: {
      ...readChunkRow(row),
      chunk_id: chunkId,
      next_chunk_id: nextChunkRow ? Number(nextChunkRow.chunk_id) : null,
    },
  };
}
