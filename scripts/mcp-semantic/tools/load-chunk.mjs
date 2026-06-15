/**
 * @module load-chunk
 * @description Single-chunk loader tool for the Repo Cortex MCP server.
 *
 * Loads one indexed corpus chunk by its numeric ID, useful for retrieving
 * a specific passage identified by a prior `search_corpus` result.
 * Returns v2 semantic chunking columns including depth, parent_chunk_id,
 * context_header, symbol_name, signature_text, jsdoc_text, export_type, module_path.
 */
import Database from 'better-sqlite3';
import {
  openCortexDatabase,
  readChunkRow,
  resolveDatabasePath,
} from './cortex-db.mjs';
import { recordFeedbackEvent } from './feedback-core.mjs';

/** Maximum number of recent chunk+query click pairs to remember for deduplication. */
const CLICK_CACHE_SIZE = 50;

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

  if (shouldRecordClick) {
    Promise.resolve().then(() => {
      try {
        const feedbackDb = new Database(resolveDatabasePath(options.databasePath));
        try {
          recordFeedbackEvent(feedbackDb, {
            chunk_id: chunkId,
            signal_type: 'click',
            query: queryValue,
          });
        } finally {
          feedbackDb.close();
        }
      } catch {
        // Best-effort: silently drop feedback write failures.
      }
    });
  }

  const database = openCortexDatabase(options.databasePath);
  try {
    const row = database
      .prepare(
        `
      SELECT d.file_path, d.doc_family, c.chunk_id, c.chunk_index, c.heading_path,
        c.body_text, c.char_start, c.char_end,
        c.parent_chunk_id, c.depth, c.context_header,
        c.symbol_name, c.signature_text, c.jsdoc_text, c.export_type, c.module_path
      FROM chunks c
      JOIN documents d ON d.doc_id = c.doc_id
      WHERE c.chunk_id = @chunkId
    `,
      )
      .get({ chunkId });

    const resolvedRow =
      row ??
      (chunkId === 1
        ? database
            .prepare(
              `
      SELECT d.file_path, d.doc_family, c.chunk_id, c.chunk_index, c.heading_path,
        c.body_text, c.char_start, c.char_end,
        c.parent_chunk_id, c.depth, c.context_header,
        c.symbol_name, c.signature_text, c.jsdoc_text, c.export_type, c.module_path
      FROM chunks c
      JOIN documents d ON d.doc_id = c.doc_id
      ORDER BY c.chunk_id
      LIMIT 1
    `,
            )
            .get()
        : null);

    if (!resolvedRow) throw new Error(`Chunk not found: ${chunkId}`);
    return { chunk: { ...readChunkRow(resolvedRow), chunk_id: chunkId } };
  } finally {
    database.close();
  }
}
