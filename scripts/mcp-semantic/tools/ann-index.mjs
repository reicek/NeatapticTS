/**
 * @module ann-index
 * @description DiskANN vector index builder for the Repo Cortex dense
 * retrieval layer.
 *
 * Creates a libSQL DiskANN vector index on the `chunks.embedding` column using
 * `CREATE INDEX ... USING libsql_vector_idx(embedding)`. DiskANN is a built-in
 * libSQL feature — no external native dependency is required.
 *
 * The index is configured with cosine metric, max_neighbors=59, alpha=1.2,
 * and search_l=80 for a latency/recall tradeoff tuned for 384-dimensional
 * embeddings and RAG retrieval limits (top-5/top-10).
 */
import { getTursoClient } from './cortex-db.mjs';
import {
  DEFAULT_ANN_THRESHOLD,
  DEFAULT_DISKANN_ALPHA,
  DEFAULT_DISKANN_MAX_NEIGHBORS,
  DEFAULT_DISKANN_SEARCH_L,
  resolveDenseStrategy,
} from './ann-strategy.mjs';

/**
 * Build the DiskANN vector index SQL with tuned parameters and an optional
 * partial-index WHERE clause.
 *
 * When `whereClause` is provided, the SQL includes a `WHERE` clause so the
 * index only covers rows matching the condition (pre-filtered ANN). This is
 * useful for creating family-specific or layer-specific ANN indexes.
 *
 * Uses `WITH KEY(metric='cosine', max_neighbors=59, alpha=1.2, search_l=80)`
 * for production Turso Cloud. Local libSQL may not support the `WITH KEY`
 * clause — in that case {@link buildAnnIndex} falls back to the basic index
 * syntax.
 *
 * @param {string | null} whereClause - Optional WHERE clause for partial indexing.
 * @returns {string} Tuned DiskANN CREATE INDEX SQL.
 */
function buildDiskAnnSqlTuned(whereClause) {
  const partialWhere = whereClause ? `WHERE ${whereClause}` : '';
  return `
    CREATE INDEX IF NOT EXISTS idx_chunks_embedding_diskann
    ON chunks USING libsql_vector_idx(embedding)
    WITH KEY(metric='cosine', max_neighbors=${DEFAULT_DISKANN_MAX_NEIGHBORS}, alpha=${DEFAULT_DISKANN_ALPHA}, search_l=${DEFAULT_DISKANN_SEARCH_L})
    ${partialWhere}
  `.trim();
}

/**
 * Build the fallback DiskANN vector index SQL for local libSQL environments
 * that do not support `WITH KEY`, with an optional partial-index WHERE clause.
 *
 * @param {string | null} whereClause - Optional WHERE clause for partial indexing.
 * @returns {string} Basic DiskANN CREATE INDEX SQL.
 */
function buildDiskAnnSqlBasic(whereClause) {
  const partialWhere = whereClause ? `WHERE ${whereClause}` : '';
  return `
    CREATE INDEX IF NOT EXISTS idx_chunks_embedding_diskann
    ON chunks USING libsql_vector_idx(embedding)
    ${partialWhere}
  `.trim();
}

/**
 * Build the DiskANN vector index on the `chunks.embedding` column.
 *
 * Executes the DiskANN `CREATE INDEX` SQL on the corpus database. When the
 * tuned `WITH KEY` syntax is not available (e.g. local libSQL), falls back to
 * the basic `libsql_vector_idx` index without tuning parameters.
 *
 * @param {object} options - Build inputs.
 * @param {string} [options.databasePath] - Path to the corpus database.
 * @param {import('@libsql/client').Client} [options.client] - Pre-opened libSQL client.
 * @param {string} [options.modelId='all-MiniLM-L6-v2'] - Embedding model identifier.
 * @param {number} [options.dimension=384] - Embedding dimension.
 * @param {string} [options.forceStrategy] - Force override (retained for compat).
 * @param {string} [options.whereClause] - Optional WHERE clause for partial index creation (e.g., "d.doc_family = 'src'").
 * @param {string} [options.partialFilter] - Alias for whereClause; partial index filter condition.
 * @returns {Promise<{ strategy: string, build_status: string, index_id: string | null, index_type: string | null, current_elements: number, build_error?: string | null }>} Build result summary.
 */
export async function buildAnnIndex(options = {}) {
  if (!options.databasePath && !options.client) {
    throw new Error(
      'databasePath or client is required to build a DiskANN index.',
    );
  }

  const modelId = String(options.modelId ?? 'all-MiniLM-L6-v2');
  const dimension = Number(options.dimension ?? 384);
  const whereClause = options.whereClause ?? options.partialFilter ?? null;

  const client = options.client ?? (await getTursoClient(options.databasePath));
  const startedAt = new Date().toISOString();

  // Count chunks for metadata.
  const chunkResult = await client.execute(
    'SELECT COUNT(*) as count FROM chunks',
  );
  const chunkCount = Number(chunkResult.rows[0]?.count ?? 0);

  const strategy = resolveDenseStrategy({
    chunkCount,
    annThreshold: DEFAULT_ANN_THRESHOLD,
    indexStatus: 'pending',
    forceStrategy: options.forceStrategy,
  });

  // Try the tuned DiskANN index first, then fall back to the basic syntax
  // for local libSQL environments that do not support WITH KEY.
  let buildError = null;
  let buildStatus = 'ready';

  try {
    await client.execute(buildDiskAnnSqlTuned(whereClause));
  } catch {
    try {
      await client.execute(buildDiskAnnSqlBasic(whereClause));
    } catch (err) {
      buildError = err instanceof Error ? err.message : String(err);
      buildStatus = 'error';
    }
  }

  const now = new Date().toISOString();
  const indexId = `diskann_${modelId}_${dimension}`;

  return {
    strategy,
    index_id: buildStatus === 'ready' ? indexId : null,
    index_type: buildStatus === 'ready' ? 'diskann' : null,
    build_status: buildStatus,
    build_error: buildError,
    current_elements: chunkCount,
    build_started_at: startedAt,
    build_completed_at: now,
  };
}
