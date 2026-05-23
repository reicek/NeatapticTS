import { requireString } from '../../agent-customization/mcp/mcp-utils.mjs';
import { queryDenseIndex } from '../../semantic-index/query-dense.mjs';
import { normalizeLimit, openCortexDatabase, readChunkRow, sanitizeFtsQuery } from './cortex-db.mjs';

export async function searchCorpus(options = {}) {
  const rawQuery = requireString(options.query, 'query');
  const query = sanitizeFtsQuery(rawQuery);
  const limit = normalizeLimit(options.limit, 10);
  const family = typeof options.family === 'string' && options.family.trim() ? options.family.trim() : null;
  const useDense = options.use_dense === true;

  if (!query) {
    return { query: rawQuery, limit, ...(family ? { family } : {}), ...(useDense ? { alpha: Number(options.alpha ?? 0.5), use_dense: true } : { use_dense: false }), results: [] };
  }

  if (useDense) {
    return queryDenseIndex({
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
  }

  const database = openCortexDatabase(options.databasePath);

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