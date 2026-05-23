import { requireString } from '../../agent-customization/mcp/mcp-utils.mjs';
import { normalizeLimit, openCortexDatabase, readChunkRow } from './cortex-db.mjs';

export async function searchCorpus(options = {}) {
  const query = requireString(options.query, 'query');
  const limit = normalizeLimit(options.limit, 10);
  const family = typeof options.family === 'string' && options.family.trim() ? options.family.trim() : null;
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
      results: rows.map((row) => ({ ...readChunkRow(row), score: Number(row.score) })),
    };
  } finally {
    database.close();
  }
}