/**
 * @module list-families
 * @description Document family listing tool for the Repo Cortex MCP server.
 *
 * Returns all indexed document families with their document and chunk counts,
 * useful for scoping corpus searches to a specific project area.
 */
import { openCortexDatabase } from './cortex-db.mjs';

/**
 * List all indexed document families with document and chunk counts.
 *
 * @param {object} [options={}] - Tool options.
 * @param {string} [options.databasePath] - Override corpus database path.
 * @returns {Promise<{ families: Array<{ family: string, documents: number, chunks: number }> }>} Family list.
 */
export async function listFamilies(options = {}) {
  const database = openCortexDatabase(options.databasePath);

  try {
    const rows = database
      .prepare(
        `
      SELECT d.doc_family AS family, COUNT(DISTINCT d.doc_id) AS documents, COUNT(c.chunk_id) AS chunks
      FROM documents d
      LEFT JOIN chunks c ON c.doc_id = d.doc_id
      GROUP BY d.doc_family
      ORDER BY d.doc_family
    `,
      )
      .all();

    return {
      families: rows.map((row) => ({
        family: row.family,
        documents: Number(row.documents),
        chunks: Number(row.chunks),
      })),
    };
  } finally {
    database.close();
  }
}
