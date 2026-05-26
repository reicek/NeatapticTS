/**
 * @module load-document
 * @description Full-document loader tool for the Repo Cortex MCP server.
 *
 * Loads all ordered chunks for one indexed repository path, allowing agents
 * to read the full indexed content of a source file or document.
 */
import { normalizeRepoPath, openCortexDatabase, readChunkRow } from './cortex-db.mjs';

/**
 * Load all ordered chunks for one indexed repository path.
 *
 * Validates the path with {@link normalizeRepoPath} to prevent path traversal
 * before querying the database.
 *
 * @param {object} [options={}] - Tool options.
 * @param {string} options.file_path - Repo-relative file path to load.
 * @param {string} [options.databasePath] - Override corpus database path.
 * @returns {Promise<{ file_path: string, chunks: Array<object> }>} Ordered chunks for the document.
 * @throws {Error} When `file_path` escapes the repository root or is not found in the index.
 */
export async function loadDocument(options = {}) {
  const filePath = normalizeRepoPath(options.file_path);
  const database = openCortexDatabase(options.databasePath);

  try {
    const rows = database.prepare(`
      SELECT d.file_path, d.doc_family, c.chunk_id, c.chunk_index, c.heading_path,
        c.body_text, c.char_start, c.char_end
      FROM chunks c
      JOIN documents d ON d.doc_id = c.doc_id
      WHERE d.file_path = @filePath
      ORDER BY c.chunk_index
    `).all({ filePath });

    if (rows.length === 0) throw new Error(`Document not found: ${filePath}`);
    return { file_path: filePath, chunks: rows.map(readChunkRow) };
  } finally {
    database.close();
  }
}