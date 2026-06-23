/**
 * @module load-document
 * @description Full-document loader tool for the Repo Cortex MCP server.
 *
 * Loads all ordered chunks for one indexed repository path, allowing agents
 * to read the full indexed content of a source file or document.
 * Returns v2 semantic chunking columns and hierarchy summary metadata.
 */
import {
  normalizeRepoPath,
  getTursoClient,
  readChunkRow,
} from './cortex-db.mjs';

/**
 * Load all ordered chunks for one indexed repository path.
 *
 * Validates the path with {@link normalizeRepoPath} to prevent path traversal
 * before querying the database. Returns a hierarchy summary with depth-0 and
 * depth-1 chunk counts for v2 semantic chunking awareness.
 *
 * @param {object} [options={}] - Tool options.
 * @param {string} options.file_path - Repo-relative file path to load.
 * @param {string} [options.databasePath] - Override corpus database path.
 * @param {import('@libsql/client').Client} [options.client] - Pre-existing libsql client.
 * @returns {Promise<{ file_path: string, chunks: Array<object>, hierarchy: { depth_0_count: number, depth_1_count: number, has_sub_chunks: boolean } }>} Ordered chunks with hierarchy info.
 * @throws {Error} When `file_path` escapes the repository root or is not found in the index.
 */
export async function loadDocument(options = {}) {
  const filePath = normalizeRepoPath(options.file_path);
  const client = options.client ?? (await getTursoClient(options.databasePath));

  const result = await client.execute({
    sql: `
      SELECT d.file_path, d.doc_family, c.chunk_id, c.chunk_index, c.heading_path,
        c.body_text, c.char_start, c.char_end,
        c.parent_chunk_id, c.depth, c.context_header,
        c.symbol_name, c.signature_text, c.jsdoc_text, c.export_type, c.module_path
      FROM chunks c
      JOIN documents d ON d.doc_id = c.doc_id
      WHERE d.file_path = ?
      ORDER BY c.chunk_index
    `,
    args: [filePath],
  });

  const rows = result.rows;
  if (rows.length === 0) throw new Error(`Document not found: ${filePath}`);

  const chunks = rows.map(readChunkRow);
  const depth0 = chunks.filter((c) => c.depth === 0).length;
  const depth1 = chunks.filter((c) => c.depth === 1).length;

  return {
    file_path: filePath,
    chunks,
    hierarchy: {
      depth_0_count: depth0,
      depth_1_count: depth1,
      has_sub_chunks: depth1 > 0,
    },
  };
}
