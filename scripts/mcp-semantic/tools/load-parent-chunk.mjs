/**
 * @module load-parent-chunk
 * @description Parent-chunk loader tool for the Repo Cortex MCP server.
 *
 * Given a depth-1 sub-chunk ID, resolves and loads its parent (depth-0) chunk.
 * Useful for agents that need to navigate up the semantic chunk hierarchy to
 * understand the broader context around a sub-chunk (e.g., a method within a class).
 */
import { getTursoClient, readChunkRow } from './cortex-db.mjs';

/**
 * Load the parent chunk for a given depth-1 sub-chunk.
 *
 * Looks up the specified chunk, reads its `parent_chunk_id`, then loads
 * and returns the parent chunk descriptor with full v2 semantic metadata.
 *
 * @param {object} [options={}] - Tool options.
 * @param {number} options.chunk_id - Numeric chunk identifier of the sub-chunk.
 * @param {string} [options.databasePath] - Override corpus database path.
 * @param {import('@libsql/client').Client} [options.client] - Pre-existing libsql client.
 * @returns {Promise<{ parent_chunk: object }>} The parent chunk descriptor with v2 metadata.
 * @throws {Error} When `chunk_id` is not a positive integer, the chunk is not found,
 *   the chunk has no parent (depth-0), or the parent chunk is missing.
 */
export async function loadParentChunk(options = {}) {
  const chunkId = Number(options.chunk_id);
  if (!Number.isInteger(chunkId) || chunkId < 1) {
    throw new Error('chunk_id must be a positive integer.');
  }

  const client = options.client ?? (await getTursoClient(options.databasePath));

  const subResult = await client.execute({
    sql: `
      SELECT chunk_id, parent_chunk_id, depth
      FROM chunks
      WHERE chunk_id = ?
    `,
    args: [chunkId],
  });

  const subRow = subResult.rows[0];
  if (!subRow) {
    throw new Error(`Chunk not found: ${chunkId}`);
  }

  if (subRow.parent_chunk_id == null || Number(subRow.depth) === 0) {
    throw new Error(
      `Chunk ${chunkId} is a top-level chunk (depth 0) with no parent.`,
    );
  }

  const parentId = Number(subRow.parent_chunk_id);

  const parentResult = await client.execute({
    sql: `
      SELECT d.file_path, d.doc_family, c.chunk_id, c.chunk_index, c.heading_path,
        c.body_text, c.char_start, c.char_end,
        c.parent_chunk_id, c.depth, c.context_header,
        c.symbol_name, c.signature_text, c.jsdoc_text, c.export_type, c.module_path
      FROM chunks c
      JOIN documents d ON d.doc_id = c.doc_id
      WHERE c.chunk_id = ?
    `,
    args: [parentId],
  });

  const parentRow = parentResult.rows[0];
  if (!parentRow) {
    throw new Error(`Parent chunk not found: ${parentId}`);
  }

  return { parent_chunk: { ...readChunkRow(parentRow), chunk_id: parentId } };
}
