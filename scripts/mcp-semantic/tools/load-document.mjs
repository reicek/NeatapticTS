import { normalizeRepoPath, openCortexDatabase, readChunkRow } from './cortex-db.mjs';

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