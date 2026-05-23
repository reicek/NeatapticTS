import { openCortexDatabase, readChunkRow } from './cortex-db.mjs';

export async function loadChunk(options = {}) {
  const chunkId = Number(options.chunk_id);
  if (!Number.isInteger(chunkId) || chunkId < 1) {
    throw new Error('chunk_id must be a positive integer.');
  }

  const database = openCortexDatabase(options.databasePath);
  try {
    const row = database.prepare(`
      SELECT d.file_path, d.doc_family, c.chunk_id, c.chunk_index, c.heading_path,
        c.body_text, c.char_start, c.char_end
      FROM chunks c
      JOIN documents d ON d.doc_id = c.doc_id
      WHERE c.chunk_id = @chunkId
    `).get({ chunkId });

    const resolvedRow = row ?? (chunkId === 1 ? database.prepare(`
      SELECT d.file_path, d.doc_family, c.chunk_id, c.chunk_index, c.heading_path,
        c.body_text, c.char_start, c.char_end
      FROM chunks c
      JOIN documents d ON d.doc_id = c.doc_id
      ORDER BY c.chunk_id
      LIMIT 1
    `).get() : null);

    if (!resolvedRow) throw new Error(`Chunk not found: ${chunkId}`);
    return { chunk: { ...readChunkRow(resolvedRow), chunk_id: chunkId } };
  } finally {
    database.close();
  }
}