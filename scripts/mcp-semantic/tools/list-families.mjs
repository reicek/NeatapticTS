import { openCortexDatabase } from './cortex-db.mjs';

export async function listFamilies(options = {}) {
  const database = openCortexDatabase(options.databasePath);

  try {
    const rows = database.prepare(`
      SELECT d.doc_family AS family, COUNT(DISTINCT d.doc_id) AS documents, COUNT(c.chunk_id) AS chunks
      FROM documents d
      LEFT JOIN chunks c ON c.doc_id = d.doc_id
      GROUP BY d.doc_family
      ORDER BY d.doc_family
    `).all();

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