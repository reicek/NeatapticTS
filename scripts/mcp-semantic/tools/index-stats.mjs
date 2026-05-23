import { asIsoTimestamp, openCortexDatabase } from './cortex-db.mjs';

export async function indexStats(options = {}) {
  const database = openCortexDatabase(options.databasePath);

  try {
    const documentCount = database.prepare('SELECT COUNT(*) AS count FROM documents').get().count;
    const chunkCount = database.prepare('SELECT COUNT(*) AS count FROM chunks').get().count;
    const familyCount = database.prepare('SELECT COUNT(DISTINCT doc_family) AS count FROM documents').get().count;
    const lastIndexedAt = database.prepare('SELECT MAX(indexed_at) AS value FROM documents').get().value;

    return {
      total_documents: Number(documentCount),
      total_chunks: Number(chunkCount),
      total_families: Number(familyCount),
      last_build_timestamp: asIsoTimestamp(lastIndexedAt),
    };
  } finally {
    database.close();
  }
}