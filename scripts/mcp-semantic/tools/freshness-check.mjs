import path from 'node:path';

import { getFreshnessProof, isFreshDocument } from '../../semantic-index/freshness.mjs';
import { repoRoot } from '../../semantic-index/init-schema.mjs';
import { normalizeRepoPath, openCortexDatabase } from './cortex-db.mjs';

export async function freshnessCheck(options = {}) {
  const filePath = typeof options.file_path === 'string' && options.file_path.trim()
    ? normalizeRepoPath(options.file_path)
    : null;
  const database = openCortexDatabase(options.databasePath);

  try {
    const rows = filePath
      ? database.prepare('SELECT file_path, mtime_ms, file_size, sha256, indexed_at FROM documents WHERE file_path = @filePath').all({ filePath })
      : database.prepare('SELECT file_path, mtime_ms, file_size, sha256, indexed_at FROM documents ORDER BY file_path').all();

    if (filePath && rows.length === 0) throw new Error(`Document not found: ${filePath}`);

    const checks = await Promise.all(rows.map(async (row) => checkDocument(row, options.freshnessProof)));
    const stale = checks.filter((check) => !check.fresh).map((check) => check.file_path);
    return { fresh: stale.length === 0, stale, documents: checks };
  } finally {
    database.close();
  }
}

async function checkDocument(row, suppliedProof) {
  const proof = suppliedProof && typeof suppliedProof === 'object'
    ? suppliedProof
    : await getFreshnessProof(path.join(repoRoot, row.file_path));

  return {
    file_path: row.file_path,
    fresh: isFreshDocument(row, proof),
    indexed: {
      mtime_ms: Number(row.mtime_ms),
      file_size: Number(row.file_size),
      sha256: row.sha256,
    },
    current: proof,
  };
}