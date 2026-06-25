/**
 * @module freshness-check
 * @description Corpus freshness probe tool for the Repo Cortex MCP server.
 *
 * Compares the indexed document metadata (mtime, file size, sha256) against
 * current filesystem state to identify stale documents that need re-indexing.
 */
import path from 'node:path';

import {
  getFreshnessProof,
  isFreshDocument,
} from '../../semantic-index/freshness.mjs';
import { repoRoot } from '../../semantic-index/init-schema.mjs';
import { normalizeRepoPath, getTursoClient } from './cortex-db.mjs';

/**
 * Check corpus document freshness against current filesystem metadata.
 *
 * When `file_path` is provided, checks only that document; otherwise checks
 * all indexed documents and returns a summary of stale paths.
 *
 * @param {object} [options={}] - Tool options.
 * @param {string} [options.file_path] - Repo-relative file path to check.
 * @param {object} [options.freshnessProof] - Pre-computed freshness proof (for testing).
 * @param {string} [options.databasePath] - Override corpus database path.
 * @returns {Promise<{ fresh: boolean, stale: string[], freshness: object, documents: Array<object> }>} Freshness report.
 * @throws {Error} When a specific `file_path` is not found in the index.
 */
export async function freshnessCheck(options = {}) {
  let filePath =
    typeof options.file_path === 'string' && options.file_path.trim()
      ? normalizeRepoPath(options.file_path)
      : null;
  if (filePath === '.' || filePath === '') {
    filePath = null;
  }
  const client = options.client ?? (await getTursoClient(options.databasePath));

  const result = filePath
    ? await client.execute({
        sql: 'SELECT file_path, mtime_ms, file_size, sha256, indexed_at FROM documents WHERE file_path = ?',
        args: [filePath],
      })
    : await client.execute({
        sql: 'SELECT file_path, mtime_ms, file_size, sha256, indexed_at FROM documents ORDER BY file_path',
      });

  const rows = result.rows;
  if (filePath && rows.length === 0)
    throw new Error(`Document not found: ${filePath}`);

  const checks = await Promise.all(
    rows.map(async (row) => checkDocument(row, options.freshnessProof)),
  );
  const stale = checks
    .filter((check) => !check.fresh)
    .map((check) => check.file_path);
  const timestamp = Date.now();
  const freshness = {
    timestamp,
    stale: stale.length > 0,
    last_update_source: 'filesystem',
    last_sync: null,
    sync_lag_ms: null,
  };
  return {
    fresh: stale.length === 0,
    stale,
    freshness,
    documents: checks.map((check) => ({
      ...check,
      freshness: {
        timestamp,
        stale: !check.fresh,
        last_update_source: 'filesystem',
        last_sync: null,
        sync_lag_ms: null,
      },
    })),
  };
}

/**
 * Compare one indexed document row against its current filesystem proof.
 *
 * @param {Record<string, unknown>} row - Raw document row from the `documents` table.
 * @param {object | undefined} suppliedProof - Optional pre-computed freshness proof.
 * @returns {Promise<{ file_path: string, fresh: boolean, indexed: object, current: object }>} Per-document freshness result.
 */
async function checkDocument(row, suppliedProof) {
  const proof =
    suppliedProof && typeof suppliedProof === 'object'
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
