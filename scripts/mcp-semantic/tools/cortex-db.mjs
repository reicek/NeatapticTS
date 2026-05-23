import Database from 'better-sqlite3';
import { existsSync } from 'node:fs';
import path from 'node:path';

import { defaultDatabasePath, repoRoot } from '../../semantic-index/init-schema.mjs';
import { requireString } from '../../agent-customization/mcp/mcp-utils.mjs';

export const CORTEX_FIX_HINT = 'Run: node scripts/semantic-index/build-index.mjs';

export function resolveDatabasePath(databasePath) {
  return path.resolve(databasePath ?? process.env.CORTEX_DB_PATH ?? defaultDatabasePath);
}

export function openCortexDatabase(databasePath) {
  const resolvedDatabasePath = resolveDatabasePath(databasePath);
  if (!existsSync(resolvedDatabasePath)) {
    throw new Error(`Semantic index not found: ${resolvedDatabasePath}. ${CORTEX_FIX_HINT}`);
  }

  return new Database(resolvedDatabasePath, { readonly: true, fileMustExist: true });
}

export function normalizeLimit(value, fallback = 10) {
  const numericLimit = Number(value ?? fallback);
  if (!Number.isFinite(numericLimit)) return fallback;
  return Math.min(Math.max(Math.trunc(numericLimit), 1), 50);
}

export function normalizeRepoPath(filePath) {
  const requestedPath = requireString(filePath, 'file_path').replaceAll('\\', '/');
  const normalizedPath = path.posix.normalize(requestedPath);
  if (normalizedPath.startsWith('../') || path.isAbsolute(normalizedPath)) {
    throw new Error('file_path must stay inside the repository.');
  }

  return normalizedPath;
}

export function toAbsoluteRepoPath(filePath) {
  return path.join(repoRoot, normalizeRepoPath(filePath));
}

export function asIsoTimestamp(value) {
  const numericValue = Number(value);
  return Number.isFinite(numericValue) && numericValue > 0
    ? new Date(numericValue).toISOString()
    : null;
}

export function readChunkRow(row) {
  return {
    chunk_id: Number(row.chunk_id),
    file_path: row.file_path,
    family: row.doc_family,
    chunk_index: Number(row.chunk_index),
    heading_path: row.heading_path ?? null,
    text: row.body_text,
    char_start: Number(row.char_start),
    char_end: Number(row.char_end),
  };
}