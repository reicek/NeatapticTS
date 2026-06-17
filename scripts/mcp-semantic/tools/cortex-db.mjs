/**
 * @module cortex-db
 * @description Shared SQLite access helpers for the Repo Cortex MCP tools.
 *
 * Centralizes database path resolution, safe read-only connection opening,
 * result-row normalization, and input sanitization utilities used by every
 * corpus tool in `scripts/mcp-semantic/tools/`.
 */
import Database from 'better-sqlite3';
import { existsSync } from 'node:fs';
import path from 'node:path';

import {
  defaultDatabasePath,
  repoRoot,
} from '../../semantic-index/init-schema.mjs';
import { requireString } from '../../agent-customization/mcp/mcp-utils.mjs';
import { sanitizeFtsQuery as sanitizeFtsQueryImpl } from '../../semantic-index/tokenizer.mjs';

/** Human-readable hint for operators when the semantic index database is missing. */
export const CORTEX_FIX_HINT =
  'Run: node scripts/semantic-index/build-index.mjs';

/**
 * Resolve the corpus SQLite database path from an explicit override, the
 * `CORTEX_DB_PATH` environment variable, or the compiled-in default path.
 *
 * @param {string | undefined} databasePath - Explicit database path override.
 * @returns {string} Absolute resolved database path.
 */
export function resolveDatabasePath(databasePath) {
  return path.resolve(
    databasePath ?? process.env.CORTEX_DB_PATH ?? defaultDatabasePath,
  );
}

/**
 * Open the corpus SQLite database in read-only mode.
 *
 * Throws a descriptive error with {@link CORTEX_FIX_HINT} when the database
 * file does not exist, so operators know how to rebuild the index.
 *
 * @param {string | undefined} databasePath - Explicit database path override.
 * @returns {import('better-sqlite3').Database} Open read-only database connection.
 * @throws {Error} When the database file does not exist.
 */
export function openCortexDatabase(databasePath) {
  const resolvedDatabasePath = resolveDatabasePath(databasePath);
  if (!existsSync(resolvedDatabasePath)) {
    throw new Error(
      `Semantic index not found: ${resolvedDatabasePath}. ${CORTEX_FIX_HINT}`,
    );
  }

  return new Database(resolvedDatabasePath, {
    readonly: true,
    fileMustExist: true,
  });
}

/**
 * Clamp and coerce a raw result-limit value to an integer in [1, 50].
 *
 * @param {unknown} value - Raw limit value from the tool input.
 * @param {number} [fallback=10] - Default when `value` is absent or not a finite number.
 * @returns {number} Clamped integer limit.
 */
export function normalizeLimit(value, fallback = 10) {
  const numericLimit = Number(value ?? fallback);
  if (!Number.isFinite(numericLimit)) return fallback;
  return Math.min(Math.max(Math.trunc(numericLimit), 1), 50);
}

/**
 * Normalize and validate a file path so it stays inside the repository root.
 *
 * **Security:** Rejects any path that, after POSIX normalization, begins with
 * `../` or is absolute. This prevents path-traversal attacks where a malicious
 * caller could request files outside the repository (e.g. `../../etc/passwd`
 * or an absolute system path such as `/etc/shadow`).
 *
 * Backslashes are replaced with forward slashes before normalization so
 * Windows path separators cannot bypass the traversal check.
 *
 * @param {string} filePath - Caller-supplied file path.
 * @returns {string} POSIX-normalized repo-relative path.
 * @throws {Error} When the path escapes the repository root.
 */
export function normalizeRepoPath(filePath) {
  const requestedPath = requireString(filePath, 'file_path').replaceAll(
    '\\',
    '/',
  );
  const normalizedPath = path.posix.normalize(requestedPath);
  if (normalizedPath.startsWith('../') || path.isAbsolute(normalizedPath)) {
    throw new Error('file_path must stay inside the repository.');
  }

  return normalizedPath;
}

/**
 * Convert a validated repo-relative path to an absolute filesystem path.
 *
 * Always validates via {@link normalizeRepoPath} before joining with the
 * repository root, so path-traversal is rejected before any filesystem access.
 *
 * @param {string} filePath - Repo-relative file path.
 * @returns {string} Absolute filesystem path.
 * @throws {Error} When `filePath` escapes the repository root.
 */
export function toAbsoluteRepoPath(filePath) {
  return path.join(repoRoot, normalizeRepoPath(filePath));
}

/**
 * Convert a numeric Unix-millisecond timestamp to an ISO 8601 string.
 *
 * @param {unknown} value - Raw timestamp value (e.g. from a SQLite integer column).
 * @returns {string | null} ISO 8601 string, or `null` when the value is missing or non-numeric.
 */
export function asIsoTimestamp(value) {
  const numericValue = Number(value);
  return Number.isFinite(numericValue) && numericValue > 0
    ? new Date(numericValue).toISOString()
    : null;
}

/**
 * Sanitizes user-supplied text for safe use as an FTS5 MATCH query.
 *
 * Delegates to {@link sanitizeFtsQueryImpl} in `scripts/semantic-index/tokenizer.mjs`,
 * which preserves code identifiers (dotted, camelCase, snake_case, file-extension
 * hints) as quoted phrases and applies prefix wildcards to plain words.
 *
 * @param {string} raw - User-supplied or agent-supplied query string.
 * @returns {string} FTS5-safe query string, or empty string if no terms remain.
 */
export function sanitizeFtsQuery(raw) {
  return sanitizeFtsQueryImpl(raw);
}

/**
 * Normalize a raw SQLite chunk row into a typed chunk descriptor.
 *
 * Includes v2 semantic chunking columns (depth, parent_chunk_id,
 * context_header, symbol_name, signature_text, jsdoc_text, export_type, module_path)
 * and v3 metadata enrichment columns (arch_layer, jsdoc_quality, jsdoc_word_count,
 * cyclomatic_complexity, test_coverage, source_path_pattern).
 *
 * @param {Record<string, unknown>} row - Raw row from `chunks` joined with `documents`.
 * @returns {{ chunk_id: number, file_path: string, family: string, chunk_index: number, heading_path: string | null, text: string, char_start: number, char_end: number, depth: number, parent_chunk_id: number | null, context_header: string | null, symbol_name: string | null, signature_text: string | null, jsdoc_text: string | null, export_type: string | null, module_path: string | null, arch_layer: string | null, jsdoc_quality: string | null, jsdoc_word_count: number | null, cyclomatic_complexity: number | null, test_coverage: string | null, source_path_pattern: string | null }} Normalized chunk descriptor with v3 metadata.
 */
export function readChunkRow(row) {
  return {
    arch_layer: row.arch_layer ?? null,
    char_end: Number(row.char_end),
    char_start: Number(row.char_start),
    chunk_id: Number(row.chunk_id),
    chunk_index: Number(row.chunk_index),
    context_header: row.context_header ?? null,
    cyclomatic_complexity:
      row.cyclomatic_complexity != null
        ? Number(row.cyclomatic_complexity)
        : null,
    depth: Number(row.depth ?? 0),
    export_type: row.export_type ?? null,
    family: row.doc_family,
    file_path: row.file_path,
    heading_path: row.heading_path ?? null,
    jsdoc_quality: row.jsdoc_quality ?? null,
    jsdoc_text: row.jsdoc_text ?? null,
    jsdoc_word_count:
      row.jsdoc_word_count != null ? Number(row.jsdoc_word_count) : null,
    module_path: row.module_path ?? null,
    parent_chunk_id:
      row.parent_chunk_id != null ? Number(row.parent_chunk_id) : null,
    signature_text: row.signature_text ?? null,
    source_path_pattern: row.source_path_pattern ?? null,
    symbol_name: row.symbol_name ?? null,
    test_coverage: row.test_coverage ?? null,
    text: row.body_text,
  };
}
