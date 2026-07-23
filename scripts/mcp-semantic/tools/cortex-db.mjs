/**
 * @module cortex-db
 * @description Shared SQLite access helpers for the Repo Cortex MCP tools.
 *
 * Centralizes database path resolution, safe read-only connection opening,
 * result-row normalization, and input sanitization utilities used by every
 * corpus tool in `scripts/mcp-semantic/tools/`.
 */
import { createClient } from '@libsql/client';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  defaultDatabasePath,
  repoRoot,
} from '../../../rag-index/init-schema.mjs';
import { requireString } from '../../agent-customization/mcp/mcp-utils.mjs';
import { sanitizeFtsQuery as sanitizeFtsQueryImpl } from '../../../rag-index/tokenizer.mjs';

/** Human-readable hint for operators when the semantic index database is missing. */
export const CORTEX_FIX_HINT = 'Run: node rag-index/build-index.mjs';

/**
 * Default embedded-replica sync interval in seconds.
 *
 * Tuned for RAG workloads where the corpus changes infrequently and a
 * one-minute lag is acceptable (Phase 1 verified the unit is seconds).
 */
const DEFAULT_SYNC_INTERVAL_SECONDS = 60;

/**
 * Cache of `@libsql/client` `Client` instances keyed by database URL.
 *
 * Each unique URL gets a single cached client so repeated calls to
 * {@link getTursoClient} reuse the same connection rather than opening a
 * new one on every invocation.
 */
const tursoClientCache = new Map();

/**
 * Resolve the corpus database URL from an explicit override, the
 * `TURSO_DATABASE_URL` environment variable, or the compiled-in default path.
 *
 * When `TURSO_DATABASE_URL` is set (e.g. `libsql://my-db.turso.io` or
 * `file:./rag-index/data/turso-replica.sqlite`), it is returned verbatim — it is already
 * a URL, not a filesystem path, so `path.resolve` must not be applied.
 *
 * When no Turso URL is configured, the compiled-in default path
 * (`defaultDatabasePath`) is used and resolved to an absolute path via
 * `path.resolve`.
 *
 * @param {string | undefined} databasePath - Explicit database path override.
 * @returns {string} Database URL or absolute resolved database path.
 */
export function resolveDatabasePath(databasePath) {
  const tursoUrl = process.env.TURSO_DATABASE_URL;
  if (tursoUrl) return tursoUrl;
  return path.resolve(databasePath ?? defaultDatabasePath);
}
/**
 * Get an async `@libsql/client` `Client` for the corpus database.
 *
 * This is the sole database entry point for all Repo Cortex MCP tools. It
 * creates (or reuses a cached) `Client` instance configured from
 * environment variables and/or an explicit database path override.
 *
 * **Connection caching:** A single `Client` is cached per database URL so
 * repeated calls do not open new connections.
 *
 * **Embedded replica support:** When `TURSO_SYNC_URL` is set, the client is
 * configured as an embedded replica — a local `file:` database that syncs
 * from a remote Turso instance. The `TURSO_SYNC_INTERVAL` env var controls
 * the sync interval in **seconds** (Phase 1 verified the unit is seconds,
 * not milliseconds). Read-your-writes is enabled by default.
 *
 * **Environment variables:**
 * - `TURSO_DATABASE_URL` — primary URL (checked via {@link resolveDatabasePath})
 * - `TURSO_AUTH_TOKEN` — JWT auth token for cloud access (optional)
 * - `TURSO_SYNC_URL` — sync URL for embedded replica (optional)
 * - `TURSO_SYNC_INTERVAL` — sync interval in seconds (optional, default 60)
 *
 * @param {string | undefined} [databasePath] - Explicit database URL or path
 *   override. When omitted, {@link resolveDatabasePath} determines the URL.
 * @returns {Promise<import('@libsql/client').Client>} Cached or newly created
 *   async libSQL client.
 */
export async function getTursoClient(databasePath) {
  const url = databasePath ?? resolveDatabasePath();

  const cachedClient = tursoClientCache.get(url);
  if (cachedClient) return cachedClient;

  // Convert plain filesystem paths to file: URLs for @libsql/client.
  // Only known URL schemes (libsql:, file:, http:, https:, ws:, wss:, :memory:) are passed as-is.
  // Windows drive letters (C:) look like schemes but are not valid URL schemes.
  const isUrl =
    /^(libsql|wss|ws|https|http|file):/i.test(url) || url === ':memory:';
  const resolvedUrl = isUrl ? url : pathToFileURL(url).href;

  const clientConfig = { url: resolvedUrl };
  clientConfig.authToken = process.env.TURSO_AUTH_TOKEN;
  clientConfig.syncUrl = process.env.TURSO_SYNC_URL;

  const syncIntervalEnv = process.env.TURSO_SYNC_INTERVAL;
  clientConfig.syncInterval =
    syncIntervalEnv != null && syncIntervalEnv !== ''
      ? Number(syncIntervalEnv)
      : DEFAULT_SYNC_INTERVAL_SECONDS;

  const client = createClient(clientConfig);
  tursoClientCache.set(url, client);
  return client;
}

/**
 * Close and evict a cached `@libsql/client` `Client` from the internal cache.
 *
 * Useful for tests and short-lived CLI scripts that need to release file
 * handles before deleting temp directories. In the long-lived MCP server
 * process, clients are cached for the process lifetime and never closed.
 *
 * @param {string | undefined} [databasePath] - Database path or URL that was
 *   passed to {@link getTursoClient}. Must match the key used for caching.
 * @returns {Promise<void>}
 */
export async function closeTursoClient(databasePath) {
  const url = databasePath ?? resolveDatabasePath();
  const cached = tursoClientCache.get(url);
  if (cached) {
    await cached.close();
    tursoClientCache.delete(url);
  }
}

/**
 * Return the number of cached libSQL clients.
 *
 * Used by diagnostics and benchmarks to verify that connection pooling is
 * working (a healthy long-lived process should have one client per database
 * URL rather than a new client per query).
 *
 * @returns {number} Size of the internal client cache.
 */
export function getCachedClientCount() {
  return tursoClientCache.size;
}

/**
 * Insert or replace a cached libSQL client for a specific database URL.
 *
 * Primarily intended for tests that need to inject an in-memory or
 * pre-configured client so tools do not open new file-backed connections.
 * The caller is responsible for closing any previously cached client and for
 * clearing the entry when the test finishes.
 *
 * @param {string} url - Database URL or path key to cache the client under.
 * @param {import('@libsql/client').Client | undefined} client - Client to cache, or `undefined` to evict.
 * @returns {void}
 */
export function setTursoClient(url, client) {
  requireString(url, 'url');
  if (client === undefined) {
    tursoClientCache.delete(url);
  } else {
    tursoClientCache.set(url, client);
  }
}

/**
 * Read a single chunk by ID using an async `@libsql/client` `Client`.
 *
 * This is the Turso/libSQL replacement for the synchronous
 * `db.prepare(sql).get()` + {@link readChunkRow} pattern used by the MCP
 * load-chunk tool. It queries the `chunks` table joined with `documents`
 * and returns a normalized chunk descriptor via {@link readChunkRow}.
 *
 * @param {import('@libsql/client').Client} client - Async libSQL client
 *   (obtained from {@link getTursoClient}).
 * @param {number} chunkId - Numeric chunk ID to read.
 * @returns {Promise<ReturnType<typeof readChunkRow>>} Normalized chunk
 *   descriptor.
 * @throws {Error} When the chunk ID does not exist in the database.
 */
export async function readChunk(client, chunkId) {
  const result = await client.execute({
    sql: `
      SELECT d.file_path, d.doc_family, c.doc_id, c.chunk_id, c.chunk_index,
        c.heading_path, c.body_text, c.char_start, c.char_end,
        c.parent_chunk_id, c.depth, c.context_header,
        c.symbol_name, c.signature_text, c.jsdoc_text, c.export_type,
        c.module_path, c.arch_layer, c.jsdoc_quality, c.jsdoc_word_count,
        c.cyclomatic_complexity, c.test_coverage, c.source_path_pattern,
        c.slice_id, c.step_number, c.phase, c.status
      FROM chunks c
      JOIN documents d ON d.doc_id = c.doc_id
      WHERE c.chunk_id = ?
    `,
    args: [chunkId],
  });

  const row = result.rows[0];
  if (!row) throw new Error(`Chunk not found: ${chunkId}`);

  return readChunkRow(row);
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
 * **Security:** Rejects any path that resolves outside `repoRoot` (i.e. the
 * relative result begins with `../` or is absolute). This prevents
 * path-traversal attacks where a malicious caller could request files outside
 * the repository (e.g. `../../etc/passwd` or an absolute system path such as
 * `/etc/shadow`).
 *
 * Absolute paths that resolve inside `repoRoot` are accepted and converted to
 * POSIX repo-relative paths, which is required on Windows where callers may
 * pass the absolute repository root (e.g. `C:\NeatapticTS`).
 *
 * @param {string} filePath - Caller-supplied file path.
 * @returns {string} POSIX-normalized repo-relative path (`'.'` for the repo root).
 * @throws {Error} When the path escapes the repository root.
 */
export function normalizeRepoPath(filePath) {
  const requestedPath = requireString(filePath, 'file_path');
  const absoluteRequested = path.resolve(repoRoot, requestedPath);
  const relativeToRepo = path
    .relative(repoRoot, absoluteRequested)
    .replaceAll('\\', '/');
  if (relativeToRepo.startsWith('../') || path.isAbsolute(relativeToRepo)) {
    throw new Error('file_path must stay inside the repository.');
  }

  return relativeToRepo === '' ? '.' : relativeToRepo;
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
 * Delegates to {@link sanitizeFtsQueryImpl} in `rag-index/tokenizer.mjs`,
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
 * context_header, symbol_name, signature_text, jsdoc_text, export_type, module_path),
 * v3 metadata enrichment columns (arch_layer, jsdoc_quality, jsdoc_word_count,
 * cyclomatic_complexity, test_coverage, source_path_pattern), and the A1
 * step-packet slice metadata columns (slice_id, step_number, phase, status).
 *
 * @param {Record<string, unknown>} row - Raw row from `chunks` joined with `documents`.
 * @returns {{ chunk_id: number, file_path: string, family: string, chunk_index: number, heading_path: string | null, text: string, char_start: number, char_end: number, depth: number, parent_chunk_id: number | null, context_header: string | null, symbol_name: string | null, signature_text: string | null, jsdoc_text: string | null, export_type: string | null, module_path: string | null, arch_layer: string | null, jsdoc_quality: string | null, jsdoc_word_count: number | null, cyclomatic_complexity: number | null, test_coverage: string | null, source_path_pattern: string | null, slice_id: string | null, step_number: number | null, phase: string | null, status: string | null }} Normalized chunk descriptor with v3 and A1 metadata.
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
    phase: row.phase ?? null,
    signature_text: row.signature_text ?? null,
    slice_id: row.slice_id ?? null,
    source_path_pattern: row.source_path_pattern ?? null,
    status: row.status ?? null,
    step_number: row.step_number != null ? Number(row.step_number) : null,
    symbol_name: row.symbol_name ?? null,
    test_coverage: row.test_coverage ?? null,
    text: row.body_text,
  };
}
