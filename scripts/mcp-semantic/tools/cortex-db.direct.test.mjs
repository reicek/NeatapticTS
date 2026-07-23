/**
 * @module cortex-db.direct.test
 * @description Direct-import coverage tests for cortex-db.mjs.
 *
 * Runs in the mcp-semantic-mjs project so Jest can instrument the native ESM
 * source file via the V8 coverage provider.
 */
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createClient } from '@libsql/client';
import {
  CORTEX_FIX_HINT,
  asIsoTimestamp,
  closeTursoClient,
  getCachedClientCount,
  getTursoClient,
  normalizeLimit,
  normalizeRepoPath,
  readChunk,
  readChunkRow,
  resolveDatabasePath,
  sanitizeFtsQuery,
  setTursoClient,
  toAbsoluteRepoPath,
} from './cortex-db.mjs';
import { defaultDatabasePath } from '../../../rag-index/init-schema.mjs';

const REPO_ROOT = path.resolve();
const CREATED_URLS = new Set();
const TEMP_DIRS = [];

function trackTempDir() {
  const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'cortex-db-direct-'));
  TEMP_DIRS.push(tempDir);
  return tempDir;
}

async function trackedGetTursoClient(databasePath) {
  const client = await getTursoClient(databasePath);
  const url = databasePath ?? resolveDatabasePath();
  CREATED_URLS.add(url);
  return client;
}

async function seedSharedDatabase(client) {
  await client.executeMultiple(
    fs.readFileSync(
      path.resolve(REPO_ROOT, './rag-index/schema-turso.sql'),
      'utf8',
    ),
  );
  await client.executeMultiple(`
    INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
      VALUES (1, 'src/network.ts', 'ts-source', 1, 100, 'sha', 1);
    INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
      VALUES (2, 'src/utils.ts', 'ts-source', 1, 100, 'sha', 1);
    INSERT INTO chunks (
      chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end,
      depth, parent_chunk_id, context_header, symbol_name, signature_text, jsdoc_text,
      export_type, module_path, arch_layer, jsdoc_quality, jsdoc_word_count,
      cyclomatic_complexity, test_coverage, source_path_pattern,
      slice_id, step_number, phase, status
    ) VALUES (
      1, 1, 0, 'activate', 'fixture body', 0, 12,
      1, NULL, 'ctx', 'sym', 'sig()', 'jsdoc',
      'function', 'src/network.ts', 'layer', 'good', 7,
      3, 'full', 'pattern',
      'A1-green', 1, 'A', 'done'
    );
    INSERT INTO chunks (
      chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end,
      depth, parent_chunk_id, context_header, symbol_name, signature_text, jsdoc_text,
      export_type, module_path, arch_layer, jsdoc_quality, jsdoc_word_count,
      cyclomatic_complexity, test_coverage, source_path_pattern,
      slice_id, step_number, phase, status
    ) VALUES (
      2, 2, 1, NULL, 'minimal body', 4, 14,
      0, NULL, NULL, NULL, NULL, NULL,
      NULL, NULL, NULL, NULL, NULL,
      NULL, NULL, NULL,
      NULL, NULL, NULL, NULL
    );
  `);
}

describe('cortex-db.mjs direct import coverage', () => {
  let sharedClient;

  beforeAll(async () => {
    sharedClient = await trackedGetTursoClient(':memory:');
    await seedSharedDatabase(sharedClient);
  });

  afterAll(async () => {
    for (const url of CREATED_URLS) {
      await closeTursoClient(url === ':memory:' ? ':memory:' : url);
    }
    CREATED_URLS.clear();
    for (const tempDir of TEMP_DIRS) {
      try {
        fs.rmSync(tempDir, { recursive: true, force: true });
      } catch {
        // Ignore Windows file-handle cleanup races for temp directories.
      }
    }
  });

  it('exposes the fix hint constant', () => {
    expect(CORTEX_FIX_HINT).toMatch(/Run:/);
  });

  it('resolves to TURSO_DATABASE_URL when set', () => {
    const original = process.env.TURSO_DATABASE_URL;
    process.env.TURSO_DATABASE_URL = 'libsql://env.turso.io';
    try {
      expect(resolveDatabasePath()).toBe('libsql://env.turso.io');
    } finally {
      if (original === undefined) {
        delete process.env.TURSO_DATABASE_URL;
      } else {
        process.env.TURSO_DATABASE_URL = original;
      }
    }
  });

  it('resolves an explicit database path when no env is set', () => {
    delete process.env.TURSO_DATABASE_URL;
    expect(resolveDatabasePath('/tmp/explicit.sqlite')).toBe(
      path.resolve('/tmp/explicit.sqlite'),
    );
  });

  it('falls back to the compiled-in default database path', () => {
    delete process.env.TURSO_DATABASE_URL;
    expect(resolveDatabasePath()).toBe(path.resolve(defaultDatabasePath));
  });

  it('reuses a cached client for the same URL', async () => {
    const first = await trackedGetTursoClient(':memory:');
    const second = await trackedGetTursoClient(':memory:');
    expect(second).toBe(first);
  });

  it('creates a client for a plain filesystem path', async () => {
    const tempDir = trackTempDir();
    const dbPath = path.join(tempDir, 'plain.sqlite');
    const client = await trackedGetTursoClient(dbPath);
    await client.execute('SELECT 1');
    expect(getCachedClientCount()).toBeGreaterThanOrEqual(2);
  });

  it('honors environment variables when building a client', async () => {
    const originalUrl = process.env.TURSO_DATABASE_URL;
    const originalToken = process.env.TURSO_AUTH_TOKEN;
    const originalInterval = process.env.TURSO_SYNC_INTERVAL;
    const tempDir = trackTempDir();
    const envDbPath = path.join(tempDir, 'env.sqlite');

    process.env.TURSO_DATABASE_URL = envDbPath;
    process.env.TURSO_AUTH_TOKEN = 'test-token';
    process.env.TURSO_SYNC_INTERVAL = '90';

    try {
      const client = await trackedGetTursoClient();
      await client.execute('SELECT 1');
      expect(getCachedClientCount()).toBeGreaterThanOrEqual(1);
    } finally {
      await closeTursoClient();
      CREATED_URLS.delete(envDbPath);
      if (originalUrl === undefined) {
        delete process.env.TURSO_DATABASE_URL;
      } else {
        process.env.TURSO_DATABASE_URL = originalUrl;
      }
      if (originalToken === undefined) {
        delete process.env.TURSO_AUTH_TOKEN;
      } else {
        process.env.TURSO_AUTH_TOKEN = originalToken;
      }
      if (originalInterval === undefined) {
        delete process.env.TURSO_SYNC_INTERVAL;
      } else {
        process.env.TURSO_SYNC_INTERVAL = originalInterval;
      }
    }
  });

  it('closes a cached client and removes it from the cache', async () => {
    const tempDir = trackTempDir();
    const dbPath = path.join(tempDir, 'close.sqlite');
    await trackedGetTursoClient(dbPath);
    const beforeClose = getCachedClientCount();
    await closeTursoClient(dbPath);
    expect(getCachedClientCount()).toBe(beforeClose - 1);
  });

  it('does not throw when closing an uncached database path', async () => {
    await expect(
      closeTursoClient('/nonexistent/uncached.sqlite'),
    ).resolves.toBeUndefined();
  });

  it('caches and evicts an injected client', () => {
    const before = getCachedClientCount();
    setTursoClient('injected-key', sharedClient);
    setTursoClient('injected-key', undefined);
    expect(getCachedClientCount()).toBe(before);
  });

  it('requires a string key when caching a client', () => {
    expect(() => setTursoClient(123, sharedClient)).toThrow(/url/);
  });

  it('reads a chunk and exposes the new A1 slice columns', async () => {
    const chunk = await readChunk(sharedClient, 1);
    expect(chunk.slice_id).toBe('A1-green');
  });

  it('throws when reading a chunk that does not exist', async () => {
    await expect(readChunk(sharedClient, 9999)).rejects.toThrow(
      /Chunk not found/,
    );
  });

  it('normalizes a row with all optional fields populated', () => {
    const populated = {
      arch_layer: 'layer',
      char_end: 12,
      char_start: 0,
      chunk_id: 1,
      chunk_index: 0,
      context_header: 'ctx',
      cyclomatic_complexity: 3,
      depth: 1,
      doc_family: 'ts-source',
      export_type: 'function',
      file_path: 'src/network.ts',
      heading_path: 'activate',
      jsdoc_quality: 'good',
      jsdoc_text: 'jsdoc',
      jsdoc_word_count: 7,
      module_path: 'src/network.ts',
      parent_chunk_id: 2,
      phase: 'A',
      signature_text: 'sig()',
      slice_id: 'A1-green',
      source_path_pattern: 'pattern',
      status: 'done',
      step_number: 1,
      symbol_name: 'sym',
      test_coverage: 'full',
      body_text: 'fixture body',
    };
    const row = readChunkRow(populated);
    expect(row.parent_chunk_id).toBe(2);
  });

  it('normalizes a row with all optional fields null', () => {
    const sparse = {
      char_end: 14,
      char_start: 4,
      chunk_id: 2,
      chunk_index: 1,
      doc_family: 'ts-source',
      file_path: 'src/utils.ts',
      body_text: 'minimal body',
    };
    const row = readChunkRow(sparse);
    expect(row.parent_chunk_id).toBeNull();
  });

  it('clamps a limit above the upper bound', () => {
    expect(normalizeLimit(100, 10)).toBe(50);
  });

  it('clamps a limit below the lower bound', () => {
    expect(normalizeLimit(0, 10)).toBe(1);
  });

  it('falls back when the limit is not finite', () => {
    expect(normalizeLimit(Number.NaN, 10)).toBe(10);
  });

  it('falls back when no limit value is provided', () => {
    expect(normalizeLimit(undefined, 10)).toBe(10);
  });

  it('uses the default fallback when no arguments are provided', () => {
    expect(normalizeLimit()).toBe(10);
  });

  it('returns a repo-relative path for a relative input', () => {
    expect(normalizeRepoPath('src/network.ts')).toBe('src/network.ts');
  });

  it('returns a dot for the repository root path', () => {
    expect(normalizeRepoPath('.')).toBe('.');
  });

  it('accepts an absolute path inside the repository', () => {
    expect(normalizeRepoPath(path.resolve(REPO_ROOT, 'src/network.ts'))).toBe(
      'src/network.ts',
    );
  });

  it('rejects a path that escapes the repository', () => {
    expect(() => normalizeRepoPath('../outside')).toThrow(
      /file_path must stay inside/,
    );
  });

  it('rejects an absolute path outside the repository', () => {
    expect(() => normalizeRepoPath(os.tmpdir())).toThrow(
      /file_path must stay inside/,
    );
  });

  it('converts a repo-relative path to an absolute filesystem path', () => {
    expect(toAbsoluteRepoPath('src/network.ts')).toBe(
      path.join(REPO_ROOT, 'src/network.ts'),
    );
  });

  it('converts a numeric timestamp to an ISO string', () => {
    expect(asIsoTimestamp(1_000_000)).toBe(new Date(1_000_000).toISOString());
  });

  it('returns null for a non-numeric timestamp', () => {
    expect(asIsoTimestamp('not a number')).toBeNull();
  });

  it('returns null for a zero timestamp', () => {
    expect(asIsoTimestamp(0)).toBeNull();
  });

  it('sanitizes an FTS query by delegating to the tokenizer', () => {
    expect(sanitizeFtsQuery('hello world')).toContain('hello');
  });
});
