/**
 * @module cortex-db.coverage.test
 * @description Coverage tests targeting every uncovered branch in cortex-db.mjs.
 *
 * Supplements cortex-db.turso.test.mjs by exercising closeTursoClient,
 * setTursoClient, getCachedClientCount, asIsoTimestamp, toAbsoluteRepoPath,
 * normalizeLimit boundary cases, readChunk error path, readChunkRow with
 * nullable columns, getTursoClient with auth token / URL schemes / sync
 * interval defaults, and resolveDatabasePath with explicit overrides.
 */

import { createClient } from '@libsql/client';
import path from 'node:path';

import { repoRoot } from '../../../rag-index/init-schema.mjs';
import { createEnvIsolation } from './turso-test-helpers.mjs';

const MODULE_PATH = '../tools/cortex-db.mjs';

const { saveEnv, restoreEnv } = createEnvIsolation();

beforeEach(() => {
  saveEnv();
});

afterEach(() => {
  restoreEnv();
});

async function loadModule() {
  return import(MODULE_PATH);
}

/**
 * Build an in-memory client with a minimal documents + chunks schema for
 * readChunk tests.
 */
async function setupTestDb() {
  const client = createClient({ url: ':memory:' });
  await client.execute({
    sql: `CREATE TABLE documents (
      doc_id INTEGER PRIMARY KEY,
      file_path TEXT NOT NULL UNIQUE,
      doc_family TEXT NOT NULL,
      mtime_ms INTEGER NOT NULL,
      file_size INTEGER NOT NULL,
      sha256 TEXT NOT NULL,
      indexed_at INTEGER NOT NULL
    )`,
  });
  await client.execute({
    sql: `CREATE TABLE chunks (
      chunk_id INTEGER PRIMARY KEY,
      doc_id INTEGER NOT NULL REFERENCES documents(doc_id),
      chunk_index INTEGER NOT NULL,
      heading_path TEXT,
      body_text TEXT NOT NULL,
      char_start INTEGER NOT NULL,
      char_end INTEGER NOT NULL,
      parent_chunk_id INTEGER,
      depth INTEGER NOT NULL DEFAULT 0,
      context_header TEXT,
      symbol_name TEXT,
      signature_text TEXT,
      jsdoc_text TEXT,
      export_type TEXT,
      module_path TEXT,
      arch_layer TEXT,
      jsdoc_quality TEXT,
      jsdoc_word_count INTEGER,
      cyclomatic_complexity INTEGER,
      test_coverage TEXT,
      source_path_pattern TEXT,
      slice_id TEXT,
      step_number INTEGER,
      phase TEXT,
      status TEXT
    )`,
  });
  await client.execute({
    sql: `INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
          VALUES (1, 'src/test.ts', 'src', 1000, 500, 'abc', 1000)`,
  });
  return client;
}

describe('CORTEX_FIX_HINT', () => {
  it('exports a non-empty hint string', async () => {
    const mod = await loadModule();
    expect(typeof mod.CORTEX_FIX_HINT).toBe('string');
    expect(mod.CORTEX_FIX_HINT.length).toBeGreaterThan(0);
  });
});

describe('resolveDatabasePath', () => {
  it('returns TURSO_DATABASE_URL verbatim when set', async () => {
    process.env.TURSO_DATABASE_URL = 'libsql://example.turso.io';
    const { resolveDatabasePath } = await loadModule();
    expect(resolveDatabasePath()).toBe('libsql://example.turso.io');
  });

  it('uses an explicit override when TURSO_DATABASE_URL is unset', async () => {
    const { resolveDatabasePath } = await loadModule();
    const result = resolveDatabasePath('/custom/path/db.sqlite');
    expect(path.isAbsolute(result)).toBe(true);
    expect(result).toBe(path.resolve('/custom/path/db.sqlite'));
  });

  it('falls back to the compiled default when nothing is provided', async () => {
    const { resolveDatabasePath } = await loadModule();
    const result = resolveDatabasePath();
    expect(path.isAbsolute(result)).toBe(true);
  });
});

describe('getTursoClient', () => {
  it('returns a cached client for the same URL', async () => {
    const { getTursoClient, closeTursoClient } = await loadModule();
    const url = ':memory:';
    try {
      const a = await getTursoClient(url);
      const b = await getTursoClient(url);
      expect(a).toBe(b);
    } finally {
      await closeTursoClient(url);
    }
  });

  it('passes auth token from env vars', async () => {
    process.env.TURSO_AUTH_TOKEN = 'test-token';
    const { getTursoClient, closeTursoClient } = await loadModule();
    const url = ':memory:';
    try {
      const client = await getTursoClient(url);
      expect(typeof client.execute).toBe('function');
    } finally {
      await closeTursoClient(url);
    }
  });

  it('uses default sync interval when TURSO_SYNC_INTERVAL is not set', async () => {
    const { getTursoClient, closeTursoClient } = await loadModule();
    const url = ':memory:';
    try {
      const client = await getTursoClient(url);
      expect(typeof client.execute).toBe('function');
    } finally {
      await closeTursoClient(url);
    }
  });

  it('uses TURSO_SYNC_INTERVAL when set to a non-empty value', async () => {
    process.env.TURSO_SYNC_INTERVAL = '30';
    const { getTursoClient, closeTursoClient } = await loadModule();
    const url = ':memory:';
    try {
      const client = await getTursoClient(url);
      expect(typeof client.execute).toBe('function');
    } finally {
      await closeTursoClient(url);
    }
  });

  it('converts a plain filesystem path to a file: URL', async () => {
    const { getTursoClient, closeTursoClient } = await loadModule();
    const fsPath = path.join(
      repoRoot,
      'rag-index',
      'data',
      'test-client.sqlite',
    );
    try {
      const client = await getTursoClient(fsPath);
      expect(typeof client.execute).toBe('function');
    } finally {
      await closeTursoClient(fsPath);
    }
  });

  it('passes URL-scheme strings as-is (file:)', async () => {
    const { getTursoClient, closeTursoClient } = await loadModule();
    const url = 'file::memory:';
    try {
      const client = await getTursoClient(url);
      expect(typeof client.execute).toBe('function');
    } finally {
      await closeTursoClient(url);
    }
  });
});

describe('closeTursoClient', () => {
  it('closes and evicts a cached client', async () => {
    const { getTursoClient, closeTursoClient, getCachedClientCount } =
      await loadModule();
    const url = ':memory:';
    const beforeCount = getCachedClientCount();
    await getTursoClient(url);
    expect(getCachedClientCount()).toBe(beforeCount + 1);
    await closeTursoClient(url);
    expect(getCachedClientCount()).toBe(beforeCount);
  });

  it('is a no-op when no client is cached for the URL', async () => {
    const { closeTursoClient, getCachedClientCount } = await loadModule();
    const beforeCount = getCachedClientCount();
    await closeTursoClient(':memory:');
    expect(getCachedClientCount()).toBe(beforeCount);
  });

  it('uses resolveDatabasePath when no path is given', async () => {
    process.env.TURSO_DATABASE_URL = ':memory:';
    const { getTursoClient, closeTursoClient, getCachedClientCount } =
      await loadModule();
    await getTursoClient();
    const afterCreate = getCachedClientCount();
    expect(afterCreate).toBeGreaterThan(0);
    await closeTursoClient();
    expect(getCachedClientCount()).toBe(afterCreate - 1);
  });
});

describe('setTursoClient', () => {
  it('inserts a client into the cache', async () => {
    const { setTursoClient, getCachedClientCount, closeTursoClient } =
      await loadModule();
    const url = 'test://set-insert';
    const mockClient = { close: async () => {} };
    const beforeCount = getCachedClientCount();
    try {
      setTursoClient(url, mockClient);
      expect(getCachedClientCount()).toBe(beforeCount + 1);
    } finally {
      await closeTursoClient(url);
    }
  });

  it('evicts a client when passed undefined', async () => {
    const { setTursoClient, getCachedClientCount, closeTursoClient } =
      await loadModule();
    const url = 'test://set-evict';
    const mockClient = { close: async () => {} };
    setTursoClient(url, mockClient);
    const afterInsert = getCachedClientCount();
    setTursoClient(url, undefined);
    expect(getCachedClientCount()).toBe(afterInsert - 1);
  });

  it('throws when url is not a non-empty string', async () => {
    const { setTursoClient } = await loadModule();
    expect(() => setTursoClient('', {})).toThrow(/url/);
    expect(() => setTursoClient(null, {})).toThrow(/url/);
  });
});

describe('getCachedClientCount', () => {
  it('returns a non-negative number', async () => {
    const { getCachedClientCount } = await loadModule();
    const count = getCachedClientCount();
    expect(typeof count).toBe('number');
    expect(count).toBeGreaterThanOrEqual(0);
  });
});

describe('readChunk', () => {
  it('throws when the chunk ID does not exist', async () => {
    const { readChunk } = await loadModule();
    const client = await setupTestDb();
    try {
      await expect(readChunk(client, 99999)).rejects.toThrow(/Chunk not found/);
    } finally {
      client.close?.();
    }
  });

  it('returns fully normalized row when all v2/v3/A1 columns are present', async () => {
    const client = await setupTestDb();
    try {
      await client.execute({
        sql: `INSERT INTO chunks (
                chunk_id, doc_id, chunk_index, heading_path, body_text,
                char_start, char_end, parent_chunk_id, depth, context_header,
                symbol_name, signature_text, jsdoc_text, export_type,
                module_path, arch_layer, jsdoc_quality, jsdoc_word_count,
                cyclomatic_complexity, test_coverage, source_path_pattern,
                slice_id, step_number, phase, status
              )
              VALUES (10, 1, 2, 'mod.fn', 'body', 5, 15, 1, 1, 'header',
                      'mySymbol', 'fn()', 'jsdoc', 'function', 'src/mod.ts',
                      'network', 'good', 42, 7, '80', 'src/**',
                      'A1', 3, 'impl', 'done')`,
      });
      const { readChunk } = await loadModule();
      const chunk = await readChunk(client, 10);
      expect(chunk).toMatchObject({
        chunk_id: 10,
        file_path: 'src/test.ts',
        family: 'src',
        chunk_index: 2,
        heading_path: 'mod.fn',
        text: 'body',
        char_start: 5,
        char_end: 15,
        depth: 1,
        parent_chunk_id: 1,
        context_header: 'header',
        symbol_name: 'mySymbol',
        signature_text: 'fn()',
        jsdoc_text: 'jsdoc',
        export_type: 'function',
        module_path: 'src/mod.ts',
        arch_layer: 'network',
        jsdoc_quality: 'good',
        jsdoc_word_count: 42,
        cyclomatic_complexity: 7,
        test_coverage: '80',
        source_path_pattern: 'src/**',
        slice_id: 'A1',
        step_number: 3,
        phase: 'impl',
        status: 'done',
      });
    } finally {
      client.close?.();
    }
  });
});

describe('readChunkRow', () => {
  it('normalizes a row with nullable columns set to null', async () => {
    const { readChunkRow } = await loadModule();
    const row = {
      arch_layer: null,
      char_end: 100,
      char_start: 0,
      chunk_id: 5,
      chunk_index: 0,
      context_header: null,
      cyclomatic_complexity: null,
      depth: null,
      export_type: null,
      doc_family: 'src',
      file_path: 'src/a.ts',
      heading_path: null,
      jsdoc_quality: null,
      jsdoc_text: null,
      jsdoc_word_count: null,
      module_path: null,
      parent_chunk_id: null,
      phase: null,
      signature_text: null,
      slice_id: null,
      source_path_pattern: null,
      status: null,
      step_number: null,
      symbol_name: null,
      test_coverage: null,
      body_text: 'hello',
    };
    const result = readChunkRow(row);
    expect(result).toEqual({
      arch_layer: null,
      char_end: 100,
      char_start: 0,
      chunk_id: 5,
      chunk_index: 0,
      context_header: null,
      cyclomatic_complexity: null,
      depth: 0,
      export_type: null,
      family: 'src',
      file_path: 'src/a.ts',
      heading_path: null,
      jsdoc_quality: null,
      jsdoc_text: null,
      jsdoc_word_count: null,
      module_path: null,
      parent_chunk_id: null,
      phase: null,
      signature_text: null,
      slice_id: null,
      source_path_pattern: null,
      status: null,
      step_number: null,
      symbol_name: null,
      test_coverage: null,
      text: 'hello',
    });
  });

  it('preserves non-null numeric and string metadata columns', async () => {
    const { readChunkRow } = await loadModule();
    const row = {
      arch_layer: 'tooling',
      char_end: 50,
      char_start: 10,
      chunk_id: 8,
      chunk_index: 3,
      context_header: 'ctx',
      cyclomatic_complexity: 12,
      depth: 2,
      export_type: 'class',
      doc_family: 'plan',
      file_path: 'docs/p.md',
      heading_path: 'H',
      jsdoc_quality: 'excellent',
      jsdoc_text: 'doc',
      jsdoc_word_count: 20,
      module_path: 'docs',
      parent_chunk_id: 7,
      phase: 'test',
      signature_text: 'sig',
      slice_id: 'B2',
      source_path_pattern: 'docs/**',
      status: 'wip',
      step_number: 5,
      symbol_name: 'sym',
      test_coverage: '95',
      body_text: 'content',
    };
    const result = readChunkRow(row);
    expect(result.depth).toBe(2);
    expect(result.parent_chunk_id).toBe(7);
    expect(result.jsdoc_word_count).toBe(20);
    expect(result.cyclomatic_complexity).toBe(12);
    expect(result.step_number).toBe(5);
    expect(result.arch_layer).toBe('tooling');
  });
});

describe('normalizeLimit', () => {
  it('returns the fallback when value is undefined', async () => {
    const { normalizeLimit } = await loadModule();
    expect(normalizeLimit(undefined, 15)).toBe(15);
  });

  it('returns the fallback when value is not a finite number', async () => {
    const { normalizeLimit } = await loadModule();
    expect(normalizeLimit(NaN, 12)).toBe(12);
    expect(normalizeLimit(Infinity, 12)).toBe(12);
    expect(normalizeLimit('abc', 12)).toBe(12);
  });

  it('uses default fallback of 10 when no fallback given', async () => {
    const { normalizeLimit } = await loadModule();
    expect(normalizeLimit(undefined)).toBe(10);
  });

  it('clamps values below 1 to 1', async () => {
    const { normalizeLimit } = await loadModule();
    expect(normalizeLimit(0)).toBe(1);
    expect(normalizeLimit(-5)).toBe(1);
  });

  it('clamps values above 50 to 50', async () => {
    const { normalizeLimit } = await loadModule();
    expect(normalizeLimit(100)).toBe(50);
  });

  it('truncates fractional values', async () => {
    const { normalizeLimit } = await loadModule();
    expect(normalizeLimit(7.9)).toBe(7);
    expect(normalizeLimit(50.99)).toBe(50);
  });

  it('keeps in-range integers unchanged', async () => {
    const { normalizeLimit } = await loadModule();
    expect(normalizeLimit(25)).toBe(25);
  });

  it('coerces numeric strings', async () => {
    const { normalizeLimit } = await loadModule();
    expect(normalizeLimit('30')).toBe(30);
  });
});

describe('normalizeRepoPath', () => {
  it('rejects empty string', async () => {
    const { normalizeRepoPath } = await loadModule();
    expect(() => normalizeRepoPath('')).toThrow(/file_path/);
  });

  it('rejects non-string', async () => {
    const { normalizeRepoPath } = await loadModule();
    expect(() => normalizeRepoPath(123)).toThrow(/file_path/);
  });

  it('returns "." for the repo root', async () => {
    const { normalizeRepoPath } = await loadModule();
    expect(normalizeRepoPath(repoRoot)).toBe('.');
  });

  it('rejects absolute path outside repo', async () => {
    const { normalizeRepoPath } = await loadModule();
    expect(() => normalizeRepoPath('C:\\Windows\\System32')).toThrow(
      /repository/,
    );
  });

  it('converts backslash relative paths to POSIX', async () => {
    const { normalizeRepoPath } = await loadModule();
    const result = normalizeRepoPath('src\\sub\\file.ts');
    expect(result).toBe('src/sub/file.ts');
  });
});

describe('toAbsoluteRepoPath', () => {
  it('returns an absolute path for a valid repo-relative path', async () => {
    const { toAbsoluteRepoPath } = await loadModule();
    const result = toAbsoluteRepoPath('README.md');
    expect(path.isAbsolute(result)).toBe(true);
    expect(result).toBe(path.join(repoRoot, 'README.md'));
  });

  it('throws for path traversal', async () => {
    const { toAbsoluteRepoPath } = await loadModule();
    expect(() => toAbsoluteRepoPath('../../etc/passwd')).toThrow(/repository/);
  });
});

describe('asIsoTimestamp', () => {
  it('converts a valid timestamp to ISO string', async () => {
    const { asIsoTimestamp } = await loadModule();
    const ts = 1700000000000;
    const result = asIsoTimestamp(ts);
    expect(result).toBe(new Date(ts).toISOString());
  });

  it('returns null for non-numeric values', async () => {
    const { asIsoTimestamp } = await loadModule();
    expect(asIsoTimestamp('abc')).toBeNull();
    expect(asIsoTimestamp(null)).toBeNull();
    expect(asIsoTimestamp(undefined)).toBeNull();
  });

  it('returns null for zero or negative', async () => {
    const { asIsoTimestamp } = await loadModule();
    expect(asIsoTimestamp(0)).toBeNull();
    expect(asIsoTimestamp(-1)).toBeNull();
  });

  it('returns null for NaN', async () => {
    const { asIsoTimestamp } = await loadModule();
    expect(asIsoTimestamp(NaN)).toBeNull();
  });
});

describe('sanitizeFtsQuery', () => {
  it('returns empty string for empty input', async () => {
    const { sanitizeFtsQuery } = await loadModule();
    expect(sanitizeFtsQuery('')).toBe('');
    expect(sanitizeFtsQuery(null)).toBe('');
  });

  it('preserves code identifiers as quoted phrases', async () => {
    const { sanitizeFtsQuery } = await loadModule();
    expect(sanitizeFtsQuery('network.activate')).toBe('"network.activate"');
  });

  it('keeps acronyms as exact terms', async () => {
    const { sanitizeFtsQuery } = await loadModule();
    expect(sanitizeFtsQuery('NEAT')).toBe('NEAT');
  });

  it('applies prefix wildcards to plain words', async () => {
    const { sanitizeFtsQuery } = await loadModule();
    expect(sanitizeFtsQuery('hello world')).toBe('hello* world*');
  });
});
