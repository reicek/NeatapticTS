import { jest } from '@jest/globals';
import { existsSync, mkdirSync, rmSync, writeFileSync, readFileSync } from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ---------------------------------------------------------------------------
// Import real @libsql/client BEFORE mocking so we can create real :memory: clients
// ---------------------------------------------------------------------------
const realLibsql = await import('@libsql/client');

// ---------------------------------------------------------------------------
// Mock state
// ---------------------------------------------------------------------------

let throwOnCreate = false;
let throwString = false;
let sharedMemoryClient = null;

// ---------------------------------------------------------------------------
// Mock @libsql/client — returns a shared :memory: client that persists across
// initTurso calls (close is no-op'd via Proxy so the DB isn't destroyed).
// ---------------------------------------------------------------------------

jest.unstable_mockModule('@libsql/client', () => ({
  createClient: (config) => {
    if (throwOnCreate) throw new Error('createClient failed');
    if (throwString) throw 'string error'; // eslint-disable-line no-throw-literal
    if (!sharedMemoryClient) {
      sharedMemoryClient = realLibsql.createClient({ url: ':memory:' });
    }
    return {
      execute: (...args) => sharedMemoryClient.execute(...args),
      batch: (...args) => sharedMemoryClient.batch(...args),
      close: async () => {},
    };
  },
}));

// ---------------------------------------------------------------------------
// Import module under test AFTER mocks
// ---------------------------------------------------------------------------
const { initTurso } = await import('./init-turso.mjs');

// Path to the real schema file
const SCHEMA_PATH = path.resolve(__dirname, 'schema-turso.sql');

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
describe('init-turso.mjs', () => {
  beforeEach(() => {
    throwOnCreate = false;
    throwString = false;
    sharedMemoryClient = realLibsql.createClient({ url: ':memory:' });
  });

  afterEach(async () => {
    if (sharedMemoryClient) {
      try {
        await sharedMemoryClient.close();
      } catch {
        // already closed
      }
    }
    sharedMemoryClient = null;
  });

  describe('initTurso', () => {
    it('initializes database with default schema and verifies FTS5', async () => {
      const result = await initTurso({
        url: ':memory:',
        schemaPath: SCHEMA_PATH,
      });
      expect(result.success).toBe(true);
      expect(result.schemaApplied).toBe(true);
      expect(result.statementsApplied).toBeGreaterThan(0);
      expect(result.diskannSkipped).toBe(0);
      expect(result.fts5Verified).toBe(true);
      expect(result.fts5MatchedRows).toBeGreaterThan(0);
      expect(result.tableCounts).toBeDefined();
      expect(result.tableCounts.documents).toBe(0);
      expect(result.tableCounts.chunks).toBe(0);
      expect(result.tableCounts._schema_version).toBeDefined();
    });

    it('skips DiskANN indexes when skipDiskann is true', async () => {
      const result = await initTurso({
        url: ':memory:',
        schemaPath: SCHEMA_PATH,
        skipDiskann: true,
      });
      expect(result.success).toBe(true);
      expect(result.diskannSkipped).toBeGreaterThan(0);
    });

    it('skips FTS5 verification when skipFts5Verify is true', async () => {
      const result = await initTurso({
        url: ':memory:',
        schemaPath: SCHEMA_PATH,
        skipFts5Verify: true,
      });
      expect(result.success).toBe(true);
      expect(result.fts5Verified).toBe(false);
      expect(result.fts5MatchedRows).toBe(0);
    });

    it('returns error when schema file does not exist', async () => {
      const result = await initTurso({
        url: ':memory:',
        schemaPath: 'nonexistent-schema.sql',
      });
      expect(result.success).toBe(false);
      expect(result.schemaApplied).toBe(false);
      expect(result.error).toBeDefined();
      expect(result.statementsApplied).toBe(0);
    });

    it('throws when createClient throws', async () => {
      throwOnCreate = true;
      await expect(initTurso({
        url: ':memory:',
        schemaPath: SCHEMA_PATH,
      })).rejects.toThrow('createClient failed');
    });

    it('creates parent directory for file: URLs', async () => {
      const tempDir = path.resolve(__dirname, '__test_tmp_init__');
      try {
        rmSync(tempDir, { recursive: true, force: true });
        const dbPath = path.join(tempDir, 'test.sqlite');
        const result = await initTurso({
          url: 'file:' + dbPath,
          schemaPath: SCHEMA_PATH,
        });
        expect(result.success).toBe(true);
        expect(existsSync(tempDir)).toBe(true);
      } finally {
        rmSync(tempDir, { recursive: true, force: true });
      }
    });

    it('passes authToken when provided', async () => {
      const result = await initTurso({
        url: ':memory:',
        authToken: 'test-token',
        schemaPath: SCHEMA_PATH,
      });
      expect(result.success).toBe(true);
    });

    it('uses env vars when options not provided', async () => {
      process.env.TURSO_DATABASE_URL = ':memory:';
      const result = await initTurso({ schemaPath: SCHEMA_PATH });
      expect(result.url).toBe(':memory:');
      delete process.env.TURSO_DATABASE_URL;
    });

    it('handles stale FTS5 test data from prior interrupted run', async () => {
      // First run — creates schema and verifies FTS5 (cleans up after itself)
      const result1 = await initTurso({
        url: ':memory:',
        schemaPath: SCHEMA_PATH,
      });
      expect(result1.success).toBe(true);

      // Manually insert stale test data into the shared :memory: client
      // (simulates an interrupted prior run that didn't clean up)
      await sharedMemoryClient.execute({
        sql: `INSERT INTO documents (file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
              VALUES (?, ?, ?, ?, ?, ?)`,
        args: ['__init_turso_fts5_test__.ts', '__test__', 0, 0, 'stale-sha', 0],
      });

      // Run initTurso again — should detect and clean up stale data, then verify FTS5
      const result2 = await initTurso({
        url: ':memory:',
        schemaPath: SCHEMA_PATH,
      });
      expect(result2.success).toBe(true);
      expect(result2.fts5Verified).toBe(true);
    });

    it('throws non-Error when createClient throws a string', async () => {
      throwString = true;
      await expect(initTurso({
        url: ':memory:',
        schemaPath: SCHEMA_PATH,
      })).rejects.toBe('string error');
    });
  });

  describe('splitSqlStatements (via initTurso)', () => {
    it('handles trigger statements with BEGIN...END blocks', async () => {
      // The real schema-turso.sql has CREATE TRIGGER with BEGIN...END
      // Applying it via initTurso tests splitSqlStatements
      const result = await initTurso({
        url: ':memory:',
        schemaPath: SCHEMA_PATH,
      });
      expect(result.success).toBe(true);
      // If triggers were split incorrectly, the schema would fail to apply
    });

    it('handles SQL with trailing content (no final semicolon)', async () => {
      // Create a temp SQL file with no trailing semicolon
      const tempDir = path.resolve(__dirname, '__test_tmp_split__');
      const tempSqlPath = path.join(tempDir, 'test-schema.sql');
      try {
        mkdirSync(tempDir, { recursive: true });
        writeFileSync(
          tempSqlPath,
          [
            "CREATE TABLE IF NOT EXISTS test_a (id INTEGER PRIMARY KEY);",
            "CREATE TABLE IF NOT EXISTS test_b (id INTEGER PRIMARY KEY);",
            "-- trailing comment without semicolon",
          ].join('\n'),
        );

        // We can't use initTurso directly because it expects specific tables
        // But we can verify splitSqlStatements handles it by checking the result
        // The schema application would fail because expected tables don't exist
        const result = await initTurso({
          url: ':memory:',
          schemaPath: tempSqlPath,
          skipFts5Verify: true,
        });
        // The schema applies (tables created), but FTS5 verify is skipped
        // getTableCounts would fail because expected tables don't exist
        expect(result.success).toBe(false);
      } finally {
        rmSync(tempDir, { recursive: true, force: true });
      }
    });
  });

  describe('CLI main()', () => {
    let writeSpy;
    let errorSpy;
    let originalExit;
    let exitCode;

    beforeEach(() => {
      writeSpy = jest
        .spyOn(console, 'log')
        .mockImplementation(() => {});
      errorSpy = jest
        .spyOn(console, 'error')
        .mockImplementation(() => {});
      originalExit = process.exit;
      exitCode = null;
      process.exit = (code) => {
        exitCode = code;
      };
    });

    afterEach(() => {
      writeSpy.mockRestore();
      errorSpy.mockRestore();
      process.exit = originalExit;
    });

    it('runs CLI with --json and successful init', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'init-turso.mjs',
      );
      process.argv = ['node', scriptPath, '--json', '--schema', SCHEMA_PATH];

      await import('./init-turso.mjs?cli-test=' + Date.now());
      const __start = Date.now();
      while (exitCode === null && Date.now() - __start < 10000) {
        await new Promise((r) => setTimeout(r, 10));
      }

      const stdout = writeSpy.mock.calls.map((c) => c[0]).join('');
      expect(stdout).toContain('"success": true');
      expect(exitCode).toBe(0);
    });

    it('runs CLI with human-readable output on success', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'init-turso.mjs',
      );
      process.argv = ['node', scriptPath, '--schema', SCHEMA_PATH];

      await import('./init-turso.mjs?cli-test=' + Date.now() + '2');
      const __start = Date.now();
      while (exitCode === null && Date.now() - __start < 10000) {
        await new Promise((r) => setTimeout(r, 10));
      }

      const stdout = writeSpy.mock.calls.map((c) => c[0]).join('');
      expect(stdout).toContain('initialized successfully');
      expect(stdout).toContain('URL:');
      expect(stdout).toContain('Schema applied:');
      expect(stdout).toContain('FTS5 verified:');
      expect(stdout).toContain('Table counts:');
      expect(exitCode).toBe(0);
    });

    it('runs CLI with --skip-diskann showing skipped count', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'init-turso.mjs',
      );
      process.argv = [
        'node',
        scriptPath,
        '--schema',
        SCHEMA_PATH,
        '--skip-diskann',
      ];

      await import('./init-turso.mjs?cli-test=' + Date.now() + '3');
      const __start = Date.now();
      while (exitCode === null && Date.now() - __start < 10000) {
        await new Promise((r) => setTimeout(r, 10));
      }

      const stdout = writeSpy.mock.calls.map((c) => c[0]).join('');
      expect(stdout).toContain('DiskANN skipped:');
      expect(exitCode).toBe(0);
    });

    it('runs CLI with human-readable output on failure', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'init-turso.mjs',
      );
      process.argv = [
        'node',
        scriptPath,
        '--schema',
        'nonexistent.sql',
      ];

      await import('./init-turso.mjs?cli-test=' + Date.now() + '4');
      const __start = Date.now();
      while (exitCode === null && Date.now() - __start < 10000) {
        await new Promise((r) => setTimeout(r, 10));
      }

      const stderr = errorSpy.mock.calls.map((c) => c[0]).join('\n');
      expect(stderr).toContain('FAILED');
      expect(stderr).toContain('Error:');
      expect(exitCode).toBe(1);
    });

    it('runs CLI with --json on failure', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'init-turso.mjs',
      );
      process.argv = [
        'node',
        scriptPath,
        '--json',
        '--schema',
        'nonexistent.sql',
      ];

      await import('./init-turso.mjs?cli-test=' + Date.now() + '5');
      const __start = Date.now();
      while (exitCode === null && Date.now() - __start < 10000) {
        await new Promise((r) => setTimeout(r, 10));
      }

      const stdout = writeSpy.mock.calls.map((c) => c[0]).join('');
      expect(stdout).toContain('"success": false');
      expect(exitCode).toBe(1);
    });
  });
});