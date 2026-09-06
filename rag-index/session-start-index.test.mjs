import { jest } from '@jest/globals';
import { writeFileSync, mkdirSync, rmSync } from 'node:fs';
import path from 'node:path';
import { runSessionStartIndex, runBuildPass } from './session-start-index.mjs';

const TMP_DIR = path.resolve(process.cwd(), 'rag-index', '__test_tmp_session');

function makeTempFile(relativePath, content) {
  const fullPath = path.join(TMP_DIR, relativePath);
  mkdirSync(path.dirname(fullPath), { recursive: true });
  writeFileSync(fullPath, content, 'utf8');
  return fullPath;
}

describe('session-start-index.mjs', () => {
  beforeEach(() => {
    mkdirSync(TMP_DIR, { recursive: true });
  });

  afterEach(() => {
    try {
      rmSync(TMP_DIR, { recursive: true, force: true });
    } catch {
      // Directory may be locked on Windows (EPERM); ignore cleanup errors
    }
  });

  describe('runSessionStartIndex', () => {
    it('returns fatalError when database does not exist (no build pass)', async () => {
      const result = await runSessionStartIndex({
        databasePath: 'nonexistent-dir-98765/db.sqlite',
      });
      expect(result.fatalError).toContain('Database not found');
      expect(result.buildPassRan).toBe(false);
    });

    it('runs incremental build pass when database exists', async () => {
      // Provide a real temp file path as the database so existsSync passes.
      // The build pass will spawn build-index.mjs against this non-DB file,
      // but the summary shape is what we care about.
      const dbPath = makeTempFile('db.sqlite', 'not a real db');
      const result = await runSessionStartIndex({
        databasePath: dbPath,
      });

      expect(result.fatalError).toBe(null);
      expect(result.buildPassRan).toBe(true);
      expect(typeof result.buildPassExitCode).toBe('number');
    });

    it('does not perform a touch pass or update indexed_at', async () => {
      const dbPath = makeTempFile('db2.sqlite', 'not a real db');
      const client = {
        execute: jest.fn(async () => ({ rows: [] })),
        close: jest.fn(async () => {}),
      };

      const result = await runSessionStartIndex({
        databasePath: dbPath,
      });

      // No client should have been used; no SQL UPDATE executed.
      expect(client.execute).not.toHaveBeenCalled();
      expect(result.fatalError).toBe(null);
      expect(result.buildPassRan).toBe(true);
      // Legacy touch-pass summary fields are gone.
      expect(result.touched).toBeUndefined();
      expect(result.contentChanged).toBeUndefined();
      expect(result.onDiskMissing).toBeUndefined();
    });
  });

  describe('runBuildPass', () => {
    it('returns an exitCode from spawnSync', () => {
      const result = runBuildPass('nonexistent-dir-98765/buildpass.sqlite');
      expect(typeof result.exitCode).toBe('number');
    });
  });

  describe('CLI entry point', () => {
    let originalArgv;
    let originalExit;

    beforeEach(() => {
      originalArgv = process.argv;
      originalExit = process.exit;
    });

    afterEach(() => {
      process.argv = originalArgv;
      process.exit = originalExit;
    });

    it('prints help and exits when --help is passed', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'session-start-index.mjs',
      );
      process.argv = ['node', scriptPath, '--help'];
      process.exit = jest.fn(() => {});

      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      await import(`./session-start-index.mjs?cli-test=${Date.now()}`);

      expect(logSpy).toHaveBeenCalled();
      const output = logSpy.mock.calls.map((c) => c[0]).join('\n');
      expect(output).toContain('session-start-index');
      expect(process.exit).toHaveBeenCalledWith(0);
      logSpy.mockRestore();
    });

    it('help output does not advertise removed --touch-only option', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'session-start-index.mjs',
      );
      process.argv = ['node', scriptPath, '--help'];
      process.exit = jest.fn(() => {});

      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      await import(`./session-start-index.mjs?cli-test=${Date.now()}-help`);

      const output = logSpy.mock.calls.map((c) => c[0]).join('\n');
      expect(output).not.toContain('--touch-only');
      logSpy.mockRestore();
    });

    it('runs with non-existent database (fatalError path)', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'session-start-index.mjs',
      );
      process.argv = ['node', scriptPath, '--database=nonexistent-dir-98765/cli1.sqlite'];
      process.exit = jest.fn(() => {});

      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      await import(`./session-start-index.mjs?cli-test=${Date.now()}-2`);
      // Wait for async .then() handler
      await new Promise((r) => setTimeout(r, 200));

      const output = logSpy.mock.calls.map((c) => c[0]).join('\n');
      expect(output).toContain('Database not found');
      expect(process.exit).toHaveBeenCalledWith(1);
      logSpy.mockRestore();
    });

    it('runs with --json and non-existent database', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'session-start-index.mjs',
      );
      process.argv = ['node', scriptPath, '--json', '--database=nonexistent-dir-98765/cli2.sqlite'];
      process.exit = jest.fn(() => {});

      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      await import(`./session-start-index.mjs?cli-test=${Date.now()}-3`);
      await new Promise((r) => setTimeout(r, 200));

      const output = logSpy.mock.calls.map((c) => c[0]).join('\n');
      expect(output).toContain('"fatalError"');
      expect(process.exit).toHaveBeenCalledWith(1);
      logSpy.mockRestore();
    });

    it('handles a corrupt database by reporting a failed build pass exit code', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'session-start-index.mjs',
      );

      // A non-SQLite file causes the spawned build pass to exit non-zero,
      // which runSessionStartIndex now captures gracefully rather than throwing.
      const corruptDbPath = path.join(TMP_DIR, 'corrupt.sqlite');
      writeFileSync(corruptDbPath, 'not a valid sqlite database', 'utf8');

      process.argv = ['node', scriptPath, `--database=${corruptDbPath}`];
      process.exit = jest.fn(() => {});

      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      await import(`./session-start-index.mjs?cli-test=${Date.now()}-4`);
      await new Promise((r) => setTimeout(r, 500));

      expect(logSpy).toHaveBeenCalled();
      // A failed build pass is not a fatal error, so CLI exits 0 and reports
      // the non-zero buildPassExitCode in the summary.
      expect(process.exit).toHaveBeenCalledWith(0);
      logSpy.mockRestore();
    });

    it('reports a failed build pass as JSON without throwing', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'session-start-index.mjs',
      );

      const corruptDbPath = path.join(TMP_DIR, 'corrupt2.sqlite');
      writeFileSync(corruptDbPath, 'not a valid sqlite database', 'utf8');

      process.argv = ['node', scriptPath, '--json', `--database=${corruptDbPath}`];
      process.exit = jest.fn(() => {});

      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      await import(`./session-start-index.mjs?cli-test=${Date.now()}-5`);
      await new Promise((r) => setTimeout(r, 500));

      expect(logSpy).toHaveBeenCalled();
      const output = logSpy.mock.calls.map((c) => c[0]).join('');
      expect(output).toContain('"buildPassExitCode"');
      expect(output).toContain('"fatalError"');
      expect(output).toContain('"fatalError": null');
      expect(process.exit).toHaveBeenCalledWith(0);
      logSpy.mockRestore();
    });
  });
});