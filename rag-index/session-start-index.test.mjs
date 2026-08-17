import { jest } from '@jest/globals';
import { writeFileSync, mkdirSync, rmSync, readFileSync, statSync } from 'node:fs';
import path from 'node:path';
import { createHash } from 'node:crypto';
import { runSessionStartIndex, runTouchPass, runBuildPass } from './session-start-index.mjs';

const TMP_DIR = path.resolve(process.cwd(), 'rag-index', '__test_tmp_session');

function makeTempFile(relativePath, content) {
  const fullPath = path.join(TMP_DIR, relativePath);
  mkdirSync(path.dirname(fullPath), { recursive: true });
  writeFileSync(fullPath, content, 'utf8');
  return fullPath;
}

function computeFreshness(filePath) {
  const content = readFileSync(filePath);
  return {
    mtime_ms: Math.trunc(statSync(filePath).mtimeMs),
    size: content.byteLength,
    sha256: createHash('sha256').update(content).digest('hex'),
  };
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
    it('returns fatalError when database does not exist (no client)', async () => {
      const result = await runSessionStartIndex({
        databasePath: 'nonexistent-dir-98765/db.sqlite',
      });
      expect(result.fatalError).toContain('Database not found');
      expect(result.touched).toBe(0);
      expect(result.buildPassRan).toBe(false);
    });

    it('runs touch pass with client and touchOnly=true (skips build pass)', async () => {
      const client = {
        execute: jest.fn(async ({ sql }) => {
          if (sql.includes('SELECT file_path')) {
            return {
              rows: [
                { file_path: 'nonexistent/file1.ts', mtime_ms: 1000, file_size: 100, sha256: 'abc' },
              ],
            };
          }
          return { rows: [] };
        }),
        close: jest.fn(async () => {}),
      };

      const result = await runSessionStartIndex({
        client,
        touchOnly: true,
      });

      expect(result.fatalError).toBe(null);
      expect(result.touched).toBe(0);
      expect(result.onDiskMissing).toBe(1);
      expect(result.buildPassRan).toBe(false);
    });

    it('runs both touch and build pass with client (non-touchOnly)', async () => {
      const client = {
        execute: jest.fn(async ({ sql }) => {
          if (sql.includes('SELECT file_path')) {
            return { rows: [] };
          }
          return { rows: [] };
        }),
        close: jest.fn(async () => {}),
      };

      // Mock spawnSync indirectly through runBuildPass
      // Actually, runBuildPass is called with the database path, not the client
      // So it will try to spawn a real subprocess
      // Let's test with touchOnly: false but mock runBuildPass by using a non-existent database path
      // Actually, runBuildPass uses the databasePath, not the client
      // The build pass always runs with the path, even when client is provided
      // So we need to handle the spawnSync call

      // Actually, looking at the code, runBuildPass(databasePath) is always called with the path
      // regardless of whether a client was used for the touch pass.
      // This will spawn a real subprocess. Let's just verify the summary shape.
      const result = await runSessionStartIndex({
        client,
        touchOnly: false,
        databasePath: 'nonexistent-dir-98765/touchbuild.sqlite',
      });

      expect(result.buildPassRan).toBe(true);
      expect(typeof result.buildPassExitCode).toBe('number');
    });

    it('handles empty documents list from client', async () => {
      const client = {
        execute: jest.fn(async () => ({ rows: [] })),
        close: jest.fn(async () => {}),
      };

      const result = await runSessionStartIndex({
        client,
        touchOnly: true,
      });

      expect(result.touched).toBe(0);
      expect(result.contentChanged).toBe(0);
      expect(result.onDiskMissing).toBe(0);
    });
  });

  describe('runTouchPass', () => {
    it('touches fresh documents and updates indexed_at via client', async () => {
      // Create a real temp file
      const tempFile = makeTempFile('fresh-file.ts', 'export const x = 1;');
      const stat = statSync(tempFile);
      const content = readFileSync(tempFile);
      const freshness = {
        mtime_ms: Math.trunc(stat.mtimeMs),
        size: content.byteLength,
        sha256: createHash('sha256').update(content).digest('hex'),
      };

      // Use a repo-relative path for the document row
      const relPath = path.relative(process.cwd(), tempFile).replaceAll('\\', '/');

      const updateCalls = [];
      const client = {
        execute: jest.fn(async ({ sql }) => {
          if (sql.includes('SELECT file_path')) {
            return {
              rows: [
                {
                  file_path: relPath,
                  mtime_ms: freshness.mtime_ms,
                  file_size: freshness.size,
                  sha256: freshness.sha256,
                },
              ],
            };
          }
          if (sql.includes('UPDATE documents')) {
            updateCalls.push(true);
          }
          return { rows: [] };
        }),
        close: jest.fn(async () => {}),
      };

      const result = await runTouchPass({ client });

      expect(result.touched).toBe(1);
      expect(result.contentChanged).toBe(0);
      expect(result.onDiskMissing).toBe(0);
      expect(updateCalls).toHaveLength(1);
    });

    it('reports changed when file content differs from stored hash', async () => {
      const tempFile = makeTempFile('changed-file.ts', 'export const y = 2;');
      const stat = statSync(tempFile);
      const relPath = path.relative(process.cwd(), tempFile).replaceAll('\\', '/');

      const client = {
        execute: jest.fn(async ({ sql }) => {
          if (sql.includes('SELECT file_path')) {
            return {
              rows: [
                {
                  file_path: relPath,
                  mtime_ms: Math.trunc(stat.mtimeMs),
                  file_size: 999, // wrong size
                  sha256: 'wrong-hash',
                },
              ],
            };
          }
          return { rows: [] };
        }),
        close: jest.fn(async () => {}),
      };

      const result = await runTouchPass({ client });

      expect(result.touched).toBe(0);
      expect(result.contentChanged).toBe(1);
    });

    it('reports missing when file does not exist on disk', async () => {
      const client = {
        execute: jest.fn(async ({ sql }) => {
          if (sql.includes('SELECT file_path')) {
            return {
              rows: [
                { file_path: 'totally/nonexistent/file.ts', mtime_ms: 1000, file_size: 100, sha256: 'abc' },
              ],
            };
          }
          return { rows: [] };
        }),
        close: jest.fn(async () => {}),
      };

      const result = await runTouchPass({ client });

      expect(result.touched).toBe(0);
      expect(result.onDiskMissing).toBe(1);
    });

    it('handles mixed fresh, changed, and missing documents', async () => {
      const freshFile = makeTempFile('fresh.ts', 'hello');
      const changedFile = makeTempFile('changed.ts', 'world');
      const freshStat = statSync(freshFile);
      const freshContent = readFileSync(freshFile);
      const freshHash = createHash('sha256').update(freshContent).digest('hex');
      const changedStat = statSync(changedFile);
      const freshRel = path.relative(process.cwd(), freshFile).replaceAll('\\', '/');
      const changedRel = path.relative(process.cwd(), changedFile).replaceAll('\\', '/');

      const client = {
        execute: jest.fn(async ({ sql }) => {
          if (sql.includes('SELECT file_path')) {
            return {
              rows: [
                // Fresh
                { file_path: freshRel, mtime_ms: Math.trunc(freshStat.mtimeMs), file_size: freshContent.byteLength, sha256: freshHash },
                // Changed (wrong hash)
                { file_path: changedRel, mtime_ms: Math.trunc(changedStat.mtimeMs), file_size: 999, sha256: 'wrong' },
                // Missing
                { file_path: 'no/such/file.ts', mtime_ms: 1, file_size: 1, sha256: 'x' },
              ],
            };
          }
          return { rows: [] };
        }),
        close: jest.fn(async () => {}),
      };

      const result = await runTouchPass({ client });

      expect(result.touched).toBe(1);
      expect(result.contentChanged).toBe(1);
      expect(result.onDiskMissing).toBe(1);
    });

    it('returns zero counts for empty documents from client', async () => {
      const client = {
        execute: jest.fn(async () => ({ rows: [] })),
        close: jest.fn(async () => {}),
      };

      const result = await runTouchPass({ client });
      expect(result.touched).toBe(0);
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

    it('runs with --touch-only and non-existent database (fatalError path)', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'session-start-index.mjs',
      );
      process.argv = ['node', scriptPath, '--touch-only', '--database=nonexistent-dir-98765/cli1.sqlite'];
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

    it('runs with --json and --touch-only with non-existent database', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'session-start-index.mjs',
      );
      process.argv = ['node', scriptPath, '--json', '--touch-only', '--database=nonexistent-dir-98765/cli2.sqlite'];
      process.exit = jest.fn(() => {});

      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      await import(`./session-start-index.mjs?cli-test=${Date.now()}-3`);
      await new Promise((r) => setTimeout(r, 200));

      const output = logSpy.mock.calls.map((c) => c[0]).join('\n');
      expect(output).toContain('"fatalError"');
      expect(process.exit).toHaveBeenCalledWith(1);
      logSpy.mockRestore();
    });

    it('handles .catch() path when runSessionStartIndex throws', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'session-start-index.mjs',
      );

      // Create a corrupt SQLite file that will cause runTouchPass to throw
      const corruptDbPath = path.join(TMP_DIR, 'corrupt.sqlite');
      writeFileSync(corruptDbPath, 'not a valid sqlite database', 'utf8');

      process.argv = ['node', scriptPath, '--touch-only', `--database=${corruptDbPath}`];
      process.exit = jest.fn(() => {});

      const errorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
      await import(`./session-start-index.mjs?cli-test=${Date.now()}-4`);
      await new Promise((r) => setTimeout(r, 500));

      expect(errorSpy).toHaveBeenCalled();
      expect(process.exit).toHaveBeenCalledWith(1);
      errorSpy.mockRestore();
    });

    it('handles .catch() path with --json when runSessionStartIndex throws', async () => {
      const scriptPath = path.resolve(
        process.cwd(),
        'rag-index',
        'session-start-index.mjs',
      );

      const corruptDbPath = path.join(TMP_DIR, 'corrupt2.sqlite');
      writeFileSync(corruptDbPath, 'not a valid sqlite database', 'utf8');

      process.argv = ['node', scriptPath, '--json', '--touch-only', `--database=${corruptDbPath}`];
      process.exit = jest.fn(() => {});

      const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
      await import(`./session-start-index.mjs?cli-test=${Date.now()}-5`);
      await new Promise((r) => setTimeout(r, 500));

      expect(logSpy).toHaveBeenCalled();
      const output = logSpy.mock.calls.map((c) => c[0]).join('');
      expect(output).toContain('"error"');
      expect(process.exit).toHaveBeenCalledWith(1);
      logSpy.mockRestore();
    });
  });
});