import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import * as realFsPromises from 'node:fs/promises';

const REPO_ROOT = path.resolve();
const SCRIPT_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/validate-tsconfig-docs.mjs',
);

jest.unstable_mockModule('fast-glob', () => ({
  default: jest.fn(),
}));

jest.unstable_mockModule('node:fs/promises', () => ({
  ...realFsPromises,
  readFile: jest.fn(),
}));

let mockFg;
let mockFs;

beforeEach(async () => {
  jest.resetModules();
  mockFg = (await import('fast-glob')).default;
  mockFs = await import('node:fs/promises');
  jest.clearAllMocks();
  // Default: all globs match
  mockFs.readFile.mockResolvedValue(
    JSON.stringify({ include: ['src/**/*.ts'] }),
  );
  mockFg.mockResolvedValue(['src/index.ts']);
});

/**
 * Imports the module as the entry point so the top-level `main()` runs.
 *
 * @param {string[]} argv - Flags to pass (e.g. `['--json']`, `['--help']`).
 * @returns {Promise<{exitCode: number | undefined}>} The exitCode set by main().
 */
async function importModuleAsEntry(argv) {
  const origArgv = process.argv;
  const origExitCode = process.exitCode;
  process.argv = [process.execPath, SCRIPT_PATH, ...argv];
  process.exitCode = undefined;
  try {
    await import(SCRIPT_PATH);
  } finally {
    process.argv = origArgv;
  }
  const exitCode = process.exitCode;
  process.exitCode = origExitCode;
  return { exitCode };
}

/**
 * Captures `console.log` output during an async operation.
 *
 * @param {() => Promise<void>} fn - Async function whose console.log to capture.
 * @returns {Promise<string[]>} Array of captured log lines.
 */
async function captureConsoleLog(fn) {
  const originalLog = console.log;
  const chunks = [];
  console.log = (...args) => {
    chunks.push(args.map((a) => String(a)).join(' '));
  };
  try {
    await fn();
    return chunks;
  } finally {
    console.log = originalLog;
  }
}

describe('validate-tsconfig-docs', () => {
  describe('validateTsconfigDocs (direct import)', () => {
    let validateTsconfigDocs;

    beforeEach(async () => {
      validateTsconfigDocs = (await import('./validate-tsconfig-docs.mjs'))
        .validateTsconfigDocs;
    });

    it('returns pass:true when all include globs resolve', async () => {
      const result = await validateTsconfigDocs();
      assert.strictEqual(result.pass, true);
      assert.ok(Array.isArray(result.checked_paths));
      assert.deepStrictEqual(result.missing_paths, []);
      assert.strictEqual(result.fixHint, null);
    });

    it('provides fixHint only when paths are missing', async () => {
      mockFg.mockResolvedValue([]);
      const result = await validateTsconfigDocs();
      assert.strictEqual(result.pass, false);
      assert.deepStrictEqual(result.missing_paths, ['src/**/*.ts']);
      assert.ok(result.fixHint);
    });

    it('handles non-array include as empty checked_paths', async () => {
      mockFs.readFile.mockResolvedValue(
        JSON.stringify({ include: 'not-an-array' }),
      );
      const result = await validateTsconfigDocs();
      assert.deepStrictEqual(result.checked_paths, []);
      assert.strictEqual(result.pass, true);
    });

    it('filters non-string entries in include array', async () => {
      mockFs.readFile.mockResolvedValue(
        JSON.stringify({ include: ['src/**/*.ts', 42, null, true] }),
      );
      const result = await validateTsconfigDocs();
      assert.deepStrictEqual(result.checked_paths, ['src/**/*.ts']);
    });
  });

  describe('main() via entry point', () => {
    it('prints usage on --help', async () => {
      const logs = await captureConsoleLog(async () => {
        await importModuleAsEntry(['--help']);
      });
      assert.ok(logs.some((l) => l.includes('Validate docs TSConfig')));
    });

    it('prints usage on -h', async () => {
      const logs = await captureConsoleLog(async () => {
        await importModuleAsEntry(['-h']);
      });
      assert.ok(logs.some((l) => l.includes('Validate docs TSConfig')));
    });

    it('emits JSON on --json when all paths match', async () => {
      let exitCode;
      const logs = await captureConsoleLog(async () => {
        ({ exitCode } = await importModuleAsEntry(['--json']));
      });
      assert.strictEqual(exitCode, 0);
      const parsed = JSON.parse(logs.join('\n'));
      assert.strictEqual(parsed.pass, true);
    });

    it('emits JSON on --json with missing paths and exitCode 1', async () => {
      mockFg.mockResolvedValue([]);
      let exitCode;
      const logs = await captureConsoleLog(async () => {
        ({ exitCode } = await importModuleAsEntry(['--json']));
      });
      assert.strictEqual(exitCode, 1);
      const parsed = JSON.parse(logs.join('\n'));
      assert.strictEqual(parsed.pass, false);
      assert.ok(parsed.fixHint);
    });

    it('prints PASS in plain mode when all paths match', async () => {
      let exitCode;
      const logs = await captureConsoleLog(async () => {
        ({ exitCode } = await importModuleAsEntry([]));
      });
      assert.strictEqual(exitCode, 0);
      assert.ok(logs.some((l) => l.includes('PASS validate-tsconfig-docs')));
    });

    it('prints FAIL and fixHint in plain mode when paths are missing', async () => {
      mockFg.mockResolvedValue([]);
      let exitCode;
      const logs = await captureConsoleLog(async () => {
        ({ exitCode } = await importModuleAsEntry([]));
      });
      assert.strictEqual(exitCode, 1);
      assert.ok(logs.some((l) => l.includes('FAIL validate-tsconfig-docs')));
      assert.ok(logs.some((l) => l.includes('fixHint:')));
    });

    it('handles Error thrown in validateTsconfigDocs (JSON mode)', async () => {
      mockFs.readFile.mockRejectedValue(new Error('read error'));
      let exitCode;
      const logs = await captureConsoleLog(async () => {
        ({ exitCode } = await importModuleAsEntry(['--json']));
      });
      assert.strictEqual(exitCode, 1);
      const parsed = JSON.parse(logs.join('\n'));
      assert.strictEqual(parsed.pass, false);
      assert.strictEqual(parsed.error, 'read error');
    });

    it('handles Error thrown in validateTsconfigDocs (plain mode)', async () => {
      mockFs.readFile.mockRejectedValue(new Error('read error'));
      let exitCode;
      const logs = await captureConsoleLog(async () => {
        ({ exitCode } = await importModuleAsEntry([]));
      });
      assert.strictEqual(exitCode, 1);
      assert.ok(logs.some((l) => l.includes('FAIL validate-tsconfig-docs')));
      assert.ok(logs.some((l) => l.includes('fixHint:')));
    });

    it('handles non-Error throw in main (JSON mode)', async () => {
      mockFs.readFile.mockRejectedValue('string error');
      let exitCode;
      const logs = await captureConsoleLog(async () => {
        ({ exitCode } = await importModuleAsEntry(['--json']));
      });
      assert.strictEqual(exitCode, 1);
      const parsed = JSON.parse(logs.join('\n'));
      assert.strictEqual(parsed.pass, false);
      assert.strictEqual(parsed.error, 'string error');
    });

    it('handles non-Error throw in main (plain mode)', async () => {
      mockFs.readFile.mockRejectedValue('string error');
      let exitCode;
      const logs = await captureConsoleLog(async () => {
        ({ exitCode } = await importModuleAsEntry([]));
      });
      assert.strictEqual(exitCode, 1);
      assert.ok(logs.some((l) => l.includes('FAIL validate-tsconfig-docs')));
    });
  });

  describe('CLI entry (--help)', () => {
    it('prints usage and exits 0', () => {
      const result = spawnSync(process.execPath, [SCRIPT_PATH, '--help'], {
        cwd: REPO_ROOT,
        encoding: 'utf8',
      });
      assert.strictEqual(result.status, 0);
      assert.ok(result.stdout.includes('Validate docs TSConfig'));
    });
  });

  describe('CLI entry (--json)', () => {
    it('emits JSON with pass/checked_paths/missing_paths', () => {
      const result = spawnSync(process.execPath, [SCRIPT_PATH, '--json'], {
        cwd: REPO_ROOT,
        encoding: 'utf8',
      });
      const payload = JSON.parse(result.stdout.trim());
      assert.ok(typeof payload.pass === 'boolean');
      assert.ok(Array.isArray(payload.checked_paths));
      assert.ok(Array.isArray(payload.missing_paths));
    });
  });

  describe('CLI entry (plain text)', () => {
    it('prints PASS or FAIL', () => {
      const result = spawnSync(process.execPath, [SCRIPT_PATH], {
        cwd: REPO_ROOT,
        encoding: 'utf8',
      });
      assert.ok(
        result.stdout.includes('PASS validate-tsconfig-docs') ||
          result.stdout.includes('FAIL validate-tsconfig-docs'),
      );
    });
  });
});