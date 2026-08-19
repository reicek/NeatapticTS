import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import path from 'node:path';

jest.unstable_mockModule('./customization-utils.mjs', () => ({
  repoRoot: 'C:\\NeatapticTS',
  parseArgs: jest.fn(),
  issue: jest.fn((severity, path, message) => ({ severity, path, message })),
  writeReport: jest.fn(),
  listMarkdownFiles: jest.fn(),
  readWorkspaceFile: jest.fn(),
  parseFrontmatter: jest.fn(),
  printUsage: jest.fn(),
  summarizeIssues: jest.fn(),
  extractMarkdownLinks: jest.fn(),
  fileExists: jest.fn(),
}));

jest.unstable_mockModule('node:fs', () => ({
  existsSync: jest.fn(),
  watch: jest.fn(),
}));

jest.unstable_mockModule('node:fs/promises', () => ({
  mkdir: jest.fn(),
  readFile: jest.fn(),
  writeFile: jest.fn(),
  rm: jest.fn(),
  readdir: jest.fn(),
  stat: jest.fn(),
  access: jest.fn(),
  constants: { R_OK: 4, W_OK: 2, F_OK: 0 },
  appendFile: jest.fn(),
  glob: jest.fn(),
}));

let mockUtils;
let mockFs;
let mockFsPromises;

beforeEach(async () => {
  jest.resetModules();
  mockUtils = await import('./customization-utils.mjs');
  mockFs = await import('node:fs');
  mockFsPromises = await import('node:fs/promises');
  jest.clearAllMocks();
});

function captureConsole() {
  const logs = [];
  const errors = [];
  const origLog = console.log;
  const origError = console.error;
  console.log = (...args) => logs.push(args.join(' '));
  console.error = (...args) => errors.push(args.join(' '));
  return {
    logs,
    errors,
    restore() {
      console.log = origLog;
      console.error = origError;
    },
  };
}

const scriptPath = path.resolve(
  'C:\\NeatapticTS\\scripts\\agent-customization\\plan-session-redirect.mjs',
);

async function importModuleWithGuard() {
  const origArgv = process.argv;
  // Ensure argv[1] points to the script for the import.meta.url entry guard,
  // while preserving any flags the test already set in argv[2:].
  process.argv = [origArgv[0] ?? 'node', scriptPath, ...origArgv.slice(2)];
  try {
    await import('./plan-session-redirect.mjs');
  } catch {
    // process.exit may throw
  }
  process.argv = origArgv;
}

describe('plan-session-redirect', () => {
  it('prints help when --help', async () => {
    const origArgv = process.argv;
    process.argv = ['node', scriptPath, '--help'];
    const cap = captureConsole();

    await importModuleWithGuard();

    cap.restore();
    assert.ok(cap.logs.some((l) => l.includes('Plan session redirect')));
  });

  it('writes session override for valid plan path', async () => {
    const origArgv = process.argv;
    process.argv = ['node', scriptPath, '--plan=plans/test.md', '--json'];
    mockFs.existsSync.mockReturnValue(true);
    mockFsPromises.mkdir.mockResolvedValue();
    mockFsPromises.writeFile.mockResolvedValue();
    mockFsPromises.readFile.mockResolvedValue(
      JSON.stringify({ plan_path: 'plans/test.md' }),
    );
    const cap = captureConsole();

    await importModuleWithGuard();

    cap.restore();
    assert.ok(cap.logs.some((l) => l.includes('"pass": true')));
    assert.ok(mockFsPromises.writeFile.mock.calls.length > 0);
  });

  it('clears session override with --clear', async () => {
    const origArgv = process.argv;
    process.argv = ['node', scriptPath, '--clear', '--json'];
    mockFs.existsSync.mockReturnValue(true);
    mockFsPromises.rm.mockResolvedValue();
    const cap = captureConsole();

    await importModuleWithGuard();

    cap.restore();
    const json = JSON.parse(cap.logs[0]);
    assert.strictEqual(json.pass, true);
    assert.strictEqual(json.override_cleared, true);
    assert.strictEqual(json.was_present, true);
  });

  it('clears session override when already absent', async () => {
    const origArgv = process.argv;
    process.argv = ['node', scriptPath, '--clear', '--json'];
    mockFs.existsSync.mockReturnValue(false);
    const cap = captureConsole();

    await importModuleWithGuard();

    cap.restore();
    const json = JSON.parse(cap.logs[0]);
    assert.strictEqual(json.was_present, false);
  });

  it('errors when --plan not provided', async () => {
    const origArgv = process.argv;
    process.argv = ['node', scriptPath, '--json'];
    const cap = captureConsole();

    await importModuleWithGuard();

    cap.restore();
    const json = JSON.parse(cap.logs[0]);
    assert.strictEqual(json.pass, false);
    assert.ok(json.error.includes('Provide --plan'));
  });

  it('errors when plan path escapes plans/', async () => {
    const origArgv = process.argv;
    process.argv = ['node', scriptPath, '--plan=../evil.md', '--json'];
    const cap = captureConsole();

    await importModuleWithGuard();

    cap.restore();
    const json = JSON.parse(cap.logs[0]);
    assert.strictEqual(json.pass, false);
    assert.ok(json.error.includes('must resolve within plans/'));
  });

  it('errors when plan file does not exist', async () => {
    const origArgv = process.argv;
    process.argv = ['node', scriptPath, '--plan=plans/missing.md', '--json'];
    mockFs.existsSync.mockReturnValue(false);
    const cap = captureConsole();

    await importModuleWithGuard();

    cap.restore();
    const json = JSON.parse(cap.logs[0]);
    assert.strictEqual(json.pass, false);
    assert.ok(json.error.includes('Plan file not found'));
  });

  it('outputs text format without --json', async () => {
    const origArgv = process.argv;
    process.argv = ['node', scriptPath, '--plan=plans/test.md'];
    mockFs.existsSync.mockReturnValue(true);
    mockFsPromises.mkdir.mockResolvedValue();
    mockFsPromises.writeFile.mockResolvedValue();
    mockFsPromises.readFile.mockResolvedValue(
      JSON.stringify({ plan_path: 'plans/test.md' }),
    );
    const cap = captureConsole();

    await importModuleWithGuard();

    cap.restore();
    assert.ok(cap.logs.some((l) => l.includes('PASS plan-session-redirect')));
  });

  it('outputs text error format without --json', async () => {
    const origArgv = process.argv;
    process.argv = ['node', scriptPath];
    const cap = captureConsole();

    await importModuleWithGuard();

    cap.restore();
    assert.ok(cap.errors.some((e) => e.includes('Provide --plan')));
  });

  it('uses absolute plan path correctly', async () => {
    const origArgv = process.argv;
    const absPath = path.resolve('C:\\NeatapticTS\\plans\\abs-test.md');
    process.argv = ['node', scriptPath, `--plan=${absPath}`, '--json'];
    mockFs.existsSync.mockReturnValue(true);
    mockFsPromises.mkdir.mockResolvedValue();
    mockFsPromises.writeFile.mockResolvedValue();
    mockFsPromises.readFile.mockResolvedValue(
      JSON.stringify({ plan_path: 'plans/abs-test.md' }),
    );
    const cap = captureConsole();

    await importModuleWithGuard();

    cap.restore();
    const json = JSON.parse(cap.logs[0]);
    assert.strictEqual(json.pass, true);
  });

  it('handles readback mismatch', async () => {
    const origArgv = process.argv;
    process.argv = ['node', scriptPath, '--plan=plans/test.md', '--json'];
    mockFs.existsSync.mockReturnValue(true);
    mockFsPromises.mkdir.mockResolvedValue();
    mockFsPromises.writeFile.mockResolvedValue();
    mockFsPromises.readFile.mockResolvedValue(
      JSON.stringify({ plan_path: 'plans/different.md' }),
    );
    const cap = captureConsole();

    await importModuleWithGuard();

    cap.restore();
    const json = JSON.parse(cap.logs[0]);
    assert.strictEqual(json.pass, false);
    assert.ok(json.error.includes('readback did not match'));
  });

  it('handles non-Error rejection in text mode', async () => {
    const origArgv = process.argv;
    const origExitCode = process.exitCode;
    process.argv = ['node', 'unused', '--plan=plans/test.md'];
    mockFs.existsSync.mockReturnValue(true);
    mockFsPromises.mkdir.mockRejectedValue('string error');
    const cap = captureConsole();

    await importModuleWithGuard();

    cap.restore();
    process.argv = origArgv;
    process.exitCode = origExitCode;
    assert.ok(cap.errors.some((e) => e === 'string error'));
  });

  it('does not call main when argv[1] is falsy', async () => {
    const origArgv = process.argv;
    process.argv = ['node'];
    const cap = captureConsole();

    try {
      await import('./plan-session-redirect.mjs');
    } catch {
      // ignore
    }

    cap.restore();
    process.argv = origArgv;
    assert.strictEqual(cap.logs.length, 0);
  });

  it('does not call main when argv[1] does not match script path', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'other-script.mjs'];
    const cap = captureConsole();

    try {
      await import('./plan-session-redirect.mjs');
    } catch {
      // ignore
    }

    cap.restore();
    process.argv = origArgv;
    assert.strictEqual(cap.logs.length, 0);
  });
});
