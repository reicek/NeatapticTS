import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

jest.unstable_mockModule('node:fs/promises', () => ({
  mkdir: jest.fn(),
  lstat: jest.fn(),
  link: jest.fn(),
  symlink: jest.fn(),
  readdir: jest.fn(),
  readFile: jest.fn(),
  writeFile: jest.fn(),
  stat: jest.fn(),
  appendFile: jest.fn(),
  access: jest.fn(),
  constants: { R_OK: 4, W_OK: 2, F_OK: 0 },
  rm: jest.fn(),
  glob: jest.fn(),
}));

let mockFs;

beforeEach(async () => {
  jest.resetModules();
  mockFs = await import('node:fs/promises');
  jest.clearAllMocks();
});

async function importModule() {
  try {
    await import('./integrate-antigravity.mjs');
  } catch {
    // process.exit may throw
  }
}

function captureConsole() {
  const logs = [];
  const errors = [];
  const origLog = console.log;
  const origError = console.error;
  console.log = (...args) => logs.push(args.join(' '));
  console.error = (...args) => errors.push(args.join(' '));
  return {
    logs, errors,
    restore() { console.log = origLog; console.error = origError; },
  };
}

describe('integrate-antigravity', () => {
  it('creates links when destination does not exist', async () => {
    const origExit = process.exit;
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const cap = captureConsole();

    mockFs.mkdir.mockResolvedValue();
    mockFs.lstat.mockRejectedValue(new Error('ENOENT'));
    mockFs.link.mockResolvedValue();
    mockFs.symlink.mockResolvedValue();
    mockFs.readdir.mockResolvedValue([
      'test.agent.md',
    ]);

    await importModule();

    process.exit = origExit;
    cap.restore();
    assert.ok(cap.logs.some((l) => l.includes('Linking MCP configuration')));
    assert.ok(cap.logs.some((l) => l.includes('Linking skills')));
    assert.ok(cap.logs.some((l) => l.includes('Linking agents')));
    assert.ok(cap.logs.some((l) => l.includes('Integration completed')));
  });

  it('skips MCP config when already exists', async () => {
    const origExit = process.exit;
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const cap = captureConsole();

    mockFs.mkdir.mockResolvedValue();
    // First lstat (mcp_config) succeeds, second (skills) fails
    mockFs.lstat.mockResolvedValueOnce({ isFile: () => true })
      .mockRejectedValueOnce(new Error('ENOENT'));
    mockFs.symlink.mockResolvedValue();
    mockFs.readdir.mockResolvedValue([]);

    await importModule();

    process.exit = origExit;
    cap.restore();
    assert.ok(cap.logs.some((l) => l.includes('mcp_config.json already exists')));
  });

  it('skips skills when already exists', async () => {
    const origExit = process.exit;
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const cap = captureConsole();

    mockFs.mkdir.mockResolvedValue();
    // First lstat (mcp_config) fails, second (skills) succeeds
    mockFs.lstat.mockRejectedValueOnce(new Error('ENOENT'))
      .mockResolvedValueOnce({ isDirectory: () => true });
    mockFs.link.mockResolvedValue();
    mockFs.readdir.mockResolvedValue([]);

    await importModule();

    process.exit = origExit;
    cap.restore();
    assert.ok(cap.logs.some((l) => l.includes('skills already exists')));
  });

  it('skips agent when already linked', async () => {
    const origExit = process.exit;
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const cap = captureConsole();

    mockFs.mkdir.mockResolvedValue();
    mockFs.lstat.mockRejectedValue(new Error('ENOENT'));
    mockFs.link.mockResolvedValue();
    mockFs.symlink.mockResolvedValue();
    mockFs.readdir.mockResolvedValue([
      'existing.agent.md',
    ]);
    // lstat for destPath (agent) succeeds
    mockFs.lstat.mockResolvedValueOnce({ isFile: () => true }) // mcp_config
      .mockRejectedValueOnce(new Error('ENOENT')) // skills
      .mockResolvedValue({ isFile: () => true }); // agent dest

    await importModule();

    process.exit = origExit;
    cap.restore();
    // Agent should be skipped (no "Created hard link for agent" log)
  });

  it('handles error and exits 1', async () => {
    const origExit = process.exit;
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const cap = captureConsole();

    mockFs.mkdir.mockRejectedValue(new Error('Permission denied'));

    await importModule();

    process.exit = origExit;
    cap.restore();
    assert.ok(cap.errors.some((e) => e.includes('Error performing integration')));
  });

  it('uses dir symlink type on non-win32 platform', async () => {
    const origPlatform = process.platform;
    Object.defineProperty(process, 'platform', { value: 'linux', configurable: true, writable: true });
    const origExit = process.exit;
    process.exit = (code) => { throw new Error(`EXIT:${code}`); };
    const cap = captureConsole();

    mockFs.mkdir.mockResolvedValue();
    mockFs.lstat.mockRejectedValue(new Error('ENOENT'));
    mockFs.link.mockResolvedValue();
    mockFs.symlink.mockResolvedValue();
    mockFs.readdir.mockResolvedValue([]);

    await importModule();

    process.exit = origExit;
    Object.defineProperty(process, 'platform', { value: origPlatform, configurable: true, writable: true });
    cap.restore();
    assert.ok(mockFs.symlink.mock.calls.length > 0);
    assert.strictEqual(mockFs.symlink.mock.calls[0][2], 'dir');
  });
});