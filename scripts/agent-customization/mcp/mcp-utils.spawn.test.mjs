/**
 * @fileoverview Coverage tests for runShellFreeCommand in mcp-utils.mjs.
 * Uses jest.unstable_mockModule to mock node:child_process spawn.
 */

import { jest } from '@jest/globals';
import { EventEmitter } from 'node:events';

jest.unstable_mockModule('node:child_process', () => ({
  spawn: jest.fn(),
}));

let spawnMock;
let runShellFreeCommand;
let MCP_REPO_ROOT;

beforeAll(async () => {
  const childProcess = await import('node:child_process');
  spawnMock = childProcess.spawn;
  const mcpUtils = await import('./mcp-utils.mjs');
  runShellFreeCommand = mcpUtils.runShellFreeCommand;
  MCP_REPO_ROOT = mcpUtils.MCP_REPO_ROOT;
});

beforeEach(() => {
  spawnMock.mockClear();
});

/**
 * Create a mock child process with EventEmitter-based stdout/stderr.
 * @returns {{ child: EventEmitter, stdout: EventEmitter, stderr: EventEmitter }}
 */
function createMockChild() {
  const child = new EventEmitter();
  child.stdout = new EventEmitter();
  child.stderr = new EventEmitter();
  spawnMock.mockReturnValue(child);
  return child;
}

/** Helper to flush microtasks. */
function flush(ms = 10) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

describe('runShellFreeCommand', () => {
  it('spawns node using process.execPath', async () => {
    const child = createMockChild();
    const promise = runShellFreeCommand('node -e "hello"');
    await flush();
    expect(spawnMock).toHaveBeenCalledTimes(1);
    const [executable, argv, options] = spawnMock.mock.calls[0];
    expect(executable).toBe(process.execPath);
    expect(argv).toEqual(['-e', 'hello']);
    expect(options.shell).toBe(false);
    expect(options.cwd).toBe(MCP_REPO_ROOT);
    child.stdout.emit('data', Buffer.from('output'));
    child.emit('close', 0);
    const result = await promise;
    expect(result.exitCode).toBe(0);
    expect(result.stdout).toBe('output');
    expect(result.executable).toBe('node');
  });

  it('spawns npx using cmd.exe on Windows', async () => {
    const child = createMockChild();
    const promise = runShellFreeCommand('npx --version');
    await flush();
    const [executable, argv] = spawnMock.mock.calls[0];
    if (process.platform === 'win32') {
      expect(executable).toBe(process.env.ComSpec ?? 'cmd.exe');
      expect(argv[0]).toBe('/c');
      expect(argv[1].endsWith('npx.cmd')).toBe(true);
      expect(argv[2]).toBe('--version');
    } else {
      expect(executable).toBe('npx');
      expect(argv).toEqual(['--version']);
    }
    child.emit('close', 0);
    await promise;
  });

  it('spawns npm using cmd.exe on Windows', async () => {
    const child = createMockChild();
    const promise = runShellFreeCommand('npm --version');
    await flush();
    const [executable, argv] = spawnMock.mock.calls[0];
    if (process.platform === 'win32') {
      expect(executable).toBe(process.env.ComSpec ?? 'cmd.exe');
      expect(argv[0]).toBe('/c');
      expect(argv[1].endsWith('npm.cmd')).toBe(true);
    }
    child.emit('close', 0);
    await promise;
  });

  it('passes through unknown executable unchanged', async () => {
    const child = createMockChild();
    const promise = runShellFreeCommand('somecmd arg1 arg2');
    await flush();
    const [executable, argv] = spawnMock.mock.calls[0];
    expect(executable).toBe('somecmd');
    expect(argv).toEqual(['arg1', 'arg2']);
    child.emit('close', 0);
    await promise;
  });

  it('captures stderr output', async () => {
    const child = createMockChild();
    const promise = runShellFreeCommand('node -e "test"');
    await flush();
    child.stderr.emit('data', Buffer.from('error output'));
    child.emit('close', 1);
    const result = await promise;
    expect(result.stderr).toBe('error output');
    expect(result.exitCode).toBe(1);
    expect(result.truncated.stderr).toBe(false);
  });

  it('truncates stdout when exceeding maxOutputBytes (single chunk)', async () => {
    const child = createMockChild();
    const promise = runShellFreeCommand('node -e "big"', { maxOutputBytes: 10 });
    await flush();
    child.stdout.emit('data', Buffer.from('a'.repeat(100)));
    child.emit('close', 0);
    const result = await promise;
    expect(result.stdout.length).toBeLessThanOrEqual(10 + '\n[output truncated]'.length);
    expect(result.stdout).toContain('[output truncated]');
    expect(result.truncated.stdout).toBe(true);
  });

  it('truncates stdout with multi-chunk: first fits, second triggers else branch', async () => {
    const child = createMockChild();
    const promise = runShellFreeCommand('node -e "multi"', { maxOutputBytes: 10 });
    await flush();
    // First chunk fits within limit
    child.stdout.emit('data', Buffer.from('12345'));
    // Second chunk: stdoutBytes (5) < maxOutputBytes (10), so it takes the if branch
    child.stdout.emit('data', Buffer.from('67890'));
    // Third chunk: stdoutBytes (10) >= maxOutputBytes, so it takes the else branch
    child.stdout.emit('data', Buffer.from('overflow'));
    child.emit('close', 0);
    const result = await promise;
    expect(result.truncated.stdout).toBe(true);
    expect(result.stdout).toContain('[output truncated]');
  });

  it('truncates stderr when exceeding maxOutputBytes', async () => {
    const child = createMockChild();
    const promise = runShellFreeCommand('node -e "bigerr"', { maxOutputBytes: 5 });
    await flush();
    child.stderr.emit('data', Buffer.from('x'.repeat(50)));
    child.emit('close', 0);
    const result = await promise;
    expect(result.truncated.stderr).toBe(true);
    expect(result.stderr).toContain('[output truncated]');
  });

  it('truncates stderr with multi-chunk else branch', async () => {
    const child = createMockChild();
    const promise = runShellFreeCommand('node -e "multierr"', { maxOutputBytes: 5 });
    await flush();
    child.stderr.emit('data', Buffer.from('ab'));
    child.stderr.emit('data', Buffer.from('cd'));
    child.stderr.emit('data', Buffer.from('ef'));
    child.stderr.emit('data', Buffer.from('overflow-extra'));
    child.emit('close', 0);
    const result = await promise;
    expect(result.truncated.stderr).toBe(true);
    expect(result.stderr).toContain('[output truncated]');
  });

  it('rejects on error event', async () => {
    const child = createMockChild();
    const promise = runShellFreeCommand('node -e "fail"');
    await flush();
    child.emit('error', new Error('spawn failed'));
    await expect(promise).rejects.toThrow('spawn failed');
  });

  it('handles null exitCode (signal kill) by defaulting to 1', async () => {
    const child = createMockChild();
    const promise = runShellFreeCommand('node -e "signal"');
    await flush();
    child.emit('close', null);
    const result = await promise;
    expect(result.exitCode).toBe(1);
  });

  it('uses default output limit when maxOutputBytes not provided', async () => {
    const child = createMockChild();
    const promise = runShellFreeCommand('node -e "test"');
    await flush();
    child.stdout.emit('data', Buffer.from('small'));
    child.emit('close', 0);
    const result = await promise;
    expect(result.stdout).toBe('small');
    expect(result.truncated.stdout).toBe(false);
  });

  it('uses default output limit when maxOutputBytes is 0', async () => {
    const child = createMockChild();
    const promise = runShellFreeCommand('node -e "test"', { maxOutputBytes: 0 });
    await flush();
    child.stdout.emit('data', Buffer.from('small'));
    child.emit('close', 0);
    const result = await promise;
    expect(result.stdout).toBe('small');
  });

  it('uses default output limit when maxOutputBytes is NaN', async () => {
    const child = createMockChild();
    const promise = runShellFreeCommand('node -e "test"', { maxOutputBytes: NaN });
    await flush();
    child.stdout.emit('data', Buffer.from('small'));
    child.emit('close', 0);
    const result = await promise;
    expect(result.stdout).toBe('small');
  });

  it('uses default output limit when maxOutputBytes is negative', async () => {
    const child = createMockChild();
    const promise = runShellFreeCommand('node -e "test"', { maxOutputBytes: -1 });
    await flush();
    child.stdout.emit('data', Buffer.from('small'));
    child.emit('close', 0);
    const result = await promise;
    expect(result.stdout).toBe('small');
  });

  it('returns command string in result', async () => {
    const child = createMockChild();
    const promise = runShellFreeCommand('node -e "test"');
    await flush();
    child.emit('close', 0);
    const result = await promise;
    expect(result.command).toBe('node -e "test"');
  });

  it('returns durationMs as a number', async () => {
    const child = createMockChild();
    const promise = runShellFreeCommand('node -e "test"');
    await flush();
    child.emit('close', 0);
    const result = await promise;
    expect(typeof result.durationMs).toBe('number');
    expect(result.durationMs).toBeGreaterThanOrEqual(0);
  });

  it('handles node.exe executable name on Windows', async () => {
    const child = createMockChild();
    const promise = runShellFreeCommand('node.exe -e "hello"');
    await flush();
    const [executable] = spawnMock.mock.calls[0];
    expect(executable).toBe(process.execPath);
    child.emit('close', 0);
    await promise;
  });

  it('handles npx.cmd executable name on Windows', async () => {
    const child = createMockChild();
    const promise = runShellFreeCommand('npx.cmd --version');
    await flush();
    const [executable, argv] = spawnMock.mock.calls[0];
    if (process.platform === 'win32') {
      expect(executable).toBe(process.env.ComSpec ?? 'cmd.exe');
      expect(argv[1].endsWith('npx.cmd')).toBe(true);
    }
    child.emit('close', 0);
    await promise;
  });

  it('falls back to cmd.exe when ComSpec env is unset (Windows)', async () => {
    const savedComSpec = process.env.ComSpec;
    delete process.env.ComSpec;
    try {
      const child = createMockChild();
      const promise = runShellFreeCommand('npx --version');
      await flush();
      const [executable, argv] = spawnMock.mock.calls[0];
      if (process.platform === 'win32') {
        expect(executable).toBe('cmd.exe');
        expect(argv[0]).toBe('/c');
        expect(argv[1].endsWith('npx.cmd')).toBe(true);
      } else {
        expect(executable).toBe('npx');
      }
      child.emit('close', 0);
      await promise;
    } finally {
      if (savedComSpec !== undefined) {
        process.env.ComSpec = savedComSpec;
      }
    }
  });
});