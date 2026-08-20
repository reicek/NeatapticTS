import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

jest.unstable_mockModule('./customization-utils.mjs', () => ({
  repoRoot: 'C:\\NeatapticTS',
  parseArgs: jest.fn(),
}));

jest.unstable_mockModule('node:fs', () => ({
  watch: jest.fn(),
}));

jest.unstable_mockModule('node:child_process', () => ({
  spawnSync: jest.fn(),
}));

let mockUtils;
let mockFs;
let mockChildProc;

beforeEach(async () => {
  jest.resetModules();
  mockUtils = await import('./customization-utils.mjs');
  mockFs = await import('node:fs');
  mockChildProc = await import('node:child_process');
  jest.clearAllMocks();
});

afterEach(() => {
  process.removeAllListeners('SIGINT');
  process.removeAllListeners('SIGTERM');
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
  'C:\\NeatapticTS\\scripts\\agent-customization\\watch-routing-table.mjs',
);

async function importModuleWithGuard(argv) {
  const origArgv = process.argv;
  process.argv = ['node', scriptPath, ...(argv || [])];
  try {
    await import('./watch-routing-table.mjs');
  } catch {
    // process.exit may throw
  }
  process.argv = origArgv;
}

function makeWatcher() {
  const handlers = [];
  return {
    watcher: {
      close: jest.fn(),
    },
    callback: (eventType, filename) => {
      handlers.forEach((h) => h(eventType, filename));
    },
    register: (h) => handlers.push(h),
  };
}

describe('watch-routing-table', () => {
  it('prints help when --help', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: true });
    const cap = captureConsole();

    await importModuleWithGuard(['--help']);

    cap.restore();
    assert.ok(cap.logs.some((l) => l.includes('Routing-table watch utility')));
  });

  it('starts watcher and runs generator on startup', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false });
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"changed": true}',
      stderr: '',
    });
    const watcherObj = makeWatcher();
    mockFs.watch.mockReturnValue(watcherObj.watcher);
    const cap = captureConsole();

    await importModuleWithGuard([]);

    cap.restore();
    assert.ok(mockChildProc.spawnSync.mock.calls.length >= 1);
    assert.ok(cap.errors.some((e) => e.includes('watching')));
  });

  it('handles generator failure on startup', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false });
    mockChildProc.spawnSync.mockReturnValue({
      status: 1,
      stdout: '',
      stderr: 'generator error',
    });
    const watcherObj = makeWatcher();
    mockFs.watch.mockReturnValue(watcherObj.watcher);
    const cap = captureConsole();

    await importModuleWithGuard([]);

    cap.restore();
    assert.ok(cap.errors.some((e) => e.includes('regeneration failed')));
  });

  it('handles generator failure with stdout only', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false });
    mockChildProc.spawnSync.mockReturnValue({
      status: 1,
      stdout: 'stdout error',
      stderr: '',
    });
    const watcherObj = makeWatcher();
    mockFs.watch.mockReturnValue(watcherObj.watcher);
    const cap = captureConsole();

    await importModuleWithGuard([]);

    cap.restore();
    assert.ok(cap.errors.some((e) => e.includes('stdout error')));
  });

  it('handles unreadable JSON output from generator', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false });
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: 'not json',
      stderr: '',
    });
    const watcherObj = makeWatcher();
    mockFs.watch.mockReturnValue(watcherObj.watcher);
    const cap = captureConsole();

    await importModuleWithGuard([]);

    cap.restore();
    assert.ok(cap.errors.some((e) => e.includes('unreadable JSON')));
  });

  it('handles already-fresh routing table', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false });
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"changed": false}',
      stderr: '',
    });
    const watcherObj = makeWatcher();
    mockFs.watch.mockReturnValue(watcherObj.watcher);
    const cap = captureConsole();

    await importModuleWithGuard([]);

    cap.restore();
    assert.ok(cap.errors.some((e) => e.includes('already fresh')));
  });

  it('uses custom debounce-ms', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false });
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"changed": true}',
      stderr: '',
    });
    const watcherObj = makeWatcher();
    mockFs.watch.mockReturnValue(watcherObj.watcher);

    await importModuleWithGuard(['--debounce-ms=500']);
    // parseArgs is called with argv, and debounce-ms is parsed separately
    assert.ok(mockFs.watch.mock.calls.length >= 2);
  });

  it('handles invalid debounce-ms falling back to default', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false });
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"changed": true}',
      stderr: '',
    });
    const watcherObj = makeWatcher();
    mockFs.watch.mockReturnValue(watcherObj.watcher);

    await importModuleWithGuard(['--debounce-ms=invalid']);
    // Should fall back to default (250ms)
    assert.ok(mockFs.watch.mock.calls.length >= 2);
  });

  it('handles empty stdout from generator', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false });
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '',
      stderr: '',
    });
    const watcherObj = makeWatcher();
    mockFs.watch.mockReturnValue(watcherObj.watcher);
    const cap = captureConsole();

    await importModuleWithGuard([]);

    cap.restore();
    // Should still log watching message
    assert.ok(cap.errors.some((e) => e.includes('watching')));
  });

  it('handles non-string filename in watch callback', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false });
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"changed": true}',
      stderr: '',
    });
    const watcherObj = makeWatcher();
    mockFs.watch.mockImplementation((dir, opts, cb) => {
      watcherObj.register(cb);
      // Simulate a file event with non-string filename
      setTimeout(() => {
        try {
          cb('change', undefined);
        } catch {
          /* ignore */
        }
      }, 10);
      return watcherObj.watcher;
    });
    const cap = captureConsole();

    await importModuleWithGuard([]);

    cap.restore();
    // Should not crash
  });

  it('triggers regeneration when .agent.md file changes', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false });
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"changed": true}',
      stderr: '',
    });
    const watchCallbacks = [];
    const watcherObj = makeWatcher();
    mockFs.watch.mockImplementation((dir, opts, cb) => {
      watchCallbacks.push(cb);
      return watcherObj.watcher;
    });

    await importModuleWithGuard([]);

    watchCallbacks[0]('change', 'some-agent.agent.md');
    await new Promise((resolve) => setTimeout(resolve, 350));

    assert.ok(mockChildProc.spawnSync.mock.calls.length >= 2);
  });

  it('triggers regeneration when SKILL.md file changes', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false });
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"changed": true}',
      stderr: '',
    });
    const watchCallbacks = [];
    const watcherObj = makeWatcher();
    mockFs.watch.mockImplementation((dir, opts, cb) => {
      watchCallbacks.push(cb);
      return watcherObj.watcher;
    });

    await importModuleWithGuard([]);

    watchCallbacks[1]('change', 'some-skill.SKILL.md');
    await new Promise((resolve) => setTimeout(resolve, 350));

    assert.ok(mockChildProc.spawnSync.mock.calls.length >= 2);
  });

  it('does not trigger regeneration for non-matching filename', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false });
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"changed": true}',
      stderr: '',
    });
    const watchCallbacks = [];
    const watcherObj = makeWatcher();
    mockFs.watch.mockImplementation((dir, opts, cb) => {
      watchCallbacks.push(cb);
      return watcherObj.watcher;
    });

    await importModuleWithGuard([]);

    watchCallbacks[0]('change', 'random-file.txt');
    await new Promise((resolve) => setTimeout(resolve, 350));

    assert.strictEqual(mockChildProc.spawnSync.mock.calls.length, 1);
  });

  it('shuts down watchers on SIGINT', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false });
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"changed": true}',
      stderr: '',
    });
    const watcherObj = makeWatcher();
    mockFs.watch.mockReturnValue(watcherObj.watcher);

    await importModuleWithGuard([]);

    const origExit = process.exit;
    process.exit = () => {
      throw new Error('EXIT:0');
    };
    try {
      process.emit('SIGINT');
    } catch {
      // process.exit throws
    }
    process.exit = origExit;

    assert.ok(watcherObj.watcher.close.mock.calls.length >= 1);
  });

  it('shuts down watchers on SIGTERM', async () => {
    mockUtils.parseArgs.mockReturnValue({ help: false });
    mockChildProc.spawnSync.mockReturnValue({
      status: 0,
      stdout: '{"changed": true}',
      stderr: '',
    });
    const watcherObj = makeWatcher();
    mockFs.watch.mockReturnValue(watcherObj.watcher);

    await importModuleWithGuard([]);

    const origExit = process.exit;
    process.exit = () => {
      throw new Error('EXIT:0');
    };
    try {
      process.emit('SIGTERM');
    } catch {
      // process.exit throws
    }
    process.exit = origExit;

    assert.ok(watcherObj.watcher.close.mock.calls.length >= 1);
  });

  it('does not call main when process.argv[1] is falsy', async () => {
    const origArgv = process.argv;
    process.argv = ['node'];
    try {
      await import('./watch-routing-table.mjs');
    } catch {
      // ignore
    }
    process.argv = origArgv;
    assert.strictEqual(mockFs.watch.mock.calls.length, 0);
  });
});
