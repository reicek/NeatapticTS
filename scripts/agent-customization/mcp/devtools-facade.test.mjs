import {
  describe,
  it,
  expect,
  beforeEach,
  afterEach,
  jest,
} from '@jest/globals';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const FACADE_PATH = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  'devtools-facade.mjs',
);

// jest.unstable_mockModule persists across jest.resetModules() — once mocked,
// stays mocked for all subsequent imports in this test file.
jest.unstable_mockModule('node:child_process', () => ({
  spawn: jest.fn(() => ({ pid: 12345, unref: jest.fn() })),
  spawnSync: jest.fn(() => ({ status: 0, stdout: '', stderr: '' })),
  execSync: jest.fn(() => ''),
}));

let spawnMock, spawnSyncMock, runFacadeMainMock, createLazyFacadeMock;

beforeEach(async () => {
  jest.resetModules();
  jest.clearAllMocks();
  const cp = await import('node:child_process');
  spawnMock = cp.spawn;
  spawnSyncMock = cp.spawnSync;
});

afterEach(() => {
  jest.restoreAllMocks();
});

describe('devtools-facade', () => {
  describe('launchVisibleChrome', () => {
    it('uses provided options for chrome path, port, and user data dir', async () => {
      const mod = await import('./devtools-facade.mjs');
      const proc = mod.launchVisibleChrome('http://localhost:3000', {
        chromePath: '/custom/chrome',
        remotePort: 9333,
        userDataDir: '/custom/profile',
      });
      expect(spawnMock).toHaveBeenCalledTimes(1);
      const [cmd, args, opts] = spawnMock.mock.calls[0];
      expect(cmd).toBe('/custom/chrome');
      expect(args).toContain('--remote-debugging-port=9333');
      expect(args).toContain('--user-data-dir=/custom/profile');
      expect(args).toContain('http://localhost:3000');
      expect(opts.detached).toBe(true);
      expect(opts.stdio).toBe('ignore');
      expect(proc.pid).toBe(12345);
    });

    it('falls back to env vars when options not provided', async () => {
      const origChromePath = process.env.CHROME_PATH;
      const origUserDataDir = process.env.CHROME_USER_DATA_DIR;
      process.env.CHROME_PATH = '/env/chrome';
      process.env.CHROME_USER_DATA_DIR = '/env/profile';
      try {
        const mod = await import('./devtools-facade.mjs');
        mod.launchVisibleChrome('http://localhost:8080', {});
        const [cmd, args] = spawnMock.mock.calls[0];
        expect(cmd).toBe('/env/chrome');
        expect(args).toContain('--remote-debugging-port=9222');
        expect(args).toContain('--user-data-dir=/env/profile');
        expect(args).toContain('http://localhost:8080');
      } finally {
        if (origChromePath !== undefined) {
          process.env.CHROME_PATH = origChromePath;
        } else {
          delete process.env.CHROME_PATH;
        }
        if (origUserDataDir !== undefined) {
          process.env.CHROME_USER_DATA_DIR = origUserDataDir;
        } else {
          delete process.env.CHROME_USER_DATA_DIR;
        }
      }
    });

    it('uses Windows defaults when no options or env vars', async () => {
      const origChromePath = process.env.CHROME_PATH;
      const origUserDataDir = process.env.CHROME_USER_DATA_DIR;
      delete process.env.CHROME_PATH;
      delete process.env.CHROME_USER_DATA_DIR;
      try {
        const mod = await import('./devtools-facade.mjs');
        mod.launchVisibleChrome(undefined);
        const [cmd, args] = spawnMock.mock.calls[0];
        // On Windows the default is 'chrome.exe'
        if (process.platform === 'win32') {
          expect(cmd).toBe('chrome.exe');
          expect(args).toContain('--user-data-dir=C:\\temp\\chrome-debug');
        } else {
          expect(cmd).toBe('google-chrome');
          expect(args).toContain('--user-data-dir=/tmp/chrome-debug');
        }
        expect(args).toContain('--remote-debugging-port=9222');
      } finally {
        if (origChromePath !== undefined) {
          process.env.CHROME_PATH = origChromePath;
        } else {
          delete process.env.CHROME_PATH;
        }
        if (origUserDataDir !== undefined) {
          process.env.CHROME_USER_DATA_DIR = origUserDataDir;
        } else {
          delete process.env.CHROME_USER_DATA_DIR;
        }
      }
    });

    it('uses non-Windows defaults on linux', async () => {
      const origPlatform = process.platform;
      const origChromePath = process.env.CHROME_PATH;
      const origUserDataDir = process.env.CHROME_USER_DATA_DIR;
      delete process.env.CHROME_PATH;
      delete process.env.CHROME_USER_DATA_DIR;
      Object.defineProperty(process, 'platform', {
        value: 'linux',
        configurable: true,
      });
      try {
        const mod = await import('./devtools-facade.mjs');
        mod.launchVisibleChrome('http://localhost:4000', {});
        const [cmd, args] = spawnMock.mock.calls[0];
        expect(cmd).toBe('google-chrome');
        expect(args).toContain('--user-data-dir=/tmp/chrome-debug');
        expect(args).toContain('http://localhost:4000');
      } finally {
        Object.defineProperty(process, 'platform', {
          value: origPlatform,
          configurable: true,
        });
        if (origChromePath !== undefined) {
          process.env.CHROME_PATH = origChromePath;
        } else {
          delete process.env.CHROME_PATH;
        }
        if (origUserDataDir !== undefined) {
          process.env.CHROME_USER_DATA_DIR = origUserDataDir;
        } else {
          delete process.env.CHROME_USER_DATA_DIR;
        }
      }
    });
  });

  describe('createDevtoolsFacade', () => {
    it('creates a facade with devtools config', async () => {
      const mod = await import('./devtools-facade.mjs');
      const facade = mod.createDevtoolsFacade();
      expect(facade).toBeDefined();
      expect(typeof facade).toBe('object');
    });
  });

  // MUST BE LAST TEST — mocks lazy-facade-core.mjs which affects module loading
  it('runs runFacadeMain when invoked as main entry point', async () => {
    runFacadeMainMock = jest.fn();
    createLazyFacadeMock = jest.fn(() => ({
      name: 'devtools',
      tools: [],
      connect: jest.fn(),
    }));
    jest.unstable_mockModule('./lazy-facade-core.mjs', () => ({
      createLazyFacade: createLazyFacadeMock,
      runFacadeMain: runFacadeMainMock,
    }));

    const origArgv1 = process.argv[1];
    process.argv[1] = FACADE_PATH;
    try {
      jest.resetModules();
      // Re-import with the mock in place and argv[1] pointing to devtools-facade.mjs
      await import('./devtools-facade.mjs');
      expect(runFacadeMainMock.mock.calls.length).toBeGreaterThan(0);
      const config = runFacadeMainMock.mock.calls[0][0];
      expect(config.name).toBe('devtools');
    } finally {
      process.argv[1] = origArgv1;
    }
  });
});
