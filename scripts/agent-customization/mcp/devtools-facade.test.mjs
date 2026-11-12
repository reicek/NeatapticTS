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

// The CDP-readiness law forbids a bare TCP-socket channel check: node:net is
// intercepted for the whole file (nothing under scripts/agent-customization
// imports it at HEAD) so the readiness red home can assert that the exported
// CDP waiter never opens Socket / createConnection / connect.
const netChannelMocks = {
  Socket: jest.fn(),
  createConnection: jest.fn(),
  connect: jest.fn(),
};
jest.unstable_mockModule('node:net', () => ({
  ...netChannelMocks,
  default: { ...netChannelMocks },
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

  // ─────────────────────────────────────────────────────────────────────────────
  // RED HOME (slice P4-S2B-harness-red) — the two failing harness contracts.
  // Both reds FAIL AT HEAD and are NEVER relaxed by P4-S2B-harness-attach: the
  // implementing slice flips them green with ZERO frozen-expectation edits.
  // No pre-existing assertion above is touched (the RED-P4-SHAPE law).
  // ─────────────────────────────────────────────────────────────────────────────

  // RED-P4-S2B-1 — THE ATTACH ARGV LAW.
  describe('browserUrl attach mode', () => {
    // The spawn argv tail (after the resolved executable element). The lazy
    // facade spawn transport may resolve the bare `npx` head to a platform
    // wrapper, but resolveSpawnCommand never rewrites the argv tail, so the
    // tail is the deterministic byte-identity surface for DEFAULT_SPAWN_COMMAND.
    const FROZEN_DEFAULT_SPAWN_ARGV = [
      '-y',
      'chrome-devtools-mcp@1.6.0',
      '--headless=false',
      '--usage-statistics=false',
      '--performance-crux=false',
      '--allow-unrestricted-paths=true',
    ];

    /**
     * Push a constructed facade through the child transport so the real
     * Chrome DevTools MCP server command is spawned exactly once, and hand
     * back the spawned argv (the resolved head excluded).
     *
     * @param {object} createOptions - Options handed to createDevtoolsFacade.
     * @returns {Promise<string[]>} The spawned command argv minus its head.
     */
    async function spawnRealServerCommand(createOptions) {
      const mod = await import('./devtools-facade.mjs');
      const facade = mod.createDevtoolsFacade(createOptions);
      await facade.server.dispatch({
        jsonrpc: '2.0',
        id: 7,
        method: 'tools/call',
        params: {
          name: 'devtools',
          arguments: { operation: 'initialize' },
        },
      });
      const spawnHead = spawnMock.mock.calls[0];
      return spawnHead[1];
    }

    it('appends --browserUrl=<url> to the spawned command when the browserUrl option is set', async () => {
      // Act — construct the facade in attach mode toward the prepared instance.
      const attachArgv = await spawnRealServerCommand({
        browserUrl: 'http://127.0.0.1:9222',
      });
      // Assert — the additive law: the frozen prefix is untouched and the
      // attach flag is carried on the spawned command.
      expect(attachArgv.slice(0, FROZEN_DEFAULT_SPAWN_ARGV.length)).toEqual(
        FROZEN_DEFAULT_SPAWN_ARGV,
      );
      expect(attachArgv).toContain('--browserUrl=http://127.0.0.1:9222');
    });

    it('keeps the default spawned command byte-identical to DEFAULT_SPAWN_COMMAND when browserUrl is not set', async () => {
      // Act — the frozen no-regression guard (green at HEAD by design).
      const defaultArgv = await spawnRealServerCommand({});
      // Assert — byte-identical to the frozen DEFAULT_SPAWN_COMMAND tail.
      expect(defaultArgv).toEqual(FROZEN_DEFAULT_SPAWN_ARGV);
    });
  });

  // RED-P4-S2B-2 — THE CDP /json/version READINESS LAW.
  describe('waitForCdpReadiness (the CDP readiness law)', () => {
    const CDP_VERSION_URL = 'http://127.0.0.1:9222/json/version';
    const CDP_BROWSER_IDENTITY = {
      Browser: 'Chrome/141.0.0.0',
      'Protocol-Version': '1.3',
      webSocketDebuggerUrl: 'ws://127.0.0.1:9222/devtools/browser/fixed-id',
    };

    /**
     * Deterministic duck-typed /json/version response stub (no live network).
     *
     * @param {number} status - HTTP status of the probe answer.
     * @param {unknown} body - JSON body of the probe answer.
     */
    const cdpResponse = (status, body) => {
      const text = JSON.stringify(body);
      return {
        status,
        ok: status >= 200 && status < 300,
        text: async () => text,
        json: async () => body,
      };
    };

    /**
     * Await the exported readiness waiter and capture its rejection instead
     * of throwing, so the red contract can assert the NAMED timeout error.
     *
     * @param {string} url - The /json/version endpoint under test.
     * @param {{ timeoutMs: number, pollIntervalMs: number }} budgetOptions - Poll budget.
     * @returns {Promise<null | Error>} The rejection, or null when it resolved.
     */
    async function capturedReadinessRejection(url, budgetOptions) {
      const mod = await import('./devtools-facade.mjs');
      try {
        await mod.waitForCdpReadiness(url, budgetOptions);
      } catch (error) {
        return error;
      }
      return null;
    }

    it('polls the /json/version HTTP channel until READY and never substitutes a bare TCP-socket check', async () => {
      // Arrange — first probe not ready, second probe carries the identity.
      let probeIndex = 0;
      const probeUrls = [];
      jest.spyOn(globalThis, 'fetch').mockImplementation(async (probeInput) => {
        probeUrls.push(
          String(
            typeof probeInput === 'string'
              ? probeInput
              : probeInput && probeInput.url,
          ),
        );
        probeIndex += 1;
        return probeIndex === 1
          ? cdpResponse(503, {})
          : cdpResponse(200, CDP_BROWSER_IDENTITY);
      });
      const netModule = await import('node:net');
      const mod = await import('./devtools-facade.mjs');

      // Act
      const readiness = await mod.waitForCdpReadiness(CDP_VERSION_URL, {
        timeoutMs: 60,
        pollIntervalMs: 5,
      });

      // Assert — the READY identity arrives through the HTTP channel only.
      expect(readiness).toEqual(CDP_BROWSER_IDENTITY);
      expect(probeUrls).toEqual([CDP_VERSION_URL, CDP_VERSION_URL]);
      const netTouchCount =
        netModule.Socket.mock.calls.length +
        netModule.createConnection.mock.calls.length +
        netModule.connect.mock.calls.length;
      expect(netTouchCount).toBe(0);
    });

    it('does not accept a 200 without the browser identity as READY within the budget', async () => {
      // Arrange — the endpoint serves 200 with a body lacking the identity.
      jest
        .spyOn(globalThis, 'fetch')
        .mockImplementation(async () =>
          cdpResponse(200, { note: 'not the CDP version identity' }),
        );

      // Act
      const rejection = await capturedReadinessRejection(CDP_VERSION_URL, {
        timeoutMs: 60,
        pollIntervalMs: 5,
      });

      // Assert — a bare 200 is never READY, so the waiter must not resolve.
      expect(rejection?.name).toBe('CdpReadinessTimeoutError');
    });

    it('rejects with the named CdpReadinessTimeoutError when READY never arrives within the budget', async () => {
      // Arrange — the CDP channel never becomes ready within the budget.
      jest
        .spyOn(globalThis, 'fetch')
        .mockImplementation(async () => cdpResponse(503, {}));

      // Act
      const rejection = await capturedReadinessRejection(CDP_VERSION_URL, {
        timeoutMs: 60,
        pollIntervalMs: 5,
      });

      // Assert — the timeout is a NAMED error, not a generic failure.
      expect(rejection).toBeInstanceOf(Error);
      expect(rejection.name).toBe('CdpReadinessTimeoutError');
    });
  });

  // ─────────────────────────────────────────────────────────────────────────────
  // GREEN COMPANIONS (slice P4-S2B-harness-attach) — the attach-flow guards.
  // The RED homes above are BYTE-FROZEN; these two its EXTEND beside them
  // (the RED-P4-SHAPE law) and measure the probe-THEN-attach ordering plus
  // the default-path probe-free guarantee end-to-end.
  // ─────────────────────────────────────────────────────────────────────────────

  describe('attach flow (the green companions)', () => {
    const COMPANION_DEFAULT_SPAWN_ARGV = [
      '-y',
      'chrome-devtools-mcp@1.6.0',
      '--headless=false',
      '--usage-statistics=false',
      '--performance-crux=false',
      '--allow-unrestricted-paths=true',
    ];
    const COMPANION_CDP_VERSION_URL = 'http://127.0.0.1:9222/json/version';
    const COMPANION_CDP_BROWSER_IDENTITY = {
      Browser: 'Chrome/141.0.0.0',
      'Protocol-Version': '1.3',
      webSocketDebuggerUrl: 'ws://127.0.0.1:9222/devtools/browser/fixed-id',
    };

    /**
     * Deterministic duck-typed /json/version response stub (a fresh
     * companion-local copy beside the frozen red home — this describe is
     * self-contained and shares NOTHING with the frozen scopes).
     *
     * @param {number} status - HTTP status of the probe answer.
     * @param {unknown} body - JSON body of the probe answer.
     */
    const cdpResponse = (status, body) => {
      const text = JSON.stringify(body);
      return {
        status,
        ok: status >= 200 && status < 300,
        text: async () => text,
        json: async () => body,
      };
    };

    /**
     * Companion-local equivalent of the frozen red-home helper: construct the
     * facade, push it through the standard first tools/call (the transport's
     * spawn trigger), and hand back the spawned argv tail.
     *
     * @param {object} createOptions - Options handed to createDevtoolsFacade.
     * @returns {Promise<string[]>} The spawned command argv minus its head.
     */
    async function spawnRealServerCommandForCompanion(createOptions) {
      const mod = await import('./devtools-facade.mjs');
      const facade = mod.createDevtoolsFacade(createOptions);
      await facade.server.dispatch({
        jsonrpc: '2.0',
        id: 7,
        method: 'tools/call',
        params: {
          name: 'devtools',
          arguments: { operation: 'initialize' },
        },
      });
      return spawnMock.mock.calls[0][1];
    }

    it('probes the /json/version HTTP channel BEFORE the MCP spawn in attach mode', async () => {
      // Arrange — the prepared instance answers READY on the first probe.
      const probeUrls = [];
      jest.spyOn(globalThis, 'fetch').mockImplementation(async (probeInput) => {
        probeUrls.push(
          String(
            typeof probeInput === 'string'
              ? probeInput
              : probeInput && probeInput.url,
          ),
        );
        return cdpResponse(200, COMPANION_CDP_BROWSER_IDENTITY);
      });

      // Act — construct the facade in attach mode and push the standard
      // dispatch through (the gated wrapper must probe, then let the
      // transport spawn the MCP server).
      const spawnResult = await spawnRealServerCommandForCompanion({
        browserUrl: 'http://127.0.0.1:9222',
      });

      // Assert — the probe-THEN-attach ordering (by invocation order), the
      // single probed endpoint, and the additive attach flag on the spawn.
      const firstProbeOrder = globalThis.fetch.mock.invocationCallOrder[0];
      const firstSpawnOrder = spawnMock.mock.invocationCallOrder[0];
      expect(firstProbeOrder).toBeLessThan(firstSpawnOrder);
      expect(probeUrls).toEqual([COMPANION_CDP_VERSION_URL]);
      expect(spawnResult).toContain('--browserUrl=http://127.0.0.1:9222');
    });

    it('never probes the CDP channel when browserUrl is not set (launch mode stays probe-free)', async () => {
      // Arrange — a fetch spy that would flag even ONE accidental probe.
      const fetchSpy = jest.spyOn(globalThis, 'fetch');

      // Act — the DEFAULT launch-mode facade with the standard dispatch.
      const spawnResult = await spawnRealServerCommandForCompanion({});

      // Assert — zero CDP HTTP probes; the default argv stays byte-identical.
      expect(fetchSpy.mock.calls.length).toBe(0);
      expect(spawnResult).toEqual(COMPANION_DEFAULT_SPAWN_ARGV);
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
