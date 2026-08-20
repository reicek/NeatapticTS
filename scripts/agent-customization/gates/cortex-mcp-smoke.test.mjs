/**
 * @module cortex-mcp-smoke.test
 * @description Coverage tests for cortex-mcp-smoke.mjs.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const REPO_ROOT = path.resolve();
const GATE_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/gates/cortex-mcp-smoke.mjs',
);

let mockExistsSync;
let mockStatSync;
let mockResolveDatabasePath;
let mockCreateServer;
let mockInvokeServerRequest;

jest.unstable_mockModule('node:fs', () => ({
  existsSync: (...args) => mockExistsSync(...args),
  statSync: (...args) => mockStatSync(...args),
}));

jest.unstable_mockModule('../mcp/mcp-utils.mjs', () => ({
  invokeServerRequest: (...args) => mockInvokeServerRequest(...args),
  MCP_REPO_ROOT: REPO_ROOT,
}));

jest.unstable_mockModule('../../mcp-semantic/tools/cortex-db.mjs', () => ({
  CORTEX_FIX_HINT: 'Run: node rag-index/build-index.mjs',
  resolveDatabasePath: (...args) => mockResolveDatabasePath(...args),
}));

jest.unstable_mockModule('../../mcp-semantic/repo-cortex-mcp.mjs', () => ({
  createRepoCortexMcpServer: (...args) => mockCreateServer(...args),
}));

async function withArgv(argv, fn) {
  const original = process.argv;
  process.argv = argv;
  try {
    return await fn();
  } finally {
    process.argv = original;
  }
}

async function importGate() {
  jest.resetModules();
  return await withArgv(
    [process.execPath, 'dummy'],
    () => import('./cortex-mcp-smoke.mjs'),
  );
}

async function importGateMain(argv) {
  const logs = [];
  const originalLog = console.log;
  console.log = (...args) => logs.push(args.map(String).join(' '));
  try {
    jest.resetModules();
    await withArgv([process.execPath, GATE_PATH, ...argv], async () => {
      await import('./cortex-mcp-smoke.mjs');
      await new Promise((r) => setTimeout(r, 200));
    });
  } finally {
    console.log = originalLog;
  }
  return logs;
}

function setupPassingMocks() {
  mockExistsSync = () => true;
  mockStatSync = () => ({ size: 100 });
  mockResolveDatabasePath = () => '/fake/db.sqlite';
  mockCreateServer = () => ({});
  let callCount = 0;
  mockInvokeServerRequest = async () => {
    callCount++;
    if (callCount === 1) {
      return { isError: false, structuredContent: { total_chunks: 10 } };
    }
    return {
      isError: false,
      structuredContent: { results: [{ id: 'chunk1' }] },
    };
  };
}

function setupFailingMocks() {
  mockExistsSync = () => false;
  mockStatSync = () => ({ size: 0 });
  mockResolveDatabasePath = () => '/fake/db.sqlite';
  mockCreateServer = () => ({});
  mockInvokeServerRequest = async () => ({
    isError: true,
    structuredContent: null,
  });
}

describe('cortex-mcp-smoke', () => {
  let originalExitCode;

  beforeEach(() => {
    setupPassingMocks();
    originalExitCode = process.exitCode;
    jest.resetModules();
  });

  afterEach(() => {
    process.exitCode = originalExitCode ?? 0;
  });

  describe('runCortexMcpSmoke', () => {
    it('fails when database does not exist', async () => {
      mockExistsSync = () => false;
      const { runCortexMcpSmoke } = await importGate();
      const result = await runCortexMcpSmoke({});
      assert.equal(result.pass, false);
      assert.ok(result.evidence.message.includes('missing or empty'));
      assert.equal(result.fixHint, 'Run: node rag-index/build-index.mjs');
    });

    it('fails when database is empty (size 0)', async () => {
      mockExistsSync = () => true;
      mockStatSync = () => ({ size: 0 });
      const { runCortexMcpSmoke } = await importGate();
      const result = await runCortexMcpSmoke({});
      assert.equal(result.pass, false);
      assert.ok(result.evidence.message.includes('missing or empty'));
    });

    it('fails when index_stats returns error', async () => {
      mockExistsSync = () => true;
      mockStatSync = () => ({ size: 100 });
      mockResolveDatabasePath = () => '/fake/db.sqlite';
      mockCreateServer = () => ({});
      mockInvokeServerRequest = async () => ({
        isError: true,
        structuredContent: null,
      });
      const { runCortexMcpSmoke } = await importGate();
      const result = await runCortexMcpSmoke({});
      assert.equal(result.pass, false);
      assert.ok(result.evidence.message.includes('index_stats'));
    });

    it('fails when stats is null/undefined', async () => {
      mockExistsSync = () => true;
      mockStatSync = () => ({ size: 100 });
      mockResolveDatabasePath = () => '/fake/db.sqlite';
      mockCreateServer = () => ({});
      mockInvokeServerRequest = async () => ({
        isError: false,
        structuredContent: null,
      });
      const { runCortexMcpSmoke } = await importGate();
      const result = await runCortexMcpSmoke({});
      assert.equal(result.pass, false);
    });

    it('fails when total_chunks < 1', async () => {
      mockExistsSync = () => true;
      mockStatSync = () => ({ size: 100 });
      mockResolveDatabasePath = () => '/fake/db.sqlite';
      mockCreateServer = () => ({});
      mockInvokeServerRequest = async () => ({
        isError: false,
        structuredContent: { total_chunks: 0 },
      });
      const { runCortexMcpSmoke } = await importGate();
      const result = await runCortexMcpSmoke({});
      assert.equal(result.pass, false);
    });

    it('fails when search_corpus returns error', async () => {
      mockExistsSync = () => true;
      mockStatSync = () => ({ size: 100 });
      mockResolveDatabasePath = () => '/fake/db.sqlite';
      mockCreateServer = () => ({});
      let callCount = 0;
      mockInvokeServerRequest = async () => {
        callCount++;
        if (callCount === 1) {
          return { isError: false, structuredContent: { total_chunks: 10 } };
        }
        return { isError: true, structuredContent: null };
      };
      const { runCortexMcpSmoke } = await importGate();
      const result = await runCortexMcpSmoke({});
      assert.equal(result.pass, false);
      assert.ok(result.evidence.message.includes('search_corpus'));
    });

    it('fails when search returns no results', async () => {
      mockExistsSync = () => true;
      mockStatSync = () => ({ size: 100 });
      mockResolveDatabasePath = () => '/fake/db.sqlite';
      mockCreateServer = () => ({});
      let callCount = 0;
      mockInvokeServerRequest = async () => {
        callCount++;
        if (callCount === 1) {
          return { isError: false, structuredContent: { total_chunks: 10 } };
        }
        return { isError: false, structuredContent: { results: [] } };
      };
      const { runCortexMcpSmoke } = await importGate();
      const result = await runCortexMcpSmoke({});
      assert.equal(result.pass, false);
    });

    it('fails when search structuredContent is undefined (?. and ?? branches)', async () => {
      mockExistsSync = () => true;
      mockStatSync = () => ({ size: 100 });
      mockResolveDatabasePath = () => '/fake/db.sqlite';
      mockCreateServer = () => ({});
      let callCount = 0;
      mockInvokeServerRequest = async () => {
        callCount++;
        if (callCount === 1) {
          return { isError: false, structuredContent: { total_chunks: 10 } };
        }
        return { isError: false };
      };
      const { runCortexMcpSmoke } = await importGate();
      const result = await runCortexMcpSmoke({});
      assert.equal(result.pass, false);
    });

    it('passes when both stats and search succeed', async () => {
      setupPassingMocks();
      const { runCortexMcpSmoke } = await importGate();
      const result = await runCortexMcpSmoke({});
      assert.equal(result.pass, true);
      assert.equal(result.evidence.searchResults, 1);
      assert.equal(result.fixHint, null);
      assert.equal(result.owner, '05-green-testing');
    });

    it('passes with no options argument (default param)', async () => {
      setupPassingMocks();
      const { runCortexMcpSmoke } = await importGate();
      const result = await runCortexMcpSmoke();
      assert.equal(result.pass, true);
    });

    it('passes databasePath option to resolveDatabasePath', async () => {
      setupPassingMocks();
      let receivedPath;
      mockResolveDatabasePath = (p) => {
        receivedPath = p;
        return '/fake/db.sqlite';
      };
      const { runCortexMcpSmoke } = await importGate();
      await runCortexMcpSmoke({ databasePath: '/custom/path.sqlite' });
      assert.equal(receivedPath, '/custom/path.sqlite');
    });
  });

  describe('main via import.meta.url guard', () => {
    it('prints usage with --help', async () => {
      setupPassingMocks();
      const logs = await importGateMain(['--help']);
      assert.ok(logs.some((l) => l.includes('Repo Cortex MCP smoke gate')));
    });

    it('prints usage with -h', async () => {
      setupPassingMocks();
      const logs = await importGateMain(['-h']);
      assert.ok(logs.some((l) => l.includes('Repo Cortex MCP smoke gate')));
    });

    it('emits JSON with --json when pass=true', async () => {
      setupPassingMocks();
      const logs = await importGateMain(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, true);
    });

    it('emits PASS text without --json when pass=true', async () => {
      setupPassingMocks();
      const logs = await importGateMain([]);
      assert.ok(logs.some((l) => l.includes('PASS')));
    });

    it('emits JSON with --json when pass=false', async () => {
      setupFailingMocks();
      const logs = await importGateMain(['--json']);
      const parsed = JSON.parse(logs[0]);
      assert.equal(parsed.pass, false);
    });

    it('emits FAIL text without --json when pass=false', async () => {
      setupFailingMocks();
      const logs = await importGateMain([]);
      assert.ok(logs.some((l) => l.includes('FAIL')));
    });

    it('passes --databasePath option through parseArgs', async () => {
      setupPassingMocks();
      let receivedPath;
      mockResolveDatabasePath = (p) => {
        receivedPath = p;
        return '/fake/db.sqlite';
      };
      await importGateMain(['--databasePath=/custom/db.sqlite']);
      assert.equal(receivedPath, '/custom/db.sqlite');
    });
  });

  describe('import.meta.url guard', () => {
    it('does not run main when argv[1] does not match', async () => {
      setupPassingMocks();
      const logs = [];
      const originalLog = console.log;
      console.log = (...args) => logs.push(args.map(String).join(' '));
      try {
        jest.resetModules();
        await withArgv([process.execPath, 'dummy'], async () => {
          await import('./cortex-mcp-smoke.mjs');
          await new Promise((r) => setTimeout(r, 100));
        });
      } finally {
        console.log = originalLog;
      }
      assert.equal(logs.length, 0);
    });
  });
});
