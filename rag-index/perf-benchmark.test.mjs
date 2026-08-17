/**
 * @module perf-benchmark.test
 * @description Comprehensive tests for perf-benchmark.mjs targeting 100% coverage.
 * This file has no exports — main().catch() runs at import time.
 * All 6 problematic imports are mocked.
 */
import { jest } from '@jest/globals';

// ---------------------------------------------------------------------------
// Mock setup — all mocks defined before module import
// ---------------------------------------------------------------------------

const mockReadFile = jest.fn();
const mockCreateClient = jest.fn();
const mockCreateServer = jest.fn();
const mockInvokeServerRequest = jest.fn();
const mockGetTursoClient = jest.fn();
const mockCloseTursoClient = jest.fn();
const mockGetCachedClientCount = jest.fn();
let mockDefaultDbPath = 'C:\\mock\\database.sqlite';

jest.unstable_mockModule('node:fs/promises', () => ({ readFile: mockReadFile }));
jest.unstable_mockModule('@libsql/client', () => ({ createClient: mockCreateClient }));
jest.unstable_mockModule('../scripts/mcp-semantic/repo-cortex-mcp.mjs', () => ({
  createRepoCortexMcpServer: mockCreateServer,
}));
jest.unstable_mockModule('../scripts/agent-customization/mcp/mcp-utils.mjs', () => ({
  invokeServerRequest: mockInvokeServerRequest,
}));
jest.unstable_mockModule('../scripts/mcp-semantic/tools/cortex-db.mjs', () => ({
  getTursoClient: mockGetTursoClient,
  closeTursoClient: mockCloseTursoClient,
  getCachedClientCount: mockGetCachedClientCount,
}));
jest.unstable_mockModule('./init-schema.mjs', () => ({
  defaultDatabasePath: mockDefaultDbPath,
}));

// ---------------------------------------------------------------------------
// Helper: create a mock client that handles different SQL queries
// ---------------------------------------------------------------------------

function makeMockClient(ftsThrows = false) {
  return {
    execute: jest.fn().mockImplementation((params) => {
      const sql = typeof params === 'string' ? params : params.sql;
      if (sql.includes('SELECT chunk_id FROM chunks ORDER BY chunk_id LIMIT 1')) {
        return Promise.resolve({ rows: [{ chunk_id: 1 }] });
      }
      if (sql.includes('WHERE depth = 1')) {
        return Promise.resolve({ rows: [{ chunk_id: 2 }] });
      }
      if (sql.includes('SELECT file_path FROM documents')) {
        return Promise.resolve({ rows: [{ file_path: 'test.ts' }] });
      }
      if (sql.includes('EXPLAIN QUERY PLAN') && sql.includes('feedback_scores')) {
        if (ftsThrows) throw new Error('no such table: feedback_scores');
        return Promise.resolve({ rows: [{ detail: 'SCAN TABLE chunks_fts' }] });
      }
      if (sql.includes('EXPLAIN QUERY PLAN')) {
        return Promise.resolve({ rows: [{ detail: 'SCAN TABLE chunks_fts' }] });
      }
      return Promise.resolve({ rows: [] });
    }),
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

describe('perf-benchmark', () => {
  let exitSpy;
  let logSpy;
  let errorSpy;
  let origSyncUrl;
  let origAuthToken;
  let origSyncInterval;
  let unhandledHandler;

  beforeEach(() => {
    exitSpy = jest.spyOn(process, 'exit').mockImplementation(() => {});
    logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    errorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    origSyncUrl = process.env.TURSO_SYNC_URL;
    origAuthToken = process.env.TURSO_AUTH_TOKEN;
    origSyncInterval = process.env.TURSO_SYNC_INTERVAL;
    delete process.env.TURSO_SYNC_URL;
    delete process.env.TURSO_AUTH_TOKEN;
    delete process.env.TURSO_SYNC_INTERVAL;
    unhandledHandler = () => {};
    process.on('unhandledRejection', unhandledHandler);
    jest.resetModules();
  });

  afterEach(() => {
    exitSpy.mockRestore();
    logSpy.mockRestore();
    errorSpy.mockRestore();
    if (unhandledHandler) {
      process.off('unhandledRejection', unhandledHandler);
      unhandledHandler = null;
    }
    if (origSyncUrl !== undefined) process.env.TURSO_SYNC_URL = origSyncUrl;
    else delete process.env.TURSO_SYNC_URL;
    if (origAuthToken !== undefined) process.env.TURSO_AUTH_TOKEN = origAuthToken;
    else delete process.env.TURSO_AUTH_TOKEN;
    if (origSyncInterval !== undefined) process.env.TURSO_SYNC_INTERVAL = origSyncInterval;
    else delete process.env.TURSO_SYNC_INTERVAL;
  });

  /** Wait for main() to call process.exit before making assertions. */
  async function waitForExit() {
    while (exitSpy.mock.calls.length === 0) {
      await new Promise((r) => setTimeout(r, 10));
    }
  }

  it('shows help and exits 0 with --help', async () => {
    let exitCalled = false;
    exitSpy.mockImplementation((code) => {
      if (exitCalled) return;
      exitCalled = true;
      throw new Error(`EXIT:${code}`);
    });
    process.argv = ['node', 'perf-benchmark.mjs', '--help'];
    mockReadFile.mockResolvedValue('');
    await import('./perf-benchmark.mjs');
    await new Promise((r) => setTimeout(r, 0));
    expect(exitSpy).toHaveBeenCalledWith(0);
    expect(logSpy).toHaveBeenCalled();
    const helpText = logSpy.mock.calls[0][0];
    expect(helpText).toContain('Benchmark all 18 repo-cortex-mcp tools');
  });

  it('shows help and exits 0 with -h', async () => {
    let exitCalled = false;
    exitSpy.mockImplementation((code) => {
      if (exitCalled) return;
      exitCalled = true;
      throw new Error(`EXIT:${code}`);
    });
    process.argv = ['node', 'perf-benchmark.mjs', '-h'];
    mockReadFile.mockResolvedValue('');
    await import('./perf-benchmark.mjs');
    await new Promise((r) => setTimeout(r, 0));
    expect(exitSpy).toHaveBeenCalledWith(0);
  });

  it('runs full benchmark in text mode (no sync, all tools succeed)', async () => {
    process.argv = ['node', 'perf-benchmark.mjs'];
    mockReadFile.mockResolvedValue('# comment\n\nTURSO_TEST=test\nNO_EQUALS\n');
    const client = makeMockClient(false);
    mockGetTursoClient.mockResolvedValue(client);
    mockCreateServer.mockReturnValue({});
    mockInvokeServerRequest.mockResolvedValue({ content: [] });
    mockGetCachedClientCount.mockReturnValue(1);
    mockCloseTursoClient.mockResolvedValue(undefined);
    mockCreateClient.mockReturnValue({
      execute: jest.fn().mockResolvedValue({ rows: [] }),
      sync: jest.fn().mockResolvedValue(undefined),
      close: jest.fn().mockResolvedValue(undefined),
    });

    await import('./perf-benchmark.mjs');
    await waitForExit();

    expect(mockGetTursoClient).toHaveBeenCalled();
    expect(mockCloseTursoClient).toHaveBeenCalled();
    expect(exitSpy).toHaveBeenCalled();
    // Text output should contain table header
    const tableOutput = logSpy.mock.calls.find(
      (c) => typeof c[0] === 'string' && c[0].includes('Repo Cortex MCP Tool Latency Benchmark'),
    );
    expect(tableOutput).toBeDefined();
  });

  it('runs full benchmark in JSON mode with sync configured', async () => {
    process.argv = ['node', 'perf-benchmark.mjs', '--json'];
    process.env.TURSO_SYNC_URL = 'libsql://sync.example.com';
    process.env.TURSO_AUTH_TOKEN = 'token123';
    process.env.TURSO_SYNC_INTERVAL = '30';
    mockReadFile.mockResolvedValue('');
    const client = makeMockClient(false);
    mockGetTursoClient.mockResolvedValue(client);
    mockCreateServer.mockReturnValue({});
    mockInvokeServerRequest.mockResolvedValue({ content: [] });
    mockGetCachedClientCount.mockReturnValue(1);
    mockCloseTursoClient.mockResolvedValue(undefined);
    const syncClient = {
      execute: jest.fn().mockResolvedValue({ rows: [] }),
      sync: jest.fn().mockResolvedValue(undefined),
      close: jest.fn().mockResolvedValue(undefined),
    };
    mockCreateClient.mockReturnValue(syncClient);

    await import('./perf-benchmark.mjs');
    await waitForExit();

    const jsonOutput = logSpy.mock.calls.find((c) => {
      try {
        const parsed = JSON.parse(c[0]);
        return parsed.benchmark === 'repo-cortex-mcp-latency';
      } catch {
        return false;
      }
    });
    expect(jsonOutput).toBeDefined();
    const parsed = JSON.parse(jsonOutput[0]);
    expect(parsed.embedded_replica.sync_configured).toBe(true);
    expect(parsed.embedded_replica.sync_lag_ms).toBeGreaterThanOrEqual(0);
  });

  it('handles tool errors (isError with structuredContent, content text, no content)', async () => {
    process.argv = ['node', 'perf-benchmark.mjs', '--json'];
    mockReadFile.mockResolvedValue('');
    const client = makeMockClient(false);
    mockGetTursoClient.mockResolvedValue(client);
    mockCreateServer.mockReturnValue({});
    mockGetCachedClientCount.mockReturnValue(1);
    mockCloseTursoClient.mockResolvedValue(undefined);
    mockCreateClient.mockReturnValue({
      execute: jest.fn().mockResolvedValue({ rows: [] }),
      sync: jest.fn().mockResolvedValue(undefined),
      close: jest.fn().mockResolvedValue(undefined),
    });

    let callCount = 0;
    mockInvokeServerRequest.mockImplementation((_server, request) => {
      const toolName = request.params.name;
      // Return error results for specific tools to cover extractToolErrorMessage branches
      if (toolName === 'search_corpus') {
        return Promise.resolve({
          isError: true,
          structuredContent: { error: 'structured error' },
        });
      }
      if (toolName === 'search_context') {
        return Promise.resolve({
          isError: true,
          content: [{ type: 'text', text: 'text error msg' }],
        });
      }
      if (toolName === 'search_advanced') {
        return Promise.resolve({
          isError: true,
          content: [{ type: 'not_text' }],
        });
      }
      if (toolName === 'load_chunk') {
        return Promise.resolve({ isError: true });
      }
      if (toolName === 'list_families') {
        return Promise.reject(new Error('thrown error'));
      }
      if (toolName === 'expand_query') {
        return Promise.reject('string error');
      }
      return Promise.resolve({ content: [] });
    });

    await import('./perf-benchmark.mjs');
    await waitForExit();

    const jsonOutput = logSpy.mock.calls.find((c) => {
      try {
        const parsed = JSON.parse(c[0]);
        return parsed.benchmark === 'repo-cortex-mcp-latency';
      } catch {
        return false;
      }
    });
    expect(jsonOutput).toBeDefined();
  });

  it('handles FTS plan catch path (first EXPLAIN fails)', async () => {
    process.argv = ['node', 'perf-benchmark.mjs', '--json'];
    mockReadFile.mockResolvedValue('');
    const client = makeMockClient(true); // FTS with feedback_scores throws
    mockGetTursoClient.mockResolvedValue(client);
    mockCreateServer.mockReturnValue({});
    mockInvokeServerRequest.mockResolvedValue({ content: [] });
    mockGetCachedClientCount.mockReturnValue(1);
    mockCloseTursoClient.mockResolvedValue(undefined);
    mockCreateClient.mockReturnValue({
      execute: jest.fn().mockResolvedValue({ rows: [] }),
      sync: jest.fn().mockResolvedValue(undefined),
      close: jest.fn().mockResolvedValue(undefined),
    });

    await import('./perf-benchmark.mjs');
    await waitForExit();

    const jsonOutput = logSpy.mock.calls.find((c) => {
      try {
        return JSON.parse(c[0]).benchmark === 'repo-cortex-mcp-latency';
      } catch {
        return false;
      }
    });
    expect(jsonOutput).toBeDefined();
  });

  it('handles sync error and close error', async () => {
    process.argv = ['node', 'perf-benchmark.mjs'];
    process.env.TURSO_SYNC_URL = 'libsql://sync.example.com';
    mockReadFile.mockResolvedValue('');
    const client = makeMockClient(false);
    mockGetTursoClient.mockResolvedValue(client);
    mockCreateServer.mockReturnValue({});
    mockInvokeServerRequest.mockResolvedValue({ content: [] });
    mockGetCachedClientCount.mockReturnValue(1);
    mockCloseTursoClient.mockResolvedValue(undefined);
    mockCreateClient.mockReturnValue({
      execute: jest.fn().mockResolvedValue({ rows: [] }),
      sync: jest.fn().mockRejectedValue(new Error('sync failed')),
      close: jest.fn().mockRejectedValue(new Error('close failed')),
    });

    await import('./perf-benchmark.mjs');
    await waitForExit();

    // Should still complete and exit
    expect(exitSpy).toHaveBeenCalled();
    // Text output should include sync lag error
    const syncOutput = logSpy.mock.calls.find(
      (c) => typeof c[0] === 'string' && c[0].includes('Sync lag'),
    );
    expect(syncOutput).toBeDefined();
  });

  it('handles loadEnvFile catch (readFile throws)', async () => {
    process.argv = ['node', 'perf-benchmark.mjs', '--json'];
    mockReadFile.mockRejectedValue(new Error('ENOENT'));
    const client = makeMockClient(false);
    mockGetTursoClient.mockResolvedValue(client);
    mockCreateServer.mockReturnValue({});
    mockInvokeServerRequest.mockResolvedValue({ content: [] });
    mockGetCachedClientCount.mockReturnValue(1);
    mockCloseTursoClient.mockResolvedValue(undefined);
    mockCreateClient.mockReturnValue({
      execute: jest.fn().mockResolvedValue({ rows: [] }),
      sync: jest.fn().mockResolvedValue(undefined),
      close: jest.fn().mockResolvedValue(undefined),
    });

    await import('./perf-benchmark.mjs');
    await waitForExit();

    expect(exitSpy).toHaveBeenCalled();
  });

  it('handles --database flag override', async () => {
    process.argv = ['node', 'perf-benchmark.mjs', '--database', 'C:\\custom\\db.sqlite', '--json'];
    mockReadFile.mockResolvedValue('');
    const client = makeMockClient(false);
    mockGetTursoClient.mockResolvedValue(client);
    mockCreateServer.mockReturnValue({});
    mockInvokeServerRequest.mockResolvedValue({ content: [] });
    mockGetCachedClientCount.mockReturnValue(1);
    mockCloseTursoClient.mockResolvedValue(undefined);
    mockCreateClient.mockReturnValue({
      execute: jest.fn().mockResolvedValue({ rows: [] }),
      sync: jest.fn().mockResolvedValue(undefined),
      close: jest.fn().mockResolvedValue(undefined),
    });

    await import('./perf-benchmark.mjs');
    await waitForExit();

    const jsonOutput = logSpy.mock.calls.find((c) => {
      try {
        return JSON.parse(c[0]).benchmark === 'repo-cortex-mcp-latency';
      } catch {
        return false;
      }
    });
    expect(jsonOutput).toBeDefined();
    const parsed = JSON.parse(jsonOutput[0]);
    expect(parsed.environment.database_path).toBe('C:\\custom\\db.sqlite');
  });

  it('handles main() catch (error thrown)', async () => {
    process.argv = ['node', 'perf-benchmark.mjs', '--json'];
    mockReadFile.mockResolvedValue('');
    mockGetTursoClient.mockRejectedValue(new Error('database connection failed'));

    await import('./perf-benchmark.mjs');
    await waitForExit();

    expect(errorSpy).toHaveBeenCalled();
    expect(exitSpy).toHaveBeenCalledWith(1);
  });

  it('handles TURSO_SYNC_INTERVAL default in text output', async () => {
    process.argv = ['node', 'perf-benchmark.mjs'];
    mockReadFile.mockResolvedValue('');
    const client = makeMockClient(false);
    mockGetTursoClient.mockResolvedValue(client);
    mockCreateServer.mockReturnValue({});
    mockInvokeServerRequest.mockResolvedValue({ content: [] });
    mockGetCachedClientCount.mockReturnValue(1);
    mockCloseTursoClient.mockResolvedValue(undefined);
    mockCreateClient.mockReturnValue({
      execute: jest.fn().mockResolvedValue({ rows: [] }),
      sync: jest.fn().mockResolvedValue(undefined),
      close: jest.fn().mockResolvedValue(undefined),
    });

    await import('./perf-benchmark.mjs');
    await waitForExit();

    const syncIntervalOutput = logSpy.mock.calls.find(
      (c) => typeof c[0] === 'string' && c[0].includes('Sync interval'),
    );
    expect(syncIntervalOutput).toBeDefined();
    expect(syncIntervalOutput[0]).toContain('default(60)');
  });
});