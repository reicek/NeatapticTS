import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

jest.unstable_mockModule('../../../mcp-semantic/repo-cortex-mcp.mjs', () => ({
  createRepoCortexMcpServer: jest.fn(() => ({
    dispatch: jest.fn(async (req) => {
      if (req.method === 'initialize') {
        return { protocolVersion: '2024-11-05', capabilities: {}, serverInfo: { name: 'cortex' } };
      }
      if (req.method === 'tools/list') {
        return { tools: [{ name: 'index_stats' }, { name: 'list_families' }, { name: 'search_corpus' }] };
      }
      if (req.method === 'tools/call') {
        if (req.params.name === 'search_corpus') {
          return { content: [{ type: 'text', text: 'results' }] };
        }
        return { content: [{ type: 'text', text: 'ok' }] };
      }
      return {};
    }),
  })),
}));

jest.unstable_mockModule('node:fs', () => ({
  writeFileSync: jest.fn(),
  readFileSync: jest.fn(),
  existsSync: jest.fn(),
}));

let mockCortex;
let mockFs;

beforeEach(async () => {
  jest.resetModules();
  mockCortex = await import('../../../mcp-semantic/repo-cortex-mcp.mjs');
  mockFs = await import('node:fs');
  jest.clearAllMocks();
});

function captureConsole() {
  const logs = [];
  const origLog = console.log;
  console.log = (...args) => logs.push(args.join(' '));
  return { logs, restore() { console.log = origLog; } };
}

describe('capture-cortex-snapshot', () => {
  it('captures and writes a snapshot successfully', async () => {
    const cap = captureConsole();

    await import('./capture-cortex-snapshot.mjs');

    cap.restore();
    assert.ok(mockCortex.createRepoCortexMcpServer.mock.calls.length >= 1);
    assert.ok(mockFs.writeFileSync.mock.calls.length >= 1);
    const snapshotPath = mockFs.writeFileSync.mock.calls[0][0];
    const snapshotData = JSON.parse(mockFs.writeFileSync.mock.calls[0][1]);
    assert.ok(snapshotPath.includes('cortex-real-snapshot.json'));
    assert.ok(snapshotData.meta);
    assert.ok(snapshotData.initialize);
    assert.ok(snapshotData.toolsList);
    assert.ok(snapshotData.calls);
  });

  it('handles tool call errors in snapshot', async () => {
    const cap = captureConsole();
    jest.resetModules();
    mockCortex = await import('../../../mcp-semantic/repo-cortex-mcp.mjs');
    mockFs = await import('node:fs');
    jest.clearAllMocks();

    // Override dispatch to throw for specific calls
    mockCortex.createRepoCortexMcpServer.mockReturnValue({
      dispatch: jest.fn(async (req) => {
        if (req.method === 'initialize') return { protocolVersion: '2024-11-05' };
        if (req.method === 'tools/list') return { tools: [] };
        if (req.method === 'tools/call') {
          throw new Error('tool error');
        }
        return {};
      }),
    });

    await import('./capture-cortex-snapshot.mjs');

    cap.restore();
    const snapshotData = JSON.parse(mockFs.writeFileSync.mock.calls[0][1]);
    assert.ok(snapshotData.calls.indexStats.isError);
    assert.ok(snapshotData.calls.indexStats.error.includes('tool error'));
  });

  it('handles non-Error throw in tool call', async () => {
    const cap = captureConsole();
    jest.resetModules();
    mockCortex = await import('../../../mcp-semantic/repo-cortex-mcp.mjs');
    mockFs = await import('node:fs');
    jest.clearAllMocks();

    mockCortex.createRepoCortexMcpServer.mockReturnValue({
      dispatch: jest.fn(async (req) => {
        if (req.method === 'initialize') return { protocolVersion: '2024-11-05' };
        if (req.method === 'tools/list') return { tools: [] };
        if (req.method === 'tools/call') {
          throw 'string error';
        }
        return {};
      }),
    });

    await import('./capture-cortex-snapshot.mjs');

    cap.restore();
    const snapshotData = JSON.parse(mockFs.writeFileSync.mock.calls[0][1]);
    assert.ok(snapshotData.calls.indexStats.isError);
    assert.strictEqual(snapshotData.calls.indexStats.error, 'string error');
  });
});