/**
 * @module perf-step28.test
 * @description Comprehensive tests for perf-step28.mjs targeting 100% coverage.
 * This file has no exports — main().catch() runs at import time.
 */
import { jest } from '@jest/globals';

const mockTraverseGraph = jest.fn();
jest.unstable_mockModule('../scripts/mcp-semantic/tools/traverse-graph.mjs', () => ({
  traverseGraph: mockTraverseGraph,
}));

describe('perf-step28', () => {
  let exitSpy;
  let logSpy;
  let errorSpy;
  let nowSpy;

  beforeEach(() => {
    mockTraverseGraph.mockReset();
    exitSpy = jest.spyOn(process, 'exit').mockImplementation(() => {});
    logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    errorSpy = jest.spyOn(console, 'error').mockImplementation(() => {});
    jest.resetModules();
  });

  afterEach(() => {
    exitSpy.mockRestore();
    logSpy.mockRestore();
    errorSpy.mockRestore();
    if (nowSpy) {
      nowSpy.mockRestore();
      nowSpy = null;
    }
  });

  /** Wait for main() to call process.exit before making assertions. */
  async function waitForExit() {
    while (exitSpy.mock.calls.length === 0) {
      await new Promise((r) => setTimeout(r, 10));
    }
  }

  it('runs and passes when p50 <= 20ms (exit 0)', async () => {
    mockTraverseGraph.mockResolvedValue({ nodes: [], edges: [] });
    await import('./perf-step28.mjs');
    await waitForExit();
    expect(exitSpy).toHaveBeenCalledWith(0);
    const output = JSON.parse(logSpy.mock.calls[0][0]);
    expect(output.subsystem).toBe('graph_traversal');
    expect(output.budget_ms).toBe(20);
    expect(output.passes).toBe(true);
    expect(output.results.samples).toBe(150); // 30 runs * 5 seed sets
  });

  it('runs and fails when p50 > 20ms (exit 1)', async () => {
    // Mock performance.now to simulate slow traversals
    let nowValue = 0;
    nowSpy = jest.spyOn(performance, 'now').mockImplementation(() => {
      nowValue += 25; // each call advances 25ms → each traversal takes 25ms
      return nowValue;
    });
    mockTraverseGraph.mockResolvedValue({ nodes: [], edges: [] });
    await import('./perf-step28.mjs');
    await waitForExit();
    expect(exitSpy).toHaveBeenCalledWith(1);
    const output = JSON.parse(logSpy.mock.calls[0][0]);
    expect(output.passes).toBe(false);
    expect(output.results.p50).toBeGreaterThan(20);
  });

  it('exits 1 when main() throws (catch handler)', async () => {
    mockTraverseGraph.mockRejectedValue(new Error('traversal failed'));
    await import('./perf-step28.mjs');
    await waitForExit();
    expect(exitSpy).toHaveBeenCalledWith(1);
    expect(errorSpy).toHaveBeenCalled();
  });

  it('uses defaultDatabasePath when no argv[2] provided', async () => {
    mockTraverseGraph.mockResolvedValue({ nodes: [], edges: [] });
    await import('./perf-step28.mjs');
    await waitForExit();
    // traverseGraph should have been called with a databasePath
    expect(mockTraverseGraph).toHaveBeenCalled();
    const callArgs = mockTraverseGraph.mock.calls[0][0];
    expect(callArgs).toHaveProperty('databasePath');
    expect(callArgs).toHaveProperty('max_hops', 2);
    expect(callArgs).toHaveProperty('max_results', 20);
  });

  it('uses custom database path from argv[2]', async () => {
    const customPath = '/custom/db.sqlite';
    process.argv = ['node', 'perf-step28.mjs', customPath];
    mockTraverseGraph.mockResolvedValue({ nodes: [], edges: [] });
    await import('./perf-step28.mjs');
    await waitForExit();
    const callArgs = mockTraverseGraph.mock.calls[0][0];
    expect(callArgs.databasePath).toBe(customPath);
    // Reset argv
    process.argv = ['node', 'perf-step28.mjs'];
  });
});