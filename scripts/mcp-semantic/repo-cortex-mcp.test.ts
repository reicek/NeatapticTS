/**
 * @module repo-cortex-mcp.premium.test
 * @description Red tests for premium primary search defaults across Cortex MCP tools.
 *
 * These contracts verify that the integrated `search_corpus`, `search_context`,
 * and `search_advanced` tools apply the LLM-facing premium defaults consistently
 * when no explicit caller option overrides them. Until Step 11 implementation
 * lands, the defaults remain off and the contracts fail.
 */

import { execFileSync, spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';

// Tests run from the repository root, so cwd is a stable anchor for repo-relative
// paths without needing import.meta.url (which currently breaks ts-jest for the
// mcp-semantic-scripts project).
const REPO_ROOT = path.resolve();

interface ToolCallResult {
  isError: boolean;
  structuredContent?: Record<string, unknown>;
}

const BUILD_INDEX_PATH = path.join(
  REPO_ROOT,
  'scripts',
  'semantic-index',
  'build-index.mjs',
);
const SNAPSHOT_SCRIPT_PATH = path.join(
  REPO_ROOT,
  'scripts',
  'semantic-index',
  'build-browser-snapshot.mjs',
);
const DATABASE_PATH = path.join(REPO_ROOT, 'data', 'turso-replica.sqlite');

const runModuleEvaluation = <Result>(source: string): Result => {
  const output = execFileSync(
    process.execPath,
    ['--input-type=module', '--eval', source],
    {
      cwd: REPO_ROOT,
      encoding: 'utf8',
    },
  );

  const trimmed = output.trim();
  if (trimmed.length === 0) {
    throw new Error('Module evaluation produced empty output');
  }
  return JSON.parse(trimmed) as Result;
};

describe('repo cortex MCP premium primary search defaults', () => {
  beforeAll(() => {
    if (!existsSync(DATABASE_PATH)) {
      spawnSync(process.execPath, [BUILD_INDEX_PATH], {
        cwd: REPO_ROOT,
        encoding: 'utf8',
        timeout: 600000,
      });
    }
    spawnSync(process.execPath, [SNAPSHOT_SCRIPT_PATH], {
      cwd: REPO_ROOT,
      encoding: 'utf8',
      timeout: 120000,
    });
  }, 600000);

  describe('search_corpus', () => {
    it('returns compact results by default', () => {
      const result = runModuleEvaluation<ToolCallResult>(`
        import { createRepoCortexMcpServer } from './scripts/mcp-semantic/repo-cortex-mcp.mjs';
        const server = createRepoCortexMcpServer({ databasePath: './rag-index/data/turso-replica.sqlite' });
        const result = await server.dispatch({
          jsonrpc: '2.0',
          id: 1,
          method: 'tools/call',
          params: { name: 'search_corpus', arguments: { query: 'NEAT activation' } },
        });
        console.log(JSON.stringify(result));
      `);

      expect(result.structuredContent?.compact).toBe(true);
    });

    it('returns a freshness proof after a successful index update', () => {
      const result = runModuleEvaluation<ToolCallResult>(`
        import { createRepoCortexMcpServer } from './scripts/mcp-semantic/repo-cortex-mcp.mjs';
        const server = createRepoCortexMcpServer({ databasePath: './rag-index/data/turso-replica.sqlite' });
        const result = await server.dispatch({
          jsonrpc: '2.0',
          id: 1,
          method: 'tools/call',
          params: { name: 'search_corpus', arguments: { query: 'NEAT activation' } },
        });
        console.log(JSON.stringify(result));
      `);

      expect(result.structuredContent?.freshness).toEqual(
        expect.objectContaining({
          stale: false,
          last_indexed_at: expect.any(Number),
          freshness_proof: expect.objectContaining({
            mtime_ms: expect.any(Number),
            size: expect.any(Number),
            sha256: expect.any(String),
          }),
        }),
      );
    });
  });

  describe('search_context', () => {
    it('returns compact results and an inline top result by default', () => {
      const result = runModuleEvaluation<ToolCallResult>(`
        import { createRepoCortexMcpServer } from './scripts/mcp-semantic/repo-cortex-mcp.mjs';
        const server = createRepoCortexMcpServer({ databasePath: './rag-index/data/turso-replica.sqlite' });
        const result = await server.dispatch({
          jsonrpc: '2.0',
          id: 1,
          method: 'tools/call',
          params: { name: 'search_context', arguments: { query: 'NEAT activation' } },
        });
        console.log(JSON.stringify(result));
      `);

      expect(result.structuredContent?.compact).toBe(true);
      expect(result.structuredContent?.top_result).toEqual(
        expect.objectContaining({
          chunk_id: expect.any(Number),
          text: expect.any(String),
        }),
      );
    });
  });

  describe('search_advanced', () => {
    it('triggers auto_fallback when the primary pipeline returns no results', () => {
      const result = runModuleEvaluation<ToolCallResult>(`
        import { createRepoCortexMcpServer } from './scripts/mcp-semantic/repo-cortex-mcp.mjs';
        const server = createRepoCortexMcpServer({ databasePath: './rag-index/data/turso-replica.sqlite' });
        const result = await server.dispatch({
          jsonrpc: '2.0',
          id: 1,
          method: 'tools/call',
          params: { name: 'search_advanced', arguments: { query: 'completely nonexistent xyzabc123', use_dense: false } },
        });
        console.log(JSON.stringify(result));
      `);

      expect(result.structuredContent?.fallback_triggered).toBe(true);
    });

    it('applies include_code_only by default for code_specific queries', () => {
      const result = runModuleEvaluation<ToolCallResult>(`
        import { createRepoCortexMcpServer } from './scripts/mcp-semantic/repo-cortex-mcp.mjs';
        const server = createRepoCortexMcpServer({ databasePath: './rag-index/data/turso-replica.sqlite' });
        const result = await server.dispatch({
          jsonrpc: '2.0',
          id: 1,
          method: 'tools/call',
          params: { name: 'search_advanced', arguments: { query: 'Network class architecture', query_class: 'code_specific' } },
        });
        console.log(JSON.stringify(result));
      `);

      expect(result.structuredContent?.include_code_only).toBe(true);
    });
  });
});
