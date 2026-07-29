import { execFileSync, spawnSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import path from 'node:path';

interface ToolDescriptor {
  name: string;
  description: string;
  inputSchema: Record<string, unknown>;
}

interface ToolCallResult {
  isError: boolean;
  structuredContent?: Record<string, unknown>;
}

interface SmokeReport {
  pass: boolean;
  fixHint?: string;
  owner?: string;
}

const REPO_ROOT = path.resolve(__dirname, '..', '..', '..');
const MCP_CONFIG_PATH = path.join(REPO_ROOT, '.vscode', 'mcp.json');
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

  return JSON.parse(output) as Result;
};

const isCI = Boolean(process.env.CI || process.env.GITHUB_ACTIONS);
const describeOrSkip = isCI ? describe.skip : describe;

describeOrSkip('repo cortex MCP red contracts', () => {
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

  describe('server shape', () => {
    it('registers all semantic tools through createTool-compatible descriptors', () => {
      const descriptors = runModuleEvaluation<ToolDescriptor[]>(`
        import { createRepoCortexMcpServer } from './scripts/mcp-semantic/repo-cortex-mcp.mjs';
        const server = createRepoCortexMcpServer({ databasePath: './rag-index/data/turso-replica.sqlite' });
        const listed = await server.dispatch({ jsonrpc: '2.0', id: 1, method: 'tools/list' });
        console.log(JSON.stringify(listed.tools));
      `);

      expect(descriptors.map(({ name }) => name)).toEqual([
        'search_corpus',
        'search_context',
        'search_advanced',
        'load_chunk',
        'load_parent_chunk',
        'load_document',
        'freshness_check',
        'index_stats',
        'ann_build_index',
        'list_families',
        'scan_code_quality',
        'traverse_graph',
        'expand_query',
        'submit_feedback',
        'parallel_search',
        'multi_hop_search',
        'turso_branch',
        'turso_pitr',
      ]);
    });

    it('converts handler exceptions into structured MCP tool error results', () => {
      const result = runModuleEvaluation<ToolCallResult>(`
        import { createRepoCortexMcpServer } from './scripts/mcp-semantic/repo-cortex-mcp.mjs';
        const server = createRepoCortexMcpServer({ databasePath: './missing-semantic-index.sqlite' });
        const result = await server.dispatch({
          jsonrpc: '2.0',
          id: 1,
          method: 'tools/call',
          params: { name: 'load_chunk', arguments: { chunk_id: 1 } },
        });
        console.log(JSON.stringify(result));
      `);

      expect(result).toEqual(
        expect.objectContaining({
          isError: true,
          structuredContent: expect.objectContaining({
            error: expect.any(String),
          }),
        }),
      );
    });

    it('adds the cortex server registration without replacing existing MCP servers', () => {
      const mcpConfig = JSON.parse(readFileSync(MCP_CONFIG_PATH, 'utf8'));

      expect(Object.keys(mcpConfig.servers)).toEqual([
        'devtools',
        'cortex',
        'neataptic-dispatch-mcp',
        'neataptic-gate-mcp',
        'neataptic-validation-mcp',
        'neataptic-workflow-mcp',
      ]);
    });
  });

  describe('tool contracts', () => {
    it('search_corpus returns ranked structured chunk results', () => {
      const result = runModuleEvaluation<Record<string, unknown>>(`
        import { searchCorpus } from './scripts/mcp-semantic/tools/search-corpus.mjs';
        const result = await searchCorpus({ databasePath: './rag-index/data/turso-replica.sqlite', query: 'NEAT activation', limit: 3 });
        console.log(JSON.stringify(result));
      `);

      expect(result).toEqual(
        expect.objectContaining({
          results: expect.arrayContaining([
            expect.objectContaining({
              chunk_id: expect.any(Number),
              score: expect.any(Number),
              text: expect.any(String),
            }),
          ]),
        }),
      );
    });

    it('load_chunk returns one structured chunk by numeric id', () => {
      const result = runModuleEvaluation<Record<string, unknown>>(`
        import { loadChunk } from './scripts/mcp-semantic/tools/load-chunk.mjs';
        const result = await loadChunk({ databasePath: './rag-index/data/turso-replica.sqlite', chunk_id: 1 });
        console.log(JSON.stringify(result));
      `);

      expect(result).toEqual(
        expect.objectContaining({
          chunk: expect.objectContaining({
            chunk_id: 1,
            file_path: expect.any(String),
            text: expect.any(String),
          }),
        }),
      );
    });

    it('load_document returns ordered chunks for a repo file path', () => {
      const result = runModuleEvaluation<Record<string, unknown>>(`
        import { loadDocument } from './scripts/mcp-semantic/tools/load-document.mjs';
        const result = await loadDocument({ databasePath: './rag-index/data/turso-replica.sqlite', file_path: 'README.md' });
        console.log(JSON.stringify(result));
      `);

      expect(result).toEqual(
        expect.objectContaining({
          file_path: 'README.md',
          chunks: expect.arrayContaining([
            expect.objectContaining({ text: expect.any(String) }),
          ]),
        }),
      );
    });

    it('freshness_check reports stale documents when filesystem proof differs', () => {
      const result = runModuleEvaluation<Record<string, unknown>>(`
        import { freshnessCheck } from './scripts/mcp-semantic/tools/freshness-check.mjs';
        const result = await freshnessCheck({
          databasePath: './rag-index/data/turso-replica.sqlite',
          file_path: 'README.md',
          freshnessProof: { mtime_ms: 1, file_size: 1, sha256: 'stale-proof' },
        });
        console.log(JSON.stringify(result));
      `);

      expect(result).toEqual(
        expect.objectContaining({
          fresh: false,
          stale: expect.arrayContaining(['README.md']),
        }),
      );
    });

    it('index_stats returns corpus row counts and last build timestamp', () => {
      const result = runModuleEvaluation<Record<string, unknown>>(`
        import { indexStats } from './scripts/mcp-semantic/tools/index-stats.mjs';
        const result = await indexStats({ databasePath: './rag-index/data/turso-replica.sqlite' });
        console.log(JSON.stringify(result));
      `);

      expect(result).toEqual(
        expect.objectContaining({
          total_documents: expect.any(Number),
          total_chunks: expect.any(Number),
          last_build_timestamp: expect.any(String),
        }),
      );
    });

    it('list_families returns indexed document families with counts', () => {
      const result = runModuleEvaluation<Record<string, unknown>>(`
        import { listFamilies } from './scripts/mcp-semantic/tools/list-families.mjs';
        const result = await listFamilies({ databasePath: './rag-index/data/turso-replica.sqlite' });
        console.log(JSON.stringify(result));
      `);

      expect(result).toEqual(
        expect.objectContaining({
          families: expect.arrayContaining([
            expect.objectContaining({
              family: expect.any(String),
              documents: expect.any(Number),
            }),
          ]),
        }),
      );
    });
  });

  describe('CLI and smoke contracts', () => {
    it('repo-cortex-mcp supports --help with the semantic tool list', () => {
      const output = execFileSync(
        process.execPath,
        ['scripts/mcp-semantic/repo-cortex-mcp.mjs', '--help'],
        {
          cwd: REPO_ROOT,
          encoding: 'utf8',
        },
      );

      expect(output).toContain('search_corpus');
    });

    it('repo-cortex-mcp supports --self-check --json with a passing report for the real index', () => {
      const report = JSON.parse(
        execFileSync(
          process.execPath,
          [
            'scripts/mcp-semantic/repo-cortex-mcp.mjs',
            '--self-check',
            '--json',
          ],
          { cwd: REPO_ROOT, encoding: 'utf8' },
        ),
      );

      expect(report).toEqual(expect.objectContaining({ ok: true }));
    });

    it('cortex-mcp-smoke returns a failing JSON gate report when the index is absent', () => {
      const report = JSON.parse(
        execFileSync(
          process.execPath,
          [
            'scripts/agent-customization/gates/cortex-mcp-smoke.mjs',
            '--json',
            '--databasePath=./missing-semantic-index.sqlite',
          ],
          { cwd: REPO_ROOT, encoding: 'utf8' },
        ),
      ) as SmokeReport;

      expect(report).toEqual(
        expect.objectContaining({
          pass: false,
          fixHint: 'Run: node rag-index/build-index.mjs',
        }),
      );
    });
  });
});
