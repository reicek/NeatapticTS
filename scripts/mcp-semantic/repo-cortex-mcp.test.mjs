/**
 * @module repo-cortex-mcp.test
 * @description Coverage tests for repo-cortex-mcp.mjs — server creation, tool
 * registration, handler delegation, self-check, CLI main(), and parseDatabasePath.
 */
import { jest } from '@jest/globals';
import { pathToFileURL as realPathToFileURL } from 'node:url';
import { resolve as realResolve } from 'node:path';

// Compute the module file path and URL using real path APIs.
const moduleAbsPath = realResolve(
  'scripts',
  'mcp-semantic',
  'repo-cortex-mcp.mjs',
);
const MODULE_FILE_URL = realPathToFileURL(moduleAbsPath).href;

/* ---- Mock mcp-utils ---- */
const mockCreateMcpServer = jest.fn();
const mockCreateSelfCheckReport = jest.fn();
const mockCreateTool = jest.fn();
const mockEmitSelfCheckReport = jest.fn();
const mockParseMcpCliArgs = jest.fn();
const mockPrintMcpUsage = jest.fn();
const mockRunStdioMcpServer = jest.fn();
const mockSelfCheckError = jest.fn();

jest.unstable_mockModule('../agent-customization/mcp/mcp-utils.mjs', () => ({
  createMcpServer: mockCreateMcpServer,
  createSelfCheckReport: mockCreateSelfCheckReport,
  createTool: mockCreateTool,
  emitSelfCheckReport: mockEmitSelfCheckReport,
  parseMcpCliArgs: mockParseMcpCliArgs,
  printMcpUsage: mockPrintMcpUsage,
  runStdioMcpServer: mockRunStdioMcpServer,
  selfCheckError: mockSelfCheckError,
  __esModule: true,
}));

/* ---- Mock all tool modules ---- */
const mockSearchCorpus = jest.fn();
const mockSearchContext = jest.fn();
const mockSearchAdvanced = jest.fn();
const mockLoadChunk = jest.fn();
const mockLoadParentChunk = jest.fn();
const mockLoadDocument = jest.fn();
const mockFreshnessCheck = jest.fn();
const mockIndexStats = jest.fn();
const mockListFamilies = jest.fn();
const mockExpandQueryHandler = jest.fn();
const mockSubmitFeedback = jest.fn();
const mockTraverseGraphHandler = jest.fn();
const mockMultiHopSearchHandler = jest.fn();
const mockTursoBranch = jest.fn();
const mockTursoPitr = jest.fn();
const mockBuildAnnIndex = jest.fn();
const mockGetTursoClient = jest.fn();
const mockRunParallelQueries = jest.fn();
const mockRunDocsQualityMetrics = jest.fn();

jest.unstable_mockModule('./tools/search-corpus.mjs', () => ({
  searchCorpus: mockSearchCorpus,
  __esModule: true,
}));
jest.unstable_mockModule('./tools/search-context.mjs', () => ({
  searchContext: mockSearchContext,
  __esModule: true,
}));
jest.unstable_mockModule('./tools/search-advanced.mjs', () => ({
  searchAdvanced: mockSearchAdvanced,
  __esModule: true,
}));
jest.unstable_mockModule('./tools/load-chunk.mjs', () => ({
  loadChunk: mockLoadChunk,
  __esModule: true,
}));
jest.unstable_mockModule('./tools/load-parent-chunk.mjs', () => ({
  loadParentChunk: mockLoadParentChunk,
  __esModule: true,
}));
jest.unstable_mockModule('./tools/load-document.mjs', () => ({
  loadDocument: mockLoadDocument,
  __esModule: true,
}));
jest.unstable_mockModule('./tools/freshness-check.mjs', () => ({
  freshnessCheck: mockFreshnessCheck,
  __esModule: true,
}));
jest.unstable_mockModule('./tools/index-stats.mjs', () => ({
  indexStats: mockIndexStats,
  __esModule: true,
}));
jest.unstable_mockModule('./tools/list-families.mjs', () => ({
  listFamilies: mockListFamilies,
  __esModule: true,
}));
jest.unstable_mockModule('./tools/expand-query.mjs', () => ({
  expandQueryHandler: mockExpandQueryHandler,
  __esModule: true,
}));
jest.unstable_mockModule('./tools/submit-feedback.mjs', () => ({
  submitFeedback: mockSubmitFeedback,
  __esModule: true,
}));
jest.unstable_mockModule('./tools/traverse-graph.mjs', () => ({
  traverseGraphHandler: mockTraverseGraphHandler,
  __esModule: true,
}));
jest.unstable_mockModule('./tools/multi-hop-search.mjs', () => ({
  multiHopSearchHandler: mockMultiHopSearchHandler,
  __esModule: true,
}));
jest.unstable_mockModule('./tools/turso-branch.mjs', () => ({
  tursoBranch: mockTursoBranch,
  __esModule: true,
}));
jest.unstable_mockModule('./tools/turso-pitr.mjs', () => ({
  tursoPitr: mockTursoPitr,
  __esModule: true,
}));
jest.unstable_mockModule('./tools/ann-index.mjs', () => ({
  buildAnnIndex: mockBuildAnnIndex,
  __esModule: true,
}));
jest.unstable_mockModule('./tools/cortex-db.mjs', () => ({
  getTursoClient: mockGetTursoClient,
  __esModule: true,
}));
jest.unstable_mockModule('../../rag-index/parallel-search.mjs', () => ({
  runParallelQueries: mockRunParallelQueries,
  __esModule: true,
}));
jest.unstable_mockModule(
  '../../rag-index/docs-quality/docs-quality.metrics.mjs',
  () => ({
    runDocsQualityMetrics: mockRunDocsQualityMetrics,
    __esModule: true,
  }),
);

/* ---- Import the module (main() won't run — argv[1] won't match) ---- */
const { createRepoCortexMcpServer, createRepoCortexTools, runSelfCheck } =
  await import('./repo-cortex-mcp.mjs');

/** All expected tool names from createRepoCortexTools. */
const EXPECTED_TOOL_NAMES = [
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
];

describe('repo-cortex-mcp', () => {
  const originalArgv = process.argv;

  beforeEach(() => {
    mockCreateTool.mockReset();
    mockCreateTool.mockImplementation((obj) => obj);
    mockCreateMcpServer.mockReset();
    mockCreateMcpServer.mockReturnValue({
      serverInfo: { name: 'cortex', version: '0.1.0' },
      tools: [],
      dispatch: jest.fn(),
    });
    mockCreateSelfCheckReport.mockReset();
    mockCreateSelfCheckReport.mockImplementation((_name, issues, extra) => ({
      ok: issues.length === 0,
      issues,
      ...extra,
    }));
    mockEmitSelfCheckReport.mockReset();
    mockParseMcpCliArgs.mockReset();
    mockPrintMcpUsage.mockReset();
    mockRunStdioMcpServer.mockReset();
    mockRunStdioMcpServer.mockResolvedValue(undefined);
    mockSelfCheckError.mockReset();
    mockSelfCheckError.mockImplementation((file, message) => ({
      file,
      message,
    }));

    // Reset all tool mocks
    for (const m of [
      mockSearchCorpus,
      mockSearchContext,
      mockSearchAdvanced,
      mockLoadChunk,
      mockLoadParentChunk,
      mockLoadDocument,
      mockFreshnessCheck,
      mockIndexStats,
      mockListFamilies,
      mockExpandQueryHandler,
      mockSubmitFeedback,
      mockTraverseGraphHandler,
      mockMultiHopSearchHandler,
      mockTursoBranch,
      mockTursoPitr,
      mockBuildAnnIndex,
      mockGetTursoClient,
      mockRunParallelQueries,
      mockRunDocsQualityMetrics,
    ]) {
      m.mockReset();
      m.mockResolvedValue({ ok: true });
    }
  });

  afterEach(() => {
    process.argv = originalArgv;
  });

  /* ====================== createRepoCortexTools ====================== */

  describe('createRepoCortexTools', () => {
    it('returns all expected tool names', () => {
      const tools = createRepoCortexTools();
      const names = tools.map((t) => t.name);
      for (const expected of EXPECTED_TOOL_NAMES) {
        expect(names).toContain(expected);
      }
      expect(tools).toHaveLength(EXPECTED_TOOL_NAMES.length);
    });

    it('every tool has a handler function', () => {
      const tools = createRepoCortexTools();
      for (const tool of tools) {
        expect(typeof tool.handler).toBe('function');
      }
    });

    it('every tool has a name and description', () => {
      const tools = createRepoCortexTools();
      for (const tool of tools) {
        expect(tool.name).toBeTruthy();
        expect(tool.description).toBeTruthy();
      }
    });

    /* --- search_corpus handler --- */
    it('search_corpus handler delegates with compact default true', async () => {
      const tools = createRepoCortexTools('/db');
      const tool = tools.find((t) => t.name === 'search_corpus');
      await tool.handler({ query: 'test' });
      expect(mockSearchCorpus).toHaveBeenCalledWith(
        expect.objectContaining({
          query: 'test',
          compact: true,
          databasePath: '/db',
        }),
      );
    });

    it('search_corpus handler respects compact=false', async () => {
      const tools = createRepoCortexTools();
      const tool = tools.find((t) => t.name === 'search_corpus');
      await tool.handler({ query: 'test', compact: false });
      expect(mockSearchCorpus).toHaveBeenCalledWith(
        expect.objectContaining({ compact: false }),
      );
    });

    /* --- search_context handler --- */
    it('search_context handler delegates with compact and read_top_result defaults', async () => {
      const tools = createRepoCortexTools('/db');
      const tool = tools.find((t) => t.name === 'search_context');
      await tool.handler({ query: 'test' });
      expect(mockSearchContext).toHaveBeenCalledWith(
        expect.objectContaining({
          query: 'test',
          compact: true,
          read_top_result: true,
          databasePath: '/db',
        }),
      );
    });

    it('search_context handler respects explicit compact and read_top_result', async () => {
      const tools = createRepoCortexTools();
      const tool = tools.find((t) => t.name === 'search_context');
      await tool.handler({
        query: 'test',
        compact: false,
        read_top_result: false,
      });
      expect(mockSearchContext).toHaveBeenCalledWith(
        expect.objectContaining({ compact: false, read_top_result: false }),
      );
    });

    /* --- search_advanced handler --- */
    it('search_advanced handler delegates with auto_fallback default true', async () => {
      const tools = createRepoCortexTools('/db');
      const tool = tools.find((t) => t.name === 'search_advanced');
      await tool.handler({ query: 'test' });
      expect(mockSearchAdvanced).toHaveBeenCalledWith(
        expect.objectContaining({
          query: 'test',
          auto_fallback: true,
          databasePath: '/db',
        }),
      );
    });

    it('search_advanced handler respects auto_fallback=false', async () => {
      const tools = createRepoCortexTools();
      const tool = tools.find((t) => t.name === 'search_advanced');
      await tool.handler({ query: 'test', auto_fallback: false });
      expect(mockSearchAdvanced).toHaveBeenCalledWith(
        expect.objectContaining({ auto_fallback: false }),
      );
    });

    /* --- load_chunk handler --- */
    it('load_chunk handler delegates', async () => {
      const tools = createRepoCortexTools('/db');
      const tool = tools.find((t) => t.name === 'load_chunk');
      await tool.handler({ chunk_id: 42 });
      expect(mockLoadChunk).toHaveBeenCalledWith(
        expect.objectContaining({ chunk_id: 42, databasePath: '/db' }),
      );
    });

    /* --- load_parent_chunk handler --- */
    it('load_parent_chunk handler delegates', async () => {
      const tools = createRepoCortexTools('/db');
      const tool = tools.find((t) => t.name === 'load_parent_chunk');
      await tool.handler({ chunk_id: 42 });
      expect(mockLoadParentChunk).toHaveBeenCalledWith(
        expect.objectContaining({ chunk_id: 42, databasePath: '/db' }),
      );
    });

    /* --- load_document handler --- */
    it('load_document handler delegates', async () => {
      const tools = createRepoCortexTools('/db');
      const tool = tools.find((t) => t.name === 'load_document');
      await tool.handler({ file_path: 'src/test.ts' });
      expect(mockLoadDocument).toHaveBeenCalledWith(
        expect.objectContaining({
          file_path: 'src/test.ts',
          databasePath: '/db',
        }),
      );
    });

    /* --- freshness_check handler --- */
    it('freshness_check handler delegates', async () => {
      const tools = createRepoCortexTools('/db');
      const tool = tools.find((t) => t.name === 'freshness_check');
      await tool.handler({ file_path: 'src/test.ts' });
      expect(mockFreshnessCheck).toHaveBeenCalledWith(
        expect.objectContaining({
          file_path: 'src/test.ts',
          databasePath: '/db',
        }),
      );
    });

    /* --- index_stats handler --- */
    it('index_stats handler delegates', async () => {
      const tools = createRepoCortexTools('/db');
      const tool = tools.find((t) => t.name === 'index_stats');
      await tool.handler({ include_metadata_coverage: true });
      expect(mockIndexStats).toHaveBeenCalledWith(
        expect.objectContaining({
          include_metadata_coverage: true,
          databasePath: '/db',
        }),
      );
    });

    /* --- ann_build_index handler --- */
    it('ann_build_index handler delegates with default model and dimension', async () => {
      const tools = createRepoCortexTools('/db');
      const tool = tools.find((t) => t.name === 'ann_build_index');
      await tool.handler({ force: 'diskann' });
      expect(mockBuildAnnIndex).toHaveBeenCalledWith(
        expect.objectContaining({
          databasePath: '/db',
          modelId: 'all-MiniLM-L6-v2',
          dimension: 384,
          forceStrategy: 'diskann',
        }),
      );
    });

    it('ann_build_index handler works without force', async () => {
      const tools = createRepoCortexTools();
      const tool = tools.find((t) => t.name === 'ann_build_index');
      await tool.handler({});
      expect(mockBuildAnnIndex).toHaveBeenCalledWith(
        expect.objectContaining({ forceStrategy: undefined }),
      );
    });

    /* --- list_families handler --- */
    it('list_families handler delegates with databasePath', async () => {
      const tools = createRepoCortexTools('/db');
      const tool = tools.find((t) => t.name === 'list_families');
      await tool.handler();
      expect(mockListFamilies).toHaveBeenCalledWith({ databasePath: '/db' });
    });

    /* --- scan_code_quality handler --- */
    it('scan_code_quality handler uses scope=paths with non-empty source_paths', async () => {
      const tools = createRepoCortexTools();
      const tool = tools.find((t) => t.name === 'scan_code_quality');
      await tool.handler({ source_paths: ['src/foo.ts'] });
      expect(mockRunDocsQualityMetrics).toHaveBeenCalledWith(
        expect.objectContaining({
          scope: 'paths',
          sourcePaths: ['src/foo.ts'],
        }),
      );
    });

    it('scan_code_quality handler uses scope=src with empty source_paths', async () => {
      const tools = createRepoCortexTools();
      const tool = tools.find((t) => t.name === 'scan_code_quality');
      await tool.handler({ source_paths: [] });
      expect(mockRunDocsQualityMetrics).toHaveBeenCalledWith(
        expect.objectContaining({ scope: 'src', sourcePaths: [] }),
      );
    });

    it('scan_code_quality handler uses scope=src with undefined source_paths', async () => {
      const tools = createRepoCortexTools();
      const tool = tools.find((t) => t.name === 'scan_code_quality');
      await tool.handler({});
      expect(mockRunDocsQualityMetrics).toHaveBeenCalledWith(
        expect.objectContaining({ scope: 'src', sourcePaths: undefined }),
      );
    });

    it('scan_code_quality handler maps complexity_threshold and min_jsdoc_words', async () => {
      const tools = createRepoCortexTools();
      const tool = tools.find((t) => t.name === 'scan_code_quality');
      await tool.handler({ complexity_threshold: 10, min_jsdoc_words: 5 });
      expect(mockRunDocsQualityMetrics).toHaveBeenCalledWith(
        expect.objectContaining({
          complexityThreshold: 10,
          minJsdocWords: 5,
        }),
      );
    });

    /* --- traverse_graph handler --- */
    it('traverse_graph handler delegates', async () => {
      const tools = createRepoCortexTools();
      const tool = tools.find((t) => t.name === 'traverse_graph');
      await tool.handler({ seed_names: ['Test'] });
      expect(mockTraverseGraphHandler).toHaveBeenCalledWith({
        seed_names: ['Test'],
      });
    });

    /* --- expand_query handler --- */
    it('expand_query handler delegates', async () => {
      const tools = createRepoCortexTools();
      const tool = tools.find((t) => t.name === 'expand_query');
      await tool.handler({ query: 'test' });
      expect(mockExpandQueryHandler).toHaveBeenCalledWith({ query: 'test' });
    });

    /* --- submit_feedback handler --- */
    it('submit_feedback handler delegates with databasePath', async () => {
      const tools = createRepoCortexTools('/db');
      const tool = tools.find((t) => t.name === 'submit_feedback');
      await tool.handler({ chunk_id: 1, signal_type: 'positive' });
      expect(mockSubmitFeedback).toHaveBeenCalledWith(
        expect.objectContaining({
          chunk_id: 1,
          signal_type: 'positive',
          databasePath: '/db',
        }),
      );
    });

    /* --- parallel_search handler --- */
    it('parallel_search handler returns results without errors', async () => {
      mockGetTursoClient.mockResolvedValue({});
      mockRunParallelQueries.mockResolvedValue({ results: [], errors: [] });
      const tools = createRepoCortexTools('/db');
      const tool = tools.find((t) => t.name === 'parallel_search');
      const result = await tool.handler({ queries: [{ sql: 'SELECT 1' }] });
      expect(mockGetTursoClient).toHaveBeenCalledWith('/db');
      expect(mockRunParallelQueries).toHaveBeenCalledWith(
        expect.objectContaining({ queries: [{ sql: 'SELECT 1' }] }),
      );
      expect(result.errors).toBeUndefined();
    });

    it('parallel_search handler includes errors when present', async () => {
      mockGetTursoClient.mockResolvedValue({});
      const testErrors = [{ query: 0, error: 'fail' }];
      mockRunParallelQueries.mockResolvedValue({
        results: [],
        errors: testErrors,
      });
      const tools = createRepoCortexTools();
      const tool = tools.find((t) => t.name === 'parallel_search');
      const result = await tool.handler({ queries: [{ sql: 'SELECT 1' }] });
      expect(result.errors).toEqual(testErrors);
    });

    it('parallel_search handler handles undefined errors with ?? default', async () => {
      mockGetTursoClient.mockResolvedValue({});
      mockRunParallelQueries.mockResolvedValue({ results: [] });
      const tools = createRepoCortexTools();
      const tool = tools.find((t) => t.name === 'parallel_search');
      const result = await tool.handler({ queries: [] });
      expect(result.errors).toBeUndefined();
    });

    /* --- multi_hop_search handler --- */
    it('multi_hop_search handler delegates', async () => {
      const tools = createRepoCortexTools();
      const tool = tools.find((t) => t.name === 'multi_hop_search');
      await tool.handler({ query: 'test' });
      expect(mockMultiHopSearchHandler).toHaveBeenCalledWith({ query: 'test' });
    });

    /* --- turso_branch handler --- */
    it('turso_branch handler uses provided action', async () => {
      const tools = createRepoCortexTools();
      const tool = tools.find((t) => t.name === 'turso_branch');
      await tool.handler({ branch_name: 'test', action: 'delete' });
      expect(mockTursoBranch).toHaveBeenCalledWith(
        expect.objectContaining({ action: 'delete', branchName: 'test' }),
      );
    });

    it('turso_branch handler defaults action to create', async () => {
      const tools = createRepoCortexTools();
      const tool = tools.find((t) => t.name === 'turso_branch');
      await tool.handler({ branch_name: 'test' });
      expect(mockTursoBranch).toHaveBeenCalledWith(
        expect.objectContaining({ action: 'create' }),
      );
    });

    /* --- turso_pitr handler --- */
    it('turso_pitr handler delegates', async () => {
      const tools = createRepoCortexTools();
      const tool = tools.find((t) => t.name === 'turso_pitr');
      await tool.handler({ database_name: 'db', timestamp: '2024-01-01' });
      expect(mockTursoPitr).toHaveBeenCalledWith(
        expect.objectContaining({
          databaseName: 'db',
          timestamp: '2024-01-01',
        }),
      );
    });
  });

  /* ====================== createRepoCortexMcpServer ====================== */

  describe('createRepoCortexMcpServer', () => {
    it('creates server with correct name and version', () => {
      createRepoCortexMcpServer({ databasePath: '/db' });
      expect(mockCreateMcpServer).toHaveBeenCalledWith(
        expect.objectContaining({
          serverName: 'cortex',
          serverVersion: '0.1.0',
        }),
      );
    });

    it('passes tools from createRepoCortexTools', () => {
      mockCreateMcpServer.mockReturnValue({ tools: ['tool1'] });
      const server = createRepoCortexMcpServer({ databasePath: '/db' });
      expect(mockCreateMcpServer).toHaveBeenCalledWith(
        expect.objectContaining({ tools: expect.any(Array) }),
      );
    });

    it('works with no options (default {})', () => {
      createRepoCortexMcpServer();
      expect(mockCreateMcpServer).toHaveBeenCalled();
    });

    it('works with empty options', () => {
      createRepoCortexMcpServer({});
      expect(mockCreateMcpServer).toHaveBeenCalled();
    });
  });

  /* ====================== runSelfCheck ====================== */

  describe('runSelfCheck', () => {
    it('returns all_ok=true when all tools have handlers', async () => {
      const result = await runSelfCheck({ databasePath: '/db' });
      expect(result.all_ok).toBe(true);
      expect(result.tool_checks_count).toBe(EXPECTED_TOOL_NAMES.length);
      expect(result.tools_checked).toHaveLength(EXPECTED_TOOL_NAMES.length);
      expect(result.issues).toEqual([]);
    });

    it('returns all_ok=false when a tool handler is missing', async () => {
      mockCreateTool.mockImplementationOnce(() => ({
        name: 'broken',
        description: 'no handler',
      }));
      mockSelfCheckError.mockReturnValue({ file: 'test', message: 'broken' });
      const result = await runSelfCheck();
      expect(result.all_ok).toBe(false);
      expect(result.issues.length).toBeGreaterThan(0);
      expect(mockSelfCheckError).toHaveBeenCalled();
    });

    it('calls createSelfCheckReport with correct arguments', async () => {
      await runSelfCheck();
      expect(mockCreateSelfCheckReport).toHaveBeenCalledWith(
        'repo-cortex-mcp',
        expect.any(Array),
        expect.objectContaining({
          tools_checked: expect.any(Array),
          tool_checks_count: expect.any(Number),
          all_ok: expect.any(Boolean),
        }),
      );
    });
  });

  /* ====================== main() via dynamic import ====================== */

  describe('main() CLI entrypoint', () => {
    /**
     * Helper: dynamically import the module with a unique query param so
     * the module re-evaluates and main() runs (because process.argv[1]
     * matches import.meta.url via the mocked pathToFileURL).
     *
     * @param {string} tag - Unique query-param tag.
     * @param {string[]} extraArgs - Extra CLI args after the script path.
     */
    async function importAndRunMain(tag, extraArgs = []) {
      // Use the real file path (not a file:// URL) for process.argv[1] so
      // the real pathToFileURL in the source module converts it correctly.
      // jest.resetModules() clears the cache so the module re-evaluates
      // and main() runs (because process.argv[1] resolves to import.meta.url).
      process.argv = ['node', moduleAbsPath, ...extraArgs];
      jest.resetModules();
      await import('./repo-cortex-mcp.mjs');
    }

    it('prints help with --help flag', async () => {
      mockParseMcpCliArgs.mockReturnValue({ help: true });
      await importAndRunMain('help', ['--help']);
      expect(mockPrintMcpUsage).toHaveBeenCalledTimes(1);
      expect(mockPrintMcpUsage).toHaveBeenCalledWith(
        expect.objectContaining({
          title: 'Repo Cortex MCP server',
          entrypoint: 'scripts/mcp-semantic/repo_cortex_mcp.mjs',
        }),
      );
    });

    it('emits self-check report with --self-check flag', async () => {
      mockParseMcpCliArgs.mockReturnValue({ selfCheck: true, json: false });
      mockCreateSelfCheckReport.mockReturnValue({ ok: true, issues: [] });
      await importAndRunMain('selfcheck', ['--self-check']);
      expect(mockEmitSelfCheckReport).toHaveBeenCalledTimes(1);
      expect(mockEmitSelfCheckReport).toHaveBeenCalledWith(
        expect.objectContaining({ ok: true }),
        { json: false },
      );
    });

    it('starts stdio server without flags', async () => {
      mockParseMcpCliArgs.mockReturnValue({});
      mockCreateMcpServer.mockReturnValue({ server: 'mock' });
      await importAndRunMain('normal');
      expect(mockRunStdioMcpServer).toHaveBeenCalledTimes(1);
      expect(mockRunStdioMcpServer).toHaveBeenCalledWith({ server: 'mock' });
    });

    /* --- parseDatabasePath branches --- */

    it('parses --databasePath flag', async () => {
      mockParseMcpCliArgs.mockReturnValue({ help: true });
      await importAndRunMain('dbpath', ['--databasePath=/custom/db', '--help']);
      // databasePath is passed to createRepoCortexTools via printMcpUsage tools
      expect(mockPrintMcpUsage).toHaveBeenCalledWith(
        expect.objectContaining({
          tools: expect.any(Array),
        }),
      );
    });

    it('parses --database flag (fallback)', async () => {
      mockParseMcpCliArgs.mockReturnValue({ help: true });
      await importAndRunMain('dbflag', ['--database=/alt/db', '--help']);
      expect(mockPrintMcpUsage).toHaveBeenCalled();
    });

    it('returns undefined when neither database flag is present', async () => {
      mockParseMcpCliArgs.mockReturnValue({ help: true });
      await importAndRunMain('nodb', ['--help']);
      expect(mockPrintMcpUsage).toHaveBeenCalled();
    });
  });
});

describe('repo-cortex-mcp self-heal response augmentation', () => {
  const mockEvaluateSelfHeal = jest.fn();
  let localCreateRepoCortexTools;

  beforeEach(async () => {
    jest.resetModules();
    mockEvaluateSelfHeal.mockReset();
    mockEvaluateSelfHeal.mockResolvedValue({
      action: 'started',
      guidanceFields: {
        state: 'model-only',
        reason: 'embeddings missing',
        action: 'started',
        attempt: 1,
        max_attempts: 3,
        cooldown_s: 600,
        next_allowed_at: 0,
        est_duration_min: 1,
        manual_recovery: null,
        guidance: 'self-heal guidance text',
      },
      spawnDecision: { pid: 123, command: 'node cortex-self-heal.mjs' },
    });
    jest.unstable_mockModule(
      '../agent-customization/cortex/cortex-health-guard.mjs',
      () => ({
        evaluateSelfHeal: mockEvaluateSelfHeal,
        __esModule: true,
      }),
    );

    // Keep createTool transparent so the handler itself is returned.
    mockCreateTool.mockReset();
    mockCreateTool.mockImplementation((obj) => obj);

    const mod = await import('./repo-cortex-mcp.mjs');
    localCreateRepoCortexTools = mod.createRepoCortexTools;
  });

  it('search_corpus handler augments a degraded response with self_heal when the tool omits it', async () => {
    mockSearchCorpus.mockReset();
    mockSearchCorpus.mockResolvedValue({
      query: 'test',
      results: [],
      dense_degraded: true,
      dense_state: 'model-only',
      dense_reason: 'embeddings missing',
    });

    const tools = localCreateRepoCortexTools();
    const tool = tools.find((t) => t.name === 'search_corpus');
    const response = await tool.handler({ query: 'test' });

    expect(response.self_heal).toBeDefined();
    expect(response.self_heal.action).toBe('started');
  });

  it('search_context handler augments a degraded response with self_heal when the tool omits it', async () => {
    mockSearchContext.mockReset();
    mockSearchContext.mockResolvedValue({
      context: 'bm25-only context',
      token_count: 10,
      tier_counts: { essential: 1, supporting: 0, supplementary: 0 },
      dense_degraded: true,
      dense_state: 'model-only',
      dense_reason: 'embeddings missing',
    });

    const tools = localCreateRepoCortexTools();
    const tool = tools.find((t) => t.name === 'search_context');
    const response = await tool.handler({ query: 'test' });

    expect(response.self_heal).toBeDefined();
    expect(response.self_heal.action).toBe('started');
  });

  it('index_stats handler augments the response with self_heal on a dense mismatch', async () => {
    mockIndexStats.mockReset();
    mockIndexStats.mockResolvedValue({
      total_documents: 1,
      total_chunks: 2,
      total_families: 1,
      last_build_timestamp: new Date(1000).toISOString(),
      feedback_stats: { total_events: 0, events_by_type: {} },
      ann: {
        strategy: 'none',
        threshold: 50000,
        current_chunk_count: 2,
        build_status: 'not_applicable',
      },
    });

    const tools = localCreateRepoCortexTools();
    const tool = tools.find((t) => t.name === 'index_stats');
    const response = await tool.handler({});

    expect(response.self_heal).toBeDefined();
    expect(response.self_heal.action).toMatch(
      /started|in_flight|cooldown|exhausted|disabled/,
    );
  });

  it('returns a non-object tool response unchanged', async () => {
    mockSearchCorpus.mockReset();
    mockSearchCorpus.mockResolvedValue(null);

    const tools = localCreateRepoCortexTools();
    const tool = tools.find((t) => t.name === 'search_corpus');
    const response = await tool.handler({ query: 'test' });

    expect(response).toBeNull();
  });

  it('makes an existing self_heal property enumerable on the wire', async () => {
    mockSearchCorpus.mockReset();
    mockSearchCorpus.mockResolvedValue({
      query: 'test',
      results: [],
      dense_state: 'model-only',
      self_heal: {
        action: 'started',
        guidance: 'tool-level self-heal',
      },
    });

    const tools = localCreateRepoCortexTools();
    const tool = tools.find((t) => t.name === 'search_corpus');
    const response = await tool.handler({ query: 'test' });

    expect(response.self_heal).toEqual(
      expect.objectContaining({ guidance: 'tool-level self-heal' }),
    );
    expect(
      Object.getOwnPropertyDescriptor(response, 'self_heal').enumerable,
    ).toBe(true);
  });

  it('skips self-heal augmentation for warm responses', async () => {
    mockSearchCorpus.mockReset();
    mockSearchCorpus.mockResolvedValue({
      query: 'test',
      results: [],
      dense_state: 'warm',
    });

    const tools = localCreateRepoCortexTools();
    const tool = tools.find((t) => t.name === 'search_corpus');
    const response = await tool.handler({ query: 'test' });

    expect(response.self_heal).toBeUndefined();
    expect(response.dense_state).toBe('warm');
  });

  it('falls back to deterministic guidance when the guard returns no guidanceFields', async () => {
    const originalDenseForceState = process.env.DENSE_FORCE_STATE;
    mockEvaluateSelfHeal.mockReset();
    mockEvaluateSelfHeal.mockResolvedValue({ action: 'noop' });
    delete process.env.DENSE_FORCE_STATE;

    try {
      mockSearchCorpus.mockReset();
      mockSearchCorpus.mockResolvedValue({
        query: 'test',
        results: [],
        dense_degraded: true,
        dense_state: 'model-only',
        dense_reason: 'embeddings missing',
      });

      const tools = localCreateRepoCortexTools();
      const tool = tools.find((t) => t.name === 'search_corpus');
      const response = await tool.handler({ query: 'test' });

      expect(response.self_heal).toEqual(
        expect.objectContaining({
          action: 'started',
          state: 'model-only',
          guidance: expect.stringMatching(/self-heal/i),
        }),
      );
    } finally {
      if (originalDenseForceState === undefined) {
        delete process.env.DENSE_FORCE_STATE;
      } else {
        process.env.DENSE_FORCE_STATE = originalDenseForceState;
      }
    }
  });

  it('falls back to deterministic guidance when the guard throws', async () => {
    mockEvaluateSelfHeal.mockReset();
    mockEvaluateSelfHeal.mockRejectedValue(new Error('guard failure'));

    mockSearchCorpus.mockReset();
    mockSearchCorpus.mockResolvedValue({
      query: 'test',
      results: [],
      dense_degraded: true,
      dense_state: 'model-only',
    });

    const tools = localCreateRepoCortexTools();
    const tool = tools.find((t) => t.name === 'search_corpus');
    const response = await tool.handler({ query: 'test' });

    expect(response.self_heal).toEqual(
      expect.objectContaining({ action: 'started' }),
    );
  });

  it('invokes the guard probe and omits test-only seams outside the test environment', async () => {
    const originalNodeEnv = process.env.NODE_ENV;
    const originalDenseForceState = process.env.DENSE_FORCE_STATE;
    process.env.NODE_ENV = 'production';
    delete process.env.DENSE_FORCE_STATE;

    try {
      mockEvaluateSelfHeal.mockReset();
      mockEvaluateSelfHeal.mockImplementation(async ({ probe }) => {
        const report = await probe();
        return {
          action: 'noop',
          guidanceFields: {
            state: report.state,
            reason: report.reason,
            action: 'noop',
            guidance: 'probe guidance',
          },
        };
      });

      mockSearchCorpus.mockReset();
      mockSearchCorpus.mockResolvedValue({
        query: 'test',
        results: [],
        dense_degraded: true,
        dense_state: 'model-only',
      });

      const tools = localCreateRepoCortexTools();
      const tool = tools.find((t) => t.name === 'search_corpus');
      const response = await tool.handler({ query: 'test' });

      const callArg = mockEvaluateSelfHeal.mock.calls[0][0];
      expect(callArg.spawner).toBeUndefined();
      expect(callArg.stateDir).toBeUndefined();
      expect(response.self_heal).toEqual(
        expect.objectContaining({
          action: 'noop',
          guidance: 'probe guidance',
        }),
      );
    } finally {
      process.env.NODE_ENV = originalNodeEnv;
      if (originalDenseForceState === undefined) {
        delete process.env.DENSE_FORCE_STATE;
      } else {
        process.env.DENSE_FORCE_STATE = originalDenseForceState;
      }
    }
  });

  it('uses explicit chunk_count and embedding_count when present', async () => {
    mockEvaluateSelfHeal.mockReset();
    mockEvaluateSelfHeal.mockImplementation(async ({ probe }) => {
      const report = await probe();
      return {
        action: 'started',
        guidanceFields: {
          state: report.state,
          reason: report.reason,
          action: 'started',
          guidance: 'count guidance',
        },
      };
    });

    mockSearchCorpus.mockReset();
    mockSearchCorpus.mockResolvedValue({
      query: 'test',
      results: [],
      dense_degraded: true,
      dense_state: 'model-only',
      chunk_count: 42,
      embedding_count: 7,
    });

    const tools = localCreateRepoCortexTools();
    const tool = tools.find((t) => t.name === 'search_corpus');
    const response = await tool.handler({ query: 'test' });

    expect(response.self_heal).toEqual(
      expect.objectContaining({ guidance: 'count guidance' }),
    );
    const callArg = mockEvaluateSelfHeal.mock.calls[0][0];
    const report = await callArg.probe();
    expect(report.chunk_count).toBe(42);
    expect(report.embedding_count).toBe(7);
  });

  it('exercises the test-only self-heal spawner seam', async () => {
    mockEvaluateSelfHeal.mockReset();
    mockEvaluateSelfHeal.mockImplementation(async (seams) => {
      const spawnResult = await seams.spawner();
      return {
        action: 'started',
        guidanceFields: {
          state: 'model-only',
          reason: 'embeddings missing',
          action: 'started',
          spawn_pid: spawnResult.pid,
          guidance: 'spawner exercised',
        },
      };
    });

    mockSearchCorpus.mockReset();
    mockSearchCorpus.mockResolvedValue({
      query: 'test',
      results: [],
      dense_degraded: true,
      dense_state: 'model-only',
    });

    const tools = localCreateRepoCortexTools();
    const tool = tools.find((t) => t.name === 'search_corpus');
    const response = await tool.handler({ query: 'test' });

    expect(response.self_heal).toEqual(
      expect.objectContaining({
        spawn_pid: 0,
        guidance: 'spawner exercised',
      }),
    );
  });
});
