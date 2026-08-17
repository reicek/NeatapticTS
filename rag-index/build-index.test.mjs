/**
 * @module build-index.test
 * @description 100% coverage tests for build-index.mjs.
 */

import { jest } from '@jest/globals';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const sourceFilePath = path.join(__dirname, 'build-index.mjs');

// Shared mock functions
const mockFg = jest.fn();
const mockReadFile = jest.fn();
const mockChunkMarkdownV2 = jest.fn();
const mockFail = jest.fn();
const mockParseCliArgs = jest.fn();
const mockPrintHelp = jest.fn();
const mockToRepoRelative = jest.fn();
const mockWriteJsonOrText = jest.fn();
const mockGetFreshnessProof = jest.fn();
const mockIsFreshDocument = jest.fn();
const mockInitSemanticIndex = jest.fn();
const mockEnrichChunkMetadata = jest.fn();
const mockEnrichDocumentMetadata = jest.fn();
const mockLoadCoverageReport = jest.fn();
const mockChunkTypeScriptSourcesV2 = jest.fn();
const mockBuildEntityGraph = jest.fn();

jest.unstable_mockModule('fast-glob', () => ({ default: mockFg }));
jest.unstable_mockModule('node:fs/promises', () => ({ readFile: mockReadFile }));
jest.unstable_mockModule('./chunker-v2.mjs', () => ({
  chunkMarkdownV2: mockChunkMarkdownV2,
}));
jest.unstable_mockModule('./cli-utils.mjs', () => ({
  fail: mockFail,
  parseCliArgs: mockParseCliArgs,
  printHelp: mockPrintHelp,
  toRepoRelative: mockToRepoRelative,
  writeJsonOrText: mockWriteJsonOrText,
}));
jest.unstable_mockModule('./freshness.mjs', () => ({
  getFreshnessProof: mockGetFreshnessProof,
  isFreshDocument: mockIsFreshDocument,
}));
jest.unstable_mockModule('./init-schema.mjs', () => ({
  defaultDatabasePath: '/fake/default-db.sqlite',
  initSemanticIndex: mockInitSemanticIndex,
  repoRoot: '/fake/repo',
}));
jest.unstable_mockModule('./metadata-enrichment.mjs', () => ({
  enrichChunkMetadata: mockEnrichChunkMetadata,
  enrichDocumentMetadata: mockEnrichDocumentMetadata,
  loadCoverageReport: mockLoadCoverageReport,
}));
jest.unstable_mockModule('./ts-chunker-v2.mjs', () => ({
  chunkTypeScriptSourcesV2: mockChunkTypeScriptSourcesV2,
}));
jest.unstable_mockModule('./build-entity-graph.mjs', () => ({
  buildEntityGraph: mockBuildEntityGraph,
}));

const { buildSemanticIndex } = await import('./build-index.mjs');

beforeEach(() => {
  jest.clearAllMocks();
  // Default mock implementations
  mockToRepoRelative.mockImplementation((p) => p);
  mockGetFreshnessProof.mockResolvedValue({
    mtime_ms: 1000, size: 100, sha256: 'abc',
  });
  mockIsFreshDocument.mockReturnValue(false);
  mockEnrichDocumentMetadata.mockReturnValue({
    arch_layer: 'core', test_coverage: 0.8, source_path_pattern: 'src/**',
  });
  mockEnrichChunkMetadata.mockReturnValue({
    arch_layer: 'core', jsdoc_quality: 'good', jsdoc_word_count: 10,
    cyclomatic_complexity: 5, test_coverage: 0.8, source_path_pattern: 'src/**',
  });
  mockLoadCoverageReport.mockResolvedValue({});
  mockChunkMarkdownV2.mockReturnValue([
    { body_text: 'chunk1', heading_path: 'h1', char_start: 0, char_end: 6, chunk_index: 0 },
  ]);
  mockChunkTypeScriptSourcesV2.mockResolvedValue([]);
});

// ---------------------------------------------------------------------------
// normalizeFiles (tested via buildSemanticIndex behavior)
// ---------------------------------------------------------------------------

describe('build-index: normalizeFiles', () => {
  it('returns undefined for undefined input', async () => {
    const result = await buildSemanticIndex({
      dryRun: true,
      corpusDocuments: [{ filePath: 'a.md', family: 'readme' }],
    });
    // dryRun returns summary without DB, normalizeFiles(undefined) → undefined → not targeted
    expect(result.scanned).toBe(1);
  });

  it('returns undefined for null input', async () => {
    const result = await buildSemanticIndex({
      dryRun: true,
      files: null,
      corpusDocuments: [{ filePath: 'a.md', family: 'readme' }],
    });
    expect(result.scanned).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// buildSemanticIndex — dry-run mode
// ---------------------------------------------------------------------------

describe('build-index: dry-run mode', () => {
  it('returns summary without DB operations', async () => {
    const docs = [{ filePath: 'a.md', family: 'readme' }];
    const result = await buildSemanticIndex({
      dryRun: true,
      corpusDocuments: docs,
      coverageReport: {},
    });
    expect(result.dryRun).toBe(true);
    expect(result.scanned).toBe(1);
    expect(result.indexed).toBe(0);
    expect(mockInitSemanticIndex).not.toHaveBeenCalled();
  });

  it('filters documents by files when provided', async () => {
    const docs = [
      { filePath: 'a.md', family: 'readme' },
      { filePath: 'b.md', family: 'readme' },
    ];
    const result = await buildSemanticIndex({
      dryRun: true,
      corpusDocuments: docs,
      files: ['a.md'],
      coverageReport: {},
    });
    expect(result.scanned).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// buildSemanticIndex — non-dry-run mode
// ---------------------------------------------------------------------------

function makeMockClient(existingDocs = [], docIdResult = { doc_id: 1 }) {
  const client = {
    execute: jest.fn().mockImplementation((params) => {
      const sql = params.sql || params;
      if (sql.includes('SELECT file_path FROM documents')) {
        return Promise.resolve({ rows: existingDocs });
      }
      if (sql.includes('SELECT file_path, mtime_ms')) {
        return Promise.resolve({ rows: [] }); // no existing doc
      }
      if (sql.includes('SELECT doc_id FROM documents')) {
        return Promise.resolve({ rows: [docIdResult] });
      }
      if (sql.includes('DELETE FROM documents')) {
        return Promise.resolve({});
      }
      if (sql.includes('INSERT INTO documents')) {
        return Promise.resolve({});
      }
      return Promise.resolve({ rows: [] });
    }),
    batch: jest.fn().mockResolvedValue([{ lastInsertRowid: 1 }]),
    close: jest.fn().mockResolvedValue(undefined),
  };
  return client;
}

describe('build-index: non-dry-run mode', () => {
  it('creates and closes client when not provided', async () => {
    const mockClient = makeMockClient();
    mockInitSemanticIndex.mockResolvedValue(mockClient);
    const docs = [{ filePath: 'a.md', family: 'readme' }];
    const result = await buildSemanticIndex({
      corpusDocuments: docs,
      coverageReport: {},
    });
    expect(result.indexed).toBe(1);
    expect(mockClient.close).toHaveBeenCalled();
  });

  it('does not close client when provided', async () => {
    const mockClient = makeMockClient();
    const docs = [{ filePath: 'a.md', family: 'readme' }];
    const result = await buildSemanticIndex({
      client: mockClient,
      corpusDocuments: docs,
      coverageReport: {},
    });
    expect(result.indexed).toBe(1);
    expect(mockClient.close).not.toHaveBeenCalled();
  });

  it('purges stale documents when not targeted', async () => {
    const mockClient = makeMockClient([{ file_path: 'stale.md' }]);
    mockInitSemanticIndex.mockResolvedValue(mockClient);
    const docs = [{ filePath: 'a.md', family: 'readme' }];
    const result = await buildSemanticIndex({
      corpusDocuments: docs,
      coverageReport: {},
    });
    expect(result.purged).toBe(1);
  });

  it('skips purge when targeted', async () => {
    const mockClient = makeMockClient([{ file_path: 'stale.md' }]);
    mockInitSemanticIndex.mockResolvedValue(mockClient);
    const docs = [{ filePath: 'a.md', family: 'readme' }];
    const result = await buildSemanticIndex({
      corpusDocuments: docs,
      files: ['a.md'],
      coverageReport: {},
    });
    expect(result.purged).toBe(0);
  });

  it('skips fresh documents', async () => {
    mockIsFreshDocument.mockReturnValue(true);
    const mockClient = makeMockClient();
    mockInitSemanticIndex.mockResolvedValue(mockClient);
    const docs = [{ filePath: 'a.md', family: 'readme' }];
    const result = await buildSemanticIndex({
      corpusDocuments: docs,
      coverageReport: {},
    });
    expect(result.skipped).toBe(1);
    expect(result.indexed).toBe(0);
  });

  it('force re-indexes fresh documents', async () => {
    mockIsFreshDocument.mockReturnValue(true);
    const mockClient = makeMockClient();
    mockInitSemanticIndex.mockResolvedValue(mockClient);
    const docs = [{ filePath: 'a.md', family: 'readme' }];
    const result = await buildSemanticIndex({
      force: true,
      corpusDocuments: docs,
      coverageReport: {},
    });
    expect(result.indexed).toBe(1);
  });

  it('increments newDocuments for new docs', async () => {
    const mockClient = makeMockClient();
    mockInitSemanticIndex.mockResolvedValue(mockClient);
    const docs = [{ filePath: 'a.md', family: 'readme' }];
    const result = await buildSemanticIndex({
      corpusDocuments: docs,
      coverageReport: {},
    });
    expect(result.newDocuments).toBe(1);
  });

  it('handles ts-source documents', async () => {
    const tsChunk = {
      body_text: 'code', heading_path: 'func', char_start: 0, char_end: 4,
      chunk_index: 0, file_path: 'src/test.ts',
    };
    mockChunkTypeScriptSourcesV2.mockResolvedValue([tsChunk]);
    const mockClient = makeMockClient();
    mockInitSemanticIndex.mockResolvedValue(mockClient);
    const docs = [{ filePath: 'src/test.ts', family: 'ts-source' }];
    const result = await buildSemanticIndex({
      corpusDocuments: docs,
      coverageReport: {},
    });
    expect(result.indexed).toBe(1);
    expect(result.chunks).toBe(1);
  });

  it('handles parent and child chunks', async () => {
    const chunks = [
      { body_text: 'parent', heading_path: 'h', char_start: 0, char_end: 6, chunk_index: 0 },
      { body_text: 'child', heading_path: 'h', char_start: 0, char_end: 5, chunk_index: 1, parent_chunk_id: 0 },
    ];
    mockChunkMarkdownV2.mockReturnValue(chunks);
    const mockClient = makeMockClient();
    mockInitSemanticIndex.mockResolvedValue(mockClient);
    const docs = [{ filePath: 'a.md', family: 'readme' }];
    const result = await buildSemanticIndex({
      corpusDocuments: docs,
      coverageReport: {},
    });
    expect(result.indexed).toBe(1);
    expect(result.chunks).toBe(2);
  });

  it('handles chunks without chunk_index (uses indexOf)', async () => {
    const chunks = [
      { body_text: 'parent', heading_path: 'h', char_start: 0, char_end: 6 },
      { body_text: 'child', heading_path: 'h', char_start: 0, char_end: 5, parent_chunk_id: 0 },
    ];
    mockChunkMarkdownV2.mockReturnValue(chunks);
    const mockClient = makeMockClient();
    mockInitSemanticIndex.mockResolvedValue(mockClient);
    const docs = [{ filePath: 'a.md', family: 'readme' }];
    const result = await buildSemanticIndex({
      corpusDocuments: docs,
      coverageReport: {},
    });
    expect(result.chunks).toBe(2);
  });

  it('handles empty documents list', async () => {
    const mockClient = makeMockClient();
    mockInitSemanticIndex.mockResolvedValue(mockClient);
    const result = await buildSemanticIndex({
      corpusDocuments: [],
      coverageReport: {},
    });
    expect(result.indexed).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// collectCorpusDocuments (via buildSemanticIndex without corpusDocuments)
// ---------------------------------------------------------------------------

describe('build-index: collectCorpusDocuments', () => {
  it('scans corpus sources via fast-glob', async () => {
    mockFg.mockResolvedValue(['src/README.md']);
    mockLoadCoverageReport.mockResolvedValue({});
    mockInitSemanticIndex.mockResolvedValue(makeMockClient());
    const result = await buildSemanticIndex({
      dryRun: true,
      coverageReport: {},
    });
    expect(result.scanned).toBeGreaterThanOrEqual(1);
    expect(mockFg).toHaveBeenCalled();
  });
});

// ---------------------------------------------------------------------------
// main() — CLI guard
// ---------------------------------------------------------------------------

describe('build-index: main()', () => {
  const origArgv1 = process.argv[1];
  const origExitCode = process.exitCode;

  afterEach(() => {
    process.argv[1] = origArgv1;
    process.exitCode = origExitCode;
  });

  it('prints help when --help is passed', async () => {
    mockParseCliArgs.mockReturnValue({ help: true });
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./build-index.mjs');
    });
    expect(mockPrintHelp).toHaveBeenCalledTimes(1);
  });

  it('emits JSON health summary on success', async () => {
    mockParseCliArgs.mockReturnValue({
      'json-health': true,
      _: [],
    });
    mockFg.mockResolvedValue([]);
    mockLoadCoverageReport.mockResolvedValue({});
    const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./build-index.mjs');
    });
    expect(logSpy).toHaveBeenCalled();
    const logged = JSON.parse(logSpy.mock.calls[0][0]);
    expect(logged.status).toBe('ok');
    logSpy.mockRestore();
  });

  it('emits JSON health failure on error', async () => {
    mockParseCliArgs.mockReturnValue({
      'json-health': true,
      _: [],
    });
    mockFg.mockRejectedValue(new Error('glob failed'));
    mockLoadCoverageReport.mockResolvedValue({});
    const logSpy = jest.spyOn(console, 'log').mockImplementation(() => {});
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./build-index.mjs');
    });
    expect(logSpy).toHaveBeenCalled();
    const logged = JSON.parse(logSpy.mock.calls[0][0]);
    expect(logged.status).toBe('error');
    expect(logged.message).toBe('glob failed');
    expect(process.exitCode).toBe(1);
    logSpy.mockRestore();
  });

  it('writes text output on success', async () => {
    mockParseCliArgs.mockReturnValue({
      json: false,
      _: [],
    });
    mockFg.mockResolvedValue([]);
    mockLoadCoverageReport.mockResolvedValue({});
    mockInitSemanticIndex.mockResolvedValue(makeMockClient());
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./build-index.mjs');
    });
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
  });

  it('writes JSON output on success', async () => {
    mockParseCliArgs.mockReturnValue({
      json: true,
      _: [],
    });
    mockFg.mockResolvedValue([]);
    mockLoadCoverageReport.mockResolvedValue({});
    mockInitSemanticIndex.mockResolvedValue(makeMockClient());
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./build-index.mjs');
    });
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
    expect(mockWriteJsonOrText.mock.calls[0][1]).toBe(true);
  });

  it('builds entity graph with --with-graph', async () => {
    mockParseCliArgs.mockReturnValue({
      'with-graph': true,
      json: false,
      _: [],
    });
    mockFg.mockResolvedValue([]);
    mockLoadCoverageReport.mockResolvedValue({});
    mockInitSemanticIndex.mockResolvedValue(makeMockClient());
    mockBuildEntityGraph.mockResolvedValue({ entities: 5, edges: 10 });
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./build-index.mjs');
    });
    expect(mockBuildEntityGraph).toHaveBeenCalledTimes(1);
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(2);
  });

  it('calls fail on Error (non-json-health)', async () => {
    mockParseCliArgs.mockReturnValue({
      json: false,
      _: [],
    });
    mockFg.mockRejectedValue(new Error('scan failed'));
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./build-index.mjs');
    });
    expect(mockFail).toHaveBeenCalledTimes(1);
    expect(mockFail.mock.calls[0][0]).toBe('scan failed');
  });

  it('calls fail with String(error) for non-Error (non-json-health)', async () => {
    mockParseCliArgs.mockReturnValue({
      json: true,
      _: [],
    });
    mockFg.mockRejectedValue('string error');
    process.argv[1] = sourceFilePath;
    await jest.isolateModulesAsync(async () => {
      await import('./build-index.mjs');
    });
    expect(mockFail).toHaveBeenCalledTimes(1);
    expect(mockFail.mock.calls[0][0]).toBe('string error');
  });
});