import { jest } from '@jest/globals';
import path from 'node:path';

const REPO_ROOT = 'C:\\NeatapticTS';
const FIXTURES = path.join(REPO_ROOT, 'rag-index', '__tests__', 'fixtures');
const SRC_FIXTURES = path.join(REPO_ROOT, 'src', '__rag_test__');

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------
const mockParseCliArgs = jest.fn();
const mockPrintHelp = jest.fn();
const mockWriteJsonOrText = jest.fn();
const mockFail = jest.fn();

jest.unstable_mockModule('./cli-utils.mjs', () => ({
  parseCliArgs: mockParseCliArgs,
  printHelp: mockPrintHelp,
  writeJsonOrText: mockWriteJsonOrText,
  fail: mockFail,
  toRepoRelative: jest.fn((filePath) =>
    path.relative(REPO_ROOT, String(filePath)).replaceAll(path.sep, '/'),
  ),
}));

jest.unstable_mockModule('./init-schema.mjs', () => ({ repoRoot: REPO_ROOT }));

const mockGlob = jest.fn();
jest.unstable_mockModule('fast-glob', () => ({ default: mockGlob }));

const { chunkTypeScriptSourcesV2, main } = await import('./ts-chunker-v2.mjs');

// ---------------------------------------------------------------------------
// chunkTypeScriptSourcesV2
// ---------------------------------------------------------------------------
describe('chunkTypeScriptSourcesV2', () => {
  it('chunks small declarations as single depth=0 chunks', async () => {
    const chunks = await chunkTypeScriptSourcesV2({
      sourcePaths: [path.join(FIXTURES, 'simple.ts')],
    });
    expect(chunks.length).toBeGreaterThan(0);
    const doubleChunk = chunks.find((c) => c.symbol_name === 'double');
    expect(doubleChunk).toBeDefined();
    expect(doubleChunk.depth).toBe(0);
    expect(doubleChunk.parent_chunk_id).toBeNull();
    expect(doubleChunk.chunk_index).toBeDefined();
    expect(doubleChunk.context_header).toContain('double');
    expect(doubleChunk.module_path).toBeDefined();
    expect(doubleChunk.export_type).toBe('function');
  });

  it('produces module-index chunk for re-export files', async () => {
    const chunks = await chunkTypeScriptSourcesV2({
      sourcePaths: [path.join(FIXTURES, 'reexports.ts')],
    });
    const moduleIndexChunk = chunks.find(
      (c) => c.symbol_name === 'module-index',
    );
    expect(moduleIndexChunk).toBeDefined();
    expect(moduleIndexChunk.export_type).toBe('reexport');
    expect(moduleIndexChunk.heading_path).toBe('module-index');
    expect(moduleIndexChunk.body_text).toContain('Re-exports:');
  });

  it('creates class parent chunk and method sub-chunks', async () => {
    const chunks = await chunkTypeScriptSourcesV2({
      sourcePaths: [path.join(FIXTURES, 'simple.ts')],
    });
    const calcChunks = chunks.filter(
      (c) => c.symbol_name === 'Calculator' || c.heading_path?.startsWith('Calculator >'),
    );
    expect(calcChunks.length).toBeGreaterThan(1);
    const parentChunk = calcChunks.find((c) => c.depth === 0);
    expect(parentChunk).toBeDefined();
    expect(parentChunk.export_type).toBe('class');
    const methodChunks = calcChunks.filter((c) => c.depth === 1);
    expect(methodChunks.length).toBeGreaterThan(0);
    for (const methodChunk of methodChunks) {
      expect(methodChunk.parent_chunk_id).toBe(parentChunk.chunk_index);
      expect(methodChunk.export_type).toBe('method');
    }
  });

  it('groups small methods into a small-methods sub-chunk', async () => {
    const chunks = await chunkTypeScriptSourcesV2({
      sourcePaths: [path.join(FIXTURES, 'simple.ts')],
    });
    const smallMethodsChunk = chunks.find(
      (c) => c.heading_path === 'Calculator > small-methods',
    );
    expect(smallMethodsChunk).toBeDefined();
    expect(smallMethodsChunk.depth).toBe(1);
  });

  it('creates property sub-chunks for large interfaces', async () => {
    const chunks = await chunkTypeScriptSourcesV2({
      sourcePaths: [path.join(FIXTURES, 'simple.ts')],
    });
    const dataChunks = chunks.filter(
      (c) => c.symbol_name === 'DataRecord' ||
        c.heading_path?.startsWith('DataRecord >'),
    );
    const parentChunk = dataChunks.find((c) => c.depth === 0);
    expect(parentChunk).toBeDefined();
    expect(parentChunk.export_type).toBe('interface');
    const propChunks = dataChunks.filter(
      (c) => c.depth === 1 && c.symbol_name === 'properties',
    );
    expect(propChunks.length).toBeGreaterThan(0);
  });

  it('assigns sequential chunk indices within each file', async () => {
    const chunks = await chunkTypeScriptSourcesV2({
      sourcePaths: [path.join(FIXTURES, 'simple.ts')],
    });
    for (let i = 0; i < chunks.length; i++) {
      expect(chunks[i].chunk_index).toBe(i);
    }
  });

  it('splits large functions at statement-group boundaries', async () => {
    const chunks = await chunkTypeScriptSourcesV2({
      sourcePaths: [path.join(SRC_FIXTURES, 'large.ts')],
    });
    const largeChunks = chunks.filter(
      (c) => c.symbol_name === 'veryLargeFunction',
    );
    expect(largeChunks.length).toBeGreaterThan(1);
    for (const chunk of largeChunks) {
      expect(chunk.depth).toBe(0);
      expect(chunk.body_text.length).toBeLessThanOrEqual(2048);
    }
  });

  it('handles small function in large.ts', async () => {
    const chunks = await chunkTypeScriptSourcesV2({
      sourcePaths: [path.join(SRC_FIXTURES, 'large.ts')],
    });
    const smallChunk = chunks.find((c) => c.symbol_name === 'smallFunc');
    expect(smallChunk).toBeDefined();
    expect(smallChunk.depth).toBe(0);
  });

  it('handles multiple files', async () => {
    const chunks = await chunkTypeScriptSourcesV2({
      sourcePaths: [
        path.join(FIXTURES, 'simple.ts'),
        path.join(FIXTURES, 'reexports.ts'),
      ],
    });
    expect(chunks.length).toBeGreaterThan(0);
    const filePaths = new Set(chunks.map((c) => c.file_path));
    expect(filePaths.size).toBeGreaterThanOrEqual(2);
  });

  it('produces correct context headers for methods', async () => {
    const chunks = await chunkTypeScriptSourcesV2({
      sourcePaths: [path.join(FIXTURES, 'simple.ts')],
    });
    const methodChunk = chunks.find(
      (c) => c.depth === 1 && c.symbol_name === 'addLarge',
    );
    if (methodChunk) {
      expect(methodChunk.context_header).toContain('Calculator');
      expect(methodChunk.context_header).toContain('addLarge');
    }
  });

  it('handles empty source files', async () => {
    const chunks = await chunkTypeScriptSourcesV2({
      sourcePaths: [path.join(SRC_FIXTURES, 'index.ts')],
    });
    expect(chunks.length).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
describe('ts-chunker-v2 main', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockGlob.mockReset();
  });

  it('prints help and returns when --help is passed', async () => {
    mockParseCliArgs.mockReturnValue({ help: true });
    await main();
    expect(mockPrintHelp).toHaveBeenCalledTimes(1);
  });

  it('runs successfully with --source in text mode', async () => {
    mockParseCliArgs.mockReturnValue({
      source: path.join(FIXTURES, 'simple.ts'),
    });
    await main();
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
    expect(mockWriteJsonOrText.mock.calls[0][1]).toBe(false);
  });

  it('runs successfully with --json flag', async () => {
    mockParseCliArgs.mockReturnValue({
      source: path.join(FIXTURES, 'simple.ts'),
      json: true,
    });
    await main();
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
    expect(mockWriteJsonOrText.mock.calls[0][1]).toBe(true);
  });

  it('runs with _ array sources', async () => {
    mockParseCliArgs.mockReturnValue({
      _: [path.join(FIXTURES, 'simple.ts')],
    });
    await main();
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
  });

  it('catches Error and calls fail', async () => {
    mockParseCliArgs.mockReturnValue({});
    mockGlob.mockRejectedValue(new Error('Glob failed'));
    await main();
    expect(mockFail).toHaveBeenCalledTimes(1);
    expect(mockFail.mock.calls[0][0]).toBe('Glob failed');
  });

  it('catches non-Error throw and calls fail with String()', async () => {
    mockParseCliArgs.mockReturnValue({});
    mockGlob.mockRejectedValue('string error');
    await main();
    expect(mockFail).toHaveBeenCalledTimes(1);
    expect(mockFail.mock.calls[0][0]).toBe('string error');
  });
});