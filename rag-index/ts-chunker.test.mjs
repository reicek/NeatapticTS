import { jest } from '@jest/globals';
import path from 'node:path';
import fs from 'node:fs';
import { Node, Project } from 'ts-morph';

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

const {
  resolveTypeScriptSourcePaths,
  createTypeScriptProject,
  loadExportedTypeScriptDeclarations,
  chunkTypeScriptSources,
  resolveJsdocSummaryText,
  countWords,
  resolveSignatureText,
  main,
} = await import('./ts-chunker.mjs');

// ---------------------------------------------------------------------------
// resolveTypeScriptSourcePaths
// ---------------------------------------------------------------------------
describe('resolveTypeScriptSourcePaths', () => {
  beforeEach(() => mockGlob.mockReset());

  it('resolves and sorts explicit sourcePaths', async () => {
    const result = await resolveTypeScriptSourcePaths({
      sourcePaths: [
        path.join(SRC_FIXTURES, 'module-b.ts'),
        path.join(SRC_FIXTURES, 'module-a.ts'),
      ],
    });
    expect(result).toHaveLength(2);
    expect(result[0]).toContain('module-a');
    expect(result[1]).toContain('module-b');
    expect(mockGlob).not.toHaveBeenCalled();
  });

  it('falls back to glob when sourcePaths is empty array', async () => {
    mockGlob.mockResolvedValue([
      path.join(SRC_FIXTURES, 'module-a.ts'),
    ]);
    const result = await resolveTypeScriptSourcePaths({
      sourcePaths: [],
      patterns: ['src/__rag_test__/*.ts'],
      ignore: [],
    });
    expect(result).toHaveLength(1);
    expect(mockGlob).toHaveBeenCalledTimes(1);
  });

  it('falls back to glob when sourcePaths is undefined', async () => {
    mockGlob.mockResolvedValue([
      path.join(SRC_FIXTURES, 'module-b.ts'),
      path.join(SRC_FIXTURES, 'module-a.ts'),
    ]);
    const result = await resolveTypeScriptSourcePaths({
      patterns: ['src/__rag_test__/*.ts'],
      ignore: ['src/__rag_test__/index.ts'],
    });
    expect(result).toHaveLength(2);
    expect(result[0]).toContain('module-a');
    expect(result[1]).toContain('module-b');
  });

  it('uses default patterns and ignore when not specified', async () => {
    mockGlob.mockResolvedValue([]);
    await resolveTypeScriptSourcePaths();
    expect(mockGlob).toHaveBeenCalledWith(
      ['src/**/*.ts'],
      expect.objectContaining({
        cwd: REPO_ROOT,
        absolute: true,
        onlyFiles: true,
        dot: false,
        ignore: ['src/**/*.d.ts', 'src/**/*.test.ts', 'src/**/*.spec.ts'],
      }),
    );
  });
});

// ---------------------------------------------------------------------------
// createTypeScriptProject
// ---------------------------------------------------------------------------
describe('createTypeScriptProject', () => {
  it('creates a project with default tsconfig path', () => {
    const project = createTypeScriptProject();
    expect(project).toBeInstanceOf(Project);
  });

  it('creates a project with custom tsconfig path', () => {
    const customTsconfigPath = path.join(REPO_ROOT, 'custom-tsconfig.json');
    fs.writeFileSync(customTsconfigPath, JSON.stringify({ compilerOptions: {} }));
    try {
      const project = createTypeScriptProject({
        tsConfigFilePath: customTsconfigPath,
      });
      expect(project).toBeInstanceOf(Project);
    } finally {
      fs.rmSync(customTsconfigPath, { force: true });
    }
  });
});

// ---------------------------------------------------------------------------
// loadExportedTypeScriptDeclarations
// ---------------------------------------------------------------------------
describe('loadExportedTypeScriptDeclarations', () => {
  it('loads exported declarations from simple.ts', async () => {
    const result = await loadExportedTypeScriptDeclarations({
      sourcePaths: [path.join(FIXTURES, 'simple.ts')],
    });
    expect(result.length).toBeGreaterThan(0);
    const names = result.map((r) => r.symbol_name);
    expect(names).toContain('double');
    expect(names).toContain('Calculator');
    expect(names).toContain('DataRecord');
    expect(names).toContain('ID');
    expect(names).toContain('greet');
    expect(names).toContain('MAX_VALUE');
    expect(names).toContain('ValidationError');
    expect(names).toContain('Empty');
    expect(names).toContain('default');
  });

  it('loads re-exported declarations from reexports.ts', async () => {
    const result = await loadExportedTypeScriptDeclarations({
      sourcePaths: [path.join(FIXTURES, 'reexports.ts')],
    });
    const names = result.map((r) => r.symbol_name);
    expect(names).toContain('simpleNs');
  });

  it('deduplicates declarations from same file', async () => {
    const result = await loadExportedTypeScriptDeclarations({
      sourcePaths: [path.join(FIXTURES, 'simple.ts')],
    });
    const ids = result.map((r) => `${r.file_path}:${r.declaration.getStart()}`);
    const uniqueIds = new Set(ids);
    expect(ids.length).toBe(uniqueIds.size);
  });

  it('sorts by file_path then symbol_name', async () => {
    const result = await loadExportedTypeScriptDeclarations({
      sourcePaths: [
        path.join(FIXTURES, 'simple.ts'),
        path.join(FIXTURES, 'imports.ts'),
      ],
    });
    for (let i = 1; i < result.length; i++) {
      const prev = result[i - 1];
      const curr = result[i];
      const pathCmp = prev.file_path.localeCompare(curr.file_path, 'en');
      expect(pathCmp <= 0).toBe(true);
      if (pathCmp === 0) {
        expect(
          prev.symbol_name.localeCompare(curr.symbol_name, 'en') <= 0,
        ).toBe(true);
      }
    }
  });

  it('uses provided project instance', async () => {
    const project = createTypeScriptProject();
    const result = await loadExportedTypeScriptDeclarations({
      sourcePaths: [path.join(FIXTURES, 'simple.ts')],
      project,
    });
    expect(result.length).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// chunkTypeScriptSources
// ---------------------------------------------------------------------------
describe('chunkTypeScriptSources', () => {
  it('produces chunks for all exported declarations', async () => {
    const chunks = await chunkTypeScriptSources({
      sourcePaths: [path.join(FIXTURES, 'simple.ts')],
    });
    expect(chunks.length).toBeGreaterThan(0);
    for (const chunk of chunks) {
      expect(chunk.body_text).toBeDefined();
      expect(chunk.char_start).toBeGreaterThanOrEqual(0);
      expect(chunk.char_end).toBeGreaterThanOrEqual(chunk.char_start);
      expect(chunk.chunk_index).toBe(0);
      expect(chunk.doc_family).toBe('ts-source');
      expect(chunk.file_path).toBeDefined();
      expect(chunk.heading_path).toBeDefined();
      expect(chunk.jsdoc_text).toBeDefined();
      expect(chunk.signature_text).toBeDefined();
      expect(chunk.symbol_name).toBeDefined();
    }
  });

  it('includes JSDoc text for declarations with JSDoc', async () => {
    const chunks = await chunkTypeScriptSources({
      sourcePaths: [path.join(FIXTURES, 'simple.ts')],
    });
    const doubleChunk = chunks.find((c) => c.symbol_name === 'double');
    expect(doubleChunk).toBeDefined();
    expect(doubleChunk.jsdoc_text).toContain('simple exported function');
  });

  it('includes signature text for functions', async () => {
    const chunks = await chunkTypeScriptSources({
      sourcePaths: [path.join(FIXTURES, 'simple.ts')],
    });
    const doubleChunk = chunks.find((c) => c.symbol_name === 'double');
    expect(doubleChunk.signature_text).toContain('function double');
  });

  it('handles re-export files', async () => {
    const chunks = await chunkTypeScriptSources({
      sourcePaths: [path.join(FIXTURES, 'reexports.ts')],
    });
    expect(chunks.length).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// resolveJsdocSummaryText
// ---------------------------------------------------------------------------
describe('resolveJsdocSummaryText', () => {
  it('returns direct JSDoc description when present', async () => {
    const decls = await loadExportedTypeScriptDeclarations({
      sourcePaths: [path.join(FIXTURES, 'simple.ts')],
    });
    const doubleDecl = decls.find((d) => d.symbol_name === 'double');
    const jsdoc = resolveJsdocSummaryText(
      doubleDecl.declaration,
      doubleDecl.jsdoc_source_node,
    );
    expect(jsdoc).toContain('simple exported function');
  });

  it('returns empty string for declaration without JSDoc', async () => {
    const decls = await loadExportedTypeScriptDeclarations({
      sourcePaths: [path.join(FIXTURES, 'reexports.ts')],
    });
    const nsDecl = decls.find((d) => d.symbol_name === 'simpleNs');
    if (nsDecl) {
      const jsdoc = resolveJsdocSummaryText(
        nsDecl.declaration,
        nsDecl.jsdoc_source_node,
      );
      expect(jsdoc).toBe('');
    }
  });

  it('falls back to jsdocSourceNode when direct has no description', async () => {
    const decls = await loadExportedTypeScriptDeclarations({
      sourcePaths: [path.join(FIXTURES, 'reexports.ts')],
    });
    const nsDecl = decls.find((d) => d.symbol_name === 'simpleNs');
    if (nsDecl && nsDecl.jsdoc_source_node) {
      const jsdoc = resolveJsdocSummaryText(
        nsDecl.declaration,
        nsDecl.jsdoc_source_node,
      );
      // Both source file and export declaration have no JSDoc
      expect(jsdoc).toBe('');
    }
  });

  it('returns empty string when no jsdocSourceNode is provided', () => {
    const mockDecl = {
      getJsDocs: () => [{ getDescription: () => '' }],
    };
    const jsdoc = resolveJsdocSummaryText(mockDecl, null);
    expect(jsdoc).toBe('');
  });
});

// ---------------------------------------------------------------------------
// countWords
// ---------------------------------------------------------------------------
describe('countWords', () => {
  it('counts words in normal text', () => {
    expect(countWords('hello world foo')).toBe(3);
  });
  it('returns 0 for empty string', () => {
    expect(countWords('')).toBe(0);
  });
  it('handles multiple spaces', () => {
    expect(countWords('  hello   world  ')).toBe(2);
  });
  it('handles null/undefined', () => {
    expect(countWords(null)).toBe(0);
    expect(countWords(undefined)).toBe(0);
  });
  it('handles single word', () => {
    expect(countWords('hello')).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// resolveSignatureText
// ---------------------------------------------------------------------------
describe('resolveSignatureText', () => {
  it('returns signature for function declarations', async () => {
    const decls = await loadExportedTypeScriptDeclarations({
      sourcePaths: [path.join(FIXTURES, 'simple.ts')],
    });
    const doubleDecl = decls.find((d) => d.symbol_name === 'double');
    const sig = resolveSignatureText(doubleDecl.declaration);
    expect(sig).toContain('function double');
    expect(sig).not.toContain('return');
  });

  it('returns signature for class declarations', async () => {
    const decls = await loadExportedTypeScriptDeclarations({
      sourcePaths: [path.join(FIXTURES, 'simple.ts')],
    });
    const calcDecl = decls.find((d) => d.symbol_name === 'Calculator');
    const sig = resolveSignatureText(calcDecl.declaration);
    expect(sig).toContain('class Calculator');
  });

  it('returns signature for interface declarations', async () => {
    const decls = await loadExportedTypeScriptDeclarations({
      sourcePaths: [path.join(FIXTURES, 'simple.ts')],
    });
    const dataDecl = decls.find((d) => d.symbol_name === 'DataRecord');
    const sig = resolveSignatureText(dataDecl.declaration);
    expect(sig).toContain('interface DataRecord');
  });

  it('returns full text for type alias declarations', async () => {
    const decls = await loadExportedTypeScriptDeclarations({
      sourcePaths: [path.join(FIXTURES, 'simple.ts')],
    });
    const idDecl = decls.find((d) => d.symbol_name === 'ID');
    const sig = resolveSignatureText(idDecl.declaration);
    expect(sig).toContain('type ID');
  });

  it('returns parent statement for variable declarations', async () => {
    const decls = await loadExportedTypeScriptDeclarations({
      sourcePaths: [path.join(FIXTURES, 'simple.ts')],
    });
    const greetDecl = decls.find((d) => d.symbol_name === 'greet');
    const sig = resolveSignatureText(greetDecl.declaration);
    expect(sig).toContain('greet');
  });

  it('returns declaration text for other types (SourceFile)', async () => {
    const decls = await loadExportedTypeScriptDeclarations({
      sourcePaths: [path.join(FIXTURES, 'reexports.ts')],
    });
    const nsDecl = decls.find((d) => d.symbol_name === 'simpleNs');
    if (nsDecl) {
      const sig = resolveSignatureText(nsDecl.declaration);
      expect(typeof sig).toBe('string');
    }
  });
});

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
describe('ts-chunker main', () => {
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

  it('runs successfully with _ array for sources', async () => {
    mockParseCliArgs.mockReturnValue({
      _: [path.join(FIXTURES, 'simple.ts')],
    });
    await main();
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
  });

  it('runs successfully with both source and _ array', async () => {
    mockParseCliArgs.mockReturnValue({
      source: path.join(FIXTURES, 'simple.ts'),
      _: [path.join(FIXTURES, 'reexports.ts')],
    });
    await main();
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
  });

  it('runs successfully with _ as non-array (undefined)', async () => {
    mockParseCliArgs.mockReturnValue({
      source: path.join(FIXTURES, 'simple.ts'),
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