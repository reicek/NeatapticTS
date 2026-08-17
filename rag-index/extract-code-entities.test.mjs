import { jest } from '@jest/globals';
import path from 'node:path';

const REPO_ROOT = 'C:\\NeatapticTS';
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

const { extractCodeEntities, deriveModulePath, main } = await import(
  './extract-code-entities.mjs'
);

// ---------------------------------------------------------------------------
// deriveModulePath
// ---------------------------------------------------------------------------
describe('deriveModulePath', () => {
  it('strips .ts extension', () => {
    expect(deriveModulePath('src/neat.ts')).toBe('src/neat');
  });
  it('strips leading ./', () => {
    expect(deriveModulePath('./src/foo.ts')).toBe('src/foo');
  });
  it('strips leading /', () => {
    expect(deriveModulePath('/src/foo.ts')).toBe('src/foo');
  });
  it('handles nested paths', () => {
    expect(deriveModulePath('src/architecture/network/network.ts')).toBe(
      'src/architecture/network/network',
    );
  });
  it('handles index files', () => {
    expect(deriveModulePath('src/methods/selection/index.ts')).toBe(
      'src/methods/selection/index',
    );
  });
  it('handles no .ts extension', () => {
    expect(deriveModulePath('src/foo')).toBe('src/foo');
  });
});

// ---------------------------------------------------------------------------
// extractCodeEntities
// ---------------------------------------------------------------------------
describe('extractCodeEntities', () => {
  it('extracts module entities', async () => {
    const result = await extractCodeEntities({
      sourcePaths: [path.join(SRC_FIXTURES, 'module-a.ts')],
    });
    const moduleEntities = result.entities.filter(
      (e) => e.entity_type === 'module',
    );
    expect(moduleEntities.length).toBeGreaterThan(0);
    const moduleEntity = moduleEntities[0];
    expect(moduleEntity.qualified_name).toBe('src/__rag_test__/module-a');
    expect(moduleEntity.name).toBe('module-a');
    const metadata = JSON.parse(moduleEntity.extra_metadata);
    expect(metadata.symbol_count).toBeGreaterThan(0);
    expect(metadata.file_count).toBeGreaterThanOrEqual(1);
  });

  it('extracts function entities', async () => {
    const result = await extractCodeEntities({
      sourcePaths: [path.join(SRC_FIXTURES, 'module-a.ts')],
    });
    const funcEntities = result.entities.filter(
      (e) => e.entity_type === 'function',
    );
    expect(funcEntities.length).toBeGreaterThan(0);
    const helperFunc = funcEntities.find((e) => e.name === 'helperFunc');
    expect(helperFunc).toBeDefined();
    expect(helperFunc.qualified_name).toBe(
      'src/__rag_test__/module-a.helperFunc',
    );
  });

  it('extracts class entities', async () => {
    const result = await extractCodeEntities({
      sourcePaths: [path.join(SRC_FIXTURES, 'module-b.ts')],
    });
    const classEntities = result.entities.filter(
      (e) => e.entity_type === 'class',
    );
    const consumer = classEntities.find((e) => e.name === 'Consumer');
    expect(consumer).toBeDefined();
    expect(consumer.qualified_name).toBe(
      'src/__rag_test__/module-b.Consumer',
    );
  });

  it('extracts interface entities', async () => {
    const result = await extractCodeEntities({
      sourcePaths: [path.join(SRC_FIXTURES, 'module-a.ts')],
    });
    const ifaceEntities = result.entities.filter(
      (e) => e.entity_type === 'interface',
    );
    expect(ifaceEntities.length).toBeGreaterThan(0);
  });

  it('extracts type-alias entities', async () => {
    const result = await extractCodeEntities({
      sourcePaths: [path.join(SRC_FIXTURES, 'module-a.ts')],
    });
    const typeEntities = result.entities.filter(
      (e) => e.entity_type === 'type-alias',
    );
    const helperType = typeEntities.find((e) => e.name === 'HelperType');
    expect(helperType).toBeDefined();
  });

  it('extracts variable entities (non-function)', async () => {
    const result = await extractCodeEntities({
      sourcePaths: [path.join(SRC_FIXTURES, 'module-a.ts')],
    });
    const varEntities = result.entities.filter(
      (e) => e.entity_type === 'variable',
    );
    const helperConst = varEntities.find((e) => e.name === 'helperConst');
    expect(helperConst).toBeDefined();
  });

  it('extracts arrow function as function entity', async () => {
    const result = await extractCodeEntities({
      sourcePaths: [path.join(SRC_FIXTURES, 'module-a.ts')],
    });
    const funcEntities = result.entities.filter(
      (e) => e.entity_type === 'function',
    );
    const helperArrow = funcEntities.find((e) => e.name === 'helperArrow');
    expect(helperArrow).toBeDefined();
  });

  it('extracts error-class entities', async () => {
    const result = await extractCodeEntities({
      sourcePaths: [path.join(SRC_FIXTURES, 'module-a.ts')],
    });
    const errorEntities = result.entities.filter(
      (e) => e.entity_type === 'error-class',
    );
    const helperError = errorEntities.find((e) => e.name === 'HelperError');
    expect(helperError).toBeDefined();
  });

  it('extracts default export entity', async () => {
    const result = await extractCodeEntities({
      sourcePaths: [path.join(SRC_FIXTURES, 'module-a.ts')],
    });
    const defaultEntity = result.entities.find(
      (e) => e.name === 'helperDefault' || e.name === 'default',
    );
    expect(defaultEntity).toBeDefined();
  });

  it('extracts method entities for class methods', async () => {
    const result = await extractCodeEntities({
      sourcePaths: [path.join(SRC_FIXTURES, 'module-b.ts')],
    });
    const methodEntities = result.entities.filter(
      (e) =>
        e.entity_type === 'function' &&
        e.extra_metadata &&
        JSON.parse(e.extra_metadata).parent_class,
    );
    expect(methodEntities.length).toBeGreaterThan(0);
  });

  it('creates export edges from module to symbols', async () => {
    const result = await extractCodeEntities({
      sourcePaths: [path.join(SRC_FIXTURES, 'module-a.ts')],
    });
    const exportEdges = result.edges.filter(
      (e) => e.relationship === 'exports',
    );
    expect(exportEdges.length).toBeGreaterThan(0);
  });

  it('creates owns edges from module to symbols', async () => {
    const result = await extractCodeEntities({
      sourcePaths: [path.join(SRC_FIXTURES, 'module-a.ts')],
    });
    const ownsEdges = result.edges.filter((e) => e.relationship === 'owns');
    expect(ownsEdges.length).toBeGreaterThan(0);
  });

  it('creates part-of inverse edges from owns edges', async () => {
    const result = await extractCodeEntities({
      sourcePaths: [path.join(SRC_FIXTURES, 'module-a.ts')],
    });
    const partOfEdges = result.edges.filter(
      (e) => e.relationship === 'part-of',
    );
    expect(partOfEdges.length).toBeGreaterThan(0);
  });

  it('creates import edges between modules', async () => {
    const result = await extractCodeEntities({
      sourcePaths: [
        path.join(SRC_FIXTURES, 'module-a.ts'),
        path.join(SRC_FIXTURES, 'module-b.ts'),
      ],
    });
    const importEdges = result.edges.filter(
      (e) => e.relationship === 'imports',
    );
    expect(importEdges.length).toBeGreaterThan(0);
  });

  it('creates implements edges', async () => {
    const result = await extractCodeEntities({
      sourcePaths: [
        path.join(SRC_FIXTURES, 'module-a.ts'),
        path.join(SRC_FIXTURES, 'module-b.ts'),
      ],
    });
    const implementsEdges = result.edges.filter(
      (e) => e.relationship === 'implements',
    );
    expect(implementsEdges.length).toBeGreaterThan(0);
    const impl = implementsEdges.find(
      (e) => e.target_qualified_name.includes('HelperInterface'),
    );
    expect(impl).toBeDefined();
  });

  it('creates depends-on edges', async () => {
    const result = await extractCodeEntities({
      sourcePaths: [
        path.join(SRC_FIXTURES, 'module-a.ts'),
        path.join(SRC_FIXTURES, 'module-b.ts'),
      ],
    });
    const dependsOnEdges = result.edges.filter(
      (e) => e.relationship === 'depends-on',
    );
    expect(dependsOnEdges.length).toBeGreaterThan(0);
  });

  it('creates owns edges from class to methods', async () => {
    const result = await extractCodeEntities({
      sourcePaths: [path.join(SRC_FIXTURES, 'module-b.ts')],
    });
    const classToMethodOwns = result.edges.filter(
      (e) =>
        e.relationship === 'owns' &&
        e.source_qualified_name.includes('module-b.Consumer'),
    );
    expect(classToMethodOwns.length).toBeGreaterThan(0);
  });

  it('deduplicates entities by qualified_name', async () => {
    const result = await extractCodeEntities({
      sourcePaths: [
        path.join(SRC_FIXTURES, 'module-a.ts'),
        path.join(SRC_FIXTURES, 'index.ts'),
      ],
    });
    const qNames = result.entities.map((e) => e.qualified_name);
    const uniqueQNames = new Set(qNames);
    expect(qNames.length).toBe(uniqueQNames.size);
  });

  it('handles re-export barrel files', async () => {
    const result = await extractCodeEntities({
      sourcePaths: [
        path.join(SRC_FIXTURES, 'module-a.ts'),
        path.join(SRC_FIXTURES, 'index.ts'),
      ],
    });
    expect(result.entities.length).toBeGreaterThan(0);
  });

  it('returns moduleEntityMap and symbolEntityMap', async () => {
    const result = await extractCodeEntities({
      sourcePaths: [path.join(SRC_FIXTURES, 'module-a.ts')],
    });
    expect(result.moduleEntityMap).toBeInstanceOf(Map);
    expect(result.symbolEntityMap).toBeInstanceOf(Map);
    expect(result.moduleEntityMap.size).toBeGreaterThan(0);
    expect(result.symbolEntityMap.size).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
describe('extract-code-entities main', () => {
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
      source: path.join(SRC_FIXTURES, 'module-a.ts'),
    });
    await main();
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
    expect(mockWriteJsonOrText.mock.calls[0][1]).toBe(false);
  });

  it('runs successfully with --json flag', async () => {
    mockParseCliArgs.mockReturnValue({
      source: path.join(SRC_FIXTURES, 'module-a.ts'),
      json: true,
    });
    await main();
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
    expect(mockWriteJsonOrText.mock.calls[0][1]).toBe(true);
  });

  it('runs with _ array sources', async () => {
    mockParseCliArgs.mockReturnValue({
      _: [path.join(SRC_FIXTURES, 'module-a.ts')],
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