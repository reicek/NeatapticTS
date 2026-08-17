import { jest } from '@jest/globals';
import path from 'node:path';

// ---------------------------------------------------------------------------
// Mocks for main() tests
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
  toRepoRelative: jest.fn((p) => p),
}));

const mockExtractCodeEntities = jest.fn();
jest.unstable_mockModule('./extract-code-entities.mjs', () => ({
  extractCodeEntities: mockExtractCodeEntities,
  deriveModulePath: jest.fn(),
}));

const mockExtractDocEntities = jest.fn();
jest.unstable_mockModule('./extract-doc-entities.mjs', () => ({
  extractDocEntities: mockExtractDocEntities,
}));

const mockGlob = jest.fn();
jest.unstable_mockModule('fast-glob', () => ({ default: mockGlob }));

const mockReadFile = jest.fn();
jest.unstable_mockModule('node:fs/promises', () => ({ readFile: mockReadFile }));

jest.unstable_mockModule('./init-schema.mjs', () => ({ repoRoot: '/fake/repo' }));

const { extractCrossRefs, main } = await import('./extract-cross-refs.mjs');

// ---------------------------------------------------------------------------
// extractCrossRefs — pure function tests
// ---------------------------------------------------------------------------
describe('extractCrossRefs', () => {
  it('returns empty edges for no doc entities', () => {
    const result = extractCrossRefs({
      docEntities: [],
      codeEntityMap: new Map(),
      docEntityMap: new Map(),
      headingsByDoc: new Map(),
    });
    expect(result.edges).toEqual([]);
  });

  it('skips doc entities with no heading text', () => {
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/test' }],
      codeEntityMap: new Map(),
      docEntityMap: new Map(),
      headingsByDoc: new Map(),
    });
    expect(result.edges).toEqual([]);
  });

  it('skips doc entities with empty heading text', () => {
    const headingsByDoc = new Map([['plans/test', []]]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/test' }],
      codeEntityMap: new Map(),
      docEntityMap: new Map(),
      headingsByDoc,
    });
    expect(result.edges).toEqual([]);
  });

  it('extracts src/*.ts path references (direct module match)', () => {
    const codeEntityMap = new Map([
      ['src/neat', { qualified_name: 'src/neat', name: 'neat', entity_type: 'module' }],
    ]);
    const headingsByDoc = new Map([
      ['plans/test', ['See src/neat.ts for details.']],
    ]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/test' }],
      codeEntityMap,
      docEntityMap: new Map(),
      headingsByDoc,
    });
    expect(result.edges).toHaveLength(1);
    expect(result.edges[0].target_qualified_name).toBe('src/neat');
    expect(result.edges[0].confidence).toBe('medium');
    expect(result.edges[0].relationship).toBe('references');
  });

  it('extracts src/*.ts path references (directory module match)', () => {
    const codeEntityMap = new Map([
      ['src/architecture/network', { qualified_name: 'src/architecture/network', name: 'network', entity_type: 'module' }],
    ]);
    const headingsByDoc = new Map([
      ['plans/test', ['See src/architecture/network/network.ts.']],
    ]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/test' }],
      codeEntityMap,
      docEntityMap: new Map(),
      headingsByDoc,
    });
    expect(result.edges).toHaveLength(1);
    expect(result.edges[0].target_qualified_name).toBe('src/architecture/network');
  });

  it('extracts src/*.ts path references (no match)', () => {
    const codeEntityMap = new Map([
      ['src/other', { qualified_name: 'src/other', name: 'other', entity_type: 'module' }],
    ]);
    const headingsByDoc = new Map([
      ['plans/test', ['See src/nonexistent/file.ts.']],
    ]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/test' }],
      codeEntityMap,
      docEntityMap: new Map(),
      headingsByDoc,
    });
    expect(result.edges).toHaveLength(0);
  });

  it('extracts ClassName.methodName references', () => {
    const codeEntityMap = new Map([
      ['src/calc.Calculator.add', { qualified_name: 'src/calc.Calculator.add', name: 'add', entity_type: 'function' }],
    ]);
    const headingsByDoc = new Map([
      ['plans/test', ['Use Calculator.add to compute.']],
    ]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/test' }],
      codeEntityMap,
      docEntityMap: new Map(),
      headingsByDoc,
    });
    const methodEdge = result.edges.find((e) => e.target_qualified_name === 'src/calc.Calculator.add');
    expect(methodEdge).toBeDefined();
    expect(methodEdge.confidence).toBe('medium');
  });

  it('extracts ClassName references (class match)', () => {
    const codeEntityMap = new Map([
      ['src/calc.Calculator', { qualified_name: 'src/calc.Calculator', name: 'Calculator', entity_type: 'class' }],
    ]);
    const headingsByDoc = new Map([
      ['plans/test', ['The Calculator class handles math.']],
    ]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/test' }],
      codeEntityMap,
      docEntityMap: new Map(),
      headingsByDoc,
    });
    const classEdge = result.edges.find((e) => e.target_qualified_name === 'src/calc.Calculator');
    expect(classEdge).toBeDefined();
    expect(classEdge.confidence).toBe('low');
  });

  it('extracts ClassName references (error-class match)', () => {
    const codeEntityMap = new Map([
      ['src/errors.ValidationError', { qualified_name: 'src/errors.ValidationError', name: 'ValidationError', entity_type: 'error-class' }],
    ]);
    const headingsByDoc = new Map([
      ['plans/test', ['ValidationError is thrown on bad input.']],
    ]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/test' }],
      codeEntityMap,
      docEntityMap: new Map(),
      headingsByDoc,
    });
    const errEdge = result.edges.find((e) => e.target_qualified_name === 'src/errors.ValidationError');
    expect(errEdge).toBeDefined();
  });

  it('skips common non-class words in ClassName pattern', () => {
    const codeEntityMap = new Map([
      ['src/test.The', { qualified_name: 'src/test.The', name: 'The', entity_type: 'class' }],
    ]);
    const headingsByDoc = new Map([
      ['plans/test', ['The system works well.']],
    ]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/test' }],
      codeEntityMap,
      docEntityMap: new Map(),
      headingsByDoc,
    });
    // "The" should be skipped as a common word
    expect(result.edges.find((e) => e.target_qualified_name === 'src/test.The')).toBeUndefined();
  });

  it('extracts backtick-quoted code symbols (exact code match)', () => {
    const codeEntityMap = new Map([
      ['src/calc.double', { qualified_name: 'src/calc.double', name: 'double', entity_type: 'function' }],
    ]);
    const headingsByDoc = new Map([
      ['plans/test', ['Use `src/calc.double` function.']],
    ]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/test' }],
      codeEntityMap,
      docEntityMap: new Map(),
      headingsByDoc,
    });
    const edge = result.edges.find((e) => e.target_qualified_name === 'src/calc.double');
    expect(edge).toBeDefined();
    expect(edge.confidence).toBe('low');
  });

  it('extracts backtick-quoted code symbols (exact doc match)', () => {
    const docEntityMap = new Map([
      ['plans/other', { qualified_name: 'plans/other', name: 'other', entity_type: 'plan' }],
    ]);
    const headingsByDoc = new Map([
      ['plans/test', ['See `plans/other` for details.']],
    ]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/test' }],
      codeEntityMap: new Map(),
      docEntityMap,
      headingsByDoc,
    });
    const edge = result.edges.find((e) => e.target_qualified_name === 'plans/other');
    expect(edge).toBeDefined();
  });

  it('extracts backtick-quoted code symbols (fuzzy name match)', () => {
    const codeEntityMap = new Map([
      ['src/calc.greet', { qualified_name: 'src/calc.greet', name: 'greet', entity_type: 'function' }],
    ]);
    const headingsByDoc = new Map([
      ['plans/test', ['Call `greet` to say hello.']],
    ]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/test' }],
      codeEntityMap,
      docEntityMap: new Map(),
      headingsByDoc,
    });
    const edge = result.edges.find((e) => e.target_qualified_name === 'src/calc.greet');
    expect(edge).toBeDefined();
  });

  it('extracts backtick-quoted symbols (no match)', () => {
    const headingsByDoc = new Map([
      ['plans/test', ['Use `unknownSymbol` here.']],
    ]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/test' }],
      codeEntityMap: new Map(),
      docEntityMap: new Map(),
      headingsByDoc,
    });
    expect(result.edges).toHaveLength(0);
  });

  it('extracts plans/<plan-name> references', () => {
    const docEntityMap = new Map([
      ['plans/my_plan', { qualified_name: 'plans/my_plan', name: 'my_plan', entity_type: 'plan' }],
    ]);
    const headingsByDoc = new Map([
      ['plans/test', ['See plans/my-plan for details.']],
    ]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/test' }],
      codeEntityMap: new Map(),
      docEntityMap,
      headingsByDoc,
    });
    const edge = result.edges.find((e) => e.target_qualified_name === 'plans/my_plan');
    expect(edge).toBeDefined();
    expect(edge.confidence).toBe('medium');
  });

  it('extracts plans references with lowercase candidate', () => {
    const docEntityMap = new Map([
      ['plans/myplan', { qualified_name: 'plans/myplan', name: 'myplan', entity_type: 'plan' }],
    ]);
    const headingsByDoc = new Map([
      ['plans/test', ['See plans/MyPlan for details.']],
    ]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/test' }],
      codeEntityMap: new Map(),
      docEntityMap,
      headingsByDoc,
    });
    const edge = result.edges.find((e) => e.target_qualified_name === 'plans/myplan');
    expect(edge).toBeDefined();
  });

  it('extracts plans references with underscore candidate', () => {
    const docEntityMap = new Map([
      ['plans/my_plan', { qualified_name: 'plans/my_plan', name: 'my_plan', entity_type: 'plan' }],
    ]);
    const headingsByDoc = new Map([
      ['plans/test', ['See plans/my-plan.']],
    ]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/test' }],
      codeEntityMap: new Map(),
      docEntityMap,
      headingsByDoc,
    });
    expect(result.edges.find((e) => e.target_qualified_name === 'plans/my_plan')).toBeDefined();
  });

  it('skips plan references for non-plan entities', () => {
    const docEntityMap = new Map([
      ['plans/my_plan', { qualified_name: 'plans/my_plan', name: 'my_plan', entity_type: 'skill' }],
    ]);
    const headingsByDoc = new Map([
      ['plans/test', ['See plans/my-plan.']],
    ]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/test' }],
      codeEntityMap: new Map(),
      docEntityMap,
      headingsByDoc,
    });
    expect(result.edges.find((e) => e.target_qualified_name === 'plans/my_plan')).toBeUndefined();
  });

  it('extracts .github/skills/<skill-name> references', () => {
    const docEntityMap = new Map([
      ['skills/coverage-guard', { qualified_name: 'skills/coverage-guard', name: 'coverage-guard', entity_type: 'skill' }],
    ]);
    const headingsByDoc = new Map([
      ['plans/test', ['See .github/skills/coverage-guard.']],
    ]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/test' }],
      codeEntityMap: new Map(),
      docEntityMap,
      headingsByDoc,
    });
    const edge = result.edges.find((e) => e.target_qualified_name === 'skills/coverage-guard');
    expect(edge).toBeDefined();
    expect(edge.confidence).toBe('medium');
  });

  it('skips skill references for non-skill entities', () => {
    const docEntityMap = new Map([
      ['skills/coverage-guard', { qualified_name: 'skills/coverage-guard', name: 'coverage-guard', entity_type: 'plan' }],
    ]);
    const headingsByDoc = new Map([
      ['plans/test', ['See .github/skills/coverage-guard.']],
    ]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/test' }],
      codeEntityMap: new Map(),
      docEntityMap,
      headingsByDoc,
    });
    expect(result.edges.find((e) => e.target_qualified_name === 'skills/coverage-guard')).toBeUndefined();
  });

  it('deduplicates edges (same source→target)', () => {
    const codeEntityMap = new Map([
      ['src/neat', { qualified_name: 'src/neat', name: 'neat', entity_type: 'module' }],
    ]);
    const headingsByDoc = new Map([
      ['plans/test', ['src/neat.ts and src/neat.ts again.']],
    ]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/test' }],
      codeEntityMap,
      docEntityMap: new Map(),
      headingsByDoc,
    });
    // Two mentions of src/neat.ts but only one edge (deduped)
    const pathEdges = result.edges.filter((e) => e.target_qualified_name === 'src/neat');
    expect(pathEdges).toHaveLength(1);
  });

  it('handles getParentQualifiedName: heading entity with lowercase+hyphen suffix', () => {
    const headingsByDoc = new Map([
      ['plans/myplan', ['Some text with src/neat.ts.']],
    ]);
    const codeEntityMap = new Map([
      ['src/neat', { qualified_name: 'src/neat', name: 'neat', entity_type: 'module' }],
    ]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/myplan.some-heading' }],
      codeEntityMap,
      docEntityMap: new Map(),
      headingsByDoc,
    });
    expect(result.edges).toHaveLength(1);
  });

  it('handles getParentQualifiedName: heading entity with non-lowercase suffix', () => {
    // suffix 'Heading' is not all lowercase, so parent = full qualified name
    const headingsByDoc = new Map([
      ['plans/myplan.Heading', ['Some text with src/neat.ts.']],
    ]);
    const codeEntityMap = new Map([
      ['src/neat', { qualified_name: 'src/neat', name: 'neat', entity_type: 'module' }],
    ]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/myplan.Heading' }],
      codeEntityMap,
      docEntityMap: new Map(),
      headingsByDoc,
    });
    expect(result.edges).toHaveLength(1);
  });

  it('handles getParentQualifiedName: heading entity with no hyphen suffix', () => {
    // suffix 'nohyphen' is lowercase but has no hyphen, so parent = full name
    const headingsByDoc = new Map([
      ['plans/myplan.nohyphen', ['Some text with src/neat.ts.']],
    ]);
    const codeEntityMap = new Map([
      ['src/neat', { qualified_name: 'src/neat', name: 'neat', entity_type: 'module' }],
    ]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/myplan.nohyphen' }],
      codeEntityMap,
      docEntityMap: new Map(),
      headingsByDoc,
    });
    expect(result.edges).toHaveLength(1);
  });

  it('handles getParentQualifiedName: no dot after slash', () => {
    const headingsByDoc = new Map([
      ['plans/myplan', ['Some text with src/neat.ts.']],
    ]);
    const codeEntityMap = new Map([
      ['src/neat', { qualified_name: 'src/neat', name: 'neat', entity_type: 'module' }],
    ]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/myplan' }],
      codeEntityMap,
      docEntityMap: new Map(),
      headingsByDoc,
    });
    expect(result.edges).toHaveLength(1);
  });

  it('handles headingsByDoc being null/undefined', () => {
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/test' }],
      codeEntityMap: new Map(),
      docEntityMap: new Map(),
      headingsByDoc: undefined,
    });
    expect(result.edges).toEqual([]);
  });

  it('truncates scan text to 2000 chars', () => {
    const longText = 'X'.repeat(2500) + ' src/neat.ts';
    const headingsByDoc = new Map([
      ['plans/test', [longText]],
    ]);
    const codeEntityMap = new Map([
      ['src/neat', { qualified_name: 'src/neat', name: 'neat', entity_type: 'module' }],
    ]);
    const result = extractCrossRefs({
      docEntities: [{ qualified_name: 'plans/test' }],
      codeEntityMap,
      docEntityMap: new Map(),
      headingsByDoc,
    });
    // src/neat.ts is after char 2000, so it should be truncated and not found
    expect(result.edges).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// main — CLI entry point tests
// ---------------------------------------------------------------------------
describe('extract-cross-refs main', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    process.exitCode = 0;
  });

  afterEach(() => {
    process.exitCode = 0;
  });

  it('prints help and returns when --help is passed', async () => {
    mockParseCliArgs.mockReturnValue({ help: true });
    await main();
    expect(mockPrintHelp).toHaveBeenCalledTimes(1);
    expect(mockPrintHelp.mock.calls[0][0]).toHaveProperty('title', 'Cross-Reference Extractor');
  });

  it('runs full pipeline successfully with --json', async () => {
    mockParseCliArgs.mockReturnValue({ json: true });
    mockExtractCodeEntities.mockResolvedValue({
      entities: [{ qualified_name: 'src/neat', name: 'neat', entity_type: 'module' }],
    });
    mockGlob.mockResolvedValue(['plans/test.md']);
    mockExtractDocEntities.mockResolvedValue({
      entities: [{
        qualified_name: 'plans/test',
        file_path: 'plans/test.md',
        char_start: 0,
        char_end: 100,
      }],
    });
    mockReadFile.mockResolvedValue('Some doc text with src/neat.ts reference.');

    await main();

    expect(mockExtractCodeEntities).toHaveBeenCalledTimes(1);
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
    expect(mockWriteJsonOrText.mock.calls[0][1]).toBe(true);
  });

  it('runs full pipeline in text mode', async () => {
    mockParseCliArgs.mockReturnValue({});
    mockExtractCodeEntities.mockResolvedValue({ entities: [] });
    mockGlob.mockResolvedValue([]);
    mockExtractDocEntities.mockResolvedValue({ entities: [] });

    await main();

    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
    expect(mockWriteJsonOrText.mock.calls[0][1]).toBe(false);
  });

  it('handles file read errors gracefully in heading text map', async () => {
    mockParseCliArgs.mockReturnValue({});
    mockExtractCodeEntities.mockResolvedValue({ entities: [] });
    mockGlob.mockResolvedValue(['plans/test.md']);
    mockExtractDocEntities.mockResolvedValue({
      entities: [{
        qualified_name: 'plans/test',
        file_path: 'plans/test.md',
        char_start: 0,
        char_end: 100,
      }],
    });
    mockReadFile.mockRejectedValue(new Error('ENOENT'));

    await main();

    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
  });

  it('catches errors and calls fail with Error message', async () => {
    mockParseCliArgs.mockReturnValue({ json: true });
    mockExtractCodeEntities.mockRejectedValue(new Error('Extract failed'));

    await main();

    expect(mockFail).toHaveBeenCalledTimes(1);
    expect(mockFail.mock.calls[0][0]).toBe('Extract failed');
    expect(mockFail.mock.calls[0][1]).toBe(true);
  });

  it('catches non-Error throws and calls fail with String()', async () => {
    mockParseCliArgs.mockReturnValue({});
    mockExtractCodeEntities.mockRejectedValue('string error');

    await main();

    expect(mockFail).toHaveBeenCalledTimes(1);
    expect(mockFail.mock.calls[0][0]).toBe('string error');
  });
});