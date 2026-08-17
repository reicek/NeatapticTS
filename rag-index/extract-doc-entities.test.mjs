import { jest } from '@jest/globals';
import path from 'node:path';

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
  toRepoRelative: jest.fn((p) => p),
}));

const mockReadFile = jest.fn();
jest.unstable_mockModule('node:fs/promises', () => ({ readFile: mockReadFile }));

jest.unstable_mockModule('./init-schema.mjs', () => ({ repoRoot: '/fake/repo' }));

const mockGlob = jest.fn();
jest.unstable_mockModule('fast-glob', () => ({ default: mockGlob }));

const {
  extractDocEntities,
  mapFamilyToEntityType,
  constructDocQualifiedName,
  extractDocMetadata,
  parseYamlFrontmatter,
  extractHeadings,
  main,
} = await import('./extract-doc-entities.mjs');

// ---------------------------------------------------------------------------
// mapFamilyToEntityType
// ---------------------------------------------------------------------------
describe('mapFamilyToEntityType', () => {
  it('maps plan to plan', () => {
    expect(mapFamilyToEntityType('plan')).toBe('plan');
  });
  it('maps completed-plan to plan', () => {
    expect(mapFamilyToEntityType('completed-plan')).toBe('plan');
  });
  it('maps skill to skill', () => {
    expect(mapFamilyToEntityType('skill')).toBe('skill');
  });
  it('maps agent to agent', () => {
    expect(mapFamilyToEntityType('agent')).toBe('agent');
  });
  it('maps demo to demo', () => {
    expect(mapFamilyToEntityType('demo')).toBe('demo');
  });
  it('maps benchmark to benchmark', () => {
    expect(mapFamilyToEntityType('benchmark')).toBe('benchmark');
  });
  it('returns null for unknown family', () => {
    expect(mapFamilyToEntityType('unknown')).toBeNull();
  });
  it('returns null for undefined', () => {
    expect(mapFamilyToEntityType(undefined)).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// constructDocQualifiedName
// ---------------------------------------------------------------------------
describe('constructDocQualifiedName', () => {
  it('constructs plan qualified name from simple file', () => {
    expect(constructDocQualifiedName('plans/MyPlan.md', 'plan')).toBe('plans/my_plan');
  });
  it('constructs plan qualified name from completed plan', () => {
    expect(constructDocQualifiedName('plans/completed/MyPlan.md', 'plan')).toBe('plans/my_plan');
  });
  it('constructs plan qualified name with .plans.md extension', () => {
    expect(constructDocQualifiedName('plans/Some_Plan.plans.md', 'plan')).toBe('plans/some_plan.plans');
  });
  it('constructs plan qualified name with camelCase', () => {
    expect(constructDocQualifiedName('plans/MyPlan.md', 'plan')).toBe('plans/my_plan');
  });
  it('constructs skill qualified name from .github/skills path', () => {
    expect(constructDocQualifiedName('.github/skills/coverage-guard/SKILL.md', 'skill')).toBe('skills/coverage-guard');
  });
  it('constructs skill qualified name when skills folder in path', () => {
    expect(constructDocQualifiedName('skills/my-skill/SKILL.md', 'skill')).toBe('skills/my-skill');
  });
  it('constructs skill qualified name fallback using dirname basename', () => {
    expect(constructDocQualifiedName('other/SKILL.md', 'skill')).toBe('skills/other');
  });
  it('constructs agent qualified name', () => {
    expect(constructDocQualifiedName('.github/agents/04-implementing.agent.md', 'agent')).toBe('agents/04-implementing');
  });
  it('constructs demo qualified name from examples subdirectory', () => {
    const result = constructDocQualifiedName('examples/flappy-bird-lstm/README.md', 'demo');
    expect(result).toMatch(/^demos\//);
  });
  it('constructs benchmark qualified name from benchmarks subdirectory', () => {
    const result = constructDocQualifiedName('benchmarks/memory-optimization/README.md', 'benchmark');
    expect(result).toMatch(/^benchmarks\//);
  });
  it('falls back to dot-separated path for unknown entity type', () => {
    expect(constructDocQualifiedName('other/file.md', 'unknown')).toBe('other.file');
  });
});

// ---------------------------------------------------------------------------
// parseYamlFrontmatter
// ---------------------------------------------------------------------------
describe('parseYamlFrontmatter', () => {
  it('returns empty object for content without frontmatter', () => {
    expect(parseYamlFrontmatter('Just some text.')).toEqual({});
  });
  it('returns empty object for unclosed frontmatter', () => {
    expect(parseYamlFrontmatter('---\ntier: 3\n')).toEqual({});
  });
  it('parses simple key-value pairs', () => {
    const content = '---\ntier: 3\nname: test\n---\nBody.';
    expect(parseYamlFrontmatter(content)).toEqual({ tier: '3', name: 'test' });
  });
  it('parses array values', () => {
    const content = '---\nskills: [a, b, c]\n---\nBody.';
    expect(parseYamlFrontmatter(content)).toEqual({ skills: ['a', 'b', 'c'] });
  });
  it('parses array values with quotes', () => {
    const content = '---\nskills: ["a", \'b\']\n---\nBody.';
    expect(parseYamlFrontmatter(content)).toEqual({ skills: ['a', 'b'] });
  });
  it('parses single-quoted values', () => {
    const content = '---\nkey: \'value\'\n---\nBody.';
    expect(parseYamlFrontmatter(content)).toEqual({ key: 'value' });
  });
  it('parses double-quoted values', () => {
    const content = '---\nkey: "value"\n---\nBody.';
    expect(parseYamlFrontmatter(content)).toEqual({ key: 'value' });
  });
  it('skips comment lines', () => {
    const content = '---\n# comment\nkey: value\n---\nBody.';
    expect(parseYamlFrontmatter(content)).toEqual({ key: 'value' });
  });
  it('skips empty lines', () => {
    const content = '---\n\nkey: value\n---\nBody.';
    expect(parseYamlFrontmatter(content)).toEqual({ key: 'value' });
  });
  it('skips lines without colons', () => {
    const content = '---\nnot-a-key-value\nkey: value\n---\nBody.';
    expect(parseYamlFrontmatter(content)).toEqual({ key: 'value' });
  });
  it('handles empty frontmatter', () => {
    const content = '---\n---\nBody.';
    expect(parseYamlFrontmatter(content)).toEqual({});
  });
});

// ---------------------------------------------------------------------------
// extractDocMetadata
// ---------------------------------------------------------------------------
describe('extractDocMetadata', () => {
  it('extracts tier and skills from agent frontmatter', () => {
    const content = '---\ntier: 3\nskills: [a, b]\n---\nBody.';
    expect(extractDocMetadata(content, 'agent', 'agent')).toEqual({ tier: '3', skills: ['a', 'b'] });
  });
  it('extracts only tier when skills not present', () => {
    const content = '---\ntier: 2\n---\nBody.';
    expect(extractDocMetadata(content, 'agent', 'agent')).toEqual({ tier: '2' });
  });
  it('extracts only skills when tier not present', () => {
    const content = '---\nskills: [x]\n---\nBody.';
    expect(extractDocMetadata(content, 'agent', 'agent')).toEqual({ skills: ['x'] });
  });
  it('returns empty for agent without frontmatter', () => {
    expect(extractDocMetadata('Just text.', 'agent', 'agent')).toEqual({});
  });
  it('extracts status marker from plan', () => {
    const content = '**Status:** [IN PROGRESS]\nSome text.';
    expect(extractDocMetadata(content, 'plan', 'plan')).toEqual({ status_marker: 'IN PROGRESS' });
  });
  it('returns empty for plan without status marker', () => {
    expect(extractDocMetadata('Just text.', 'plan', 'plan')).toEqual({});
  });
  it('returns empty for other entity types', () => {
    expect(extractDocMetadata('Some text.', 'skill', 'skill')).toEqual({});
  });
});

// ---------------------------------------------------------------------------
// extractHeadings
// ---------------------------------------------------------------------------
describe('extractHeadings', () => {
  it('returns empty array for content with no headings', () => {
    expect(extractHeadings('Just some text without headings.')).toEqual([]);
  });
  it('extracts a single heading', () => {
    const content = '# Title\nSome body text.';
    const headings = extractHeadings(content);
    expect(headings).toHaveLength(1);
    expect(headings[0].text).toBe('Title');
    expect(headings[0].level).toBe(1);
    expect(headings[0].slug).toBe('title');
  });
  it('sets charEnd to content length for last heading', () => {
    const content = '# Title\nBody.';
    const headings = extractHeadings(content);
    expect(headings[0].charEnd).toBe(content.length);
  });
  it('extracts multiple headings with correct charStart', () => {
    const content = '# First\nbody1\n## Second\nbody2';
    const headings = extractHeadings(content);
    expect(headings).toHaveLength(2);
    expect(headings[0].text).toBe('First');
    expect(headings[1].text).toBe('Second');
  });
  it('handles level 2 and 3 headings', () => {
    const content = '## Level 2\n### Level 3';
    const headings = extractHeadings(content);
    expect(headings).toHaveLength(2);
    expect(headings[0].level).toBe(2);
    expect(headings[1].level).toBe(3);
  });
  it('slugifies heading text', () => {
    const content = '# Hello World!';
    const headings = extractHeadings(content);
    expect(headings[0].slug).toBe('hello-world');
  });
  it('handles non-heading lines between headings', () => {
    const content = '# A\nsome text\nmore text\n# B\nbody';
    const headings = extractHeadings(content);
    expect(headings).toHaveLength(2);
    expect(headings[0].text).toBe('A');
    expect(headings[1].text).toBe('B');
  });
});

// ---------------------------------------------------------------------------
// extractDocEntities
// ---------------------------------------------------------------------------
describe('extractDocEntities', () => {
  beforeEach(() => {
    mockReadFile.mockReset();
  });

  it('skips documents with unknown family', async () => {
    const result = await extractDocEntities({
      documents: [{ filePath: 'unknown/file.md', family: 'unknown' }],
    });
    expect(result.entities).toEqual([]);
    expect(result.edges).toEqual([]);
  });

  it('skips documents that cannot be read', async () => {
    mockReadFile.mockRejectedValue(new Error('ENOENT'));
    const result = await extractDocEntities({
      documents: [{ filePath: 'plans/test.md', family: 'plan' }],
    });
    expect(result.entities).toEqual([]);
  });

  it('extracts plan entity with heading sections', async () => {
    const content = '# Plan Title\n\nSome body.\n\n## Section\n\nMore body.\n';
    mockReadFile.mockResolvedValue(content);
    const result = await extractDocEntities({
      documents: [{ filePath: 'plans/TestPlan.md', family: 'plan' }],
    });
    expect(result.entities.length).toBeGreaterThanOrEqual(1);
    const planEntity = result.entities.find((e) => e.entity_type === 'plan' && !e.qualified_name.includes('.'));
    expect(planEntity).toBeDefined();
    expect(result.entityMap.has(planEntity.qualified_name)).toBe(true);
  });

  it('creates contains edges for heading entities', async () => {
    const content = '# Plan\n\nBody.\n\n## Section\n\nMore.\n';
    mockReadFile.mockResolvedValue(content);
    const result = await extractDocEntities({
      documents: [{ filePath: 'plans/TestPlan.md', family: 'plan' }],
    });
    const containsEdges = result.edges.filter((e) => e.relationship === 'contains');
    expect(containsEdges.length).toBeGreaterThan(0);
  });

  it('handles document with no headings', async () => {
    mockReadFile.mockResolvedValue('Just some text without headings.');
    const result = await extractDocEntities({
      documents: [{ filePath: 'plans/Simple.md', family: 'plan' }],
    });
    expect(result.entities).toHaveLength(1);
    expect(result.edges).toHaveLength(0);
  });

  it('extracts agent entity with metadata', async () => {
    const content = '---\ntier: 3\nskills: [a, b]\n---\n# Agent\n\nBody.\n';
    mockReadFile.mockResolvedValue(content);
    const result = await extractDocEntities({
      documents: [{ filePath: '.github/agents/test.agent.md', family: 'agent' }],
    });
    const agentEntity = result.entities.find((e) => e.entity_type === 'agent');
    expect(agentEntity).toBeDefined();
    const metadata = JSON.parse(agentEntity.extra_metadata);
    expect(metadata.tier).toBe('3');
  });

  it('extracts plan with status metadata', async () => {
    const content = '**Status:** [DONE]\n# Plan\n\nBody.\n';
    mockReadFile.mockResolvedValue(content);
    const result = await extractDocEntities({
      documents: [{ filePath: 'plans/TestPlan.md', family: 'plan' }],
    });
    const planEntity = result.entities.find((e) => e.entity_type === 'plan');
    const metadata = JSON.parse(planEntity.extra_metadata);
    expect(metadata.status_marker).toBe('DONE');
  });

  it('handles multiple documents', async () => {
    mockReadFile.mockResolvedValue('# Title\nBody.');
    const result = await extractDocEntities({
      documents: [
        { filePath: 'plans/A.md', family: 'plan' },
        { filePath: 'plans/B.md', family: 'plan' },
      ],
    });
    expect(result.entities.length).toBeGreaterThanOrEqual(2);
  });
});

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
describe('extract-doc-entities main', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockReadFile.mockReset();
    process.exitCode = 0;
  });

  afterEach(() => {
    process.exitCode = 0;
  });

  it('prints help and returns when --help is passed', async () => {
    mockParseCliArgs.mockReturnValue({ help: true });
    await main();
    expect(mockPrintHelp).toHaveBeenCalledTimes(1);
  });

  it('runs full pipeline successfully in text mode', async () => {
    mockParseCliArgs.mockReturnValue({});
    mockGlob.mockResolvedValue(['plans/test.md']);
    mockReadFile.mockResolvedValue('# Plan\n\nBody text.\n');
    await main();
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
    expect(mockWriteJsonOrText.mock.calls[0][1]).toBe(false);
  });

  it('runs full pipeline with --json flag', async () => {
    mockParseCliArgs.mockReturnValue({ json: true });
    mockGlob.mockResolvedValue(['plans/test.md']);
    mockReadFile.mockResolvedValue('# Plan\n\nBody.\n');
    await main();
    expect(mockWriteJsonOrText).toHaveBeenCalledTimes(1);
    expect(mockWriteJsonOrText.mock.calls[0][1]).toBe(true);
  });

  it('catches errors and calls fail with Error message', async () => {
    mockParseCliArgs.mockReturnValue({ json: true });
    mockGlob.mockRejectedValue(new Error('Glob failed'));
    await main();
    expect(mockFail).toHaveBeenCalledTimes(1);
    expect(mockFail.mock.calls[0][0]).toBe('Glob failed');
    expect(mockFail.mock.calls[0][1]).toBe(true);
  });

  it('catches non-Error throws and calls fail with String()', async () => {
    mockParseCliArgs.mockReturnValue({});
    mockGlob.mockRejectedValue('string error');
    await main();
    expect(mockFail).toHaveBeenCalledTimes(1);
    expect(mockFail.mock.calls[0][0]).toBe('string error');
  });
});