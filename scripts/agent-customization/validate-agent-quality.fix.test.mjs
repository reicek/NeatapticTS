import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import path from 'node:path';

jest.unstable_mockModule('./customization-utils.mjs', () => ({
  listMarkdownFiles: jest.fn(),
  readWorkspaceFile: jest.fn(),
  issue: jest.fn((severity, path, message) => ({ severity, path, message })),
  summarizeIssues: jest.fn(() => ({
    name: 'mock',
    errorCount: 0,
    warningCount: 0,
    issues: [],
  })),
  parseFrontmatter: jest.fn(),
}));

jest.unstable_mockModule('node:fs/promises', () => ({
  writeFile: jest.fn(),
}));

let runFix;
let mockUtils;
let mockFs;

beforeAll(async () => {
  mockUtils = await import('./customization-utils.mjs');
  mockFs = await import('node:fs/promises');
  const mod = await import('./validate-agent-quality.fix.mjs');
  runFix = mod.runFix;
});

beforeEach(() => {
  jest.clearAllMocks();
});

describe('validate-agent-quality.fix', () => {
  it('fixes agents with frontmatter and tier', async () => {
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/agents/test.agent.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue(
      '---\nname: test-agent\ntier: 3\ndescription: Test\n---\n\nBody content.',
    );
    mockUtils.parseFrontmatter.mockReturnValue({
      data: { name: 'test-agent', tier: 3 },
    });

    const report = await runFix({});

    assert.strictEqual(report.ok, true);
    assert.strictEqual(report.fixed.length, 1);
    assert.ok(mockFs.writeFile.mock.calls[0][1].includes('## Output format'));
    assert.ok(mockFs.writeFile.mock.calls[0][1].includes('```structured-v1'));
  });

  it('reports error for file without frontmatter', async () => {
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/agents/no-fm.agent.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue('Just text, no frontmatter.');

    const report = await runFix({});

    assert.strictEqual(report.fixed.length, 0);
    assert.ok(report.issues.length > 0);
    assert.ok(report.issues[0].message.includes('Missing YAML frontmatter'));
  });

  it('handles parseFrontmatter throwing', async () => {
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/agents/bad.agent.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue(
      '---\nname: bad-agent\n---\n\nBody.',
    );
    mockUtils.parseFrontmatter.mockImplementation(() => {
      throw new Error('parse error');
    });

    const report = await runFix({});

    assert.strictEqual(report.fixed.length, 1);
    assert.ok(mockFs.writeFile.mock.calls[0][1].includes('## Output format'));
  });

  it('handles writeFile failure', async () => {
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/agents/fail.agent.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue(
      '---\nname: fail-agent\ntier: 2\n---\n\nBody.',
    );
    mockUtils.parseFrontmatter.mockReturnValue({
      data: { name: 'fail-agent', tier: 2 },
    });
    mockFs.writeFile.mockRejectedValue(new Error('disk full'));

    const report = await runFix({});

    assert.strictEqual(report.ok, false);
    assert.ok(report.issues.some((i) => i.message.includes('Failed to write')));
  });

  it('generates tier-1 footer with PHASE_COMPLETE and SUB_ORCHESTRATORS_USED', async () => {
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/agents/t1.agent.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue(
      '---\nname: t1-agent\ntier: 1\n---\n\nBody.',
    );
    mockUtils.parseFrontmatter.mockReturnValue({
      data: { name: 't1-agent', tier: 1 },
    });

    await runFix({});

    const written = mockFs.writeFile.mock.calls[0][1];
    assert.ok(written.includes('PHASE_COMPLETE'));
    assert.ok(written.includes('SUB_ORCHESTRATORS_USED'));
  });

  it('generates tier-2 footer with SPECIALISTS_USED and HANDOFF', async () => {
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/agents/t2.agent.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue(
      '---\nname: t2-agent\ntier: 2\n---\n\nBody.',
    );
    mockUtils.parseFrontmatter.mockReturnValue({
      data: { name: 't2-agent', tier: 2 },
    });

    await runFix({});

    const written = mockFs.writeFile.mock.calls[0][1];
    assert.ok(written.includes('SPECIALISTS_USED'));
    assert.ok(written.includes('HANDOFF'));
  });

  it('generates tier-4 footer without HANDOFF', async () => {
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/agents/t4.agent.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue(
      '---\nname: t4-agent\ntier: 4\n---\n\nBody.',
    );
    mockUtils.parseFrontmatter.mockReturnValue({
      data: { name: 't4-agent', tier: 4 },
    });

    await runFix({});

    const written = mockFs.writeFile.mock.calls[0][1];
    assert.ok(!written.includes('HANDOFF'));
    assert.ok(!written.includes('SPECIALISTS_USED'));
    assert.ok(written.includes('ACTIONS_TAKEN'));
  });

  it('uses tier 1 footer when tier is unknown', async () => {
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/agents/unknown.agent.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue(
      '---\nname: unknown-agent\n---\n\nBody.',
    );
    mockUtils.parseFrontmatter.mockReturnValue({
      data: { name: 'unknown-agent' },
    });

    await runFix({});

    const written = mockFs.writeFile.mock.calls[0][1];
    assert.ok(written.includes('PHASE_COMPLETE'));
    assert.ok(written.includes('SUB_ORCHESTRATORS_USED'));
  });

  it('uses empty role when name is empty', async () => {
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/agents/noname.agent.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue('---\ntier: 3\n---\n\nBody.');
    mockUtils.parseFrontmatter.mockReturnValue({ data: {} });

    await runFix({});

    const written = mockFs.writeFile.mock.calls[0][1];
    assert.ok(written.includes('ROLE: '));
  });

  it('passes json flag through to runFix', async () => {
    mockUtils.listMarkdownFiles.mockResolvedValue([]);
    const report = await runFix({ json: true });
    assert.strictEqual(report.ok, true);
  });

  it('handles being called with no arguments', async () => {
    mockUtils.listMarkdownFiles.mockResolvedValue([]);
    const report = await runFix();
    assert.strictEqual(report.ok, true);
  });

  it('preserves content before existing Output format heading', async () => {
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/agents/existing.agent.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue(
      '---\nname: existing\ntier: 3\n---\n\nImportant content.\n\n## Output format\n\n```structured-v1\nOLD: stuff\n```\n',
    );
    mockUtils.parseFrontmatter.mockReturnValue({
      data: { name: 'existing', tier: 3 },
    });

    await runFix({});

    const written = mockFs.writeFile.mock.calls[0][1];
    assert.ok(written.includes('Important content.'));
    assert.ok(!written.includes('OLD: stuff'));
  });

  it('handles empty agent list', async () => {
    mockUtils.listMarkdownFiles.mockResolvedValue([]);

    const report = await runFix({});

    assert.strictEqual(report.counts.checked, 0);
    assert.strictEqual(report.fixed.length, 0);
  });

  it('exercises the listMarkdownFiles predicate callback (false branch)', async () => {
    mockUtils.listMarkdownFiles.mockImplementation(async (dir, filter) => {
      // Exercise both true and false branches of the .endsWith('.agent.md') predicate
      assert.strictEqual(filter('.github/agents/test.agent.md'), true);
      assert.strictEqual(filter('.github/agents/test.txt'), false);
      return ['.github/agents/test.agent.md'];
    });
    mockUtils.readWorkspaceFile.mockResolvedValue(
      '---\nname: test-agent\ntier: 3\ndescription: Test\n---\n\nBody content.',
    );
    mockUtils.parseFrontmatter.mockReturnValue({
      data: { name: 'test-agent', tier: 3 },
    });
    mockFs.writeFile.mockResolvedValue(undefined);

    const report = await runFix({});

    assert.strictEqual(report.fixed.length, 1);
  });
});
