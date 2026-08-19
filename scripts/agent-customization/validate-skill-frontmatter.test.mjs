import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

jest.unstable_mockModule('./customization-utils.mjs', () => ({
  extractMarkdownLinks: jest.fn(),
  fileExists: jest.fn(),
  issue: jest.fn((severity, path, message) => ({ severity, path, message })),
  listMarkdownFiles: jest.fn(),
  parseArgs: jest.fn(),
  parseFrontmatter: jest.fn(),
  printUsage: jest.fn(),
  readWorkspaceFile: jest.fn(),
  summarizeIssues: jest.fn((name, issues) => ({
    name,
    ok: issues.length === 0,
    issues,
    counts: {
      errors: issues.filter((i) => i.severity === 'error').length,
      warnings: 0,
    },
    summaryText: `${issues.length === 0 ? 'PASS' : 'FAIL'} ${name}`,
  })),
  writeReport: jest.fn(),
}));

let mockUtils;

beforeEach(async () => {
  jest.resetModules();
  mockUtils = await import('./customization-utils.mjs');
  jest.clearAllMocks();
});

async function importModule() {
  try {
    await import('./validate-skill-frontmatter.mjs');
  } catch {
    // process.exit may throw
  }
}

describe('validate-skill-frontmatter', () => {
  it('prints help and exits 0 when --help is passed', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    process.argv = ['node', 'validate-skill-frontmatter.mjs', '--help'];
    process.exit = (code) => {
      throw new Error(`EXIT:${code}`);
    };
    mockUtils.parseArgs.mockReturnValue({ help: true, json: false });

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    assert.ok(mockUtils.printUsage.mock.calls.length >= 1);
  });

  it('validates a well-formed skill with no issues', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'validate-skill-frontmatter.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      strict: false,
    });
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/skills/my-skill/SKILL.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue('body text');
    mockUtils.parseFrontmatter.mockReturnValue({
      data: {
        name: 'my-skill',
        description: 'A valid skill.',
        'user-invocable': true,
      },
      body: 'body text',
      issues: [],
    });
    mockUtils.extractMarkdownLinks.mockReturnValue([]);
    mockUtils.fileExists.mockResolvedValue(true);

    await importModule();

    process.argv = origArgv;
    const report = mockUtils.writeReport.mock.calls[0][0];
    assert.strictEqual(report.ok, true);
  });

  it('flags error when skill name is missing', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'validate-skill-frontmatter.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      strict: false,
    });
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/skills/my-skill/SKILL.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue('');
    mockUtils.parseFrontmatter.mockReturnValue({
      data: { description: 'A skill.' },
      body: '',
      issues: [],
    });
    mockUtils.extractMarkdownLinks.mockReturnValue([]);

    await importModule();

    process.argv = origArgv;
    const nameIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('name is required'),
    );
    assert.ok(nameIssue);
  });

  it('flags error when skill name has invalid characters', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'validate-skill-frontmatter.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      strict: false,
    });
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/skills/BadName/SKILL.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue('');
    mockUtils.parseFrontmatter.mockReturnValue({
      data: { name: 'BadName', description: 'd' },
      body: '',
      issues: [],
    });
    mockUtils.extractMarkdownLinks.mockReturnValue([]);

    await importModule();

    process.argv = origArgv;
    const nameIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('lowercase alphanumeric'),
    );
    assert.ok(nameIssue);
  });

  it('flags error when skill name does not match folder', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'validate-skill-frontmatter.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      strict: false,
    });
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/skills/wrong-folder/SKILL.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue('');
    mockUtils.parseFrontmatter.mockReturnValue({
      data: { name: 'right-name', description: 'd' },
      body: '',
      issues: [],
    });
    mockUtils.extractMarkdownLinks.mockReturnValue([]);

    await importModule();

    process.argv = origArgv;
    const mismatch = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('must match folder'),
    );
    assert.ok(mismatch);
  });

  it('flags error when description is missing', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'validate-skill-frontmatter.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      strict: false,
    });
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/skills/my-skill/SKILL.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue('');
    mockUtils.parseFrontmatter.mockReturnValue({
      data: { name: 'my-skill' },
      body: '',
      issues: [],
    });
    mockUtils.extractMarkdownLinks.mockReturnValue([]);

    await importModule();

    process.argv = origArgv;
    const descIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('description is required'),
    );
    assert.ok(descIssue);
  });

  it('flags error when description exceeds 1024 chars', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'validate-skill-frontmatter.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      strict: false,
    });
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/skills/my-skill/SKILL.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue('');
    mockUtils.parseFrontmatter.mockReturnValue({
      data: { name: 'my-skill', description: 'x'.repeat(1025) },
      body: '',
      issues: [],
    });
    mockUtils.extractMarkdownLinks.mockReturnValue([]);

    await importModule();

    process.argv = origArgv;
    const descIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('1024-character'),
    );
    assert.ok(descIssue);
  });

  it('flags error when user-invocable is not boolean', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'validate-skill-frontmatter.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      strict: false,
    });
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/skills/my-skill/SKILL.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue('');
    mockUtils.parseFrontmatter.mockReturnValue({
      data: { name: 'my-skill', description: 'd', 'user-invocable': 'yes' },
      body: '',
      issues: [],
    });
    mockUtils.extractMarkdownLinks.mockReturnValue([]);

    await importModule();

    process.argv = origArgv;
    const uiIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('user-invocable'),
    );
    assert.ok(uiIssue);
  });

  it('flags error when disable-model-invocation is not boolean', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'validate-skill-frontmatter.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      strict: false,
    });
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/skills/my-skill/SKILL.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue('');
    mockUtils.parseFrontmatter.mockReturnValue({
      data: {
        name: 'my-skill',
        description: 'd',
        'disable-model-invocation': 'no',
      },
      body: '',
      issues: [],
    });
    mockUtils.extractMarkdownLinks.mockReturnValue([]);

    await importModule();

    process.argv = origArgv;
    const dmiIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('disable-model-invocation'),
    );
    assert.ok(dmiIssue);
  });

  it('flags error when compatibility exceeds 500 chars', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'validate-skill-frontmatter.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      strict: false,
    });
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/skills/my-skill/SKILL.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue('');
    mockUtils.parseFrontmatter.mockReturnValue({
      data: {
        name: 'my-skill',
        description: 'd',
        compatibility: 'x'.repeat(501),
      },
      body: '',
      issues: [],
    });
    mockUtils.extractMarkdownLinks.mockReturnValue([]);

    await importModule();

    process.argv = origArgv;
    const compIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('500-character'),
    );
    assert.ok(compIssue);
  });

  it('warns in strict mode when user-invocable not present', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'validate-skill-frontmatter.mjs',
      '--json',
      '--strict',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      strict: true,
    });
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/skills/my-skill/SKILL.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue('');
    mockUtils.parseFrontmatter.mockReturnValue({
      data: { name: 'my-skill', description: 'd', 'argument-hint': 'hint' },
      body: '',
      issues: [],
    });
    mockUtils.extractMarkdownLinks.mockReturnValue([]);

    await importModule();

    process.argv = origArgv;
    const strictIssue = mockUtils.issue.mock.calls.find(
      (c) => c[0] === 'warning' && c[2].includes('user-invocable'),
    );
    assert.ok(strictIssue);
  });

  it('errors in strict mode when argument-hint missing', async () => {
    const origArgv = process.argv;
    process.argv = [
      'node',
      'validate-skill-frontmatter.mjs',
      '--json',
      '--strict',
    ];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      strict: true,
    });
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/skills/my-skill/SKILL.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue('');
    mockUtils.parseFrontmatter.mockReturnValue({
      data: { name: 'my-skill', description: 'd', 'user-invocable': true },
      body: '',
      issues: [],
    });
    mockUtils.extractMarkdownLinks.mockReturnValue([]);

    await importModule();

    process.argv = origArgv;
    const hintIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('argument-hint'),
    );
    assert.ok(hintIssue);
  });

  it('flags error for missing local resource links', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'validate-skill-frontmatter.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      strict: false,
    });
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/skills/my-skill/SKILL.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue('See [guide](./guide.md)');
    mockUtils.parseFrontmatter.mockReturnValue({
      data: { name: 'my-skill', description: 'd' },
      body: 'See [guide](./guide.md)',
      issues: [],
    });
    mockUtils.extractMarkdownLinks.mockReturnValue(['./guide.md']);
    mockUtils.fileExists.mockResolvedValue(false);

    await importModule();

    process.argv = origArgv;
    const linkIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('does not exist'),
    );
    assert.ok(linkIssue);
  });

  it('includes parse issues from frontmatter parsing', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'validate-skill-frontmatter.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      strict: false,
    });
    const parseIssue = {
      severity: 'error',
      path: '.github/skills/my-skill/SKILL.md',
      message: 'YAML parse error',
    };
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/skills/my-skill/SKILL.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue('');
    mockUtils.parseFrontmatter.mockReturnValue({
      data: { name: 'my-skill', description: 'd' },
      body: '',
      issues: [parseIssue],
    });
    mockUtils.extractMarkdownLinks.mockReturnValue([]);

    await importModule();

    process.argv = origArgv;
    const report = mockUtils.writeReport.mock.calls[0][0];
    assert.ok(report.issues.some((i) => i.message === 'YAML parse error'));
  });

  it('handles empty skills list', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'validate-skill-frontmatter.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      strict: false,
    });
    mockUtils.listMarkdownFiles.mockResolvedValue([]);

    await importModule();

    process.argv = origArgv;
    const report = mockUtils.writeReport.mock.calls[0][0];
    assert.strictEqual(report.ok, true);
    assert.deepStrictEqual(report.skills, []);
  });

  it('passes when local resource links exist', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'validate-skill-frontmatter.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      strict: false,
    });
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/skills/my-skill/SKILL.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue('See [guide](./guide.md)');
    mockUtils.parseFrontmatter.mockReturnValue({
      data: { name: 'my-skill', description: 'd' },
      body: 'See [guide](./guide.md)',
      issues: [],
    });
    mockUtils.extractMarkdownLinks.mockReturnValue(['./guide.md']);
    mockUtils.fileExists.mockResolvedValue(true);

    await importModule();

    process.argv = origArgv;
    const linkIssue = mockUtils.issue.mock.calls.find((c) =>
      c[2].includes('does not exist'),
    );
    assert.strictEqual(linkIssue, undefined);
  });

  it('exercises the listMarkdownFiles predicate callback (false branch)', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'validate-skill-frontmatter.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({
      help: false,
      json: true,
      strict: false,
    });
    mockUtils.listMarkdownFiles.mockImplementation(async (dir, filter) => {
      // Exercise both true and false branches of the .endsWith('/SKILL.md') predicate
      assert.strictEqual(filter('.github/skills/my-skill/SKILL.md'), true);
      assert.strictEqual(filter('.github/skills/my-skill/README.md'), false);
      return [];
    });

    await importModule();

    process.argv = origArgv;
    assert.ok(mockUtils.listMarkdownFiles.mock.calls.length >= 1);
  });
});
