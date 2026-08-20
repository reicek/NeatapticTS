import { jest } from '@jest/globals';
import assert from 'node:assert/strict';

jest.unstable_mockModule('./customization-utils.mjs', () => ({
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
    await import('./validate-sdlc-skill-coverage.mjs');
  } catch {
    // process.exit may throw
  }
}

describe('validate-sdlc-skill-coverage', () => {
  it('prints help and exits 0 when --help is passed', async () => {
    const origArgv = process.argv;
    const origExit = process.exit;
    process.argv = ['node', 'validate-sdlc-skill-coverage.mjs', '--help'];
    process.exit = (code) => {
      throw new Error(`EXIT:${code}`);
    };
    mockUtils.parseArgs.mockReturnValue({ help: true, json: false });

    await importModule();

    process.argv = origArgv;
    process.exit = origExit;
    assert.ok(mockUtils.printUsage.mock.calls.length >= 1);
  });

  it('reports ok when all required skills exist', async () => {
    const origArgv = process.argv;
    const origExitCode = process.exitCode;
    process.argv = ['node', 'validate-sdlc-skill-coverage.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    const allSkillNames = [
      'planning-acceptance-criteria',
      'creating-unit-tests',
      'red-test-contracts',
      'running-unit-tests',
      'green-validation-gates',
      'triaging-test-failures',
      'test-fix-workflow',
      'auditing-js-docs',
      'docs-academic-citation-audit',
      'updating-js-docs',
      'educational-docs',
      'agent-frontmatter-standards',
      'updating-agent-frontmatter',
      'skill-frontmatter-standards',
      'updating-skill-frontmatter',
      'creating-specialist-agent',
      'splitting-monolithic-agent',
      'capturing-learning-event',
      'summarizing-session-log',
      'agent-inventory-audit',
      'agent-script-tooling',
      'skill-description-evals',
      'skill-output-evals',
      'subagent-delegation-patterns',
      'model-routing-and-budget',
    ];
    mockUtils.listMarkdownFiles.mockResolvedValue(
      allSkillNames.map((n) => `.github/skills/${n}/SKILL.md`),
    );
    mockUtils.readWorkspaceFile.mockResolvedValue(
      '---\nname: test\nargument-hint: hint\n---\nbody',
    );
    mockUtils.parseFrontmatter.mockImplementation((text, path) => ({
      data: {
        name: path.split('/').at(-2),
        'argument-hint': 'some-hint',
      },
      body: '',
      issues: [],
    }));

    await importModule();

    process.argv = origArgv;
    process.exitCode = origExitCode;
    const report = mockUtils.writeReport.mock.calls[0][0];
    assert.strictEqual(report.ok, true);
    assert.strictEqual(report.summary.skills, allSkillNames.length);
  });

  it('flags errors for missing required skills', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'validate-sdlc-skill-coverage.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.listMarkdownFiles.mockResolvedValue([]);
    mockUtils.readWorkspaceFile.mockResolvedValue('');
    mockUtils.parseFrontmatter.mockReturnValue({
      data: {},
      body: '',
      issues: [],
    });

    await importModule();

    process.argv = origArgv;
    const errorCalls = mockUtils.issue.mock.calls.filter(
      (c) => c[0] === 'error',
    );
    assert.ok(errorCalls.length >= 1);
  });

  it('flags warnings for skills without argument-hint', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'validate-sdlc-skill-coverage.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/skills/test-skill/SKILL.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue('');
    mockUtils.parseFrontmatter.mockReturnValue({
      data: { name: 'test-skill', 'argument-hint': null },
      body: '',
      issues: [],
    });

    await importModule();

    process.argv = origArgv;
    const warningCalls = mockUtils.issue.mock.calls.filter(
      (c) => c[0] === 'warning' && c[2].includes('argument-hint'),
    );
    assert.ok(warningCalls.length >= 1);
  });

  it('includes capabilities array in report', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'validate-sdlc-skill-coverage.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.listMarkdownFiles.mockResolvedValue([]);
    mockUtils.readWorkspaceFile.mockResolvedValue('');
    mockUtils.parseFrontmatter.mockReturnValue({
      data: {},
      body: '',
      issues: [],
    });

    await importModule();

    process.argv = origArgv;
    const report = mockUtils.writeReport.mock.calls[0][0];
    assert.ok(Array.isArray(report.capabilities));
    assert.ok(report.capabilities.length > 0);
    assert.ok(typeof report.capabilities[0].capability === 'string');
    assert.ok(Array.isArray(report.capabilities[0].skills));
    assert.ok(typeof report.capabilities[0].present === 'boolean');
  });

  it('sets exitCode based on report.ok', async () => {
    const origArgv = process.argv;
    const origExitCode = process.exitCode;
    process.argv = ['node', 'validate-sdlc-skill-coverage.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.listMarkdownFiles.mockResolvedValue([]);
    mockUtils.readWorkspaceFile.mockResolvedValue('');
    mockUtils.parseFrontmatter.mockReturnValue({
      data: {},
      body: '',
      issues: [],
    });

    await importModule();

    process.argv = origArgv;
    assert.strictEqual(process.exitCode, 1);
    process.exitCode = origExitCode;
  });

  it('exercises the listMarkdownFiles predicate callback (false branch)', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'validate-sdlc-skill-coverage.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.listMarkdownFiles.mockImplementation(async (dir, filter) => {
      // Exercise both true and false branches of the .endsWith('/SKILL.md') predicate
      assert.strictEqual(filter('.github/skills/my-skill/SKILL.md'), true);
      assert.strictEqual(filter('.github/skills/my-skill/README.md'), false);
      return [];
    });
    mockUtils.readWorkspaceFile.mockResolvedValue('');
    mockUtils.parseFrontmatter.mockReturnValue({
      data: {},
      body: '',
      issues: [],
    });

    await importModule();

    process.argv = origArgv;
    assert.ok(mockUtils.listMarkdownFiles.mock.calls.length >= 1);
  });

  it('covers data.name ?? "" fallback when skill has no name in frontmatter', async () => {
    const origArgv = process.argv;
    process.argv = ['node', 'validate-sdlc-skill-coverage.mjs', '--json'];
    mockUtils.parseArgs.mockReturnValue({ help: false, json: true });
    mockUtils.listMarkdownFiles.mockResolvedValue([
      '.github/skills/test-skill/SKILL.md',
    ]);
    mockUtils.readWorkspaceFile.mockResolvedValue('---\n---\nbody');
    mockUtils.parseFrontmatter.mockReturnValue({
      data: {},
      body: '',
      issues: [],
    });

    await importModule();

    process.argv = origArgv;
    const report = mockUtils.writeReport.mock.calls[0][0];
    assert.strictEqual(report.summary.skills, 1);
    assert.strictEqual(report.ok, false);
  });
});
