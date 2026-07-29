import { spawnSync } from 'node:child_process';
import { mkdir, rm, writeFile } from 'node:fs/promises';
import path from 'node:path';

/**
 * Tests for the file-path validation behaviour added to
 * validate-plan-phase-packets.mjs (Issue 5 / B4).
 *
 * The validator must accept validation entries that are file paths and
 * reject stale Jest `--testPathPattern` / `--testPathPatterns` CLI flags.
 */

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const VALIDATOR_SCRIPT = path.join(
  REPO_ROOT,
  'scripts',
  'agent-customization',
  'validate-plan-phase-packets.mjs',
);
const TEMP_DIR = path.join(REPO_ROOT, 'plans', '_test-temp');

interface ValidatorIssue {
  severity: string;
  path: string;
  message: string;
}

interface ValidatorReport {
  name: string;
  ok: boolean;
  issues: ValidatorIssue[];
  counts: { errors: number; warnings: number };
}

const MAX_BUFFER_BYTES = 10 * 1024 * 1024;

async function loadValidator() {
  // @ts-ignore - tested module is authored in plain ESM without a declaration file.
  return import('./validate-plan-phase-packets.mjs');
}

function buildPlan(
  validationEntries: string[],
  planRelativePath: string,
): string {
  const validationYaml = validationEntries
    .map((entry) => `  - '${entry}'`)
    .join('\n');

  return [
    '## Implementation phases',
    '',
    '### Phase B — Orchestration fixes [WIP]',
    '',
    '- **Phase objective:** Fix validator.',
    '- **Stop conditions:** Done.',
    '- **Required validation:** Tests pass.',
    '',
    '```yaml',
    'phase: B',
    'title: Orchestration fixes',
    'status: WIP',
    'goal: planning',
    'expansion: steps',
    'auto_expand: false',
    'mode: fresh-session',
    `source_of_truth: ${planRelativePath}`,
    'copy_paste: true',
    'next_phase: C',
    'skills:',
    '  - implementation-standards',
    'validation:',
    validationYaml,
    'acceptance_criteria:',
    '  - AC passes',
    'placeholder_steps:',
    '  - placeholder',
    '```',
    '',
    '## Validation gates',
    '',
  ].join('\n');
}

async function runValidator(
  validationEntries: string[],
): Promise<ValidatorReport> {
  await mkdir(TEMP_DIR, { recursive: true });
  const planRelativePath = `plans/_test-temp/validator-${Date.now()}.plans.md`;
  const planFile = path.join(REPO_ROOT, planRelativePath);
  await writeFile(
    planFile,
    buildPlan(validationEntries, planRelativePath),
    'utf8',
  );

  const result = spawnSync(
    process.execPath,
    [VALIDATOR_SCRIPT, `--plan=${planRelativePath}`, '--json'],
    {
      cwd: REPO_ROOT,
      encoding: 'utf8',
      maxBuffer: MAX_BUFFER_BYTES,
    },
  );

  await rm(planFile, { force: true });

  if (result.status !== 0 && result.status !== 1) {
    throw new Error(
      `Validator process failed (exit ${result.status}): ${result.stderr || result.stdout}`,
    );
  }

  const report: ValidatorReport = JSON.parse(result.stdout);
  return report;
}

describe('validate-plan-phase-packets validation entries', () => {
  it('AC-B4-005: path-only validation entries pass', async () => {
    const report = await runValidator([
      'scripts/agent-customization/validate-plan-phase-packets.test.ts',
    ]);
    expect(report.ok).toBe(true);
    expect(report.counts.errors).toBe(0);
  });

  it('AC-B4-005: mixed path and stale flag entries fail on the flag', async () => {
    const report = await runValidator([
      'scripts/agent-customization/validate-plan-phase-packets.test.ts',
      '--testPathPatterns=scripts/foo.test.ts',
    ]);
    expect(report.ok).toBe(false);
    expect(report.counts.errors).toBeGreaterThan(0);
    expect(
      report.issues.some(
        (issue: ValidatorIssue) =>
          issue.severity === 'error' &&
          issue.message.includes('--testPathPattern'),
      ),
    ).toBe(true);
  });

  it('AC-B4-004/005: flag-only validation entries are rejected', async () => {
    const report = await runValidator([
      '--testPathPattern=scripts/agent-customization/validate-plan-phase-packets.test.ts',
    ]);
    expect(report.ok).toBe(false);
    expect(report.counts.errors).toBeGreaterThan(0);
    expect(
      report.issues.some(
        (issue: ValidatorIssue) =>
          issue.severity === 'error' &&
          issue.message.includes('stale --testPathPattern'),
      ),
    ).toBe(true);
  });
});

describe('looksLikeFilePath direct unit tests', () => {
  it('rejects non-string input', async () => {
    const { looksLikeFilePath } = await loadValidator();
    expect(looksLikeFilePath(123 as unknown as string)).toBe(false);
  });

  it('rejects empty strings', async () => {
    const { looksLikeFilePath } = await loadValidator();
    expect(looksLikeFilePath('')).toBe(false);
  });

  it('rejects strings starting with --', async () => {
    const { looksLikeFilePath } = await loadValidator();
    expect(looksLikeFilePath('--testPathPattern=foo')).toBe(false);
  });

  it('rejects multi-line strings', async () => {
    const { looksLikeFilePath } = await loadValidator();
    expect(looksLikeFilePath('foo\nbar')).toBe(false);
  });

  it('accepts slash-containing paths', async () => {
    const { looksLikeFilePath } = await loadValidator();
    expect(looksLikeFilePath('scripts/foo.mjs')).toBe(true);
  });

  it('accepts extension-only paths', async () => {
    const { looksLikeFilePath } = await loadValidator();
    expect(looksLikeFilePath('foo.test.ts')).toBe(true);
  });

  it('accepts dot and dot-dot relative paths', async () => {
    const { looksLikeFilePath } = await loadValidator();
    expect(looksLikeFilePath('.')).toBe(true);
    expect(looksLikeFilePath('..')).toBe(true);
  });
});

describe('validateValidationList direct unit tests', () => {
  it('errors on non-string validation entries', async () => {
    const { validateValidationList } = await loadValidator();
    const issues: ValidatorIssue[] = [];
    validateValidationList(
      [123 as unknown as string],
      'test/validation',
      issues,
    );
    expect(issues).toHaveLength(1);
    expect(issues[0].severity).toBe('error');
    expect(issues[0].message).toContain('Validation entry must be a string');
  });

  it('warns on entries that do not look like file paths', async () => {
    const { validateValidationList } = await loadValidator();
    const issues: ValidatorIssue[] = [];
    validateValidationList(['run all tests'], 'test/validation', issues);
    expect(issues).toHaveLength(1);
    expect(issues[0].severity).toBe('warning');
    expect(issues[0].message).toContain('does not look like a file path');
  });

  it('passes on valid file path entries', async () => {
    const { validateValidationList } = await loadValidator();
    const issues: ValidatorIssue[] = [];
    validateValidationList(['scripts/foo.test.ts'], 'test/validation', issues);
    expect(issues).toHaveLength(0);
  });

  it('errors on stale --testPathPattern flag entries', async () => {
    const { validateValidationList } = await loadValidator();
    const issues: ValidatorIssue[] = [];
    validateValidationList(
      ['--testPathPattern=scripts/foo.test.ts'],
      'test/validation',
      issues,
    );
    expect(issues).toHaveLength(1);
    expect(issues[0].severity).toBe('error');
    expect(issues[0].message).toContain('stale --testPathPattern');
  });
});

describe('validatePlanText direct unit tests', () => {
  it('passes a plan with valid file-path validation entries', async () => {
    const { validatePlanText } = await loadValidator();
    const text = buildPlan(
      ['scripts/agent-customization/validate-plan-phase-packets.test.ts'],
      'plans/_test-temp/direct-valid.plans.md',
    );
    const report = await validatePlanText(
      text,
      'plans/_test-temp/direct-valid.plans.md',
    );
    expect(report.ok).toBe(true);
    expect(report.counts.errors).toBe(0);
  });

  it('fails a plan containing a stale --testPathPattern flag', async () => {
    const { validatePlanText } = await loadValidator();
    const text = buildPlan(
      ['--testPathPattern=scripts/foo.test.ts'],
      'plans/_test-temp/direct-stale.plans.md',
    );
    const report = await validatePlanText(
      text,
      'plans/_test-temp/direct-stale.plans.md',
    );
    expect(report.ok).toBe(false);
    expect(report.counts.errors).toBeGreaterThan(0);
    expect(
      report.issues.some(
        (issue: ValidatorIssue) =>
          issue.severity === 'error' &&
          issue.message.includes('stale --testPathPattern'),
      ),
    ).toBe(true);
  });
});
