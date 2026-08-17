/**
 * @module code-coverage.gate.test
 * @description Native-ESM coverage tests for code-coverage.gate.mjs.
 *
 * Runs in the agent-customization-mjs Jest project so V8 instruments the
 * source .mjs file directly, producing accurate coverage for the gate script.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';

const REPO_ROOT = path.resolve();
const GATE_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/gates/code-coverage.gate.mjs',
);

function makeSummary(dir, entries, fileName = 'coverage-summary.json') {
  const summaryPath = path.join(dir, fileName);
  const absoluteEntries = {};
  for (const [key, metrics] of Object.entries(entries)) {
    absoluteEntries[path.resolve(REPO_ROOT, key)] = metrics;
  }
  writeFileSync(summaryPath, JSON.stringify(absoluteEntries, null, 2));
  return summaryPath;
}

function runGateCli(args) {
  return execFileSync(process.execPath, [GATE_PATH, ...args], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
  });
}

async function loadGate() {
  return import('./code-coverage.gate.mjs');
}

describe('code-coverage gate native-ESM coverage', () => {
  let tempDir;

  beforeEach(() => {
    tempDir = mkdtempSync(path.join(os.tmpdir(), 'coverage-gate-mjs-'));
  });

  afterEach(() => {
    rmSync(tempDir, { recursive: true, force: true });
    jest.resetModules();
    jest.restoreAllMocks();
    jest.clearAllMocks();
    process.exitCode = 0;
  });

  it('passes when all target files have 100% coverage', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/architecture/foo.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      changedFiles: ['src/architecture/foo.ts'],
    });

    assert.equal(result.pass, true);
    assert.deepEqual(result.failedFiles, []);
    assert.deepEqual(result.missingFiles, []);
  });

  it('fails when a target file is below 100% on any metric', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/architecture/bar.ts': {
        lines: { pct: 100 },
        statements: { pct: 90 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      changedFiles: ['src/architecture/bar.ts'],
    });

    assert.equal(result.pass, false);
    assert.ok(result.failedFiles.includes('src/architecture/bar.ts'));
  });

  it('reports files missing from the coverage summary', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {});

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      changedFiles: ['src/architecture/missing.ts'],
    });

    assert.equal(result.pass, false);
    assert.ok(result.missingFiles.includes('src/architecture/missing.ts'));
  });

  it('supports repo-relative keys in the coverage summary and baseline', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = path.join(tempDir, 'relative-summary.json');
    const baselinePath = path.join(tempDir, 'relative-baseline.json');
    writeFileSync(
      summaryPath,
      JSON.stringify({
        'src/architecture/relative.ts': {
          lines: { pct: 100 },
          statements: { pct: 100 },
          functions: { pct: 100 },
          branches: { pct: 100 },
        },
      }),
    );
    writeFileSync(
      baselinePath,
      JSON.stringify({
        'src/architecture/relative.ts': {
          lines: { pct: 80 },
          statements: { pct: 80 },
          functions: { pct: 80 },
          branches: { pct: 80 },
        },
      }),
    );

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      coverageBaselinePath: path.relative(REPO_ROOT, baselinePath),
      changedFiles: ['src/architecture/relative.ts'],
    });

    assert.equal(result.pass, true);
    assert.equal(result.evidence.fileReports[0].found, true);
    assert.equal(result.evidence.fileReports[0].baseline.statements, 80);
  });

  it('passes trivially when no coverage-relevant files are changed', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {});

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      changedFiles: ['README.md'],
    });

    assert.equal(result.pass, true);
    assert.deepEqual(result.evidence.targetFiles, []);
  });

  it('allows legacy files to match their baseline coverage instead of 100%', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/architecture/legacy.ts': {
        lines: { pct: 80 },
        statements: { pct: 80 },
        functions: { pct: 100 },
        branches: { pct: 70 },
      },
    });
    const baselinePath = makeSummary(
      tempDir,
      {
        'src/architecture/legacy.ts': {
          lines: { pct: 80 },
          statements: { pct: 80 },
          functions: { pct: 100 },
          branches: { pct: 70 },
        },
      },
      'coverage-baseline.json',
    );

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      coverageBaselinePath: path.relative(REPO_ROOT, baselinePath),
      changedFiles: ['src/architecture/legacy.ts'],
    });

    assert.equal(result.pass, true);
    assert.deepEqual(result.failedFiles, []);
  });

  it('treats a missing baseline file as an empty baseline and still requires 100% for new files', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/architecture/new-no-baseline.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });
    const missingBaselinePath = path.join(tempDir, 'no-baseline.json');

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      coverageBaselinePath: path.relative(REPO_ROOT, missingBaselinePath),
      changedFiles: ['src/architecture/new-no-baseline.ts'],
    });

    assert.equal(result.pass, true);
  });

  it('fails when a legacy file regresses below its baseline', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/architecture/legacy.ts': {
        lines: { pct: 70 },
        statements: { pct: 80 },
        functions: { pct: 100 },
        branches: { pct: 80 },
      },
    });
    const baselinePath = makeSummary(
      tempDir,
      {
        'src/architecture/legacy.ts': {
          lines: { pct: 80 },
          statements: { pct: 80 },
          functions: { pct: 100 },
          branches: { pct: 80 },
        },
      },
      'coverage-baseline.json',
    );

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      coverageBaselinePath: path.relative(REPO_ROOT, baselinePath),
      changedFiles: ['src/architecture/legacy.ts'],
    });

    assert.equal(result.pass, false);
    assert.ok(result.failedFiles.includes('src/architecture/legacy.ts'));
  });

  it('treats a missing legacy file as 0% and passes when the baseline is also 0%', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {});
    const baselinePath = makeSummary(
      tempDir,
      {
        'src/architecture/not-run.ts': {
          lines: { pct: 0 },
          statements: { pct: 0 },
          functions: { pct: 0 },
          branches: { pct: 0 },
        },
      },
      'coverage-baseline.json',
    );

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      coverageBaselinePath: path.relative(REPO_ROOT, baselinePath),
      changedFiles: ['src/architecture/not-run.ts'],
    });

    assert.equal(result.pass, true);
    assert.deepEqual(result.failedFiles, []);
  });

  it('falls back to 0% for missing baseline metrics when a legacy file is not in the summary', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {});
    const baselinePath = makeSummary(
      tempDir,
      {
        'src/architecture/partial-baseline.ts': {
          lines: { pct: 0 },
        },
      },
      'coverage-baseline.json',
    );

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      coverageBaselinePath: path.relative(REPO_ROOT, baselinePath),
      changedFiles: ['src/architecture/partial-baseline.ts'],
    });

    assert.equal(result.pass, true);
    assert.deepEqual(result.evidence.fileReports[0].thresholds, {
      lines: 0,
      statements: 0,
      functions: 0,
      branches: 0,
    });
  });

  it('treats a missing legacy file as 0% and fails when the baseline is above 0%', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {});
    const baselinePath = makeSummary(
      tempDir,
      {
        'src/architecture/not-run.ts': {
          lines: { pct: 50 },
          statements: { pct: 50 },
          functions: { pct: 50 },
          branches: { pct: 50 },
        },
      },
      'coverage-baseline.json',
    );

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      coverageBaselinePath: path.relative(REPO_ROOT, baselinePath),
      changedFiles: ['src/architecture/not-run.ts'],
    });

    assert.equal(result.pass, false);
    assert.ok(result.failedFiles.includes('src/architecture/not-run.ts'));
  });

  it('still requires new files to reach 100% even when a baseline exists', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/architecture/new.ts': {
        lines: { pct: 90 },
        statements: { pct: 90 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });
    const baselinePath = makeSummary(
      tempDir,
      {
        'src/architecture/old.ts': {
          lines: { pct: 50 },
          statements: { pct: 50 },
          functions: { pct: 50 },
          branches: { pct: 50 },
        },
      },
      'coverage-baseline.json',
    );

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      coverageBaselinePath: path.relative(REPO_ROOT, baselinePath),
      changedFiles: ['src/architecture/new.ts'],
    });

    assert.equal(result.pass, false);
    assert.ok(result.failedFiles.includes('src/architecture/new.ts'));
  });

  it('treats missing current metric values as 0', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/architecture/missing-metric.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
      },
    });

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      changedFiles: ['src/architecture/missing-metric.ts'],
    });

    assert.equal(result.pass, false);
    assert.ok(
      result.failedFiles.includes('src/architecture/missing-metric.ts'),
    );
  });

  it('treats missing baseline metric values as 0 in the baseline report', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/architecture/missing-baseline-metric.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });
    const baselinePath = makeSummary(
      tempDir,
      {
        'src/architecture/missing-baseline-metric.ts': {
          lines: { pct: 100 },
          statements: { pct: 100 },
          functions: { pct: 100 },
        },
      },
      'coverage-baseline.json',
    );

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      coverageBaselinePath: path.relative(REPO_ROOT, baselinePath),
      changedFiles: ['src/architecture/missing-baseline-metric.ts'],
    });

    assert.equal(result.pass, true);
    assert.equal(result.evidence.fileReports[0].baseline.branches, 0);
  });

  it('uses default paths when called without options', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const result = await runCodeCoverageGate();
    assert.equal(typeof result.pass, 'boolean');
    assert.equal(result.owner, 'code-coverage');
  });

  it('emits JSON output from the CLI entry point', () => {
    const summaryPath = makeSummary(tempDir, {
      'src/architecture/qux.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });

    const output = runGateCli([
      '--json',
      '--coverage-summary-path',
      path.relative(REPO_ROOT, summaryPath),
      '--changed-files',
      'src/architecture/qux.ts',
    ]);

    const parsed = JSON.parse(output);
    assert.equal(parsed.pass, true);
    assert.equal(parsed.owner, 'code-coverage');
  });

  it('parses --changed-files= equal argument form from the CLI', () => {
    const summaryPath = makeSummary(tempDir, {
      'src/equal/form.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });

    const output = runGateCli([
      '--json',
      `--coverage-summary-path=${path.relative(REPO_ROOT, summaryPath)}`,
      '--changed-files=src/equal/form.ts',
    ]);

    const parsed = JSON.parse(output);
    assert.equal(parsed.pass, true);
  });

  it('treats --scripts as an alias for --changed-files from the CLI', () => {
    const summaryPath = makeSummary(tempDir, {
      'src/scripts/alias.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });

    const output = runGateCli([
      '--json',
      `--coverage-summary-path=${path.relative(REPO_ROOT, summaryPath)}`,
      '--scripts=src/scripts/alias.ts',
    ]);

    const parsed = JSON.parse(output);
    assert.equal(parsed.pass, true);
  });

  it('parses direct main() equal argument forms', async () => {
    const { main } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/equal/form.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
      'src/scripts/alias.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });

    const result = await main([
      '--json',
      `--coverage-summary-path=${path.relative(REPO_ROOT, summaryPath)}`,
      '--changed-files=src/equal/form.ts',
      '--scripts=src/scripts/alias.ts',
    ]);
    assert.equal(result.pass, true);
  });

  it('parses direct main() token argument forms', async () => {
    const { main } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/token/baseline.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });
    const baselinePath = path.join(tempDir, 'baseline-token.json');
    writeFileSync(
      baselinePath,
      JSON.stringify({
        [path.resolve(REPO_ROOT, 'src/token/baseline.ts')]: {
          lines: { pct: 90 },
          statements: { pct: 90 },
          functions: { pct: 90 },
          branches: { pct: 90 },
        },
      }),
    );

    const result = await main([
      '--json',
      '--coverage-summary-path',
      path.relative(REPO_ROOT, summaryPath),
      '--coverage-baseline-path',
      path.relative(REPO_ROOT, baselinePath),
      '--changed-files',
      'src/token/baseline.ts',
    ]);
    assert.equal(result.pass, true);
  });

  it('parses direct main() --coverage-baseline-path= equal form', async () => {
    const { main } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/equal/baseline.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });
    const baselinePath = path.join(tempDir, 'baseline-equal.json');
    writeFileSync(
      baselinePath,
      JSON.stringify({
        [path.resolve(REPO_ROOT, 'src/equal/baseline.ts')]: {
          lines: { pct: 90 },
          statements: { pct: 90 },
          functions: { pct: 90 },
          branches: { pct: 90 },
        },
      }),
    );

    const result = await main([
      '--json',
      `--coverage-summary-path=${path.relative(REPO_ROOT, summaryPath)}`,
      `--coverage-baseline-path=${path.relative(REPO_ROOT, baselinePath)}`,
      '--changed-files=src/equal/baseline.ts',
    ]);
    assert.equal(result.pass, true);
  });

  it('main() uses process.argv defaults when called without arguments', async () => {
    const { main } = await loadGate();
    const result = await main();
    assert.equal(typeof result.pass, 'boolean');
  });

  it('main() returns a failure contract when the summary is missing', async () => {
    const { main } = await loadGate();
    const result = await main([
      '--json',
      `--coverage-summary-path=${path.relative(REPO_ROOT, path.join(tempDir, 'missing-summary.json'))}`,
      '--changed-files',
      'src/direct/missing.ts',
    ]);

    assert.equal(result.pass, false);
  });

  it('main() returns a contract using the default summary path when none is provided', async () => {
    const { main } = await loadGate();
    const result = await main(['--json']);
    assert.equal(
      result.evidence.coverageSummaryPath,
      'coverage/coverage-summary.json',
    );
    assert.equal(typeof result.pass, 'boolean');
  });

  it('main() prints human-readable output when --json is omitted', async () => {
    const { main } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/human/output.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });

    const logSpy = jest
      .spyOn(console, 'log')
      .mockImplementation(() => undefined);
    try {
      await main([
        `--coverage-summary-path=${path.relative(REPO_ROOT, summaryPath)}`,
        '--changed-files',
        'src/human/output.ts',
      ]);
      assert.ok(logSpy.mock.calls.length > 0);
    } finally {
      logSpy.mockRestore();
    }
  });

  it('main() logs a fix hint when human-readable output reports a failure', async () => {
    const { main } = await loadGate();
    const logSpy = jest
      .spyOn(console, 'log')
      .mockImplementation(() => undefined);
    try {
      await main([
        `--coverage-summary-path=${path.relative(REPO_ROOT, path.join(tempDir, 'missing-summary.json'))}`,
        '--changed-files',
        'src/human/failing.ts',
      ]);
      const fixHintCall = logSpy.mock.calls.find(([a]) => a === 'fixHint:');
      assert.ok(fixHintCall);
    } finally {
      logSpy.mockRestore();
    }
  });

  it('parseGateArgs skips two-token flags when the next argument is another flag', async () => {
    const { main } = await loadGate();
    const result = await main([
      '--json',
      '--changed-files',
      '--coverage-summary-path=src/nowhere.json',
    ]);
    assert.equal(result.pass, false);
  });

  it('parseGateArgs skips --coverage-summary-path token when followed by a flag', async () => {
    const { main } = await loadGate();
    const result = await main([
      '--json',
      '--coverage-summary-path',
      '--changed-files=src/nowhere.ts',
    ]);
    assert.equal(result.pass, false);
  });

  it('parseGateArgs skips --coverage-baseline-path token when followed by a flag', async () => {
    const { main } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/baseline-flag.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });
    const result = await main([
      '--json',
      `--coverage-summary-path=${path.relative(REPO_ROOT, summaryPath)}`,
      '--coverage-baseline-path',
      '--changed-files=src/baseline-flag.ts',
    ]);
    assert.equal(result.pass, true);
  });

  it('returns a failure contract when the coverage summary cannot be read', async () => {
    const { runCodeCoverageGate } = await loadGate();

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.join(tempDir, 'does-not-exist.json'),
      changedFiles: ['src/architecture/foo.ts'],
    });

    assert.equal(result.pass, false);
  });

  it('derives changed files from git status when none are provided', async () => {
    jest.unstable_mockModule('node:child_process', () => ({
      spawnSync: jest.fn(() => ({
        status: 0,
        stdout: ' M src/git/derived.ts\n',
      })),
    }));
    jest.resetModules();

    const { main } = await import('./code-coverage.gate.mjs');
    const summaryPath = makeSummary(tempDir, {
      'src/git/derived.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });

    const result = await main([
      '--json',
      `--coverage-summary-path=${path.relative(REPO_ROOT, summaryPath)}`,
    ]);
    assert.equal(result.pass, true);
  });

  it('falls back to an empty changed-file list when git status fails', async () => {
    jest.unstable_mockModule('node:child_process', () => ({
      spawnSync: jest.fn(() => ({ status: 1, stdout: '' })),
    }));
    jest.resetModules();

    const { main } = await import('./code-coverage.gate.mjs');
    const summaryPath = makeSummary(tempDir, {});

    const result = await main([
      '--json',
      `--coverage-summary-path=${path.relative(REPO_ROOT, summaryPath)}`,
    ]);
    assert.equal(result.pass, true);
  });

  it('filterTargetFiles and isSourceFile exclude test files', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/architecture/source.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      changedFiles: [
        'src/architecture/source.ts',
        'src/architecture/source.test.ts',
        'src/__tests__/helper.ts',
        'README.md',
      ],
    });

    assert.equal(result.pass, true);
    assert.deepEqual(result.evidence.targetFiles, [
      'src/architecture/source.ts',
    ]);
  });

  it('splitPathList handles comma and newline separated paths', async () => {
    const { main } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/a.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
      'src/b.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });

    const result = await main([
      '--json',
      `--coverage-summary-path=${path.relative(REPO_ROOT, summaryPath)}`,
      '--changed-files',
      'src/a.ts\nsrc/b.ts',
    ]);
    assert.equal(result.pass, true);
  });

  it('buildFixHint includes both missing and failed files', async () => {
    const { main } = await loadGate();
    const summaryPath = makeSummary(tempDir, {});

    const result = await main([
      '--json',
      `--coverage-summary-path=${path.relative(REPO_ROOT, summaryPath)}`,
      '--changed-files',
      'src/missing-a.ts,src/missing-b.ts',
    ]);

    assert.equal(result.pass, false);
    assert.ok(result.fixHint.includes('Missing from coverage summary'));
    assert.ok(result.fixHint.includes('src/missing-a.ts'));
  });

  it('buildFixHint handles each combination of missing and failed files', async () => {
    const { buildFixHint } = await loadGate();
    assert.ok(
      buildFixHint(['src/missing.ts'], []).includes(
        'Missing from coverage summary',
      ),
    );
    assert.ok(
      buildFixHint([], ['src/failed.ts']).includes('Files below 100% coverage'),
    );
    const combined = buildFixHint(['src/missing.ts'], ['src/failed.ts']);
    assert.ok(combined.includes('Missing from coverage summary'));
    assert.ok(combined.includes('Files below 100% coverage'));
    assert.equal(buildFixHint([], []), '');
  });

  it('type-only exemption removes a file from target files', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/architecture/executable.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      changedFiles: [
        'src/architecture/executable.ts',
        'src/architecture/types.ts',
      ],
      exemptions: {
        'src/architecture/types.ts': 'type-only',
      },
    });

    assert.equal(result.pass, true);
    assert.deepEqual(result.evidence.targetFiles, [
      'src/architecture/executable.ts',
    ]);
    assert.deepEqual(result.evidence.exemptFiles, [
      { file: 'src/architecture/types.ts', kind: 'type-only' },
    ]);
  });

  it('legacy-dominant exemption accepts current coverage when no baseline exists', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/architecture/legacy.ts': {
        lines: { pct: 57.21 },
        statements: { pct: 56.66 },
        functions: { pct: 22.1 },
        branches: { pct: 64.51 },
      },
    });

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      changedFiles: ['src/architecture/legacy.ts'],
      exemptions: {
        'src/architecture/legacy.ts': 'legacy-dominant',
      },
    });

    assert.equal(result.pass, true);
    assert.equal(
      result.evidence.fileReports[0].thresholds.lines,
      result.evidence.fileReports[0].metrics.lines,
    );
    assert.equal(result.evidence.fileReports[0].exempt, 'legacy-dominant');
  });

  it('legacy-dominant exemption still uses baseline threshold when baseline exists', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/architecture/legacy.ts': {
        lines: { pct: 90 },
        statements: { pct: 90 },
        functions: { pct: 90 },
        branches: { pct: 90 },
      },
    });
    const baselinePath = makeSummary(
      tempDir,
      {
        'src/architecture/legacy.ts': {
          lines: { pct: 80 },
          statements: { pct: 80 },
          functions: { pct: 80 },
          branches: { pct: 80 },
        },
      },
      'coverage-baseline.json',
    );

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      coverageBaselinePath: path.relative(REPO_ROOT, baselinePath),
      changedFiles: ['src/architecture/legacy.ts'],
      exemptions: {
        'src/architecture/legacy.ts': 'legacy-dominant',
      },
    });

    assert.equal(result.pass, true);
    assert.equal(result.evidence.fileReports[0].thresholds.lines, 80);
  });

  it('CLI --exemptions reads exemptions from a JSON file', () => {
    const summaryPath = makeSummary(tempDir, {
      'src/exempt/legacy.ts': {
        lines: { pct: 57.21 },
        statements: { pct: 56.66 },
        functions: { pct: 22.1 },
        branches: { pct: 64.51 },
      },
      'src/exempt/fallback.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });
    const exemptionsPath = path.join(tempDir, 'exemptions.json');
    writeFileSync(
      exemptionsPath,
      JSON.stringify({
        'src/exempt/legacy.ts': 'legacy-dominant',
        'src/exempt/types.ts': 'type-only',
      }),
    );

    const output = runGateCli([
      '--json',
      `--coverage-summary-path=${path.relative(REPO_ROOT, summaryPath)}`,
      '--changed-files',
      'src/exempt/legacy.ts,src/exempt/types.ts,src/exempt/fallback.ts',
      `--exemptions=${path.relative(REPO_ROOT, exemptionsPath)}`,
    ]);

    const parsed = JSON.parse(output);
    assert.equal(parsed.pass, true);
    assert.deepEqual(parsed.evidence.targetFiles, [
      'src/exempt/legacy.ts',
      'src/exempt/fallback.ts',
    ]);
    assert.ok(
      parsed.evidence.exemptFiles.some(
        (exempt) =>
          exempt.file === 'src/exempt/legacy.ts' &&
          exempt.kind === 'legacy-dominant',
      ),
    );
  });

  it('ignores explicit changed files outside coverage-relevant directories', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {});

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      changedFiles: [
        'rag-index/docs-quality/docs-quality.metrics.mjs',
        'plans/docs-quality-metrics-gap.plans.md',
      ],
    });

    assert.equal(result.pass, true);
    assert.deepEqual(result.evidence.targetFiles, []);
  });

  it('parses --exemptions via main()', async () => {
    const { main } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/exempt/legacy.ts': {
        lines: { pct: 57.21 },
        statements: { pct: 56.66 },
        functions: { pct: 22.1 },
        branches: { pct: 64.51 },
      },
    });
    const exemptionsPath = path.join(tempDir, 'exemptions.json');
    writeFileSync(
      exemptionsPath,
      JSON.stringify({ 'src/exempt/legacy.ts': 'legacy-dominant' }),
    );

    const result = await main([
      '--json',
      `--coverage-summary-path=${path.relative(REPO_ROOT, summaryPath)}`,
      `--exemptions=${path.relative(REPO_ROOT, exemptionsPath)}`,
      '--changed-files',
      'src/exempt/legacy.ts',
    ]);

    assert.equal(result.pass, true);
  });

  it('marks a missing legacy-dominant file as exempt but still fails for 0% coverage', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {});

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      changedFiles: ['src/exempt/missing-legacy.ts'],
      exemptions: { 'src/exempt/missing-legacy.ts': 'legacy-dominant' },
    });

    assert.equal(result.pass, false);
    assert.ok(
      result.evidence.missingFiles.includes('src/exempt/missing-legacy.ts'),
    );
    const report = result.evidence.fileReports.find(
      (r) => r.file === 'src/exempt/missing-legacy.ts',
    );
    assert.equal(report.exempt, 'legacy-dominant');
  });

  it('parseGateArgs handles --exemptions as two-token flag', async () => {
    const { main } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/exempt/token.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });

    const result = await main([
      '--json',
      `--coverage-summary-path=${path.relative(REPO_ROOT, summaryPath)}`,
      '--exemptions',
      '{"src/exempt/types.ts":"type-only"}',
      '--changed-files',
      'src/exempt/token.ts,src/exempt/types.ts',
    ]);

    assert.equal(result.pass, true);
    assert.deepEqual(result.evidence.targetFiles, ['src/exempt/token.ts']);
  });

  it('loadExemptions parses inline JSON string (not a file path)', async () => {
    const { main } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/exempt/inline.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });

    const result = await main([
      '--json',
      `--coverage-summary-path=${path.relative(REPO_ROOT, summaryPath)}`,
      `--exemptions={"src/exempt/inline.ts":"type-only"}`,
      '--changed-files',
      'src/exempt/inline.ts',
    ]);

    assert.equal(result.pass, true);
    assert.deepEqual(result.evidence.targetFiles, []);
  });

  it('loadExemptions parses single-quoted inline JSON string', async () => {
    const { main } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/exempt/quoted.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });

    const result = await main([
      '--json',
      `--coverage-summary-path=${path.relative(REPO_ROOT, summaryPath)}`,
      `--exemptions='{"src/exempt/quoted.ts":"type-only"}'`,
      '--changed-files',
      'src/exempt/quoted.ts',
    ]);

    assert.equal(result.pass, true);
    assert.deepEqual(result.evidence.targetFiles, []);
  });

  it('normalizeExemptions throws on unsupported exemption kind', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/bad/exempt.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });

    await assert.rejects(
      runCodeCoverageGate({
        coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
        changedFiles: ['src/bad/exempt.ts'],
        exemptions: { 'src/bad/exempt.ts': 'unsupported-kind' },
      }),
      (error) => {
        assert.ok(error.message.includes('Unsupported exemption kind'));
        return true;
      },
    );
  });

  it('normalizeExemptions handles object-form exemption with kind property', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/obj/exempt.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      changedFiles: ['src/obj/exempt.ts', 'src/obj/types.ts'],
      exemptions: { 'src/obj/types.ts': { kind: 'type-only' } },
    });

    assert.equal(result.pass, true);
    assert.deepEqual(result.evidence.exemptFiles, [
      { file: 'src/obj/types.ts', kind: 'type-only' },
    ]);
  });

  it('skips --exemptions value when next arg starts with --', async () => {
    const { main } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/test/covered.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });

    const result = await main([
      '--exemptions',
      '--json',
      `--coverage-summary-path=${path.relative(REPO_ROOT, summaryPath)}`,
      '--coverage-baseline-path=nonexistent-baseline.json',
      '--changed-files',
      'src/test/covered.ts',
    ]);

    assert.equal(result.pass, true);
  });

  it('resolves legacy-dominant threshold with missing metric in entry', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/legacy/missing-metric.ts': {
        lines: { pct: 80 },
        statements: { pct: 75 },
        functions: { pct: 50 },
      },
    });

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      coverageBaselinePath: 'nonexistent-baseline.json',
      changedFiles: ['src/legacy/missing-metric.ts'],
      exemptions: { 'src/legacy/missing-metric.ts': 'legacy-dominant' },
    });

    assert.equal(result.pass, true);
  });
});
