/**
 * @module code-coverage.gate.test
 * @description Green tests for the code-coverage gate contract.
 */
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const REPO_ROOT = path.resolve();

/**
 * Lazily load the ESM gate module inside each test so Jest does not try to
 * transform its top-level await through the mjs-to-cjs transformer.
 */
async function loadGate() {
  // @ts-ignore - tested module is authored in plain ESM without a declaration file.
  return import('./code-coverage.gate.mjs');
}

function makeSummary(
  dir: string,
  entries: Record<string, Record<string, { pct: number }>>,
  fileName = 'coverage-summary.json',
) {
  const summaryPath = path.join(dir, fileName);
  const absoluteEntries: Record<string, Record<string, { pct: number }>> = {};
  for (const [key, metrics] of Object.entries(entries)) {
    absoluteEntries[path.resolve(REPO_ROOT, key)] = metrics;
  }
  writeFileSync(summaryPath, JSON.stringify(absoluteEntries, null, 2));
  return summaryPath;
}

describe('code-coverage gate contract', () => {
  let tempDir: string;

  beforeEach(() => {
    tempDir = mkdtempSync(path.join(os.tmpdir(), 'coverage-gate-'));
  });

  afterEach(() => {
    rmSync(tempDir, { recursive: true, force: true });
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

    expect(result.pass).toBe(true);
    expect(result.failedFiles).toEqual([]);
    expect(result.missingFiles).toEqual([]);
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

    expect(result.pass).toBe(false);
    expect(result.failedFiles).toContain('src/architecture/bar.ts');
  });

  it('reports files missing from the coverage summary', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {});

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      changedFiles: ['src/architecture/missing.ts'],
    });

    expect(result.pass).toBe(false);
    expect(result.missingFiles).toContain('src/architecture/missing.ts');
  });

  it('passes trivially when no coverage-relevant files are changed', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {});

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      changedFiles: ['README.md'],
    });

    expect(result.pass).toBe(true);
    expect(result.evidence.targetFiles).toEqual([]);
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

    const output = execFileSync(
      process.execPath,
      [
        path.resolve(
          REPO_ROOT,
          'scripts/agent-customization/gates/code-coverage.gate.mjs',
        ),
        '--json',
        '--coverage-summary-path',
        path.relative(REPO_ROOT, summaryPath),
        '--changed-files',
        'src/architecture/qux.ts',
      ],
      {
        cwd: REPO_ROOT,
        encoding: 'utf8',
      },
    );

    const parsed = JSON.parse(output);
    expect(parsed.pass).toBe(true);
    expect(parsed.owner).toBe('code-coverage');
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

    expect(result.pass).toBe(true);
    expect(result.failedFiles).toEqual([]);
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

    expect(result.pass).toBe(false);
    expect(result.failedFiles).toContain('src/architecture/legacy.ts');
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

    expect(result.pass).toBe(true);
    expect(result.failedFiles).toEqual([]);
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

    expect(result.pass).toBe(true);
    expect(result.evidence.fileReports[0].thresholds).toEqual({
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

    expect(result.pass).toBe(false);
    expect(result.failedFiles).toContain('src/architecture/not-run.ts');
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

    expect(result.pass).toBe(false);
    expect(result.failedFiles).toContain('src/architecture/new.ts');
  });

  it('accepts an explicit baseline path from the CLI', () => {
    const summaryPath = makeSummary(tempDir, {
      'src/architecture/qux.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });
    const baselinePath = makeSummary(tempDir, {}, 'coverage-baseline.json');

    const output = execFileSync(
      process.execPath,
      [
        path.resolve(
          REPO_ROOT,
          'scripts/agent-customization/gates/code-coverage.gate.mjs',
        ),
        '--json',
        '--coverage-summary-path',
        path.relative(REPO_ROOT, summaryPath),
        '--coverage-baseline-path',
        path.relative(REPO_ROOT, baselinePath),
        '--changed-files',
        'src/architecture/qux.ts',
      ],
      {
        cwd: REPO_ROOT,
        encoding: 'utf8',
      },
    );

    const parsed = JSON.parse(output);
    expect(parsed.pass).toBe(true);
  });

  it('parses --changed-files= equal argument form', () => {
    const summaryPath = makeSummary(tempDir, {
      'src/equal/form.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });

    const output = execFileSync(
      process.execPath,
      [
        path.resolve(
          REPO_ROOT,
          'scripts/agent-customization/gates/code-coverage.gate.mjs',
        ),
        '--json',
        `--coverage-summary-path=${path.relative(REPO_ROOT, summaryPath)}`,
        '--changed-files=src/equal/form.ts',
      ],
      {
        cwd: REPO_ROOT,
        encoding: 'utf8',
      },
    );

    const parsed = JSON.parse(output);
    expect(parsed.pass).toBe(true);
  });

  it('treats --scripts as an alias for --changed-files', () => {
    const summaryPath = makeSummary(tempDir, {
      'src/scripts/alias.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });

    const output = execFileSync(
      process.execPath,
      [
        path.resolve(
          REPO_ROOT,
          'scripts/agent-customization/gates/code-coverage.gate.mjs',
        ),
        '--json',
        `--coverage-summary-path=${path.relative(REPO_ROOT, summaryPath)}`,
        '--scripts=src/scripts/alias.ts',
      ],
      {
        cwd: REPO_ROOT,
        encoding: 'utf8',
      },
    );

    const parsed = JSON.parse(output);
    expect(parsed.pass).toBe(true);
  });

  it('parses direct main() --changed-files= and --scripts= equal forms', async () => {
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
    expect(result.pass).toBe(true);
  });

  it('parses direct main() --coverage-summary-path and --coverage-baseline-path token forms', async () => {
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
    expect(result.pass).toBe(true);
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
    expect(result.pass).toBe(true);
  });

  it('exposes main() for direct testing with controlled argv', async () => {
    const { main } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/direct/main.ts': {
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
      'src/direct/main.ts',
    ]);

    expect(result.pass).toBe(true);
  });

  it('runCodeCoverageGate() uses default paths when called without options', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const result = await runCodeCoverageGate();
    expect(typeof result.pass).toBe('boolean');
    expect(result.owner).toBe('code-coverage');
  });

  it('main() uses process.argv defaults when called without arguments', async () => {
    const { main } = await loadGate();
    const result = await main();
    expect(typeof result.pass).toBe('boolean');
  });

  it('treats missing current metric values as 0', async () => {
    const { runCodeCoverageGate } = await loadGate();
    const summaryPath = makeSummary(tempDir, {
      'src/architecture/missing-metric.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        // branches metric intentionally omitted
      },
    });

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      changedFiles: ['src/architecture/missing-metric.ts'],
    });

    expect(result.pass).toBe(false);
    expect(result.failedFiles).toContain('src/architecture/missing-metric.ts');
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
          // branches metric intentionally omitted
        },
      },
      'coverage-baseline.json',
    );

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.relative(REPO_ROOT, summaryPath),
      coverageBaselinePath: path.relative(REPO_ROOT, baselinePath),
      changedFiles: ['src/architecture/missing-baseline-metric.ts'],
    });

    expect(result.pass).toBe(true);
    expect(result.evidence.fileReports[0]?.baseline).toEqual(
      expect.objectContaining({ branches: 0 }),
    );
  });

  it('main() returns a failure contract when the summary is missing', async () => {
    const { main } = await loadGate();
    const result = await main([
      '--json',
      `--coverage-summary-path=${path.relative(REPO_ROOT, path.join(tempDir, 'missing-summary.json'))}`,
      '--changed-files',
      'src/direct/missing.ts',
    ]);

    expect(result.pass).toBe(false);
  });

  it('main() returns a failure contract when no summary path is provided', async () => {
    const { main } = await loadGate();
    const result = await main([
      '--json',
      '--changed-files',
      'src/no-summary-path.ts',
    ]);
    expect(result.pass).toBe(false);
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
      expect(logSpy).toHaveBeenCalled();
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
      expect(fixHintCall).toBeDefined();
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
    expect(result.pass).toBe(false);
  });

  it('parseGateArgs skips --coverage-summary-path token when followed by a flag', async () => {
    const { main } = await loadGate();
    const result = await main([
      '--json',
      '--coverage-summary-path',
      '--changed-files=src/nowhere.ts',
    ]);
    expect(result.pass).toBe(false);
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
    expect(result.pass).toBe(true);
  });

  it('executes the CLI entry path when imported as the main module', () => {
    jest.resetModules();
    const originalArgv = process.argv;
    const modulePath = path.resolve(
      REPO_ROOT,
      'scripts/agent-customization/gates/code-coverage.gate.mjs',
    );
    const summaryPath = makeSummary(tempDir, {
      'src/cli/entry.ts': {
        lines: { pct: 100 },
        statements: { pct: 100 },
        functions: { pct: 100 },
        branches: { pct: 100 },
      },
    });
    process.argv = [
      'node',
      modulePath,
      '--json',
      `--coverage-summary-path=${path.relative(REPO_ROOT, summaryPath)}`,
      '--changed-files=src/cli/entry.ts',
    ];

    return jest
      .isolateModulesAsync(async () => {
        // @ts-ignore - tested module is authored in plain ESM without a declaration file.
        await import('./code-coverage.gate.mjs');
        // Give the async top-level entry a tick to finish.
        await new Promise((resolve) => setTimeout(resolve, 200));
      })
      .finally(() => {
        process.argv = originalArgv;
      });
  });

  it('returns a failure contract when the coverage summary cannot be read', async () => {
    const { runCodeCoverageGate } = await loadGate();

    const result = await runCodeCoverageGate({
      coverageSummaryPath: path.join(tempDir, 'does-not-exist.json'),
      changedFiles: ['src/architecture/foo.ts'],
    });

    expect(result.pass).toBe(false);
  });

  it('derives changed files from git status when none are provided', () => {
    jest.resetModules();
    jest.doMock('node:child_process', () => ({
      spawnSync: jest.fn(() => ({
        status: 0,
        stdout: ' M src/git/derived.ts\n',
      })),
    }));

    return jest.isolateModulesAsync(async () => {
      // @ts-ignore - tested module is authored in plain ESM without a declaration file.
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
      expect(result.pass).toBe(true);
    });
  });

  it('falls back to an empty changed-file list when git status fails', () => {
    jest.resetModules();
    jest.doMock('node:child_process', () => ({
      spawnSync: jest.fn(() => ({ status: 1, stdout: '' })),
    }));

    return jest.isolateModulesAsync(async () => {
      // @ts-ignore - tested module is authored in plain ESM without a declaration file.
      const { main } = await import('./code-coverage.gate.mjs');
      const summaryPath = makeSummary(tempDir, {});

      const result = await main([
        '--json',
        `--coverage-summary-path=${path.relative(REPO_ROOT, summaryPath)}`,
      ]);
      expect(result.pass).toBe(true);
    });
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

    expect(result.pass).toBe(true);
    expect(result.evidence.targetFiles).toEqual([
      'src/architecture/executable.ts',
    ]);
    expect(result.evidence.exemptFiles).toEqual([
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

    expect(result.pass).toBe(true);
    expect(result.evidence.fileReports[0].thresholds.lines).toEqual(
      result.evidence.fileReports[0].metrics.lines,
    );
    expect(result.evidence.fileReports[0].exempt).toBe('legacy-dominant');
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

    expect(result.pass).toBe(true);
    expect(result.evidence.fileReports[0].thresholds.lines).toBe(80);
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

    const output = execFileSync(
      process.execPath,
      [
        path.resolve(
          REPO_ROOT,
          'scripts/agent-customization/gates/code-coverage.gate.mjs',
        ),
        '--json',
        `--coverage-summary-path=${path.relative(REPO_ROOT, summaryPath)}`,
        '--changed-files',
        'src/exempt/legacy.ts,src/exempt/types.ts,src/exempt/fallback.ts',
        `--exemptions=${path.relative(REPO_ROOT, exemptionsPath)}`,
      ],
      {
        cwd: REPO_ROOT,
        encoding: 'utf8',
      },
    );

    const parsed = JSON.parse(output);
    expect(parsed.pass).toBe(true);
    expect(parsed.evidence.targetFiles).toEqual([
      'src/exempt/legacy.ts',
      'src/exempt/fallback.ts',
    ]);
    expect(
      parsed.evidence.exemptFiles.some(
        (exempt: { file: string; kind: string }) =>
          exempt.file === 'src/exempt/legacy.ts' &&
          exempt.kind === 'legacy-dominant',
      ),
    ).toBe(true);
  });
});
