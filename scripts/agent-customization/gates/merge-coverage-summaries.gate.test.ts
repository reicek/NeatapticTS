/**
 * @module merge-coverage-summaries.gate.test
 * @description Green tests for the coverage-summary merge utility.
 */
import {
  mkdtempSync,
  rmSync,
  writeFileSync,
  readFileSync,
  mkdirSync,
  existsSync,
} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const REPO_ROOT = path.resolve();

/**
 * Lazily load the ESM merge module inside each test so Jest does not try to
 * transform its top-level await through the mjs-to-cjs transformer.
 */
async function loadMerge() {
  // @ts-ignore - tested module is authored in plain ESM without a declaration file.
  return import('./merge-coverage-summaries.mjs');
}

function makeSummaryData(
  fileName: string,
  overrides: Record<string, number> = {},
) {
  const absolutePath = path.join(REPO_ROOT, fileName);
  return {
    [absolutePath]: {
      lines: {
        total: 10,
        covered: overrides.linesCovered ?? 10,
        skipped: 0,
        pct: overrides.linesPct ?? 100,
      },
      statements: {
        total: 12,
        covered: overrides.statementsCovered ?? 12,
        skipped: 0,
        pct: overrides.statementsPct ?? 100,
      },
      functions: {
        total: 3,
        covered: overrides.functionsCovered ?? 3,
        skipped: 0,
        pct: overrides.functionsPct ?? 100,
      },
      branches: {
        total: 4,
        covered: overrides.branchesCovered ?? 4,
        skipped: 0,
        pct: overrides.branchesPct ?? 100,
      },
    },
  };
}

function makeTotal() {
  return {
    total: {
      lines: { total: 10, covered: 10, skipped: 0, pct: 100 },
      statements: { total: 12, covered: 12, skipped: 0, pct: 100 },
      functions: { total: 3, covered: 3, skipped: 0, pct: 100 },
      branches: { total: 4, covered: 4, skipped: 0, pct: 100 },
      branchesTrue: { total: 0, covered: 0, skipped: 0, pct: 100 },
    },
  };
}

describe('merge-coverage-summaries utility contract', () => {
  let tempDir: string;
  let mergeCoverageSummaries: (options?: {
    coverageDir?: string;
    summaryPath?: string;
  }) => Promise<{ mergedFiles: string[]; summaryPath: string }>;

  beforeEach(async () => {
    tempDir = mkdtempSync(path.join(os.tmpdir(), 'merge-cov-'));
    ({ mergeCoverageSummaries } = await loadMerge());
  });

  afterEach(() => {
    rmSync(tempDir, { recursive: true, force: true });
  });

  function runMerge(options?: { coverageDir: string; summaryPath?: string }) {
    return mergeCoverageSummaries(
      options ?? { coverageDir: path.join(tempDir, 'coverage') },
    );
  }

  it('merges multiple project coverage-summary.json files into one summary', async () => {
    const coverageDir = path.join(tempDir, 'coverage');
    const projectA = path.join(coverageDir, 'project-a');
    const projectB = path.join(coverageDir, 'project-b');
    const summaryPath = path.join(coverageDir, 'merged-summary.json');
    mkdirSync(projectA, { recursive: true });
    mkdirSync(projectB, { recursive: true });
    writeFileSync(
      path.join(projectA, 'coverage-summary.json'),
      JSON.stringify({ ...makeTotal(), ...makeSummaryData('src/a.ts') }),
    );
    writeFileSync(
      path.join(projectB, 'coverage-summary.json'),
      JSON.stringify({ ...makeTotal(), ...makeSummaryData('src/b.ts') }),
    );

    const result = await runMerge({ coverageDir, summaryPath });
    expect(result.mergedFiles.length).toBeGreaterThanOrEqual(2);
    const merged = JSON.parse(readFileSync(summaryPath, 'utf8'));
    expect(merged.total.statements.pct).toBe(100);
    expect(merged[path.join(REPO_ROOT, 'src/a.ts')]).toBeDefined();
    expect(merged[path.join(REPO_ROOT, 'src/b.ts')]).toBeDefined();
    expect(path.resolve(REPO_ROOT, result.summaryPath)).toBe(summaryPath);
  });

  it('skips project directories that have no coverage-summary.json yet', async () => {
    const coverageDir = path.join(tempDir, 'coverage');
    const projectA = path.join(coverageDir, 'project-a');
    const projectEmpty = path.join(coverageDir, 'project-empty');
    mkdirSync(projectA, { recursive: true });
    mkdirSync(projectEmpty, { recursive: true });
    writeFileSync(
      path.join(projectA, 'coverage-summary.json'),
      JSON.stringify(makeSummaryData('src/a.ts')),
    );

    const result = await runMerge({ coverageDir });
    expect(result.mergedFiles).toHaveLength(1);
  });

  it('ignores non-project entries in the coverage directory', async () => {
    const coverageDir = path.join(tempDir, 'coverage');
    const projectA = path.join(coverageDir, 'project-a');
    const otherDir = path.join(coverageDir, 'other');
    const strayFile = path.join(coverageDir, 'not-a-dir.txt');
    mkdirSync(projectA, { recursive: true });
    mkdirSync(otherDir, { recursive: true });
    writeFileSync(strayFile, 'x');
    writeFileSync(
      path.join(projectA, 'coverage-summary.json'),
      JSON.stringify(makeSummaryData('src/a.ts')),
    );

    const result = await runMerge({ coverageDir });
    expect(result.mergedFiles).toHaveLength(1);
  });

  it('throws when no project coverage-summary.json files exist', async () => {
    const coverageDir = path.join(tempDir, 'coverage');
    mkdirSync(path.join(coverageDir, 'project-empty'), { recursive: true });
    await expect(runMerge({ coverageDir })).rejects.toThrow(
      /No coverage-summary.json files found/,
    );
  });

  it('emits merged summary metadata from the CLI entry point', () => {
    const projectA = path.join(tempDir, 'coverage', 'project-cli');
    mkdirSync(projectA, { recursive: true });
    writeFileSync(
      path.join(projectA, 'coverage-summary.json'),
      JSON.stringify(makeSummaryData('src/cli.ts')),
    );

    const coverageDir = path.join(tempDir, 'coverage');
    const summaryPath = path.join(coverageDir, 'cli-summary.json');
    const output = execFileSync(
      process.execPath,
      [
        path.resolve(
          REPO_ROOT,
          'scripts/agent-customization/gates/merge-coverage-summaries.mjs',
        ),
        '--coverage-dir',
        coverageDir,
        '--summary-path',
        summaryPath,
      ],
      { cwd: REPO_ROOT, encoding: 'utf8' },
    );

    const parsed = JSON.parse(output);
    expect(Array.isArray(parsed.mergedFiles)).toBe(true);
    expect(parsed.mergedFiles.length).toBeGreaterThanOrEqual(1);
    expect(path.resolve(REPO_ROOT, parsed.summaryPath)).toBe(summaryPath);
  });

  it('parses --coverage-dir= and --summary-path= argument forms', () => {
    const projectA = path.join(tempDir, 'coverage', 'project-eq');
    mkdirSync(projectA, { recursive: true });
    writeFileSync(
      path.join(projectA, 'coverage-summary.json'),
      JSON.stringify(makeSummaryData('src/eq.ts')),
    );

    const coverageDir = path.join(tempDir, 'coverage');
    const summaryPath = path.join(coverageDir, 'eq-summary.json');
    const output = execFileSync(
      process.execPath,
      [
        path.resolve(
          REPO_ROOT,
          'scripts/agent-customization/gates/merge-coverage-summaries.mjs',
        ),
        `--coverage-dir=${coverageDir}`,
        `--summary-path=${summaryPath}`,
      ],
      { cwd: REPO_ROOT, encoding: 'utf8' },
    );

    const parsed = JSON.parse(output);
    expect(parsed.mergedFiles.length).toBeGreaterThanOrEqual(1);
    expect(path.resolve(REPO_ROOT, parsed.summaryPath)).toBe(summaryPath);
  });

  it('exposes main() for direct testing with controlled argv', async () => {
    const projectA = path.join(tempDir, 'coverage', 'project-main');
    mkdirSync(projectA, { recursive: true });
    writeFileSync(
      path.join(projectA, 'coverage-summary.json'),
      JSON.stringify(makeSummaryData('src/main.ts')),
    );

    const coverageDir = path.join(tempDir, 'coverage');
    const summaryPath = path.join(coverageDir, 'main-summary.json');
    const { main } = await loadMerge();
    const result = await main([
      `--coverage-dir=${coverageDir}`,
      `--summary-path=${summaryPath}`,
    ]);

    expect(result.mergedFiles.length).toBeGreaterThanOrEqual(1);
    expect(path.resolve(REPO_ROOT, result.summaryPath)).toBe(summaryPath);
  });

  it('parses --coverage-dir and --summary-path token forms via parseCliOptions', async () => {
    const projectA = path.join(tempDir, 'coverage', 'project-token');
    mkdirSync(projectA, { recursive: true });
    writeFileSync(
      path.join(projectA, 'coverage-summary.json'),
      JSON.stringify(makeSummaryData('src/token.ts')),
    );

    const coverageDir = path.join(tempDir, 'coverage');
    const summaryPath = path.join(coverageDir, 'token-summary.json');
    const { parseCliOptions, main } = await loadMerge();
    const options = parseCliOptions([
      '--coverage-dir',
      coverageDir,
      '--summary-path',
      summaryPath,
    ]);
    expect(options.coverageDir).toBe(path.resolve(coverageDir));
    expect(options.summaryPath).toBe(path.resolve(summaryPath));

    const result = await main([
      '--coverage-dir',
      coverageDir,
      '--summary-path',
      summaryPath,
    ]);
    expect(result.mergedFiles.length).toBeGreaterThanOrEqual(1);
  });

  it('parseCliOptions ignores flags without a following value', async () => {
    const { parseCliOptions } = await loadMerge();
    expect(parseCliOptions([])).toEqual({});
    expect(parseCliOptions(['--coverage-dir'])).toEqual({});
    expect(parseCliOptions(['--summary-path'])).toEqual({});
  });

  it('parseCliOptions ignores unrecognized arguments', async () => {
    const { parseCliOptions } = await loadMerge();
    expect(parseCliOptions(['--unknown', 'value', 'extra'])).toEqual({});
    expect(parseCliOptions(['foo'])).toEqual({});
  });

  it('main() and mergeCoverageSummaries() use default coverage directory when called without arguments', async () => {
    const defaultProject = path.join(
      REPO_ROOT,
      'coverage',
      'project-merge-default',
    );
    mkdirSync(defaultProject, { recursive: true });
    const finalPath = path.join(defaultProject, 'coverage-summary.json');
    writeFileSync(finalPath, JSON.stringify(makeSummaryData('src/default.ts')));
    const defaultSummary = path.join(
      REPO_ROOT,
      'coverage',
      'coverage-summary.json',
    );
    const { main, mergeCoverageSummaries } = await loadMerge();
    try {
      const mainResult = await main();
      expect(typeof mainResult.summaryPath).toBe('string');
      const directResult = await mergeCoverageSummaries();
      expect(typeof directResult.summaryPath).toBe('string');
    } finally {
      rmSync(defaultProject, { recursive: true, force: true });
      if (existsSync(defaultSummary)) {
        rmSync(defaultSummary, { force: true });
      }
    }
  });

  it('prints merge result JSON to stdout via printMergeResult', async () => {
    const { printMergeResult } = await loadMerge();
    const logSpy = jest
      .spyOn(console, 'log')
      .mockImplementation(() => undefined);
    try {
      printMergeResult({ mergedFiles: ['a'], summaryPath: 's.json' });
      expect(logSpy).toHaveBeenCalledWith(
        JSON.stringify({ mergedFiles: ['a'], summaryPath: 's.json' }, null, 2),
      );
    } finally {
      logSpy.mockRestore();
    }
  });

  it('prints merge errors to stderr and sets exit code via printMergeError', async () => {
    const { printMergeError } = await loadMerge();
    const errorSpy = jest
      .spyOn(console, 'error')
      .mockImplementation(() => undefined);
    const originalExitCode = process.exitCode;
    try {
      printMergeError(new Error('boom'));
      expect(errorSpy).toHaveBeenCalledWith('boom');
      expect(process.exitCode).toBe(1);
    } finally {
      errorSpy.mockRestore();
      process.exitCode = originalExitCode;
    }
  });

  it('picks the entry with the better line coverage when statement coverage is tied', async () => {
    const coverageDir = path.join(tempDir, 'coverage');
    const projectA = path.join(coverageDir, 'project-a');
    const projectB = path.join(coverageDir, 'project-b');
    const summaryPath = path.join(coverageDir, 'merged-summary.json');
    mkdirSync(projectA, { recursive: true });
    mkdirSync(projectB, { recursive: true });

    const fileName = 'src/tie.ts';
    const tiedStatements = {
      total: 12,
      covered: 6,
      skipped: 0,
      pct: 50,
    };
    const projectAData = makeSummaryData(fileName, {
      statementsCovered: 6,
      statementsPct: 50,
      linesCovered: 5,
      linesPct: 50,
    });
    const projectBData = makeSummaryData(fileName, {
      statementsCovered: 6,
      statementsPct: 50,
      linesCovered: 8,
      linesPct: 80,
    });

    writeFileSync(
      path.join(projectA, 'coverage-summary.json'),
      JSON.stringify({ ...makeTotal(), ...projectAData }),
    );
    writeFileSync(
      path.join(projectB, 'coverage-summary.json'),
      JSON.stringify({ ...makeTotal(), ...projectBData }),
    );

    await runMerge({ coverageDir, summaryPath });
    const merged = JSON.parse(readFileSync(summaryPath, 'utf8'));
    expect(merged[path.join(REPO_ROOT, fileName)].lines.pct).toBe(80);
  });

  it('replaces an existing entry when the candidate has better statement coverage', async () => {
    const coverageDir = path.join(tempDir, 'coverage');
    const projectA = path.join(coverageDir, 'project-a');
    const projectB = path.join(coverageDir, 'project-b');
    const summaryPath = path.join(coverageDir, 'merged-summary.json');
    mkdirSync(projectA, { recursive: true });
    mkdirSync(projectB, { recursive: true });

    const fileName = 'src/better-statements.ts';
    const projectAData = makeSummaryData(fileName, {
      statementsCovered: 6,
      statementsPct: 50,
    });
    const projectBData = makeSummaryData(fileName, {
      statementsCovered: 10,
      statementsPct: 83,
    });

    writeFileSync(
      path.join(projectA, 'coverage-summary.json'),
      JSON.stringify({ ...makeTotal(), ...projectAData }),
    );
    writeFileSync(
      path.join(projectB, 'coverage-summary.json'),
      JSON.stringify({ ...makeTotal(), ...projectBData }),
    );

    await runMerge({ coverageDir, summaryPath });
    const merged = JSON.parse(readFileSync(summaryPath, 'utf8'));
    expect(merged[path.join(REPO_ROOT, fileName)].statements.pct).toBe(83);
  });

  it('keeps the existing entry when the candidate is worse', async () => {
    const coverageDir = path.join(tempDir, 'coverage');
    const projectA = path.join(coverageDir, 'project-a');
    const projectB = path.join(coverageDir, 'project-b');
    const summaryPath = path.join(coverageDir, 'merged-summary.json');
    mkdirSync(projectA, { recursive: true });
    mkdirSync(projectB, { recursive: true });

    const fileName = 'src/worse.ts';
    const projectAData = makeSummaryData(fileName, {
      statementsCovered: 10,
      statementsPct: 83,
    });
    const projectBData = makeSummaryData(fileName, {
      statementsCovered: 6,
      statementsPct: 50,
    });

    writeFileSync(
      path.join(projectA, 'coverage-summary.json'),
      JSON.stringify({ ...makeTotal(), ...projectAData }),
    );
    writeFileSync(
      path.join(projectB, 'coverage-summary.json'),
      JSON.stringify({ ...makeTotal(), ...projectBData }),
    );

    await runMerge({ coverageDir, summaryPath });
    const merged = JSON.parse(readFileSync(summaryPath, 'utf8'));
    expect(merged[path.join(REPO_ROOT, fileName)].statements.pct).toBe(83);
  });

  it('skips missing metrics while computing the total', async () => {
    const coverageDir = path.join(tempDir, 'coverage');
    const projectA = path.join(coverageDir, 'project-a');
    mkdirSync(projectA, { recursive: true });

    const absolutePath = path.join(REPO_ROOT, 'src/missing.ts');
    const partial = {
      [absolutePath]: {
        lines: { total: 10, covered: 10, skipped: 0, pct: 100 },
      },
    };

    writeFileSync(
      path.join(projectA, 'coverage-summary.json'),
      JSON.stringify({ ...makeTotal(), ...partial }),
    );

    const summaryPath = path.join(coverageDir, 'merged-summary.json');
    await runMerge({ coverageDir, summaryPath });
    const merged = JSON.parse(readFileSync(summaryPath, 'utf8'));
    expect(merged.total.lines.pct).toBe(100);
    expect(merged.total.statements.pct).toBe(100);
  });

  it('prints a CLI error and exits non-zero on failure', () => {
    const emptyDir = path.join(tempDir, 'empty-coverage');
    mkdirSync(emptyDir, { recursive: true });
    let err: { status?: number } | undefined;
    try {
      execFileSync(
        process.execPath,
        [
          path.resolve(
            REPO_ROOT,
            'scripts/agent-customization/gates/merge-coverage-summaries.mjs',
          ),
          '--coverage-dir',
          emptyDir,
        ],
        { cwd: REPO_ROOT, encoding: 'utf8' },
      );
    } catch (caught) {
      err = caught as { status?: number };
    }
    expect(err).toBeDefined();
    expect(err?.status).not.toBe(0);
  });
});
