/**
 * @module merge-coverage-summaries.test
 * @description Native-ESM coverage tests for merge-coverage-summaries.mjs.
 *
 * Runs in the agent-customization-mjs Jest project so V8 instruments the
 * source .mjs file directly, producing accurate coverage for the merge script.
 */
import { jest } from '@jest/globals';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
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

const REPO_ROOT = path.resolve();
const MERGE_PATH = path.resolve(
  REPO_ROOT,
  'scripts/agent-customization/gates/merge-coverage-summaries.mjs',
);

function makeSummaryData(fileName, overrides = {}) {
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

async function loadMerge() {
  return import('./merge-coverage-summaries.mjs');
}

describe('merge-coverage-summaries native-ESM coverage', () => {
  let tempDir;

  beforeEach(() => {
    tempDir = mkdtempSync(path.join(os.tmpdir(), 'merge-cov-mjs-'));
  });

  afterEach(() => {
    rmSync(tempDir, { recursive: true, force: true });
  });

  it('merges multiple project coverage-summary.json files into one summary', async () => {
    const { mergeCoverageSummaries } = await loadMerge();
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

    const result = await mergeCoverageSummaries({ coverageDir, summaryPath });
    assert.ok(result.mergedFiles.length >= 2);
    const merged = JSON.parse(readFileSync(summaryPath, 'utf8'));
    assert.equal(merged.total.statements.pct, 100);
    assert.ok(merged['src/a.ts']);
    assert.ok(merged['src/b.ts']);
  });

  it('toRepoRelativeKey preserves total and converts absolute paths', async () => {
    const { toRepoRelativeKey } = await loadMerge();
    assert.equal(toRepoRelativeKey('total'), 'total');
    assert.equal(
      toRepoRelativeKey(path.join(REPO_ROOT, 'src/x.ts')),
      'src/x.ts',
    );
  });

  it('discovers coverage-summary.json in nested subdirectories', async () => {
    const { mergeCoverageSummaries } = await loadMerge();
    const coverageDir = path.join(tempDir, 'coverage');
    const nestedProject = path.join(coverageDir, 'nested', 'project-a');
    mkdirSync(nestedProject, { recursive: true });
    writeFileSync(
      path.join(nestedProject, 'coverage-summary.json'),
      JSON.stringify({ ...makeTotal(), ...makeSummaryData('src/nested.ts') }),
    );

    const result = await mergeCoverageSummaries({ coverageDir });
    assert.equal(result.mergedFiles.length, 1);
    const merged = JSON.parse(
      readFileSync(path.join(coverageDir, 'coverage-summary.json'), 'utf8'),
    );
    assert.ok(merged['src/nested.ts']);
  });

  it('discovers coverage-summary.json in non-project-named directories', async () => {
    const { mergeCoverageSummaries } = await loadMerge();
    const coverageDir = path.join(tempDir, 'coverage');
    const customDir = path.join(coverageDir, 'custom-named-dir');
    mkdirSync(customDir, { recursive: true });
    writeFileSync(
      path.join(customDir, 'coverage-summary.json'),
      JSON.stringify({ ...makeTotal(), ...makeSummaryData('src/custom.ts') }),
    );

    const result = await mergeCoverageSummaries({ coverageDir });
    assert.equal(result.mergedFiles.length, 1);
    const merged = JSON.parse(
      readFileSync(path.join(coverageDir, 'coverage-summary.json'), 'utf8'),
    );
    assert.ok(merged['src/custom.ts']);
  });

  it('ignores the previous merged coverage-summary.json output path', async () => {
    const { mergeCoverageSummaries } = await loadMerge();
    const coverageDir = path.join(tempDir, 'coverage');
    const projectA = path.join(coverageDir, 'project-a');
    const summaryPath = path.join(coverageDir, 'coverage-summary.json');
    mkdirSync(projectA, { recursive: true });
    writeFileSync(
      path.join(projectA, 'coverage-summary.json'),
      JSON.stringify({ ...makeTotal(), ...makeSummaryData('src/a.ts') }),
    );
    writeFileSync(
      summaryPath,
      JSON.stringify({ ...makeTotal(), ...makeSummaryData('src/old.ts') }),
    );

    const result = await mergeCoverageSummaries({ coverageDir, summaryPath });
    assert.equal(result.mergedFiles.length, 1);
    const merged = JSON.parse(readFileSync(summaryPath, 'utf8'));
    assert.ok(merged['src/a.ts']);
    assert.equal(merged['src/old.ts'], undefined);
  });

  it('skips project directories that have no coverage-summary.json yet', async () => {
    const { mergeCoverageSummaries } = await loadMerge();
    const coverageDir = path.join(tempDir, 'coverage');
    const projectA = path.join(coverageDir, 'project-a');
    const projectEmpty = path.join(coverageDir, 'project-empty');
    mkdirSync(projectA, { recursive: true });
    mkdirSync(projectEmpty, { recursive: true });
    writeFileSync(
      path.join(projectA, 'coverage-summary.json'),
      JSON.stringify(makeSummaryData('src/a.ts')),
    );

    const result = await mergeCoverageSummaries({ coverageDir });
    assert.equal(result.mergedFiles.length, 1);
  });

  it('ignores non-project entries in the coverage directory', async () => {
    const { mergeCoverageSummaries } = await loadMerge();
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

    const result = await mergeCoverageSummaries({ coverageDir });
    assert.equal(result.mergedFiles.length, 1);
  });

  it('throws when no project coverage-summary.json files exist', async () => {
    const { mergeCoverageSummaries } = await loadMerge();
    const coverageDir = path.join(tempDir, 'coverage');
    mkdirSync(path.join(coverageDir, 'project-empty'), { recursive: true });
    await assert.rejects(
      mergeCoverageSummaries({ coverageDir }),
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
        MERGE_PATH,
        '--coverage-dir',
        coverageDir,
        '--summary-path',
        summaryPath,
      ],
      { cwd: REPO_ROOT, encoding: 'utf8' },
    );

    const parsed = JSON.parse(output);
    assert.ok(Array.isArray(parsed.mergedFiles));
    assert.ok(parsed.mergedFiles.length >= 1);
    assert.equal(path.resolve(REPO_ROOT, parsed.summaryPath), summaryPath);
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
        MERGE_PATH,
        `--coverage-dir=${coverageDir}`,
        `--summary-path=${summaryPath}`,
      ],
      { cwd: REPO_ROOT, encoding: 'utf8' },
    );

    const parsed = JSON.parse(output);
    assert.ok(parsed.mergedFiles.length >= 1);
    assert.equal(path.resolve(REPO_ROOT, parsed.summaryPath), summaryPath);
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

    assert.ok(result.mergedFiles.length >= 1);
    assert.equal(path.resolve(REPO_ROOT, result.summaryPath), summaryPath);
  });

  it('parses token forms via parseCliOptions', async () => {
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
    assert.equal(options.coverageDir, path.resolve(coverageDir));
    assert.equal(options.summaryPath, path.resolve(summaryPath));

    const result = await main([
      '--coverage-dir',
      coverageDir,
      '--summary-path',
      summaryPath,
    ]);
    assert.ok(result.mergedFiles.length >= 1);
  });

  it('parseCliOptions ignores flags without a following value', async () => {
    const { parseCliOptions } = await loadMerge();
    assert.deepEqual(parseCliOptions([]), {});
    assert.deepEqual(parseCliOptions(['--coverage-dir']), {});
    assert.deepEqual(parseCliOptions(['--summary-path']), {});
  });

  it('parseCliOptions ignores unrecognized arguments', async () => {
    const { parseCliOptions } = await loadMerge();
    assert.deepEqual(parseCliOptions(['--unknown', 'value', 'extra']), {});
    assert.deepEqual(parseCliOptions(['foo']), {});
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
      assert.equal(typeof mainResult.summaryPath, 'string');
      const directResult = await mergeCoverageSummaries();
      assert.equal(typeof directResult.summaryPath, 'string');
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
      assert.deepEqual(Array.from(logSpy.mock.calls[0]), [
        JSON.stringify({ mergedFiles: ['a'], summaryPath: 's.json' }, null, 2),
      ]);
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
      assert.deepEqual(Array.from(errorSpy.mock.calls[0]), ['boom']);
      assert.equal(process.exitCode, 1);
    } finally {
      errorSpy.mockRestore();
      process.exitCode = originalExitCode;
    }
  });

  it('picks the entry with the better line coverage when statement coverage is tied', async () => {
    const { mergeCoverageSummaries } = await loadMerge();
    const coverageDir = path.join(tempDir, 'coverage');
    const projectA = path.join(coverageDir, 'project-a');
    const projectB = path.join(coverageDir, 'project-b');
    const summaryPath = path.join(coverageDir, 'merged-summary.json');
    mkdirSync(projectA, { recursive: true });
    mkdirSync(projectB, { recursive: true });

    const fileName = 'src/tie.ts';
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

    await mergeCoverageSummaries({ coverageDir, summaryPath });
    const merged = JSON.parse(readFileSync(summaryPath, 'utf8'));
    assert.equal(merged[fileName].lines.pct, 80);
  });

  it('replaces an existing entry when the candidate has better statement coverage', async () => {
    const { mergeCoverageSummaries } = await loadMerge();
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

    await mergeCoverageSummaries({ coverageDir, summaryPath });
    const merged = JSON.parse(readFileSync(summaryPath, 'utf8'));
    assert.equal(merged[fileName].statements.pct, 83);
  });

  it('keeps the existing entry when line coverage is worse despite tied statements', async () => {
    const { mergeCoverageSummaries } = await loadMerge();
    const coverageDir = path.join(tempDir, 'coverage');
    const projectA = path.join(coverageDir, 'project-a');
    const projectB = path.join(coverageDir, 'project-b');
    const summaryPath = path.join(coverageDir, 'merged-summary.json');
    mkdirSync(projectA, { recursive: true });
    mkdirSync(projectB, { recursive: true });

    const fileName = 'src/worse-lines.ts';
    const projectAData = makeSummaryData(fileName, {
      statementsCovered: 6,
      statementsPct: 50,
      linesCovered: 8,
      linesPct: 80,
    });
    const projectBData = makeSummaryData(fileName, {
      statementsCovered: 6,
      statementsPct: 50,
      linesCovered: 5,
      linesPct: 50,
    });

    writeFileSync(
      path.join(projectA, 'coverage-summary.json'),
      JSON.stringify({ ...makeTotal(), ...projectAData }),
    );
    writeFileSync(
      path.join(projectB, 'coverage-summary.json'),
      JSON.stringify({ ...makeTotal(), ...projectBData }),
    );

    await mergeCoverageSummaries({ coverageDir, summaryPath });
    const merged = JSON.parse(readFileSync(summaryPath, 'utf8'));
    assert.equal(merged[fileName].lines.pct, 80);
  });

  it('keeps the existing entry when the candidate is worse', async () => {
    const { mergeCoverageSummaries } = await loadMerge();
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

    await mergeCoverageSummaries({ coverageDir, summaryPath });
    const merged = JSON.parse(readFileSync(summaryPath, 'utf8'));
    assert.equal(merged[fileName].statements.pct, 83);
  });

  it('handles sparse entries that are missing statements and lines', async () => {
    const { mergeCoverageSummaries } = await loadMerge();
    const coverageDir = path.join(tempDir, 'coverage');
    const projectA = path.join(coverageDir, 'project-a');
    const projectB = path.join(coverageDir, 'project-b');
    const summaryPath = path.join(coverageDir, 'merged-summary.json');
    mkdirSync(projectA, { recursive: true });
    mkdirSync(projectB, { recursive: true });

    const relativePath = 'src/sparse.ts';
    const absolutePath = path.join(REPO_ROOT, relativePath);
    const sparseA = {
      [absolutePath]: {
        functions: { total: 3, covered: 3, skipped: 0, pct: 100 },
        branches: { total: 4, covered: 4, skipped: 0, pct: 100 },
      },
    };
    const sparseB = {
      [absolutePath]: {
        lines: { pct: 100 },
      },
    };

    writeFileSync(
      path.join(projectA, 'coverage-summary.json'),
      JSON.stringify({ ...makeTotal(), ...sparseA }),
    );
    writeFileSync(
      path.join(projectB, 'coverage-summary.json'),
      JSON.stringify({ ...makeTotal(), ...sparseB }),
    );

    await mergeCoverageSummaries({ coverageDir, summaryPath });
    const merged = JSON.parse(readFileSync(summaryPath, 'utf8'));
    assert.ok(merged[relativePath]);
    assert.equal(merged.total.lines.pct, 100);
  });

  it('handles sparse entries where the candidate lacks lines', async () => {
    const { mergeCoverageSummaries } = await loadMerge();
    const coverageDir = path.join(tempDir, 'coverage');
    const projectA = path.join(coverageDir, 'project-a');
    const projectB = path.join(coverageDir, 'project-b');
    const summaryPath = path.join(coverageDir, 'merged-summary.json');
    mkdirSync(projectA, { recursive: true });
    mkdirSync(projectB, { recursive: true });

    const relativePath = 'src/sparse-candidate.ts';
    const absolutePath = path.join(REPO_ROOT, relativePath);
    const sparseA = {
      [absolutePath]: {
        lines: { pct: 100 },
      },
    };
    const sparseB = {
      [absolutePath]: {
        functions: { total: 3, covered: 3, skipped: 0, pct: 100 },
      },
    };

    writeFileSync(
      path.join(projectA, 'coverage-summary.json'),
      JSON.stringify({ ...makeTotal(), ...sparseA }),
    );
    writeFileSync(
      path.join(projectB, 'coverage-summary.json'),
      JSON.stringify({ ...makeTotal(), ...sparseB }),
    );

    await mergeCoverageSummaries({ coverageDir, summaryPath });
    const merged = JSON.parse(readFileSync(summaryPath, 'utf8'));
    assert.ok(merged[relativePath]);
  });

  it('skips missing metrics while computing the total', async () => {
    const { mergeCoverageSummaries } = await loadMerge();
    const coverageDir = path.join(tempDir, 'coverage');
    const projectA = path.join(coverageDir, 'project-a');
    mkdirSync(projectA, { recursive: true });

    const relativePath = 'src/missing.ts';
    const absolutePath = path.join(REPO_ROOT, relativePath);
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
    await mergeCoverageSummaries({ coverageDir, summaryPath });
    const merged = JSON.parse(readFileSync(summaryPath, 'utf8'));
    assert.equal(merged.total.lines.pct, 100);
    assert.equal(merged.total.statements.pct, 100);
  });

  it('skips summary files that cannot be read', async () => {
    const { mergeCoverageSummaries } = await loadMerge();
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

    const realReadFile = (await import('node:fs/promises')).readFile;
    const failingPath = path.join(projectB, 'coverage-summary.json');
    const fakeReadFile = async (filePath, options) => {
      if (
        typeof filePath === 'string' &&
        path.normalize(filePath) === path.normalize(failingPath)
      ) {
        throw new Error('permission denied');
      }
      return realReadFile(filePath, options);
    };

    const result = await mergeCoverageSummaries({
      coverageDir,
      summaryPath,
      readFile: fakeReadFile,
    });
    assert.equal(result.mergedFiles.length, 1);
    const merged = JSON.parse(readFileSync(summaryPath, 'utf8'));
    assert.ok(merged['src/a.ts']);
    assert.equal(merged['src/b.ts'], undefined);
  });

  it('generateCoverageBaseline records coverage for changed source files', async () => {
    const { generateCoverageBaseline } = await loadMerge();
    const coverageDir = path.join(tempDir, 'coverage');
    const summaryPath = path.join(coverageDir, 'baseline-input-summary.json');
    const baselinePath = path.join(coverageDir, 'baseline-output.json');
    mkdirSync(coverageDir, { recursive: true });
    const summary = {
      ...makeTotal(),
      ...makeSummaryData('src/baseline-feature.ts'),
    };
    writeFileSync(summaryPath, JSON.stringify(summary));

    const fileA = 'src/baseline-feature.ts';
    const spawnSync = () => ({
      status: 0,
      stdout: ` M ${fileA}\n`,
      stderr: '',
    });

    const result = await generateCoverageBaseline({
      coverageSummaryPath: summaryPath,
      baselinePath,
      spawnSync,
    });
    assert.equal(result.files, 1);
    assert.equal(result.zeroFiles, 0);
    const baseline = JSON.parse(readFileSync(baselinePath, 'utf8'));
    assert.equal(baseline[fileA].statements.pct, 100);
    assert.equal(baseline[fileA].lines.pct, 100);
  });

  it('generateCoverageBaseline records 0% for changed files not in the summary', async () => {
    const { generateCoverageBaseline } = await loadMerge();
    const coverageDir = path.join(tempDir, 'coverage');
    const summaryPath = path.join(coverageDir, 'baseline-input-summary.json');
    const baselinePath = path.join(coverageDir, 'baseline-output.json');
    mkdirSync(coverageDir, { recursive: true });
    writeFileSync(summaryPath, JSON.stringify(makeTotal()));

    const fileA = 'src/missing-coverage.ts';
    const spawnSync = () => ({
      status: 0,
      stdout: `?? ${fileA}\n`,
      stderr: '',
    });

    const result = await generateCoverageBaseline({
      coverageSummaryPath: summaryPath,
      baselinePath,
      spawnSync,
    });
    assert.equal(result.files, 1);
    assert.equal(result.zeroFiles, 1);
    const baseline = JSON.parse(readFileSync(baselinePath, 'utf8'));
    assert.equal(baseline[fileA].statements.pct, 0);
    assert.equal(baseline[fileA].lines.pct, 0);
  });

  it('generateCoverageBaseline writes an empty baseline when no files changed', async () => {
    const { generateCoverageBaseline } = await loadMerge();
    const coverageDir = path.join(tempDir, 'coverage');
    const summaryPath = path.join(coverageDir, 'baseline-input-summary.json');
    const baselinePath = path.join(coverageDir, 'baseline-output.json');
    mkdirSync(coverageDir, { recursive: true });
    writeFileSync(summaryPath, JSON.stringify(makeTotal()));
    const spawnSync = () => ({ status: 0, stdout: '', stderr: '' });

    const result = await generateCoverageBaseline({
      coverageSummaryPath: summaryPath,
      baselinePath,
      spawnSync,
    });
    assert.equal(result.files, 0);
    assert.equal(result.zeroFiles, 0);
    const baseline = JSON.parse(readFileSync(baselinePath, 'utf8'));
    assert.deepEqual(baseline, {});
  });

  it('generateCoverageBaseline throws when git status fails', async () => {
    const { generateCoverageBaseline } = await loadMerge();
    const spawnSync = () => ({
      status: 1,
      stdout: '',
      stderr: 'not a repo',
    });

    await assert.rejects(
      generateCoverageBaseline({ spawnSync }),
      /git status failed: not a repo/,
    );
  });

  it('generateCoverageBaseline ignores test files, __tests__ directories and non-source files', async () => {
    const { generateCoverageBaseline } = await loadMerge();
    const coverageDir = path.join(tempDir, 'coverage');
    const summaryPath = path.join(coverageDir, 'baseline-input-summary.json');
    const baselinePath = path.join(coverageDir, 'baseline-output.json');
    mkdirSync(coverageDir, { recursive: true });
    const summary = {
      ...makeTotal(),
      ...makeSummaryData('scripts/agent-customization/gates/valid.mjs'),
    };
    writeFileSync(summaryPath, JSON.stringify(summary));
    const stdout = [
      ' M src/feature.test.ts',
      '?? src/feature.spec.mjs',
      ' M scripts/agent-customization/gates/__tests__/helper.ts',
      ' M scripts/agent-customization/gates/valid.mjs',
      ' M scripts/agent-customization/gates/readme.md',
      ' M scripts/agent-customization/gates/config.json',
    ].join('\n');
    const spawnSync = () => ({ status: 0, stdout, stderr: '' });

    const result = await generateCoverageBaseline({
      coverageSummaryPath: summaryPath,
      baselinePath,
      spawnSync,
    });
    assert.equal(result.files, 1);
    const baseline = JSON.parse(readFileSync(baselinePath, 'utf8'));
    assert.ok(baseline['scripts/agent-customization/gates/valid.mjs']);
  });

  it('generateCoverageBaseline uses default paths and a mocked git status', async () => {
    const { generateCoverageBaseline } = await loadMerge();
    const defaultSummaryPath = path.join(
      REPO_ROOT,
      'coverage',
      'coverage-summary.json',
    );
    const defaultBaselinePath = path.join(
      REPO_ROOT,
      'coverage',
      'coverage-baseline.json',
    );
    mkdirSync(path.dirname(defaultSummaryPath), { recursive: true });
    const originalSummary = existsSync(defaultSummaryPath)
      ? readFileSync(defaultSummaryPath, 'utf8')
      : null;
    const originalBaseline = existsSync(defaultBaselinePath)
      ? readFileSync(defaultBaselinePath, 'utf8')
      : null;
    const summary = {
      ...makeTotal(),
      ...makeSummaryData(
        'src/architecture/network/gpu/network.gpu.activate.ts',
      ),
    };
    writeFileSync(defaultSummaryPath, JSON.stringify(summary));
    const stdout = ' M src/architecture/network/gpu/network.gpu.activate.ts';
    const spawnSync = () => ({ status: 0, stdout, stderr: '' });

    try {
      const result = await generateCoverageBaseline({ spawnSync });
      assert.equal(result.files, 1);
      assert.equal(
        path.normalize(result.baselinePath),
        path.normalize(path.join('coverage', 'coverage-baseline.json')),
      );
      assert.ok(existsSync(defaultBaselinePath));
      const baseline = JSON.parse(readFileSync(defaultBaselinePath, 'utf8'));
      assert.ok(
        baseline['src/architecture/network/gpu/network.gpu.activate.ts'],
      );
    } finally {
      if (originalSummary !== null) {
        writeFileSync(defaultSummaryPath, originalSummary);
      } else {
        rmSync(defaultSummaryPath, { force: true });
      }
      if (originalBaseline !== null) {
        writeFileSync(defaultBaselinePath, originalBaseline);
      } else {
        rmSync(defaultBaselinePath, { force: true });
      }
    }
  });

  it('generateCoverageBaseline can be invoked without arguments', async () => {
    const { generateCoverageBaseline } = await loadMerge();
    const defaultSummaryPath = path.join(
      REPO_ROOT,
      'coverage',
      'coverage-summary.json',
    );
    const defaultBaselinePath = path.join(
      REPO_ROOT,
      'coverage',
      'coverage-baseline.json',
    );
    mkdirSync(path.dirname(defaultSummaryPath), { recursive: true });
    const originalSummary = existsSync(defaultSummaryPath)
      ? readFileSync(defaultSummaryPath, 'utf8')
      : null;
    const originalBaseline = existsSync(defaultBaselinePath)
      ? readFileSync(defaultBaselinePath, 'utf8')
      : null;
    writeFileSync(defaultSummaryPath, JSON.stringify(makeTotal()));

    try {
      const result = await generateCoverageBaseline();
      assert.equal(
        path.normalize(result.baselinePath),
        path.normalize(path.join('coverage', 'coverage-baseline.json')),
      );
      assert.ok(existsSync(defaultBaselinePath));
    } finally {
      if (originalSummary !== null) {
        writeFileSync(defaultSummaryPath, originalSummary);
      } else {
        rmSync(defaultSummaryPath, { force: true });
      }
      if (originalBaseline !== null) {
        writeFileSync(defaultBaselinePath, originalBaseline);
      } else {
        rmSync(defaultBaselinePath, { force: true });
      }
    }
  });

  it('generateCoverageBaseline uses default summary and baseline paths', async () => {
    const { generateCoverageBaseline } = await loadMerge();
    let capturedBaselinePath;
    let capturedContent;
    const spawnSync = () => ({ status: 0, stdout: '', stderr: '' });
    const readFile = async (filePath) => {
      if (
        path.normalize(filePath) ===
        path.normalize(
          path.join(REPO_ROOT, 'coverage', 'coverage-summary.json'),
        )
      ) {
        return JSON.stringify(makeTotal());
      }
      throw new Error(`unexpected read ${filePath}`);
    };
    const writeFile = async (filePath, content) => {
      capturedBaselinePath = filePath;
      capturedContent = content;
    };

    const result = await generateCoverageBaseline({
      spawnSync,
      readFile,
      writeFile,
    });
    assert.equal(result.files, 0);
    assert.equal(
      path.normalize(result.baselinePath),
      path.normalize(path.join('coverage', 'coverage-baseline.json')),
    );
    assert.ok(capturedContent);
    assert.equal(
      path.normalize(capturedBaselinePath),
      path.normalize(
        path.join(REPO_ROOT, 'coverage', 'coverage-baseline.json'),
      ),
    );
  });

  it('generateCoverageBaseline uses a fallback error message when stderr is missing', async () => {
    const { generateCoverageBaseline } = await loadMerge();
    const spawnSync = () => ({ status: 1, stdout: '', stderr: undefined });
    await assert.rejects(
      generateCoverageBaseline({ spawnSync }),
      /git status failed: unknown error/,
    );
  });

  it('generateCoverageBaseline accepts an explicit --source-files list bypassing git status', async () => {
    const { generateCoverageBaseline } = await loadMerge();
    const coverageDir = path.join(tempDir, 'coverage');
    const summaryPath = path.join(coverageDir, 'baseline-input-summary.json');
    const baselinePath = path.join(coverageDir, 'baseline-output.json');
    mkdirSync(coverageDir, { recursive: true });
    const summary = {
      ...makeTotal(),
      ...makeSummaryData('src/explicit-a.ts'),
      ...makeSummaryData('src/explicit-b.ts'),
    };
    writeFileSync(summaryPath, JSON.stringify(summary));

    const result = await generateCoverageBaseline({
      coverageSummaryPath: summaryPath,
      baselinePath,
      sourceFiles:
        'src/explicit-a.ts\nsrc/explicit-b.ts,scripts/agent-customization/gates/valid.mjs',
    });
    assert.equal(result.files, 3);
    assert.equal(result.zeroFiles, 1);
    const baseline = JSON.parse(readFileSync(baselinePath, 'utf8'));
    assert.equal(baseline['src/explicit-a.ts'].statements.pct, 100);
    assert.equal(baseline['src/explicit-b.ts'].statements.pct, 100);
    assert.equal(
      baseline['scripts/agent-customization/gates/valid.mjs'].statements.pct,
      0,
    );
  });

  it('generateCoverageBaseline records zero for missing per-metric entries', async () => {
    const { generateCoverageBaseline } = await loadMerge();
    const coverageDir = path.join(tempDir, 'coverage');
    const summaryPath = path.join(coverageDir, 'baseline-input-summary.json');
    const baselinePath = path.join(coverageDir, 'baseline-output.json');
    mkdirSync(coverageDir, { recursive: true });
    const relativePath = 'src/partial.ts';
    const absolutePath = path.resolve(REPO_ROOT, relativePath);
    const summary = {
      total: { lines: { total: 1, covered: 1, skipped: 0, pct: 100 } },
      [absolutePath]: {
        lines: { total: 10, covered: 10, skipped: 0, pct: 100 },
      },
    };
    writeFileSync(summaryPath, JSON.stringify(summary));
    const spawnSync = () => ({
      status: 0,
      stdout: ` M ${relativePath}\n`,
      stderr: '',
    });

    await generateCoverageBaseline({
      coverageSummaryPath: summaryPath,
      baselinePath,
      spawnSync,
    });
    const baseline = JSON.parse(readFileSync(baselinePath, 'utf8'));
    assert.equal(baseline[relativePath].lines.pct, 100);
    assert.equal(baseline[relativePath].statements.pct, 0);
  });

  it('parseCliOptions handles --baseline in flag, token and equals forms', async () => {
    const { parseCliOptions } = await loadMerge();
    assert.equal(
      parseCliOptions(['--baseline']).baselinePath,
      path.join(REPO_ROOT, 'coverage', 'coverage-baseline.json'),
    );
    assert.equal(
      parseCliOptions(['--baseline', path.join(tempDir, 'custom.json')])
        .baselinePath,
      path.join(tempDir, 'custom.json'),
    );
    assert.equal(
      parseCliOptions([`--baseline=${path.join(tempDir, 'eq.json')}`])
        .baselinePath,
      path.join(tempDir, 'eq.json'),
    );
  });

  it('parseCliOptions handles --source-files in token and equals forms', async () => {
    const { parseCliOptions } = await loadMerge();
    assert.equal(
      parseCliOptions(['--source-files', 'src/a.ts,src/b.ts']).sourceFiles,
      'src/a.ts,src/b.ts',
    );
    assert.equal(
      parseCliOptions(['--source-files=src/c.ts']).sourceFiles,
      'src/c.ts',
    );
  });

  it('main() generates a baseline when --baseline is provided', async () => {
    const projectA = path.join(tempDir, 'coverage', 'project-main-baseline');
    mkdirSync(projectA, { recursive: true });
    writeFileSync(
      path.join(projectA, 'coverage-summary.json'),
      JSON.stringify({
        ...makeTotal(),
        ...makeSummaryData('src/main-baseline.ts'),
      }),
    );
    const coverageDir = path.join(tempDir, 'coverage');
    const summaryPath = path.join(coverageDir, 'main-baseline-summary.json');
    const baselinePath = path.join(coverageDir, 'main-baseline.json');
    const { main } = await loadMerge();
    const spawnSync = () => ({ status: 0, stdout: '', stderr: '' });

    const result = await main(
      [
        `--coverage-dir=${coverageDir}`,
        `--summary-path=${summaryPath}`,
        `--baseline=${baselinePath}`,
      ],
      { spawnSync },
    );
    assert.ok(result.mergedFiles.length >= 1);
    assert.ok(existsSync(baselinePath));
    const baseline = JSON.parse(readFileSync(baselinePath, 'utf8'));
    assert.deepEqual(baseline, {});
  });

  it('prints a CLI error and exits non-zero on failure', () => {
    const emptyDir = path.join(tempDir, 'empty-coverage');
    mkdirSync(emptyDir, { recursive: true });
    let err;
    try {
      execFileSync(process.execPath, [MERGE_PATH, '--coverage-dir', emptyDir], {
        cwd: REPO_ROOT,
        encoding: 'utf8',
      });
    } catch (caught) {
      err = caught;
    }
    assert.ok(err);
    assert.notEqual(err.status, 0);
  });
});
