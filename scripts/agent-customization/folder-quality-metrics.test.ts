import { spawnSync } from 'node:child_process';
import { access, mkdtemp, rename, rm, writeFile } from 'node:fs/promises';
import { constants as fsConstants } from 'node:fs';
import path from 'node:path';

interface FolderQualityMetricsReport {
  evidence: string[];
  folderChecked: string;
  pass: boolean;
  smells: Array<{
    detail: string;
    file: string;
    kind: string;
  }>;
}

interface SpawnedFolderQualityMetricsResult {
  report: FolderQualityMetricsReport | null;
  status: number | null;
  stderr: string;
  stdout: string;
}

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const FOLDER_QUALITY_METRICS_PATH = path.join(
  REPO_ROOT,
  'scripts',
  'folder-quality-metrics.mjs',
);
const COVERAGE_LCOV_PATH = path.join(REPO_ROOT, 'coverage', 'lcov.info');

describe('folder-quality-metrics.mjs', () => {
  describe('red CLI contract', () => {
    it('accepts a target folder and emits JSON with a missing-test-file smell', async () => {
      const fixture = await createFolderQualityFixture();

      try {
        const result = runFolderQualityMetrics([
          `--folder=${fixture.relativeFolderPath}`,
          '--json',
        ]);

        expect(result).toEqual(
          expect.objectContaining({
            report: expect.objectContaining({
              folderChecked: fixture.relativeFolderPath,
              pass: false,
              smells: expect.arrayContaining([
                expect.objectContaining({
                  file: expect.stringContaining('probe.ts'),
                  kind: 'missing-test-file',
                }),
              ]),
            }),
            status: 1,
          }),
        );
      } finally {
        await rm(fixture.fixtureDirectory, { force: true, recursive: true });
      }
    });

    it('skips optional lcov lookup gracefully when coverage/lcov.info is absent', async () => {
      const fixture = await createFolderQualityFixture();

      try {
        const result = await withMissingCoverageLcov(() =>
          runFolderQualityMetrics([
            `--folder=${fixture.relativeFolderPath}`,
            '--json',
          ]),
        );

        expect(result).toEqual(
          expect.objectContaining({
            report: expect.objectContaining({
              evidence: expect.any(Array),
              folderChecked: fixture.relativeFolderPath,
              pass: false,
              smells: expect.arrayContaining([
                expect.objectContaining({
                  kind: 'missing-test-file',
                }),
              ]),
            }),
            status: 1,
          }),
        );
      } finally {
        await rm(fixture.fixtureDirectory, { force: true, recursive: true });
      }
    });
  });
});

async function createFolderQualityFixture(): Promise<{
  fixtureDirectory: string;
  relativeFolderPath: string;
}> {
  const fixtureDirectory = await mkdtemp(
    path.join(REPO_ROOT, 'testing', 'folder-quality-fixture-'),
  );
  const probeFilePath = path.join(fixtureDirectory, 'probe.ts');

  await writeFile(
    probeFilePath,
    [
      '/**',
      ' * Folder-quality fixture used by the red contract.',
      ' */',
      'export function probeFolderQuality(): number {',
      '  return 1;',
      '}',
      '',
    ].join('\n'),
    'utf8',
  );

  return {
    fixtureDirectory,
    relativeFolderPath: path.relative(REPO_ROOT, fixtureDirectory).split(path.sep).join('/'),
  };
}

function runFolderQualityMetrics(
  args: string[],
): SpawnedFolderQualityMetricsResult {
  const spawnedResult = spawnSync(process.execPath, [
    FOLDER_QUALITY_METRICS_PATH,
    ...args,
  ], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
    timeout: 600_000,
  });

  return {
    report: tryParseJson<FolderQualityMetricsReport>(spawnedResult.stdout ?? ''),
    status: spawnedResult.status,
    stderr: spawnedResult.stderr ?? '',
    stdout: spawnedResult.stdout ?? '',
  };
}

async function withMissingCoverageLcov<ResultType>(
  callback: () => ResultType,
): Promise<ResultType> {
  const coverageLcovExists = await pathExists(COVERAGE_LCOV_PATH);
  const coverageLcovBackupPath = path.join(
    REPO_ROOT,
    'coverage',
    `lcov.info.folder-quality-backup-${process.pid}`,
  );

  if (coverageLcovExists) {
    await rename(COVERAGE_LCOV_PATH, coverageLcovBackupPath);
  }

  try {
    return callback();
  } finally {
    if (coverageLcovExists) {
      await rename(coverageLcovBackupPath, COVERAGE_LCOV_PATH);
    }
  }
}

async function pathExists(targetPath: string): Promise<boolean> {
  try {
    await access(targetPath, fsConstants.F_OK);
    return true;
  } catch {
    return false;
  }
}

function tryParseJson<ResultType>(stdout: string): ResultType | null {
  if (!stdout.trim()) {
    return null;
  }

  try {
    return JSON.parse(stdout) as ResultType;
  } catch {
    return null;
  }
}