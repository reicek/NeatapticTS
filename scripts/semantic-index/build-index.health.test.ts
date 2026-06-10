import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

interface BuildIndexHealthReport {
  elapsed_ms: number;
  new_documents: number;
  removed_documents: number;
  status: string;
  total_documents: number;
}

interface SpawnedBuildHealthResult {
  report: BuildIndexHealthReport | null;
  status: number | null;
  stderr: string;
  stdout: string;
}

const REPO_ROOT = path.resolve(__dirname, '..', '..');
const BUILD_INDEX_PATH = path.join(
  REPO_ROOT,
  'scripts',
  'semantic-index',
  'build-index.mjs',
);

describe('build-index.mjs', () => {
  describe('red json-health contract', () => {
    it('emits compact health summary fields when --json-health is requested', () => {
      const result = runBuildIndexJsonHealth();

      expect(result).toEqual(
        expect.objectContaining({
          status: 0,
          report: expect.objectContaining({
            status: 'ok',
            total_documents: expect.any(Number),
            new_documents: expect.any(Number),
            removed_documents: expect.any(Number),
            elapsed_ms: expect.any(Number),
          }),
        }),
      );
    });
  });
});

function runBuildIndexJsonHealth(): SpawnedBuildHealthResult {
  const spawned = spawnSync(
    process.execPath,
    [BUILD_INDEX_PATH, '--json-health', '--dry-run'],
    {
      cwd: REPO_ROOT,
      encoding: 'utf8',
    },
  );

  return {
    report: tryParseJson<BuildIndexHealthReport>(spawned.stdout ?? ''),
    status: spawned.status,
    stderr: spawned.stderr ?? '',
    stdout: spawned.stdout ?? '',
  };
}

function tryParseJson<ReportType>(stdout: string) {
  if (!stdout.trim()) return null;

  try {
    return JSON.parse(stdout) as ReportType;
  } catch {
    return null;
  }
}
