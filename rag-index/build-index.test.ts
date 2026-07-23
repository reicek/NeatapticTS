import { spawnSync } from 'node:child_process';
import path from 'node:path';

const REPO_ROOT = path.resolve(process.cwd());

interface BuildIndexSummary {
  scanned: number;
  indexed: number;
  skipped: number;
  chunks: number;
  dryRun: boolean;
  totalDocuments: number;
}

interface SpawnedBuildResult {
  stderr: string;
  status: number | null;
  stdout: string;
  summary: BuildIndexSummary | null;
}

const BUILD_INDEX_PATH = path.join(REPO_ROOT, 'rag-index', 'build-index.mjs');

function runBuildIndex(args: string[]): SpawnedBuildResult {
  const spawned = spawnSync(process.execPath, [BUILD_INDEX_PATH, ...args], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
  });

  let summary: BuildIndexSummary | null = null;
  const stdout = spawned.stdout ?? '';
  if (stdout.trim()) {
    try {
      summary = JSON.parse(stdout) as BuildIndexSummary;
    } catch {
      summary = null;
    }
  }

  return {
    summary,
    status: spawned.status,
    stderr: spawned.stderr ?? '',
    stdout,
  };
}

let targetedResult: SpawnedBuildResult;

beforeAll(() => {
  targetedResult = runBuildIndex(['--dry-run', '--json', '--files=README.md']);
});

describe('build-index.mjs --files filtering', () => {
  it('exits successfully when given a targeted dry-run invocation', () => {
    expect(targetedResult.status).toBe(0);
  });

  it('limits the scanned document count to the single targeted path', () => {
    expect(targetedResult.summary?.scanned).toBe(1);
  });

  it('still reports the full corpus size separately from the targeted scan', () => {
    expect(targetedResult.summary?.totalDocuments).toBeGreaterThan(1);
  });
});
