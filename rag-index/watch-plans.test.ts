/**
 * @module watch-plans.test
 * @description Red test for Phase C Step E2: optional dev-mode file watcher that
 * re-indexes changed `plans/*.plans.md` files in real time.
 *
 * This test encodes the contract that `rag-index/watch-plans.mjs` must:
 *   - use Node `fs.watch` to observe plan directories;
 *   - trigger a targeted `rag-index/embed-index.mjs --files=<changed>`
 *     re-index when a plan file changes.
 */
import { spawnSync } from 'node:child_process';
import path from 'node:path';

const REPO_ROOT = path.resolve();

interface SpawnedJsonResult<ReportType> {
  report: ReportType | null;
  status: number | null;
  stderr: string;
  stdout: string;
}

interface WatcherReport {
  triggered: boolean;
  changedFile: string | null;
  commands: string[];
}

function runModuleEvaluation<ReportType>(
  source: string,
): SpawnedJsonResult<ReportType> {
  const spawned = spawnSync(
    process.execPath,
    ['--input-type=module', '--eval', source],
    {
      cwd: REPO_ROOT,
      encoding: 'utf8',
      maxBuffer: 16 * 1024 * 1024,
    },
  );

  return {
    report: tryParseJson<ReportType>(spawned.stdout ?? ''),
    status: spawned.status,
    stderr: spawned.stderr ?? '',
    stdout: spawned.stdout ?? '',
  };
}

function tryParseJson<ReportType>(stdout: string): ReportType | null {
  if (!stdout.trim()) return null;
  try {
    return JSON.parse(stdout) as ReportType;
  } catch {
    return null;
  }
}

describe('rag-index/watch-plans.mjs', () => {
  it('uses fs.watch to trigger targeted re-index when a plan file changes', () => {
    const result = runModuleEvaluation<WatcherReport>(`
      import fs from 'node:fs';
      import os from 'node:os';
      import path from 'node:path';
      import { runPlanWatcher } from './rag-index/watch-plans.mjs';

      const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), 'watch-plans-test-'));
      const plansDir = path.join(tempDir, 'plans');
      fs.mkdirSync(plansDir, { recursive: true });
      const planFile = path.join(plansDir, 'example.plans.md');
      fs.writeFileSync(planFile, '# plan\\n');

      const commands = [];
      const watcher = await runPlanWatcher({
        planDirs: [plansDir],
        debounceMs: 25,
        onChange: async (changedPath) => {
          const repoRelative = path.relative(${JSON.stringify(REPO_ROOT)}, changedPath).replaceAll(path.sep, '/');
          commands.push('node rag-index/embed-index.mjs --files=' + repoRelative);
        },
      });

      fs.writeFileSync(planFile, '# plan updated\\n');
      await new Promise((resolve) => setTimeout(resolve, 300));
      await watcher.close();

      console.log(JSON.stringify({
        triggered: commands.length > 0,
        changedFile: planFile,
        commands,
      }));
    `);

    expect(result.report?.commands).toEqual([
      expect.stringMatching(
        /^node rag-index\/embed-index\.mjs --files=.*example\.plans\.md$/,
      ),
    ]);
  });
});
