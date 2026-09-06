/**
 * @module auto-reindex.test
 * @description Red tests for Phase C Step E2: post-commit hook orchestration,
 * optional plan-file watcher, and cortex-index gate fixHint contracts.
 *
 * These tests encode the contract that the E2-impl slice must:
 *   - detect changed `plans/*.plans.md` files and invoke targeted
 *     `rag-index/build-index.mjs --files=<changed>` then
 *     `rag-index/embed-index.mjs --files=<changed>`;
 *   - degrade gracefully when a re-index command fails (log the failure,
 *     exit 0 for the post-commit wrapper);
 *   - write a freshness proof to `rag-index/freshness-proofs/plans-reindex.log`;
 *   - expose a `resolveStalePlanFixHint` helper that produces a targeted
 *     `--files=` re-index fixHint for the cortex-index gate.
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

interface ReindexSummary {
  changed: string[];
  commands: string[][];
  exitCode: number;
  failures: string[][];
  logPath: string;
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

describe('rag-index/auto-reindex.mjs', () => {
  describe('post-commit hook orchestration', () => {
    it('detects changed plan files and invokes targeted build-index then embed-index', () => {
      const result = runModuleEvaluation<ReindexSummary>(`
        import { reindexChangedFiles } from './rag-index/auto-reindex.mjs';
        const summary = await reindexChangedFiles({
          changedFiles: ['plans/example.plans.md'],
        });
        console.log(JSON.stringify(summary));
      `);

      expect(result.report?.commands).toEqual([
        ['node', 'rag-index/build-index.mjs', '--files=plans/example.plans.md'],
        ['node', 'rag-index/embed-index.mjs', '--files=plans/example.plans.md'],
      ]);
    });

    it('exits 0 and records failures when a re-index command fails', () => {
      const result = runModuleEvaluation<ReindexSummary>(`
        import { reindexChangedFiles } from './rag-index/auto-reindex.mjs';
        const summary = await reindexChangedFiles({
          changedFiles: ['plans/example.plans.md'],
          runCommand: async () => ({ success: false }),
        });
        console.log(JSON.stringify(summary));
      `);

      expect(result.report).toEqual(
        expect.objectContaining({
          exitCode: 0,
          failures: expect.arrayContaining([
            expect.arrayContaining([
              expect.stringMatching(/build-index|embed-index/),
            ]),
          ]),
        }),
      );
    });

    it('logs command failures to stderr', () => {
      const result = runModuleEvaluation<ReindexSummary>(`
        import { reindexChangedFiles } from './rag-index/auto-reindex.mjs';
        const summary = await reindexChangedFiles({
          changedFiles: ['plans/example.plans.md'],
          runCommand: async () => ({ success: false }),
        });
        console.log(JSON.stringify(summary));
      `);

      expect(result.stderr).toMatch(/auto-reindex: command failed:/);
    });

    it('writes re-index results to rag-index/freshness-proofs/auto-reindex.log', () => {
      const result = runModuleEvaluation<ReindexSummary>(`
        import { reindexChangedFiles } from './rag-index/auto-reindex.mjs';
        const summary = await reindexChangedFiles({
          changedFiles: ['plans/example.plans.md'],
        });
        console.log(JSON.stringify(summary));
      `);

      expect(result.report?.logPath).toMatch(
        /rag-index\/freshness-proofs\/auto-reindex\.log$/,
      );
    });
  });

  describe('cortex-index gate fixHint contract', () => {
    it('mentions targeted re-index for stale plan files', () => {
      const result = runModuleEvaluation<{ fixHint: string }>(`
        import { resolveStalePlanFixHint } from './rag-index/auto-reindex.mjs';
        const fixHint = resolveStalePlanFixHint(['plans/example.plans.md']);
        console.log(JSON.stringify({ fixHint }));
      `);

      expect(result.report?.fixHint).toMatch(
        /node\s+rag-index\/(build-index|embed-index)\.mjs\s+--files=/,
      );
    });
  });
});
