import { spawnSync } from 'node:child_process';
import path from 'node:path';

interface DensePrewarmStepReport {
  error?: string;
  name: string;
  status: string;
}

interface DensePrewarmJsonReport {
  error?: string;
  failedStep?: string;
  pass: boolean;
  steps?: DensePrewarmStepReport[];
}

interface DensePrewarmContractReport {
  commandRunnerCalls?: number;
  exitCode: number;
  logText?: string;
  report?: DensePrewarmJsonReport;
  stepNames?: string[];
}

interface SpawnedJsonResult<ReportType> {
  report: ReportType | null;
  status: number | null;
  stderr: string;
  stdout: string;
}

const REPO_ROOT = path.resolve(process.cwd());

describe('prewarm-dense.mjs', () => {
  describe('red prewarm bootstrap contract', () => {
    it('orchestrates download-model, embed-index, validate-embeddings, download-reranker, and validate-reranker in order', () => {
      // Arrange and Act
      const result = runModuleEvaluation<DensePrewarmContractReport>(`
        import { runDensePrewarm } from './scripts/semantic-index/prewarm-dense.mjs';

        const stepNames = [];
        const prewarmResult = await runDensePrewarm({
          commandRunner: (step) => {
            stepNames.push(step.name);
            return { status: 0, stderr: '', stdout: JSON.stringify({ ok: true }) };
          },
          json: true,
          logger: () => {},
          modelExists: () => false,
          rerankerModelExists: () => false,
        });

        console.log(JSON.stringify({
          exitCode: prewarmResult.exitCode,
          report: prewarmResult.report,
          stepNames,
        }));
      `);

      // Assert
      expect(result).toEqual(
        expect.objectContaining({
          report: expect.objectContaining({
            exitCode: 0,
            stepNames: [
              'download-model',
              'embed-index',
              'validate-embeddings',
              'download-reranker',
              'validate-reranker',
            ],
          }),
          status: 0,
        }),
      );
    });

    it('logs dry-run steps in order without invoking subprocesses', () => {
      // Arrange and Act
      const result = runModuleEvaluation<DensePrewarmContractReport>(`
        import { runDensePrewarm } from './scripts/semantic-index/prewarm-dense.mjs';

        const commandRunnerCalls = [];
        const logLines = [];
        const prewarmResult = await runDensePrewarm({
          commandRunner: (step) => {
            commandRunnerCalls.push(step.name);
            return { status: 0, stderr: '', stdout: '' };
          },
          dryRun: true,
          logger: (line) => logLines.push(String(line)),
          modelExists: () => false,
        });

        console.log(JSON.stringify({
          commandRunnerCalls: commandRunnerCalls.length,
          exitCode: prewarmResult.exitCode,
          logText: logLines.join('\\n'),
        }));
      `);

      // Assert
      expect(result).toEqual(
        expect.objectContaining({
          report: expect.objectContaining({
            commandRunnerCalls: 0,
            exitCode: 0,
            logText: expect.stringMatching(
              /download-model[\s\S]*embed-index[\s\S]*validate-embeddings[\s\S]*download-reranker[\s\S]*validate-reranker/u,
            ),
          }),
          status: 0,
        }),
      );
    });

    it('emits passing JSON with per-step statuses on success', () => {
      // Arrange and Act
      const result = runModuleEvaluation<DensePrewarmContractReport>(`
        import { runDensePrewarm } from './scripts/semantic-index/prewarm-dense.mjs';

        const prewarmResult = await runDensePrewarm({
          commandRunner: () => ({ status: 0, stderr: '', stdout: JSON.stringify({ ok: true }) }),
          json: true,
          logger: () => {},
          modelExists: () => false,
        });

        console.log(JSON.stringify({
          exitCode: prewarmResult.exitCode,
          report: prewarmResult.report,
        }));
      `);

      // Assert
      expect(result).toEqual(
        expect.objectContaining({
          report: expect.objectContaining({
            exitCode: 0,
            report: {
              pass: true,
              steps: [
                { name: 'download-model', status: 'ok' },
                { name: 'embed-index', status: 'ok' },
                { name: 'validate-embeddings', status: 'ok' },
                { name: 'download-reranker', status: 'ok' },
                { name: 'validate-reranker', status: 'ok' },
              ],
            },
          }),
          status: 0,
        }),
      );
    });

    it('emits failing JSON and stops when embed-index exits non-zero', () => {
      // Arrange and Act
      const result = runModuleEvaluation<DensePrewarmContractReport>(`
        import { runDensePrewarm } from './scripts/semantic-index/prewarm-dense.mjs';

        const stepNames = [];
        const prewarmResult = await runDensePrewarm({
          commandRunner: (step) => {
            stepNames.push(step.name);
            return step.name === 'embed-index'
              ? { status: 1, stderr: 'fixture embed-index failure', stdout: '' }
              : { status: 0, stderr: '', stdout: JSON.stringify({ ok: true }) };
          },
          json: true,
          logger: () => {},
          modelExists: () => false,
          rerankerModelExists: () => false,
        });

        console.log(JSON.stringify({
          exitCode: prewarmResult.exitCode,
          report: prewarmResult.report,
          stepNames,
        }));
      `);

      // Assert
      expect(result).toEqual(
        expect.objectContaining({
          report: expect.objectContaining({
            exitCode: 1,
            report: {
              error: expect.stringMatching(/\S/u),
              failedStep: 'embed-index',
              pass: false,
            },
            stepNames: ['download-model', 'embed-index'],
          }),
          status: 0,
        }),
      );
    });
  });
});

function runModuleEvaluation<ReportType>(
  source: string,
): SpawnedJsonResult<ReportType> {
  const spawned = spawnSync(
    process.execPath,
    ['--input-type=module', '--eval', source],
    {
      cwd: REPO_ROOT,
      encoding: 'utf8',
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
