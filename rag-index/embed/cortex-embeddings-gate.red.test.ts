import { spawnSync } from 'node:child_process';
import path from 'node:path';

interface CortexEmbeddingsGateContractReport {
  issues: string[][];
  passes: boolean[];
}

interface SpawnedJsonResult<ReportType> {
  report: ReportType | null;
  status: number | null;
  stderr: string;
  stdout: string;
}

const REPO_ROOT = path.resolve(process.cwd());

describe('cortex-embeddings.gate.mjs', () => {
  describe('red embeddings gate contract', () => {
    it('returns pass false when no usable embeddings or hybrid MRR improvement is too small', () => {
      // Arrange and Act
      const result = runModuleEvaluation<CortexEmbeddingsGateContractReport>(`
        import { evaluateCortexEmbeddingsGate } from './scripts/agent-customization/gates/cortex-embeddings.gate.mjs';

        const noEmbeddingsReport = evaluateCortexEmbeddingsGate({
          bm25MrrAt5: 0.4,
          chunkCount: 3,
          embeddingCount: 0,
          hybridMrrAt5: 0.45,
          minHybridImprovement: 0.02,
        });
        const lowImprovementReport = evaluateCortexEmbeddingsGate({
          bm25MrrAt5: 0.5,
          chunkCount: 3,
          embeddingCount: 3,
          hybridMrrAt5: 0.515,
          minHybridImprovement: 0.02,
        });

        console.log(JSON.stringify({
          issues: [
            noEmbeddingsReport.evidence.map(({ issue }) => issue),
            lowImprovementReport.evidence.map(({ issue }) => issue),
          ],
          passes: [noEmbeddingsReport.pass, lowImprovementReport.pass],
        }));
      `);

      // Assert
      expect(result).toEqual(
        expect.objectContaining({
          report: {
            issues: [
              ['no usable embeddings for model'],
              ['hybrid MRR@5 improvement below threshold'],
            ],
            passes: [false, false],
          },
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
