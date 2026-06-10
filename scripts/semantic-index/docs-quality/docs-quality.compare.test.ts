import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import path from 'node:path';

interface CompareResult {
  accepted: boolean;
  delta?: Record<string, number>;
  mismatches?: string[];
}

interface SpawnedJsonResult<ReportType> {
  report: ReportType | null;
  status: number | null;
  stderr: string;
  stdout: string;
}

const REPO_ROOT = path.resolve(process.cwd());
const FIXTURES_ROOT = path.join(
  REPO_ROOT,
  'scripts',
  'semantic-index',
  'docs-quality',
  '__fixtures__',
);

describe('docs-quality compare red contracts', () => {
  it('rejects metric dimension mismatches with machine-readable reason codes', () => {
    const baseManifest = JSON.parse(
      readFileSync(path.join(FIXTURES_ROOT, 'manifest.v1.base.json'), 'utf8'),
    ) as Record<string, unknown>;

    const result = runModuleEvaluation<{ reasons: string[] }>(`
      import { compareDocsQualityRuns } from './scripts/semantic-index/docs-quality/docs-quality.compare.mjs';
      const base = ${JSON.stringify(baseManifest)};
      const mismatchedCandidates = [
        { ...base, metricVersion: 2 },
        { ...base, threshold: { minJsdocWords: 11, complexityThreshold: 10 } },
        { ...base, scopeType: 'src' },
        { ...base, scopeDigest: 'ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff' },
        { ...base, scannerVersion: '2.0.0' },
      ];
      const reasons = mismatchedCandidates.map((candidate) => {
        const comparison = compareDocsQualityRuns({ leftManifest: base, rightManifest: candidate, leftSummary: {}, rightSummary: {} });
        return comparison.reasonCode;
      });
      console.log(JSON.stringify({ reasons }));
    `);

    expect(result).toEqual(
      expect.objectContaining({
        report: {
          reasons: [
            'METRIC_VERSION_MISMATCH',
            'THRESHOLD_MISMATCH',
            'SCOPE_TYPE_MISMATCH',
            'SCOPE_DIGEST_MISMATCH',
            'SCANNER_VERSION_MISMATCH',
          ],
        },
        status: 0,
      }),
    );
  });

  it('emits metric deltas only when all comparison dimensions match', () => {
    const leftManifest = JSON.parse(
      readFileSync(path.join(FIXTURES_ROOT, 'manifest.v1.base.json'), 'utf8'),
    ) as Record<string, unknown>;
    const rightManifest = JSON.parse(
      readFileSync(
        path.join(FIXTURES_ROOT, 'manifest.v1.matching-right.json'),
        'utf8',
      ),
    ) as Record<string, unknown>;

    const result = runModuleEvaluation<CompareResult>(`
      import { compareDocsQualityRuns } from './scripts/semantic-index/docs-quality/docs-quality.compare.mjs';
      const output = compareDocsQualityRuns({
        leftManifest: ${JSON.stringify(leftManifest)},
        rightManifest: ${JSON.stringify(rightManifest)},
        leftSummary: {
          missingJsdoc: 20,
          weakJsdoc: 12,
          highComplexity: 8,
          evidenceCount: 40,
        },
        rightSummary: {
          missingJsdoc: 18,
          weakJsdoc: 10,
          highComplexity: 9,
          evidenceCount: 37,
        },
      });
      console.log(JSON.stringify(output));
    `);

    expect(result).toEqual(
      expect.objectContaining({
        report: {
          accepted: true,
          delta: {
            missingJsdoc: -2,
            weakJsdoc: -2,
            highComplexity: 1,
            evidenceCount: -3,
          },
        },
        status: 0,
      }),
    );
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
