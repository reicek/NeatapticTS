import { readFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import path from 'node:path';

interface MetricsContractReport {
  canonicalEvidence: Array<{ file: string; issue: string; numericValue: number; symbol: string }>;
  digestA: string;
  digestB: string;
  firstKey?: string;
  isSchemaValid: boolean;
  lastKey?: string;
  missingFields: string[];
}

interface SpawnedJsonResult<ReportType> {
  report: ReportType | null;
  status: number | null;
  stderr: string;
  stdout: string;
}

const REPO_ROOT = path.resolve(process.cwd());
const FIXTURES_ROOT = path.join(REPO_ROOT, 'scripts', 'semantic-index', 'docs-quality', '__fixtures__');
const DOCS_QUALITY_METRICS_PATH = path.join(REPO_ROOT, 'scripts', 'semantic-index', 'docs-quality', 'docs-quality.metrics.mjs');

describe('docs-quality metrics red contracts', () => {
  it('requires the v1 manifest schema fields and rejects missing scannerVersion', () => {
    const manifest = JSON.parse(readFileSync(path.join(FIXTURES_ROOT, 'manifest.v1.base.json'), 'utf8')) as Record<string, unknown>;
    delete manifest.scannerVersion;

    const result = runModuleEvaluation<MetricsContractReport>(`
      import { validateDocsQualityManifestV1 } from './scripts/semantic-index/docs-quality/docs-quality.contract.mjs';
      const validation = validateDocsQualityManifestV1(${JSON.stringify(manifest)});
      console.log(JSON.stringify({
        canonicalEvidence: [],
        digestA: '',
        digestB: '',
        isSchemaValid: validation.valid,
        missingFields: validation.errors.map(({ field }) => field),
      }));
    `);

    expect(result).toEqual(expect.objectContaining({
      report: expect.objectContaining({
        isSchemaValid: false,
        missingFields: expect.arrayContaining(['scannerVersion']),
      }),
      status: 0,
    }));
  });

  it('normalizes ordering and dedupe deterministically and keeps scope digest stable across path order variants', () => {
    const evidence = JSON.parse(readFileSync(path.join(FIXTURES_ROOT, 'metrics.v1.evidence.unsorted.json'), 'utf8')) as Array<Record<string, unknown>>;

    const result = runModuleEvaluation<MetricsContractReport>(`
      import { normalizeDocsQualityEvidence, normalizeScopeInputAndDigest } from './scripts/semantic-index/docs-quality/docs-quality.normalize.mjs';
      const canonicalEvidence = normalizeDocsQualityEvidence(${JSON.stringify(evidence)});
      const scopeA = normalizeScopeInputAndDigest({ scopeType: 'paths', scopeValue: ['src/neat.ts', 'src/architecture/network.ts', 'src/neat.ts'] });
      const scopeB = normalizeScopeInputAndDigest({ scopeType: 'paths', scopeValue: ['src/architecture/network.ts', 'src/neat.ts'] });
      console.log(JSON.stringify({
        canonicalEvidence,
        digestA: scopeA.scopeDigest,
        digestB: scopeB.scopeDigest,
        isSchemaValid: true,
        missingFields: [],
      }));
    `);

    expect(result).toEqual(expect.objectContaining({
      report: {
        canonicalEvidence: [
          {
            file: 'src/architecture/network.ts',
            issue: 'high complexity',
            numericValue: 14,
            symbol: 'buildDenseLayer',
          },
          {
            file: 'src/neat.ts',
            issue: 'missing JSDoc',
            numericValue: 0,
            symbol: 'undocumentedMutation',
          },
          {
            file: 'src/architecture/network.ts',
            issue: 'weak JSDoc',
            numericValue: 5,
            symbol: 'allocateGenome',
          },
        ],
        digestA: 'd36d8aef57058f860ee0e6ff49fd2d34413806b0bafcbf4f818a4378515f7f53',
        digestB: 'd36d8aef57058f860ee0e6ff49fd2d34413806b0bafcbf4f818a4378515f7f53',
        isSchemaValid: true,
        missingFields: [],
      },
      status: 0,
    }));
  });

  it('serializes CLI JSON with summary first and evidence last', () => {
    const result = runMetricsCliEvaluation([
      '--json',
      '--scope',
      'paths',
      '--source',
      'src/neat.ts',
      '--source',
      'src/architecture/network.ts',
      '--run-id',
      'red-contract-cli-order',
    ]);

    expect(result).toEqual(expect.objectContaining({
      report: expect.objectContaining({
        firstKey: 'summary',
        lastKey: 'evidence',
      }),
      status: 0,
    }));
  });

  it('sorts canonical evidence hottest-first with descending numeric values inside each severity tier', () => {
    const evidence = [
      { file: 'src/zeta.ts', issue: 'weak JSDoc', numericValue: 8, symbol: 'zetaWeak' },
      { file: 'src/alpha.ts', issue: 'high complexity', numericValue: 12, symbol: 'alphaHot' },
      { file: 'src/beta.ts', issue: 'missing JSDoc', numericValue: 0, symbol: 'betaMissing' },
      { file: 'src/gamma.ts', issue: 'high complexity', numericValue: 20, symbol: 'gammaHot' },
      { file: 'src/epsilon.ts', issue: 'high complexity', numericValue: 20, symbol: 'epsilonHot' },
      { file: 'src/delta.ts', issue: 'weak JSDoc', numericValue: 3, symbol: 'deltaWeak' },
      { file: 'src/alpha.ts', issue: 'missing JSDoc', numericValue: 0, symbol: 'alphaMissing' },
    ];

    const result = runModuleEvaluation<MetricsContractReport>(`
      import { normalizeDocsQualityEvidence } from './scripts/semantic-index/docs-quality/docs-quality.normalize.mjs';
      console.log(JSON.stringify({
        canonicalEvidence: normalizeDocsQualityEvidence(${JSON.stringify(evidence)}),
        digestA: '',
        digestB: '',
        isSchemaValid: true,
        missingFields: [],
      }));
    `);

    expect(result).toEqual(expect.objectContaining({
      report: expect.objectContaining({
        canonicalEvidence: [
          { file: 'src/epsilon.ts', issue: 'high complexity', numericValue: 20, symbol: 'epsilonHot' },
          { file: 'src/gamma.ts', issue: 'high complexity', numericValue: 20, symbol: 'gammaHot' },
          { file: 'src/alpha.ts', issue: 'high complexity', numericValue: 12, symbol: 'alphaHot' },
          { file: 'src/alpha.ts', issue: 'missing JSDoc', numericValue: 0, symbol: 'alphaMissing' },
          { file: 'src/beta.ts', issue: 'missing JSDoc', numericValue: 0, symbol: 'betaMissing' },
          { file: 'src/zeta.ts', issue: 'weak JSDoc', numericValue: 8, symbol: 'zetaWeak' },
          { file: 'src/delta.ts', issue: 'weak JSDoc', numericValue: 3, symbol: 'deltaWeak' },
        ],
      }),
      status: 0,
    }));
  });
});

function runMetricsCliEvaluation(argumentsVector: string[]): SpawnedJsonResult<MetricsContractReport> {
  const spawned = spawnSync(process.execPath, [DOCS_QUALITY_METRICS_PATH, ...argumentsVector], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
  });

  const parsedReport = tryParseJson<Record<string, unknown>>(spawned.stdout ?? '');
  const rootKeys = parsedReport ? Object.keys(parsedReport) : [];

  return {
    report: parsedReport
      ? ({
        ...parsedReport,
        firstKey: rootKeys.at(0),
        lastKey: rootKeys.at(-1),
      } as MetricsContractReport)
      : null,
    status: spawned.status,
    stderr: spawned.stderr ?? '',
    stdout: spawned.stdout ?? '',
  };
}

function runModuleEvaluation<ReportType>(source: string): SpawnedJsonResult<ReportType> {
  const spawned = spawnSync(process.execPath, ['--input-type=module', '--eval', source], {
    cwd: REPO_ROOT,
    encoding: 'utf8',
  });

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
