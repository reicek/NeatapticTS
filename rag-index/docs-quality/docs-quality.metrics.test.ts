import {
  existsSync,
  mkdirSync,
  readFileSync,
  renameSync,
  rmSync,
  writeFileSync,
} from 'node:fs';
import { spawnSync } from 'node:child_process';
import path from 'node:path';

interface MetricsContractReport {
  canonicalEvidence: Array<{
    file: string;
    issue: string;
    numericValue: number;
    symbol: string;
  }>;
  digestA: string;
  digestB: string;
  firstKey?: string;
  isSchemaValid: boolean;
  lastKey?: string;
  manifest?: Record<string, unknown>;
  missingFields: string[];
  pass?: boolean;
  summary?: {
    coverage?: {
      available: boolean;
      filesBelow100: number;
      filesBelow100Detail: Array<{
        branches: number;
        file: string;
        functions: number;
        lines: number;
        statementCoverageSource: string;
        statements: number;
        uncoveredBranches: Array<{
          branch: string;
          block: string;
          line: number;
          taken: number | null;
        }>;
        uncoveredFunctions: string[];
        uncoveredLines: number[];
      }>;
      overallBranches: number;
      overallFunctions: number;
      overallLines: number;
      totalFiles: number;
    };
  };
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
  'rag-index',
  'docs-quality',
  '__fixtures__',
);
const DOCS_QUALITY_METRICS_PATH = path.join(
  REPO_ROOT,
  'rag-index',
  'docs-quality',
  'docs-quality.metrics.mjs',
);
const COVERAGE_DIRECTORY_PATH = path.join(REPO_ROOT, 'coverage');
const COVERAGE_FIXTURE_LCOV = [
  'TN:',
  'SF:src/neat.ts',
  'FN:1,activate',
  'FNDA:1,activate',
  'FNF:1',
  'FNH:1',
  'DA:1,1',
  'LF:1',
  'LH:1',
  'BRDA:1,0,0,1',
  'BRF:1',
  'BRH:1',
  'end_of_record',
  '',
].join('\n');
const COVERAGE_FIXTURE_SUMMARY = {
  total: {
    statements: {
      pct: 100,
    },
  },
  'src/neat.ts': {
    statements: {
      pct: 100,
    },
  },
};
const PARTIAL_COVERAGE_FIXTURE_SUMMARY = {
  total: {
    statements: {
      pct: 21.5,
    },
  },
  'src/neat.ts': {
    statements: {
      pct: 100,
    },
  },
};
const MIXED_SCOPE_FIXTURE_PATH = path.join(
  FIXTURES_ROOT,
  'lcov-mixed-scope.json',
);
const MIXED_SCOPE_FIXTURE = JSON.parse(
  readFileSync(MIXED_SCOPE_FIXTURE_PATH, 'utf8'),
) as {
  lcov: string;
  summary: Record<string, unknown>;
};

interface CoverageDirectorySwapState {
  backupDirectoryPath: string | null;
}

describe('docs-quality metrics red contracts', () => {
  it('requires the cleaned manifest schema fields and rejects missing scannerVersion', () => {
    const manifest = JSON.parse(
      readFileSync(path.join(FIXTURES_ROOT, 'manifest.v1.base.json'), 'utf8'),
    ) as Record<string, unknown>;
    delete manifest.scannerVersion;

    const result = runModuleEvaluation<MetricsContractReport>(`
      import { validateDocsQualityManifestV1 } from './rag-index/docs-quality/docs-quality.contract.mjs';
      const validation = validateDocsQualityManifestV1(${JSON.stringify(manifest)});
      console.log(JSON.stringify({
        canonicalEvidence: [],
        digestA: '',
        digestB: '',
        isSchemaValid: validation.valid,
        missingFields: validation.errors.map(({ field }) => field),
      }));
    `);

    expect(result).toEqual(
      expect.objectContaining({
        report: expect.objectContaining({
          isSchemaValid: false,
          missingFields: expect.arrayContaining(['scannerVersion']),
        }),
        status: 0,
      }),
    );
  });

  it('normalizes ordering and dedupe deterministically and keeps scope digest stable across path order variants', () => {
    const evidence = JSON.parse(
      readFileSync(
        path.join(FIXTURES_ROOT, 'metrics.v1.evidence.unsorted.json'),
        'utf8',
      ),
    ) as Array<Record<string, unknown>>;

    const result = runModuleEvaluation<MetricsContractReport>(`
      import { normalizeDocsQualityEvidence, normalizeScopeInputAndDigest } from './rag-index/docs-quality/docs-quality.normalize.mjs';
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

    expect(result).toEqual(
      expect.objectContaining({
        report: {
          canonicalEvidence: [
            {
              file: 'src/architecture/network.ts',
              issue: 'high complexity',
              numericValue: 14,
              symbol: 'buildDenseLayer',
            },
            {
              file: 'src/architecture/network.ts',
              issue: 'high complexity',
              numericValue: 9,
              symbol: 'es2023Heavy',
            },
            {
              file: 'src/neat.ts',
              issue: 'missing JSDoc',
              numericValue: 0,
              symbol: 'undocumentedMutation',
            },
            {
              file: 'src/neat.ts',
              issue: 'incomplete JSDoc tags',
              numericValue: 0,
              symbol: 'incompleteTags',
            },
            {
              file: 'src/architecture/network.ts',
              issue: 'weak JSDoc',
              numericValue: 5,
              symbol: 'allocateGenome',
            },
          ],
          digestA:
            'd36d8aef57058f860ee0e6ff49fd2d34413806b0bafcbf4f818a4378515f7f53',
          digestB:
            'd36d8aef57058f860ee0e6ff49fd2d34413806b0bafcbf4f818a4378515f7f53',
          isSchemaValid: true,
          missingFields: [],
        },
        status: 0,
      }),
    );
  });

  it('collapses re-exported symbol duplicates across files into one canonical evidence row', () => {
    const evidence = JSON.parse(
      readFileSync(
        path.join(FIXTURES_ROOT, 're-export-duplicate.evidence.json'),
        'utf8',
      ),
    ) as Array<Record<string, unknown>>;

    const result = runModuleEvaluation<MetricsContractReport>(`
      import { normalizeDocsQualityEvidence } from './rag-index/docs-quality/docs-quality.normalize.mjs';
      console.log(JSON.stringify({
        canonicalEvidence: normalizeDocsQualityEvidence(${JSON.stringify(evidence)}),
        digestA: '',
        digestB: '',
        isSchemaValid: true,
        missingFields: [],
      }));
    `);

    expect(result).toEqual(
      expect.objectContaining({
        report: expect.objectContaining({
          canonicalEvidence: [
            {
              file: 'src/architecture/re-exports.ts',
              issue: 'missing JSDoc',
              numericValue: 0,
              symbol: 'sharedUtility',
            },
          ],
        }),
        status: 0,
      }),
    );
  });

  it('serializes CLI JSON with summary first and evidence last', () => {
    const result = runMetricsCliEvaluationWithCoverageFixture([
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

    expect(result).toEqual(
      expect.objectContaining({
        report: expect.objectContaining({
          firstKey: 'summary',
          lastKey: 'evidence',
          summary: expect.objectContaining({
            coverage: expect.objectContaining({
              available: true,
              totalFiles: expect.any(Number),
              filesBelow100: expect.any(Number),
              filesBelow100Detail: expect.any(Array),
              overallLines: expect.any(Number),
              overallBranches: expect.any(Number),
              overallFunctions: expect.any(Number),
            }),
          }),
        }),
        status: 0,
      }),
    );
  });

  it('keeps the CLI coverage contract deterministic when live repo coverage is absent at startup', () => {
    const result = withCoverageDirectoryTemporarilyMissing(() =>
      runMetricsCliEvaluationWithCoverageFixture([
        '--json',
        '--scope',
        'paths',
        '--source',
        'src/neat.ts',
        '--source',
        'src/architecture/network.ts',
        '--run-id',
        'red-contract-cli-order-coverage-absent',
      ]),
    );

    expect(result).toEqual(
      expect.objectContaining({
        report: expect.objectContaining({
          firstKey: 'summary',
          lastKey: 'evidence',
          summary: expect.objectContaining({
            coverage: expect.objectContaining({
              available: true,
              totalFiles: expect.any(Number),
              filesBelow100: expect.any(Number),
              filesBelow100Detail: expect.any(Array),
              overallLines: expect.any(Number),
              overallBranches: expect.any(Number),
              overallFunctions: expect.any(Number),
            }),
          }),
        }),
        status: 0,
      }),
    );
  });

  it('surfaces partial coverage artifacts as partial coverage instead of a normal repo-wide metric state', () => {
    const result = runMetricsCliEvaluationWithPartialCoverageFixture([
      '--json',
      '--scope',
      'paths',
      '--source',
      'src/neat.ts',
      '--source',
      'src/architecture/network.ts',
      '--run-id',
      'red-contract-cli-partial-coverage',
    ]);

    expect(result).toEqual(
      expect.objectContaining({
        report: expect.objectContaining({
          summary: expect.objectContaining({
            coverage: expect.objectContaining({
              available: true,
              isPartial: true,
            }),
          }),
          manifest: expect.objectContaining({
            coverage: expect.objectContaining({
              available: true,
              isPartial: true,
            }),
          }),
        }),
        status: 0,
      }),
    );
  });

  it('filters coverage records to the resolved scan scope', () => {
    const result = runMetricsCliEvaluationWithMixedScopeFixture([
      '--json',
      '--scope',
      'paths',
      '--source',
      'src/docs-quality/scope.ts',
      '--run-id',
      'red-contract-scope-filter',
    ]);

    expect(result).toEqual(
      expect.objectContaining({
        report: expect.objectContaining({
          summary: expect.objectContaining({
            coverage: expect.objectContaining({
              available: true,
              totalFiles: 1,
              filesBelow100: 1,
              filesBelow100Detail: [
                expect.objectContaining({
                  file: 'src/docs-quality/scope.ts',
                }),
              ],
            }),
          }),
        }),
        status: 0,
      }),
    );
  });

  it('reports coverage separately from the docs-quality pass which requires zero evidence', () => {
    const result = runMetricsCliEvaluationWithMixedScopeFixture([
      '--json',
      '--scope',
      'paths',
      '--source',
      'src/docs-quality/scope.ts',
      '--run-id',
      'red-contract-coverage-gate',
    ]);

    expect(result).toEqual(
      expect.objectContaining({
        report: expect.objectContaining({
          pass: true,
          summary: expect.objectContaining({
            coverage: expect.objectContaining({
              coveragePass: false,
            }),
          }),
        }),
        status: 0,
      }),
    );
  });

  it('sorts canonical evidence hottest-first with descending numeric values inside each severity tier', () => {
    const evidence = [
      {
        file: 'src/zeta.ts',
        issue: 'weak JSDoc',
        numericValue: 8,
        symbol: 'zetaWeak',
      },
      {
        file: 'src/alpha.ts',
        issue: 'high complexity',
        numericValue: 12,
        symbol: 'alphaHot',
      },
      {
        file: 'src/beta.ts',
        issue: 'missing JSDoc',
        numericValue: 0,
        symbol: 'betaMissing',
      },
      {
        file: 'src/gamma.ts',
        issue: 'high complexity',
        numericValue: 20,
        symbol: 'gammaHot',
      },
      {
        file: 'src/epsilon.ts',
        issue: 'high complexity',
        numericValue: 20,
        symbol: 'epsilonHot',
      },
      {
        file: 'src/delta.ts',
        issue: 'weak JSDoc',
        numericValue: 3,
        symbol: 'deltaWeak',
      },
      {
        file: 'src/alpha.ts',
        issue: 'missing JSDoc',
        numericValue: 0,
        symbol: 'alphaMissing',
      },
    ];

    const result = runModuleEvaluation<MetricsContractReport>(`
      import { normalizeDocsQualityEvidence } from './rag-index/docs-quality/docs-quality.normalize.mjs';
      console.log(JSON.stringify({
        canonicalEvidence: normalizeDocsQualityEvidence(${JSON.stringify(evidence)}),
        digestA: '',
        digestB: '',
        isSchemaValid: true,
        missingFields: [],
      }));
    `);

    expect(result).toEqual(
      expect.objectContaining({
        report: expect.objectContaining({
          canonicalEvidence: [
            {
              file: 'src/epsilon.ts',
              issue: 'high complexity',
              numericValue: 20,
              symbol: 'epsilonHot',
            },
            {
              file: 'src/gamma.ts',
              issue: 'high complexity',
              numericValue: 20,
              symbol: 'gammaHot',
            },
            {
              file: 'src/alpha.ts',
              issue: 'high complexity',
              numericValue: 12,
              symbol: 'alphaHot',
            },
            {
              file: 'src/alpha.ts',
              issue: 'missing JSDoc',
              numericValue: 0,
              symbol: 'alphaMissing',
            },
            {
              file: 'src/beta.ts',
              issue: 'missing JSDoc',
              numericValue: 0,
              symbol: 'betaMissing',
            },
            {
              file: 'src/zeta.ts',
              issue: 'weak JSDoc',
              numericValue: 8,
              symbol: 'zetaWeak',
            },
            {
              file: 'src/delta.ts',
              issue: 'weak JSDoc',
              numericValue: 3,
              symbol: 'deltaWeak',
            },
          ],
        }),
        status: 0,
      }),
    );
  });

  it('manifest generatedAt is a fresh ISO timestamp, not the frozen canonical value', () => {
    const beforeMs = Date.now();
    const result = runMetricsCliEvaluationWithCoverageFixture([
      '--json',
      '--scope',
      'paths',
      '--source',
      'src/neat.ts',
      '--run-id',
      'red-contract-generated-at-freshness',
    ]);
    const afterMs = Date.now();

    expect(result.status).toBe(0);
    const generatedAt = (
      result.report?.manifest as Record<string, unknown> | undefined
    )?.generatedAt;
    expect(generatedAt).not.toBe('1970-01-01T00:00:00.000Z');
    expect(generatedAt).toBeTruthy();
    const generatedAtMs = Date.parse(String(generatedAt));
    expect(generatedAtMs).toBeGreaterThanOrEqual(beforeMs - 1000);
    expect(generatedAtMs).toBeLessThanOrEqual(afterMs + 1000);
  });

  it('manifest gitCommit falls back to unknown without spawning git when GIT_COMMIT is absent', () => {
    const result = runMetricsCliEvaluationWithCoverageFixtureAndEnv(
      [
        '--json',
        '--scope',
        'paths',
        '--source',
        'src/neat.ts',
        '--run-id',
        'red-contract-git-free-fallback',
      ],
      Object.fromEntries(
        Object.entries(process.env).filter(([key]) => key !== 'GIT_COMMIT'),
      ),
    );

    expect(result.status).toBe(0);
    const gitCommit = (
      result.report?.manifest as Record<string, unknown> | undefined
    )?.gitCommit;
    expect(gitCommit).toBe('unknown');
  });

  it('rejects manifests that include legacy top-level fields after schema cleanup', () => {
    const cleanManifest = JSON.parse(
      readFileSync(path.join(FIXTURES_ROOT, 'manifest.v1.base.json'), 'utf8'),
    ) as Record<string, unknown>;
    const scopeConfig = cleanManifest.scopeConfig as
      Record<string, unknown> | undefined;
    const legacyManifest = {
      ...cleanManifest,
      scopeDigest: scopeConfig?.scopeDigest,
      scopeType: scopeConfig?.scopeType,
      threshold: cleanManifest.thresholdConfig,
    };

    const result = runModuleEvaluation<{
      errors: Array<{ field: string; message: string }>;
      isSchemaValid: boolean;
    }>(`
      import { validateDocsQualityManifestV1 } from './rag-index/docs-quality/docs-quality.contract.mjs';
      const validation = validateDocsQualityManifestV1(${JSON.stringify(legacyManifest)});
      console.log(JSON.stringify({
        isSchemaValid: validation.valid,
        errors: validation.errors,
      }));
    `);

    expect(result).toEqual(
      expect.objectContaining({
        report: expect.objectContaining({
          isSchemaValid: false,
          errors: expect.arrayContaining([
            expect.objectContaining({ field: 'scopeDigest' }),
            expect.objectContaining({ field: 'scopeType' }),
            expect.objectContaining({ field: 'threshold' }),
          ]),
        }),
        status: 0,
      }),
    );
  });

  it('manifest uses bumped metric and scanner versions after schema cleanup', () => {
    const result = runMetricsCliEvaluationWithCoverageFixture([
      '--json',
      '--scope',
      'paths',
      '--source',
      'src/neat.ts',
      '--source',
      'src/architecture/network.ts',
      '--run-id',
      'red-contract-schema-version-bump',
    ]);

    expect(result.status).toBe(0);
    const manifest = result.report?.manifest as
      Record<string, unknown> | undefined;
    expect(manifest?.metricVersion).toBe(2);
    expect(manifest?.scannerVersion).toBe('2.0.0');
  });

  it('generated manifest excludes legacy top-level fields after schema cleanup', () => {
    const result = runMetricsCliEvaluationWithCoverageFixture([
      '--json',
      '--scope',
      'paths',
      '--source',
      'src/neat.ts',
      '--source',
      'src/architecture/network.ts',
      '--run-id',
      'red-contract-legacy-fields-absent',
    ]);

    expect(result.status).toBe(0);
    const manifest = result.report?.manifest as
      Record<string, unknown> | undefined;
    expect(manifest).not.toHaveProperty('threshold');
    expect(manifest).not.toHaveProperty('scopeType');
    expect(manifest).not.toHaveProperty('scopeDigest');
  });
});

function runMetricsCliEvaluation(
  argumentsVector: string[],
  environment: NodeJS.ProcessEnv = process.env,
): SpawnedJsonResult<MetricsContractReport> {
  const spawned = spawnSync(
    process.execPath,
    [DOCS_QUALITY_METRICS_PATH, ...argumentsVector],
    {
      cwd: REPO_ROOT,
      encoding: 'utf8',
      env: environment,
    },
  );

  const parsedReport = tryParseJson<Record<string, unknown>>(
    spawned.stdout ?? '',
  );
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

function runMetricsCliEvaluationWithCoverageFixture(
  argumentsVector: string[],
): SpawnedJsonResult<MetricsContractReport> {
  return withTemporaryCoverageFixture(() =>
    runMetricsCliEvaluation(argumentsVector),
  );
}

function runMetricsCliEvaluationWithCoverageFixtureAndEnv(
  argumentsVector: string[],
  environment: NodeJS.ProcessEnv,
): SpawnedJsonResult<MetricsContractReport> {
  return withTemporaryCoverageFixture(() =>
    runMetricsCliEvaluation(argumentsVector, environment),
  );
}

function runMetricsCliEvaluationWithPartialCoverageFixture(
  argumentsVector: string[],
): SpawnedJsonResult<MetricsContractReport> {
  return withTemporaryCoverageFixture(
    () => runMetricsCliEvaluation(argumentsVector),
    PARTIAL_COVERAGE_FIXTURE_SUMMARY,
  );
}

function runMetricsCliEvaluationWithMixedScopeFixture(
  argumentsVector: string[],
): SpawnedJsonResult<MetricsContractReport> {
  return withTemporaryLcovFixture(() =>
    runMetricsCliEvaluation(argumentsVector),
  );
}

function withCoverageDirectoryTemporarilyMissing<ResultType>(
  callback: () => ResultType,
): ResultType {
  const coverageDirectorySwapState = hideCoverageDirectory();

  try {
    return callback();
  } finally {
    restoreCoverageDirectory(coverageDirectorySwapState);
  }
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
    },
  );

  return {
    report: tryParseJson<ReportType>(spawned.stdout ?? ''),
    status: spawned.status,
    stderr: spawned.stderr ?? '',
    stdout: spawned.stdout ?? '',
  };
}

function withTemporaryCoverageFixture<ResultType>(
  callback: () => ResultType,
  coverageSummary = COVERAGE_FIXTURE_SUMMARY,
): ResultType {
  const coverageDirectorySwapState =
    installTemporaryCoverageFixture(coverageSummary);

  try {
    return callback();
  } finally {
    restoreCoverageDirectory(coverageDirectorySwapState);
  }
}

function withTemporaryLcovFixture<ResultType>(
  callback: () => ResultType,
): ResultType {
  const coverageDirectorySwapState = hideCoverageDirectory();

  mkdirSync(COVERAGE_DIRECTORY_PATH, { recursive: true });
  writeFileSync(
    path.join(COVERAGE_DIRECTORY_PATH, 'lcov.info'),
    MIXED_SCOPE_FIXTURE.lcov,
  );
  writeFileSync(
    path.join(COVERAGE_DIRECTORY_PATH, 'coverage-summary.json'),
    JSON.stringify(MIXED_SCOPE_FIXTURE.summary, null, 2),
  );

  try {
    return callback();
  } finally {
    restoreCoverageDirectory(coverageDirectorySwapState);
  }
}

function hideCoverageDirectory(): CoverageDirectorySwapState {
  const backupDirectoryPath = existsSync(COVERAGE_DIRECTORY_PATH)
    ? `${COVERAGE_DIRECTORY_PATH}.docs-quality-backup.${process.pid}.${Date.now()}.${Math.random().toString(36).slice(2)}`
    : null;

  if (backupDirectoryPath) {
    retryRenameSync(COVERAGE_DIRECTORY_PATH, backupDirectoryPath);
  }

  return { backupDirectoryPath };
}

function installTemporaryCoverageFixture(
  coverageSummary = COVERAGE_FIXTURE_SUMMARY,
): CoverageDirectorySwapState {
  const coverageDirectorySwapState = hideCoverageDirectory();

  mkdirSync(COVERAGE_DIRECTORY_PATH, { recursive: true });
  writeFileSync(
    path.join(COVERAGE_DIRECTORY_PATH, 'lcov.info'),
    COVERAGE_FIXTURE_LCOV,
  );
  writeFileSync(
    path.join(COVERAGE_DIRECTORY_PATH, 'coverage-summary.json'),
    JSON.stringify(coverageSummary, null, 2),
  );

  return coverageDirectorySwapState;
}

function restoreCoverageDirectory({
  backupDirectoryPath,
}: CoverageDirectorySwapState): void {
  rmSync(COVERAGE_DIRECTORY_PATH, { recursive: true, force: true });

  if (backupDirectoryPath && existsSync(backupDirectoryPath)) {
    retryRenameSync(backupDirectoryPath, COVERAGE_DIRECTORY_PATH);
  }
}

function retryRenameSync(source: string, destination: string): void {
  const maxRetries = 10;
  const delayMs = 50;
  let lastError: unknown;

  for (let attempt = 0; attempt < maxRetries; attempt += 1) {
    try {
      renameSync(source, destination);
      return;
    } catch (error) {
      lastError = error;
      if (attempt < maxRetries - 1) {
        Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, delayMs);
      }
    }
  }

  throw lastError;
}

function tryParseJson<ReportType>(stdout: string): ReportType | null {
  if (!stdout.trim()) return null;

  try {
    return JSON.parse(stdout) as ReportType;
  } catch {
    return null;
  }
}
