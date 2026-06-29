import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  fail,
  parseCliArgs,
  printHelp,
  writeJsonOrText,
} from '../cli-utils.mjs';
import { scanCodeQuality } from '../code-quality-scanner.mjs';
import {
  DOCS_QUALITY_METRIC_VERSION,
  DOCS_QUALITY_SCANNER_VERSION,
  validateDocsQualityManifestV1,
} from './docs-quality.contract.mjs';
import {
  computeNormalizedEvidenceDigest,
  computeSourcePathsDigest,
  normalizeDocsQualityEvidence,
  normalizeScopeInputAndDigest,
} from './docs-quality.normalize.mjs';
import { writeDocsQualityRunArtifacts } from './docs-quality.artifacts.mjs';

const DEFAULT_COMPLEXITY_THRESHOLD = 10;
const DEFAULT_MIN_JSDOC_WORDS = 10;
const DEFAULT_RUN_ID = 'default';
const CANONICAL_GENERATED_AT = '1970-01-01T00:00:00.000Z';
const FULL_COVERAGE_PCT_THRESHOLD = 99;

/**
 * Run canonical docs-quality metrics and persist run artifacts.
 *
 * @param {{ complexityThreshold?: number, minJsdocWords?: number, runId?: string, scope?: string, sourcePaths?: string[] }} options - Runner options.
 * @returns {Promise<{ artifacts: { evidencePath: string, manifestPath: string, runDirectory: string, summaryPath: string }, evidence: Array<{ file: string, issue: string, numericValue: number, symbol: string }>, manifest: Record<string, unknown>, pass: boolean, summary: Record<string, unknown> }>} Canonical run output.
 */
export async function runDocsQualityMetrics(options = {}) {
  const thresholds = resolveThresholds(options);
  const scopeConfig = resolveScopeConfig(options);
  const scannerReport = await scanCodeQuality({
    complexityThreshold: thresholds.complexityThreshold,
    minJsdocWords: thresholds.minJsdocWords,
    sourcePaths:
      scopeConfig.scopeType === 'paths' ? scopeConfig.scopeValue : undefined,
  });
  const coverage = await parseLcovSummary(
    path.resolve(process.cwd(), 'coverage', 'lcov.info'),
  );

  const canonicalEvidence = normalizeDocsQualityEvidence(
    scannerReport.evidence,
  );
  const issueBreakdown = summarizeIssueBreakdown(canonicalEvidence);
  const summary = {
    evidenceCount: canonicalEvidence.length,
    highComplexity: issueBreakdown.highComplexity,
    missingJsdoc: issueBreakdown.missingJsdoc,
    weakCount: issueBreakdown.weakJsdoc,
    weakJsdoc: issueBreakdown.weakJsdoc,
    coverage,
  };

  const normalizedEvidenceDigest =
    computeNormalizedEvidenceDigest(canonicalEvidence);
  const sourcePathsDigest = computeSourcePathsDigest(
    scopeConfig.scopeType === 'paths'
      ? scopeConfig.scopeValue
      : [scopeConfig.scopeType],
  );
  const runId = String(options.runId ?? DEFAULT_RUN_ID);
  const manifest = {
    metricVersion: DOCS_QUALITY_METRIC_VERSION,
    scannerVersion: DOCS_QUALITY_SCANNER_VERSION,
    gitCommit: resolveGitCommit(),
    generatedAt: CANONICAL_GENERATED_AT,
    thresholdConfig: {
      minJsdocWords: thresholds.minJsdocWords,
      complexityThreshold: thresholds.complexityThreshold,
    },
    scopeConfig: {
      scopeType: scopeConfig.scopeType,
      scopeValue: scopeConfig.scopeValue,
      scopeDigest: scopeConfig.scopeDigest,
    },
    sourcePathsDigest,
    issueBreakdown,
    weakCount: issueBreakdown.weakJsdoc,
    normalizedEvidenceDigest,
    threshold: {
      minJsdocWords: thresholds.minJsdocWords,
      complexityThreshold: thresholds.complexityThreshold,
    },
    scopeType: scopeConfig.scopeType,
    scopeDigest: sourcePathsDigest,
    coverage,
  };

  const validation = validateDocsQualityManifestV1(manifest);
  if (!validation.valid) {
    throw new Error(
      `Invalid docs-quality manifest: ${validation.errors.map(({ field }) => field).join(', ')}`,
    );
  }

  const artifacts = await writeDocsQualityRunArtifacts({
    summary,
    evidence: {
      evidence: canonicalEvidence,
      normalizedEvidenceDigest,
    },
    manifest: {
      ...manifest,
      summaryPath: toRepoRelativePath(
        path.join('artifacts', 'docs-quality', 'runs', runId, 'summary.json'),
      ),
      evidencePath: toRepoRelativePath(
        path.join('artifacts', 'docs-quality', 'runs', runId, 'evidence.json'),
      ),
      manifestPath: toRepoRelativePath(
        path.join('artifacts', 'docs-quality', 'runs', runId, 'manifest.json'),
      ),
    },
    runId,
  });

  const finalManifest = JSON.parse(
    await readFile(artifacts.manifestPath, 'utf8'),
  );

  return {
    pass: scannerReport.pass,
    evidence: canonicalEvidence,
    summary,
    manifest: finalManifest,
    artifacts,
  };
}

function resolveThresholds(options) {
  const complexityThreshold = Number(
    options.complexityThreshold ?? DEFAULT_COMPLEXITY_THRESHOLD,
  );
  const minJsdocWords = Number(
    options.minJsdocWords ?? DEFAULT_MIN_JSDOC_WORDS,
  );
  if (!Number.isFinite(complexityThreshold) || complexityThreshold < 0) {
    throw new Error('complexityThreshold must be a non-negative number.');
  }

  if (!Number.isFinite(minJsdocWords) || minJsdocWords < 0) {
    throw new Error('minJsdocWords must be a non-negative number.');
  }

  return {
    complexityThreshold,
    minJsdocWords,
  };
}

function resolveScopeConfig(options) {
  const normalizedScope = String(
    options.scope ??
      (Array.isArray(options.sourcePaths) && options.sourcePaths.length > 0
        ? 'paths'
        : 'src'),
  );
  const scopeType = normalizedScope === 'paths' ? 'paths' : 'src';
  const scopeValue =
    scopeType === 'paths'
      ? Array.isArray(options.sourcePaths)
        ? options.sourcePaths
        : []
      : ['src'];
  const normalizedScopeConfig = normalizeScopeInputAndDigest({
    scopeType,
    scopeValue,
  });

  if (
    normalizedScopeConfig.scopeType === 'paths' &&
    normalizedScopeConfig.scopeValue.length === 0
  ) {
    throw new Error('scope=paths requires at least one source path.');
  }

  return normalizedScopeConfig;
}

function summarizeIssueBreakdown(canonicalEvidence) {
  const issueBreakdown = {
    missingJsdoc: 0,
    weakJsdoc: 0,
    highComplexity: 0,
  };

  for (const evidenceEntry of canonicalEvidence) {
    if (evidenceEntry.issue === 'missing JSDoc')
      issueBreakdown.missingJsdoc += 1;
    if (evidenceEntry.issue === 'weak JSDoc') issueBreakdown.weakJsdoc += 1;
    if (evidenceEntry.issue === 'high complexity')
      issueBreakdown.highComplexity += 1;
  }

  return issueBreakdown;
}

/**
 * Parse a repo coverage artifact into a small additive summary.
 *
 * @param {string} lcovPath - Absolute path to coverage/lcov.info.
 * @returns {Promise<{ available: false } | { available: true, filesBelow100: number, overallBranches: number, overallFunctions: number, overallLines: number, totalFiles: number }>} Coverage summary.
 */
async function parseLcovSummary(lcovPath) {
  if (!existsSync(lcovPath)) {
    return { available: false };
  }

  const coverageSummaryPath = path.resolve(
    process.cwd(),
    'coverage',
    'coverage-summary.json',
  );
  const statementCoverageByFile =
    await readStatementCoverageByFile(coverageSummaryPath);
  const totalStatementCoverage =
    await readTotalStatementCoverage(coverageSummaryPath);
  const lcovContent = await readFile(lcovPath, 'utf8');
  const coverageRecords = lcovContent
    .split('end_of_record')
    .map((record) => record.trim())
    .filter(Boolean);

  const aggregate = {
    totalFiles: 0,
    filesBelow100: 0,
    lineHits: 0,
    lineFound: 0,
    branchHits: 0,
    branchFound: 0,
    functionHits: 0,
    functionFound: 0,
    filesBelow100Detail: [],
  };

  for (const coverageRecord of coverageRecords) {
    if (!coverageRecord.includes('SF:')) continue;

    const coverageFilePath = normalizeCoverageFilePath(
      readLcovSourceFile(coverageRecord),
    );
    const lineHits = readLcovCounter(coverageRecord, /^LH:(\d+)$/m);
    const lineFound = readLcovCounter(coverageRecord, /^LF:(\d+)$/m);
    const branchHits = readLcovCounter(coverageRecord, /^BRH:(\d+)$/m);
    const branchFound = readLcovCounter(coverageRecord, /^BRF:(\d+)$/m);
    const functionHits = readLcovCounter(coverageRecord, /^FNH:(\d+)$/m);
    const functionFound = readLcovCounter(coverageRecord, /^FNF:(\d+)$/m);

    aggregate.totalFiles += 1;
    aggregate.lineHits += lineHits;
    aggregate.lineFound += lineFound;
    aggregate.branchHits += branchHits;
    aggregate.branchFound += branchFound;
    aggregate.functionHits += functionHits;
    aggregate.functionFound += functionFound;

    const lineCoverage = toCoveragePercent(lineHits, lineFound);
    const branchCoverage = toCoveragePercent(branchHits, branchFound);
    const functionCoverage = toCoveragePercent(functionHits, functionFound);
    if (lineCoverage < 100 || branchCoverage < 100 || functionCoverage < 100) {
      aggregate.filesBelow100 += 1;
      const statementCoverage = statementCoverageByFile.get(coverageFilePath);
      aggregate.filesBelow100Detail.push({
        file: coverageFilePath,
        statements: Number.isFinite(statementCoverage)
          ? statementCoverage
          : lineCoverage,
        statementCoverageSource: Number.isFinite(statementCoverage)
          ? 'coverage-summary'
          : 'lcov-line-fallback',
        branches: branchCoverage,
        functions: functionCoverage,
        lines: lineCoverage,
        uncoveredLines: readUncoveredLineNumbers(coverageRecord),
        uncoveredBranches: readUncoveredBranches(coverageRecord),
        uncoveredFunctions: readUncoveredFunctionNames(coverageRecord),
      });
    }
  }

  const isPartialCoverage =
    Number.isFinite(totalStatementCoverage) &&
    totalStatementCoverage < FULL_COVERAGE_PCT_THRESHOLD;

  return {
    available: true,
    ...(isPartialCoverage ? { isPartial: true } : {}),
    totalFiles: aggregate.totalFiles,
    filesBelow100: aggregate.filesBelow100,
    filesBelow100Detail: aggregate.filesBelow100Detail.toSorted(
      compareCoverageDetailRows,
    ),
    overallLines: toCoveragePercent(aggregate.lineHits, aggregate.lineFound),
    overallBranches: toCoveragePercent(
      aggregate.branchHits,
      aggregate.branchFound,
    ),
    overallFunctions: toCoveragePercent(
      aggregate.functionHits,
      aggregate.functionFound,
    ),
  };
}

/**
 * Read per-file statement percentages from Istanbul's coverage summary when available.
 *
 * @param {string} coverageSummaryPath - Absolute path to coverage/coverage-summary.json.
 * @returns {Promise<Map<string, number>>} Statement coverage percentages keyed by normalized repo path.
 */
async function readStatementCoverageByFile(coverageSummaryPath) {
  if (!existsSync(coverageSummaryPath)) {
    return new Map();
  }

  try {
    const coverageSummary = JSON.parse(
      await readFile(coverageSummaryPath, 'utf8'),
    );
    const coverageSummaryEntries = Object.entries(coverageSummary).filter(
      ([coverageFilePath]) => coverageFilePath !== 'total',
    );

    return new Map(
      coverageSummaryEntries
        .map(([coverageFilePath, coverageEntry]) => {
          if (
            !isPlainObject(coverageEntry) ||
            !isPlainObject(coverageEntry.statements)
          ) {
            return null;
          }

          const statementPercent = Number(coverageEntry.statements.pct);
          if (!Number.isFinite(statementPercent)) {
            return null;
          }

          return [
            normalizeCoverageFilePath(coverageFilePath),
            statementPercent,
          ];
        })
        .filter(Boolean),
    );
  } catch {
    return new Map();
  }
}

/**
 * Read repo-wide total statement coverage from Istanbul's coverage summary when available.
 *
 * @param {string} coverageSummaryPath - Absolute path to coverage/coverage-summary.json.
 * @returns {Promise<number | null>} Total statement coverage percentage.
 */
async function readTotalStatementCoverage(coverageSummaryPath) {
  if (!existsSync(coverageSummaryPath)) {
    return null;
  }

  try {
    const coverageSummary = JSON.parse(
      await readFile(coverageSummaryPath, 'utf8'),
    );
    if (
      !isPlainObject(coverageSummary.total) ||
      !isPlainObject(coverageSummary.total.statements)
    ) {
      return null;
    }

    const statementPercent = Number(coverageSummary.total.statements.pct);
    return Number.isFinite(statementPercent) ? statementPercent : null;
  } catch {
    return null;
  }
}

/**
 * Read the source file path from an LCOV record.
 *
 * @param {string} coverageRecord - One LCOV file record.
 * @returns {string} Source file path as emitted by LCOV.
 */
function readLcovSourceFile(coverageRecord) {
  const sourceFileMatch = coverageRecord.match(/^SF:(.+)$/m);
  return sourceFileMatch?.[1]?.trim() ?? 'unknown';
}

/**
 * Normalize a coverage source path to a stable repo-relative path when possible.
 *
 * @param {string} coverageFilePath - Source file path from coverage artifacts.
 * @returns {string} Normalized coverage file path.
 */
function normalizeCoverageFilePath(coverageFilePath) {
  if (!coverageFilePath || coverageFilePath === 'unknown') {
    return 'unknown';
  }

  if (path.isAbsolute(coverageFilePath)) {
    const relativePath = path.relative(process.cwd(), coverageFilePath);
    if (!relativePath.startsWith('..')) {
      return toRepoRelativePath(relativePath);
    }
  }

  return toRepoRelativePath(coverageFilePath);
}

/**
 * Read uncovered line numbers from an LCOV record.
 *
 * @param {string} coverageRecord - One LCOV file record.
 * @returns {number[]} Uncovered line numbers.
 */
function readUncoveredLineNumbers(coverageRecord) {
  return coverageRecord
    .split('\n')
    .filter((coverageLine) => coverageLine.startsWith('DA:'))
    .map((coverageLine) => coverageLine.slice(3).split(','))
    .filter(([, hitCount]) => Number.parseInt(hitCount ?? '0', 10) === 0)
    .map(([lineNumber]) => Number.parseInt(lineNumber ?? '0', 10))
    .filter((lineNumber) => Number.isFinite(lineNumber));
}

/**
 * Read uncovered branch descriptors from an LCOV record.
 *
 * @param {string} coverageRecord - One LCOV file record.
 * @returns {Array<{ branch: string, block: string, line: number, taken: number | null }>} Uncovered branch metadata.
 */
function readUncoveredBranches(coverageRecord) {
  return coverageRecord
    .split('\n')
    .filter((coverageLine) => coverageLine.startsWith('BRDA:'))
    .map((coverageLine) => coverageLine.slice(5).split(','))
    .filter(
      ([, , , takenCount]) =>
        takenCount === '-' || Number.parseInt(takenCount ?? '0', 10) === 0,
    )
    .map(([lineNumber, blockNumber, branchNumber, takenCount]) => ({
      line: Number.parseInt(lineNumber ?? '0', 10),
      block: blockNumber ?? '0',
      branch: branchNumber ?? '0',
      taken: takenCount === '-' ? null : Number.parseInt(takenCount ?? '0', 10),
    }))
    .filter((branchEntry) => Number.isFinite(branchEntry.line));
}

/**
 * Read uncovered function names from an LCOV record.
 *
 * @param {string} coverageRecord - One LCOV file record.
 * @returns {string[]} Uncovered function names.
 */
function readUncoveredFunctionNames(coverageRecord) {
  return coverageRecord
    .split('\n')
    .filter((coverageLine) => coverageLine.startsWith('FNDA:'))
    .map((coverageLine) => coverageLine.slice(5).split(','))
    .filter(([hitCount]) => Number.parseInt(hitCount ?? '0', 10) === 0)
    .map(([, functionName]) => functionName ?? 'unknown')
    .filter(Boolean);
}

/**
 * Compare per-file coverage rows so the lowest-coverage files sort first.
 *
 * @param {{ branches: number, file: string, functions: number, lines: number, statements: number }} leftEntry - Left detail row.
 * @param {{ branches: number, file: string, functions: number, lines: number, statements: number }} rightEntry - Right detail row.
 * @returns {number} Sort comparator result.
 */
function compareCoverageDetailRows(leftEntry, rightEntry) {
  const leftWorstCoverage = Math.min(
    leftEntry.statements,
    leftEntry.branches,
    leftEntry.functions,
    leftEntry.lines,
  );
  const rightWorstCoverage = Math.min(
    rightEntry.statements,
    rightEntry.branches,
    rightEntry.functions,
    rightEntry.lines,
  );

  if (leftWorstCoverage !== rightWorstCoverage) {
    return leftWorstCoverage - rightWorstCoverage;
  }

  return leftEntry.file.localeCompare(rightEntry.file);
}

/**
 * Read one integer counter from an LCOV record.
 *
 * @param {string} coverageRecord - One LCOV file record.
 * @param {RegExp} counterPattern - Regex for the target counter.
 * @returns {number} Parsed counter value.
 */
function readLcovCounter(coverageRecord, counterPattern) {
  const match = coverageRecord.match(counterPattern);
  return Number.parseInt(match?.[1] ?? '0', 10);
}

/**
 * Convert hit/found counters into an integer coverage percentage.
 *
 * @param {number} coveredCount - Covered item count.
 * @param {number} totalCount - Total item count.
 * @returns {number} Rounded percentage.
 */
function toCoveragePercent(coveredCount, totalCount) {
  if (totalCount === 0) {
    return 100;
  }

  return Math.round((coveredCount / totalCount) * 100);
}

function isPlainObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function resolveGitCommit() {
  const fromEnvironment = process.env.GIT_COMMIT;
  if (typeof fromEnvironment === 'string' && fromEnvironment.trim()) {
    return fromEnvironment.trim();
  }

  const gitResult = spawnSync('git', ['rev-parse', 'HEAD'], {
    cwd: process.cwd(),
    encoding: 'utf8',
  });
  if (gitResult.status === 0) {
    const commitHash = String(gitResult.stdout ?? '').trim();
    if (commitHash.length > 0) return commitHash;
  }

  return 'unknown';
}

function toRepoRelativePath(candidatePath) {
  return candidatePath.replaceAll('\\', '/');
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Docs quality metrics runner',
      usage: 'node rag-index/docs-quality/docs-quality.metrics.mjs [--json]',
      options: [
        '--json                       Emit full run payload as JSON.',
        '--scope <src|paths>          Scope type (default: src, paths when --source is provided).',
        '--source <path>              Explicit source path when --scope=paths (repeatable).',
        '--min-jsdoc-words <n>        Minimum non-whitespace words in exported JSDoc (default: 10).',
        '--complexity-threshold <n>   Maximum allowed cyclomatic complexity (default: 10).',
        '--run-id <value>             Run folder ID under artifacts/docs-quality/runs/ (default: default).',
        '--help                       Show this help.',
      ],
    });
    return;
  }

  const providedSources = [
    args.source,
    ...(Array.isArray(args._) ? args._ : []),
  ]
    .flat()
    .filter(Boolean);

  try {
    const run = await runDocsQualityMetrics({
      scope: args.scope,
      runId: args['run-id'],
      sourcePaths: providedSources.length > 0 ? providedSources : undefined,
      minJsdocWords: args['min-jsdoc-words'],
      complexityThreshold: args['complexity-threshold'],
    });
    const serializedRun = {
      summary: run.summary,
      pass: run.pass,
      manifest: run.manifest,
      artifacts: run.artifacts,
      evidence: run.evidence,
    };

    writeJsonOrText(serializedRun, Boolean(args.json), (payload) =>
      [
        `Docs-quality metrics completed: ${payload.summary.evidenceCount} evidence row(s).`,
        `Manifest: ${payload.manifest.manifestPath}`,
      ].join('\n'),
    );
  } catch (error) {
    fail(
      error instanceof Error ? error.message : String(error),
      Boolean(args.json),
    );
  }
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  if (existsSync(path.resolve(process.argv[1]))) {
    await main();
  }
}
