import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { spawnSync } from 'node:child_process';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import { fail, parseCliArgs, printHelp, writeJsonOrText } from '../cli-utils.mjs';
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
    sourcePaths: scopeConfig.scopeType === 'paths' ? scopeConfig.scopeValue : undefined,
  });

  const canonicalEvidence = normalizeDocsQualityEvidence(scannerReport.evidence);
  const issueBreakdown = summarizeIssueBreakdown(canonicalEvidence);
  const summary = {
    evidenceCount: canonicalEvidence.length,
    highComplexity: issueBreakdown.highComplexity,
    missingJsdoc: issueBreakdown.missingJsdoc,
    weakCount: issueBreakdown.weakJsdoc,
    weakJsdoc: issueBreakdown.weakJsdoc,
  };

  const normalizedEvidenceDigest = computeNormalizedEvidenceDigest(canonicalEvidence);
  const sourcePathsDigest = computeSourcePathsDigest(scopeConfig.scopeType === 'paths'
    ? scopeConfig.scopeValue
    : [scopeConfig.scopeType]);
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
  };

  const validation = validateDocsQualityManifestV1(manifest);
  if (!validation.valid) {
    throw new Error(`Invalid docs-quality manifest: ${validation.errors.map(({ field }) => field).join(', ')}`);
  }

  const artifacts = await writeDocsQualityRunArtifacts({
    summary,
    evidence: {
      evidence: canonicalEvidence,
      normalizedEvidenceDigest,
    },
    manifest: {
      ...manifest,
      summaryPath: toRepoRelativePath(path.join('artifacts', 'docs-quality', 'runs', runId, 'summary.json')),
      evidencePath: toRepoRelativePath(path.join('artifacts', 'docs-quality', 'runs', runId, 'evidence.json')),
      manifestPath: toRepoRelativePath(path.join('artifacts', 'docs-quality', 'runs', runId, 'manifest.json')),
    },
    runId,
  });

  const finalManifest = JSON.parse(await readFile(artifacts.manifestPath, 'utf8'));

  return {
    pass: scannerReport.pass,
    evidence: canonicalEvidence,
    summary,
    manifest: finalManifest,
    artifacts,
  };
}

function resolveThresholds(options) {
  const complexityThreshold = Number(options.complexityThreshold ?? DEFAULT_COMPLEXITY_THRESHOLD);
  const minJsdocWords = Number(options.minJsdocWords ?? DEFAULT_MIN_JSDOC_WORDS);
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
  const normalizedScope = String(options.scope ?? (Array.isArray(options.sourcePaths) && options.sourcePaths.length > 0 ? 'paths' : 'src'));
  const scopeType = normalizedScope === 'paths' ? 'paths' : 'src';
  const scopeValue = scopeType === 'paths'
    ? (Array.isArray(options.sourcePaths) ? options.sourcePaths : [])
    : ['src'];
  const normalizedScopeConfig = normalizeScopeInputAndDigest({
    scopeType,
    scopeValue,
  });

  if (normalizedScopeConfig.scopeType === 'paths' && normalizedScopeConfig.scopeValue.length === 0) {
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
    if (evidenceEntry.issue === 'missing JSDoc') issueBreakdown.missingJsdoc += 1;
    if (evidenceEntry.issue === 'weak JSDoc') issueBreakdown.weakJsdoc += 1;
    if (evidenceEntry.issue === 'high complexity') issueBreakdown.highComplexity += 1;
  }

  return issueBreakdown;
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
      usage: 'node scripts/semantic-index/docs-quality/docs-quality.metrics.mjs [--json]',
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

  const providedSources = [args.source, ...(Array.isArray(args._) ? args._ : [])]
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

    writeJsonOrText(serializedRun, Boolean(args.json), (payload) => [
      `Docs-quality metrics completed: ${payload.summary.evidenceCount} evidence row(s).`,
      `Manifest: ${payload.manifest.manifestPath}`,
    ].join('\n'));
  } catch (error) {
    fail(error instanceof Error ? error.message : String(error), Boolean(args.json));
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (existsSync(path.resolve(process.argv[1]))) {
    await main();
  }
}