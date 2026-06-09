import { readFileSync } from 'node:fs';
import { pathToFileURL } from 'node:url';

import {
  fail,
  parseCliArgs,
  printHelp,
  writeJsonOrText,
} from '../cli-utils.mjs';

const REASON_CODES = {
  METRIC_VERSION_MISMATCH: 'METRIC_VERSION_MISMATCH',
  THRESHOLD_MISMATCH: 'THRESHOLD_MISMATCH',
  SCOPE_TYPE_MISMATCH: 'SCOPE_TYPE_MISMATCH',
  SCOPE_DIGEST_MISMATCH: 'SCOPE_DIGEST_MISMATCH',
  SCANNER_VERSION_MISMATCH: 'SCANNER_VERSION_MISMATCH',
};

/**
 * Compare two docs-quality runs with strict compatibility guards.
 *
 * @param {{ leftManifest: Record<string, unknown>, rightManifest: Record<string, unknown>, leftSummary: Record<string, unknown>, rightSummary: Record<string, unknown> }} payload - Comparison payload.
 * @returns {{ accepted: boolean, reasonCode?: string, delta?: { evidenceCount: number, highComplexity: number, missingJsdoc: number, weakJsdoc: number } }} Comparison result.
 */
export function compareDocsQualityRuns(payload) {
  const leftDimensions = extractComparableDimensions(payload.leftManifest);
  const rightDimensions = extractComparableDimensions(payload.rightManifest);

  if (leftDimensions.metricVersion !== rightDimensions.metricVersion) {
    return {
      accepted: false,
      reasonCode: REASON_CODES.METRIC_VERSION_MISMATCH,
    };
  }

  if (
    leftDimensions.threshold.minJsdocWords !==
      rightDimensions.threshold.minJsdocWords ||
    leftDimensions.threshold.complexityThreshold !==
      rightDimensions.threshold.complexityThreshold
  ) {
    return { accepted: false, reasonCode: REASON_CODES.THRESHOLD_MISMATCH };
  }

  if (leftDimensions.scopeType !== rightDimensions.scopeType) {
    return { accepted: false, reasonCode: REASON_CODES.SCOPE_TYPE_MISMATCH };
  }

  if (leftDimensions.scopeDigest !== rightDimensions.scopeDigest) {
    return { accepted: false, reasonCode: REASON_CODES.SCOPE_DIGEST_MISMATCH };
  }

  if (leftDimensions.scannerVersion !== rightDimensions.scannerVersion) {
    return {
      accepted: false,
      reasonCode: REASON_CODES.SCANNER_VERSION_MISMATCH,
    };
  }

  const leftSummary = normalizeSummary(payload.leftSummary);
  const rightSummary = normalizeSummary(payload.rightSummary);

  return {
    accepted: true,
    delta: {
      missingJsdoc: rightSummary.missingJsdoc - leftSummary.missingJsdoc,
      weakJsdoc: rightSummary.weakJsdoc - leftSummary.weakJsdoc,
      highComplexity: rightSummary.highComplexity - leftSummary.highComplexity,
      evidenceCount: rightSummary.evidenceCount - leftSummary.evidenceCount,
    },
  };
}

function extractComparableDimensions(manifest) {
  const thresholdConfig = isPlainObject(manifest.thresholdConfig)
    ? manifest.thresholdConfig
    : isPlainObject(manifest.threshold)
      ? manifest.threshold
      : {};
  const scopeConfig = isPlainObject(manifest.scopeConfig)
    ? manifest.scopeConfig
    : {};

  return {
    metricVersion: Number(manifest.metricVersion ?? 0),
    scannerVersion: String(manifest.scannerVersion ?? ''),
    threshold: {
      minJsdocWords: Number(thresholdConfig.minJsdocWords ?? 0),
      complexityThreshold: Number(thresholdConfig.complexityThreshold ?? 0),
    },
    scopeType: String(scopeConfig.scopeType ?? manifest.scopeType ?? ''),
    scopeDigest: String(
      manifest.scopeDigest ??
        scopeConfig.scopeDigest ??
        manifest.sourcePathsDigest ??
        '',
    ),
  };
}

function normalizeSummary(summary) {
  const issueBreakdown = isPlainObject(summary.issueBreakdown)
    ? summary.issueBreakdown
    : {};
  return {
    missingJsdoc: Number(
      summary.missingJsdoc ?? issueBreakdown.missingJsdoc ?? 0,
    ),
    weakJsdoc: Number(summary.weakJsdoc ?? issueBreakdown.weakJsdoc ?? 0),
    highComplexity: Number(
      summary.highComplexity ?? issueBreakdown.highComplexity ?? 0,
    ),
    evidenceCount: Number(summary.evidenceCount ?? 0),
  };
}

function isPlainObject(value) {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function readJsonFile(filePath) {
  return JSON.parse(readFileSync(filePath, 'utf8'));
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Docs quality compare',
      usage:
        'node scripts/semantic-index/docs-quality/docs-quality.compare.mjs --left=<manifest> --right=<manifest> [--json]',
      options: [
        '--left <manifestPath>        Left run manifest path.',
        '--right <manifestPath>       Right run manifest path.',
        '--json                       Emit comparison payload as JSON.',
        '--help                       Show this help.',
      ],
    });
    return;
  }

  const leftManifestPath = String(args.left ?? '').trim();
  const rightManifestPath = String(args.right ?? '').trim();
  if (!leftManifestPath || !rightManifestPath) {
    fail(
      'Both --left and --right manifest paths are required.',
      Boolean(args.json),
    );
    return;
  }

  try {
    const leftManifest = readJsonFile(leftManifestPath);
    const rightManifest = readJsonFile(rightManifestPath);
    const leftSummary = readJsonFile(leftManifest.summaryPath);
    const rightSummary = readJsonFile(rightManifest.summaryPath);
    const result = compareDocsQualityRuns({
      leftManifest,
      rightManifest,
      leftSummary,
      rightSummary,
    });

    writeJsonOrText(result, Boolean(args.json), (payload) =>
      payload.accepted
        ? `Docs-quality compare accepted. Delta evidence count: ${payload.delta.evidenceCount}`
        : `Docs-quality compare rejected: ${payload.reasonCode}`,
    );
    if (!result.accepted) process.exitCode = 1;
  } catch (error) {
    fail(
      error instanceof Error ? error.message : String(error),
      Boolean(args.json),
    );
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();

export { REASON_CODES };
