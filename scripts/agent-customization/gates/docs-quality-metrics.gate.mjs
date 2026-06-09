#!/usr/bin/env node
import { pathToFileURL } from 'node:url';

import { createRepoCortexMcpServer } from '../../mcp-semantic/repo-cortex-mcp.mjs';
import {
  DOCS_QUALITY_METRIC_VERSION,
  DOCS_QUALITY_SCANNER_VERSION,
  validateDocsQualityManifestV1,
} from '../../semantic-index/docs-quality/docs-quality.contract.mjs';
import {
  compareDocsQualityRuns,
  REASON_CODES,
} from '../../semantic-index/docs-quality/docs-quality.compare.mjs';
import { runDocsQualityMetrics } from '../../semantic-index/docs-quality/docs-quality.metrics.mjs';

const OWNER = '05-green-testing';

function parseArgs(argumentsVector) {
  return {
    help: argumentsVector.includes('--help') || argumentsVector.includes('-h'),
    json: argumentsVector.includes('--json'),
  };
}

function printUsage() {
  console.log(
    [
      'Docs-quality metrics gate',
      '',
      'Usage:',
      '  node scripts/agent-customization/gates/docs-quality-metrics.gate.mjs [--json]',
      '  node scripts/agent-customization/gates/docs-quality-metrics.gate.mjs --help',
      '',
      'Checks:',
      '  - schema validity',
      '  - deterministic ordering/digest stability',
      '  - comparator guard behavior',
      '  - CLI/MCP parity',
      '  - contract-invalid output rejection',
    ].join('\n'),
  );
}

export async function runDocsQualityMetricsGate() {
  const baselineRun = await runDocsQualityMetrics({
    scope: 'paths',
    sourcePaths: ['src/neat.ts', 'src/architecture/network.ts'],
    minJsdocWords: 10,
    complexityThreshold: 10,
    runId: 'gate-baseline',
  });
  const deterministicRun = await runDocsQualityMetrics({
    scope: 'paths',
    sourcePaths: ['src/architecture/network.ts', 'src/neat.ts', 'src/neat.ts'],
    minJsdocWords: 10,
    complexityThreshold: 10,
    runId: 'gate-deterministic',
  });
  const thresholdMismatchRun = await runDocsQualityMetrics({
    scope: 'paths',
    sourcePaths: ['src/neat.ts', 'src/architecture/network.ts'],
    minJsdocWords: 11,
    complexityThreshold: 10,
    runId: 'gate-threshold-mismatch',
  });

  const baselineSchema = validateDocsQualityManifestV1(baselineRun.manifest);
  const deterministicSchema = validateDocsQualityManifestV1(
    deterministicRun.manifest,
  );

  const deterministicCompare = compareDocsQualityRuns({
    leftManifest: baselineRun.manifest,
    rightManifest: deterministicRun.manifest,
    leftSummary: baselineRun.summary,
    rightSummary: deterministicRun.summary,
  });
  const mismatchCompare = compareDocsQualityRuns({
    leftManifest: baselineRun.manifest,
    rightManifest: thresholdMismatchRun.manifest,
    leftSummary: baselineRun.summary,
    rightSummary: thresholdMismatchRun.summary,
  });

  const mcpServer = createRepoCortexMcpServer({
    databasePath: './data/semantic-index.sqlite',
  });
  const mcpResponse = await mcpServer.dispatch({
    jsonrpc: '2.0',
    id: 1,
    method: 'tools/call',
    params: {
      name: 'scan_code_quality',
      arguments: {
        complexity_threshold: 10,
        min_jsdoc_words: 10,
        source_paths: ['src/neat.ts', 'src/architecture/network.ts'],
      },
    },
  });
  const mcpManifest = mcpResponse?.structuredContent?.manifest;
  const parityMatch = hasMatchingManifestContractFields(
    baselineRun.manifest,
    mcpManifest,
  );

  const invalidManifestCandidate = {
    ...baselineRun.manifest,
  };
  delete invalidManifestCandidate.scannerVersion;
  const invalidValidation = validateDocsQualityManifestV1(
    invalidManifestCandidate,
  );

  const pass =
    baselineSchema.valid &&
    deterministicSchema.valid &&
    Number(baselineRun.manifest.metricVersion) ===
      DOCS_QUALITY_METRIC_VERSION &&
    String(baselineRun.manifest.scannerVersion) ===
      DOCS_QUALITY_SCANNER_VERSION &&
    deterministicCompare.accepted &&
    deterministicCompare.delta?.evidenceCount === 0 &&
    deterministicCompare.delta?.highComplexity === 0 &&
    deterministicCompare.delta?.missingJsdoc === 0 &&
    deterministicCompare.delta?.weakJsdoc === 0 &&
    baselineRun.manifest.normalizedEvidenceDigest ===
      deterministicRun.manifest.normalizedEvidenceDigest &&
    baselineRun.manifest.scopeDigest ===
      deterministicRun.manifest.scopeDigest &&
    mismatchCompare.accepted === false &&
    mismatchCompare.reasonCode === REASON_CODES.THRESHOLD_MISMATCH &&
    parityMatch &&
    invalidValidation.valid === false &&
    invalidValidation.errors.some(
      (errorEntry) => errorEntry.field === 'scannerVersion',
    );

  return {
    schema_version: 1,
    pass,
    gateType: 'mechanism-only',
    probeScope:
      'docs-quality gate mechanism checks only: schema validity, deterministic ordering, comparator guards, CLI/MCP parity, and invalid contract rejection.',
    repoWideDebtCommand: 'npm run docs:quality:metrics',
    evidence: {
      schema_valid: baselineSchema.valid && deterministicSchema.valid,
      deterministic_ordering:
        baselineRun.manifest.normalizedEvidenceDigest ===
        deterministicRun.manifest.normalizedEvidenceDigest,
      comparator_guards:
        mismatchCompare.accepted === false &&
        mismatchCompare.reasonCode === REASON_CODES.THRESHOLD_MISMATCH,
      parity_cli_mcp: parityMatch,
      contract_invalid_rejected: invalidValidation.valid === false,
      metric_version: baselineRun.manifest.metricVersion,
      scanner_version: baselineRun.manifest.scannerVersion,
    },
    fixHint: pass
      ? null
      : 'Re-run docs-quality unit tests and ensure contract/compare/parity modules remain aligned with docs-quality-metrics.gate checks.',
    owner: OWNER,
  };
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printUsage();
    return;
  }

  const report = await runDocsQualityMetricsGate();
  if (options.json) {
    console.log(JSON.stringify(report, null, 2));
  } else {
    console.log(
      report.pass
        ? 'PASS docs-quality-metrics gate'
        : 'FAIL docs-quality-metrics gate',
    );
    if (!report.pass) {
      console.log(`fixHint: ${report.fixHint}`);
    }
  }

  process.exitCode = report.pass ? 0 : 1;
}

if (
  process.argv[1] &&
  import.meta.url === pathToFileURL(process.argv[1]).href
) {
  await main();
}

function hasMatchingManifestContractFields(leftManifest, rightManifest) {
  if (!rightManifest || typeof rightManifest !== 'object') {
    return false;
  }

  return (
    Number(leftManifest.metricVersion) ===
      Number(rightManifest.metricVersion) &&
    String(leftManifest.scannerVersion) ===
      String(rightManifest.scannerVersion) &&
    String(leftManifest.sourcePathsDigest) ===
      String(rightManifest.sourcePathsDigest) &&
    String(leftManifest.scopeDigest) === String(rightManifest.scopeDigest) &&
    String(leftManifest.normalizedEvidenceDigest) ===
      String(rightManifest.normalizedEvidenceDigest) &&
    Number(leftManifest.thresholdConfig?.minJsdocWords) ===
      Number(rightManifest.thresholdConfig?.minJsdocWords) &&
    Number(leftManifest.thresholdConfig?.complexityThreshold) ===
      Number(rightManifest.thresholdConfig?.complexityThreshold) &&
    String(leftManifest.scopeConfig?.scopeType) ===
      String(rightManifest.scopeConfig?.scopeType) &&
    JSON.stringify(leftManifest.scopeConfig?.scopeValue ?? []) ===
      JSON.stringify(rightManifest.scopeConfig?.scopeValue ?? [])
  );
}
