/**
 * @description Standard gate for the Repo Cortex dense embedding layer. Checks three
 * conditions in order:
 * 1. ONNX model assets are present in `scripts/semantic-index/models/`
 *    (`model.onnx` and `model-meta.json`).
 * 2. The consolidated corpus DB (`data/turso-replica.sqlite`) has usable embeddings
 *    for the active model in the `chunks.embedding` column
 *    (delegates to `validate-embeddings.mjs`).
 * 3. Hybrid MRR\@5 exceeds BM25-only MRR\@5 by at least the minimum improvement
 *    threshold (default: +0.02, measured by `eval-embeddings.mjs`).
 *
 * Emits the standard gate JSON contract `{ pass, evidence, fixHint, owner }`. Exits 0
 * when all three conditions pass.
 *
 * @param {boolean} [--json]                           - Emit the standard gate JSON contract.
 * @param {string}  [--database=<path>]                - Override corpus database path.
 * @param {string}  [--model-directory=<path>]         - Override ONNX model cache directory.
 * @param {string}  [--model-id=<id>]                  - Restrict validation to one model id.
 * @param {string}  [--query-file=<path>]              - Override eval query set path.
 * @param {number}  [--min-hybrid-improvement=<n>]     - Override the hybrid MRR\@5 minimum improvement.
 * @param {boolean} [--help] [-h]                      - Show help and exit.
 *
 * @returns {void} Exits 0 when the gate passes, 1 when any condition fails.
 */
import { existsSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { validateEmbeddings } from '../../semantic-index/validate-embeddings.mjs';
import { DEFAULT_MODEL_DIRECTORY } from '../../semantic-index/embed-index.mjs';

const OWNER = '05-green-testing';
const DEFAULT_MIN_HYBRID_IMPROVEMENT = 0.02;

export function evaluateCortexEmbeddingsGate(options = {}) {
  const minHybridImprovement = Number(
    options.minHybridImprovement ?? DEFAULT_MIN_HYBRID_IMPROVEMENT,
  );
  const chunkCount = Number(options.chunkCount ?? 0);
  const embeddingCount = Number(options.embeddingCount ?? 0);
  const bm25MrrAt5 = Number(options.bm25MrrAt5 ?? 0);
  const hybridMrrAt5 = Number(options.hybridMrrAt5 ?? 0);
  const evidence = [];

  if (embeddingCount === 0) {
    evidence.push({
      actual: embeddingCount,
      expected: chunkCount,
      issue: 'no usable embeddings for model',
    });
  }

  if (hybridMrrAt5 < bm25MrrAt5 + minHybridImprovement) {
    evidence.push({
      bm25MrrAt5,
      hybridMrrAt5,
      issue: 'hybrid MRR@5 improvement below threshold',
      minHybridImprovement,
    });
  }

  return createGateReport(evidence);
}

export async function runCortexEmbeddingsGate(options = {}) {
  const modelDirectory = path.resolve(
    options.modelDirectory ?? DEFAULT_MODEL_DIRECTORY,
  );
  const evidence = [];

  if (
    !existsSync(path.join(modelDirectory, 'model-meta.json')) ||
    !existsSync(path.join(modelDirectory, 'model.onnx'))
  ) {
    evidence.push({
      issue: 'model assets missing',
      modelDirectory,
    });
  }

  try {
    const validationReport = await validateEmbeddings(options);
    evidence.push(...validationReport.evidence);

    const evaluationReport =
      options.evaluationReport ?? (await loadEvaluationReport(options));
    evidence.push(
      ...evaluateCortexEmbeddingsGate({
        bm25MrrAt5: evaluationReport.bm25MrrAt5,
        chunkCount: evaluationReport.chunkCount,
        embeddingCount: evaluationReport.embeddingCount,
        hybridMrrAt5: evaluationReport.hybridMrrAt5,
        minHybridImprovement: options.minHybridImprovement,
      }).evidence,
    );
  } catch (error) {
    evidence.push({
      issue: 'gate execution error',
      message: error instanceof Error ? error.message : String(error),
    });
  }

  return createGateReport(evidence);
}

function createGateReport(evidence) {
  return {
    pass: evidence.length === 0,
    evidence,
    fixHint:
      evidence.length === 0
        ? null
        : 'Run: node scripts/semantic-index/download-model.mjs; node scripts/semantic-index/embed-index.mjs; node scripts/semantic-index/eval-embeddings.mjs --json',
    owner: OWNER,
  };
}

async function loadEvaluationReport(options) {
  const evaluationModule =
    await import('../../semantic-index/eval-embeddings.mjs');
  return evaluationModule.evaluateEmbeddings({
    alpha: options.alpha,
    corpusDatabasePath: options.corpusDatabasePath,
    modelId: options.modelId,
    queryFilePath: options.queryFilePath,
  });
}

function parseArgs(argv) {
  const flags = { json: false };

  for (const argument of argv) {
    if (argument === '--json') flags.json = true;
    else if (argument === '--help' || argument === '-h') flags.help = true;
    else if (argument.startsWith('--database='))
      flags.corpusDatabasePath = argument.slice('--database='.length);
    else if (argument.startsWith('--model-directory='))
      flags.modelDirectory = argument.slice('--model-directory='.length);
    else if (argument.startsWith('--model-id='))
      flags.modelId = argument.slice('--model-id='.length);
    else if (argument.startsWith('--query-file='))
      flags.queryFilePath = argument.slice('--query-file='.length);
    else if (argument.startsWith('--min-hybrid-improvement='))
      flags.minHybridImprovement = argument.slice(
        '--min-hybrid-improvement='.length,
      );
  }

  return flags;
}

function printUsage() {
  console.log(
    [
      'Cortex embeddings gate',
      '',
      'Usage:',
      '  node scripts/agent-customization/gates/cortex-embeddings.gate.mjs [--json]',
      '  node scripts/agent-customization/gates/cortex-embeddings.gate.mjs --help',
      '',
      'Options:',
      '  --json                          Emit the standard gate JSON contract.',
      '  --database=<path>               Override the corpus database path.',
      '  --model-directory=<path>        Override the ONNX model cache directory.',
      '  --model-id=<id>                 Restrict validation to one model id.',
      '  --query-file=<path>             Override the eval query set path.',
      '  --min-hybrid-improvement=<n>    Override the hybrid MRR@5 minimum improvement.',
    ].join('\n'),
  );
}

async function main() {
  const options = parseArgs(process.argv.slice(2));
  if (options.help) {
    printUsage();
    return;
  }

  const report = await runCortexEmbeddingsGate(options);
  console.log(
    options.json
      ? JSON.stringify(report, null, 2)
      : `${report.pass ? 'PASS' : 'FAIL'} cortex-embeddings.gate`,
  );
  if (!report.pass) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();
