/**
 * @description Standard gate wrapper for dense-search readiness. The gate imports the
 * local readiness probe directly, reports `pass: true` only for the `warm` state,
 * and otherwise points operators to the prewarm bootstrap command.
 *
 * @param {boolean} [--json] - Emit the standard gate JSON contract.
 * @param {string}  [--database <path>] - Override the corpus database path.
 * @param {string}  [--model-directory <path>] - Override the local model directory.
 * @param {string}  [--model-id <id>] - Override the embedding model identifier.
 * @param {boolean} [--help] - Show help and exit.
 *
 * @returns {void} Exits 0 when the dense runtime is warm, 1 otherwise.
 */
import { pathToFileURL } from 'node:url';
import {
  parseCliArgs,
  printHelp,
  writeJsonOrText,
} from '../../../rag-index/cli-utils.mjs';
import { checkDenseReadiness } from '../../../rag-index/dense-readiness.mjs';

const FIX_HINT = 'Run `npm run index:prewarm` to build the embedding index.';
const OWNER = '01-planning';

/**
 * Evaluate the dense readiness gate.
 *
 * @param {{
 *   corpusDatabasePath?: string,
 *   modelDirectory?: string,
 *   modelId?: string,
 *   readinessProbe?: (options?: Record<string, unknown>) => Promise<{ state: 'cold' | 'model-only' | 'warm', chunk_count?: number | null, embedding_count?: number | null }>,
 * }} [options] - Gate options and test doubles.
 * @returns {Promise<{ pass: boolean, evidence: { state: 'cold' | 'model-only' } | { state: 'warm', chunk_count: number | null, embedding_count: number | null }, fixHint: string | null, owner: string }>} Gate result.
 */
export async function evaluateDenseReadinessGate(options = {}) {
  const readinessProbe = options.readinessProbe ?? checkDenseReadiness;
  const readinessReport = await readinessProbe(options);

  if (readinessReport.state === 'warm') {
    return {
      evidence: {
        chunk_count: readinessReport.chunk_count ?? null,
        embedding_count: readinessReport.embedding_count ?? null,
        state: 'warm',
      },
      fixHint: null,
      owner: OWNER,
      pass: true,
    };
  }

  return {
    evidence: { state: readinessReport.state },
    fixHint: FIX_HINT,
    owner: OWNER,
    pass: false,
  };
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));

  if (args.help) {
    printHelp({
      title: 'Dense readiness gate',
      usage:
        'node scripts/agent-customization/gates/dense-readiness.gate.mjs [--json]',
      options: [
        '--json                       Emit the standard gate JSON contract.',
        '--database <path>            Override the semantic-index corpus database path.',
        '--model-directory <path>     Override the local dense model directory.',
        '--model-id <id>              Override the embedding model identifier.',
        '--help                       Show this help.',
      ],
    });
    return;
  }

  const report = await evaluateDenseReadinessGate({
    corpusDatabasePath: args.database,
    modelDirectory: args['model-directory'],
    modelId: args['model-id'],
  });
  writeJsonOrText(
    report,
    Boolean(args.json),
    (payload) => `${payload.pass ? 'PASS' : 'FAIL'} dense-readiness.gate`,
  );
  if (!report.pass) process.exitCode = 1;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();
