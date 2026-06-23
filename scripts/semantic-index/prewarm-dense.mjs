/**
 * @description Bootstrap the local dense-search runtime by ensuring the ONNX model
 * cache exists, building any missing embeddings in the consolidated
 * `data/turso-replica.sqlite` corpus DB, and validating that usable embeddings
 * are present. The script is idempotent: when `model.onnx` is already present the
 * download step is skipped, `embed-index.mjs` reuses its existing skip-unchanged
 * behavior, and validation is read-only.
 *
 * @param {boolean} [--dry-run] - Log the planned bootstrap steps without spawning subprocesses.
 * @param {boolean} [--json] - Emit machine-readable success or failure output.
 * @param {boolean} [--help] - Show help and exit.
 *
 * @returns {void} Exits 0 when all required steps succeed, 1 when any step fails.
 */
import { spawnSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { parseCliArgs, printHelp, writeJsonOrText } from './cli-utils.mjs';
import { DEFAULT_MODEL_DIRECTORY } from './embed-index.mjs';
import { DEFAULT_RERANKER_MODEL_DIRECTORY } from './reranker-readiness.mjs';
import { repoRoot } from './init-schema.mjs';

const MODEL_PATH = path.join(DEFAULT_MODEL_DIRECTORY, 'model.onnx');
const RERANKER_MODEL_PATH = path.join(
  DEFAULT_RERANKER_MODEL_DIRECTORY,
  'model.onnx',
);
const STEP_DEFINITIONS = Object.freeze([
  {
    args: [],
    name: 'download-model',
    scriptPath: path.join(
      repoRoot,
      'scripts',
      'semantic-index',
      'download-model.mjs',
    ),
  },
  {
    args: [],
    name: 'embed-index',
    scriptPath: path.join(
      repoRoot,
      'scripts',
      'semantic-index',
      'embed-index.mjs',
    ),
  },
  {
    args: ['--json'],
    name: 'validate-embeddings',
    scriptPath: path.join(
      repoRoot,
      'scripts',
      'semantic-index',
      'validate-embeddings.mjs',
    ),
  },
  {
    args: [],
    name: 'download-reranker',
    scriptPath: path.join(
      repoRoot,
      'scripts',
      'semantic-index',
      'download-reranker.mjs',
    ),
  },
  {
    args: ['--json'],
    name: 'validate-reranker',
    scriptPath: path.join(
      repoRoot,
      'scripts',
      'semantic-index',
      'reranker-readiness.mjs',
    ),
  },
]);

/**
 * Run the dense prewarm bootstrap with injectable side effects for tests.
 *
 * @param {{
 *   commandRunner?: (step: { name: string, scriptPath: string, args: string[] }) => { status?: number | null, stderr?: string, stdout?: string },
 *   dryRun?: boolean,
 *   json?: boolean,
 *   logger?: (line: string) => void,
 *   modelExists?: () => boolean,
 *   rerankerModelExists?: () => boolean,
 * }} [options] - Runtime options and test doubles.
 * @returns {Promise<{ exitCode: number, report: { pass: boolean, steps: Array<{ name: string, status: 'ok' | 'skipped' }> } | { pass: false, failedStep: string, error: string } }>} Execution summary.
 */
export async function runDensePrewarm(options = {}) {
  const logger = options.logger ?? console.log;
  const dryRun = Boolean(options.dryRun);
  const modelExists = options.modelExists ?? (() => existsSync(MODEL_PATH));
  const rerankerModelExists =
    options.rerankerModelExists ?? (() => existsSync(RERANKER_MODEL_PATH));
  const commandRunner = options.commandRunner ?? runBootstrapStep;
  const steps = [];

  for (const stepDefinition of STEP_DEFINITIONS) {
    const stepShouldSkip =
      (stepDefinition.name === 'download-model' && modelExists()) ||
      (stepDefinition.name === 'download-reranker' && rerankerModelExists());

    if (dryRun) {
      logger(formatDryRunMessage(stepDefinition, stepShouldSkip));
      continue;
    }

    if (stepShouldSkip) {
      logger(`${stepDefinition.name}: model present, skipping download`);
      steps.push({ name: stepDefinition.name, status: 'skipped' });
      continue;
    }

    logger(`${stepDefinition.name}: running`);
    const stepResult = commandRunner(stepDefinition);
    const exitStatus = Number(stepResult?.status ?? 1);

    if (exitStatus !== 0) {
      const errorMessage = createErrorMessage(stepDefinition.name, stepResult);
      return {
        exitCode: 1,
        report: {
          error: errorMessage,
          failedStep: stepDefinition.name,
          pass: false,
        },
      };
    }

    steps.push({ name: stepDefinition.name, status: 'ok' });
  }

  return {
    exitCode: 0,
    report: {
      pass: true,
      steps,
    },
  };
}

function formatDryRunMessage(stepDefinition, stepShouldSkip) {
  if (stepShouldSkip)
    return `${stepDefinition.name}: model present, skipping download`;
  return `${stepDefinition.name}: dry-run`;
}

function runBootstrapStep(step) {
  return spawnSync(process.execPath, [step.scriptPath, ...step.args], {
    cwd: repoRoot,
    encoding: 'utf8',
  });
}

function createErrorMessage(stepName, stepResult) {
  const stderrText = String(stepResult?.stderr ?? '').trim();
  const stdoutText = String(stepResult?.stdout ?? '').trim();
  return (
    stderrText || stdoutText || `${stepName} exited with a non-zero status.`
  );
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));

  if (args.help) {
    printHelp({
      title: 'Dense prewarm bootstrap',
      usage:
        'node scripts/semantic-index/prewarm-dense.mjs [--dry-run] [--json]',
      options: [
        '--dry-run  Log the planned bootstrap steps without spawning subprocesses.',
        '--json     Emit a machine-readable success or failure summary.',
        '--help     Show this help.',
      ],
    });
    return;
  }

  const result = await runDensePrewarm({
    dryRun: Boolean(args['dry-run']),
    json: Boolean(args.json),
  });

  writeJsonOrText(result.report, Boolean(args.json), (payload) =>
    payload.pass
      ? payload.steps.map((step) => `${step.name}: ${step.status}`).join('\n')
      : `${payload.failedStep}: ${payload.error}`,
  );

  if (result.exitCode !== 0) process.exitCode = result.exitCode;
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();
