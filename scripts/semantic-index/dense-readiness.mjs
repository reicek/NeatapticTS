/**
 * @description Report whether dense search can run without degrading to BM25-only.
 * The readiness states are:
 * - `cold`: the ONNX model cache is absent.
 * - `model-only`: the model exists, but the embeddings database is absent or incomplete.
 * - `warm`: the model exists and the embedding count matches the corpus chunk count.
 *
 * `DENSE_FORCE_STATE` supports deterministic overrides for `cold` and `model-only`.
 * When either value is present, the probe returns the forced state immediately
 * without filesystem or database checks.
 *
 * @param {boolean} [--json] - Emit machine-readable readiness output.
 * @param {string}  [--database <path>] - Override the corpus database path.
 * @param {string}  [--embeddings-database <path>] - Override the embeddings database path.
 * @param {string}  [--model-directory <path>] - Override the local model directory.
 * @param {string}  [--model-id <id>] - Override the embedding model identifier.
 * @param {boolean} [--help] - Show help and exit.
 *
 * @returns {void} Exits 0 after reporting the current readiness state.
 */
import { existsSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { fail, parseCliArgs, printHelp, writeJsonOrText } from './cli-utils.mjs';
import {
  DEFAULT_EMBEDDINGS_DATABASE_PATH,
  DEFAULT_MODEL_DIRECTORY,
  DEFAULT_MODEL_ID,
} from './embed-index.mjs';
import { defaultDatabasePath } from './init-schema.mjs';
import { validateEmbeddings } from './validate-embeddings.mjs';

const FORCED_STATES = new Set(['cold', 'model-only']);

/**
 * Check whether the dense-search runtime is warm enough to serve default-on dense queries.
 *
 * @param {{
 *   corpusDatabasePath?: string,
 *   databasePath?: string,
 *   embeddingsDatabasePath?: string,
 *   modelDirectory?: string,
 *   modelId?: string,
 * }} [options] - Probe configuration.
 * @returns {Promise<{ ready: boolean, state: 'cold' | 'model-only' | 'warm', reason: string, chunk_count: number | null, embedding_count: number | null }>} Readiness report.
 */
export async function checkDenseReadiness(options = {}) {
  const forcedState = normalizeForcedState(process.env.DENSE_FORCE_STATE);
  if (forcedState) return createForcedReadinessReport(forcedState);

  const corpusDatabasePath = path.resolve(options.corpusDatabasePath ?? options.databasePath ?? defaultDatabasePath);
  const embeddingsDatabasePath = path.resolve(options.embeddingsDatabasePath ?? DEFAULT_EMBEDDINGS_DATABASE_PATH);
  const modelDirectory = path.resolve(options.modelDirectory ?? DEFAULT_MODEL_DIRECTORY);
  const modelId = String(options.modelId ?? DEFAULT_MODEL_ID);
  const modelPath = path.join(modelDirectory, 'model.onnx');

  if (!existsSync(modelPath)) {
    return {
      chunk_count: null,
      embedding_count: null,
      ready: false,
      reason: 'Dense model assets are absent.',
      state: 'cold',
    };
  }

  if (!existsSync(embeddingsDatabasePath)) {
    return {
      chunk_count: null,
      embedding_count: null,
      ready: false,
      reason: 'Dense model is present but the embeddings database is absent.',
      state: 'model-only',
    };
  }

  try {
    const validationReport = await validateEmbeddings({
      corpusDatabasePath,
      embeddingsDatabasePath,
      modelId,
    });
    const chunkCount = normalizeCount(validationReport.chunk_count);
    const embeddingCount = normalizeCount(validationReport.embedding_count);

    if (validationReport.pass) {
      return {
        chunk_count: chunkCount,
        embedding_count: embeddingCount,
        ready: true,
        reason: `All ${chunkCount ?? 0} chunks have embeddings.`,
        state: 'warm',
      };
    }

    return {
      chunk_count: chunkCount,
      embedding_count: embeddingCount,
      ready: false,
      reason: createModelOnlyReason(chunkCount, embeddingCount),
      state: 'model-only',
    };
  } catch (error) {
    return {
      chunk_count: null,
      embedding_count: null,
      ready: false,
      reason: `Dense embeddings could not be validated: ${error instanceof Error ? error.message : String(error)}`,
      state: 'model-only',
    };
  }
}

function normalizeForcedState(forcedState) {
  const trimmedState = typeof forcedState === 'string' ? forcedState.trim() : '';
  return FORCED_STATES.has(trimmedState) ? trimmedState : null;
}

function createForcedReadinessReport(state) {
  return {
    chunk_count: null,
    embedding_count: null,
    ready: false,
    reason: `DENSE_FORCE_STATE forced ${state} readiness.`,
    state,
  };
}

function normalizeCount(value) {
  const numericValue = Number(value);
  return Number.isFinite(numericValue) ? numericValue : null;
}

function createModelOnlyReason(chunkCount, embeddingCount) {
  if (chunkCount !== null && embeddingCount !== null) {
    return `Embeddings are incomplete: expected ${chunkCount}, found ${embeddingCount}.`;
  }

  return 'Dense model is present but embeddings are missing or incomplete.';
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));

  if (args.help) {
    printHelp({
      title: 'Dense readiness probe',
      usage: 'node scripts/semantic-index/dense-readiness.mjs [--json]',
      options: [
        '--json                       Emit the readiness report as JSON.',
        '--database <path>            Override the semantic-index corpus database path.',
        '--embeddings-database <p>    Override the embeddings database path.',
        '--model-directory <path>     Override the local dense model directory.',
        '--model-id <id>              Override the embedding model identifier.',
        '--help                       Show this help.',
      ],
    });
    return;
  }

  try {
    const report = await checkDenseReadiness({
      corpusDatabasePath: args.database,
      embeddingsDatabasePath: args['embeddings-database'],
      modelDirectory: args['model-directory'],
      modelId: args['model-id'],
    });
    writeJsonOrText(report, Boolean(args.json), (payload) => `${payload.state}: ${payload.reason}`);
  } catch (error) {
    fail(error instanceof Error ? error.message : String(error), Boolean(args.json));
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();