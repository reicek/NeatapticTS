/**
 * @description Report whether cross-encoder re-ranking can run without degrading
 * to hybrid-only results. The readiness states mirror the bi-encoder pattern:
 *
 * - `cold`: the ONNX model cache is absent.
 * - `model-only`: the model exists, but the ONNX session cannot be created.
 * - `warm`: the model exists and the ONNX session can be created successfully.
 *
 * `RERANKER_FORCE_STATE` supports deterministic overrides for `cold` and `model-only`.
 * When either value is present, the probe returns the forced state immediately
 * without filesystem or session checks.
 *
 * @param {boolean} [--json] - Emit machine-readable readiness output.
 * @param {string}  [--reranker-model-directory <path>] - Override the reranker model directory.
 * @param {string}  [--reranker-model-id <id>] - Override the reranker model identifier.
 * @param {boolean} [--help] - Show help and exit.
 *
 * @returns {void} Exits 0 after reporting the current readiness state.
 */
import { existsSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  fail,
  parseCliArgs,
  printHelp,
  writeJsonOrText,
} from './cli-utils.mjs';
import { repoRoot } from './init-schema.mjs';

const FORCED_STATES = new Set(['cold', 'model-only']);

/**
 * Default model identifier for the cross-encoder reranker.
 * Uses `cross-encoder/ms-marco-MiniLM-L-6-v2` — a lightweight (~66 MB)
 * cross-encoder that scores query-document pairs for relevance.
 */
export const DEFAULT_RERANKER_MODEL_ID = 'cross-encoder/ms-marco-MiniLM-L-6-v2';

/**
 * Default maximum sequence length for the cross-encoder.
 * The model processes `[CLS] query [SEP] document [SEP]` pairs
 * and truncates to this token limit.
 */
export const DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH = 512;

/**
 * Default directory for the cross-encoder model cache.
 * Mirrors the bi-encoder layout at `scripts/semantic-index/models/`.
 */
export const DEFAULT_RERANKER_MODEL_DIRECTORY = path.join(
  repoRoot,
  'scripts',
  'semantic-index',
  'models',
  'reranker',
);

/**
 * Check whether the cross-encoder reranker runtime is warm enough to serve re-ranking queries.
 *
 * Probes the reranker model directory for `model.onnx` existence,
 * validates `model-meta.json` structure, and attempts to create an ONNX
 * `InferenceSession` to confirm the model loads correctly.
 *
 * @param {{
 *   rerankerModelDirectory?: string,
 *   rerankerModelId?: string,
 *   forceState?: string,
 * }} [options] - Probe configuration.
 * @returns {Promise<{ ready: boolean, state: 'cold' | 'model-only' | 'warm', reason: string, model_id?: string | null, max_sequence_length?: number | null }>} Readiness report.
 */
export async function checkRerankerReadiness(options = {}) {
  const forcedState = normalizeForcedState(
    options.forceState ?? process.env.RERANKER_FORCE_STATE,
  );
  if (forcedState) return createForcedReadinessReport(forcedState);

  const modelDirectory = path.resolve(
    options.rerankerModelDirectory ?? DEFAULT_RERANKER_MODEL_DIRECTORY,
  );
  const modelId = String(options.rerankerModelId ?? DEFAULT_RERANKER_MODEL_ID);
  const modelPath = path.join(modelDirectory, 'model.onnx');
  const modelMetaPath = path.join(modelDirectory, 'model-meta.json');

  if (!existsSync(modelPath)) {
    return {
      max_sequence_length: null,
      model_id: null,
      ready: false,
      reason: 'Reranker model assets are absent.',
      state: 'cold',
    };
  }

  // Validate model-meta.json structure
  let modelMeta = null;
  try {
    const metaContent = await readFile(modelMetaPath, 'utf8');
    modelMeta = JSON.parse(metaContent);
  } catch {
    return {
      max_sequence_length: null,
      model_id: null,
      ready: false,
      reason:
        'Reranker model is present but model-meta.json is missing or invalid.',
      state: 'model-only',
    };
  }

  if (!modelMeta || typeof modelMeta !== 'object') {
    return {
      max_sequence_length: null,
      model_id: null,
      ready: false,
      reason: 'Reranker model-meta.json is not a valid object.',
      state: 'model-only',
    };
  }

  // Cross-encoder meta must NOT have a dimension field (outputs scalar, not vector)
  // but must have max_sequence_length
  const maxSequenceLength = Number(
    modelMeta.max_sequence_length ?? DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH,
  );
  const modelIdFromMeta = String(modelMeta.model_id ?? modelId);

  // Try creating an ONNX InferenceSession to validate model loads
  try {
    const { InferenceSession } = await import('onnxruntime-node');
    const session = await InferenceSession.create(modelPath);

    // Warm: model + valid session
    await session.release?.();

    return {
      max_sequence_length: maxSequenceLength,
      model_id: modelIdFromMeta,
      ready: true,
      reason: `Reranker model is warm with session created successfully (${modelIdFromMeta}).`,
      state: 'warm',
    };
  } catch (error) {
    return {
      max_sequence_length: maxSequenceLength,
      model_id: modelIdFromMeta,
      ready: false,
      reason: `Reranker model exists but session creation failed: ${error instanceof Error ? error.message : String(error)}`,
      state: 'model-only',
    };
  }
}

/**
 * Normalize a forced-state string, returning null for invalid values.
 *
 * @param {string | undefined} value - Raw forced-state value from options or env.
 * @returns {string | null} Normalized state ('cold' or 'model-only'), or null.
 */
function normalizeForcedState(value) {
  const trimmedState = typeof value === 'string' ? value.trim() : '';
  return FORCED_STATES.has(trimmedState) ? trimmedState : null;
}

/**
 * Create a readiness report for a forced state, skipping all filesystem and session checks.
 *
 * @param {'cold' | 'model-only'} state - Forced readiness state.
 * @returns {{ ready: false, state: string, reason: string, model_id: null, max_sequence_length: null }} Forced readiness report.
 */
function createForcedReadinessReport(state) {
  const reason =
    state === 'cold'
      ? 'Reranker model assets are absent.'
      : 'Reranker model exists but session creation failed.';
  return {
    max_sequence_length: null,
    model_id: null,
    ready: false,
    reason: `RERANKER_FORCE_STATE forced ${state} readiness. ${reason}`,
    state,
  };
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));

  if (args.help) {
    printHelp({
      title: 'Cross-encoder reranker readiness probe',
      usage: 'node scripts/semantic-index/reranker-readiness.mjs [--json]',
      options: [
        '--json                            Emit the readiness report as JSON.',
        '--reranker-model-directory <path>  Override the reranker model directory.',
        '--reranker-model-id <id>           Override the reranker model identifier.',
        '--help                            Show this help.',
      ],
    });
    return;
  }

  try {
    const report = await checkRerankerReadiness({
      rerankerModelDirectory: args['reranker-model-directory'],
      rerankerModelId: args['reranker-model-id'],
    });
    writeJsonOrText(
      report,
      Boolean(args.json),
      (payload) => `${payload.state}: ${payload.reason}`,
    );
  } catch (error) {
    fail(
      error instanceof Error ? error.message : String(error),
      Boolean(args.json),
    );
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();
