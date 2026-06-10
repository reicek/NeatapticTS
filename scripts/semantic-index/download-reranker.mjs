/**
 * @description Download the `cross-encoder/ms-marco-MiniLM-L-6-v2` ONNX
 * cross-encoder model and its required tokenizer assets from Hugging Face
 * to the local model cache at `scripts/semantic-index/models/reranker/`.
 * Verifies the SHA-256 of `model.onnx` against Hugging Face metadata and
 * writes a `model-meta.json` sidecar with
 * `{ model_id, model_sha256, repository_id, max_sequence_length, downloaded_at }`.
 *
 * Unlike the bi-encoder `download-model.mjs`, this script does NOT write a
 * `dimension` field because cross-encoders output a scalar relevance score,
 * not a vector embedding.
 *
 * @param {boolean} [--json]                         - Emit JSON summary.
 * @param {string}  [--model-directory <path>]       - Override local model cache directory.
 * @param {string}  [--model-id <id>]                - Override local model identifier.
 * @param {string}  [--repository-id <id>]            - Override Hugging Face repository id.
 * @param {string}  [--expected-sha256 <h>]            - Override expected SHA-256 for model.onnx.
 * @param {number}  [--max-sequence-length <n>]        - Override max_sequence_length written to model-meta.json.
 * @param {boolean} [--help]                           - Show help and exit.
 *
 * @returns {void} Exits 0 on success, 1 on download or verification failure.
 */
import { createHash } from 'node:crypto';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  fail,
  parseCliArgs,
  printHelp,
  writeJsonOrText,
} from './cli-utils.mjs';
import { DEFAULT_RERANKER_MODEL_DIRECTORY } from './reranker-readiness.mjs';

export const DEFAULT_RERANKER_REPOSITORY_ID =
  'cross-encoder/ms-marco-MiniLM-L-6-v2';
export const DEFAULT_RERANKER_MODEL_ID = 'cross-encoder/ms-marco-MiniLM-L-6-v2';
export const DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH = 512;

/**
 * Assets to download for the cross-encoder model.
 * `model.onnx` is verified against its SHA-256; tokenizer files are not.
 */
export const RERANKER_ASSETS = Object.freeze([
  {
    localName: 'model.onnx',
    remotePath: 'onnx/model.onnx',
    verifySha256: true,
    urlMode: 'resolve',
  },
  {
    localName: 'tokenizer.json',
    remotePath: 'tokenizer.json',
    verifySha256: false,
    urlMode: 'raw',
  },
  {
    localName: 'tokenizer_config.json',
    remotePath: 'tokenizer_config.json',
    verifySha256: false,
    urlMode: 'raw',
  },
  {
    localName: 'special_tokens_map.json',
    remotePath: 'special_tokens_map.json',
    verifySha256: false,
    urlMode: 'raw',
  },
]);

/**
 * Fields written to model-meta.json for the cross-encoder.
 * Note: `dimension` is deliberately excluded because cross-encoders
 * output a scalar relevance score, not a vector.
 */
export const RERANKER_META_FIELDS = Object.freeze([
  'model_id',
  'model_sha256',
  'repository_id',
  'max_sequence_length',
  'downloaded_at',
]);

const DEFAULT_DOWNLOAD_RETRIES = 3;

/**
 * Download the cross-encoder model assets from Hugging Face and verify model integrity.
 *
 * Downloads `model.onnx`, `tokenizer.json`, `tokenizer_config.json`, and
 * `special_tokens_map.json` to the specified model directory. Verifies the
 * SHA-256 of `model.onnx` against Hugging Face metadata or the
 * `--expected-sha256` override. Writes `model-meta.json` with model metadata
 * including `max_sequence_length` but NOT `dimension`.
 *
 * @param {{
 *   modelDirectory?: string,
 *   modelId?: string,
 *   repositoryId?: string,
 *   expectedSha256?: string,
 *   maxSequenceLength?: number,
 * }} [options] - Download configuration.
 * @returns {Promise<{ modelId: string, modelDirectory: string, modelSha256: string, assets: Array<{ file: string, sha256: string, verified: boolean }> }>} Download summary.
 */
export async function downloadRerankerAssets(options = {}) {
  const modelDirectory = path.resolve(
    options.modelDirectory ?? DEFAULT_RERANKER_MODEL_DIRECTORY,
  );
  const repositoryId = String(
    options.repositoryId ?? DEFAULT_RERANKER_REPOSITORY_ID,
  );
  const modelId = String(options.modelId ?? DEFAULT_RERANKER_MODEL_ID);
  const maxSequenceLength = Number(
    options.maxSequenceLength ?? DEFAULT_RERANKER_MAX_SEQUENCE_LENGTH,
  );

  await mkdir(modelDirectory, { recursive: true });

  // Resolve expected SHA-256 from Hugging Face metadata
  const repositoryMetadata = await fetchJson(
    `https://huggingface.co/api/models/${repositoryId}`,
  );
  const modelSibling =
    repositoryMetadata?.siblings?.find?.(
      ({ rfilename }) => rfilename === 'onnx/model.onnx',
    ) ?? null;
  let expectedModelSha256 = String(
    options.expectedSha256 ?? modelSibling?.lfs?.oid ?? '',
  );
  if (!expectedModelSha256) {
    expectedModelSha256 = await resolveExpectedModelSha256FromPointer(
      repositoryId,
      'onnx/model.onnx',
    );
  }

  const assetReports = [];
  for (const asset of RERANKER_ASSETS) {
    const assetUrl = createAssetUrl(
      repositoryId,
      asset.remotePath,
      asset.urlMode,
    );
    const assetDownload = asset.verifySha256
      ? await downloadBinaryWithMetadata(assetUrl)
      : { buffer: await downloadBinary(assetUrl), finalUrl: assetUrl };
    const assetBuffer = assetDownload.buffer;
    const targetPath = path.join(modelDirectory, asset.localName);
    const assetSha256 = createSha256(assetBuffer);

    if (asset.verifySha256 && !expectedModelSha256) {
      throw new Error(
        'Unable to resolve the expected SHA-256 for onnx/model.onnx from the Hugging Face metadata or raw Git LFS pointer.',
      );
    }
    if (asset.verifySha256 && assetSha256 !== expectedModelSha256) {
      throw new Error(
        `SHA-256 mismatch for ${asset.localName}: expected ${expectedModelSha256}, received ${assetSha256}.`,
      );
    }

    await writeFile(targetPath, assetBuffer);
    assetReports.push({
      file: asset.localName,
      sha256: assetSha256,
      verified: asset.verifySha256,
    });
  }

  // Write model-meta.json WITHOUT dimension (cross-encoders output scalar)
  const modelMeta = {
    downloaded_at: new Date().toISOString(),
    max_sequence_length: maxSequenceLength,
    model_id: modelId,
    model_sha256: expectedModelSha256,
    repository_id: repositoryId,
  };
  await writeFile(
    path.join(modelDirectory, 'model-meta.json'),
    `${JSON.stringify(modelMeta, null, 2)}\n`,
  );

  return {
    assets: assetReports,
    maxSequenceLength,
    modelDirectory,
    modelId,
    modelSha256: expectedModelSha256,
  };
}

function createSha256(buffer) {
  return createHash('sha256').update(buffer).digest('hex');
}

function createAssetUrl(repositoryId, remotePath, urlMode) {
  const normalizedMode = urlMode === 'raw' ? 'raw' : 'resolve';
  return `https://huggingface.co/${repositoryId}/${normalizedMode}/main/${remotePath}`;
}

async function fetchJson(url) {
  const response = await fetchWithRetry(url);
  if (!response.ok)
    throw new Error(
      `Failed to fetch ${url}: ${response.status} ${response.statusText}`,
    );
  return response.json();
}

async function downloadBinary(url) {
  const response = await fetchWithRetry(url);
  if (!response.ok)
    throw new Error(
      `Failed to download ${url}: ${response.status} ${response.statusText}`,
    );
  return Buffer.from(await response.arrayBuffer());
}

async function downloadBinaryWithMetadata(url) {
  const response = await fetchWithRetry(url);
  if (!response.ok)
    throw new Error(
      `Failed to download ${url}: ${response.status} ${response.statusText}`,
    );
  return {
    buffer: Buffer.from(await response.arrayBuffer()),
    finalUrl: response.url,
  };
}

async function fetchWithRetry(url, retries = DEFAULT_DOWNLOAD_RETRIES) {
  let lastError = null;

  for (let attemptIndex = 0; attemptIndex < retries; attemptIndex += 1) {
    try {
      return await fetch(url);
    } catch (error) {
      lastError = error;
    }
  }

  throw lastError instanceof Error
    ? lastError
    : new Error(`Failed to fetch ${url}`);
}

async function resolveExpectedModelSha256FromPointer(repositoryId, remotePath) {
  const pointerUrl = `https://huggingface.co/${repositoryId}/raw/main/${remotePath}`;
  const response = await fetchWithRetry(pointerUrl);
  if (!response.ok) {
    throw new Error(
      `Failed to fetch ${pointerUrl}: ${response.status} ${response.statusText}`,
    );
  }

  const pointerText = await response.text();
  return pointerText.match(/oid sha256:([a-f0-9]{64})/iu)?.[1] ?? '';
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Cross-encoder reranker model downloader',
      usage: 'node scripts/semantic-index/download-reranker.mjs [--json]',
      options: [
        '--json                           Emit JSON summary.',
        '--model-directory <path>          Override the local model cache directory.',
        '--model-id <id>                   Override the local model identifier.',
        '--repository-id <id>              Override the Hugging Face repository id.',
        '--expected-sha256 <h>            Override the expected SHA-256 for model.onnx.',
        '--max-sequence-length <n>         Override max_sequence_length written to model-meta.json.',
        '--help                           Show this help.',
      ],
    });
    return;
  }

  try {
    const summary = await downloadRerankerAssets({
      expectedSha256: args['expected-sha256'],
      maxSequenceLength: args['max-sequence-length']
        ? Number(args['max-sequence-length'])
        : undefined,
      modelDirectory: args['model-directory'],
      modelId: args['model-id'],
      repositoryId: args['repository-id'],
    });
    writeJsonOrText(
      summary,
      Boolean(args.json),
      (payload) =>
        `Downloaded reranker model ${payload.modelId} to ${payload.modelDirectory}`,
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
