/**
 * @description Build or incrementally update the dense embedding index stored in
 * the consolidated `chunks.embedding` column of `rag-index/data/turso-replica.sqlite`.
 * Reads all chunks, runs each chunk body through the locally cached
 * `all-MiniLM-L6-v2` ONNX model (mean-pool → L2-normalize → 384-dim float32),
 * and stores vectors as F8_BLOB via Turso's `vector8()` function.
 * Skips chunks whose `chunk_sha256` and `model_id` are unchanged (incremental
 * rule). Run `download-model.mjs` once before this script.
 *
 * @param {boolean} [--dry-run]                     - Count queued chunks without writing embeddings.
 * @param {boolean} [--json]                         - Emit JSON summary `{ embedded, skipped, queued, dryRun }`.
 * @param {string}  [--database <path>]              - Override corpus database path.
 * @param {string}  [--model-directory <path>]       - Override local model cache directory.
 * @param {string}  [--model-id <id>]                - Override model identifier.
 * @param {number}  [--dimension <n>]                - Override embedding dimension.
 * @param {string}  [--model-sha256 <hex>]           - Override model SHA-256 value.
 * @param {boolean} [--help]                         - Show help and exit.
 *
 * @returns {void} Exits 0 on success, 1 on error. JSON summary written to stdout when `--json` is passed.
 */
import { createClient } from '@libsql/client';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  fail,
  parseCliArgs,
  printHelp,
  writeJsonOrText,
} from './cli-utils.mjs';
import { defaultDatabasePath, repoRoot } from './init-schema.mjs';

export const DEFAULT_MODEL_ID = 'all-MiniLM-L6-v2';
export const DEFAULT_MODEL_DIRECTORY = path.join(
  repoRoot,
  'rag-index',
  'models',
);
const DEFAULT_MAX_SEQUENCE_LENGTH = 512;

/** Maximum number of SQL statements per client.batch() call. */
const BATCH_SIZE = 1000;

export async function buildEmbeddingIndex(options = {}) {
  const corpusDatabasePath = path.resolve(
    options.corpusDatabasePath ?? options.databasePath ?? defaultDatabasePath,
  );
  const modelId = String(options.modelId ?? DEFAULT_MODEL_ID);
  const dryRun = Boolean(options.dryRun);
  const modelMeta = await readModelMeta(options);
  const dimension = Number(options.dimension ?? modelMeta.dimension ?? 0);
  const modelSha256 = String(
    options.modelSha256 ?? modelMeta.model_sha256 ?? '',
  );

  if (!Number.isInteger(dimension) || dimension < 1) {
    throw new Error(
      'Embedding dimension is required. Pass --dimension or provide rag-index/models/model-meta.json.',
    );
  }

  if (!modelSha256) {
    throw new Error(
      'Model SHA-256 is required. Pass --model-sha256 or provide rag-index/models/model-meta.json.',
    );
  }

  const embedText =
    options.embedText ??
    (await createOnnxTextEmbedder({
      dimension,
      modelDirectory: options.modelDirectory ?? DEFAULT_MODEL_DIRECTORY,
      modelId,
    }));

  const client =
    options.client ??
    createClient({ url: pathToFileURL(corpusDatabasePath).href });
  return buildEmbeddingIndexWithClient({
    client,
    embedText,
    modelId,
    modelSha256,
    dimension,
    dryRun,
  });
}

/**
 * Async client-based embedding index builder.
 *
 * Reads chunks from the client, computes embeddings, and writes them to
 * the `chunks.embedding` column (consolidated Turso schema) via UPDATE.
 *
 * @param {object} params - Build parameters.
 * @param {import('@libsql/client').Client} params.client - libSQL client.
 * @param {Function} params.embedText - Embedding function.
 * @param {string} params.modelId - Model identifier.
 * @param {string} params.modelSha256 - Model SHA-256 hash.
 * @param {number} params.dimension - Embedding dimension.
 * @param {boolean} params.dryRun - Skip writing if true.
 * @returns {Promise<object>} Summary with embedded/skipped/queued counts.
 */
async function buildEmbeddingIndexWithClient({
  client,
  embedText,
  modelId,
  modelSha256,
  dimension,
  dryRun,
}) {
  const summary = {
    dryRun,
    embedded: 0,
    modelId,
    purged: 0,
    queued: 0,
    skipped: 0,
  };

  const chunkRowsResult = await client.execute({
    sql: `
      SELECT c.chunk_id, c.chunk_index, c.heading_path, c.body_text, c.char_start, c.char_end,
        c.parent_chunk_id, c.depth, c.context_header, c.symbol_name, c.signature_text,
        c.jsdoc_text, c.export_type, c.module_path,
        d.file_path, d.doc_family
      FROM chunks c
      LEFT JOIN documents d ON d.doc_id = c.doc_id
      ORDER BY c.chunk_id
    `,
    args: [],
  });
  const chunkRows = chunkRowsResult.rows;
  const pendingUpdates = [];

  for (const chunkRow of chunkRows) {
    const chunkSha256 = createChunkSha256(chunkRow);

    // Check if embedding is already up-to-date.
    const existingResult = await client.execute({
      sql: 'SELECT chunk_sha256, embedding_model FROM chunks WHERE chunk_id = ?',
      args: [chunkRow.chunk_id],
    });
    const existing = existingResult.rows[0];
    if (
      existing?.chunk_sha256 === chunkSha256 &&
      existing?.embedding_model === modelId
    ) {
      summary.skipped += 1;
      continue;
    }

    if (dryRun) {
      summary.queued += 1;
      continue;
    }

    const embeddingVector = normalizeEmbeddingVector(
      await embedText({
        chunk: chunkRow,
        chunkId: Number(chunkRow.chunk_id),
        dimension,
        filePath: chunkRow.file_path ?? null,
        headingPath: chunkRow.heading_path ?? null,
        modelId,
        text: chunkRow.body_text,
      }),
      dimension,
    );

    pendingUpdates.push({
      sql: 'UPDATE chunks SET embedding = vector8(?), embedding_model = ?, chunk_sha256 = ?, embedded_at = ? WHERE chunk_id = ?',
      args: [
        Buffer.from(
          embeddingVector.buffer,
          embeddingVector.byteOffset,
          embeddingVector.byteLength,
        ),
        modelId,
        chunkSha256,
        Date.now(),
        chunkRow.chunk_id,
      ],
    });

    summary.embedded += 1;

    // Flush accumulated updates in batches of BATCH_SIZE.
    if (pendingUpdates.length >= BATCH_SIZE) {
      const batchSlice = pendingUpdates.splice(0, BATCH_SIZE);
      await client.batch(batchSlice, 'write');
    }
  }

  // Flush any remaining pending updates.
  if (pendingUpdates.length > 0) {
    await client.batch(pendingUpdates.splice(0), 'write');
  }

  await releaseEmbedText(embedText);
  return summary;
}

export function normalizeEmbeddingVector(vectorLike, dimension) {
  const float32Vector = toFloat32Array(vectorLike);
  if (float32Vector.length !== dimension) {
    throw new Error(
      `Expected embedding dimension ${dimension}, received ${float32Vector.length}.`,
    );
  }

  let magnitudeSquared = 0;
  for (const value of float32Vector) magnitudeSquared += value * value;

  if (magnitudeSquared === 0) return new Float32Array(float32Vector);

  const magnitude = Math.sqrt(magnitudeSquared);
  const normalizedVector = new Float32Array(float32Vector.length);
  for (let valueIndex = 0; valueIndex < float32Vector.length; valueIndex += 1) {
    normalizedVector[valueIndex] = float32Vector[valueIndex] / magnitude;
  }
  return normalizedVector;
}

function createChunkSha256(chunkRow) {
  return createHash('sha256')
    .update(
      JSON.stringify({
        body_text: chunkRow.body_text,
        char_end: Number(chunkRow.char_end),
        char_start: Number(chunkRow.char_start),
        chunk_id: Number(chunkRow.chunk_id),
        chunk_index: Number(chunkRow.chunk_index),
        context_header: chunkRow.context_header ?? null,
        depth: Number(chunkRow.depth ?? 0),
        doc_family: chunkRow.doc_family ?? null,
        file_path: chunkRow.file_path ?? null,
        heading_path: chunkRow.heading_path ?? null,
        symbol_name: chunkRow.symbol_name ?? null,
      }),
    )
    .digest('hex');
}

export async function readModelMeta(options = {}) {
  if (options.modelMeta) return options.modelMeta;

  const modelMetaPath = path.resolve(
    options.modelMetaPath ??
      path.join(
        options.modelDirectory ?? DEFAULT_MODEL_DIRECTORY,
        'model-meta.json',
      ),
  );
  try {
    return JSON.parse(await readFile(modelMetaPath, 'utf8'));
  } catch (error) {
    if (
      error &&
      typeof error === 'object' &&
      'code' in error &&
      error.code === 'ENOENT'
    ) {
      return {};
    }
    throw error;
  }
}

export async function createOnnxTextEmbedder(options = {}) {
  const { Tokenizer } = await import('@huggingface/tokenizers');
  const { InferenceSession, Tensor } = await import('onnxruntime-node');
  const modelDirectory = path.resolve(
    options.modelDirectory ?? DEFAULT_MODEL_DIRECTORY,
  );
  const modelPath = path.join(modelDirectory, 'model.onnx');
  const tokenizer = await createWordPieceTokenizer({
    modelDirectory,
    Tokenizer,
  });
  const session = await InferenceSession.create(modelPath);
  const [clsTokenId, sepTokenId] = ['[CLS]', '[SEP]'].map((token) =>
    tokenizer.token_to_id(token),
  );
  const maxSequenceLength = Number(
    options.maxSequenceLength ?? DEFAULT_MAX_SEQUENCE_LENGTH,
  );
  if (clsTokenId === undefined || sepTokenId === undefined) {
    throw new Error(
      'The local tokenizer vocabulary is missing the required [CLS] or [SEP] tokens.',
    );
  }

  const embedText = async ({ text }) => {
    const encoded = tokenizer.encode(String(text ?? ''), {
      add_special_tokens: false,
      return_token_type_ids: true,
    });
    const truncatedTokenIds = encoded.ids.slice(
      0,
      Math.max(0, maxSequenceLength - 2),
    );
    const inputIds = [clsTokenId, ...truncatedTokenIds, sepTokenId];
    const attentionMaskValues = inputIds.map(() => 1);
    const tokenTypeIdValues = inputIds.map(() => 0);
    const tokenIds = BigInt64Array.from(
      inputIds.map((tokenId) => BigInt(tokenId)),
    );
    const attentionMask = BigInt64Array.from(
      attentionMaskValues.map((maskValue) => BigInt(maskValue)),
    );
    const tokenTypeIds = BigInt64Array.from(
      tokenTypeIdValues.map((tokenTypeId) => BigInt(tokenTypeId)),
    );
    const sequenceLength = inputIds.length;
    const feeds = {
      attention_mask: new Tensor('int64', attentionMask, [1, sequenceLength]),
      input_ids: new Tensor('int64', tokenIds, [1, sequenceLength]),
    };

    if (session.inputNames.includes('token_type_ids')) {
      feeds.token_type_ids = new Tensor('int64', tokenTypeIds, [
        1,
        sequenceLength,
      ]);
    }

    const outputs = await session.run(feeds);
    const outputName =
      session.outputNames.find((name) => name === 'last_hidden_state') ??
      session.outputNames[0];
    if (!outputName || !outputs[outputName]) {
      throw new Error(
        'ONNX embedding session did not return last_hidden_state output.',
      );
    }

    return meanPoolEmbedding(
      outputs[outputName],
      attentionMask,
      options.dimension,
    );
  };

  embedText.release = async () => {
    await session.release?.();
  };

  return embedText;
}

async function createWordPieceTokenizer({ Tokenizer, modelDirectory }) {
  const tokenizerJsonPath = path.join(modelDirectory, 'tokenizer.json');
  const tokenizerConfigPath = path.join(
    modelDirectory,
    'tokenizer_config.json',
  );
  const specialTokensMapPath = path.join(
    modelDirectory,
    'special_tokens_map.json',
  );
  const [tokenizerJson, tokenizerConfig, specialTokensMap] = await Promise.all([
    readJsonFile(tokenizerJsonPath, null),
    readJsonFile(tokenizerConfigPath, {}),
    readJsonFile(specialTokensMapPath, {}),
  ]);
  const vocabulary = tokenizerJson?.model?.vocab;
  if (!vocabulary || typeof vocabulary !== 'object') {
    throw new Error(
      'tokenizer.json is missing the WordPiece vocabulary required for local tokenization.',
    );
  }
  const unkToken = resolveSpecialToken(specialTokensMap.unk_token, '[UNK]');
  const tokenizer = new Tokenizer(
    {
      added_tokens: [],
      decoder: null,
      model: {
        type: 'WordPiece',
        vocab: vocabulary,
        unk_token: unkToken,
        continuing_subword_prefix: '##',
        max_input_chars_per_word: 100,
      },
      normalizer: {
        type: 'BertNormalizer',
        clean_text: true,
        handle_chinese_chars: true,
        lowercase: Boolean(tokenizerConfig.do_lower_case ?? true),
        strip_accents: tokenizerConfig.strip_accents ?? true,
      },
      post_processor: null,
      pre_tokenizer: { type: 'BertPreTokenizer' },
    },
    {
      clean_up_tokenization_spaces: true,
    },
  );

  return tokenizer;
}

function meanPoolEmbedding(lastHiddenStateTensor, attentionMask, dimension) {
  const outputData = toFloat32Array(lastHiddenStateTensor.data);
  const normalizedDimension = Number(dimension);
  if (!Number.isInteger(normalizedDimension) || normalizedDimension < 1) {
    throw new Error(
      'A positive embedding dimension is required for mean pooling.',
    );
  }

  const pooledVector = new Float32Array(normalizedDimension);
  let includedTokenCount = 0;

  for (let tokenIndex = 0; tokenIndex < attentionMask.length; tokenIndex += 1) {
    if (Number(attentionMask[tokenIndex]) === 0) continue;
    includedTokenCount += 1;
    const tokenOffset = tokenIndex * normalizedDimension;
    for (
      let dimensionIndex = 0;
      dimensionIndex < normalizedDimension;
      dimensionIndex += 1
    ) {
      pooledVector[dimensionIndex] += outputData[tokenOffset + dimensionIndex];
    }
  }

  if (includedTokenCount === 0) return pooledVector;

  for (
    let dimensionIndex = 0;
    dimensionIndex < normalizedDimension;
    dimensionIndex += 1
  ) {
    pooledVector[dimensionIndex] /= includedTokenCount;
  }
  return pooledVector;
}

/**
 * Quantize a Float32 embedding vector to an F8 (int8) BLOB buffer.
 *
 * Each float32 value in the range [-1, 1] is mapped to an int8 value in
 * [-127, 127]. Values outside [-1, 1] are clamped. The result is a Buffer
 * of length equal to the input vector length (384 bytes for 384-dim),
 * achieving 4x compression versus the 1536-byte Float32 representation.
 *
 * @param {Float32Array} float32Vector - L2-normalized embedding vector.
 * @returns {Buffer} Int8-quantized buffer (one byte per dimension).
 */
export function toF8BlobBuffer(float32Vector) {
  const buffer = Buffer.alloc(float32Vector.length);
  for (let i = 0; i < float32Vector.length; i += 1) {
    const clamped = Math.max(-1, Math.min(1, float32Vector[i]));
    buffer.writeInt8(Math.round(clamped * 127), i);
  }
  return buffer;
}

function toFloat32Array(vectorLike) {
  if (vectorLike instanceof Float32Array) return vectorLike;
  if (ArrayBuffer.isView(vectorLike)) {
    return new Float32Array(
      vectorLike.buffer.slice(
        vectorLike.byteOffset,
        vectorLike.byteOffset + vectorLike.byteLength,
      ),
    );
  }
  return Float32Array.from(Array.isArray(vectorLike) ? vectorLike : []);
}

async function releaseEmbedText(embedText) {
  if (typeof embedText?.release === 'function') await embedText.release();
}

async function readJsonFile(filePath, fallbackValue) {
  try {
    return JSON.parse(await readFile(filePath, 'utf8'));
  } catch (error) {
    if (
      error &&
      typeof error === 'object' &&
      'code' in error &&
      error.code === 'ENOENT'
    ) {
      return fallbackValue;
    }
    throw error;
  }
}

function resolveSpecialToken(value, fallbackToken) {
  if (typeof value === 'string' && value.trim()) return value;
  if (
    value &&
    typeof value === 'object' &&
    typeof value.content === 'string' &&
    value.content.trim()
  ) {
    return value.content;
  }
  return fallbackToken;
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Embedding index builder',
      usage: 'node rag-index/embed-index.mjs [--dry-run] [--json]',
      options: [
        '--dry-run                  Count work without writing embeddings.',
        '--json                     Emit JSON summary.',
        '--database <path>          Override the corpus database path.',
        '--model-directory <path>   Override the local model cache directory.',
        '--model-id <id>            Override the model identifier.',
        '--dimension <n>            Override the embedding dimension.',
        '--model-sha256 <hex>       Override the model SHA-256 value.',
        '--help                     Show this help.',
      ],
    });
    return;
  }

  try {
    const summary = await buildEmbeddingIndex({
      corpusDatabasePath: args.database,
      dimension: args.dimension,
      dryRun: Boolean(args['dry-run']),
      modelDirectory: args['model-directory'],
      modelId: args['model-id'],
      modelSha256: args['model-sha256'],
    });
    writeJsonOrText(summary, Boolean(args.json), (payload) =>
      payload.dryRun
        ? `Embedding build dry run: queued ${payload.queued}, skipped ${payload.skipped}`
        : `Embedding build complete: embedded ${payload.embedded}, skipped ${payload.skipped}`,
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
