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
 * @param {string}  [--files=<path>]                - Re-embed chunks for the specified repo-relative file path; repeatable. Run build-index.mjs first to re-chunk changed files.
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

/**
 * Default sentence-transformer model identifier used when no model is
 * configured on the CLI or in `options.modelId`.
 */
export const DEFAULT_MODEL_ID = 'all-MiniLM-L6-v2';

/**
 * Default local directory that caches the ONNX tokenizer and model files.
 * Used when `options.modelDirectory` is not provided.
 */
export const DEFAULT_MODEL_DIRECTORY = path.join(
  repoRoot,
  'rag-index',
  'models',
);
const DEFAULT_MAX_SEQUENCE_LENGTH = 512;

/** Maximum number of SQL statements per client.batch() call. */
const BATCH_SIZE = 1000;

/**
 * Build or incrementally update the dense embedding index in the consolidated
 * `chunks` table, including step-packet slice metadata for plan-family chunks.
 *
 * Reads every chunk, computes an embedding with the configured embedder, and
 * writes the quantized vector to `chunks.embedding`. It also parses step-packet
 * YAML for `plan`-family chunks and persists `slice_id`, `step_number`,
 * `phase`, and `status` on each chunk row.
 *
 * @param {object} [options={}] - Build options.
 * @param {string} [options.corpusDatabasePath] - Override path to the corpus database.
 * @param {string} [options.databasePath] - Alias for `corpusDatabasePath`.
 * @param {string} [options.modelId] - Model identifier; defaults to {@link DEFAULT_MODEL_ID}.
 * @param {boolean} [options.dryRun] - Count queued chunks without writing.
 * @param {object} [options.modelMeta] - Pre-loaded model metadata object.
 * @param {string} [options.modelMetaPath] - Path to `model-meta.json`.
 * @param {string} [options.modelDirectory] - Directory containing `model.onnx`; defaults to {@link DEFAULT_MODEL_DIRECTORY}.
 * @param {number} [options.dimension] - Embedding dimension; falls back to `modelMeta.dimension`.
 * @param {string} [options.modelSha256] - Model SHA-256; falls back to `modelMeta.model_sha256`.
 * @param {string[]} [options.files] - Repo-relative file paths to limit re-indexing to. When provided, only chunks whose `file_path` matches one of these paths are embedded; all other chunks are skipped.
 * @param {Function} [options.embedText] - Override embedding function (used in tests).
 * @param {import('@libsql/client').Client} [options.client] - Existing libSQL client (used in tests).
 * @returns {Promise<object>} Summary with `embedded`, `skipped`, `queued`, `modelId`, `dryRun`, and `purged` counts.
 * @throws {Error} When `dimension` or `modelSha256` cannot be resolved.
 *
 * @example
 * ```js
 * const summary = await buildEmbeddingIndex({
 *   client,
 *   embedText: async ({ text }) => new Float32Array(384).fill(0.1),
 *   dimension: 384,
 *   modelSha256: 'fake-sha256',
 *   modelId: 'fake-model',
 * });
 * console.log(summary.embedded, summary.skipped);
 * ```
 */
export async function buildEmbeddingIndex(options) {
  options = /* istanbul ignore next -- defensive: options always provided in tests */ options ?? {};
  const corpusDatabasePath = path.resolve(
    /* istanbul ignore next -- defensive: corpusDatabasePath/databasePath always provided in tests */
    options.corpusDatabasePath ?? options.databasePath ?? defaultDatabasePath,
  );
  const modelId = String(options.modelId ?? DEFAULT_MODEL_ID);
  const dryRun = Boolean(options.dryRun);
  const modelMeta = await readModelMeta(options);
  /* istanbul ignore next -- defensive: fallback to modelMeta.dimension then 0 */
  const dimension = Number(options.dimension ?? modelMeta.dimension ?? 0);
  /* istanbul ignore next -- defensive: fallback to modelMeta.model_sha256 then '' */
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

  /* istanbul ignore next -- defensive: embedText fallback creates ONNX embedder (untestable in ESM vm) */
  const embedText =
    options.embedText ??
    (await createOnnxTextEmbedder({
      dimension,
      /* istanbul ignore next -- defensive: modelDirectory fallback */
      modelDirectory: options.modelDirectory ?? DEFAULT_MODEL_DIRECTORY,
      modelId,
    }));

  /* istanbul ignore next -- defensive: client fallback creates real SQLite connection */
  const client =
    options.client ??
    createClient({ url: pathToFileURL(corpusDatabasePath).href });
  return buildEmbeddingIndexWithClient({
    client,
    embedText,
    files: options.files,
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
  files,
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

  const targetFiles =
    Array.isArray(files) && files.length > 0 ? new Set(files) : null;

  if (targetFiles && targetFiles.size > 0) {
    const fileList = Array.from(targetFiles);
    const placeholders = fileList.map(() => '?').join(',');
    const matchResult = await client.execute({
      sql: `SELECT COUNT(*) as count FROM documents WHERE file_path IN (${placeholders})`,
      args: fileList,
    });
    /* istanbul ignore next -- defensive: rows[0] always exists for COUNT(*) query */
    if (Number(matchResult.rows[0]?.count ?? 0) === 0) {
      console.warn(`No matching documents found for: ${fileList.join(', ')}`);
    }
  }

  const chunkRowsResult = await client.execute({
    sql: `
      SELECT c.chunk_id, c.chunk_index, c.heading_path, c.body_text, c.char_start, c.char_end,
        c.parent_chunk_id, c.depth, c.context_header, c.symbol_name, c.signature_text,
        c.jsdoc_text, c.export_type, c.module_path,
        c.slice_id, c.step_number, c.phase, c.status,
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
    if (targetFiles && !targetFiles.has(chunkRow.file_path)) {
      summary.skipped += 1;
      continue;
    }

    const sliceMetadata = extractSliceMetadata(
      chunkRow.body_text,
      chunkRow.doc_family,
    );
    const chunkSha256 = createChunkSha256(chunkRow, sliceMetadata);

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
        filePath: /* istanbul ignore next -- defensive: file_path always present in test data */ chunkRow.file_path ?? null,
        headingPath: /* istanbul ignore next -- defensive: heading_path may be null in test data */ chunkRow.heading_path ?? null,
        modelId,
        text: chunkRow.body_text,
      }),
      dimension,
    );

    pendingUpdates.push({
      sql: 'UPDATE chunks SET embedding = vector8(?), embedding_model = ?, chunk_sha256 = ?, embedded_at = ?, slice_id = ?, step_number = ?, phase = ?, status = ? WHERE chunk_id = ?',
      args: [
        Buffer.from(
          embeddingVector.buffer,
          embeddingVector.byteOffset,
          embeddingVector.byteLength,
        ),
        modelId,
        chunkSha256,
        Date.now(),
        sliceMetadata.slice_id,
        sliceMetadata.step_number,
        sliceMetadata.phase,
        sliceMetadata.status,
        chunkRow.chunk_id,
      ],
    });

    summary.embedded += 1;

    // Flush accumulated updates in batches of BATCH_SIZE.
    /* istanbul ignore if -- requires 1000+ chunks to trigger batch flush */
    if (pendingUpdates.length >= BATCH_SIZE) {
      /* istanbul ignore next -- requires 1000+ chunks to trigger batch flush */
      const batchSlice = pendingUpdates.splice(0, BATCH_SIZE);
      await client.batch(batchSlice, 'write');
    }
  }

  // Flush any remaining pending updates.
  if (pendingUpdates.length > 0) {
    await client.batch(pendingUpdates.splice(0), 'write');
  }

  // Update freshness markers for every document targeted by this run.
  if (!dryRun && targetFiles && targetFiles.size > 0) {
    const fileList = Array.from(targetFiles);
    const placeholders = fileList.map(() => '?').join(',');
    await client.execute({
      sql: `UPDATE documents SET indexed_at = ? WHERE file_path IN (${placeholders})`,
      args: [Date.now(), ...fileList],
    });
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

function createChunkSha256(chunkRow, sliceMetadata) {
  /* istanbul ignore next -- defensive: ?? fallbacks for optional chunk fields */
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
        phase: sliceMetadata?.phase ?? null,
        slice_id: sliceMetadata?.slice_id ?? null,
        status: sliceMetadata?.status ?? null,
        step_number: sliceMetadata?.step_number ?? null,
        symbol_name: chunkRow.symbol_name ?? null,
      }),
    )
    .digest('hex');
}

/**
 * Extract step-packet slice metadata from a plan-family chunk body.
 *
 * Looks for the first fenced YAML block in `bodyText`, then reads the
 * top-level `phase`, `step`, and `status` keys plus the `slice_id` of the
 * first slice under `slices`. Non-plan chunks, missing YAML blocks, or
 * malformed values yield `null` fields.
 *
 * @param {string} bodyText - Chunk body text.
 * @param {string | null} docFamily - Document family (e.g. `'plan'`).
 * @returns {{ slice_id: string | null, step_number: number | null, phase: string | null, status: string | null }} Parsed slice metadata.
 */
function extractSliceMetadata(bodyText, docFamily) {
  const emptyMetadata = {
    phase: null,
    slice_id: null,
    status: null,
    step_number: null,
  };

  if (docFamily !== 'plan' || typeof bodyText !== 'string') {
    return emptyMetadata;
  }

  const yamlMatch = bodyText.match(/```yaml\s*\n([\s\S]*?)\n\s*```/);
  if (!yamlMatch) return emptyMetadata;

  const yamlText = yamlMatch[1];
  // Intentionally scoped to single-line scalar values. Plan step packets use
  // a constrained YAML form; multi-line or nested values are not expected here.
  const phaseMatch = yamlText.match(/^phase\s*:\s*['"]?([^'"\n]+)['"]?/m);
  const statusMatch = yamlText.match(/^status\s*:\s*['"]?([^'"\n]+)['"]?/m);
  const stepMatch = yamlText.match(/^step\s*:\s*(\d+)/m);
  const sliceMatch = yamlText.match(
    /^\s*-\s+slice_id\s*:\s*['"]?([^'"\n]+)['"]?/m,
  );

  /* istanbul ignore next -- defensive: optional chaining ?? null fallbacks for regex matches */
  return {
    phase: phaseMatch?.[1]?.trim() ?? null,
    slice_id: sliceMatch?.[1]?.trim() ?? null,
    status: statusMatch?.[1]?.trim() ?? null,
    step_number: stepMatch ? Number(stepMatch[1]) : null,
  };
}

export async function readModelMeta(options) {
  options = /* istanbul ignore next -- defensive: options always provided in tests */ options ?? {};
  if (options.modelMeta) return options.modelMeta;

  const modelMetaPath = path.resolve(
    options.modelMetaPath ??
      /* istanbul ignore next -- defensive: modelDirectory fallback */
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

export async function createOnnxTextEmbedder(options) {
  options = /* istanbul ignore next -- defensive: options always provided in tests */ options ?? {};
  const forcedState =
    typeof process.env.DENSE_FORCE_STATE === 'string'
      ? process.env.DENSE_FORCE_STATE.trim()
      : '';
  if (forcedState === 'cold' || forcedState === 'model-only') {
    throw new Error(
      `ONNX embedder disabled because DENSE_FORCE_STATE=${forcedState}.`,
    );
  }

  const { Tokenizer } = await import('@huggingface/tokenizers');
  const { InferenceSession, Tensor } = await import('onnxruntime-node');
  const modelDirectory = path.resolve(
    /* istanbul ignore next -- defensive: modelDirectory fallback */
    options.modelDirectory ?? DEFAULT_MODEL_DIRECTORY,
  );
  const modelPath = path.join(modelDirectory, 'model.onnx');
  const tokenizer = await createWordPieceTokenizer({
    modelDirectory,
    Tokenizer,
  });
  const session = await InferenceSession.create(modelPath);
  /* istanbul ignore next -- ONNX runtime unreachable: onnxruntime-node Tensor type bug in ESM/vm-modules */
  const [clsTokenId, sepTokenId] = ['[CLS]', '[SEP]'].map((token) =>
    tokenizer.token_to_id(token),
  );
  /* istanbul ignore next -- ONNX runtime unreachable */
  const maxSequenceLength = Number(
    options.maxSequenceLength ?? DEFAULT_MAX_SEQUENCE_LENGTH,
  );
  /* istanbul ignore next -- ONNX runtime unreachable */
  if (clsTokenId === undefined || sepTokenId === undefined) {
    throw new Error(
      'The local tokenizer vocabulary is missing the required [CLS] or [SEP] tokens.',
    );
  }

  /* istanbul ignore next -- ONNX runtime unreachable: embedText closure requires working ONNX session */
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

  /* istanbul ignore next -- ONNX runtime unreachable */
  embedText.release = async () => {
    await session.release?.();
  };

  /* istanbul ignore next -- ONNX runtime unreachable */
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

/* istanbul ignore next -- only callable from embedText closure which requires working ONNX runtime */
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

/* istanbul ignore next -- CLI main entrypoint, only runs when file is executed directly */
async function main() {
  const args = parseCliArgs(process.argv.slice(2), {
    repeatableFlags: ['files'],
  });
  if (args.help) {
    printHelp({
      title: 'Embedding index builder',
      usage:
        'node rag-index/embed-index.mjs [--dry-run] [--json] [--files=<path>]...',
      options: [
        '--dry-run                  Count work without writing embeddings.',
        '--json                     Emit JSON summary.',
        '--files=<path>             Re-embed chunks for the specified repo-relative file path. Repeatable. Use build-index.mjs first to re-chunk changed files.',
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
    const files = typeof args.files === 'string' ? [args.files] : args.files;
    const summary = await buildEmbeddingIndex({
      corpusDatabasePath: args.database,
      dimension: args.dimension,
      dryRun: Boolean(args['dry-run']),
      files,
      modelDirectory: args['model-directory'],
      modelId: args['model-id'],
      modelSha256: args['model-sha256'],
    });
    writeJsonOrText(summary, Boolean(args.json),
      /* istanbul ignore next -- text formatter covered when writeJsonOrText is not mocked */
      (payload) =>
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

/* istanbul ignore next -- main module guard */
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();
