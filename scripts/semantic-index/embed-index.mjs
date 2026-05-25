/**
 * @description Build or incrementally update the dense embedding index stored in
 * `data/embeddings.sqlite`. Reads all chunks from `data/semantic-index.sqlite`, runs
 * each chunk body through the locally cached `all-MiniLM-L6-v2` ONNX model
 * (mean-pool → L2-normalize → 384-dim float32 BLOB), and stores vectors in the
 * `chunk_embeddings` table. Skips chunks whose `chunk_sha256` and `model_id` are
 * unchanged (incremental rule). Run `download-model.mjs` once before this script.
 *
 * @param {boolean} [--dry-run]                     - Count queued chunks without writing embeddings.
 * @param {boolean} [--json]                         - Emit JSON summary `{ embedded, skipped, queued, dryRun }`.
 * @param {string}  [--database <path>]              - Override corpus database path.
 * @param {string}  [--embeddings-database <p>]      - Override embeddings database path.
 * @param {string}  [--model-directory <path>]       - Override local model cache directory.
 * @param {string}  [--model-id <id>]                - Override model identifier.
 * @param {number}  [--dimension <n>]                - Override embedding dimension.
 * @param {string}  [--model-sha256 <hex>]           - Override model SHA-256 value.
 * @param {boolean} [--help]                         - Show help and exit.
 *
 * @returns {void} Exits 0 on success, 1 on error. JSON summary written to stdout when `--json` is passed.
 */
import Database from 'better-sqlite3';
import { createHash } from 'node:crypto';
import { mkdir, readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { fail, parseCliArgs, printHelp, writeJsonOrText } from './cli-utils.mjs';
import { defaultDatabasePath, repoRoot } from './init-schema.mjs';

export const DEFAULT_MODEL_ID = 'all-MiniLM-L6-v2';
export const DEFAULT_MODEL_DIRECTORY = path.join(repoRoot, 'scripts', 'semantic-index', 'models');
export const DEFAULT_EMBEDDINGS_DATABASE_PATH = path.join(repoRoot, 'data', 'embeddings.sqlite');
const DEFAULT_MAX_SEQUENCE_LENGTH = 512;

export async function buildEmbeddingIndex(options = {}) {
  const corpusDatabasePath = path.resolve(options.corpusDatabasePath ?? options.databasePath ?? defaultDatabasePath);
  const embeddingsDatabasePath = path.resolve(options.embeddingsDatabasePath ?? DEFAULT_EMBEDDINGS_DATABASE_PATH);
  const modelId = String(options.modelId ?? DEFAULT_MODEL_ID);
  const dryRun = Boolean(options.dryRun);
  const modelMeta = await readModelMeta(options);
  const dimension = Number(options.dimension ?? modelMeta.dimension ?? 0);
  const modelSha256 = String(options.modelSha256 ?? modelMeta.model_sha256 ?? '');

  if (!Number.isInteger(dimension) || dimension < 1) {
    throw new Error('Embedding dimension is required. Pass --dimension or provide scripts/semantic-index/models/model-meta.json.');
  }

  if (!modelSha256) {
    throw new Error('Model SHA-256 is required. Pass --model-sha256 or provide scripts/semantic-index/models/model-meta.json.');
  }

  const embedText = options.embedText ?? await createOnnxTextEmbedder({
    dimension,
    modelDirectory: options.modelDirectory ?? DEFAULT_MODEL_DIRECTORY,
    modelId,
  });

  await mkdir(path.dirname(embeddingsDatabasePath), { recursive: true });

  const corpusDatabase = new Database(corpusDatabasePath, { readonly: true, fileMustExist: true });
  const embeddingsDatabase = new Database(embeddingsDatabasePath);
  initializeEmbeddingsSchema(embeddingsDatabase);

  const summary = {
    dryRun,
    embedded: 0,
    embeddingsDatabasePath,
    modelId,
    purged: 0,
    queued: 0,
    skipped: 0,
  };

  if (!dryRun) {
    summary.purged = purgeOrphanedEmbeddings({
      corpusDatabase,
      corpusDatabasePath,
      embeddingsDatabase,
      modelId,
    });
  }

  const chunkRows = corpusDatabase.prepare(`
    SELECT c.chunk_id, c.chunk_index, c.heading_path, c.body_text, c.char_start, c.char_end,
      d.file_path, d.doc_family
    FROM chunks c
    LEFT JOIN documents d ON d.doc_id = c.doc_id
    ORDER BY c.chunk_id
  `).all();

  const selectExistingEmbedding = embeddingsDatabase.prepare(`
    SELECT chunk_sha256, model_id
    FROM chunk_embeddings
    WHERE chunk_id = ?
  `);
  const upsertEmbedding = embeddingsDatabase.prepare(`
    INSERT INTO chunk_embeddings (
      chunk_id,
      embedding,
      chunk_sha256,
      model_id,
      model_sha256,
      dimension,
      embedded_at
    ) VALUES (?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(chunk_id) DO UPDATE SET
      embedding = excluded.embedding,
      chunk_sha256 = excluded.chunk_sha256,
      model_id = excluded.model_id,
      model_sha256 = excluded.model_sha256,
      dimension = excluded.dimension,
      embedded_at = excluded.embedded_at
  `);

  const writeEmbedding = embeddingsDatabase.transaction((chunkId, embeddingBuffer, chunkSha256) => {
    upsertEmbedding.run(
      chunkId,
      embeddingBuffer,
      chunkSha256,
      modelId,
      modelSha256,
      dimension,
      new Date().toISOString(),
    );
  });

  try {
    for (const chunkRow of chunkRows) {
      const chunkSha256 = createChunkSha256(chunkRow);
      const existingEmbedding = selectExistingEmbedding.get(chunkRow.chunk_id);
      if (existingEmbedding?.chunk_sha256 === chunkSha256 && existingEmbedding?.model_id === modelId) {
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
      writeEmbedding(chunkRow.chunk_id, toBlobBuffer(embeddingVector), chunkSha256);
      summary.embedded += 1;
    }
  } finally {
    corpusDatabase.close();
    embeddingsDatabase.close();
    await releaseEmbedText(embedText);
  }

  return summary;
}

export function normalizeEmbeddingVector(vectorLike, dimension) {
  const float32Vector = toFloat32Array(vectorLike);
  if (float32Vector.length !== dimension) {
    throw new Error(`Expected embedding dimension ${dimension}, received ${float32Vector.length}.`);
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

function initializeEmbeddingsSchema(database) {
  database.exec(`
    CREATE TABLE IF NOT EXISTS chunk_embeddings (
      chunk_id INTEGER PRIMARY KEY,
      embedding BLOB NOT NULL,
      chunk_sha256 TEXT NOT NULL,
      model_id TEXT NOT NULL,
      model_sha256 TEXT NOT NULL,
      dimension INTEGER NOT NULL,
      embedded_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS chunk_embeddings_model_idx
      ON chunk_embeddings(model_id, chunk_sha256);
  `);
}

function purgeOrphanedEmbeddings({ corpusDatabase, corpusDatabasePath, embeddingsDatabase, modelId }) {
  embeddingsDatabase.prepare('ATTACH DATABASE ? AS corpus').run(corpusDatabasePath);

  try {
    const orphanedChunkRows = embeddingsDatabase.prepare(`
      SELECT chunk_embeddings.chunk_id
      FROM chunk_embeddings
      LEFT JOIN corpus.chunks ON corpus.chunks.chunk_id = chunk_embeddings.chunk_id
      WHERE chunk_embeddings.model_id = ?
        AND corpus.chunks.chunk_id IS NULL
    `);
    const orphanedRows = orphanedChunkRows.all(modelId);
    if (orphanedRows.length === 0) return 0;

    const deleteEmbedding = embeddingsDatabase.prepare(`
      DELETE FROM chunk_embeddings
      WHERE chunk_id = ?
        AND model_id = ?
    `);
    const purgeTransaction = embeddingsDatabase.transaction((rowsToDelete) => {
      for (const { chunk_id: chunkId } of rowsToDelete) deleteEmbedding.run(chunkId, modelId);
    });

    purgeTransaction(orphanedRows);
    return orphanedRows.length;
  } finally {
    embeddingsDatabase.prepare('DETACH DATABASE corpus').run();
  }
}

function createChunkSha256(chunkRow) {
  return createHash('sha256').update(JSON.stringify({
    body_text: chunkRow.body_text,
    char_end: Number(chunkRow.char_end),
    char_start: Number(chunkRow.char_start),
    chunk_id: Number(chunkRow.chunk_id),
    chunk_index: Number(chunkRow.chunk_index),
    doc_family: chunkRow.doc_family ?? null,
    file_path: chunkRow.file_path ?? null,
    heading_path: chunkRow.heading_path ?? null,
  })).digest('hex');
}

export async function readModelMeta(options = {}) {
  if (options.modelMeta) return options.modelMeta;

  const modelMetaPath = path.resolve(options.modelMetaPath ?? path.join(options.modelDirectory ?? DEFAULT_MODEL_DIRECTORY, 'model-meta.json'));
  try {
    return JSON.parse(await readFile(modelMetaPath, 'utf8'));
  } catch (error) {
    if (error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT') {
      return {};
    }
    throw error;
  }
}

export async function createOnnxTextEmbedder(options = {}) {
  const { Tokenizer } = await import('@huggingface/tokenizers');
  const { InferenceSession, Tensor } = await import('onnxruntime-node');
  const modelDirectory = path.resolve(options.modelDirectory ?? DEFAULT_MODEL_DIRECTORY);
  const modelPath = path.join(modelDirectory, 'model.onnx');
  const tokenizer = await createWordPieceTokenizer({ modelDirectory, Tokenizer });
  const session = await InferenceSession.create(modelPath);
  const [clsTokenId, sepTokenId] = ['[CLS]', '[SEP]'].map((token) => tokenizer.token_to_id(token));
  const maxSequenceLength = Number(options.maxSequenceLength ?? DEFAULT_MAX_SEQUENCE_LENGTH);
  if (clsTokenId === undefined || sepTokenId === undefined) {
    throw new Error('The local tokenizer vocabulary is missing the required [CLS] or [SEP] tokens.');
  }

  const embedText = async ({ text }) => {
    const encoded = tokenizer.encode(String(text ?? ''), {
      add_special_tokens: false,
      return_token_type_ids: true,
    });
    const truncatedTokenIds = encoded.ids.slice(0, Math.max(0, maxSequenceLength - 2));
    const inputIds = [clsTokenId, ...truncatedTokenIds, sepTokenId];
    const attentionMaskValues = inputIds.map(() => 1);
    const tokenTypeIdValues = inputIds.map(() => 0);
    const tokenIds = BigInt64Array.from(inputIds.map((tokenId) => BigInt(tokenId)));
    const attentionMask = BigInt64Array.from(attentionMaskValues.map((maskValue) => BigInt(maskValue)));
    const tokenTypeIds = BigInt64Array.from(tokenTypeIdValues.map((tokenTypeId) => BigInt(tokenTypeId)));
    const sequenceLength = inputIds.length;
    const feeds = {
      attention_mask: new Tensor('int64', attentionMask, [1, sequenceLength]),
      input_ids: new Tensor('int64', tokenIds, [1, sequenceLength]),
    };

    if (session.inputNames.includes('token_type_ids')) {
      feeds.token_type_ids = new Tensor('int64', tokenTypeIds, [1, sequenceLength]);
    }

    const outputs = await session.run(feeds);
    const outputName = session.outputNames.find((name) => name === 'last_hidden_state') ?? session.outputNames[0];
    if (!outputName || !outputs[outputName]) {
      throw new Error('ONNX embedding session did not return last_hidden_state output.');
    }

    return meanPoolEmbedding(outputs[outputName], attentionMask, options.dimension);
  };

  embedText.release = async () => {
    await session.release?.();
  };

  return embedText;
}

async function createWordPieceTokenizer({ Tokenizer, modelDirectory }) {
  const tokenizerJsonPath = path.join(modelDirectory, 'tokenizer.json');
  const tokenizerConfigPath = path.join(modelDirectory, 'tokenizer_config.json');
  const specialTokensMapPath = path.join(modelDirectory, 'special_tokens_map.json');
  const [tokenizerJson, tokenizerConfig, specialTokensMap] = await Promise.all([
    readJsonFile(tokenizerJsonPath, null),
    readJsonFile(tokenizerConfigPath, {}),
    readJsonFile(specialTokensMapPath, {}),
  ]);
  const vocabulary = tokenizerJson?.model?.vocab;
  if (!vocabulary || typeof vocabulary !== 'object') {
    throw new Error('tokenizer.json is missing the WordPiece vocabulary required for local tokenization.');
  }
  const unkToken = resolveSpecialToken(specialTokensMap.unk_token, '[UNK]');
  const tokenizer = new Tokenizer({
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
  }, {
    clean_up_tokenization_spaces: true,
  });

  return tokenizer;
}

function meanPoolEmbedding(lastHiddenStateTensor, attentionMask, dimension) {
  const outputData = toFloat32Array(lastHiddenStateTensor.data);
  const normalizedDimension = Number(dimension);
  if (!Number.isInteger(normalizedDimension) || normalizedDimension < 1) {
    throw new Error('A positive embedding dimension is required for mean pooling.');
  }

  const pooledVector = new Float32Array(normalizedDimension);
  let includedTokenCount = 0;

  for (let tokenIndex = 0; tokenIndex < attentionMask.length; tokenIndex += 1) {
    if (Number(attentionMask[tokenIndex]) === 0) continue;
    includedTokenCount += 1;
    const tokenOffset = tokenIndex * normalizedDimension;
    for (let dimensionIndex = 0; dimensionIndex < normalizedDimension; dimensionIndex += 1) {
      pooledVector[dimensionIndex] += outputData[tokenOffset + dimensionIndex];
    }
  }

  if (includedTokenCount === 0) return pooledVector;

  for (let dimensionIndex = 0; dimensionIndex < normalizedDimension; dimensionIndex += 1) {
    pooledVector[dimensionIndex] /= includedTokenCount;
  }
  return pooledVector;
}

function toBlobBuffer(float32Vector) {
  return Buffer.from(float32Vector.buffer, float32Vector.byteOffset, float32Vector.byteLength);
}

function toFloat32Array(vectorLike) {
  if (vectorLike instanceof Float32Array) return vectorLike;
  if (ArrayBuffer.isView(vectorLike)) {
    return new Float32Array(vectorLike.buffer.slice(vectorLike.byteOffset, vectorLike.byteOffset + vectorLike.byteLength));
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
    if (error && typeof error === 'object' && 'code' in error && error.code === 'ENOENT') {
      return fallbackValue;
    }
    throw error;
  }
}

function resolveSpecialToken(value, fallbackToken) {
  if (typeof value === 'string' && value.trim()) return value;
  if (value && typeof value === 'object' && typeof value.content === 'string' && value.content.trim()) {
    return value.content;
  }
  return fallbackToken;
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Embedding index builder',
      usage: 'node scripts/semantic-index/embed-index.mjs [--dry-run] [--json]',
      options: [
        '--dry-run                  Count work without writing embeddings.',
        '--json                     Emit JSON summary.',
        '--database <path>          Override the semantic-index corpus database path.',
        '--embeddings-database <p>  Override the embeddings database path.',
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
      embeddingsDatabasePath: args['embeddings-database'],
      modelDirectory: args['model-directory'],
      modelId: args['model-id'],
      modelSha256: args['model-sha256'],
    });
    writeJsonOrText(summary, Boolean(args.json), (payload) => payload.dryRun
      ? `Embedding build dry run: queued ${payload.queued}, skipped ${payload.skipped}`
      : `Embedding build complete: embedded ${payload.embedded}, skipped ${payload.skipped}`);
  } catch (error) {
    fail(error instanceof Error ? error.message : String(error), Boolean(args.json));
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();