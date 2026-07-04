/**
 * @module build-term-index
 * @description Build or incrementally update the term embeddings index stored in
 * the consolidated `term_embeddings` table of `rag-index/data/turso-replica.sqlite`.
 * Extracts qualifying terms from corpus chunks, computes mean-pooled embeddings
 * for each term, and stores them for use by query expansion.
 *
 * Qualifying terms are tokens extracted from `heading_path` and the first 256
 * characters of `body_text`, filtered by:
 * - Minimum frequency ≥ 5 chunks
 * - Maximum frequency ≤ 30% of total chunks (stop-word filter)
 * - Length ≥ 3 characters (after Porter stemming)
 * - Contains at least one ASCII letter
 *
 * For each qualifying term, its embedding is the L2-normalized mean of chunk
 * embeddings where the term appears prominently (heading or opening text).
 *
 * @param {boolean} [--dry-run] - Compute qualifying terms without writing embeddings.
 * @param {boolean} [--json] - Emit JSON summary.
 * @param {string}  [--database <path>] - Override corpus database path.
 * @param {string}  [--model-directory <path>] - Override ONNX model directory.
 * @param {string}  [--model-id <id>] - Override model identifier.
 * @param {number}  [--dimension <n>] - Override embedding dimension.
 * @param {string}  [--model-sha256 <hex>] - Override model SHA-256.
 * @param {number}  [--min-frequency <n>] - Minimum chunk frequency for qualifying terms (default: 5).
 * @param {number}  [--max-frequency-ratio <n>] - Maximum frequency ratio (default: 0.3).
 * @param {number}  [--min-term-length <n>] - Minimum term length after Porter stemming (default: 3).
 * @param {boolean} [--help] - Show help and exit.
 *
 * @returns {void} Exits 0 on success, 1 on error. JSON summary written to stdout when `--json` is passed.
 */
import { createClient } from '@libsql/client';
import { createHash } from 'node:crypto';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

import {
  fail,
  parseCliArgs,
  printHelp,
  writeJsonOrText,
} from './cli-utils.mjs';
import { defaultDatabasePath } from './init-schema.mjs';
import {
  DEFAULT_MODEL_DIRECTORY,
  DEFAULT_MODEL_ID,
  buildEmbeddingIndex,
  createOnnxTextEmbedder,
  normalizeEmbeddingVector,
  readModelMeta,
} from './embed-index.mjs';
import { computeCosineSimilarity } from './hybrid-rank.mjs';

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Minimum chunk frequency for a term to qualify for the term index. */
export const DEFAULT_MIN_FREQUENCY = 5;

/** Maximum frequency ratio (term appears in ≤ this fraction of total chunks). */
export const DEFAULT_MAX_FREQUENCY_RATIO = 0.3;

/** Minimum term length (after Porter stemming) to qualify. */
export const DEFAULT_MIN_TERM_LENGTH = 3;

/** Text window size: first N characters of body_text to scan for terms. */
const BODY_TEXT_WINDOW = 256;

// ---------------------------------------------------------------------------
// Term extraction
// ---------------------------------------------------------------------------

/**
 * Extract qualifying terms from corpus chunks.
 *
 * Tokenizes `heading_path` and the first 256 characters of `body_text`,
 * applies frequency and length filters, and returns a map of qualifying
 * terms to their chunk IDs and frequency counts.
 *
 * @param {import('@libsql/client').Client} corpusDatabase - Read-only libSQL corpus client.
 * @param {{ minFrequency?: number, maxFrequencyRatio?: number, minTermLength?: number }} options - Filter options.
 * @returns {Promise<{ qualifyingTerms: Map<string, { frequency: number, docFamilyCount: number, chunkIds: number[] }>, totalChunks: number }>}
 */
export async function extractQualifyingTerms(corpusDatabase, options = {}) {
  const minFrequency = options.minFrequency ?? DEFAULT_MIN_FREQUENCY;
  const maxFrequencyRatio =
    options.maxFrequencyRatio ?? DEFAULT_MAX_FREQUENCY_RATIO;
  const minTermLength = options.minTermLength ?? DEFAULT_MIN_TERM_LENGTH;

  const result = await corpusDatabase.execute({
    sql: `
          SELECT c.chunk_id, c.heading_path, c.body_text, d.doc_family
          FROM chunks c
          LEFT JOIN documents d ON d.doc_id = c.doc_id
          ORDER BY c.chunk_id
        `,
    args: [],
  });
  return processTermStats(result.rows, {
    minFrequency,
    maxFrequencyRatio,
    minTermLength,
  });
}

/**
 * Process chunk rows into qualifying terms with frequency filters.
 *
 * @param {Array<{ chunk_id: number, heading_path: string, body_text: string, doc_family: string }>} chunkRows - Rows from chunks query.
 * @param {{ minFrequency: number, maxFrequencyRatio: number, minTermLength: number }} config - Filter config.
 * @returns {{ qualifyingTerms: Map<string, { frequency: number, docFamily_count: number, chunkIds: number[] }>, totalChunks: number }}
 */
function processTermStats(chunkRows, config) {
  const { minFrequency, maxFrequencyRatio, minTermLength } = config;
  const totalChunks = chunkRows.length;
  const maxFrequency = Math.floor(maxFrequencyRatio * totalChunks);

  // Collect per-term statistics
  const termStats = new Map();

  for (const row of chunkRows) {
    const text = `${row.heading_path ?? ''} ${String(row.body_text ?? '').substring(0, BODY_TEXT_WINDOW)}`;
    const tokens = porterTokenize(text);

    const uniqueTokens = new Set(tokens);
    for (const token of uniqueTokens) {
      if (token.length < minTermLength) continue;
      if (!/[a-zA-Z]/.test(token)) continue;

      const entry = termStats.get(token) ?? {
        frequency: 0,
        docFamilySet: new Set(),
        chunkIds: [],
      };
      entry.frequency += 1;
      entry.docFamilySet.add(row.doc_family ?? 'unknown');
      entry.chunkIds.push(Number(row.chunk_id));
      termStats.set(token, entry);
    }
  }

  // Apply frequency filters
  const qualifyingTerms = new Map();
  for (const [term, entry] of termStats) {
    if (entry.frequency < minFrequency) continue;
    if (entry.frequency > maxFrequency) continue;
    qualifyingTerms.set(term, {
      frequency: entry.frequency,
      doc_family_count: entry.docFamilySet.size,
      chunkIds: entry.chunkIds,
    });
  }

  return { qualifyingTerms, totalChunks };
}

/**
 * Simple Porter-like tokenizer that lowercases, splits on non-alphanumeric,
 * and applies basic stemming heuristics.
 *
 * This approximates the Porter stemmer used by FTS5 for term matching.
 * For production accuracy, FTS5's porter tokenizer is the authority;
 * this extractor produces candidate terms that are close enough for
 * embedding-based synonym discovery.
 *
 * @param {string} text - Raw text to tokenize.
 * @returns {string[]} Array of lowercased, stemmed tokens.
 */
export function porterTokenize(text) {
  const raw = String(text ?? '').toLowerCase();
  const tokens = raw.split(/[^a-z0-9]+/).filter(Boolean);
  return tokens.map(applyPorterStem);
}

/**
 * Apply a simplified Porter stemming heuristic.
 *
 * Handles the most common English suffixes. Not a full Porter stemmer,
 * but sufficient for term matching in query expansion context.
 *
 * @param {string} word - Lowercased word to stem.
 * @returns {string} Stemmed word.
 */
export function applyPorterStem(word) {
  if (word.length < 4) return word;

  // Step 1a: plural/gerund
  if (word.endsWith('sses')) return word.slice(0, -2);
  if (word.endsWith('ies')) return word.slice(0, -2);
  if (word.endsWith('ss')) return word;
  if (word.endsWith('s') && !word.endsWith('us') && !word.endsWith('ss'))
    return word.slice(0, -1);

  // Step 1b: past participle / gerund
  if (word.endsWith('eed') && word.length > 4) return word.slice(0, -1);
  if (word.endsWith('ed') && word.length > 4) {
    const stem = word.slice(0, -2);
    if (stem.match(/[aeiou]/))
      return stem.endsWith('d') || stem.endsWith('l')
        ? stem + stem[stem.length - 1]
        : stem;
    return stem;
  }
  if (word.endsWith('ing') && word.length > 5) {
    const stem = word.slice(0, -3);
    if (stem.match(/[aeiou]/))
      return stem.endsWith('d') || stem.endsWith('l')
        ? stem + stem[stem.length - 1]
        : stem;
    return stem;
  }

  // Step 2: common suffixes
  if (word.endsWith('ational')) return word.slice(0, -5) + 'e';
  if (word.endsWith('tional')) return word.slice(0, -4) + 'e';
  if (word.endsWith('ization')) return word.slice(0, -5) + 'e';
  if (word.endsWith('ation')) return word.slice(0, -3);
  if (word.endsWith('ness')) return word.slice(0, -4);
  if (word.endsWith('ment')) return word.slice(0, -4);

  return word;
}

// ---------------------------------------------------------------------------
// Term embedding computation
// ---------------------------------------------------------------------------

/**
 * Compute term embeddings by mean-pooling chunk embeddings for each qualifying term.
 *
 * For each term, loads the embeddings of all chunks where the term appears
 * prominently (in heading_path or first 256 chars of body_text), then computes
 * the L2-normalized mean of those embeddings.
 *
 * Reads chunk embeddings from the `chunks.embedding` column (consolidated
 * schema) and writes term embeddings via `await client.execute()`.
 *
 * @param {import('@libsql/client').Client} embeddingsDatabase - libSQL client.
 * @param {Map<string, { frequency: number, doc_family_count: number, chunkIds: number[] }>} qualifyingTerms - Terms and their chunk IDs.
 * @param {{ modelId: string, modelSha256: string, dimension: number }} modelInfo - Model identification.
 * @returns {Promise<{ built: number, skipped: number, purged: number }>} Summary counts.
 */
export async function buildTermEmbeddings(
  embeddingsDatabase,
  qualifyingTerms,
  modelInfo,
) {
  return buildTermEmbeddingsWithClient(
    embeddingsDatabase,
    qualifyingTerms,
    modelInfo,
  );
}

/**
 * Async client-based term embedding computation.
 *
 * Reads chunk embeddings from the `chunks.embedding` column (consolidated
 * Turso schema) instead of the separate `chunk_embeddings` table.
 *
 * @param {import('@libsql/client').Client} client - libSQL client.
 * @param {Map<string, { frequency: number, doc_family_count: number, chunkIds: number[] }>} qualifyingTerms - Terms and their chunk IDs.
 * @param {{ modelId: string, modelSha256: string, dimension: number }} modelInfo - Model identification.
 * @returns {Promise<{ built: number, skipped: number, purged: number }>} Summary counts.
 */
async function buildTermEmbeddingsWithClient(
  client,
  qualifyingTerms,
  modelInfo,
) {
  const { modelId, modelSha256, dimension } = modelInfo;

  // Load chunk embeddings from consolidated chunks.embedding column.
  const embeddingRowsResult = await client.execute({
    sql: 'SELECT chunk_id, embedding FROM chunks WHERE embedding IS NOT NULL AND embedding_model = ?',
    args: [modelId],
  });

  const chunkEmbeddingMap = new Map();
  for (const row of embeddingRowsResult.rows) {
    const buf = row.embedding;
    const float32 = new Float32Array(
      buf.buffer,
      buf.byteOffset,
      buf.byteLength / 4,
    );
    chunkEmbeddingMap.set(Number(row.chunk_id), float32);
  }

  // Purge existing term embeddings for this model.
  await client.execute({
    sql: 'DELETE FROM term_embeddings WHERE model_id = ?',
    args: [modelId],
  });

  let built = 0;

  for (const [term, entry] of qualifyingTerms) {
    const { frequency, doc_family_count: docFamilyCount, chunkIds } = entry;

    // Collect embeddings for chunks containing this term.
    const termChunkEmbeddings = [];
    for (const chunkId of chunkIds) {
      const embedding = chunkEmbeddingMap.get(chunkId);
      if (embedding) {
        termChunkEmbeddings.push(embedding);
      }
    }

    // Skip terms with no available embeddings.
    if (termChunkEmbeddings.length === 0) continue;

    // Mean-pool the chunk embeddings.
    const meanEmbedding = new Float32Array(dimension);
    for (const embedding of termChunkEmbeddings) {
      for (let i = 0; i < dimension; i += 1) {
        meanEmbedding[i] += embedding[i];
      }
    }
    for (let i = 0; i < dimension; i += 1) {
      meanEmbedding[i] /= termChunkEmbeddings.length;
    }

    // L2-normalize.
    const normalizedEmbedding = normalizeL2(meanEmbedding);

    // Compute term SHA-256.
    const termSha256 = createHash('sha256').update(term).digest('hex');

    // Store as BLOB.
    const embeddingBuffer = Buffer.from(
      normalizedEmbedding.buffer,
      normalizedEmbedding.byteOffset,
      normalizedEmbedding.byteLength,
    );

    await client.execute({
      sql: `INSERT INTO term_embeddings (
          term, embedding, term_sha256, model_id, model_sha256,
          dimension, frequency, doc_family_count
        ) VALUES (?, vector8(?), ?, ?, ?, ?, ?, ?)`,
      args: [
        term,
        embeddingBuffer,
        termSha256,
        modelId,
        modelSha256,
        dimension,
        frequency,
        docFamilyCount,
      ],
    });

    built += 1;
  }

  return {
    built,
    skipped: 0,
    purged: 0,
  };
}

/**
 * L2-normalize a vector in place and return it.
 *
 * @param {Float32Array} vector - Vector to normalize.
 * @returns {Float32Array} The normalized vector (same reference).
 */
function normalizeL2(vector) {
  let magnitudeSquared = 0;
  for (let i = 0; i < vector.length; i += 1) {
    magnitudeSquared += vector[i] * vector[i];
  }
  if (magnitudeSquared === 0) return vector;
  const magnitude = Math.sqrt(magnitudeSquared);
  for (let i = 0; i < vector.length; i += 1) {
    vector[i] /= magnitude;
  }
  return vector;
}

// ---------------------------------------------------------------------------
// CLI
// ---------------------------------------------------------------------------

async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Term index builder',
      usage:
        'node rag-index/build-term-index.mjs [--json] [--dry-run] [--database <path>]',
      options: [
        '--dry-run                     Compute qualifying terms without writing embeddings',
        '--json                        Emit JSON summary',
        '--database <path>             Override corpus database path',
        '--model-directory <path>      Override ONNX model directory',
        '--model-id <id>              Override model identifier',
        '--dimension <n>               Override embedding dimension',
        '--model-sha256 <hex>           Override model SHA-256',
        '--min-frequency <n>           Minimum chunk frequency (default: 5)',
        '--max-frequency-ratio <n>     Maximum frequency ratio (default: 0.3)',
        '--min-term-length <n>         Minimum term length (default: 3)',
        '--help                        Show this help',
      ],
    });
    return;
  }

  const corpusDatabasePath = path.resolve(args.database ?? defaultDatabasePath);

  const modelMeta = await readModelMeta(args);
  const modelId = String(args['model-id'] ?? DEFAULT_MODEL_ID);
  const dimension = Number(args.dimension ?? modelMeta.dimension ?? 0);
  const modelSha256 = String(
    args['model-sha256'] ?? modelMeta.model_sha256 ?? '',
  );

  if (!dimension || !modelSha256) {
    fail(
      'Model dimension and SHA-256 are required. Run embed-index.mjs first or pass --dimension and --model-sha256.',
    );
  }

  const corpusDatabase = createClient({
    url: pathToFileURL(corpusDatabasePath).href,
  });

  const { qualifyingTerms, totalChunks } = await extractQualifyingTerms(
    corpusDatabase,
    {
      minFrequency: Number(args['min-frequency'] ?? DEFAULT_MIN_FREQUENCY),
      maxFrequencyRatio: Number(
        args['max-frequency-ratio'] ?? DEFAULT_MAX_FREQUENCY_RATIO,
      ),
      minTermLength: Number(args['min-term-length'] ?? DEFAULT_MIN_TERM_LENGTH),
    },
  );

  if (args['dry-run']) {
    const summary = {
      dryRun: true,
      qualifyingTermCount: qualifyingTerms.size,
      totalChunks,
      built: 0,
      skipped: 0,
      purged: 0,
    };
    writeJsonOrText(summary, Boolean(args.json), (payload) =>
      JSON.stringify(payload, null, 2),
    );
    await corpusDatabase.close();
    return;
  }

  const result = await buildTermEmbeddings(corpusDatabase, qualifyingTerms, {
    modelId,
    modelSha256,
    dimension,
  });

  await corpusDatabase.close();

  const summary = {
    dryRun: false,
    qualifyingTermCount: qualifyingTerms.size,
    totalChunks,
    ...result,
  };

  writeJsonOrText(summary, Boolean(args.json), (payload) =>
    JSON.stringify(payload, null, 2),
  );
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();
