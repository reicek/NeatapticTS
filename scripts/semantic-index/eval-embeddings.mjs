/**
 * @description Evaluate hybrid dense retrieval quality against the canonical 20-query
 * eval set in `scripts/semantic-index/eval-queries.json`. Reports MRR\@5 for BM25-only
 * and hybrid modes and asserts that the hybrid score meets a minimum improvement
 * threshold (default: +0.02). Exits non-zero when the threshold is not met.
 *
 * MRR\@5 formula: `MRR@5 = (1/|Q|) * sum_q { 1 / rank_of_first_hit_at_k<=5 }` (0 if
 * no hit in top 5). Hit criteria: result belongs to an `expected_doc_families` member
 * AND the chunk heading or symbol name contains the expected substring.
 *
 * @param {boolean} [--json]                      - Emit JSON evaluation output.
 * @param {number}  [--alpha <n>]                 - Hybrid BM25 weight (0–1, default: 0.5).
 * @param {string}  [--database <path>]           - Override corpus database path.
 * @param {string}  [--model-directory <path>]    - Override local model cache directory.
 * @param {string}  [--model-id <id>]             - Override model identifier.
 * @param {string}  [--query-file <path>]         - Override eval query set path.
 * @param {number}  [--min-hybrid-improvement <n>] - Override required hybrid MRR\@5 gain (default: 0.02).
 * @param {boolean} [--help]                      - Show help and exit.
 *
 * @returns {void} Exits 0 when the hybrid improvement threshold is met, 1 otherwise.
 *   JSON report written to stdout when `--json` is passed.
 */
import { createClient } from '@libsql/client';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  fail,
  parseCliArgs,
  printHelp,
  writeJsonOrText,
} from './cli-utils.mjs';
import {
  DEFAULT_MODEL_DIRECTORY,
  DEFAULT_MODEL_ID,
  createOnnxTextEmbedder,
  readModelMeta,
} from './embed-index.mjs';
import { queryDenseIndex } from './query-dense.mjs';
import { defaultDatabasePath, repoRoot } from './init-schema.mjs';

const DEFAULT_QUERY_FILE_PATH = path.join(
  repoRoot,
  'scripts',
  'semantic-index',
  'eval-queries.json',
);
const DEFAULT_MIN_HYBRID_IMPROVEMENT = 0.02;

export async function evaluateEmbeddings(options = {}) {
  const corpusDatabasePath = path.resolve(
    options.corpusDatabasePath ?? options.databasePath ?? defaultDatabasePath,
  );
  const queryFilePath = path.resolve(
    options.queryFilePath ?? DEFAULT_QUERY_FILE_PATH,
  );
  const querySpecs = JSON.parse(await readFile(queryFilePath, 'utf8'));
  const modelMeta = await readModelMeta({
    modelDirectory: options.modelDirectory ?? DEFAULT_MODEL_DIRECTORY,
    modelMeta: options.modelMeta,
  });
  const modelId = String(
    options.modelId ?? modelMeta.model_id ?? DEFAULT_MODEL_ID,
  );
  const embedText =
    options.embedText ??
    (await createOnnxTextEmbedder({
      dimension: Number(options.dimension ?? modelMeta.dimension ?? 0),
      modelDirectory: options.modelDirectory ?? DEFAULT_MODEL_DIRECTORY,
      modelId,
    }));
  const limit = 5;
  const minHybridImprovement = Number(
    options.minHybridImprovement ?? DEFAULT_MIN_HYBRID_IMPROVEMENT,
  );
  const alpha = Number(options.alpha ?? 0.5);

  const queryReports = [];
  let bm25Total = 0;
  let hybridTotal = 0;

  try {
    for (const querySpec of querySpecs) {
      const bm25Result = await queryDenseIndex({
        corpusDatabasePath,
        dense: false,
        family: querySpec.family ?? null,
        limit,
        query: querySpec.query,
        client: options.client,
      });
      const hybridResult = await queryDenseIndex({
        alpha,
        corpusDatabasePath,
        dense: true,
        family: querySpec.family ?? null,
        limit,
        modelDirectory: options.modelDirectory ?? DEFAULT_MODEL_DIRECTORY,
        modelId,
        query: querySpec.query,
        embedText,
        client: options.client,
      });
      const bm25Rank = findHitRank(bm25Result.results, querySpec, limit);
      const hybridRank = findHitRank(hybridResult.results, querySpec, limit);
      const bm25ReciprocalRank = bm25Rank ? 1 / bm25Rank : 0;
      const hybridReciprocalRank = hybridRank ? 1 / hybridRank : 0;

      bm25Total += bm25ReciprocalRank;
      hybridTotal += hybridReciprocalRank;
      queryReports.push({
        bm25_rank: bm25Rank,
        bm25_rr_at_5: bm25ReciprocalRank,
        hybrid_rank: hybridRank,
        hybrid_rr_at_5: hybridReciprocalRank,
        query: querySpec.query,
      });
    }
  } finally {
    if (
      typeof options.embedText?.release !== 'function' &&
      typeof embedText?.release === 'function'
    ) {
      await embedText.release();
    }
  }

  const bm25MrrAt5 = querySpecs.length > 0 ? bm25Total / querySpecs.length : 0;
  const hybridMrrAt5 =
    querySpecs.length > 0 ? hybridTotal / querySpecs.length : 0;
  const tsSourceQueries = querySpecs.filter(
    (querySpec) =>
      Array.isArray(querySpec.expected_doc_families) &&
      querySpec.expected_doc_families.includes('ts-source'),
  ).length;
  const { chunkCount, embeddingCount } = options.client
    ? await loadCorpusCountsWithClient(options.client, modelId)
    : await loadCorpusCounts(corpusDatabasePath, modelId);
  const pass =
    hybridMrrAt5 >= bm25MrrAt5 + minHybridImprovement && tsSourceQueries >= 5;

  return {
    alpha,
    bm25MrrAt5,
    chunkCount,
    embeddingCount,
    hybridMrrAt5,
    improvement: hybridMrrAt5 - bm25MrrAt5,
    minHybridImprovement,
    modelId,
    pass,
    queryCount: querySpecs.length,
    queryReports,
    tsSourceQueries,
  };
}

async function loadCorpusCounts(corpusDatabasePath, modelId) {
  const corpusClient = createClient({
    url: pathToFileURL(corpusDatabasePath).href,
  });

  try {
    const chunkResult = await corpusClient.execute(
      'SELECT COUNT(*) AS count FROM chunks',
    );
    const embeddingResult = await corpusClient.execute({
      sql: 'SELECT COUNT(*) AS count FROM chunks WHERE embedding IS NOT NULL AND embedding_model = ?',
      args: [modelId],
    });
    return {
      chunkCount: Number(chunkResult.rows[0].count),
      embeddingCount: Number(embeddingResult.rows[0].count),
    };
  } finally {
    await corpusClient.close();
  }
}

/**
 * Async client path for loading corpus counts via a Turso/libSQL client.
 *
 * Reads both chunk count and embedding count from the single consolidated
 * database. Embeddings are stored in the `chunks.embedding` column (not a
 * separate `chunk_embeddings` table).
 *
 * @param {import('@libsql/client').Client} client - Turso/libSQL client.
 * @param {string} modelId - Embedding model identifier.
 * @returns {Promise<{ chunkCount: number, embeddingCount: number }>}
 */
async function loadCorpusCountsWithClient(client, modelId) {
  const chunkResult = await client.execute(
    'SELECT COUNT(*) AS count FROM chunks',
  );
  const embeddingResult = await client.execute({
    sql: 'SELECT COUNT(*) AS count FROM chunks WHERE embedding IS NOT NULL AND embedding_model = ?',
    args: [modelId],
  });
  return {
    chunkCount: Number(chunkResult.rows[0].count),
    embeddingCount: Number(embeddingResult.rows[0].count),
  };
}

function findHitRank(results, querySpec, limit) {
  const deepestAcceptedRank = Math.min(
    limit,
    Number(querySpec.min_rank_of_hit ?? limit),
  );
  for (
    let resultIndex = 0;
    resultIndex < Math.min(results.length, limit);
    resultIndex += 1
  ) {
    const candidate = results[resultIndex];
    if (!matchesQueryExpectation(candidate, querySpec)) continue;
    const rank = resultIndex + 1;
    return rank <= deepestAcceptedRank ? rank : null;
  }
  return null;
}

function matchesQueryExpectation(result, querySpec) {
  const expectedFamilies = Array.isArray(querySpec.expected_doc_families)
    ? querySpec.expected_doc_families
    : [];
  if (expectedFamilies.length > 0 && !expectedFamilies.includes(result.family))
    return false;

  const headingNeedle = normalizeNeedle(querySpec.expected_heading_contains);
  const symbolNeedle = normalizeNeedle(querySpec.expected_symbol_contains);
  const headingHaystack = normalizeNeedle(result.heading_path);

  if (headingNeedle && !headingHaystack.includes(headingNeedle)) return false;
  if (symbolNeedle && !headingHaystack.includes(symbolNeedle)) return false;
  return true;
}

function normalizeNeedle(value) {
  return String(value ?? '')
    .trim()
    .toLowerCase();
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Dense embedding evaluator',
      usage: 'node scripts/semantic-index/eval-embeddings.mjs [--json]',
      options: [
        '--json                     Emit JSON evaluation output.',
        '--alpha <n>                Hybrid BM25 weight (default: 0.5).',
        '--database <path>          Override the corpus database path.',
        '--model-directory <path>   Override the local model cache directory.',
        '--model-id <id>            Override the model identifier.',
        '--query-file <path>        Override the eval query set path.',
        '--min-hybrid-improvement <n>  Override the required hybrid MRR@5 gain.',
        '--help                     Show this help.',
      ],
    });
    return;
  }

  try {
    const report = await evaluateEmbeddings({
      alpha: args.alpha,
      corpusDatabasePath: args.database,
      minHybridImprovement: args['min-hybrid-improvement'],
      modelDirectory: args['model-directory'],
      modelId: args['model-id'],
      queryFilePath: args['query-file'],
    });
    writeJsonOrText(report, Boolean(args.json), (payload) =>
      payload.pass
        ? `Dense eval passed: hybrid MRR@5 ${payload.hybridMrrAt5.toFixed(3)} vs BM25 ${payload.bm25MrrAt5.toFixed(3)}`
        : `Dense eval failed: hybrid MRR@5 ${payload.hybridMrrAt5.toFixed(3)} vs BM25 ${payload.bm25MrrAt5.toFixed(3)}`,
    );
    if (!report.pass) process.exitCode = 1;
  } catch (error) {
    fail(
      error instanceof Error ? error.message : String(error),
      Boolean(args.json),
    );
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();
