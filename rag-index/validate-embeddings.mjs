/**
 * @description Validate that the dense embedding index in the consolidated
 * `chunks.embedding` column of `rag-index/data/turso-replica.sqlite` is usable: at least
 * one chunk must have a vector for the active model. Emits the standard gate JSON
 * contract `{ pass, evidence, fixHint, owner }`. Use before relying on hybrid
 * search results.
 *
 * @param {boolean} [--json]                     - Emit the standard gate JSON contract.
 * @param {string}  [--database <path>]          - Override corpus database path.
 * @param {string}  [--model-id <id>]            - Restrict validation to one model id.
 * @param {boolean} [--help]                     - Show help and exit.
 *
 * @returns {void} Exits 0 when embeddings are present,
 *   1 when no embeddings are found for the requested model.
 */
import { createClient } from '@libsql/client';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  fail,
  parseCliArgs,
  printHelp,
  writeJsonOrText,
} from './cli-utils.mjs';
import { defaultDatabasePath } from './init-schema.mjs';

export async function validateEmbeddings(options = {}) {
  const modelId = String(options.modelId ?? 'all-MiniLM-L6-v2');

  if (options.client) {
    return await validateEmbeddingsOnClient(options.client, modelId);
  }

  const corpusDatabasePath = path.resolve(
    options.corpusDatabasePath ?? options.databasePath ?? defaultDatabasePath,
  );

  const corpusClient = createClient({
    url: pathToFileURL(corpusDatabasePath).href,
  });

  try {
    return await validateEmbeddingsOnClient(corpusClient, modelId);
  } finally {
    await corpusClient.close();
  }
}

/**
 * Validate embedding coverage using a single libSQL client.
 *
 * The corpus is considered dense-ready when at least one chunk has an
 * embedding for the active model. The consolidated Turso schema intentionally
 * allows metadata-only chunks to omit embeddings, so the threshold is relaxed
 * to "any usable embeddings" while still surfacing the coverage ratio in the
 * report.
 *
 * @param {import('@libsql/client').Client} client - libSQL client.
 * @param {string} modelId - Embedding model identifier.
 * @returns {Promise<object>} Validation report.
 */
async function validateEmbeddingsOnClient(client, modelId) {
  const chunkCountResult = await client.execute({
    sql: 'SELECT COUNT(*) AS count FROM chunks',
    args: [],
  });
  const embeddingCountResult = await client.execute({
    sql: 'SELECT COUNT(*) AS count FROM chunks WHERE embedding IS NOT NULL AND embedding_model = ?',
    args: [modelId],
  });

  const chunkCount = Number(chunkCountResult.rows[0].count);
  const embeddingCount = Number(embeddingCountResult.rows[0].count);

  return buildValidationReport(chunkCount, embeddingCount, modelId);
}

/**
 * Build a validation report from chunk and embedding counts.
 *
 * The consolidated Turso schema intentionally allows metadata-only chunks to
 * omit embeddings, so the gate passes whenever at least one embedding for the
 * requested model is present (coverage > 0).
 *
 * @param {number} chunkCount - Total chunks in the corpus.
 * @param {number} embeddingCount - Chunks with embeddings for the model.
 * @param {string} modelId - Embedding model identifier.
 * @returns {object} Validation report.
 */
function buildValidationReport(chunkCount, embeddingCount, modelId) {
  const evidence = [];
  if (Number(embeddingCount) === 0) {
    evidence.push({
      actual: Number(embeddingCount),
      expected: Number(chunkCount),
      issue: 'no embeddings for model',
      model_id: modelId,
    });
  }

  return {
    chunk_count: Number(chunkCount),
    embedding_count: Number(embeddingCount),
    pass: evidence.length === 0,
    evidence,
    fixHint:
      evidence.length === 0 ? null : 'Run: node rag-index/embed-index.mjs',
    owner: '05-green-testing',
  };
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Embeddings validator',
      usage: 'node rag-index/validate-embeddings.mjs [--json]',
      options: [
        '--json                     Emit the standard gate JSON contract.',
        '--database <path>          Override the corpus database path.',
        '--model-id <id>            Restrict validation to one model id.',
        '--help                     Show this help.',
      ],
    });
    return;
  }

  try {
    const report = await validateEmbeddings({
      corpusDatabasePath: args.database,
      modelId: args['model-id'],
    });
    writeJsonOrText(report, Boolean(args.json), (payload) =>
      payload.pass
        ? 'Embeddings validation passed.'
        : `Embeddings validation failed: ${payload.evidence.map(({ issue }) => issue).join(', ')}`,
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
