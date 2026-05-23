/**
 * @description Validate that the dense embedding index in `data/embeddings.sqlite` is
 * complete: vector count for the active model must equal the chunk count in
 * `data/semantic-index.sqlite`. Emits the standard gate JSON contract
 * `{ pass, evidence, fixHint, owner }`. Use before relying on hybrid search results.
 *
 * @param {boolean} [--json]                     - Emit the standard gate JSON contract.
 * @param {string}  [--database <path>]          - Override corpus database path.
 * @param {string}  [--embeddings-database <p>]  - Override embeddings database path.
 * @param {string}  [--model-id <id>]            - Restrict validation to one model id.
 * @param {boolean} [--help]                     - Show help and exit.
 *
 * @returns {void} Exits 0 when the embedding count matches the corpus chunk count,
 *   1 when a mismatch is detected.
 */
import Database from 'better-sqlite3';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { fail, parseCliArgs, printHelp, writeJsonOrText } from './cli-utils.mjs';
import { defaultDatabasePath, repoRoot } from './init-schema.mjs';

export const DEFAULT_EMBEDDINGS_DATABASE_PATH = path.join(repoRoot, 'data', 'embeddings.sqlite');

export async function validateEmbeddings(options = {}) {
  const corpusDatabasePath = path.resolve(options.corpusDatabasePath ?? options.databasePath ?? defaultDatabasePath);
  const embeddingsDatabasePath = path.resolve(options.embeddingsDatabasePath ?? DEFAULT_EMBEDDINGS_DATABASE_PATH);
  const modelId = String(options.modelId ?? 'all-MiniLM-L6-v2');
  const corpusDatabase = new Database(corpusDatabasePath, { readonly: true, fileMustExist: true });
  const embeddingsDatabase = new Database(embeddingsDatabasePath, { readonly: true, fileMustExist: true });

  try {
    const [{ count: chunkCount }] = corpusDatabase.prepare('SELECT COUNT(*) AS count FROM chunks').all();
    const [{ count: embeddingCount }] = embeddingsDatabase.prepare(
      'SELECT COUNT(*) AS count FROM chunk_embeddings WHERE model_id = ?'
    ).all(modelId);

    const evidence = [];
    if (Number(embeddingCount) !== Number(chunkCount)) {
      evidence.push({
        actual: Number(embeddingCount),
        expected: Number(chunkCount),
        issue: 'embedding count mismatch',
        model_id: modelId,
      });
    }

    return {
      pass: evidence.length === 0,
      evidence,
      fixHint: evidence.length === 0 ? null : 'Run: node scripts/semantic-index/embed-index.mjs',
      owner: '05-green-testing',
    };
  } finally {
    corpusDatabase.close();
    embeddingsDatabase.close();
  }
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Embeddings validator',
      usage: 'node scripts/semantic-index/validate-embeddings.mjs [--json]',
      options: [
        '--json                     Emit the standard gate JSON contract.',
        '--database <path>          Override the semantic-index corpus database path.',
        '--embeddings-database <p>  Override the embeddings database path.',
        '--model-id <id>            Restrict validation to one model id.',
        '--help                     Show this help.',
      ],
    });
    return;
  }

  try {
    const report = await validateEmbeddings({
      corpusDatabasePath: args.database,
      embeddingsDatabasePath: args['embeddings-database'],
      modelId: args['model-id'],
    });
    writeJsonOrText(report, Boolean(args.json), (payload) => payload.pass
      ? 'Embeddings validation passed.'
      : `Embeddings validation failed: ${payload.evidence.map(({ issue }) => issue).join(', ')}`);
    if (!report.pass) process.exitCode = 1;
  } catch (error) {
    fail(error instanceof Error ? error.message : String(error), Boolean(args.json));
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();