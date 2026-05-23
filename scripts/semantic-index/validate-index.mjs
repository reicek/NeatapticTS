import Database from 'better-sqlite3';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { fail, parseCliArgs, printHelp, writeJsonOrText } from './cli-utils.mjs';
import { getFreshnessProof, isFreshDocument } from './freshness.mjs';
import { defaultDatabasePath, repoRoot } from './init-schema.mjs';

const DEFAULT_MIN_DOCUMENTS = 1;
const DEFAULT_MIN_CHUNKS = 1;
const DEFAULT_MAX_STALENESS_MS = 24 * 60 * 60 * 1000;

export async function validateSemanticIndex(input = {}) {
  const documents = input.documents ?? [];
  const freshnessChecks = input.freshnessChecks ?? [];
  const minDocuments = Number(input.minDocuments ?? DEFAULT_MIN_DOCUMENTS);
  const minChunks = Number(input.minChunks ?? 0);
  const chunks = Number(input.chunks ?? documents.length);
  const now = Number(input.now ?? Date.now());
  const maxStalenessMs = Number(input.maxStalenessMs ?? DEFAULT_MAX_STALENESS_MS);
  const failures = [];

  if (documents.length < minDocuments) failures.push(`Expected at least ${minDocuments} documents; found ${documents.length}.`);
  if (chunks < minChunks) failures.push(`Expected at least ${minChunks} chunks; found ${chunks}.`);

  const freshnessByPath = new Map(freshnessChecks.map((proof) => [proof.file_path, proof]));
  for (const documentRow of documents) {
    const freshnessProof = freshnessByPath.get(documentRow.file_path);
    if (freshnessProof && !isFreshDocument(documentRow, freshnessProof)) failures.push(`Stale freshness proof for ${documentRow.file_path}.`);
    if (documentRow.indexed_at && now - Number(documentRow.indexed_at) > maxStalenessMs) failures.push(`Index row too old for ${documentRow.file_path}.`);
  }

  return { ok: failures.length === 0, pass: failures.length === 0, failures, documents: documents.length, chunks };
}

export async function validateDatabase(options = {}) {
  const databasePath = path.resolve(options.databasePath ?? defaultDatabasePath);
  if (!existsSync(databasePath)) return { ok: false, pass: false, failures: [`Database not found: ${databasePath}`], documents: 0, chunks: 0 };

  const database = new Database(databasePath, { readonly: true, fileMustExist: true });
  const documents = database.prepare('SELECT file_path, mtime_ms, file_size, sha256, indexed_at FROM documents ORDER BY file_path').all();
  const [{ count: chunkCount }] = database.prepare('SELECT COUNT(*) AS count FROM chunks').all();
  database.close();

  const freshnessChecks = await Promise.all(documents.map(async (documentRow) => ({
    file_path: documentRow.file_path,
    ...(await getFreshnessProof(path.join(repoRoot, documentRow.file_path))),
  })));

  return validateSemanticIndex({ documents, freshnessChecks, chunks: chunkCount, minDocuments: options.minDocuments, minChunks: options.minChunks, maxStalenessMs: options.maxStalenessMs });
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Semantic index validator',
      usage: 'node scripts/semantic-index/validate-index.mjs [--json] [--min-documents 1] [--min-chunks 1]',
      options: ['--json                Emit JSON validation result', '--min-documents <n>   Minimum expected document rows (default: 1)', '--min-chunks <n>      Minimum expected chunk rows (default: 1)', '--max-age-ms <ms>     Maximum row age in milliseconds (default: 86400000 / 24 h)', '--database <path>     Path to SQLite database file (default: data/semantic-index.sqlite)', '--help                Show this help'],
    });
    return;
  }

  try {
    const result = await validateDatabase({ minDocuments: args['min-documents'], minChunks: args['min-chunks'] ?? DEFAULT_MIN_CHUNKS, maxStalenessMs: args['max-age-ms'], databasePath: args.database });
    writeJsonOrText(result, Boolean(args.json), (payload) => payload.pass ? `Semantic index valid: ${payload.documents} documents, ${payload.chunks} chunks` : `Semantic index invalid: ${payload.failures.join('; ')}`);
    if (!result.pass) process.exitCode = 1;
  } catch (error) {
    fail(error instanceof Error ? error.message : String(error), Boolean(args.json));
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();