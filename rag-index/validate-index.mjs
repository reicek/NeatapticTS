/**
 * @description Assert the health of `rag-index/data/turso-replica.sqlite`: minimum document and
 * chunk row counts, per-document freshness proof validity, and optional staleness age
 * enforcement. Emits the standard gate JSON contract `{ ok, pass, documents, chunks, failures }`.
 *
 * @param {boolean} [--json]                - Emit JSON validation result.
 * @param {number}  [--min-documents <n>]   - Minimum expected document rows (default: 1).
 * @param {number}  [--min-chunks <n>]      - Minimum expected chunk rows (default: 1).
 * @param {number}  [--max-age-ms <ms>]     - Maximum allowed row age in milliseconds (default: 86 400 000 / 24 h).
 * @param {string}  [--database <path>]     - Path to the SQLite database file (default: `rag-index/data/turso-replica.sqlite`).
 * @param {boolean} [--help]                - Show help and exit.
 *
 * @returns {void} Exits 0 when the index is healthy, 1 when any assertion fails.
 */
import { createClient } from '@libsql/client';
import { existsSync } from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  fail,
  parseCliArgs,
  printHelp,
  writeJsonOrText,
} from './cli-utils.mjs';
import { getFreshnessProof, isFreshDocument } from './freshness.mjs';
import { defaultDatabasePath, repoRoot } from './init-schema.mjs';

const DEFAULT_MIN_DOCUMENTS = 1;
const DEFAULT_MIN_CHUNKS = 1;
const DEFAULT_MAX_STALENESS_MS = 24 * 60 * 60 * 1000;
const REBUILD_COMMAND = 'node rag-index/build-index.mjs';
const STALE_FIX_HINT = `Stale paths detected. Run: ${REBUILD_COMMAND}`;
const MISSING_FIX_HINT = `Missing paths detected. Run: ${REBUILD_COMMAND}`;
const OVER_AGE_FIX_HINT = `Over-age paths detected. Run: ${REBUILD_COMMAND}`;
const GENERIC_FIX_HINT = `Run: ${REBUILD_COMMAND}`;

export async function validateSemanticIndex(input = {}) {
  const documents = input.documents ?? [];
  const freshnessChecks = input.freshnessChecks ?? [];
  const minDocuments = Number(input.minDocuments ?? DEFAULT_MIN_DOCUMENTS);
  const minChunks = Number(input.minChunks ?? 0);
  const chunks = Number(input.chunks ?? documents.length);
  const now = Number(input.now ?? Date.now());
  const maxStalenessMs = Number(
    input.maxStalenessMs ?? DEFAULT_MAX_STALENESS_MS,
  );
  const failures = [];
  const stalePaths = [];
  const missingPaths = [];
  const overAgePaths = [];

  if (documents.length < minDocuments)
    failures.push(
      `Expected at least ${minDocuments} documents; found ${documents.length}.`,
    );
  if (chunks < minChunks)
    failures.push(`Expected at least ${minChunks} chunks; found ${chunks}.`);

  const freshnessByPath = new Map(
    freshnessChecks.map((proof) => [proof.file_path, proof]),
  );
  for (const documentRow of documents) {
    const freshnessProof = freshnessByPath.get(documentRow.file_path);
    if (freshnessProof?.missing === true) {
      failures.push(`Indexed file is missing: ${documentRow.file_path}.`);
      pushUnique(missingPaths, documentRow.file_path);
      continue;
    }

    if (freshnessProof && !isFreshDocument(documentRow, freshnessProof)) {
      failures.push(`Stale freshness proof for ${documentRow.file_path}.`);
      pushUnique(stalePaths, documentRow.file_path);
    }

    if (
      documentRow.indexed_at &&
      now - Number(documentRow.indexed_at) > maxStalenessMs
    ) {
      failures.push(`Index row too old for ${documentRow.file_path}.`);
      pushUnique(overAgePaths, documentRow.file_path);
    }
  }

  return createValidationResult({
    failures,
    documents: documents.length,
    chunks,
    stalePaths,
    missingPaths,
    overAgePaths,
  });
}

export async function validateDatabase(options = {}) {
  if (options.client) {
    const docsResult = await options.client.execute({
      sql: 'SELECT file_path, mtime_ms, file_size, sha256, indexed_at FROM documents ORDER BY file_path',
      args: [],
    });
    const chunkCountResult = await options.client.execute({
      sql: 'SELECT COUNT(*) AS count FROM chunks',
      args: [],
    });
    const documents = docsResult.rows;
    const chunkCount = Number(chunkCountResult.rows[0].count);

    const freshnessChecks = await Promise.all(
      documents.map(async (documentRow) => {
        const absolutePath = path.join(repoRoot, documentRow.file_path);

        try {
          return {
            file_path: documentRow.file_path,
            ...(await getFreshnessProof(absolutePath)),
          };
        } catch (error) {
          if (
            error &&
            typeof error === 'object' &&
            'code' in error &&
            error.code === 'ENOENT'
          ) {
            return {
              file_path: documentRow.file_path,
              missing: true,
            };
          }

          throw error;
        }
      }),
    );

    return validateSemanticIndex({
      documents,
      freshnessChecks,
      chunks: chunkCount,
      minDocuments: options.minDocuments,
      minChunks: options.minChunks,
      maxStalenessMs: options.maxStalenessMs,
    });
  }

  const databasePath = path.resolve(
    options.databasePath ?? defaultDatabasePath,
  );
  if (!existsSync(databasePath)) {
    return createValidationResult({
      failures: [`Database not found: ${databasePath}`],
      documents: 0,
      chunks: 0,
    });
  }

  const database = createClient({ url: pathToFileURL(databasePath).href });
  const docsResult = await database.execute({
    sql: 'SELECT file_path, mtime_ms, file_size, sha256, indexed_at FROM documents ORDER BY file_path',
    args: [],
  });
  const chunkCountResult = await database.execute({
    sql: 'SELECT COUNT(*) AS count FROM chunks',
    args: [],
  });
  const documents = docsResult.rows;
  const chunkCount = Number(chunkCountResult.rows[0].count);
  await database.close();

  const freshnessChecks = await Promise.all(
    documents.map(async (documentRow) => {
      const absolutePath = path.join(repoRoot, documentRow.file_path);

      try {
        return {
          file_path: documentRow.file_path,
          ...(await getFreshnessProof(absolutePath)),
        };
      } catch (error) {
        if (
          error &&
          typeof error === 'object' &&
          'code' in error &&
          error.code === 'ENOENT'
        ) {
          return {
            file_path: documentRow.file_path,
            missing: true,
          };
        }

        throw error;
      }
    }),
  );

  return validateSemanticIndex({
    documents,
    freshnessChecks,
    chunks: chunkCount,
    minDocuments: options.minDocuments,
    minChunks: options.minChunks,
    maxStalenessMs: options.maxStalenessMs,
  });
}

function createValidationResult({
  failures,
  documents,
  chunks,
  stalePaths = [],
  missingPaths = [],
  overAgePaths = [],
}) {
  const pass = failures.length === 0;

  return {
    ok: pass,
    pass,
    failures,
    documents,
    chunks,
    stale_paths: stalePaths,
    missing_paths: missingPaths,
    over_age_paths: overAgePaths,
    fixHint: pass
      ? null
      : resolveFixHint({ stalePaths, missingPaths, overAgePaths, failures }),
  };
}

function resolveFixHint({ stalePaths, missingPaths, overAgePaths, failures }) {
  if (stalePaths.length > 0) return STALE_FIX_HINT;
  if (missingPaths.length > 0) return MISSING_FIX_HINT;
  if (overAgePaths.length > 0) return OVER_AGE_FIX_HINT;
  return failures.length > 0 ? GENERIC_FIX_HINT : null;
}

function pushUnique(values, value) {
  if (!values.includes(value)) values.push(value);
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Semantic index validator',
      usage:
        'node rag-index/validate-index.mjs [--json] [--min-documents 1] [--min-chunks 1]',
      options: [
        '--json                Emit JSON validation result',
        '--min-documents <n>   Minimum expected document rows (default: 1)',
        '--min-chunks <n>      Minimum expected chunk rows (default: 1)',
        '--max-age-ms <ms>     Maximum row age in milliseconds (default: 86400000 / 24 h)',
        '--database <path>     Path to SQLite database file (default: rag-index/data/turso-replica.sqlite)',
        '--help                Show this help',
      ],
    });
    return;
  }

  try {
    const result = await validateDatabase({
      minDocuments: args['min-documents'],
      minChunks: args['min-chunks'] ?? DEFAULT_MIN_CHUNKS,
      maxStalenessMs: args['max-age-ms'],
      databasePath: args.database,
    });
    writeJsonOrText(result, Boolean(args.json), (payload) =>
      payload.pass
        ? `Semantic index valid: ${payload.documents} documents, ${payload.chunks} chunks`
        : `Semantic index invalid: ${payload.failures.join('; ')}`,
    );
    if (!result.pass) process.exitCode = 1;
  } catch (error) {
    fail(
      error instanceof Error ? error.message : String(error),
      Boolean(args.json),
    );
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();
