import Database from 'better-sqlite3';
import { existsSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import {
  fail,
  parseCliArgs,
  printHelp,
  writeJsonOrText,
} from './cli-utils.mjs';
import { defaultDatabasePath, repoRoot } from './init-schema.mjs';

const SNAPSHOT_SCHEMA_VERSION = '1';
const defaultOutputPath = path.join(
  repoRoot,
  'docs',
  'assets',
  'semantic-snapshot.json',
);

export async function buildBrowserSnapshot(options = {}) {
  const databasePath = path.resolve(
    options.databasePath ?? defaultDatabasePath,
  );
  const outputPath = path.resolve(options.outputPath ?? defaultOutputPath);

  if (!existsSync(databasePath)) {
    throw new Error(
      `Semantic SQLite index not found at ${databasePath}. Run "npm run index:build" before building the browser snapshot, or pass --database <path>.`,
    );
  }

  const database = new Database(databasePath, {
    readonly: true,
    fileMustExist: true,
  });
  try {
    const documents = readSnapshotDocuments(database);
    const snapshot = {
      schema_version: SNAPSHOT_SCHEMA_VERSION,
      generated_at: new Date().toISOString(),
      families: [...new Set(documents.map(({ family }) => family))].toSorted(),
      documents,
    };

    if (!options.dryRun) {
      await mkdir(path.dirname(outputPath), { recursive: true });
      await writeFile(
        outputPath,
        `${JSON.stringify(snapshot, null, 2)}\n`,
        'utf8',
      );
    }

    return {
      outputPath,
      databasePath,
      schema_version: snapshot.schema_version,
      generated_at: snapshot.generated_at,
      families: snapshot.families,
      documents: snapshot.documents.length,
      chunks: snapshot.documents.reduce(
        (chunkCount, documentRecord) =>
          chunkCount + documentRecord.chunks.length,
        0,
      ),
      dryRun: Boolean(options.dryRun),
    };
  } finally {
    database.close();
  }
}

export function createBrowserSnapshot(databasePath = defaultDatabasePath) {
  const database = new Database(path.resolve(databasePath), {
    readonly: true,
    fileMustExist: true,
  });
  try {
    const documents = readSnapshotDocuments(database);
    return {
      schema_version: SNAPSHOT_SCHEMA_VERSION,
      generated_at: new Date().toISOString(),
      families: [...new Set(documents.map(({ family }) => family))].toSorted(),
      documents,
    };
  } finally {
    database.close();
  }
}

function readSnapshotDocuments(database) {
  const documentRows = database
    .prepare(
      'SELECT doc_id, file_path, doc_family AS family FROM documents ORDER BY file_path ASC, doc_id ASC',
    )
    .all();
  const chunkRows = database
    .prepare(
      'SELECT chunk_id, doc_id, heading_path, body_text, char_start, char_end FROM chunks ORDER BY doc_id ASC, chunk_index ASC, chunk_id ASC',
    )
    .all();
  const chunksByDocumentId = groupChunksByDocumentId(chunkRows);

  return documentRows.map((documentRow) => ({
    doc_id: documentRow.doc_id,
    file_path: normalizeRepoPath(documentRow.file_path),
    family: documentRow.family,
    chunks: chunksByDocumentId.get(documentRow.doc_id) ?? [],
  }));
}

function groupChunksByDocumentId(chunkRows) {
  return chunkRows.reduce((chunksByDocumentId, chunkRow) => {
    const chunks = chunksByDocumentId.get(chunkRow.doc_id) ?? [];
    chunks.push({
      chunk_id: chunkRow.chunk_id,
      heading_path: chunkRow.heading_path ?? '',
      body_text: chunkRow.body_text,
      char_start: chunkRow.char_start,
      char_end: chunkRow.char_end,
    });
    chunksByDocumentId.set(chunkRow.doc_id, chunks);
    return chunksByDocumentId;
  }, new Map());
}

function normalizeRepoPath(filePath) {
  return filePath.replaceAll('\\', '/');
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Semantic browser snapshot builder',
      usage:
        'node scripts/semantic-index/build-browser-snapshot.mjs [--database path] [--output path] [--dry-run] [--json]',
      options: [
        '--database <path> Path to SQLite semantic index (default: data/semantic-index.sqlite)',
        '--output <path>   Path to browser JSON snapshot (default: docs/assets/semantic-snapshot.json)',
        '--dry-run         Read and summarize the snapshot without writing JSON',
        '--json            Emit JSON summary',
        '--help            Show this help',
      ],
    });
    return;
  }

  try {
    const summary = await buildBrowserSnapshot({
      databasePath: args.database,
      outputPath: args.output,
      dryRun: Boolean(args['dry-run']),
    });
    writeJsonOrText(
      summary,
      Boolean(args.json),
      (payload) =>
        `Semantic browser snapshot: ${payload.documents} documents, ${payload.chunks} chunks${payload.dryRun ? ' (dry run)' : ` -> ${payload.outputPath}`}`,
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
