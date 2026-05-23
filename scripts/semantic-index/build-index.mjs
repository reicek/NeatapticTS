import fg from 'fast-glob';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { chunkMarkdown } from './chunker.mjs';
import { fail, parseCliArgs, printHelp, toRepoRelative, writeJsonOrText } from './cli-utils.mjs';
import { getFreshnessProof, isFreshDocument } from './freshness.mjs';
import { defaultDatabasePath, initSemanticIndex, repoRoot } from './init-schema.mjs';

const CORPUS_SOURCES = [
  { family: 'readme', patterns: ['src/**/README.md'] },
  { family: 'skill', patterns: ['.github/skills/**/SKILL.md'] },
  { family: 'agent', patterns: ['.github/agents/*.agent.md'] },
  { family: 'plan', patterns: ['plans/**/*.md'], ignore: ['plans/completed/**'] },
  { family: 'completed-plan', patterns: ['plans/completed/**/*.md'] },
  { family: 'demo', patterns: ['examples/**/README.md', 'examples/**/*.ts'] },
  { family: 'benchmark', patterns: ['benchmarks/README.md', 'benchmarks/**/*.test.ts'] },
  { family: 'root-doc', patterns: ['README.md', 'CLAUDE.md', 'STYLEGUIDE.md', 'CONTRIBUTING.md'] },
  { family: 'copilot-instructions', patterns: ['.github/copilot-instructions.md'] },
];

export async function buildSemanticIndex(options = {}) {
  const databasePath = path.resolve(options.databasePath ?? defaultDatabasePath);
  const documents = options.corpusDocuments ?? await collectCorpusDocuments();
  const summary = {
    databasePath,
    scanned: documents.length,
    indexed: 0,
    skipped: 0,
    chunks: 0,
    purged: 0,
    dryRun: Boolean(options.dryRun),
    totalDocuments: documents.length,
    newDocuments: 0,
    elapsedMs: 0,
  };

  if (options.dryRun) return summary;

  const database = await initSemanticIndex({ databasePath });
  summary.purged = deleteMissingDocuments(database, documents);
  const existingDocument = database.prepare('SELECT * FROM documents WHERE file_path = ?');
  const upsertDocument = database.prepare(`
    INSERT INTO documents(file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
    VALUES (?, ?, ?, ?, ?, ?)
    ON CONFLICT(file_path) DO UPDATE SET
      doc_family = excluded.doc_family,
      mtime_ms = excluded.mtime_ms,
      file_size = excluded.file_size,
      sha256 = excluded.sha256,
      indexed_at = excluded.indexed_at
  `);
  const findDocumentId = database.prepare('SELECT doc_id FROM documents WHERE file_path = ?');
  const deleteChunks = database.prepare('DELETE FROM chunks WHERE doc_id = ?');
  const insertChunk = database.prepare(`
    INSERT INTO chunks(doc_id, chunk_index, heading_path, body_text, char_start, char_end)
    VALUES (?, ?, ?, ?, ?, ?)
  `);
  const buildLoopStartTime = Date.now();

  const indexDocument = database.transaction((documentRecord, freshnessProof, chunks) => {
    upsertDocument.run(documentRecord.filePath, documentRecord.family, freshnessProof.mtime_ms, freshnessProof.size, freshnessProof.sha256, Date.now());
    const { doc_id: documentId } = findDocumentId.get(documentRecord.filePath);
    deleteChunks.run(documentId);
    chunks.forEach((chunk, chunkIndex) => {
      insertChunk.run(documentId, chunkIndex, chunk.heading_path, chunk.body_text, chunk.char_start, chunk.char_end);
    });
  });

  for (const documentRecord of documents) {
    const absolutePath = path.join(repoRoot, documentRecord.filePath);
    const freshnessProof = await getFreshnessProof(absolutePath);
    const currentRow = existingDocument.get(documentRecord.filePath);

    if (!options.force && isFreshDocument(currentRow, freshnessProof)) {
      summary.skipped += 1;
      continue;
    }

    if (!currentRow) summary.newDocuments += 1;

    const markdownText = await readFile(absolutePath, 'utf8');
    const chunks = chunkMarkdown(markdownText);
    indexDocument(documentRecord, freshnessProof, chunks);
    summary.indexed += 1;
    summary.chunks += chunks.length;
  }

  summary.elapsedMs = Date.now() - buildLoopStartTime;

  database.exec("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')");
  database.close();
  return summary;
}

export function deleteMissingDocuments(database, documents) {
  const currentDocumentPaths = new Set(documents.map(({ filePath }) => filePath));
  const existingDocumentPaths = database.prepare('SELECT file_path FROM documents').all();
  const deleteDocument = database.prepare('DELETE FROM documents WHERE file_path = ?');

  const purgeDeletedDocuments = database.transaction(() => {
    let purgedCount = 0;
    existingDocumentPaths.forEach(({ file_path: filePath }) => {
      if (currentDocumentPaths.has(filePath)) return;
      deleteDocument.run(filePath);
      purgedCount += 1;
    });
    return purgedCount;
  });

  return purgeDeletedDocuments();
}

async function collectCorpusDocuments() {
  const records = await Promise.all(CORPUS_SOURCES.map(async (source) => {
    const entries = await fg(source.patterns, {
      cwd: repoRoot,
      absolute: false,
      onlyFiles: true,
      dot: true,
      ignore: source.ignore ?? [],
    });
    return entries.toSorted().map((filePath) => ({ filePath: toRepoRelative(path.join(repoRoot, filePath)), family: source.family }));
  }));

  return records.flat();
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Semantic index builder',
      usage: 'node scripts/semantic-index/build-index.mjs [--dry-run] [--force] [--json] [--database path]',
      options: ['--dry-run         Scan corpus without writing SQLite rows', '--force           Re-index unchanged documents even if freshness proof matches', '--json            Emit JSON summary', '--json-health     Emit compact health summary JSON', '--database <path> Path to SQLite database file (default: data/semantic-index.sqlite)', '--help            Show this help'],
    });
    return;
  }

  const emitJsonHealth = Boolean(args['json-health']);

  try {
    const summary = await buildSemanticIndex({ dryRun: Boolean(args['dry-run']), force: Boolean(args.force), databasePath: args.database });
    if (emitJsonHealth) {
      console.log(JSON.stringify(createJsonHealthSummary(summary), null, 2));
      return;
    }

    writeJsonOrText(summary, Boolean(args.json), (payload) => `Semantic index: scanned ${payload.scanned}, indexed ${payload.indexed}, skipped ${payload.skipped}, chunks ${payload.chunks}${payload.dryRun ? ' (dry run)' : ''}`);
  } catch (error) {
    if (emitJsonHealth) {
      console.log(JSON.stringify(createJsonHealthFailure(error, args.database), null, 2));
      process.exitCode = 1;
      return;
    }

    fail(error instanceof Error ? error.message : String(error), Boolean(args.json));
  }
}

function createJsonHealthSummary(summary) {
  return {
    status: 'ok',
    total_documents: Number(summary.totalDocuments ?? summary.scanned ?? 0),
    new_documents: Number(summary.newDocuments ?? 0),
    removed_documents: Number(summary.purged ?? 0),
    elapsed_ms: Number(summary.elapsedMs ?? 0),
    index_path: toRepoRelative(path.resolve(summary.databasePath ?? defaultDatabasePath)),
  };
}

function createJsonHealthFailure(error, databasePath) {
  const message = error instanceof Error ? error.message : String(error);

  return {
    status: 'error',
    total_documents: 0,
    new_documents: 0,
    removed_documents: 0,
    elapsed_ms: 0,
    index_path: toRepoRelative(path.resolve(databasePath ?? defaultDatabasePath)),
    message,
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();