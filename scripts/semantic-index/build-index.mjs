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
  const summary = { databasePath, scanned: documents.length, indexed: 0, skipped: 0, chunks: 0, purged: 0, dryRun: Boolean(options.dryRun) };

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

    const markdownText = await readFile(absolutePath, 'utf8');
    const chunks = chunkMarkdown(markdownText);
    indexDocument(documentRecord, freshnessProof, chunks);
    summary.indexed += 1;
    summary.chunks += chunks.length;
  }

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
      options: ['--dry-run         Scan corpus without writing SQLite rows', '--force           Re-index unchanged documents even if freshness proof matches', '--json            Emit JSON summary', '--database <path> Path to SQLite database file (default: data/semantic-index.sqlite)', '--help            Show this help'],
    });
    return;
  }

  try {
    const summary = await buildSemanticIndex({ dryRun: Boolean(args['dry-run']), force: Boolean(args.force), databasePath: args.database });
    writeJsonOrText(summary, Boolean(args.json), (payload) => `Semantic index: scanned ${payload.scanned}, indexed ${payload.indexed}, skipped ${payload.skipped}, chunks ${payload.chunks}${payload.dryRun ? ' (dry run)' : ''}`);
  } catch (error) {
    fail(error instanceof Error ? error.message : String(error), Boolean(args.json));
  }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();