/**
 * @description Scan the NeatapticTS corpus, chunk documents by family, and populate
 * `data/semantic-index.sqlite` with BM25-searchable content. Uses v2 semantic chunkers
 * (AST-aware TypeScript chunking and structure-aware markdown chunking) that produce
 * enriched metadata columns: parent_chunk_id, depth, context_header, symbol_name,
 * signature_text, jsdoc_text, export_type, module_path.
 *
 * @param {boolean} [--dry-run] - Scan corpus without writing SQLite rows.
 * @param {boolean} [--force] - Re-index unchanged documents even if freshness proof matches.
 * @param {boolean} [--json] - Emit JSON summary `{ scanned, indexed, skipped, chunks, elapsedMs }`.
 * @param {boolean} [--json-health] - Emit compact health summary JSON.
 * @param {string}  [--database <path>] - Path to the SQLite database file (default: `data/semantic-index.sqlite`).
 * @param {boolean} [--help] - Show help and exit.
 *
 * @returns {void} Exits 0 on success, 1 on fatal error. JSON summary written to stdout
 *   when `--json` is passed.
 */
import fg from 'fast-glob';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { chunkMarkdownV2 } from './chunker-v2.mjs';
import {
  fail,
  parseCliArgs,
  printHelp,
  toRepoRelative,
  writeJsonOrText,
} from './cli-utils.mjs';
import { getFreshnessProof, isFreshDocument } from './freshness.mjs';
import {
  defaultDatabasePath,
  initSemanticIndex,
  repoRoot,
} from './init-schema.mjs';
import {
  enrichChunkMetadata,
  enrichDocumentMetadata,
  loadCoverageReport,
} from './metadata-enrichment.mjs';
import { chunkTypeScriptSourcesV2 } from './ts-chunker-v2.mjs';

const CORPUS_SOURCES = [
  { family: 'readme', patterns: ['src/**/README.md'] },
  {
    family: 'ts-source',
    patterns: ['src/**/*.ts'],
    ignore: ['src/**/*.d.ts', 'src/**/*.test.ts', 'src/**/*.spec.ts'],
  },
  { family: 'skill', patterns: ['.github/skills/**/SKILL.md'] },
  { family: 'agent', patterns: ['.github/agents/*.agent.md'] },
  {
    family: 'plan',
    patterns: ['plans/**/*.md'],
    ignore: ['plans/completed/**'],
  },
  { family: 'completed-plan', patterns: ['plans/completed/**/*.md'] },
  { family: 'demo', patterns: ['examples/**/README.md', 'examples/**/*.ts'] },
  {
    family: 'benchmark',
    patterns: ['benchmarks/README.md', 'benchmarks/**/*.test.ts'],
  },
  {
    family: 'root-doc',
    patterns: ['README.md', 'CLAUDE.md', 'STYLEGUIDE.md', 'CONTRIBUTING.md'],
  },
  {
    family: 'copilot-instructions',
    patterns: ['.github/copilot-instructions.md'],
  },
];

export async function buildSemanticIndex(options = {}) {
  const databasePath = path.resolve(
    options.databasePath ?? defaultDatabasePath,
  );
  const documents = options.corpusDocuments ?? (await collectCorpusDocuments());
  const tsSourceChunksByFilePath =
    await collectTypeScriptChunksByFilePath(documents);
  const coverageReport =
    options.coverageReport ?? (await loadCoverageReport(repoRoot));
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
  // Run v3 schema migration to ensure metadata columns exist
  const { migrateSchemaV2ToV3 } = await import('./migrate-schema.mjs');
  migrateSchemaV2ToV3({ databasePath });

  summary.purged = deleteMissingDocuments(database, documents);
  const existingDocument = database.prepare(
    'SELECT * FROM documents WHERE file_path = ?',
  );
  const upsertDocument = database.prepare(`
    INSERT INTO documents(file_path, doc_family, mtime_ms, file_size, sha256, indexed_at, arch_layer, test_coverage, source_path_pattern)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(file_path) DO UPDATE SET
      doc_family = excluded.doc_family,
      mtime_ms = excluded.mtime_ms,
      file_size = excluded.file_size,
      sha256 = excluded.sha256,
      indexed_at = excluded.indexed_at,
      arch_layer = excluded.arch_layer,
      test_coverage = excluded.test_coverage,
      source_path_pattern = excluded.source_path_pattern
  `);
  const findDocumentId = database.prepare(
    'SELECT doc_id FROM documents WHERE file_path = ?',
  );
  const deleteChunks = database.prepare('DELETE FROM chunks WHERE doc_id = ?');
  const insertChunk = database.prepare(`
    INSERT INTO chunks(doc_id, chunk_index, heading_path, body_text, char_start, char_end,
      parent_chunk_id, depth, context_header, symbol_name, signature_text, jsdoc_text, export_type, module_path,
      arch_layer, jsdoc_quality, jsdoc_word_count, cyclomatic_complexity, test_coverage, source_path_pattern)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
  `);
  const buildLoopStartTime = Date.now();

  const indexDocument = database.transaction(
    (documentRecord, freshnessProof, chunks) => {
      // Step 1: Enrich document-level metadata
      const docMetadata = enrichDocumentMetadata(
        documentRecord,
        coverageReport,
      );

      upsertDocument.run(
        documentRecord.filePath,
        documentRecord.family,
        freshnessProof.mtime_ms,
        freshnessProof.size,
        freshnessProof.sha256,
        Date.now(),
        docMetadata.arch_layer,
        docMetadata.test_coverage,
        docMetadata.source_path_pattern,
      );
      const { doc_id: documentId } = findDocumentId.get(
        documentRecord.filePath,
      );
      deleteChunks.run(documentId);

      // Insert all chunks with sequentially assigned chunk_index values.
      // Parent chunks (parent_chunk_id == null) are inserted first to collect
      // their database IDs, then child chunks (parent_chunk_id != null) are
      // inserted with parent_chunk_id resolved to the parent's database ID.
      // For child chunks, parent_chunk_id temporarily holds the parent's
      // chunk_index (assigned by the chunker), which we resolve via chunkIdMap.
      // Note: we separate by parent_chunk_id (not depth) because markdown chunks
      // use depth for heading level (0-6) without a parent-child relationship.
      const chunkIdMap = new Map(); // chunker chunk_index → database rowid (for parent chunks)
      const parentChunks = chunks.filter(
        (chunk) => chunk.parent_chunk_id == null,
      );
      const childChunks = chunks.filter(
        (chunk) => chunk.parent_chunk_id != null,
      );

      // First pass: insert parent chunks with sequential chunk_index starting at 0.
      for (
        let parentIndex = 0;
        parentIndex < parentChunks.length;
        parentIndex += 1
      ) {
        const chunk = parentChunks[parentIndex];
        const originalIndex = chunk.chunk_index ?? chunks.indexOf(chunk);
        // Step 2: Enrich chunk-level metadata
        const chunkMeta = enrichChunkMetadata(
          {
            jsdoc_text: chunk.jsdoc_text ?? null,
            export_type: chunk.export_type ?? null,
            module_path: chunk.module_path ?? null,
            file_path: documentRecord.filePath,
            family: documentRecord.family,
            body_text: chunk.body_text,
          },
          coverageReport,
        );

        insertChunk.run(
          documentId,
          parentIndex, // Sequential chunk_index for all parent chunks
          chunk.heading_path ?? null,
          chunk.body_text,
          chunk.char_start ?? 0,
          chunk.char_end ?? chunk.body_text.length,
          null, // parent_chunk_id is null for parent chunks
          chunk.depth ?? 0,
          chunk.context_header ?? null,
          chunk.symbol_name ?? null,
          chunk.signature_text ?? null,
          chunk.jsdoc_text ?? null,
          chunk.export_type ?? null,
          chunk.module_path ?? null,
          chunkMeta.arch_layer,
          chunkMeta.jsdoc_quality,
          chunkMeta.jsdoc_word_count,
          chunkMeta.cyclomatic_complexity,
          chunkMeta.test_coverage,
          chunkMeta.source_path_pattern,
        );
        const rowId = database
          .prepare('SELECT last_insert_rowid() AS id')
          .get();
        chunkIdMap.set(originalIndex, rowId.id);
      }

      // Second pass: insert child chunks with sequential chunk_index and resolved parent_chunk_id.
      for (
        let childIndex = 0;
        childIndex < childChunks.length;
        childIndex += 1
      ) {
        const chunk = childChunks[childIndex];
        // parent_chunk_id holds the parent's original chunk_index from the chunker.
        // Resolve it to the parent's database chunk_id via chunkIdMap.
        const parentChunkIndex = chunk.parent_chunk_id;
        const resolvedParentId = chunkIdMap.get(parentChunkIndex) ?? null;
        // Step 2: Enrich chunk-level metadata
        const chunkMeta = enrichChunkMetadata(
          {
            jsdoc_text: chunk.jsdoc_text ?? null,
            export_type: chunk.export_type ?? null,
            module_path: chunk.module_path ?? null,
            file_path: documentRecord.filePath,
            family: documentRecord.family,
            body_text: chunk.body_text,
          },
          coverageReport,
        );

        insertChunk.run(
          documentId,
          parentChunks.length + childIndex, // Sequential chunk_index for child chunks
          chunk.heading_path ?? null,
          chunk.body_text,
          chunk.char_start ?? 0,
          chunk.char_end ?? chunk.body_text.length,
          resolvedParentId,
          chunk.depth ?? 1,
          chunk.context_header ?? null,
          chunk.symbol_name ?? null,
          chunk.signature_text ?? null,
          chunk.jsdoc_text ?? null,
          chunk.export_type ?? null,
          chunk.module_path ?? null,
          chunkMeta.arch_layer,
          chunkMeta.jsdoc_quality,
          chunkMeta.jsdoc_word_count,
          chunkMeta.cyclomatic_complexity,
          chunkMeta.test_coverage,
          chunkMeta.source_path_pattern,
        );
      }
    },
  );

  for (const documentRecord of documents) {
    const absolutePath = path.join(repoRoot, documentRecord.filePath);
    const freshnessProof = await getFreshnessProof(absolutePath);
    const currentRow = existingDocument.get(documentRecord.filePath);

    if (!options.force && isFreshDocument(currentRow, freshnessProof)) {
      summary.skipped += 1;
      continue;
    }

    if (!currentRow) summary.newDocuments += 1;

    const chunks = await chunkDocument(
      documentRecord,
      absolutePath,
      tsSourceChunksByFilePath,
    );
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
  const currentDocumentPaths = new Set(
    documents.map(({ filePath }) => filePath),
  );
  const existingDocumentPaths = database
    .prepare('SELECT file_path FROM documents')
    .all();
  const deleteDocument = database.prepare(
    'DELETE FROM documents WHERE file_path = ?',
  );

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
  const records = await Promise.all(
    CORPUS_SOURCES.map(async (source) => {
      const entries = await fg(source.patterns, {
        cwd: repoRoot,
        absolute: false,
        onlyFiles: true,
        dot: true,
        ignore: source.ignore ?? [],
      });
      return entries.toSorted().map((filePath) => ({
        filePath: toRepoRelative(path.join(repoRoot, filePath)),
        family: source.family,
      }));
    }),
  );

  return records.flat();
}

async function chunkDocument(
  documentRecord,
  absolutePath,
  tsSourceChunksByFilePath,
) {
  if (documentRecord.family === 'ts-source') {
    return tsSourceChunksByFilePath.get(documentRecord.filePath) ?? [];
  }

  const markdownText = await readFile(absolutePath, 'utf8');
  return chunkMarkdownV2(markdownText, { filePath: documentRecord.filePath });
}

async function collectTypeScriptChunksByFilePath(documents) {
  const tsSourcePaths = documents
    .filter(({ family }) => family === 'ts-source')
    .map(({ filePath }) => path.join(repoRoot, filePath));
  if (tsSourcePaths.length === 0) return new Map();

  const tsSourceChunks = await chunkTypeScriptSourcesV2({
    sourcePaths: tsSourcePaths,
  });
  return tsSourceChunks.reduce((chunksByFilePath, chunk) => {
    const existingChunks = chunksByFilePath.get(chunk.file_path) ?? [];
    existingChunks.push(chunk);
    chunksByFilePath.set(chunk.file_path, existingChunks);
    return chunksByFilePath;
  }, new Map());
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Semantic index builder',
      usage:
        'node scripts/semantic-index/build-index.mjs [--dry-run] [--force] [--json] [--database path]',
      options: [
        '--dry-run         Scan corpus without writing SQLite rows',
        '--force           Re-index unchanged documents even if freshness proof matches',
        '--json            Emit JSON summary',
        '--json-health     Emit compact health summary JSON',
        '--database <path> Path to SQLite database file (default: data/semantic-index.sqlite)',
        '--with-graph      Build entity/relationship graph after corpus build',
        '--help            Show this help',
      ],
    });
    return;
  }

  const emitJsonHealth = Boolean(args['json-health']);
  const withGraph = Boolean(args['with-graph']);

  try {
    const summary = await buildSemanticIndex({
      dryRun: Boolean(args['dry-run']),
      force: Boolean(args.force),
      databasePath: args.database,
    });
    if (emitJsonHealth) {
      console.log(JSON.stringify(createJsonHealthSummary(summary), null, 2));
      return;
    }

    writeJsonOrText(
      summary,
      Boolean(args.json),
      (payload) =>
        `Semantic index: scanned ${payload.scanned}, indexed ${payload.indexed}, skipped ${payload.skipped}, chunks ${payload.chunks}${payload.dryRun ? ' (dry run)' : ''}`,
    );

    // Optionally build entity graph after corpus build.
    if (withGraph) {
      const { buildEntityGraph } = await import('./build-entity-graph.mjs');
      const graphSummary = await buildEntityGraph({
        databasePath: args.database,
      });
      if (emitJsonHealth) {
        console.log(JSON.stringify(graphSummary, null, 2));
      } else {
        writeJsonOrText(
          graphSummary,
          Boolean(args.json),
          (payload) =>
            `Entity graph: ${payload.entities} entities, ${payload.edges} edges`,
        );
      }
    }
  } catch (error) {
    if (emitJsonHealth) {
      console.log(
        JSON.stringify(createJsonHealthFailure(error, args.database), null, 2),
      );
      process.exitCode = 1;
      return;
    }

    fail(
      error instanceof Error ? error.message : String(error),
      Boolean(args.json),
    );
  }
}

function createJsonHealthSummary(summary) {
  return {
    status: 'ok',
    total_documents: Number(summary.totalDocuments ?? summary.scanned ?? 0),
    new_documents: Number(summary.newDocuments ?? 0),
    removed_documents: Number(summary.purged ?? 0),
    elapsed_ms: Number(summary.elapsedMs ?? 0),
    index_path: toRepoRelative(
      path.resolve(summary.databasePath ?? defaultDatabasePath),
    ),
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
    index_path: toRepoRelative(
      path.resolve(databasePath ?? defaultDatabasePath),
    ),
    message,
  };
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href)
  await main();
