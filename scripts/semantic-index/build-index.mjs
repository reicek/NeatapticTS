/**
 * @description Scan the NeatapticTS corpus, chunk documents by family, and populate
 * `data/turso-replica.sqlite` with BM25-searchable content. Uses v2 semantic chunkers
 * (AST-aware TypeScript chunking and structure-aware markdown chunking) that produce
 * enriched metadata columns: parent_chunk_id, depth, context_header, symbol_name,
 * signature_text, jsdoc_text, export_type, module_path.
 *
 * @param {boolean} [--dry-run] - Scan corpus without writing SQLite rows.
 * @param {boolean} [--force] - Re-index unchanged documents even if freshness proof matches.
 * @param {boolean} [--json] - Emit JSON summary `{ scanned, indexed, skipped, chunks, elapsedMs }`.
 * @param {boolean} [--json-health] - Emit compact health summary JSON.
 * @param {string}  [--database <path>] - Path to the SQLite database file (default: `data/turso-replica.sqlite`).
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

/** Maximum number of SQL statements per client.batch() call. */
const BATCH_SIZE = 1000;

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
    patterns: ['README.md', 'STYLEGUIDE.md', 'CONTRIBUTING.md'],
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

  const client = await initSemanticIndex({
    databasePath,
    client: options.client,
  });

  return buildSemanticIndexWithClient({
    client,
    documents,
    coverageReport,
    summary,
    options,
    tsSourceChunksByFilePath,
  });
}

/**
 * Async client-based implementation of the build loop.
 *
 * Uses `await client.execute()` for all database operations against a
 * Turso/libSQL client. The client is NOT closed — the caller owns its
 * lifecycle.
 *
 * @param {object} params - Build parameters.
 * @param {import('@libsql/client').Client} params.client - libSQL client.
 * @param {Array} params.documents - Corpus document records.
 * @param {object} params.coverageReport - Coverage report for metadata enrichment.
 * @param {object} params.summary - Summary object to populate.
 * @param {object} params.options - Original options (force flag, etc.).
 * @returns {Promise<object>} Populated summary.
 */
async function buildSemanticIndexWithClient({
  client,
  documents,
  coverageReport,
  summary,
  options,
  tsSourceChunksByFilePath,
}) {
  // Step 1: Purge documents that are no longer in the corpus.
  const currentDocumentPaths = new Set(
    documents.map(({ filePath }) => filePath),
  );
  const existingDocsResult = await client.execute({
    sql: 'SELECT file_path FROM documents',
    args: [],
  });
  const deletePromises = [];
  for (const row of existingDocsResult.rows) {
    if (!currentDocumentPaths.has(row.file_path)) {
      deletePromises.push(
        client.execute({
          sql: 'DELETE FROM documents WHERE file_path = ?',
          args: [row.file_path],
        }),
      );
    }
  }
  await Promise.all(deletePromises);
  summary.purged = deletePromises.length;

  const buildLoopStartTime = Date.now();

  // Step 2: Index each document.
  for (const documentRecord of documents) {
    const absolutePath = path.join(repoRoot, documentRecord.filePath);
    const freshnessProof = await getFreshnessProof(absolutePath);
    const existingResult = await client.execute({
      sql: 'SELECT file_path, mtime_ms, file_size, sha256, indexed_at FROM documents WHERE file_path = ?',
      args: [documentRecord.filePath],
    });
    const currentRow = existingResult.rows[0];

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

    // Step 2a: Enrich document-level metadata.
    const docMetadata = enrichDocumentMetadata(documentRecord, coverageReport);
    await client.execute({
      sql: `INSERT INTO documents(file_path, doc_family, mtime_ms, file_size, sha256, indexed_at, arch_layer, test_coverage, source_path_pattern)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(file_path) DO UPDATE SET
              doc_family = excluded.doc_family,
              mtime_ms = excluded.mtime_ms,
              file_size = excluded.file_size,
              sha256 = excluded.sha256,
              indexed_at = excluded.indexed_at,
              arch_layer = excluded.arch_layer,
              test_coverage = excluded.test_coverage,
              source_path_pattern = excluded.source_path_pattern`,
      args: [
        documentRecord.filePath,
        documentRecord.family,
        freshnessProof.mtime_ms,
        freshnessProof.size,
        freshnessProof.sha256,
        Date.now(),
        docMetadata.arch_layer,
        docMetadata.test_coverage,
        docMetadata.source_path_pattern,
      ],
    });

    const docIdResult = await client.execute({
      sql: 'SELECT doc_id FROM documents WHERE file_path = ?',
      args: [documentRecord.filePath],
    });
    const documentId = docIdResult.rows[0].doc_id;

    // Delete old chunks for this document via batch transaction.
    await client.batch(
      [{ sql: 'DELETE FROM chunks WHERE doc_id = ?', args: [documentId] }],
      'write',
    );

    // Insert parent chunks first, then child chunks (resolve parent_chunk_id).
    const parentChunks = chunks.filter(
      (chunk) => chunk.parent_chunk_id == null,
    );
    const childChunks = chunks.filter((chunk) => chunk.parent_chunk_id != null);
    const chunkIdMap = new Map();

    // Build parent INSERT statements and batch them, capturing row IDs from
    // each batch result's lastInsertRowid for child-chunk parent_chunk_id resolution.
    const parentStatements = parentChunks.map((chunk, parentIndex) => {
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
      return {
        sql: `INSERT INTO chunks(doc_id, chunk_index, heading_path, body_text, char_start, char_end,
              parent_chunk_id, depth, context_header, symbol_name, signature_text, jsdoc_text, export_type, module_path,
              arch_layer, jsdoc_quality, jsdoc_word_count, cyclomatic_complexity, test_coverage, source_path_pattern)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        args: [
          documentId,
          parentIndex,
          chunk.heading_path ?? null,
          chunk.body_text,
          chunk.char_start ?? 0,
          chunk.char_end ?? chunk.body_text.length,
          null,
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
        ],
      };
    });

    // Batch-insert parent chunks in groups of BATCH_SIZE, collecting row IDs.
    const parentResults = [];
    for (
      let batchStart = 0;
      batchStart < parentStatements.length;
      batchStart += BATCH_SIZE
    ) {
      const batchSlice = parentStatements.slice(
        batchStart,
        batchStart + BATCH_SIZE,
      );
      const batchResult = await client.batch(batchSlice, 'write');
      parentResults.push(...batchResult);
    }

    for (
      let parentIndex = 0;
      parentIndex < parentChunks.length;
      parentIndex += 1
    ) {
      const chunk = parentChunks[parentIndex];
      const originalIndex = chunk.chunk_index ?? chunks.indexOf(chunk);
      chunkIdMap.set(
        originalIndex,
        Number(parentResults[parentIndex].lastInsertRowid),
      );
    }

    // Build child INSERT statements, resolving parent_chunk_id from the ID map.
    const childStatements = childChunks.map((chunk, childIndex) => {
      const parentChunkIndex = chunk.parent_chunk_id;
      const resolvedParentId = chunkIdMap.get(parentChunkIndex) ?? null;
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
      return {
        sql: `INSERT INTO chunks(doc_id, chunk_index, heading_path, body_text, char_start, char_end,
              parent_chunk_id, depth, context_header, symbol_name, signature_text, jsdoc_text, export_type, module_path,
              arch_layer, jsdoc_quality, jsdoc_word_count, cyclomatic_complexity, test_coverage, source_path_pattern)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        args: [
          documentId,
          parentChunks.length + childIndex,
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
        ],
      };
    });

    // Batch-insert child chunks in groups of BATCH_SIZE.
    for (
      let batchStart = 0;
      batchStart < childStatements.length;
      batchStart += BATCH_SIZE
    ) {
      const batchSlice = childStatements.slice(
        batchStart,
        batchStart + BATCH_SIZE,
      );
      await client.batch(batchSlice, 'write');
    }

    summary.indexed += 1;
    summary.chunks += chunks.length;
  }

  summary.elapsedMs = Date.now() - buildLoopStartTime;
  return summary;
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
        '--database <path> Path to SQLite database file (default: data/turso-replica.sqlite)',
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
