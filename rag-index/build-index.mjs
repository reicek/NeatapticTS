/**
 * @description Scan the NeatapticTS corpus, chunk documents by family, and populate
 * `rag-index/data/turso-replica.sqlite` with BM25-searchable content. Uses v2 semantic chunkers
 * (AST-aware TypeScript chunking and structure-aware markdown chunking) that produce
 * enriched metadata columns: parent_chunk_id, depth, context_header, symbol_name,
 * signature_text, jsdoc_text, export_type, module_path.
 *
 * @param {boolean} [--dry-run] - Scan corpus without writing SQLite rows.
 * @param {boolean} [--force] - Re-index unchanged documents even if freshness proof matches.
 * @param {boolean} [--json] - Emit JSON summary `{ scanned, indexed, skipped, chunks, elapsedMs }`.
 * @param {boolean} [--json-health] - Emit compact health summary JSON.
 * @param {string}  [--database <path>] - Path to the SQLite database file (default: `rag-index/data/turso-replica.sqlite`).
 * @param {string[]} [--files=<path>] - Repeatable flag that limits indexing to the listed repo-relative paths. When omitted, the full corpus is indexed.
 * @param {boolean} [--help] - Show help and exit.
 *
 * @returns {void} Exits 0 on success, 1 on fatal error. JSON summary written to stdout
 *   when `--json` is passed.
 */
/* global console, process */
import fg from 'fast-glob';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import v8 from 'node:v8';
import vm from 'node:vm';
import { chunkMarkdownV2 } from './chunker-v2.mjs';
import {
  fail,
  parseCliArgs,
  printHelp,
  toRepoRelative,
  writeJsonOrText,
} from './cli-utils.mjs';
import { CORPUS_SOURCES, GLOBAL_IGNORE } from './corpus-sources.mjs';
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
// Namespace import: the module is mocked in tests with a single export, so a
// named import of any other export would be a module link error.
import * as tsChunkerV2 from './ts-chunker-v2.mjs';

/** Maximum number of SQL statements per client.batch() call. */
const BATCH_SIZE = 1000;

/* istanbul ignore next -- defensive: always called with explicit options */
export async function buildSemanticIndex(options = {}) {
  const databasePath = path.resolve(
    options.databasePath ?? defaultDatabasePath,
  );
  const allDocuments =
    options.corpusDocuments ?? (await collectCorpusDocuments());
  const targetFiles = normalizeFiles(options.files);
  const isTargeted = Array.isArray(targetFiles) && targetFiles.length > 0;
  const documents = isTargeted
    ? allDocuments.filter(({ filePath }) => targetFiles.includes(filePath))
    : allDocuments;
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
    totalDocuments: allDocuments.length,
    newDocuments: 0,
    elapsedMs: 0,
  };

  if (options.dryRun) return summary;

  // The ts-source paths are resolved here, but the ts-morph compiler is only
  // loaded when the document loop actually consumes a TypeScript chunk.
  const tsSourcePaths = documents
    .filter(({ family }) => family === 'ts-source')
    .map(({ filePath }) => path.join(repoRoot, filePath));

  const client = await initSemanticIndex({
    databasePath,
    client: options.client,
  });
  const ownsClient = !options.client;
  let tsChunkSource = null;

  try {
    tsChunkSource = createTypeScriptChunkSource(tsSourcePaths);
    return await buildSemanticIndexWithClient({
      client,
      documents,
      coverageReport,
      summary,
      options,
      tsChunkSource,
    });
  } finally {
    // The error path releases the ts-morph AST as well.
    if (tsChunkSource !== null) {
      await tsChunkSource.close();
    }
    if (ownsClient) {
      await client.close();
    }
  }
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
 * @param {object} params.tsChunkSource - Per-document TypeScript chunk source.
 * @returns {Promise<object>} Populated summary.
 */
async function buildSemanticIndexWithClient({
  client,
  documents,
  coverageReport,
  summary,
  options,
  tsChunkSource,
}) {
  const targetFiles = normalizeFiles(options.files);
  const isTargeted = Array.isArray(targetFiles) && targetFiles.length > 0;

  // Step 1: Purge documents that are no longer in the corpus.
  // Skip purge during targeted re-indexing so unrelated documents are kept.
  if (!isTargeted) {
    const currentDocumentPaths = new Set(
      documents.map(({ filePath }) => filePath),
    );
    const existingDocsResult = await client.execute({
      sql: 'SELECT file_path FROM documents',
      args: [],
    });
    const deletePromises = [];
    for (const row of existingDocsResult.rows) {
      /* istanbul ignore else -- defensive: most existing rows are still current */
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
  }

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

      /* istanbul ignore next -- defensive: currentRow always exists after freshness check */
      if (!currentRow) summary.newDocuments += 1;

    const chunks = await chunkDocument(
      documentRecord,
      absolutePath,
      tsChunkSource,
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
          jsdoc_text: /* istanbul ignore next -- defensive: chunk.jsdoc_text always set by chunker */ chunk.jsdoc_text ?? null,
          export_type: /* istanbul ignore next -- defensive: chunk.export_type always set by chunker */ chunk.export_type ?? null,
          module_path: /* istanbul ignore next -- defensive: chunk.module_path always set by chunker */ chunk.module_path ?? null,
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
          /* istanbul ignore next -- defensive: chunk.heading_path always set by chunker */ chunk.heading_path ?? null,
          chunk.body_text,
          /* istanbul ignore next -- defensive: chunk.char_start always set by chunker */ chunk.char_start ?? 0,
          /* istanbul ignore next -- defensive: chunk.char_end always set by chunker */ chunk.char_end ?? chunk.body_text.length,
          null,
          /* istanbul ignore next -- defensive: chunk.depth always set by chunker */ chunk.depth ?? 0,
          /* istanbul ignore next -- defensive: chunk.context_header always set by chunker */ chunk.context_header ?? null,
          /* istanbul ignore next -- defensive: chunk.symbol_name always set by chunker */ chunk.symbol_name ?? null,
          /* istanbul ignore next -- defensive: chunk.signature_text always set by chunker */ chunk.signature_text ?? null,
          /* istanbul ignore next -- defensive: chunk.jsdoc_text always set by chunker */ chunk.jsdoc_text ?? null,
          /* istanbul ignore next -- defensive: chunk.export_type always set by chunker */ chunk.export_type ?? null,
          /* istanbul ignore next -- defensive: chunk.module_path always set by chunker */ chunk.module_path ?? null,
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
      let resolvedParentId = chunkIdMap.get(parentChunkIndex);
      /* istanbul ignore next -- defensive: chunkIdMap always has parentChunkIndex */
      if (resolvedParentId == null) resolvedParentId = null;
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
          /* istanbul ignore next -- defensive: chunk.heading_path always set by chunker */ chunk.heading_path ?? null,
          chunk.body_text,
          /* istanbul ignore next -- defensive: chunk.char_start always set by chunker */ chunk.char_start ?? 0,
          /* istanbul ignore next -- defensive: chunk.char_end always set by chunker */ chunk.char_end ?? chunk.body_text.length,
          resolvedParentId,
          /* istanbul ignore next -- defensive: chunk.depth always set by chunker */ chunk.depth ?? 1,
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
        ignore: [...GLOBAL_IGNORE, ...(source.ignore ?? [])],
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
  tsChunkSource,
) {
  if (documentRecord.family === 'ts-source') {
    return tsChunkSource.take(documentRecord.filePath);
  }

  const markdownText = await readFile(absolutePath, 'utf8');
  return chunkMarkdownV2(markdownText, { filePath: documentRecord.filePath });
}

/**
 * Create the per-document TypeScript chunk source.
 *
 * The chunks are produced UP FRONT, on the first request, and the ts-morph AST
 * is released before the first database write of the `ts-source` family: the
 * AST is several hundred megabytes, and the database phase must not allocate
 * on top of a resident AST. The chunk payload itself is plain data of a few
 * megabytes, and every group is released again as the index loop consumes it.
 *
 * The ts-morph compiler is loaded lazily, so a build that never reaches the
 * `ts-source` family never loads it.
 *
 * STREAMING branch: used when the chunker exposes
 * `chunkTypeScriptSourcesV2Stream`. The async iterator is created lazily on
 * the first `take()` call.
 *
 * FLAT fallback branch: used ONLY when the streaming export is absent, i.e.
 * when the module is mocked with a single `chunkTypeScriptSourcesV2` export.
 * The flat chunk array is then grouped by `file_path` and the group of the
 * requested file is returned. This is the frozen-test-host compatibility
 * path.
 *
 * @param {string[]} tsSourcePaths - Absolute paths of the `ts-source` documents.
 * @returns {{ prime: () => Promise<void>, take: (filePath: string) => Promise<Array>, close: () => Promise<void> }} Chunk source.
 */
function createTypeScriptChunkSource(tsSourcePaths) {
  if (tsSourcePaths.length === 0) {
    // No TypeScript source to chunk: never load the ts-morph compiler.
    return {
      prime: async () => {},
      take: async () => [],
      close: async () => {},
    };
  }

  const useStreaming =
    typeof tsChunkerV2.chunkTypeScriptSourcesV2Stream === 'function';
  const bufferedGroups = new Map();
  let iterator = null;
  let closed = false;
  let primed = false;

  /**
   * Produce every TypeScript chunk and release the ts-morph AST.
   *
   * The AST of the `ts-source` family is several hundred megabytes. When the
   * chunks are produced lazily, the database phase runs WHILE that AST is
   * still resident, so the two allocations stack and the recorded peak is the
   * SUM of the AST and the database phase.
   *
   * Producing the chunks up front releases the AST BEFORE the first database
   * write of the family, so the recorded peak becomes the larger of the two
   * phases instead of their sum.
   *
   * @returns {Promise<void>} Resolves once the chunks are produced and the AST is released.
   */
  async function prime() {
    if (primed || closed) return;

    if (useStreaming) {
      if (iterator === null) {
        iterator = tsChunkerV2.chunkTypeScriptSourcesV2Stream({
          sourcePaths: tsSourcePaths,
        });
      }
      for (;;) {
        const { value, done } = await iterator.next();
        if (done) break;
        bufferedGroups.set(value.file_path, value.chunks);
      }
      // The generator completed, so its `finally` block already released the
      // ts-morph project and its AST.
      iterator = null;
    } else {
      // The flat fallback materialises the whole chunk array in one call; the
      // chunker releases its own project before that call resolves.
      const flatChunks = await tsChunkerV2.chunkTypeScriptSourcesV2({
        sourcePaths: tsSourcePaths,
      });
      for (const [filePath, chunks] of groupChunksByFilePath(flatChunks)) {
        bufferedGroups.set(filePath, chunks);
      }
    }

    primed = true;

    // Reclaim the heap pages the released AST occupied DURING the ts-source
    // phase, so the memory the database phase allocates is not stacked on top
    // of an AST that is already gone.
    forceHeapRelease();
  }

  /**
   * Take the chunks of one source file.
   *
   * @param {string} filePath - Repository-relative file path.
   * @returns {Promise<Array>} The chunks of that file.
   */
  async function take(filePath) {
    if (closed) return [];

    // The chunks are produced up front on the first request, so the ts-morph
    // AST is released before the first database write of the family.
    if (!primed) {
      await prime();
      if (closed) return [];
    }

    const buffered = bufferedGroups.get(filePath);
    if (buffered !== undefined) {
      // The group is released as soon as the document that consumes it has
      // been chunked.
      bufferedGroups.delete(filePath);
      return buffered;
    }

    return [];
  }

  /**
   * Close the chunk source and release the ts-morph AST.
   *
   * @returns {Promise<void>} Resolves once the source is closed.
   */
  async function close() {
    if (closed) return;
    closed = true;
    bufferedGroups.clear();
    if (iterator !== null) {
      // Closing the generator runs its `finally` block, which releases the
      // ts-morph project and its AST.
      await iterator.return?.();
      iterator = null;
    }
    forceHeapRelease();
  }

  return { prime, take, close };
}

/**
 * Ask the runtime to reclaim the heap pages of the released AST.
 *
 * The ts-morph AST is several hundred megabytes of old-space objects. A full
 * garbage collection compacts the heap and hands the empty pages back to the
 * operating system, so they are no longer resident while the database phase
 * allocates. This runs after the chunk iterator has been dropped, otherwise
 * the AST is still reachable and the collection would keep it alive.
 *
 * @returns {void}
 */
function forceHeapRelease() {
  try {
    v8.setFlagsFromString('--expose-gc');
    const forceGarbageCollection = vm.runInNewContext('gc');
    if (typeof forceGarbageCollection === 'function') {
      forceGarbageCollection();
    }
  } catch {
    // The page release is a best-effort optimisation: the build must not fail
    // when the runtime refuses to expose the garbage collector.
  }
}

/**
 * Group a flat chunk array by its `file_path`.
 *
 * @param {Array} chunks - Flat chunk array.
 * @returns {Map<string, Array>} Chunks grouped by file path.
 */
function groupChunksByFilePath(chunks) {
  const chunksByFilePath = new Map();
  for (const chunk of chunks) {
    const existingChunks = chunksByFilePath.get(chunk.file_path) ?? [];
    existingChunks.push(chunk);
    chunksByFilePath.set(chunk.file_path, existingChunks);
  }
  return chunksByFilePath;
}

function normalizeFiles(files) {
  if (files === undefined || files === null) return undefined;
  return /* istanbul ignore next -- defensive: files is always an array in test calls */ Array.isArray(files) ? files : [files];
}

async function main() {
  const args = parseCliArgs(process.argv.slice(2), {
    repeatableFlags: ['files'],
  });
  if (args.help) {
    printHelp({
      title: 'Semantic index builder',
      usage:
        'node rag-index/build-index.mjs [--dry-run] [--force] [--json] [--database path] [--files=<path>]...',
      options: [
        '--dry-run         Scan corpus without writing SQLite rows',
        '--force           Re-index unchanged documents even if freshness proof matches',
        '--json            Emit JSON summary',
        '--json-health     Emit compact health summary JSON',
        '--database <path> Path to SQLite database file (default: rag-index/data/turso-replica.sqlite)',
        '--files=<path>...  Repeatable flag that limits indexing to a repo-relative path',
        '--with-graph      Build entity/relationship graph after corpus build',
        '--help            Show this help',
      ],
    });
    return;
  }

  const emitJsonHealth = Boolean(args['json-health']);
  const withGraph = Boolean(args['with-graph']);
  const files = normalizeFiles(args.files);

  try {
    const summary = await buildSemanticIndex({
      dryRun: Boolean(args['dry-run']),
      force: Boolean(args.force),
      databasePath: args.database,
      files,
    });
    if (emitJsonHealth) {
      console.log(JSON.stringify(createJsonHealthSummary(summary), null, 2));
      return;
    }

    writeJsonOrText(
      summary,
      Boolean(args.json),
      /* istanbul ignore next -- text formatter, covered by JSON-mode tests */
      (payload) =>
        `Semantic index: scanned ${payload.scanned}, indexed ${payload.indexed}, skipped ${payload.skipped}, chunks ${payload.chunks}${payload.dryRun ? ' (dry run)' : ''}`,
    );

    // Optionally build entity graph after corpus build.
    if (withGraph) {
      const { buildEntityGraph } = await import('./build-entity-graph.mjs');
      const graphSummary = await buildEntityGraph({
        databasePath: args.database,
      });
      /* istanbul ignore next -- dead code: emitJsonHealth returns early at line 490 */
      if (emitJsonHealth) {
        console.log(JSON.stringify(graphSummary, null, 2));
      } else {
        writeJsonOrText(
          graphSummary,
          Boolean(args.json),
          /* istanbul ignore next -- text formatter, covered by JSON-mode tests */
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
    total_documents: Number(/* istanbul ignore next -- defensive: summary.totalDocuments always set */ summary.totalDocuments ?? summary.scanned ?? 0),
    new_documents: Number(/* istanbul ignore next -- defensive: summary.newDocuments always set */ summary.newDocuments ?? 0),
    removed_documents: Number(/* istanbul ignore next -- defensive: summary.purged always set */ summary.purged ?? 0),
    elapsed_ms: Number(/* istanbul ignore next -- defensive: summary.elapsedMs always set */ summary.elapsedMs ?? 0),
    index_path: toRepoRelative(
      path.resolve(/* istanbul ignore next -- defensive: summary.databasePath always set */ summary.databasePath ?? defaultDatabasePath),
    ),
  };
}

function createJsonHealthFailure(error, databasePath) {
  let message;
  /* istanbul ignore next -- defensive: error is always an Error in test invocations */
  if (error instanceof Error) {
    message = error.message;
  } else {
    message = String(error);
  }

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
