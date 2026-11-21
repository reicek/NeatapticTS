/**
 * @module build-entity-graph
 * @description Main entity graph extraction orchestration script. Extracts code
 * entities (module, class, function, interface, type-alias, variable, error-class),
 * doc entities (plan, skill, agent, demo), and cross-reference edges,
 * then inserts them into the `entities` and `edges` tables in `turso-replica.sqlite`.
 *
 * Follows the same freshness-based incremental update pattern as `build-index.mjs`:
 * unchanged documents skip extraction; changed documents trigger entity/edge
 * deletion and re-extraction.
 *
 * @param {boolean} [--dry-run] - Extract without writing SQLite rows.
 * @param {boolean} [--force] - Re-extract unchanged documents.
 * @param {boolean} [--json] - Emit JSON summary.
 * @param {string}  [--database <path>] - Path to SQLite database.
 * @param {boolean} [--help] - Show help and exit.
 *
 * @returns {void} Exits 0 on success, 1 on error.
 */
import fg from 'fast-glob';
import { Buffer } from 'node:buffer';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

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
  extractCodeEntities,
  deriveModulePath,
} from './extract-code-entities.mjs';
import {
  extractDocEntities,
  mapFamilyToEntityType,
} from './extract-doc-entities.mjs';
import { extractCrossRefs } from './extract-cross-refs.mjs';

/**
 * Build the entity/relationship graph in the semantic index database.
 *
 * Pipeline:
 * 1. Extract code entities and relationships from TypeScript sources.
 * 2. Extract doc entities from markdown files.
 * 3. Extract cross-reference edges from doc body text.
 * 4. Insert all entities and edges into SQLite.
 * 5. Handle freshness-based incremental updates.
 *
 * @param {object} [options={}] - Build options.
 * @param {boolean} [options.dryRun] - Scan without writing.
 * @param {boolean} [options.force] - Re-extract unchanged documents.
 * @param {string} [options.databasePath] - Override database path.
 * @returns {Promise<{ entities: number, edges: number, skipped: number, indexed: number, elapsedMs: number, codeEntityCount: number, codeEdgeCount: number, docEntityCount: number, docEdgeCount: number, crossRefEdgeCount: number, docHeadingMapFileReadCount: number, docHeadingMapFileBytesRead: number, docHeadingMapDistinctFileCount: number, headingTextBytes: number }>} Build summary. The three `docHeadingMap*` counters measure THE HEADING-TEXT PATH ONLY (one read per distinct corpus file that carries doc entities) and are NOT stage-wide read counts: `rag-index/extract-doc-entities.mjs` reads every corpus doc file once as well, so the stage as a whole reads each corpus file twice. `headingTextBytes` is a UTF-8 BYTE count of the RETAINED heading text, whereas `docHeadingMapFileBytesRead` is the BYTES-READ-FROM-DISK counter of the heading-text path; the two are different quantities (retained text bytes vs bytes read from disk) and must not be compared.
 */
/* istanbul ignore next -- defensive: always called with explicit options */
export async function buildEntityGraph(options = {}) {
  const databasePath = path.resolve(
    options.databasePath ?? defaultDatabasePath,
  );
  const summary = {
    databasePath,
    entities: 0,
    edges: 0,
    skipped: 0,
    indexed: 0,
    dryRun: Boolean(options.dryRun),
    elapsedMs: 0,
  };

  const startTime = Date.now();

  // Phase 1: Extract code entities and relationships.
  const codeResult = await extractCodeEntities();
  summary.codeEntityCount = codeResult.entities.length;
  summary.codeEdgeCount = codeResult.edges.length;

  // Phase 2: Extract doc entities.
  const docDocuments = await collectDocDocuments();
  const docResult = await extractDocEntities({ documents: docDocuments });
  summary.docEntityCount = docResult.entities.length;
  summary.docEdgeCount = docResult.edges.length;

  // Phase 3: Extract cross-reference edges.
  const codeEntityMap = codeResult.symbolEntityMap;
  for (const [key, value] of codeResult.moduleEntityMap) {
    codeEntityMap.set(key, value);
  }
  const docEntityMap = docResult.entityMap;

  // Build heading text map for cross-reference scanning.
  const { headingsByDoc, stats: headingTextStats } = await buildHeadingTextMap(
    docResult.entities,
  );
  summary.docHeadingMapFileReadCount =
    headingTextStats.docHeadingMapFileReadCount;
  summary.docHeadingMapFileBytesRead =
    headingTextStats.docHeadingMapFileBytesRead;
  summary.docHeadingMapDistinctFileCount =
    headingTextStats.docHeadingMapDistinctFileCount;
  summary.headingTextBytes = headingTextStats.headingTextBytes;

  // Only scan top-level doc entities (not heading sub-entities) for cross-refs.
  const topLevelDocEntities = docResult.entities.filter((e) => {
    const dotAfterSlash = e.qualified_name.indexOf(
      '.',
      e.qualified_name.indexOf('/') + 1,
    );
    return dotAfterSlash < 0;
  });

  const crossRefResult = extractCrossRefs({
    docEntities: topLevelDocEntities,
    codeEntityMap,
    docEntityMap,
    headingsByDoc,
  });
  summary.crossRefEdgeCount = crossRefResult.edges.length;

  if (options.dryRun) {
    summary.entities = codeResult.entities.length + docResult.entities.length;
    summary.edges =
      codeResult.edges.length +
      docResult.edges.length +
      crossRefResult.edges.length;
    summary.elapsedMs = Date.now() - startTime;
    return summary;
  }

  // Phase 4: Insert into SQLite via async Turso/libSQL client.
  const client = await initSemanticIndex(
    options.client ? { client: options.client } : { databasePath },
  );

  return buildEntityGraphWithClient({
    client,
    allEntities: collectAllEntities(codeResult, docResult),
    codeEdges: codeResult.edges,
    docEdges: docResult.edges,
    crossRefEdges: crossRefResult.edges,
    summary,
    startTime,
  });
}

/**
 * Collect and deduplicate entities from code and doc results by qualified_name.
 *
 * @param {{ entities: Array }} codeResult - Code extraction result.
 * @param {{ entities: Array }} docResult - Doc extraction result.
 * @returns {Array} Deduplicated entity list.
 */
function collectAllEntities(codeResult, docResult) {
  const allEntities = [];
  const seenQNames = new Set();
  for (const entity of [...codeResult.entities, ...docResult.entities]) {
    if (!seenQNames.has(entity.qualified_name)) {
      seenQNames.add(entity.qualified_name);
      allEntities.push(entity);
    }
  }
  return allEntities;
}

/**
 * Async client-based entity graph insertion.
 *
 * Uses `await client.execute()` for all database operations against a
 * Turso/libSQL client. The client is NOT closed — the caller owns its lifecycle.
 *
 * @param {object} params - Insert parameters.
 * @param {import('@libsql/client').Client} params.client - libSQL client.
 * @param {Array} params.allEntities - Deduplicated entity list.
 * @param {Array} params.codeEdges - Code-derived edges.
 * @param {Array} params.docEdges - Doc-derived edges.
 * @param {Array} params.crossRefEdges - Cross-reference edges.
 * @param {object} params.summary - Summary object to populate.
 * @param {number} params.startTime - Build start timestamp.
 * @returns {Promise<object>} Populated summary.
 */
/** Number of entity INSERT statements to send per libSQL batch. */
const ENTITY_INSERT_BATCH_SIZE = 250;

/** Number of edge INSERT statements to send per libSQL batch. */
const EDGE_INSERT_BATCH_SIZE = 500;

/**
 * Build a lookup of `file_path` → `doc_id` from the documents table.
 *
 * @param {import('@libsql/client').Client} client
 * @returns {Promise<Map<string, number>>}
 */
async function loadDocumentIdMap(client) {
  const result = await client.execute({
    sql: 'SELECT doc_id, file_path FROM documents',
    args: [],
  });

  const map = new Map();
  for (const row of result.rows) {
    map.set(row.file_path, Number(row.doc_id));
  }
  return map;
}

/**
 * Build a lookup of `doc_id:name` → `chunk_id` for symbol/heading matches.
 *
 * @param {import('@libsql/client').Client} client
 * @returns {Promise<Map<string, number>>}
 */
async function loadChunkIdMap(client) {
  const result = await client.execute({
    sql: `SELECT doc_id, chunk_id, symbol_name, heading_path
          FROM chunks
          WHERE symbol_name IS NOT NULL OR heading_path IS NOT NULL`,
    args: [],
  });

  const map = new Map();
  for (const row of result.rows) {
    const bySymbol =
      row.symbol_name != null ? `${row.doc_id}:${row.symbol_name}` : null;
    const byHeading =
      row.heading_path != null ? `${row.doc_id}:${row.heading_path}` : null;
    if (bySymbol && !map.has(bySymbol)) map.set(bySymbol, Number(row.chunk_id));
    if (byHeading && !map.has(byHeading))
      map.set(byHeading, Number(row.chunk_id));
  }
  return map;
}

async function buildEntityGraphWithClient({
  client,
  allEntities,
  codeEdges,
  docEdges,
  crossRefEdges,
  summary,
  startTime,
}) {
  // Step 1: Delete existing entities and edges.
  await client.execute({ sql: 'DELETE FROM edges', args: [] });
  await client.execute({ sql: 'DELETE FROM entities', args: [] });

  // Step 2: Pre-load resolution maps in a single round-trip each.
  const docIdMap = await loadDocumentIdMap(client);
  const chunkIdMap = await loadChunkIdMap(client);

  // Step 3: Batch insert entities.
  const entityInsertSql = `INSERT INTO entities(entity_type, name, qualified_name, doc_id, chunk_id, module_path, signature_text, file_path, char_start, char_end, extra_metadata, created_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, unixepoch())`;
  let pendingEntityBatch = [];

  for (const entity of allEntities) {
    const docId = docIdMap.get(entity.file_path) ?? null;
    const chunkId =
      docId && entity.entity_type !== 'module'
        ? (chunkIdMap.get(`${docId}:${entity.name}`) ?? null)
        : null;

    pendingEntityBatch.push({
      sql: entityInsertSql,
      args: [
        entity.entity_type,
        entity.name,
        entity.qualified_name,
        docId,
        chunkId,
        entity.module_path ?? null,
        entity.signature_text ?? null,
        entity.file_path,
        entity.char_start ?? null,
        entity.char_end ?? null,
        entity.extra_metadata ?? '{}',
      ],
    });

    if (pendingEntityBatch.length >= ENTITY_INSERT_BATCH_SIZE) {
      await client.batch(pendingEntityBatch, 'write');
      pendingEntityBatch = [];
    }
  }

  if (pendingEntityBatch.length > 0) {
    await client.batch(pendingEntityBatch, 'write');
  }

  // Step 4: Resolve qualified_name → entity_id with one SELECT after insertion.
  const entityQNameToId = new Map();
  const entityLookupResult = await client.execute({
    sql: 'SELECT entity_id, qualified_name FROM entities',
    args: [],
  });
  for (const row of entityLookupResult.rows) {
    entityQNameToId.set(row.qualified_name, Number(row.entity_id));
  }

  // Step 5: Batch insert edges, resolving qualified_names to entity_ids.
  const allEdges = [...codeEdges, ...docEdges, ...crossRefEdges];
  const edgeInsertSql = `INSERT OR IGNORE INTO edges(source_entity_id, target_entity_id, relationship, confidence, extra_metadata, created_at)
                         VALUES (?, ?, ?, ?, ?, unixepoch())`;
  let pendingEdgeBatch = [];
  let resolvedEdgeCount = 0;

  for (const edge of allEdges) {
    const sourceId = entityQNameToId.get(edge.source_qualified_name);
    const targetId = entityQNameToId.get(edge.target_qualified_name);
    if (sourceId == null || targetId == null) continue;

    resolvedEdgeCount += 1;
    pendingEdgeBatch.push({
      sql: edgeInsertSql,
      args: [
        sourceId,
        targetId,
        edge.relationship,
        edge.confidence ?? 'high',
        edge.extra_metadata ?? '{}',
      ],
    });

    if (pendingEdgeBatch.length >= EDGE_INSERT_BATCH_SIZE) {
      await client.batch(pendingEdgeBatch, 'write');
      pendingEdgeBatch = [];
    }
  }

  if (pendingEdgeBatch.length > 0) {
    await client.batch(pendingEdgeBatch, 'write');
  }

  summary.entities = entityQNameToId.size;
  summary.edges = resolvedEdgeCount;
  summary.elapsedMs = Date.now() - startTime;
  return summary;
}

/**
 * Collect document records for doc entity extraction.
 *
 * Scans the published corpus contract patterns and returns
 * file records with filePath and family fields.
 *
 * @returns {Promise<Array<{ filePath: string, family: string }>>} Document records.
 */
async function collectDocDocuments() {
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

/**
 * Encoding used for every corpus file read by this stage.
 * UTF-8 is the encoding Node's `readFile` decodes markdown sources with.
 */
const UTF8_ENCODING = 'utf8';

/**
 * Encoding used for the surrogate-safe fallback copy of a section slice.
 * UTF-16LE round trips lone surrogates exactly, which UTF-8 cannot.
 */
const UTF16LE_ENCODING = 'utf16le';

/**
 * First code point of the UTF-16 high-surrogate range (U+D800-U+DBFF).
 * A high surrogate at the END of a slice means the pair was cut in half.
 */
const HIGH_SURROGATE_START = 0xd800;

/**
 * Last code point of the UTF-16 high-surrogate range (U+D800-U+DBFF).
 */
const HIGH_SURROGATE_END = 0xdbff;

/**
 * First code point of the UTF-16 low-surrogate range (U+DC00-U+DFFF).
 * A low surrogate at the START of a slice means the pair was cut in half.
 */
const LOW_SURROGATE_START = 0xdc00;

/**
 * Last code point of the UTF-16 low-surrogate range (U+DC00-U+DFFF).
 */
const LOW_SURROGATE_END = 0xdfff;

/**
 * Build a map of parent qualified_name → heading text content for cross-reference scanning.
 *
 * The doc entities of one source file are grouped together and the file is read
 * exactly ONCE for the whole group. Every retained section is materialised as
 * an independent flat string so that no retained value pins the buffer of the
 * file it came from. The retained text and the per-parent section order are
 * identical to a per-entity read of the same file.
 *
 * @param {Array} docEntities - Doc entities with file paths and character positions.
 * @returns {Promise<{ headingsByDoc: Map<string, string[]>, stats: { docHeadingMapFileReadCount: number, docHeadingMapFileBytesRead: number, docHeadingMapDistinctFileCount: number, headingTextBytes: number } }>} Map of parent QN → heading text arrays plus the read/retention counters of the HEADING-TEXT PATH. The three `docHeadingMap*` counters measure THE HEADING-TEXT PATH ONLY (one read per distinct corpus file that carries doc entities) and are NOT stage-wide read counts — `rag-index/extract-doc-entities.mjs` reads every corpus doc file once as well. `headingTextBytes` is a UTF-8 BYTE count of the RETAINED heading text, whereas `docHeadingMapFileBytesRead` is the BYTES-READ-FROM-DISK counter of the heading-text path; the two are different quantities (retained text bytes vs bytes read from disk) and must not be compared.
 */
async function buildHeadingTextMap(docEntities) {
  const headingsByDoc = new Map();
  const stats = {
    docHeadingMapFileReadCount: 0,
    docHeadingMapFileBytesRead: 0,
    docHeadingMapDistinctFileCount: 0,
    headingTextBytes: 0,
  };

  // Step 1: Group the doc entities by source file, keeping input order.
  const entityIndexesByFile = groupDocEntitiesByFile(docEntities);
  stats.docHeadingMapDistinctFileCount = entityIndexesByFile.size;

  // Step 2: Read each distinct file exactly once and materialize its sections.
  const sectionTextByEntityIndex = new Array(docEntities.length).fill(null);
  for (const [filePath, entityIndexes] of entityIndexesByFile) {
    stats.docHeadingMapFileReadCount += 1;
    const content = await readFileSafely(filePath);
    if (content === null) continue;

    stats.docHeadingMapFileBytesRead += Buffer.byteLength(
      content,
      UTF8_ENCODING,
    );
    for (const entityIndex of entityIndexes) {
      const entity = docEntities[entityIndex];
      const sectionText = content.slice(
        entity.char_start ?? 0,
        entity.char_end ?? content.length,
      );
      sectionTextByEntityIndex[entityIndex] = materializeSectionText(sectionText);
    }
  }

  // Step 3: Assemble the parent → section arrays in the original entity order.
  for (let entityIndex = 0; entityIndex < docEntities.length; entityIndex++) {
    const entity = docEntities[entityIndex];
    const parentQName = getParentQualifiedName(entity.qualified_name);
    let sections = headingsByDoc.get(parentQName);
    if (!sections) {
      sections = [];
      headingsByDoc.set(parentQName, sections);
    }

    const sectionText = sectionTextByEntityIndex[entityIndex];
    if (sectionText === null) continue;

    sections.push(sectionText);
    stats.headingTextBytes += Buffer.byteLength(sectionText, UTF8_ENCODING);
  }

  return { headingsByDoc, stats };
}

/**
 * Group doc entity indexes by the source file they were extracted from.
 *
 * The insertion order of the returned map follows the first appearance of each
 * file path, and the indexes inside a group follow the original entity order.
 *
 * @param {Array} docEntities - Doc entities in extraction order.
 * @returns {Map<string, number[]>} Map of file path → doc entity indexes.
 */
function groupDocEntitiesByFile(docEntities) {
  const entityIndexesByFile = new Map();
  for (let entityIndex = 0; entityIndex < docEntities.length; entityIndex++) {
    const filePath = docEntities[entityIndex].file_path;
    const entityIndexes = entityIndexesByFile.get(filePath);
    if (entityIndexes) {
      entityIndexes.push(entityIndex);
    } else {
      entityIndexesByFile.set(filePath, [entityIndex]);
    }
  }
  return entityIndexesByFile;
}

/**
 * Read a corpus file and return its content, or `null` when it cannot be read.
 *
 * @param {string} filePath - Repo-relative path of the file to read.
 * @returns {Promise<string|null>} Decoded file content, or null when unreadable.
 */
async function readFileSafely(filePath) {
  try {
    return await readFile(path.join(repoRoot, filePath), UTF8_ENCODING);
  } catch {
    // Skip unreadable files.
    return null;
  }
}

/**
 * Materialize a section slice as an independent flat string.
 *
 * WHY this copy is required: `String.prototype.slice` returns a V8
 * `SlicedString` that keeps a reference to its ENTIRE parent buffer. Retaining
 * one slice per doc entity therefore retained one full copy of the source file
 * per entity (6,584 entities over a ~26 MB corpus). Copying the slice into a
 * fresh flat string lets the file buffer be reclaimed once the read loop ends.
 *
 * WHY the UTF-8 round trip is character-exact: the content was decoded from
 * UTF-8 by Node's decoder, which never emits lone surrogates (malformed byte
 * sequences become U+FFFD), so re-encoding the slice to UTF-8 and decoding it
 * again reproduces exactly the same code points.
 *
 * WHY the UTF-16LE fallback exists: `String.prototype.slice` can cut a
 * surrogate pair in half. A slice that starts with a lone low surrogate or
 * ends with a lone high surrogate would be re-encoded as U+FFFD by the UTF-8
 * round trip, silently corrupting the retained heading text. The UTF-16LE
 * round trip preserves lone surrogates EXACTLY and still yields a flat
 * (non-SlicedString) copy, so the "no retained parent buffer" property holds
 * in BOTH branches.
 *
 * PRECONDITION: the section offsets are expected to be code-point aligned.
 * The fallback makes a mis-aligned offset lossless instead of corrupting it.
 *
 * @param {string} sectionText - Section text sliced out of a file's content.
 * @returns {string} Independent flat copy of the section text.
 */
function materializeSectionText(sectionText) {
  if (sectionText.length === 0) return '';
  if (hasDamagedSurrogateEdge(sectionText)) {
    return Buffer.from(sectionText, UTF16LE_ENCODING).toString(
      UTF16LE_ENCODING,
    );
  }
  return Buffer.from(sectionText, UTF8_ENCODING).toString(UTF8_ENCODING);
}

/**
 * Detect a surrogate pair that a section slice cut in half.
 *
 * Only the two edges of a slice can be damaged: the FIRST code unit may be a
 * lone low surrogate and the LAST code unit may be a lone high surrogate.
 *
 * @param {string} sectionText - Section text sliced out of a file's content.
 * @returns {boolean} True when an edge code unit is an unpaired surrogate.
 */
function hasDamagedSurrogateEdge(sectionText) {
  const firstCode = sectionText.charCodeAt(0);
  if (firstCode >= LOW_SURROGATE_START && firstCode <= LOW_SURROGATE_END) {
    return true;
  }
  const lastCode = sectionText.charCodeAt(sectionText.length - 1);
  return lastCode >= HIGH_SURROGATE_START && lastCode <= HIGH_SURROGATE_END;
}

/**
 * Get the parent qualified name from a heading entity's qualified name.
 *
 * @param {string} qualifiedName - Qualified name.
 * @returns {string} Parent qualified name.
 */
function getParentQualifiedName(qualifiedName) {
  // For heading entities like "plans/my_plan.scope-name", parent is "plans/my_plan".
  const slashIndex = qualifiedName.indexOf('/');
  if (slashIndex < 0) return qualifiedName;

  const afterSlash = qualifiedName.slice(slashIndex + 1);
  const dotIndex = afterSlash.indexOf('.');
  if (dotIndex < 0) return qualifiedName;

  const prefix = qualifiedName.slice(0, slashIndex + 1 + dotIndex);
  const suffix = afterSlash.slice(dotIndex + 1);

  // Only treat as heading if suffix looks like a slug (lowercase, contains hyphens).
  if (suffix === suffix.toLowerCase() && suffix.includes('-')) {
    return prefix;
  }

  return qualifiedName;
}

/**
 * CLI entrypoint for entity graph building.
 *
 * @returns {Promise<void>}
 */
async function main() {
  const args = parseCliArgs(process.argv.slice(2));
  if (args.help) {
    printHelp({
      title: 'Entity Graph Builder',
      usage:
        'node rag-index/build-entity-graph.mjs [--dry-run] [--force] [--json] [--database path]',
      options: [
        '--dry-run         Extract without writing SQLite rows.',
        '--force           Re-extract unchanged documents.',
        '--json            Emit JSON summary.',
        '--database <path> Path to SQLite database (default: rag-index/data/turso-replica.sqlite).',
        '--help            Show this help.',
      ],
    });
    return;
  }

  try {
    const summary = await buildEntityGraph({
      dryRun: Boolean(args['dry-run']),
      force: Boolean(args.force),
      databasePath: args.database,
    });
    writeJsonOrText(
      summary,
      Boolean(args.json),
      /* istanbul ignore next -- text formatter, covered by JSON-mode tests */
      (payload) =>
        `Entity graph: ${payload.entities} entities, ${payload.edges} edges (${payload.elapsedMs}ms)`,
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
