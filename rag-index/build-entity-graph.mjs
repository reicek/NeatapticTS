/**
 * @module build-entity-graph
 * @description Main entity graph extraction orchestration script. Extracts code
 * entities (module, class, function, interface, type-alias, variable, error-class),
 * doc entities (plan, skill, agent, demo, benchmark), and cross-reference edges,
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

/** Document sources for doc entity extraction. */
const DOC_ENTITY_SOURCES = [
  {
    family: 'plan',
    patterns: ['plans/**/*.md'],
    ignore: ['plans/completed/**'],
  },
  { family: 'completed-plan', patterns: ['plans/completed/**/*.md'] },
  { family: 'skill', patterns: ['.github/skills/**/SKILL.md'] },
  { family: 'agent', patterns: ['.github/agents/*.agent.md'] },
  { family: 'demo', patterns: ['examples/**/README.md', 'examples/**/*.ts'] },
  {
    family: 'benchmark',
    patterns: ['benchmarks/README.md', 'benchmarks/**/*.test.ts'],
  },
];

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
 * @returns {Promise<{ entities: number, edges: number, skipped: number, indexed: number, elapsedMs: number }>} Build summary.
 */
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
  const headingsByDoc = await buildHeadingTextMap(docResult.entities);

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
 * Scans the configured DOC_ENTITY_SOURCES patterns and returns
 * file records with filePath and family fields.
 *
 * @returns {Promise<Array<{ filePath: string, family: string }>>} Document records.
 */
async function collectDocDocuments() {
  const records = await Promise.all(
    DOC_ENTITY_SOURCES.map(async (source) => {
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

/**
 * Build a map of parent qualified_name → heading text content for cross-reference scanning.
 *
 * @param {Array} docEntities - Doc entities with file paths and character positions.
 * @returns {Promise<Map<string, string[]>>} Map of parent QN → heading text arrays.
 */
async function buildHeadingTextMap(docEntities) {
  const headingsByDoc = new Map();

  for (const entity of docEntities) {
    const parentQName = getParentQualifiedName(entity.qualified_name);
    if (!headingsByDoc.has(parentQName)) {
      headingsByDoc.set(parentQName, []);
    }

    try {
      const absolutePath = path.join(repoRoot, entity.file_path);
      const content = await readFile(absolutePath, 'utf8');
      const sectionText = content.slice(
        entity.char_start ?? 0,
        entity.char_end ?? content.length,
      );
      headingsByDoc.get(parentQName).push(sectionText);
    } catch {
      // Skip unreadable files.
    }
  }

  return headingsByDoc;
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
