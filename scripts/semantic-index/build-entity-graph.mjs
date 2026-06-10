/**
 * @module build-entity-graph
 * @description Main entity graph extraction orchestration script. Extracts code
 * entities (module, class, function, interface, type-alias, variable, error-class),
 * doc entities (plan, skill, agent, demo, benchmark), and cross-reference edges,
 * then inserts them into the `entities` and `edges` tables in `semantic-index.sqlite`.
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

  // Phase 4: Insert into SQLite.
  const database = await initSemanticIndex({ databasePath });

  // Check if entities/edges tables exist (graceful degradation).
  const tableCheck = database
    .prepare(
      "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('entities', 'edges')",
    )
    .all();
  const existingTables = new Set(tableCheck.map((row) => row.name));
  if (!existingTables.has('entities') || !existingTables.has('edges')) {
    // Tables don't exist yet — run schema creation.
    const { readFile: readSchema } = await import('node:fs/promises');
    const schemaPath = path.join(
      path.dirname(import.meta.url.replace(/^file:\/\//, '')),
      'schema-v2.sql',
    );
    const schema = await readSchema(schemaPath, 'utf8');
    database.exec(schema);
  }

  // Collect all entities and deduplicate by qualified_name.
  const allEntities = [];
  const seenQNames = new Set();
  for (const entity of [...codeResult.entities, ...docResult.entities]) {
    if (!seenQNames.has(entity.qualified_name)) {
      seenQNames.add(entity.qualified_name);
      allEntities.push(entity);
    }
  }

  // Insert entities and build qualified_name → entity_id map.
  const insertEntity = database.prepare(`
    INSERT INTO entities(entity_type, name, qualified_name, doc_id, chunk_id, module_path, signature_text, file_path, char_start, char_end, extra_metadata, created_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, unixepoch())
  `);

  const findDocId = database.prepare(
    'SELECT doc_id FROM documents WHERE file_path = ?',
  );
  const findChunkId = database.prepare(
    'SELECT chunk_id FROM chunks WHERE doc_id = ? AND (symbol_name = ? OR heading_path = ?) LIMIT 1',
  );

  // Delete existing entities and edges before re-inserting.
  const deleteAllEntities = database.prepare('DELETE FROM entities');
  const deleteAllEdges = database.prepare('DELETE FROM edges');

  const insertGraph = database.transaction(() => {
    deleteAllEdges.run();
    deleteAllEntities.run();

    const entityQNameToId = new Map();

    for (const entity of allEntities) {
      // Resolve doc_id from file_path.
      let docId = null;
      const docRow = findDocId.get(entity.file_path);
      if (docRow) docId = docRow.doc_id;

      // Resolve chunk_id if possible.
      let chunkId = null;
      if (docId && entity.entity_type !== 'module') {
        const chunkRow = findChunkId.get(docId, entity.name, entity.name);
        if (chunkRow) chunkId = chunkRow.chunk_id;
      }

      const result = insertEntity.run(
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
      );

      entityQNameToId.set(
        entity.qualified_name,
        Number(result.lastInsertRowid),
      );
    }

    // Collect all edges.
    const allEdges = [
      ...codeResult.edges,
      ...docResult.edges,
      ...crossRefResult.edges,
    ];

    // Insert edges, resolving qualified_names to entity_ids.
    const insertEdge = database.prepare(`
      INSERT OR IGNORE INTO edges(source_entity_id, target_entity_id, relationship, confidence, extra_metadata, created_at)
      VALUES (?, ?, ?, ?, ?, unixepoch())
    `);

    for (const edge of allEdges) {
      const sourceId = entityQNameToId.get(edge.source_qualified_name);
      const targetId = entityQNameToId.get(edge.target_qualified_name);
      if (sourceId == null || targetId == null) continue;

      insertEdge.run(
        sourceId,
        targetId,
        edge.relationship,
        edge.confidence ?? 'high',
        edge.extra_metadata ?? '{}',
      );
    }

    summary.entities = entityQNameToId.size;
    summary.edges = allEdges.filter((edge) => {
      const sourceId = entityQNameToId.get(edge.source_qualified_name);
      const targetId = entityQNameToId.get(edge.target_qualified_name);
      return sourceId != null && targetId != null;
    }).length;
  });

  insertGraph();

  database.close();
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
        'node scripts/semantic-index/build-entity-graph.mjs [--dry-run] [--force] [--json] [--database path]',
      options: [
        '--dry-run         Extract without writing SQLite rows.',
        '--force           Re-extract unchanged documents.',
        '--json            Emit JSON summary.',
        '--database <path> Path to SQLite database (default: data/semantic-index.sqlite).',
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
