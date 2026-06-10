/**
 * @module traverse-graph
 * @description BFS multi-hop traversal of the entity/relationship graph. Resolves seed
 * entities from qualified names or text search, then follows edges of specified
 * relationship types for a configurable number of hops. Returns discovered entities,
 * relationships, and associated chunk_ids for context expansion.
 *
 * Ranking heuristic: distance (1/d) × confidence weight × connectivity × type preference.
 *
 * @param {string}  [seed_query]  - Free-text query for seed entity discovery.
 * @param {Array}   [seed_names]  - Qualified names or partial names for seed resolution.
 * @param {Array}  [relationship_types] - Relationship types to follow.
 * @param {Array}  [entity_types] - Entity types to include in results.
 * @param {number}  [max_hops]    - Maximum traversal depth (default: 2, max: 3).
 * @param {number}  [max_results] - Maximum entities to return (default: 20, max: 50).
 * @param {Array}  [confidence_filter] - Minimum confidence levels to follow.
 * @param {string}  [databasePath] - Override database path.
 *
 * @returns {object} Traversal results with entities, relationships, chunk_ids, doc_ids.
 */
import Database from 'better-sqlite3';
import path from 'node:path';

import {
  defaultDatabasePath,
  repoRoot,
} from '../../semantic-index/init-schema.mjs';

/** All valid relationship types. */
const ALL_RELATIONSHIP_TYPES = [
  'imports',
  'exports',
  'depends-on',
  'implements',
  'references',
  'owns',
  'part-of',
  'contains',
];

/** All valid entity types. */
const ALL_ENTITY_TYPES = [
  'module',
  'class',
  'function',
  'interface',
  'type-alias',
  'variable',
  'error-class',
  'plan',
  'skill',
  'agent',
  'demo',
  'benchmark',
];

/** Confidence weight mapping. */
const CONFIDENCE_WEIGHTS = { high: 1.0, medium: 0.7, low: 0.4 };

/** Code entity type preference. */
const CODE_ENTITY_TYPES = new Set([
  'module',
  'class',
  'function',
  'interface',
  'type-alias',
  'variable',
  'error-class',
]);

/** Maximum hops allowed. */
const MAX_HOPS_LIMIT = 3;

/** Maximum results allowed. */
const MAX_RESULTS_LIMIT = 50;

/**
 * Traverse the entity/relationship graph from seed entities.
 *
 * BFS traversal following specified relationship types, with confidence-weighted
 * ordering and type preference scoring.
 *
 * @param {object} options - Traversal options.
 * @param {string} [options.seed_query] - Free-text query for seed entity discovery.
 * @param {string[]} [options.seed_names] - Qualified names or partial names for seeds.
 * @param {string[]} [options.relationship_types] - Relationship types to follow.
 * @param {string[]} [options.entity_types] - Entity types to include in results.
 * @param {number} [options.max_hops=2] - Maximum traversal depth.
 * @param {number} [options.max_results=20] - Maximum entities to return.
 * @param {string[]} [options.confidence_filter] - Minimum confidence levels.
 * @param {string} [options.databasePath] - Override database path.
 * @returns {Promise<object>} Traversal results.
 */
export async function traverseGraph(options = {}) {
  const maxHops = Math.min(
    Math.max(Number(options.max_hops ?? 2), 1),
    MAX_HOPS_LIMIT,
  );
  const maxResults = Math.min(
    Math.max(Number(options.max_results ?? 20), 1),
    MAX_RESULTS_LIMIT,
  );
  const relationshipTypes = validateArrayOption(
    options.relationship_types,
    ALL_RELATIONSHIP_TYPES,
  );
  const entityTypes = validateArrayOption(
    options.entity_types,
    ALL_ENTITY_TYPES,
  );
  const confidenceFilter = validateArrayOption(options.confidence_filter, [
    'high',
    'medium',
    'low',
  ]);
  const confidenceSet = new Set(confidenceFilter);

  const databasePath = path.resolve(
    options.databasePath ?? defaultDatabasePath,
  );
  const database = new Database(databasePath, {
    readonly: true,
    fileMustExist: false,
  });

  // Check if graph tables exist.
  const tableCheck = database
    .prepare(
      "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('entities', 'edges')",
    )
    .all();
  const existingTables = new Set(tableCheck.map((row) => row.name));

  if (!existingTables.has('entities') || !existingTables.has('edges')) {
    database.close();
    return {
      seed_entities: [],
      entities: [],
      relationships: [],
      chunk_ids: [],
      doc_ids: [],
      hop_count: maxHops,
      total_discovered: 0,
      returned_count: 0,
      graph_available: false,
    };
  }

  // Phase 1: Resolve seed entities.
  const seedEntities = resolveSeedEntities(
    database,
    options.seed_names,
    options.seed_query,
  );
  if (seedEntities.length === 0) {
    database.close();
    return {
      seed_entities: [],
      entities: [],
      relationships: [],
      chunk_ids: [],
      doc_ids: [],
      hop_count: maxHops,
      total_discovered: 0,
      returned_count: 0,
      graph_available: true,
    };
  }

  // Phase 2: BFS traversal.
  const visited = new Set(seedEntities.map((e) => e.entity_id));
  const discovered = [...seedEntities];
  const distanceMap = new Map();
  for (const seed of seedEntities) {
    distanceMap.set(seed.entity_id, 0);
  }

  let currentFrontier = [...seedEntities];
  const allEdges = [];

  for (let hop = 1; hop <= maxHops; hop++) {
    const nextFrontier = [];

    for (const entity of currentFrontier) {
      // Follow outgoing edges.
      const outgoingEdges = queryOutgoingEdges(
        database,
        entity.entity_id,
        relationshipTypes,
        confidenceSet,
      );
      for (const edge of outgoingEdges) {
        const target = getEntityById(database, edge.target_entity_id);
        if (!target || visited.has(target.entity_id)) continue;
        if (!entityTypes.includes(target.entity_type)) continue;

        visited.add(target.entity_id);
        distanceMap.set(target.entity_id, hop);
        discovered.push(target);
        nextFrontier.push(target);
        allEdges.push(edge);
      }

      // Follow incoming edges (reverse traversal).
      const incomingEdges = queryIncomingEdges(
        database,
        entity.entity_id,
        relationshipTypes,
        confidenceSet,
      );
      for (const edge of incomingEdges) {
        const source = getEntityById(database, edge.source_entity_id);
        if (!source || visited.has(source.entity_id)) continue;
        if (!entityTypes.includes(source.entity_type)) continue;

        visited.add(source.entity_id);
        distanceMap.set(source.entity_id, hop);
        discovered.push(source);
        nextFrontier.push(source);
        allEdges.push(edge);
      }
    }

    currentFrontier = nextFrontier;
    if (currentFrontier.length === 0) break;
  }

  // Phase 3: Rank and limit results.
  const edgeCounts = computeEdgeCounts(discovered, allEdges);
  const rankedEntities = rankEntities(
    discovered,
    distanceMap,
    edgeCounts,
    entityTypes,
  );
  const topEntities = rankedEntities.slice(0, maxResults);

  // Phase 4: Collect associated chunks and docs.
  const topEntityIds = new Set(topEntities.map((e) => e.entity_id));
  const relevantEdges = allEdges.filter(
    (edge) =>
      topEntityIds.has(edge.source_entity_id) ||
      topEntityIds.has(edge.target_entity_id),
  );

  const chunkIds = topEntities
    .filter((e) => e.chunk_id != null)
    .map((e) => e.chunk_id);

  const docIds = topEntities
    .filter((e) => e.chunk_id == null && e.doc_id != null)
    .map((e) => e.doc_id);

  database.close();

  return {
    seed_entities: seedEntities.map(formatEntity),
    entities: topEntities.map(formatEntity),
    relationships: relevantEdges.map(formatEdge),
    chunk_ids: [...new Set(chunkIds)],
    doc_ids: [...new Set(docIds)],
    hop_count: maxHops,
    total_discovered: discovered.length,
    returned_count: topEntities.length,
    graph_available: true,
  };
}

/**
 * Resolve seed entities from seed_names or seed_query.
 *
 * @param {object} database - SQLite database connection.
 * @param {string[]} seedNames - Qualified names or partial names.
 * @param {string} seedQuery - Free-text query for seed discovery.
 * @returns {Array<object>} Seed entities.
 */
function resolveSeedEntities(database, seedNames, seedQuery) {
  if (Array.isArray(seedNames) && seedNames.length > 0) {
    return resolveByNames(database, seedNames);
  }

  if (typeof seedQuery === 'string' && seedQuery.length > 0) {
    return resolveByQuery(database, seedQuery);
  }

  return [];
}

/**
 * Resolve seed entities by qualified names or partial name matching.
 *
 * Uses SQL LIKE with wildcards for prefix matching.
 *
 * @param {object} database - SQLite database.
 * @param {string[]} names - Name patterns to search.
 * @returns {Array<object>} Matching entities.
 */
function resolveByNames(database, names) {
  const entities = [];
  const findExact = database.prepare(
    'SELECT * FROM entities WHERE qualified_name = ? LIMIT 10',
  );
  const findPrefix = database.prepare(
    "SELECT * FROM entities WHERE qualified_name LIKE ? || '%' ORDER BY LENGTH(qualified_name) LIMIT 10",
  );
  const findFuzzy = database.prepare(
    "SELECT * FROM entities WHERE qualified_name LIKE '%' || ? || '%' OR name LIKE '%' || ? || '%' ORDER BY CASE WHEN qualified_name = ? THEN 0 WHEN qualified_name LIKE ? || '%' THEN 1 WHEN name = ? THEN 2 ELSE 3 END, LENGTH(qualified_name) LIMIT 10",
  );

  const seen = new Set();
  for (const name of names) {
    // Try exact match first.
    const exact = findExact.get(name);
    if (exact && !seen.has(exact.entity_id)) {
      entities.push(exact);
      seen.add(exact.entity_id);
      continue;
    }

    // Try prefix match.
    const prefix = findPrefix.all(name);
    for (const entity of prefix) {
      if (!seen.has(entity.entity_id)) {
        entities.push(entity);
        seen.add(entity.entity_id);
      }
    }
    if (entities.length > 0) continue;

    // Try fuzzy match.
    const fuzzy = findFuzzy.all(name, name, name, name, name);
    for (const entity of fuzzy) {
      if (!seen.has(entity.entity_id)) {
        entities.push(entity);
        seen.add(entity.entity_id);
      }
    }
  }

  return entities.slice(0, 10);
}

/**
 * Resolve seed entities by free-text query using FTS on entity names.
 *
 * @param {object} database - SQLite database.
 * @param {string} query - Free-text query.
 * @returns {Array<object>} Top-5 matching entities.
 */
function resolveByQuery(database, query) {
  const sanitized = query
    .replace(/[-@^*{}():"]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim();

  if (!sanitized) return [];

  const terms = sanitized.split(' ').filter(Boolean);
  const likeClauses = terms
    .map(() => '(qualified_name LIKE ? OR name LIKE ?)')
    .join(' AND ');
  const params = terms.flatMap((term) => [`%${term}%`, `%${term}%`]);

  const stmt = database.prepare(
    `SELECT * FROM entities WHERE ${likeClauses} ORDER BY LENGTH(qualified_name) LIMIT 5`,
  );

  return stmt.all(...params);
}

/**
 * Query outgoing edges from an entity with relationship and confidence filters.
 *
 * @param {object} database - SQLite database.
 * @param {number} entityId - Source entity ID.
 * @param {string[]} relationshipTypes - Relationship types to follow.
 * @param {Set} confidenceSet - Allowed confidence levels.
 * @returns {Array<object>} Matching edges with source and target names.
 */
function queryOutgoingEdges(
  database,
  entityId,
  relationshipTypes,
  confidenceSet,
) {
  const placeholders = relationshipTypes.map(() => '?').join(',');
  const confidencePlaceholders = [...confidenceSet].map(() => '?').join(',');
  const stmt = database.prepare(`
    SELECT e.*, 
           src.qualified_name AS source_qualified_name, src.entity_type AS source_entity_type,
           tgt.qualified_name AS target_qualified_name, tgt.entity_type AS target_entity_type
    FROM edges e
    JOIN entities src ON e.source_entity_id = src.entity_id
    JOIN entities tgt ON e.target_entity_id = tgt.entity_id
    WHERE e.source_entity_id = ?
      AND e.relationship IN (${placeholders})
      AND e.confidence IN (${confidencePlaceholders})
  `);

  const params = [entityId, ...relationshipTypes, ...confidenceSet];
  return stmt.all(...params);
}

/**
 * Query incoming edges to an entity with relationship and confidence filters.
 *
 * @param {object} database - SQLite database.
 * @param {number} entityId - Target entity ID.
 * @param {string[]} relationshipTypes - Relationship types to follow.
 * @param {Set} confidenceSet - Allowed confidence levels.
 * @returns {Array<object>} Matching edges.
 */
function queryIncomingEdges(
  database,
  entityId,
  relationshipTypes,
  confidenceSet,
) {
  const placeholders = relationshipTypes.map(() => '?').join(',');
  const confidencePlaceholders = [...confidenceSet].map(() => '?').join(',');
  const stmt = database.prepare(`
    SELECT e.*,
           src.qualified_name AS source_qualified_name, src.entity_type AS source_entity_type,
           tgt.qualified_name AS target_qualified_name, tgt.entity_type AS target_entity_type
    FROM edges e
    JOIN entities src ON e.source_entity_id = src.entity_id
    JOIN entities tgt ON e.target_entity_id = tgt.entity_id
    WHERE e.target_entity_id = ?
      AND e.relationship IN (${placeholders})
      AND e.confidence IN (${confidencePlaceholders})
  `);

  const params = [entityId, ...relationshipTypes, ...confidenceSet];
  return stmt.all(...params);
}

/**
 * Get an entity by its ID.
 *
 * @param {object} database - SQLite database.
 * @param {number} entityId - Entity ID.
 * @returns {object | null} Entity row or null.
 */
function getEntityById(database, entityId) {
  const stmt = database.prepare('SELECT * FROM entities WHERE entity_id = ?');
  return stmt.get(entityId) ?? null;
}

/**
 * Compute edge counts for discovered entities.
 *
 * @param {Array} discovered - Discovered entities.
 * @param {Array} allEdges - All discovered edges.
 * @returns {Map<number, number>} Entity ID → edge count.
 */
function computeEdgeCounts(discovered, allEdges) {
  const counts = new Map();
  const entityIds = new Set(discovered.map((e) => e.entity_id));

  for (const edge of allEdges) {
    if (entityIds.has(edge.source_entity_id)) {
      counts.set(
        edge.source_entity_id,
        (counts.get(edge.source_entity_id) ?? 0) + 1,
      );
    }
    if (entityIds.has(edge.target_entity_id)) {
      counts.set(
        edge.target_entity_id,
        (counts.get(edge.target_entity_id) ?? 0) + 1,
      );
    }
  }

  return counts;
}

/**
 * Rank discovered entities by composite score.
 *
 * Score = (1/distance) × confidence_weight × (1 + 0.1 × connectivity) × type_preference
 *
 * @param {Array} discovered - Discovered entities.
 * @param {Map} distanceMap - Entity ID → distance from seed.
 * @param {Map} edgeCounts - Entity ID → edge count.
 * @param {string[]} entityTypes - Allowed entity types.
 * @returns {Array} Ranked entities (highest score first).
 */
function rankEntities(discovered, distanceMap, edgeCounts, entityTypes) {
  const scored = discovered.map((entity) => {
    const distance = distanceMap.get(entity.entity_id) ?? 1;
    const distanceScore = 1.0 / distance;

    // Use the confidence of the edge that discovered this entity.
    // For seeds (distance 0), assume high confidence.
    const confidenceWeight = distance === 0 ? 1.0 : 0.7;

    const edgeCount = edgeCounts.get(entity.entity_id) ?? 0;
    const connectivityScore = Math.log(1 + edgeCount);

    const typePreference = CODE_ENTITY_TYPES.has(entity.entity_type)
      ? 1.0
      : 0.8;

    const score =
      distanceScore *
      confidenceWeight *
      (1 + 0.1 * connectivityScore) *
      typePreference;

    return { ...entity, _rank: score };
  });

  return scored.toSorted((a, b) => b._rank - a._rank);
}

/**
 * Format an entity for output.
 *
 * @param {object} entity - Entity row with optional _rank.
 * @returns {object} Formatted entity.
 */
function formatEntity(entity) {
  return {
    entity_id: entity.entity_id,
    entity_type: entity.entity_type,
    name: entity.name,
    qualified_name: entity.qualified_name,
    doc_id: entity.doc_id,
    chunk_id: entity.chunk_id,
    module_path: entity.module_path,
    signature_text: entity.signature_text,
    file_path: entity.file_path,
  };
}

/**
 * Format an edge for output.
 *
 * @param {object} edge - Edge row with joined source/target names.
 * @returns {object} Formatted edge.
 */
function formatEdge(edge) {
  return {
    edge_id: edge.edge_id,
    source_entity_id: edge.source_entity_id,
    target_entity_id: edge.target_entity_id,
    source_qualified_name: edge.source_qualified_name,
    target_qualified_name: edge.target_qualified_name,
    source_entity_type: edge.source_entity_type,
    target_entity_type: edge.target_entity_type,
    relationship: edge.relationship,
    confidence: edge.confidence,
  };
}

/**
 * Validate an array option against allowed values.
 *
 * @param {Array | undefined} value - User-provided value.
 * @param {Array} allowedValues - All valid values.
 * @returns {Array} Filtered array of valid values, or all values if input is empty.
 */
function validateArrayOption(value, allowedValues) {
  if (!Array.isArray(value) || value.length === 0) return allowedValues;
  const allowedSet = new Set(allowedValues);
  const filtered = value.filter((item) => allowedSet.has(item));
  return filtered.length > 0 ? filtered : allowedValues;
}

/**
 * MCP tool handler for traverse_graph.
 *
 * @param {object} argumentsObject - Tool input arguments.
 * @returns {Promise<object>} Traversal results.
 */
export async function traverseGraphHandler(argumentsObject) {
  return traverseGraph({
    seed_names: argumentsObject.seed_names,
    seed_query: argumentsObject.seed_query,
    relationship_types: argumentsObject.relationship_types,
    entity_types: argumentsObject.entity_types,
    max_hops: argumentsObject.max_hops,
    max_results: argumentsObject.max_results,
    confidence_filter: argumentsObject.confidence_filter,
    databasePath: argumentsObject.databasePath,
  });
}
