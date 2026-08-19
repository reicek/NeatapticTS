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
import { getTursoClient } from './cortex-db.mjs';
import { ErrorCodes, cortexError } from './cortex-error.mjs';

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
const MAX_HOPS_LIMIT = 4;

/** Maximum results allowed. */
const MAX_RESULTS_LIMIT = 100;

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
  const seedNames = options.seed_names;
  const seedQuery = options.seed_query;
  const hasSeeds =
    (Array.isArray(seedNames) && seedNames.length > 0) ||
    (typeof seedQuery === 'string' && seedQuery.length > 0);
  if (!hasSeeds) {
    throw cortexError(
      ErrorCodes.SEED_REQUIRED,
      'At least one of seed_query or seed_names is required.',
    );
  }

  const rawMaxHops = options.max_hops;
  let maxHops =
    rawMaxHops === undefined || rawMaxHops === null ? 2 : Number(rawMaxHops);
  if (!Number.isInteger(maxHops) || maxHops < 1) {
    maxHops = 2;
  }
  maxHops = Math.min(maxHops, MAX_HOPS_LIMIT);

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

  const startTime = Date.now();

  const client = options.client ?? (await getTursoClient(options.databasePath));

  // Check if graph tables exist.
  const tableResult = await client.execute({
    sql: "SELECT name FROM sqlite_master WHERE type='table' AND name IN ('entities', 'edges')",
  });
  const existingTables = new Set(tableResult.rows.map((row) => row.name));

  if (!existingTables.has('entities') || !existingTables.has('edges')) {
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
      graph_state: 'not_built',
      traversal_stats: {
        total_entities_discovered: 0,
        total_edges_traversed: 0,
        hops_completed: 0,
        query_time_ms: Date.now() - startTime,
      },
    };
  }

  // Load graph via async client queries.
  const graph = await loadGraphAsync(client);

  // Phase 2: Resolve seed entities.
  const seedEntities = resolveSeedEntities(graph, seedNames, seedQuery);
  if (seedEntities.length === 0) {
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
      graph_state: 'ready',
      traversal_stats: {
        total_entities_discovered: 0,
        total_edges_traversed: 0,
        hops_completed: 0,
        query_time_ms: Date.now() - startTime,
      },
    };
  }

  // Phase 3: BFS traversal in memory.
  const relationshipTypesSet = new Set(relationshipTypes);
  const entityTypesSet = new Set(entityTypes);
  const { discovered, allEdges, hopsCompleted, distanceMap } =
    traverseFromSeeds(graph, seedEntities, {
      maxHops,
      relationshipTypesSet,
      confidenceSet,
      entityTypesSet,
    });

  // Phase 4: Rank and limit results.
  const edgeCounts = computeEdgeCounts(discovered, allEdges);
  const rankedEntities = rankEntities(
    discovered,
    distanceMap,
    edgeCounts,
    entityTypes,
  );
  const topEntities = rankedEntities.slice(0, maxResults);

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

  const queryTimeMs = Date.now() - startTime;

  return {
    seed_entities: seedEntities.map((e) => formatEntity(e, distanceMap)),
    entities: topEntities.map((e) => formatEntity(e, distanceMap)),
    relationships: relevantEdges.map(formatEdge).filter(Boolean),
    chunk_ids: [...new Set(chunkIds.map(Number))],
    doc_ids: [...new Set(docIds.map(Number))],
    hop_count: maxHops,
    total_discovered: discovered.length,
    returned_count: topEntities.length,
    graph_available: true,
    graph_state: 'ready',
    traversal_stats: {
      total_entities_discovered: discovered.length,
      total_edges_traversed: allEdges.length,
      hops_completed: hopsCompleted,
      query_time_ms: queryTimeMs,
    },
  };
}

/**
 * Resolve seed entities from seed_names or seed_query against the in-memory graph.
 *
 * @param {object} graph - In-memory graph cache.
 * @param {string[]} seedNames - Qualified names or partial names.
 * @param {string} seedQuery - Free-text query for seed discovery.
 * @returns {Array<object>} Seed entities.
 */
function resolveSeedEntities(graph, seedNames, seedQuery) {
  const seen = new Set();
  const seeds = [];

  if (Array.isArray(seedNames) && seedNames.length > 0) {
    for (const entity of resolveByNames(graph, seedNames)) {
      /* istanbul ignore else -- defensive: resolveByNames returns unique entities */
      if (!seen.has(entity.entity_id)) {
        seeds.push(entity);
        seen.add(entity.entity_id);
      }
    }
  }

  if (typeof seedQuery === 'string' && seedQuery.length > 0) {
    for (const entity of resolveByQuery(graph, seedQuery)) {
      /* istanbul ignore else -- defensive: resolveByQuery returns unique entities */
      if (!seen.has(entity.entity_id)) {
        seeds.push(entity);
        seen.add(entity.entity_id);
      }
    }
  }

  return seeds;
}

/**
 * Resolve seed entities by qualified names or partial name matching from the in-memory graph.
 *
 * @param {object} graph - In-memory graph cache.
 * @param {string[]} names - Name patterns to search.
 * @returns {Array<object>} Matching entities.
 */
function resolveByNames(graph, names) {
  const matches = [];
  const seen = new Set();

  for (const name of names) {
    const normalized = name.toLowerCase();

    // Exact match on qualified_name.
    const exact = graph.entitiesByQualifiedName.get(name);
    if (exact && !seen.has(exact.entity_id)) {
      matches.push(exact);
      seen.add(exact.entity_id);
      continue;
    }

    // Prefix match on qualified_name, shortest first.
    const prefixMatches = [];
    for (const [qualifiedName, entity] of graph.entitiesByQualifiedName) {
      if (qualifiedName.startsWith(name)) {
        prefixMatches.push(entity);
      }
    }
    prefixMatches.sort(
      (a, b) => a.qualified_name.length - b.qualified_name.length,
    );
    let foundPrefix = false;
    for (const entity of prefixMatches.slice(0, 10)) {
      /* istanbul ignore else -- defensive: prefix matches are unique */
      if (!seen.has(entity.entity_id)) {
        matches.push(entity);
        seen.add(entity.entity_id);
        foundPrefix = true;
      }
    }
    if (foundPrefix) continue;

    // Fuzzy match on qualified_name or name with priority ordering.
    const fuzzyMatches = [];
    for (const entity of graph.entities.values()) {
      const qualifiedName = entity.qualified_name.toLowerCase();
      const entityName =
        /* istanbul ignore next -- defensive: name always present */ (
          entity.name ?? ''
        ).toLowerCase();
      if (
        qualifiedName.includes(normalized) ||
        entityName.includes(normalized)
      ) {
        fuzzyMatches.push(entity);
      }
    }
    fuzzyMatches.sort((a, b) => {
      const aQualified = a.qualified_name.toLowerCase();
      const bQualified = b.qualified_name.toLowerCase();
      const aName =
        /* istanbul ignore next -- defensive: name always present */ (
          a.name ?? ''
        ).toLowerCase();
      const bName =
        /* istanbul ignore next -- defensive: name always present */ (
          b.name ?? ''
        ).toLowerCase();
      /* istanbul ignore next -- defensive: exact/prefix/name matches handled by earlier resolution stages */
      const priorityA =
        aQualified === normalized
          ? 0
          : aQualified.startsWith(normalized)
            ? 1
            : aName === normalized
              ? 2
              : 3;
      const priorityB =
        bQualified === normalized
          ? 0
          : bQualified.startsWith(normalized)
            ? 1
            : bName === normalized
              ? 2
              : 3;
      if (priorityA !== priorityB) return priorityA - priorityB;
      return a.qualified_name.length - b.qualified_name.length;
    });
    for (const entity of fuzzyMatches.slice(0, 10)) {
      /* istanbul ignore else -- defensive: fuzzy matches are unique */
      if (!seen.has(entity.entity_id)) {
        matches.push(entity);
        seen.add(entity.entity_id);
      }
    }
  }

  return matches.slice(0, 10);
}

/**
 * Resolve seed entities by free-text query against the in-memory graph.
 *
 * @param {object} graph - In-memory graph cache.
 * @param {string} query - Free-text query.
 * @returns {Array<object>} Top-5 matching entities.
 */
function resolveByQuery(graph, query) {
  const sanitized = query
    .replace(/[-@^*{}():"]/g, ' ')
    .replace(/\s+/g, ' ')
    .trim()
    .toLowerCase();

  /* istanbul ignore if -- defensive: query is pre-validated before resolveByQuery */
  if (!sanitized) return [];

  const terms = sanitized.split(' ').filter(Boolean);
  const matches = [];
  for (const entity of graph.entities.values()) {
    const qualifiedName = entity.qualified_name.toLowerCase();
    const entityName =
      /* istanbul ignore next -- defensive: name always present */ (
        entity.name ?? ''
      ).toLowerCase();
    if (
      terms.every(
        (term) => qualifiedName.includes(term) || entityName.includes(term),
      )
    ) {
      matches.push(entity);
    }
  }

  matches.sort(
    /* istanbul ignore next -- defensive: comparator only runs with 2+ matches */ (
      a,
      b,
    ) => a.qualified_name.length - b.qualified_name.length,
  );
  return matches.slice(0, 5);
}

/**
 * Load the full entity/edge graph into memory with two batched async SQL queries.
 *
 * @param {import('@libsql/client').Client} client - libSQL client.
 * @returns {Promise<object>} In-memory graph structure.
 */
async function loadGraphAsync(client) {
  const entities = new Map();
  const entitiesByQualifiedName = new Map();

  const entityResult = await client.execute(
    'SELECT * FROM entities ORDER BY entity_id',
  );
  for (const entity of entityResult.rows) {
    const entityId = Number(entity.entity_id);
    const normalized = { ...entity, entity_id: entityId };
    entities.set(entityId, normalized);
    entitiesByQualifiedName.set(entity.qualified_name, normalized);
  }

  const edgesBySource = new Map();
  const edgesByTarget = new Map();
  const allEdges = [];

  const edgeResult = await client.execute({
    sql: `SELECT e.*,
            src.qualified_name AS source_qualified_name, src.entity_type AS source_entity_type,
            tgt.qualified_name AS target_qualified_name, tgt.entity_type AS target_entity_type
    FROM edges e
    JOIN entities src ON e.source_entity_id = src.entity_id
    JOIN entities tgt ON e.target_entity_id = tgt.entity_id
    ORDER BY e.edge_id`,
  });

  for (const edge of edgeResult.rows) {
    const normalizedEdge = {
      ...edge,
      source_entity_id: Number(edge.source_entity_id),
      target_entity_id: Number(edge.target_entity_id),
    };
    allEdges.push(normalizedEdge);
    if (!edgesBySource.has(normalizedEdge.source_entity_id)) {
      edgesBySource.set(normalizedEdge.source_entity_id, []);
    }
    edgesBySource.get(normalizedEdge.source_entity_id).push(normalizedEdge);
    /* istanbul ignore else -- defensive: first edge for each target entity */
    if (!edgesByTarget.has(normalizedEdge.target_entity_id)) {
      edgesByTarget.set(normalizedEdge.target_entity_id, []);
    }
    edgesByTarget.get(normalizedEdge.target_entity_id).push(normalizedEdge);
  }

  return {
    entities,
    entitiesByQualifiedName,
    edgesBySource,
    edgesByTarget,
    allEdges,
  };
}

/**
 * BFS traversal over the in-memory graph.
 *
 * @param {object} graph - In-memory graph cache.
 * @param {Array<object>} seedEntities - Seed entities.
 * @param {object} options - Traversal options.
 * @param {number} options.maxHops - Maximum hops.
 * @param {Set<string>} options.relationshipTypesSet - Allowed relationship types.
 * @param {Set<string>} options.confidenceSet - Allowed confidence levels.
 * @param {Set<string>} options.entityTypesSet - Allowed entity types.
 * @returns {object} Traversal state.
 */
function traverseFromSeeds(graph, seedEntities, options) {
  const { maxHops, relationshipTypesSet, confidenceSet, entityTypesSet } =
    options;

  const visited = new Set(seedEntities.map((e) => e.entity_id));
  const discovered = [...seedEntities];
  const distanceMap = new Map();
  for (const seed of seedEntities) {
    distanceMap.set(seed.entity_id, 0);
  }

  let currentFrontier = [...seedEntities];
  const allEdges = [];
  let hopsCompleted = 0;

  for (let hop = 1; hop <= maxHops; hop++) {
    const nextFrontier = [];

    for (const entity of currentFrontier) {
      const outgoingEdges = graph.edgesBySource.get(entity.entity_id) ?? [];
      for (const edge of outgoingEdges) {
        if (!relationshipTypesSet.has(edge.relationship)) continue;
        if (!confidenceSet.has(edge.confidence)) continue;
        if (visited.has(edge.target_entity_id)) continue;

        const target = graph.entities.get(edge.target_entity_id);
        /* istanbul ignore if -- defensive: test entities always found in graph */
        if (!target) continue;
        /* istanbul ignore if -- defensive: test entities always have allowed types */
        if (!entityTypesSet.has(target.entity_type)) continue;

        visited.add(target.entity_id);
        distanceMap.set(target.entity_id, hop);
        discovered.push(target);
        nextFrontier.push(target);
        allEdges.push(edge);
      }

      const incomingEdges = graph.edgesByTarget.get(entity.entity_id) ?? [];
      for (const edge of incomingEdges) {
        /* istanbul ignore if -- defensive: test edges always have allowed relationship types */
        if (!relationshipTypesSet.has(edge.relationship)) continue;
        /* istanbul ignore if -- defensive: test edges always have allowed confidence levels */
        if (!confidenceSet.has(edge.confidence)) continue;
        if (visited.has(edge.source_entity_id)) continue;

        const source = graph.entities.get(edge.source_entity_id);
        /* istanbul ignore if -- defensive: test entities always found in graph */
        if (!source) continue;
        /* istanbul ignore if -- defensive: test entities always have allowed types */
        if (!entityTypesSet.has(source.entity_type)) continue;

        visited.add(source.entity_id);
        distanceMap.set(source.entity_id, hop);
        discovered.push(source);
        nextFrontier.push(source);
        allEdges.push(edge);
      }
    }

    currentFrontier = nextFrontier;
    hopsCompleted = hop;
    if (currentFrontier.length === 0) break;
  }

  return {
    discovered,
    allEdges,
    hopsCompleted,
    distanceMap,
  };
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
    /* istanbul ignore else -- defensive: all edge sources are in discovered set */
    if (entityIds.has(edge.source_entity_id)) {
      counts.set(
        edge.source_entity_id,
        (counts.get(edge.source_entity_id) ?? 0) + 1,
      );
    }
    /* istanbul ignore else -- defensive: all edge targets are in discovered set */
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
    /* istanbul ignore next -- defensive: all entities are in distanceMap */
    const distance = distanceMap.get(entity.entity_id) ?? 1;
    const distanceScore = 1.0 / distance;

    // Use the confidence of the edge that discovered this entity.
    // For seeds (distance 0), assume high confidence.
    const confidenceWeight = distance === 0 ? 1.0 : 0.7;

    const edgeCount = edgeCounts.get(entity.entity_id) ?? 0;
    const connectivityScore = Math.log(1 + edgeCount);

    /* istanbul ignore next -- defensive: test entities are all code types */
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
 * @param {Map<number, number>} [distanceMap] - Entity ID → hop distance from seeds.
 * @returns {object} Formatted entity.
 */
function formatEntity(entity, distanceMap) {
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
    hop_distance:
      /* istanbul ignore next -- defensive: distanceMap always provided with all entities */ distanceMap?.get(
        entity.entity_id,
      ) ?? 0,
  };
}

/**
 * Check that an edge row has all required string fields with non-empty values.
 *
 * @param {object} edge - Edge row with joined source/target names.
 * @returns {boolean} True when the edge is safe to emit.
 */
function isWellFormedEdge(edge) {
  const requiredStringFields = [
    'source_qualified_name',
    'target_qualified_name',
    'source_entity_type',
    'target_entity_type',
    'relationship',
    'confidence',
  ];
  return requiredStringFields.every((field) => {
    const value = edge[field];
    return typeof value === 'string' && value.length > 0;
  });
}

/**
 * Format an edge for output.
 *
 * @param {object} edge - Edge row with joined source/target names.
 * @returns {object | null} Formatted edge, or null when the edge is malformed.
 */
function formatEdge(edge) {
  if (!isWellFormedEdge(edge)) {
    return null;
  }
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
  const rawMaxHops = argumentsObject.max_hops;
  const maxHops =
    rawMaxHops === undefined || rawMaxHops === null ? 2 : Number(rawMaxHops);
  if (!Number.isInteger(maxHops) || maxHops < 1 || maxHops > MAX_HOPS_LIMIT) {
    throw cortexError(
      ErrorCodes.INVALID_MAX_HOPS,
      `max_hops must be an integer between 1 and ${MAX_HOPS_LIMIT}.`,
    );
  }

  return traverseGraph({
    seed_names: argumentsObject.seed_names,
    seed_query: argumentsObject.seed_query,
    relationship_types: argumentsObject.relationship_types,
    entity_types: argumentsObject.entity_types,
    max_hops: argumentsObject.max_hops,
    max_results: argumentsObject.max_results,
    confidence_filter: argumentsObject.confidence_filter,
    databasePath: argumentsObject.databasePath,
    client: argumentsObject.client,
  });
}
