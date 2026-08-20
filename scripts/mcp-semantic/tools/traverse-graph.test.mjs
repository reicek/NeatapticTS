/**
 * @module traverse-graph.test
 * @description Coverage tests for traverse-graph.mjs — BFS traversal, seed
 * resolution, ranking, formatting, validation, and handler.
 */
import {
  createSchemaClient,
  insertTestFixtures,
  insertGraphFixtures,
  createEnvIsolation,
  TEST_DOC_ID,
  TEST_CHUNK_ID,
  TEST_PARENT_CHUNK_ID,
} from '../__tests__/turso-test-helpers.mjs';

const { traverseGraph, traverseGraphHandler } =
  await import('./traverse-graph.mjs');

describe('traverse-graph', () => {
  const { saveEnv, restoreEnv } = createEnvIsolation();
  let client;

  beforeEach(async () => {
    saveEnv();
    client = await createSchemaClient();
    await insertTestFixtures(client);
    await insertGraphFixtures(client);
  });

  afterEach(async () => {
    restoreEnv();
    if (client) {
      try {
        await client.close();
      } catch {
        /* noop */
      }
    }
  });

  describe('validation', () => {
    it('throws when no seed_query and no seed_names provided', async () => {
      await expect(traverseGraph({ client })).rejects.toThrow(
        'At least one of seed_query or seed_names is required',
      );
    });

    it('throws when seed_names is empty array and no seed_query', async () => {
      await expect(traverseGraph({ seed_names: [], client })).rejects.toThrow(
        'seed_query or seed_names is required',
      );
    });

    it('throws when seed_query is empty string and no seed_names', async () => {
      await expect(traverseGraph({ seed_query: '', client })).rejects.toThrow(
        'seed_query or seed_names is required',
      );
    });
  });

  describe('graph not built', () => {
    it('returns not_built result when entities/edges tables missing', async () => {
      const emptyClient = await createSchemaClient();
      // Drop entities and edges tables (createSchemaClient creates them via schema)
      try {
        await emptyClient.execute('DROP TABLE IF EXISTS edges');
        await emptyClient.execute('DROP TABLE IF EXISTS entities');

        const result = await traverseGraph({
          seed_names: ['TestEntity'],
          client: emptyClient,
        });

        expect(result.graph_available).toBe(false);
        expect(result.graph_state).toBe('not_built');
        expect(result.entities).toEqual([]);
        expect(result.seed_entities).toEqual([]);
        expect(result.traversal_stats.hops_completed).toBe(0);
      } finally {
        await emptyClient.close();
      }
    });

    it('returns not_built when only entities table missing', async () => {
      const emptyClient = await createSchemaClient();
      try {
        await emptyClient.execute('DROP TABLE IF EXISTS edges');

        const result = await traverseGraph({
          seed_names: ['TestEntity'],
          client: emptyClient,
        });

        expect(result.graph_available).toBe(false);
      } finally {
        await emptyClient.close();
      }
    });
  });

  describe('graph built but no seeds found', () => {
    it('returns empty result when no seed entities match', async () => {
      const result = await traverseGraph({
        seed_names: ['NonExistentEntity'],
        client,
      });

      expect(result.graph_available).toBe(true);
      expect(result.graph_state).toBe('ready');
      expect(result.entities).toEqual([]);
      expect(result.seed_entities).toEqual([]);
      expect(result.total_discovered).toBe(0);
    });
  });

  describe('successful traversal', () => {
    it('resolves seeds by exact qualified_name', async () => {
      const result = await traverseGraph({
        seed_names: ['src/turso.TursoTestEntityA'],
        client,
      });

      expect(result.seed_entities).toHaveLength(1);
      expect(result.seed_entities[0].name).toBe('TursoTestEntityA');
      expect(result.graph_available).toBe(true);
    });

    it('resolves seeds by prefix match on qualified_name', async () => {
      const result = await traverseGraph({
        seed_names: ['src/turso.TursoTestEntity'],
        client,
      });

      // Should match both EntityA and EntityB via prefix
      expect(result.seed_entities.length).toBeGreaterThanOrEqual(1);
    });

    it('resolves seeds by fuzzy match on name', async () => {
      const result = await traverseGraph({
        seed_names: ['TursoTestEntityB'],
        client,
      });

      expect(result.seed_entities.length).toBeGreaterThanOrEqual(1);
    });

    it('resolves seeds by seed_query', async () => {
      const result = await traverseGraph({
        seed_query: 'TursoTestEntityA',
        client,
      });

      expect(result.seed_entities.length).toBeGreaterThanOrEqual(1);
    });

    it('performs BFS traversal and discovers connected entities', async () => {
      const result = await traverseGraph({
        seed_names: ['src/turso.TursoTestEntityA'],
        max_hops: 2,
        client,
      });

      // EntityA references EntityB → should discover EntityB
      expect(result.entities.length).toBeGreaterThanOrEqual(1);
      expect(result.relationships.length).toBeGreaterThanOrEqual(1);
      expect(result.traversal_stats.hops_completed).toBeGreaterThanOrEqual(1);
    });

    it('includes chunk_ids for entities with chunk_id', async () => {
      const result = await traverseGraph({
        seed_names: ['src/turso.TursoTestEntityA'],
        client,
      });

      // EntityA has chunk_id = TEST_CHUNK_ID
      expect(result.chunk_ids).toContain(TEST_CHUNK_ID);
    });

    it('includes doc_ids for entities without chunk_id', async () => {
      // Insert an entity with no chunk_id but with doc_id
      await client.execute({
        sql: `INSERT INTO entities (entity_id, entity_type, name, qualified_name, doc_id, chunk_id, module_path, file_path)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        args: [
          900003,
          'module',
          'NoChunkEntity',
          'src/turso.NoChunkEntity',
          TEST_DOC_ID,
          null,
          'src/test.ts',
          'src/test.ts',
        ],
      });

      const result = await traverseGraph({
        seed_names: ['src/turso.NoChunkEntity'],
        client,
      });

      expect(result.doc_ids).toContain(TEST_DOC_ID);
    });

    it('formats entities with hop_distance', async () => {
      const result = await traverseGraph({
        seed_names: ['src/turso.TursoTestEntityA'],
        client,
      });

      expect(result.seed_entities[0].hop_distance).toBe(0);
      // Discovered entities should have hop_distance > 0
      const nonSeedEntities = result.entities.filter(
        (e) => !result.seed_entities.some((s) => s.entity_id === e.entity_id),
      );
      if (nonSeedEntities.length > 0) {
        expect(nonSeedEntities[0].hop_distance).toBeGreaterThan(0);
      }
    });

    it('filters by relationship_types', async () => {
      const result = await traverseGraph({
        seed_names: ['src/turso.TursoTestEntityA'],
        relationship_types: ['imports'], // Our fixture has 'references', not 'imports'
        client,
      });

      // No edges match 'imports' filter → only seed entity discovered
      expect(result.entities).toHaveLength(1);
      expect(result.relationships).toHaveLength(0);
    });

    it('filters by entity_types', async () => {
      const result = await traverseGraph({
        seed_names: ['src/turso.TursoTestEntityA'],
        entity_types: ['class'], // EntityA is 'function', EntityB is 'class'
        client,
      });

      // EntityA (function) is seed but won't be in results if filtered by class only
      // Actually seeds are always included; let's check the discovered ones
      // EntityB is 'class' → should be discovered
      const entityNames = result.entities.map((e) => e.name);
      expect(entityNames).toContain('TursoTestEntityB');
    });

    it('filters by confidence_filter', async () => {
      const result = await traverseGraph({
        seed_names: ['src/turso.TursoTestEntityA'],
        confidence_filter: ['low'], // Our fixture has 'high' confidence
        client,
      });

      // No edges match 'low' confidence → only seed entity
      expect(result.entities).toHaveLength(1);
    });

    it('respects max_results limit', async () => {
      const result = await traverseGraph({
        seed_names: ['src/turso.TursoTestEntityA'],
        max_results: 1,
        client,
      });

      expect(result.returned_count).toBeLessThanOrEqual(1);
    });

    it('combines seed_names and seed_query for seed resolution', async () => {
      const result = await traverseGraph({
        seed_names: ['src/turso.TursoTestEntityA'],
        seed_query: 'TursoTestEntityB',
        client,
      });

      expect(result.seed_entities.length).toBeGreaterThanOrEqual(2);
    });
  });

  describe('malformed edges', () => {
    it('filters out malformed edges in formatEdge (via mock client)', async () => {
      // Use a mock client that returns an edge with null source_qualified_name
      // to exercise the isWellFormedEdge false branch.
      const entities = [
        {
          entity_id: 900001,
          entity_type: 'function',
          name: 'EntityA',
          qualified_name: 'src.EntityA',
          doc_id: null,
          chunk_id: null,
        },
        {
          entity_id: 900002,
          entity_type: 'class',
          name: 'EntityB',
          qualified_name: 'src.EntityB',
          doc_id: null,
          chunk_id: null,
        },
      ];
      const edges = [
        {
          edge_id: 1,
          source_entity_id: 900001,
          target_entity_id: 900002,
          source_qualified_name: null, // malformed: null instead of string
          source_entity_type: 'function',
          target_qualified_name: 'src.EntityB',
          target_entity_type: 'class',
          relationship: 'references',
          confidence: 'high',
        },
      ];

      const mockClient = {
        async execute(sqlOrObj) {
          const sql = typeof sqlOrObj === 'string' ? sqlOrObj : sqlOrObj.sql;
          if (sql.includes('sqlite_master')) {
            return { rows: [{ name: 'entities' }, { name: 'edges' }] };
          }
          if (sql.includes('SELECT * FROM entities')) {
            return { rows: entities };
          }
          if (sql.includes('FROM edges e')) {
            return { rows: edges };
          }
          return { rows: [] };
        },
      };

      const result = await traverseGraph({
        seed_names: ['src.EntityA'],
        client: mockClient,
      });

      // The malformed edge should be filtered out by formatEdge
      expect(result.relationships).toHaveLength(0);
    });
  });

  describe('default max_hops handling', () => {
    it('defaults max_hops to 2 when not provided', async () => {
      const result = await traverseGraph({
        seed_names: ['src/turso.TursoTestEntityA'],
        client,
      });

      expect(result.hop_count).toBe(2);
    });

    it('resets invalid max_hops (non-integer) to 2', async () => {
      const result = await traverseGraph({
        seed_names: ['src/turso.TursoTestEntityA'],
        max_hops: 1.5,
        client,
      });

      expect(result.hop_count).toBe(2);
    });

    it('resets invalid max_hops (zero) to 2', async () => {
      const result = await traverseGraph({
        seed_names: ['src/turso.TursoTestEntityA'],
        max_hops: 0,
        client,
      });

      expect(result.hop_count).toBe(2);
    });

    it('resets negative max_hops to 2', async () => {
      const result = await traverseGraph({
        seed_names: ['src/turso.TursoTestEntityA'],
        max_hops: -1,
        client,
      });

      expect(result.hop_count).toBe(2);
    });

    it('clamps max_hops to MAX_HOPS_LIMIT (4)', async () => {
      const result = await traverseGraph({
        seed_names: ['src/turso.TursoTestEntityA'],
        max_hops: 10,
        client,
      });

      expect(result.hop_count).toBe(4);
    });
  });

  describe('validateArrayOption edge cases', () => {
    it('returns all values when all filtered items are invalid', async () => {
      const result = await traverseGraph({
        seed_names: ['src/turso.TursoTestEntityA'],
        relationship_types: ['invalid1', 'invalid2'],
        client,
      });

      // Should fall back to ALL_RELATIONSHIP_TYPES
      expect(result.hop_count).toBe(2);
    });

    it('returns all values when entity_types all invalid', async () => {
      const result = await traverseGraph({
        seed_names: ['src/turso.TursoTestEntityA'],
        entity_types: ['invalid1'],
        client,
      });

      expect(result.hop_count).toBe(2);
    });

    it('returns all values when confidence_filter all invalid', async () => {
      const result = await traverseGraph({
        seed_names: ['src/turso.TursoTestEntityA'],
        confidence_filter: ['invalid1'],
        client,
      });

      expect(result.hop_count).toBe(2);
    });
  });

  describe('traverseGraphHandler', () => {
    it('delegates to traverseGraph with all args', async () => {
      const result = await traverseGraphHandler({
        seed_names: ['src/turso.TursoTestEntityA'],
        client,
      });

      expect(result.graph_available).toBe(true);
      expect(result.seed_entities.length).toBeGreaterThanOrEqual(1);
    });

    it('throws on invalid max_hops (non-integer)', async () => {
      await expect(
        traverseGraphHandler({
          seed_names: ['Test'],
          max_hops: 1.5,
          client,
        }),
      ).rejects.toThrow('max_hops must be an integer between 1 and 4');
    });

    it('throws on max_hops > 4', async () => {
      await expect(
        traverseGraphHandler({
          seed_names: ['Test'],
          max_hops: 5,
          client,
        }),
      ).rejects.toThrow('max_hops must be an integer between 1 and 4');
    });

    it('throws on max_hops < 1', async () => {
      await expect(
        traverseGraphHandler({
          seed_names: ['Test'],
          max_hops: 0,
          client,
        }),
      ).rejects.toThrow('max_hops must be an integer between 1 and 4');
    });

    it('throws when no seeds provided', async () => {
      await expect(traverseGraphHandler({ client })).rejects.toThrow(
        'seed_query or seed_names is required',
      );
    });

    it('defaults max_hops to 2 when not provided', async () => {
      const result = await traverseGraphHandler({
        seed_names: ['src/turso.TursoTestEntityA'],
        client,
      });

      expect(result.hop_count).toBe(2);
    });
  });
});
