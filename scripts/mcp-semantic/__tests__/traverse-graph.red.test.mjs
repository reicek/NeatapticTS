/**
 * @module traverse-graph.red.test
 * @description Red tests for BFS multi-hop traversal of the entity/relationship graph.
 */
import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createClient } from '@libsql/client';
import { closeTursoClient } from '../tools/cortex-db.mjs';

function createTestDb(databasePath) {
  const client = createClient({ url: 'file:' + databasePath });
  return {
    client,
    async exec(sql) {
      return client.execute(sql);
    },
    prepare(sql) {
      return {
        async run(...args) {
          return client.execute({ sql, args });
        },
        async get(...args) {
          return (await client.execute({ sql, args })).rows[0];
        },
        async all(...args) {
          return (await client.execute({ sql, args })).rows;
        },
      };
    },
    async close() {
      return client.close();
    },
  };
}

describe('traverse-graph', () => {
  describe('traverseGraph', () => {
    it('returns graph_available: false when entities/edges tables do not exist', async () => {
      const { traverseGraph } = await import('../tools/traverse-graph.mjs');

      const fixtureDirectory = await mkdtemp(
        path.join(tmpdir(), 'traverse-graph-test-'),
      );
      const databasePath = path.join(fixtureDirectory, 'empty.sqlite');

      try {
        // Create an empty database without entities/edges tables.
        const client = createTestDb(databasePath);
        await client.exec('CREATE TABLE dummy (id INTEGER PRIMARY KEY)');
        await client.close();

        const result = await traverseGraph({
          databasePath,
          seed_query: 'Network',
        });

        expect(result.graph_available).toBe(false);
        expect(result.entities).toEqual([]);
        expect(result.relationships).toEqual([]);
        expect(result.chunk_ids).toEqual([]);
        expect(result.doc_ids).toEqual([]);
      } finally {
        await closeTursoClient(databasePath);
        await rm(fixtureDirectory, {
          recursive: true,
          force: true,
          maxRetries: 10,
          retryDelay: 200,
        });
      }
    });

    it('returns empty results when no seed entities are found', async () => {
      const { traverseGraph } = await import('../tools/traverse-graph.mjs');

      const fixtureDirectory = await mkdtemp(
        path.join(tmpdir(), 'traverse-graph-test-'),
      );
      const databasePath = path.join(fixtureDirectory, 'test.sqlite');

      try {
        const client = createTestDb(databasePath);
        await client.exec(`
          CREATE TABLE entities (
            entity_id INTEGER PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            qualified_name TEXT NOT NULL UNIQUE,
            doc_id INTEGER,
            chunk_id INTEGER,
            module_path TEXT,
            signature_text TEXT,
            file_path TEXT NOT NULL,
            char_start INTEGER,
            char_end INTEGER,
            extra_metadata TEXT DEFAULT '{}',
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
          );
          CREATE TABLE edges (
            edge_id INTEGER PRIMARY KEY,
            source_entity_id INTEGER NOT NULL,
            target_entity_id INTEGER NOT NULL,
            relationship TEXT NOT NULL,
            confidence TEXT NOT NULL DEFAULT 'high',
            extra_metadata TEXT DEFAULT '{}',
            created_at INTEGER NOT NULL DEFAULT (unixepoch()),
            UNIQUE(source_entity_id, target_entity_id, relationship)
          );
        `);
        await client.close();

        const result = await traverseGraph({
          databasePath,
          seed_query: 'NonexistentEntity',
        });

        expect(result.graph_available).toBe(true);
        expect(result.seed_entities).toEqual([]);
        expect(result.entities).toEqual([]);
        expect(result.total_discovered).toBe(0);
      } finally {
        await closeTursoClient(databasePath);
        await rm(fixtureDirectory, {
          recursive: true,
          force: true,
          maxRetries: 10,
          retryDelay: 200,
        });
      }
    });

    it('resolves seed entities by exact qualified_name', async () => {
      const { traverseGraph } = await import('../tools/traverse-graph.mjs');

      const fixtureDirectory = await mkdtemp(
        path.join(tmpdir(), 'traverse-graph-test-'),
      );
      const databasePath = path.join(fixtureDirectory, 'test.sqlite');

      try {
        const client = createTestDb(databasePath);
        await client.exec(`
          CREATE TABLE entities (
            entity_id INTEGER PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            qualified_name TEXT NOT NULL UNIQUE,
            doc_id INTEGER,
            chunk_id INTEGER,
            module_path TEXT,
            signature_text TEXT,
            file_path TEXT NOT NULL,
            char_start INTEGER,
            char_end INTEGER,
            extra_metadata TEXT DEFAULT '{}',
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
          );
          CREATE TABLE edges (
            edge_id INTEGER PRIMARY KEY,
            source_entity_id INTEGER NOT NULL,
            target_entity_id INTEGER NOT NULL,
            relationship TEXT NOT NULL,
            confidence TEXT NOT NULL DEFAULT 'high',
            extra_metadata TEXT DEFAULT '{}',
            created_at INTEGER NOT NULL DEFAULT (unixepoch()),
            UNIQUE(source_entity_id, target_entity_id, relationship)
          );
        `);

        await client
          .prepare(
            'INSERT INTO entities (entity_type, name, qualified_name, file_path) VALUES (?, ?, ?, ?)',
          )
          .run(
            'class',
            'Network',
            'src/architecture/network.Network',
            'src/architecture/network/network.ts',
          );

        await client.close();

        const result = await traverseGraph({
          databasePath,
          seed_names: ['src/architecture/network.Network'],
          max_hops: 0,
        });

        expect(result.graph_available).toBe(true);
        expect(result.seed_entities.length).toBeGreaterThan(0);
        expect(result.seed_entities[0].qualified_name).toBe(
          'src/architecture/network.Network',
        );
      } finally {
        await closeTursoClient(databasePath);
        await rm(fixtureDirectory, {
          recursive: true,
          force: true,
          maxRetries: 10,
          retryDelay: 200,
        });
      }
    });

    it('performs BFS traversal following outgoing edges', async () => {
      const { traverseGraph } = await import('../tools/traverse-graph.mjs');

      const fixtureDirectory = await mkdtemp(
        path.join(tmpdir(), 'traverse-graph-test-'),
      );
      const databasePath = path.join(fixtureDirectory, 'test.sqlite');

      try {
        const client = createTestDb(databasePath);
        await client.exec(`
          CREATE TABLE entities (
            entity_id INTEGER PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            qualified_name TEXT NOT NULL UNIQUE,
            doc_id INTEGER,
            chunk_id INTEGER,
            module_path TEXT,
            signature_text TEXT,
            file_path NOT NULL,
            char_start INTEGER,
            char_end INTEGER,
            extra_metadata TEXT DEFAULT '{}',
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
          );
          CREATE TABLE edges (
            edge_id INTEGER PRIMARY KEY,
            source_entity_id INTEGER NOT NULL,
            target_entity_id INTEGER NOT NULL,
            relationship TEXT NOT NULL,
            confidence TEXT NOT NULL DEFAULT 'high',
            extra_metadata TEXT DEFAULT '{}',
            created_at INTEGER NOT NULL DEFAULT (unixepoch()),
            UNIQUE(source_entity_id, target_entity_id, relationship)
          );
        `);

        // Insert seed entity.
        await client
          .prepare(
            'INSERT INTO entities (entity_type, name, qualified_name, file_path) VALUES (?, ?, ?, ?)',
          )
          .run(
            'class',
            'Network',
            'src/architecture/network.Network',
            'src/architecture/network/network.ts',
          );

        // Insert target entity reachable via outgoing edge.
        await client
          .prepare(
            'INSERT INTO entities (entity_type, name, qualified_name, file_path) VALUES (?, ?, ?, ?)',
          )
          .run(
            'function',
            'evolve',
            'src/architecture/network.Network.evolve',
            'src/architecture/network/network.ts',
          );

        // Insert outgoing edge.
        await client
          .prepare(
            'INSERT INTO edges (source_entity_id, target_entity_id, relationship, confidence) VALUES (?, ?, ?, ?)',
          )
          .run(1, 2, 'owns', 'high');

        await client.close();

        const result = await traverseGraph({
          databasePath,
          seed_names: ['src/architecture/network.Network'],
          max_hops: 2,
          relationship_types: ['owns'],
        });

        expect(result.graph_available).toBe(true);
        expect(result.total_discovered).toBe(2);
        const entityNames = result.entities.map((e) => e.qualified_name);
        expect(entityNames).toContain(
          'src/architecture/network.Network.evolve',
        );
      } finally {
        await closeTursoClient(databasePath);
        await rm(fixtureDirectory, {
          recursive: true,
          force: true,
          maxRetries: 10,
          retryDelay: 200,
        });
      }
    });

    it('performs BFS traversal following incoming edges (reverse traversal)', async () => {
      const { traverseGraph } = await import('../tools/traverse-graph.mjs');

      const fixtureDirectory = await mkdtemp(
        path.join(tmpdir(), 'traverse-graph-test-'),
      );
      const databasePath = path.join(fixtureDirectory, 'test.sqlite');

      try {
        const client = createTestDb(databasePath);
        await client.exec(`
          CREATE TABLE entities (
            entity_id INTEGER PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            qualified_name TEXT NOT NULL UNIQUE,
            doc_id INTEGER,
            chunk_id INTEGER,
            module_path TEXT,
            signature_text TEXT,
            file_path NOT NULL,
            char_start INTEGER,
            char_end INTEGER,
            extra_metadata TEXT DEFAULT '{}',
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
          );
          CREATE TABLE edges (
            edge_id INTEGER PRIMARY KEY,
            source_entity_id INTEGER NOT NULL,
            target_entity_id INTEGER NOT NULL,
            relationship TEXT NOT NULL,
            confidence TEXT NOT NULL DEFAULT 'high',
            extra_metadata TEXT DEFAULT '{}',
            created_at INTEGER NOT NULL DEFAULT (unixepoch()),
            UNIQUE(source_entity_id, target_entity_id, relationship)
          );
        `);

        // Insert source entity that owns the seed.
        await client
          .prepare(
            'INSERT INTO entities (entity_type, name, qualified_name, file_path) VALUES (?, ?, ?, ?)',
          )
          .run(
            'module',
            'network',
            'src/architecture/network',
            'src/architecture/network/network.ts',
          );

        // Insert seed entity.
        await client
          .prepare(
            'INSERT INTO entities (entity_type, name, qualified_name, file_path) VALUES (?, ?, ?, ?)',
          )
          .run(
            'class',
            'Network',
            'src/architecture/network.Network',
            'src/architecture/network/network.ts',
          );

        // Module owns the class (incoming edge to seed).
        await client
          .prepare(
            'INSERT INTO edges (source_entity_id, target_entity_id, relationship, confidence) VALUES (?, ?, ?, ?)',
          )
          .run(1, 2, 'owns', 'high');

        await client.close();

        const result = await traverseGraph({
          databasePath,
          seed_names: ['src/architecture/network.Network'],
          max_hops: 2,
          relationship_types: ['owns'],
        });

        expect(result.graph_available).toBe(true);
        // Should discover the module via incoming "owns" edge.
        const entityNames = result.entities.map((e) => e.qualified_name);
        expect(entityNames).toContain('src/architecture/network');
      } finally {
        await closeTursoClient(databasePath);
        await rm(fixtureDirectory, {
          recursive: true,
          force: true,
          maxRetries: 10,
          retryDelay: 200,
        });
      }
    });

    it('filters by relationship types', async () => {
      const { traverseGraph } = await import('../tools/traverse-graph.mjs');

      const fixtureDirectory = await mkdtemp(
        path.join(tmpdir(), 'traverse-graph-test-'),
      );
      const databasePath = path.join(fixtureDirectory, 'test.sqlite');

      try {
        const client = createTestDb(databasePath);
        await client.exec(`
          CREATE TABLE entities (
            entity_id INTEGER PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            qualified_name TEXT NOT NULL UNIQUE,
            doc_id INTEGER,
            chunk_id INTEGER,
            module_path TEXT,
            signature_text TEXT,
            file_path NOT NULL,
            char_start INTEGER,
            char_end INTEGER,
            extra_metadata TEXT DEFAULT '{}',
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
          );
          CREATE TABLE edges (
            edge_id INTEGER PRIMARY KEY,
            source_entity_id INTEGER NOT NULL,
            target_entity_id INTEGER NOT NULL,
            relationship TEXT NOT NULL,
            confidence TEXT NOT NULL DEFAULT 'high',
            extra_metadata TEXT DEFAULT '{}',
            created_at INTEGER NOT NULL DEFAULT (unixepoch()),
            UNIQUE(source_entity_id, target_entity_id, relationship)
          );
        `);

        await client
          .prepare(
            'INSERT INTO entities (entity_type, name, qualified_name, file_path) VALUES (?, ?, ?, ?)',
          )
          .run(
            'module',
            'network',
            'src/architecture/network',
            'src/architecture/network/network.ts',
          );

        await client
          .prepare(
            'INSERT INTO entities (entity_type, name, qualified_name, file_path) VALUES (?, ?, ?, ?)',
          )
          .run(
            'class',
            'Network',
            'src/architecture/network.Network',
            'src/architecture/network/network.ts',
          );

        await client
          .prepare(
            'INSERT INTO entities (entity_type, name, qualified_name, file_path) VALUES (?, ?, ?, ?)',
          )
          .run(
            'function',
            'evolve',
            'src/architecture/network.Network.evolve',
            'src/architecture/network/network.ts',
          );

        // owns edge: module → class
        await client
          .prepare(
            'INSERT INTO edges (source_entity_id, target_entity_id, relationship, confidence) VALUES (?, ?, ?, ?)',
          )
          .run(1, 2, 'owns', 'high');

        // imports edge: module → some other module (should be filtered out)
        await client
          .prepare(
            'INSERT INTO entities (entity_type, name, qualified_name, file_path) VALUES (?, ?, ?, ?)',
          )
          .run('module', 'methods', 'src/methods', 'src/methods/index.ts');

        await client
          .prepare(
            'INSERT INTO edges (source_entity_id, target_entity_id, relationship, confidence) VALUES (?, ?, ?, ?)',
          )
          .run(1, 4, 'imports', 'high');

        await client.close();

        // Only follow "owns" relationship — imports should be ignored.
        const result = await traverseGraph({
          databasePath,
          seed_names: ['src/architecture/network'],
          max_hops: 2,
          relationship_types: ['owns'],
        });

        const entityNames = result.entities.map((e) => e.qualified_name);
        expect(entityNames).toContain('src/architecture/network.Network');
        // The methods module should NOT be discovered since we only follow "owns".
        expect(entityNames).not.toContain('src/methods');
      } finally {
        await closeTursoClient(databasePath);
        await rm(fixtureDirectory, {
          recursive: true,
          force: true,
          maxRetries: 10,
          retryDelay: 200,
        });
      }
    });

    it('respects max_hops limit', async () => {
      const { traverseGraph } = await import('../tools/traverse-graph.mjs');

      const fixtureDirectory = await mkdtemp(
        path.join(tmpdir(), 'traverse-graph-test-'),
      );
      const databasePath = path.join(fixtureDirectory, 'test.sqlite');

      try {
        const client = createTestDb(databasePath);
        await client.exec(`
          CREATE TABLE entities (
            entity_id INTEGER PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            qualified_name TEXT NOT NULL UNIQUE,
            doc_id INTEGER,
            chunk_id INTEGER,
            module_path TEXT,
            signature_text TEXT,
            file_path NOT NULL,
            char_start INTEGER,
            char_end INTEGER,
            extra_metadata TEXT DEFAULT '{}',
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
          );
          CREATE TABLE edges (
            edge_id INTEGER PRIMARY KEY,
            source_entity_id INTEGER NOT NULL,
            target_entity_id INTEGER NOT NULL,
            relationship TEXT NOT NULL,
            confidence TEXT NOT NULL DEFAULT 'high',
            extra_metadata TEXT DEFAULT '{}',
            created_at INTEGER NOT NULL DEFAULT (unixepoch()),
            UNIQUE(source_entity_id, target_entity_id, relationship)
          );
        `);

        // Create a chain: A → B → C → D
        const names = ['entity.A', 'entity.B', 'entity.C', 'entity.D'];
        for (let i = 0; i < names.length; i++) {
          await client
            .prepare(
              'INSERT INTO entities (entity_type, name, qualified_name, file_path) VALUES (?, ?, ?, ?)',
            )
            .run('class', names[i].split('.')[1], names[i], 'test.ts');
        }
        for (let i = 0; i < names.length - 1; i++) {
          await client
            .prepare(
              'INSERT INTO edges (source_entity_id, target_entity_id, relationship, confidence) VALUES (?, ?, ?, ?)',
            )
            .run(i + 1, i + 2, 'depends-on', 'high');
        }

        await client.close();

        // With max_hops=1, should only discover B (1 hop from A).
        const result = await traverseGraph({
          databasePath,
          seed_names: ['entity.A'],
          max_hops: 1,
          relationship_types: ['depends-on'],
        });

        const entityNames = result.entities.map((e) => e.qualified_name);
        expect(entityNames).toContain('entity.B');
        expect(entityNames).not.toContain('entity.C');
        expect(entityNames).not.toContain('entity.D');
      } finally {
        await closeTursoClient(databasePath);
        await rm(fixtureDirectory, {
          recursive: true,
          force: true,
          maxRetries: 10,
          retryDelay: 200,
        });
      }
    });

    it('respects max_results limit', async () => {
      const { traverseGraph } = await import('../tools/traverse-graph.mjs');

      const fixtureDirectory = await mkdtemp(
        path.join(tmpdir(), 'traverse-graph-test-'),
      );
      const databasePath = path.join(fixtureDirectory, 'test.sqlite');

      try {
        const client = createTestDb(databasePath);
        await client.exec(`
          CREATE TABLE entities (
            entity_id INTEGER PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            qualified_name TEXT NOT NULL UNIQUE,
            doc_id INTEGER,
            chunk_id INTEGER,
            module_path TEXT,
            signature_text TEXT,
            file_path NOT NULL,
            char_start INTEGER,
            char_end INTEGER,
            extra_metadata TEXT DEFAULT '{}',
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
          );
          CREATE TABLE edges (
            edge_id INTEGER PRIMARY KEY,
            source_entity_id INTEGER NOT NULL,
            target_entity_id INTEGER NOT NULL,
            relationship TEXT NOT NULL,
            confidence TEXT NOT NULL DEFAULT 'high',
            extra_metadata TEXT DEFAULT '{}',
            created_at INTEGER NOT NULL DEFAULT (unixepoch()),
            UNIQUE(source_entity_id, target_entity_id, relationship)
          );
        `);

        // Seed entity.
        await client
          .prepare(
            'INSERT INTO entities (entity_type, name, qualified_name, file_path) VALUES (?, ?, ?, ?)',
          )
          .run('module', 'network', 'src.architecture.network', 'test.ts');

        // Create 10 target entities.
        for (let i = 1; i <= 10; i++) {
          await client
            .prepare(
              'INSERT INTO entities (entity_type, name, qualified_name, file_path) VALUES (?, ?, ?, ?)',
            )
            .run(
              'class',
              `Class${i}`,
              `src.architecture.network.Class${i}`,
              'test.ts',
            );

          await client
            .prepare(
              'INSERT INTO edges (source_entity_id, target_entity_id, relationship, confidence) VALUES (?, ?, ?, ?)',
            )
            .run(1, i + 1, 'owns', 'high');
        }

        await client.close();

        const result = await traverseGraph({
          databasePath,
          seed_names: ['src.architecture.network'],
          max_hops: 2,
          max_results: 5,
        });

        // Should return at most 5 entities (plus seed).
        expect(result.returned_count).toBeLessThanOrEqual(5);
      } finally {
        await closeTursoClient(databasePath);
        await rm(fixtureDirectory, {
          recursive: true,
          force: true,
          maxRetries: 10,
          retryDelay: 200,
        });
      }
    });

    it('filters by confidence level', async () => {
      const { traverseGraph } = await import('../tools/traverse-graph.mjs');

      const fixtureDirectory = await mkdtemp(
        path.join(tmpdir(), 'traverse-graph-test-'),
      );
      const databasePath = path.join(fixtureDirectory, 'test.sqlite');

      try {
        const client = createTestDb(databasePath);
        await client.exec(`
          CREATE TABLE entities (
            entity_id INTEGER PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            qualified_name TEXT NOT NULL UNIQUE,
            doc_id INTEGER,
            chunk_id INTEGER,
            module_path TEXT,
            signature_text TEXT,
            file_path NOT NULL,
            char_start INTEGER,
            char_end INTEGER,
            extra_metadata TEXT DEFAULT '{}',
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
          );
          CREATE TABLE edges (
            edge_id INTEGER PRIMARY KEY,
            source_entity_id INTEGER NOT NULL,
            target_entity_id INTEGER NOT NULL,
            relationship TEXT NOT NULL,
            confidence TEXT NOT NULL DEFAULT 'high',
            extra_metadata TEXT DEFAULT '{}',
            created_at INTEGER NOT NULL DEFAULT (unixepoch()),
            UNIQUE(source_entity_id, target_entity_id, relationship)
          );
        `);

        // Seed.
        await client
          .prepare(
            'INSERT INTO entities (entity_type, name, qualified_name, file_path) VALUES (?, ?, ?, ?)',
          )
          .run('module', 'network', 'src.architecture.network', 'test.ts');

        // High-confidence target.
        await client
          .prepare(
            'INSERT INTO entities (entity_type, name, qualified_name, file_path) VALUES (?, ?, ?, ?)',
          )
          .run(
            'class',
            'HighConfClass',
            'src.architecture.network.HighConfClass',
            'test.ts',
          );
        await client
          .prepare(
            'INSERT INTO edges (source_entity_id, target_entity_id, relationship, confidence) VALUES (?, ?, ?, ?)',
          )
          .run(1, 2, 'owns', 'high');

        // Low-confidence target.
        await client
          .prepare(
            'INSERT INTO entities (entity_type, name, qualified_name, file_path) VALUES (?, ?, ?, ?)',
          )
          .run(
            'class',
            'LowConfClass',
            'src.architecture.network.LowConfClass',
            'test.ts',
          );
        await client
          .prepare(
            'INSERT INTO edges (source_entity_id, target_entity_id, relationship, confidence) VALUES (?, ?, ?, ?)',
          )
          .run(1, 3, 'references', 'low');

        await client.close();

        // Filter to only high+medium confidence.
        const result = await traverseGraph({
          databasePath,
          seed_names: ['src.architecture.network'],
          max_hops: 2,
          confidence_filter: ['high', 'medium'],
        });

        const entityNames = result.entities.map((e) => e.qualified_name);
        expect(entityNames).toContain('src.architecture.network.HighConfClass');
        // Low-confidence edge should not be followed.
        expect(entityNames).not.toContain(
          'src.architecture.network.LowConfClass',
        );
      } finally {
        await closeTursoClient(databasePath);
        await rm(fixtureDirectory, {
          recursive: true,
          force: true,
          maxRetries: 10,
          retryDelay: 200,
        });
      }
    });

    it('prevents cycles in BFS traversal', async () => {
      const { traverseGraph } = await import('../tools/traverse-graph.mjs');

      const fixtureDirectory = await mkdtemp(
        path.join(tmpdir(), 'traverse-graph-test-'),
      );
      const databasePath = path.join(fixtureDirectory, 'test.sqlite');

      try {
        const client = createTestDb(databasePath);
        await client.exec(`
          CREATE TABLE entities (
            entity_id INTEGER PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            qualified_name TEXT NOT NULL UNIQUE,
            doc_id INTEGER,
            chunk_id INTEGER,
            module_path TEXT,
            signature_text TEXT,
            file_path NOT NULL,
            char_start INTEGER,
            char_end INTEGER,
            extra_metadata TEXT DEFAULT '{}',
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
          );
          CREATE TABLE edges (
            edge_id INTEGER PRIMARY KEY,
            source_entity_id INTEGER NOT NULL,
            target_entity_id INTEGER NOT NULL,
            relationship TEXT NOT NULL,
            confidence TEXT NOT NULL DEFAULT 'high',
            extra_metadata TEXT DEFAULT '{}',
            created_at INTEGER NOT NULL DEFAULT (unixepoch()),
            UNIQUE(source_entity_id, target_entity_id, relationship)
          );
        `);

        // Create cycle: A → B → A
        await client
          .prepare(
            'INSERT INTO entities (entity_type, name, qualified_name, file_path) VALUES (?, ?, ?, ?)',
          )
          .run('class', 'A', 'entity.A', 'test.ts');
        await client
          .prepare(
            'INSERT INTO entities (entity_type, name, qualified_name, file_path) VALUES (?, ?, ?, ?)',
          )
          .run('class', 'B', 'entity.B', 'test.ts');

        await client
          .prepare(
            'INSERT INTO edges (source_entity_id, target_entity_id, relationship, confidence) VALUES (?, ?, ?, ?)',
          )
          .run(1, 2, 'depends-on', 'high');
        await client
          .prepare(
            'INSERT INTO edges (source_entity_id, target_entity_id, relationship, confidence) VALUES (?, ?, ?, ?)',
          )
          .run(2, 1, 'depends-on', 'high');

        await client.close();

        const result = await traverseGraph({
          databasePath,
          seed_names: ['entity.A'],
          max_hops: 3,
        });

        // Should not loop infinitely; should discover exactly 2 entities.
        expect(result.total_discovered).toBe(2);
      } finally {
        await closeTursoClient(databasePath);
        await rm(fixtureDirectory, {
          recursive: true,
          force: true,
          maxRetries: 10,
          retryDelay: 200,
        });
      }
    });

    it('resolves seed entities by fuzzy name matching', async () => {
      const { traverseGraph } = await import('../tools/traverse-graph.mjs');

      const fixtureDirectory = await mkdtemp(
        path.join(tmpdir(), 'traverse-graph-test-'),
      );
      const databasePath = path.join(fixtureDirectory, 'test.sqlite');

      try {
        const client = createTestDb(databasePath);
        await client.exec(`
          CREATE TABLE entities (
            entity_id INTEGER PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            qualified_name TEXT NOT NULL UNIQUE,
            doc_id INTEGER,
            chunk_id INTEGER,
            module_path TEXT,
            signature_text TEXT,
            file_path NOT NULL,
            char_start INTEGER,
            char_end INTEGER,
            extra_metadata TEXT DEFAULT '{}',
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
          );
          CREATE TABLE edges (
            edge_id INTEGER PRIMARY KEY,
            source_entity_id INTEGER NOT NULL,
            target_entity_id INTEGER NOT NULL,
            relationship TEXT NOT NULL,
            confidence TEXT NOT NULL DEFAULT 'high',
            extra_metadata TEXT DEFAULT '{}',
            created_at INTEGER NOT NULL DEFAULT (unixepoch()),
            UNIQUE(source_entity_id, target_entity_id, relationship)
          );
        `);

        await client
          .prepare(
            'INSERT INTO entities (entity_type, name, qualified_name, file_path) VALUES (?, ?, ?, ?)',
          )
          .run(
            'class',
            'Network',
            'src/architecture/network.Network',
            'test.ts',
          );

        await client.close();

        // Use partial name for seed resolution.
        const result = await traverseGraph({
          databasePath,
          seed_names: ['Network'],
          max_hops: 0,
        });

        expect(result.seed_entities.length).toBeGreaterThan(0);
      } finally {
        await closeTursoClient(databasePath);
        await rm(fixtureDirectory, {
          recursive: true,
          force: true,
          maxRetries: 10,
          retryDelay: 200,
        });
      }
    });

    it('resolves seed entities by free-text query', async () => {
      const { traverseGraph } = await import('../tools/traverse-graph.mjs');

      const fixtureDirectory = await mkdtemp(
        path.join(tmpdir(), 'traverse-graph-test-'),
      );
      const databasePath = path.join(fixtureDirectory, 'test.sqlite');

      try {
        const client = createTestDb(databasePath);
        await client.exec(`
          CREATE TABLE entities (
            entity_id INTEGER PRIMARY KEY,
            entity_type TEXT NOT NULL,
            name TEXT NOT NULL,
            qualified_name TEXT NOT NULL UNIQUE,
            doc_id INTEGER,
            chunk_id INTEGER,
            module_path TEXT,
            signature_text TEXT,
            file_path NOT NULL,
            char_start INTEGER,
            char_end INTEGER,
            extra_metadata TEXT DEFAULT '{}',
            created_at INTEGER NOT NULL DEFAULT (unixepoch())
          );
          CREATE TABLE edges (
            edge_id INTEGER PRIMARY KEY,
            source_entity_id INTEGER NOT NULL,
            target_entity_id INTEGER NOT NULL,
            relationship TEXT NOT NULL,
            confidence TEXT NOT NULL DEFAULT 'high',
            extra_metadata TEXT DEFAULT '{}',
            created_at INTEGER NOT NULL DEFAULT (unixepoch()),
            UNIQUE(source_entity_id, target_entity_id, relationship)
          );
        `);

        await client
          .prepare(
            'INSERT INTO entities (entity_type, name, qualified_name, file_path) VALUES (?, ?, ?, ?)',
          )
          .run(
            'class',
            'Network',
            'src/architecture/network.Network',
            'test.ts',
          );

        await client.close();

        const result = await traverseGraph({
          databasePath,
          seed_query: 'Network architecture',
          max_hops: 0,
        });

        expect(result.seed_entities.length).toBeGreaterThan(0);
      } finally {
        await closeTursoClient(databasePath);
        await rm(fixtureDirectory, {
          recursive: true,
          force: true,
          maxRetries: 10,
          retryDelay: 200,
        });
      }
    });
  });

  describe('traverseGraphHandler', () => {
    it('maps MCP tool arguments to traverseGraph options', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');

      const fixtureDirectory = await mkdtemp(
        path.join(tmpdir(), 'traverse-graph-test-'),
      );
      const databasePath = path.join(fixtureDirectory, 'empty.sqlite');

      try {
        const client = createTestDb(databasePath);
        await client.exec('CREATE TABLE dummy (id INTEGER PRIMARY KEY)');
        await client.close();

        const result = await traverseGraphHandler({
          seed_query: 'test',
          seed_names: ['Network'],
          relationship_types: ['owns', 'imports'],
          entity_types: ['class', 'module'],
          max_hops: 2,
          max_results: 10,
          confidence_filter: ['high'],
          databasePath,
        });

        // Should not throw; returns gracefully even with empty graph.
        expect(result.graph_available).toBe(false);
      } finally {
        await closeTursoClient(databasePath);
        await rm(fixtureDirectory, {
          recursive: true,
          force: true,
          maxRetries: 10,
          retryDelay: 200,
        });
      }
    });
  });
});
