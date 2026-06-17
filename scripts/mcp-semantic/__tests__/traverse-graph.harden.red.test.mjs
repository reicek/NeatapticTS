/**
 * @module traverse-graph.harden.red.test
 * @description Red tests for hardening traverse_graph to the Step 10/21 design spec.
 *
 * The spec requires seed_query/seed_names union, max_hops clamped to 4,
 * max_results clamped to 100, traversal_stats, graph_state, and structured
 * error taxonomy.
 */

import { mkdtemp, rm, readFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import Database from 'better-sqlite3';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function readSchema() {
  const schemaPath = path.join(__dirname, '../../semantic-index/schema-v2.sql');
  return readFile(schemaPath, 'utf8');
}

async function setupDb() {
  const tempDir = await mkdtemp(
    path.join(tmpdir(), 'traverse-graph-harden-test-'),
  );
  const dbPath = path.join(tempDir, 'test.sqlite');
  const db = new Database(dbPath);
  db.exec(await readSchema());
  db.exec(`
    INSERT INTO documents (file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
    VALUES ('src/graph.ts', 'ts-source', 0, 100, 'a', 1);
  `);
  const docId = db.prepare('SELECT doc_id FROM documents').get().doc_id;
  db.exec(`
    INSERT INTO entities (entity_id, entity_type, name, qualified_name, doc_id, file_path)
    VALUES
      (1, 'function', 'foo', 'src.graph.foo', ${docId}, 'src/graph.ts'),
      (2, 'function', 'bar', 'src.graph.bar', ${docId}, 'src/graph.ts'),
      (3, 'function', 'baz', 'src.graph.baz', ${docId}, 'src/graph.ts'),
      (4, 'function', 'qux', 'src.graph.qux', ${docId}, 'src/graph.ts');
  `);
  db.exec(`
    INSERT INTO edges (source_entity_id, target_entity_id, relationship, confidence)
    VALUES
      (1, 2, 'depends-on', 'high'),
      (2, 3, 'depends-on', 'medium'),
      (3, 4, 'depends-on', 'low');
  `);
  db.close();
  return { dbPath, tempDir };
}

function teardown(tempDir) {
  return rm(tempDir, { recursive: true, force: true });
}

describe('traverse-graph hardened', () => {
  describe('schema validation', () => {
    it('rejects missing seed_query and seed_names with SEED_REQUIRED', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        await expect(
          traverseGraphHandler({ databasePath: dbPath }),
        ).rejects.toThrow(/SEED_REQUIRED/);
      } finally {
        await teardown(tempDir);
      }
    });

    it('rejects max_hops greater than 4 with INVALID_MAX_HOPS', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        await expect(
          traverseGraphHandler({
            databasePath: dbPath,
            seed_names: ['src.graph.foo'],
            max_hops: 5,
          }),
        ).rejects.toThrow(/INVALID_MAX_HOPS/);
      } finally {
        await teardown(tempDir);
      }
    });

    it('rejects max_hops of zero with INVALID_MAX_HOPS', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        await expect(
          traverseGraphHandler({
            databasePath: dbPath,
            seed_names: ['src.graph.foo'],
            max_hops: 0,
          }),
        ).rejects.toThrow(/INVALID_MAX_HOPS/);
      } finally {
        await teardown(tempDir);
      }
    });
  });

  describe('result shape', () => {
    it('returns graph_state and traversal_stats for a populated graph', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        const result = await traverseGraphHandler({
          databasePath: dbPath,
          seed_names: ['src.graph.foo'],
        });

        expect(result).toEqual(
          expect.objectContaining({
            graph_state: 'ready',
            traversal_stats: expect.objectContaining({
              total_entities_discovered: expect.any(Number),
              total_edges_traversed: expect.any(Number),
              hops_completed: expect.any(Number),
              query_time_ms: expect.any(Number),
            }),
          }),
        );
      } finally {
        await teardown(tempDir);
      }
    });

    it('returns seed_entities for named seeds', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        const result = await traverseGraphHandler({
          databasePath: dbPath,
          seed_names: ['src.graph.foo'],
        });

        expect(result.seed_entities).toEqual(
          expect.arrayContaining([
            expect.objectContaining({
              entity_id: 1,
              qualified_name: 'src.graph.foo',
            }),
          ]),
        );
      } finally {
        await teardown(tempDir);
      }
    });

    it('returns edges between discovered entities', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        const result = await traverseGraphHandler({
          databasePath: dbPath,
          seed_names: ['src.graph.foo'],
          max_hops: 2,
        });

        expect(result.relationships?.length).toBeGreaterThan(0);
      } finally {
        await teardown(tempDir);
      }
    });
  });

  describe('BFS traversal', () => {
    it('discovers direct neighbors at hop 1', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        const result = await traverseGraphHandler({
          databasePath: dbPath,
          seed_names: ['src.graph.foo'],
          max_hops: 1,
        });

        const hop1 = result.entities.filter((e) => e.hop_distance === 1);
        expect(hop1.map((e) => e.qualified_name).sort()).toEqual([
          'src.graph.bar',
        ]);
      } finally {
        await teardown(tempDir);
      }
    });

    it('discovers neighbors of neighbors at hop 2', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        const result = await traverseGraphHandler({
          databasePath: dbPath,
          seed_names: ['src.graph.foo'],
          max_hops: 2,
        });

        const hop2 = result.entities.filter((e) => e.hop_distance === 2);
        expect(hop2.map((e) => e.qualified_name).sort()).toEqual([
          'src.graph.baz',
        ]);
      } finally {
        await teardown(tempDir);
      }
    });
  });

  describe('filters', () => {
    it('respects max_results', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        const result = await traverseGraphHandler({
          databasePath: dbPath,
          seed_names: ['src.graph.foo'],
          max_hops: 3,
          max_results: 2,
        });

        expect(result.entities.length).toBeLessThanOrEqual(2);
      } finally {
        await teardown(tempDir);
      }
    });

    it('respects confidence_filter', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        const result = await traverseGraphHandler({
          databasePath: dbPath,
          seed_names: ['src.graph.foo'],
          max_hops: 3,
          confidence_filter: ['high', 'medium'],
        });

        expect(
          result.relationships.every((edge) =>
            ['high', 'medium'].includes(edge.confidence),
          ),
        ).toBe(true);
      } finally {
        await teardown(tempDir);
      }
    });

    it('unions seed_query and seed_names seeds', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        const result = await traverseGraphHandler({
          databasePath: dbPath,
          seed_names: ['src.graph.foo'],
          seed_query: 'qux',
          max_hops: 1,
        });

        const seedIds = result.seed_entities.map((e) => e.entity_id).sort();
        expect(seedIds).toContain(1);
        expect(seedIds).toContain(4);
      } finally {
        await teardown(tempDir);
      }
    });
  });

  describe('empty graph', () => {
    it('returns graph_state not_built when graph tables are absent', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');
      const tempDir = await mkdtemp(
        path.join(tmpdir(), 'traverse-graph-empty-test-'),
      );
      const dbPath = path.join(tempDir, 'empty.sqlite');
      const db = new Database(dbPath);
      db.exec('CREATE TABLE dummy (id INTEGER PRIMARY KEY)');
      db.close();

      try {
        const result = await traverseGraphHandler({
          databasePath: dbPath,
          seed_names: ['src.graph.foo'],
        });

        expect(result.graph_state).toBe('not_built');
      } finally {
        await teardown(tempDir);
      }
    });
  });
});
