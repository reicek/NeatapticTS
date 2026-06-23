/**
 * @module traverse-graph.harden.red.test
 * @description Red tests for hardening traverse_graph to the Step 10/21 design spec.
 *
 * The spec requires seed_query/seed_names union, max_hops clamped to 4,
 * max_results clamped to 100, traversal_stats, graph_state, and structured
 * error taxonomy.
 */

import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { createClient } from '@libsql/client';
import { readCorpusSchema, splitSqlStatements } from './turso-test-helpers.mjs';

async function setupDb() {
  const tempDir = await mkdtemp(
    path.join(tmpdir(), 'traverse-graph-harden-test-'),
  );
  const dbPath = path.join(tempDir, 'test.sqlite');
  const client = createClient({ url: 'file:' + dbPath });
  const schemaSql = await readCorpusSchema();
  for (const stmt of splitSqlStatements(schemaSql)) {
    await client.execute(stmt);
  }
  await client.execute(`
    INSERT INTO documents (file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
    VALUES ('src/graph.ts', 'ts-source', 0, 100, 'a', 1);
  `);
  const docResult = await client.execute('SELECT doc_id FROM documents');
  const docId = docResult.rows[0].doc_id;
  await client.execute({
    sql: `INSERT INTO entities (entity_id, entity_type, name, qualified_name, doc_id, file_path)
    VALUES
      (?, 'function', 'foo', 'src.graph.foo', ?, 'src/graph.ts'),
      (?, 'function', 'bar', 'src.graph.bar', ?, 'src/graph.ts'),
      (?, 'function', 'baz', 'src.graph.baz', ?, 'src/graph.ts'),
      (?, 'function', 'qux', 'src.graph.qux', ?, 'src/graph.ts')`,
    args: [1, docId, 2, docId, 3, docId, 4, docId],
  });
  await client.execute(`
    INSERT INTO edges (source_entity_id, target_entity_id, relationship, confidence)
    VALUES
      (1, 2, 'depends-on', 'high'),
      (2, 3, 'depends-on', 'medium'),
      (3, 4, 'depends-on', 'low');
  `);
  return { client, dbPath, tempDir };
}

async function teardown(client, tempDir) {
  await client.close();
  await rm(tempDir, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });
}

describe('traverse-graph hardened', () => {
  describe('schema validation', () => {
    it('rejects missing seed_query and seed_names with SEED_REQUIRED', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        await expect(traverseGraphHandler({ client })).rejects.toThrow(
          /SEED_REQUIRED/,
        );
      } finally {
        await teardown(client, tempDir);
      }
    });

    it('rejects max_hops greater than 4 with INVALID_MAX_HOPS', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        await expect(
          traverseGraphHandler({
            client,
            seed_names: ['src.graph.foo'],
            max_hops: 5,
          }),
        ).rejects.toThrow(/INVALID_MAX_HOPS/);
      } finally {
        await teardown(client, tempDir);
      }
    });

    it('rejects max_hops of zero with INVALID_MAX_HOPS', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        await expect(
          traverseGraphHandler({
            client,
            seed_names: ['src.graph.foo'],
            max_hops: 0,
          }),
        ).rejects.toThrow(/INVALID_MAX_HOPS/);
      } finally {
        await teardown(client, tempDir);
      }
    });
  });

  describe('result shape', () => {
    it('returns graph_state and traversal_stats for a populated graph', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const result = await traverseGraphHandler({
          client,
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
        await teardown(client, tempDir);
      }
    });

    it('returns seed_entities for named seeds', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const result = await traverseGraphHandler({
          client,
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
        await teardown(client, tempDir);
      }
    });

    it('returns edges between discovered entities', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const result = await traverseGraphHandler({
          client,
          seed_names: ['src.graph.foo'],
          max_hops: 2,
        });

        expect(result.relationships?.length).toBeGreaterThan(0);
      } finally {
        await teardown(client, tempDir);
      }
    });
  });

  describe('BFS traversal', () => {
    it('discovers direct neighbors at hop 1', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const result = await traverseGraphHandler({
          client,
          seed_names: ['src.graph.foo'],
          max_hops: 1,
        });

        const hop1 = result.entities.filter((e) => e.hop_distance === 1);
        expect(hop1.map((e) => e.qualified_name).sort()).toEqual([
          'src.graph.bar',
        ]);
      } finally {
        await teardown(client, tempDir);
      }
    });

    it('discovers neighbors of neighbors at hop 2', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const result = await traverseGraphHandler({
          client,
          seed_names: ['src.graph.foo'],
          max_hops: 2,
        });

        const hop2 = result.entities.filter((e) => e.hop_distance === 2);
        expect(hop2.map((e) => e.qualified_name).sort()).toEqual([
          'src.graph.baz',
        ]);
      } finally {
        await teardown(client, tempDir);
      }
    });
  });

  describe('filters', () => {
    it('respects max_results', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const result = await traverseGraphHandler({
          client,
          seed_names: ['src.graph.foo'],
          max_hops: 3,
          max_results: 2,
        });

        expect(result.entities.length).toBeLessThanOrEqual(2);
      } finally {
        await teardown(client, tempDir);
      }
    });

    it('respects confidence_filter', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const result = await traverseGraphHandler({
          client,
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
        await teardown(client, tempDir);
      }
    });

    it('unions seed_query and seed_names seeds', async () => {
      const { traverseGraphHandler } =
        await import('../tools/traverse-graph.mjs');
      const { client, dbPath, tempDir } = await setupDb();
      try {
        const result = await traverseGraphHandler({
          client,
          seed_names: ['src.graph.foo'],
          seed_query: 'qux',
          max_hops: 1,
        });

        const seedIds = result.seed_entities.map((e) => e.entity_id).sort();
        expect(seedIds).toContain(1);
        expect(seedIds).toContain(4);
      } finally {
        await teardown(client, tempDir);
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
      const emptyClient = createClient({ url: 'file:' + dbPath });
      await emptyClient.execute('CREATE TABLE dummy (id INTEGER PRIMARY KEY)');

      try {
        const result = await traverseGraphHandler({
          client: emptyClient,
          seed_names: ['src.graph.foo'],
        });

        expect(result.graph_state).toBe('not_built');
      } finally {
        await teardown(emptyClient, tempDir);
      }
    });
  });
});
