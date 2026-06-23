/**
 * @module search-context.harden.red.test
 * @description Red tests for hardening search_context to the Step 10/21 design spec.
 *
 * The spec requires budget accounting, include_metadata, dedup_strategy,
 * rerank_state, and graceful budget-exceeded handling.
 */

import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { createClient } from '@libsql/client';
import { readCorpusSchema, splitSqlStatements } from './turso-test-helpers.mjs';

async function setupDb() {
  const tempDir = await mkdtemp(path.join(tmpdir(), 'search-context-test-'));
  const dbPath = path.join(tempDir, 'test.sqlite');
  const client = createClient({ url: pathToFileURL(dbPath).href });
  const schemaSql = await readCorpusSchema();
  for (const stmt of splitSqlStatements(schemaSql)) {
    await client.execute(stmt);
  }
  await client.execute(`
    INSERT INTO documents (file_path, doc_family, mtime_ms, file_size, sha256, indexed_at, arch_layer)
    VALUES ('src/foo.ts', 'ts-source', 0, 100, 'a', 1, 'network');
  `);
  const docResult = await client.execute('SELECT doc_id FROM documents');
  const docId = docResult.rows[0].doc_id;
  for (let i = 0; i < 5; i += 1) {
    await client.execute({
      sql: `
      INSERT INTO chunks (doc_id, chunk_index, body_text, char_start, char_end, depth, arch_layer, export_type, jsdoc_quality)
      VALUES (?, ?, ?, 0, 10, 0, 'network', 'function', 'good')
    `,
      args: [docId, i, `function foo${i}() { return ${i}; }`],
    });
  }
  return { client, dbPath, tempDir };
}

async function teardown(client, tempDir) {
  await client.close();
  await rm(tempDir, { recursive: true, force: true, maxRetries: 10, retryDelay: 200 });
}

describe('search-context hardened', () => {
  describe('schema validation', () => {
    it('rejects a missing query parameter', async () => {
      const { searchContext } = await import('../tools/search-context.mjs');
      const { client, tempDir } = await setupDb();
      try {
        await expect(searchContext({ client })).rejects.toThrow(/query/);
      } finally {
        await teardown(client, tempDir);
      }
    });
  });

  describe('budget accounting', () => {
    it('returns total_chunks_retrieved, chunks_in_context, tokens_used, budget_remaining', async () => {
      const { searchContext } = await import('../tools/search-context.mjs');
      const { client, tempDir } = await setupDb();
      try {
        const result = await searchContext({
          client,
          query: 'function foo',
          use_dense: false,
          budget: 1024,
        });

        expect(result).toEqual(
          expect.objectContaining({
            total_chunks_retrieved: expect.any(Number),
            chunks_in_context: expect.any(Number),
            tokens_used: expect.any(Number),
            budget_remaining: expect.any(Number),
          }),
        );
      } finally {
        await teardown(client, tempDir);
      }
    });

    it('reports dense_state and rerank_state', async () => {
      const { searchContext } = await import('../tools/search-context.mjs');
      const { client, tempDir } = await setupDb();
      try {
        const result = await searchContext({
          client,
          query: 'function foo',
          use_dense: false,
        });

        expect(result).toEqual(
          expect.objectContaining({
            dense_state: expect.any(String),
            rerank_state: expect.any(String),
          }),
        );
      } finally {
        await teardown(client, tempDir);
      }
    });

    it('sets budget_remaining to zero when budget is exceeded', async () => {
      const { searchContext } = await import('../tools/search-context.mjs');
      const { client, tempDir } = await setupDb();
      try {
        const result = await searchContext({
          client,
          query: 'function foo',
          use_dense: false,
          budget: 1,
        });

        expect(result.budget_remaining).toBe(0);
      } finally {
        await teardown(client, tempDir);
      }
    });
  });

  describe('include_metadata', () => {
    it('includes per-chunk metadata when include_metadata is true', async () => {
      const { searchContext } = await import('../tools/search-context.mjs');
      const { client, tempDir } = await setupDb();
      try {
        const result = await searchContext({
          client,
          query: 'function foo',
          use_dense: false,
          include_metadata: true,
        });

        expect(result.results?.[0]).toEqual(
          expect.objectContaining({
            chunk_id: expect.any(Number),
            metadata: expect.any(Object),
          }),
        );
      } finally {
        await teardown(client, tempDir);
      }
    });
  });

  describe('dedup_strategy', () => {
    it('deduplicates identical chunks when dedup_strategy is exact', async () => {
      const { searchContext } = await import('../tools/search-context.mjs');
      const { client, tempDir } = await setupDb();
      try {
        const docResult = await client.execute('SELECT doc_id FROM documents');
        const doc = docResult.rows[0];
        await client.execute({
          sql: `
          INSERT INTO chunks (doc_id, chunk_index, body_text, char_start, char_end, depth)
          VALUES (?, 99, 'function foo0() { return 0; }', 0, 10, 0)
        `,
          args: [doc.doc_id],
        });

        const result = await searchContext({
          client,
          query: 'function foo',
          use_dense: false,
          dedup_strategy: 'exact',
        });

        expect(result.chunks_in_context).toBeLessThan(
          result.total_chunks_retrieved,
        );
      } finally {
        await teardown(client, tempDir);
      }
    });
  });

  describe('internal forwarding', () => {
    it('forwards to search_corpus so results contain chunk_id and feedback_boost', async () => {
      const { searchContext } = await import('../tools/search-context.mjs');
      const { client, tempDir } = await setupDb();
      try {
        const result = await searchContext({
          client,
          query: 'function foo',
          use_dense: false,
        });

        expect(result.results?.[0]).toEqual(
          expect.objectContaining({
            chunk_id: expect.any(Number),
            feedback_boost: expect.any(Number),
          }),
        );
      } finally {
        await teardown(client, tempDir);
      }
    });
  });
});
