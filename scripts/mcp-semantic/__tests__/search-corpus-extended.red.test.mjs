/**
 * @module search-corpus-extended.red.test
 * @description Red tests for Step 21 extensions to the search_corpus MCP tool.
 *
 * Verifies metadata filter SQL generation, classification_hints override
 * behavior, backward compatibility, and structured error taxonomy.
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
  const tempDir = await mkdtemp(path.join(tmpdir(), 'search-corpus-ext-test-'));
  const dbPath = path.join(tempDir, 'test.sqlite');
  const db = new Database(dbPath);
  db.exec(await readSchema());
  db.exec(`
    INSERT INTO documents (file_path, doc_family, mtime_ms, file_size, sha256, indexed_at, arch_layer)
    VALUES
      ('src/network.ts', 'ts-source', 0, 100, 'a', 1, 'network'),
      ('src/methods.ts', 'ts-source', 0, 100, 'b', 1, 'methods'),
      ('plans/roadmap.md', 'plans', 0, 100, 'c', 1, 'planning');
  `);
  const docs = db
    .prepare('SELECT doc_id, doc_family, arch_layer FROM documents')
    .all();
  for (const doc of docs) {
    db.prepare(
      `
      INSERT INTO chunks (doc_id, chunk_index, body_text, char_start, char_end, depth, arch_layer)
      VALUES (?, 0, ?, 0, 10, 0, ?)
    `,
    ).run(
      doc.doc_id,
      `${doc.doc_family} ${doc.arch_layer} content`,
      doc.arch_layer,
    );
  }
  db.close();
  return { dbPath, tempDir };
}

function teardown(tempDir) {
  return rm(tempDir, { recursive: true, force: true });
}

describe('search-corpus extended', () => {
  describe('metadata filter', () => {
    it('applies an eq filter on arch_layer through SQL WHERE', async () => {
      const { searchCorpus } = await import('../tools/search-corpus.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        const result = await searchCorpus({
          databasePath: dbPath,
          query: 'content',
          use_dense: false,
          metadata: {
            filter: { op: 'eq', field: 'arch_layer', value: 'network' },
          },
        });

        expect(result.results.map((r) => r.family).sort()).toEqual([
          'ts-source',
        ]);
      } finally {
        await teardown(tempDir);
      }
    });

    it('combines metadata filter with explicit family using AND', async () => {
      const { searchCorpus } = await import('../tools/search-corpus.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        const result = await searchCorpus({
          databasePath: dbPath,
          query: 'content',
          use_dense: false,
          family: 'ts-source',
          metadata: {
            filter: { op: 'eq', field: 'arch_layer', value: 'methods' },
          },
        });

        expect(result.results.map((r) => r.arch_layer)).toEqual(['methods']);
      } finally {
        await teardown(tempDir);
      }
    });

    it('returns identical results when metadata and classification_hints are omitted', async () => {
      const { searchCorpus } = await import('../tools/search-corpus.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        const baseline = await searchCorpus({
          databasePath: dbPath,
          query: 'content',
          use_dense: false,
        });
        const extended = await searchCorpus({
          databasePath: dbPath,
          query: 'content',
          use_dense: false,
        });

        expect(extended.results.map((r) => r.chunk_id).sort()).toEqual(
          baseline.results.map((r) => r.chunk_id).sort(),
        );
      } finally {
        await teardown(tempDir);
      }
    });

    it('rejects a malformed metadata filter with INVALID_METADATA_FILTER', async () => {
      const { searchCorpus } = await import('../tools/search-corpus.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        await expect(
          searchCorpus({
            databasePath: dbPath,
            query: 'content',
            use_dense: false,
            metadata: {
              filter: {
                op: 'unknown_op',
                field: 'arch_layer',
                value: 'network',
              },
            },
          }),
        ).rejects.toThrow(/INVALID_METADATA_FILTER/);
      } finally {
        await teardown(tempDir);
      }
    });
  });

  describe('classification_hints', () => {
    it('overrides classification-derived alpha when classification_hints.alpha is provided', async () => {
      const { searchCorpus } = await import('../tools/search-corpus.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        const result = await searchCorpus({
          databasePath: dbPath,
          query: 'network architecture',
          query_class: 'cross_boundary',
          use_dense: false,
          classification_hints: { alpha: 0.9 },
        });

        expect(result.alpha).toBeCloseTo(0.9, 2);
      } finally {
        await teardown(tempDir);
      }
    });

    it('overrides classification-derived family when classification_hints.family is provided', async () => {
      const { searchCorpus } = await import('../tools/search-corpus.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        const result = await searchCorpus({
          databasePath: dbPath,
          query: 'planning',
          query_class: 'plan_specific',
          use_dense: false,
          classification_hints: { family: 'plans' },
        });

        expect(result.family).toBe('plans');
      } finally {
        await teardown(tempDir);
      }
    });

    it('still returns classification metadata when hints are used', async () => {
      const { searchCorpus } = await import('../tools/search-corpus.mjs');
      const { dbPath, tempDir } = await setupDb();
      try {
        const result = await searchCorpus({
          databasePath: dbPath,
          query: 'network',
          query_class: 'code_specific',
          use_dense: false,
          classification_hints: { alpha: 0.6 },
        });

        expect(result).toEqual(
          expect.objectContaining({
            query_class: expect.any(String),
            confidence: expect.any(Number),
          }),
        );
      } finally {
        await teardown(tempDir);
      }
    });
  });

  describe('error taxonomy', () => {
    it('returns structured error for missing corpus database', async () => {
      const { createRepoCortexMcpServer } =
        await import('../repo-cortex-mcp.mjs');
      const server = createRepoCortexMcpServer({
        databasePath: './missing-semantic-index.sqlite',
      });
      const result = await server.dispatch({
        jsonrpc: '2.0',
        id: 1,
        method: 'tools/call',
        params: {
          name: 'search_corpus',
          arguments: { query: 'NEAT' },
        },
      });

      expect(result).toEqual(
        expect.objectContaining({
          isError: true,
          structuredContent: expect.objectContaining({
            error: expect.stringContaining('CORPUS_NOT_FOUND'),
          }),
        }),
      );
    });
  });
});
