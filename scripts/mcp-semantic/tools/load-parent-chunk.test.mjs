/**
 * @module load-parent-chunk.test
 * @description Coverage tests for load-parent-chunk.mjs — all error paths and
 * success cases for parent chunk resolution.
 */
import {
  createSchemaClient,
  insertTestFixtures,
  createEnvIsolation,
  TEST_CHUNK_ID,
  TEST_PARENT_CHUNK_ID,
  TEST_FILE_PATH,
  TEST_FAMILY,
  TEST_DOC_ID,
} from '../__tests__/turso-test-helpers.mjs';

const { loadParentChunk } = await import('./load-parent-chunk.mjs');

describe('load-parent-chunk', () => {
  const { saveEnv, restoreEnv } = createEnvIsolation();
  let client;

  beforeEach(async () => {
    saveEnv();
    client = await createSchemaClient();
    await insertTestFixtures(client);
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

  describe('success', () => {
    it('loads the parent chunk for a depth-1 sub-chunk', async () => {
      const result = await loadParentChunk({
        chunk_id: TEST_CHUNK_ID,
        client,
      });

      expect(result.parent_chunk.chunk_id).toBe(TEST_PARENT_CHUNK_ID);
      expect(result.parent_chunk.file_path).toBe(TEST_FILE_PATH);
      expect(result.parent_chunk.family).toBe(TEST_FAMILY);
      expect(result.parent_chunk.depth).toBe(0);
      expect(result.parent_chunk.parent_chunk_id).toBeNull();
      expect(result.parent_chunk.symbol_name).toBe('TursoAsyncTestSymbol');
    });
  });

  describe('error cases', () => {
    it('throws on non-integer chunk_id', async () => {
      await expect(loadParentChunk({ chunk_id: 1.5, client })).rejects.toThrow(
        'chunk_id must be a positive integer',
      );
    });

    it('throws on zero chunk_id', async () => {
      await expect(loadParentChunk({ chunk_id: 0, client })).rejects.toThrow(
        'chunk_id must be a positive integer',
      );
    });

    it('throws on negative chunk_id', async () => {
      await expect(loadParentChunk({ chunk_id: -5, client })).rejects.toThrow(
        'chunk_id must be a positive integer',
      );
    });

    it('throws on non-number chunk_id', async () => {
      await expect(
        loadParentChunk({ chunk_id: 'abc', client }),
      ).rejects.toThrow('chunk_id must be a positive integer');
    });

    it('throws when chunk not found', async () => {
      await expect(
        loadParentChunk({ chunk_id: 999999, client }),
      ).rejects.toThrow('Chunk not found: 999999');
    });

    it('throws when chunk has depth 0 (no parent)', async () => {
      await expect(
        loadParentChunk({ chunk_id: TEST_PARENT_CHUNK_ID, client }),
      ).rejects.toThrow('top-level chunk (depth 0) with no parent');
    });

    it('throws when parent_chunk_id is null but depth is not 0', async () => {
      // Insert a chunk with depth=1 but null parent_chunk_id
      await client.execute({
        sql: `INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end, parent_chunk_id, depth)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        args: [430010, TEST_DOC_ID, 5, 'Orphan', 'orphan body', 0, 10, null, 1],
      });

      await expect(
        loadParentChunk({ chunk_id: 430010, client }),
      ).rejects.toThrow('top-level chunk (depth 0) with no parent');
    });

    it('throws when parent chunk is missing from DB', async () => {
      // Insert a chunk with parent_chunk_id pointing to non-existent parent
      await client.execute({
        sql: `INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end, parent_chunk_id, depth)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        args: [
          430020,
          TEST_DOC_ID,
          6,
          'Orphan2',
          'orphan2 body',
          0,
          10,
          777777,
          1,
        ],
      });

      await expect(
        loadParentChunk({ chunk_id: 430020, client }),
      ).rejects.toThrow('Parent chunk not found: 777777');
    });
  });

  describe('client vs databasePath', () => {
    it('uses getTursoClient when no client provided', async () => {
      const { setTursoClient, closeTursoClient } =
        await import('./cortex-db.mjs');
      const dbPath = ':memory:test-load-parent';
      setTursoClient(dbPath, client);
      try {
        const result = await loadParentChunk({
          chunk_id: TEST_CHUNK_ID,
          databasePath: dbPath,
        });
        expect(result.parent_chunk.chunk_id).toBe(TEST_PARENT_CHUNK_ID);
      } finally {
        closeTursoClient(dbPath);
      }
    });
  });
});
