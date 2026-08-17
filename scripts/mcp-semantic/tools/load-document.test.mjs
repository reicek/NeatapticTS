/**
 * @module load-document.test
 * @description Coverage tests for load-document.mjs — branch coverage for
 * depth hierarchy, has_sub_chunks, client vs databasePath, and not-found.
 */
import {
  createSchemaClient,
  insertTestFixtures,
  createEnvIsolation,
  TEST_FILE_PATH,
  TEST_FAMILY,
  TEST_DOC_ID,
  TEST_PARENT_CHUNK_ID,
  TEST_CHUNK_ID,
} from '../__tests__/turso-test-helpers.mjs';

const { loadDocument } = await import('./load-document.mjs');

describe('load-document', () => {
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
      try { await client.close(); } catch { /* noop */ }
    }
  });

  describe('basic loading', () => {
    it('loads all chunks for a document with hierarchy metadata', async () => {
      const result = await loadDocument({ file_path: TEST_FILE_PATH, client });

      expect(result.file_path).toBe(TEST_FILE_PATH);
      expect(result.chunks).toHaveLength(2);
      expect(result.chunks[0].depth).toBe(0);
      expect(result.chunks[1].depth).toBe(1);
      expect(result.hierarchy.depth_0_count).toBe(1);
      expect(result.hierarchy.depth_1_count).toBe(1);
      expect(result.hierarchy.has_sub_chunks).toBe(true);
    });

    it('returns chunks ordered by chunk_index', async () => {
      const result = await loadDocument({ file_path: TEST_FILE_PATH, client });
      expect(result.chunks[0].chunk_index).toBeLessThan(result.chunks[1].chunk_index);
    });

    it('maps body_text to text via readChunkRow', async () => {
      const result = await loadDocument({ file_path: TEST_FILE_PATH, client });
      expect(result.chunks[0].text).toContain('tursouniqueword');
    });
  });

  describe('hierarchy branches', () => {
    it('has_sub_chunks=false when only depth-0 chunks exist', async () => {
      // Create a document with only a depth-0 chunk
      await client.execute({
        sql: `INSERT INTO documents (doc_id, file_path, doc_family, mtime_ms, file_size, sha256, indexed_at)
              VALUES (?, ?, ?, ?, ?, ?, ?)`,
        args: [500010, 'src/flat.ts', 'flat-family', 1000, 100, 'sha-flat', 1000],
      });
      await client.execute({
        sql: `INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end, parent_chunk_id, depth)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        args: [430000, 500010, 0, 'Flat', 'flat body', 0, 10, null, 0],
      });

      const result = await loadDocument({ file_path: 'src/flat.ts', client });

      expect(result.hierarchy.depth_0_count).toBe(1);
      expect(result.hierarchy.depth_1_count).toBe(0);
      expect(result.hierarchy.has_sub_chunks).toBe(false);
    });

    it('counts multiple depth-0 and depth-1 chunks correctly', async () => {
      // Add additional chunks to the existing document
      await client.execute({
        sql: `INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end, parent_chunk_id, depth)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        args: [430001, TEST_DOC_ID, 2, 'Extra0', 'extra0 body', 60, 70, null, 0],
      });
      await client.execute({
        sql: `INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end, parent_chunk_id, depth)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        args: [430002, TEST_DOC_ID, 3, 'Extra1', 'extra1 body', 70, 80, TEST_PARENT_CHUNK_ID, 1],
      });

      const result = await loadDocument({ file_path: TEST_FILE_PATH, client });

      expect(result.chunks).toHaveLength(4);
      expect(result.hierarchy.depth_0_count).toBe(2);
      expect(result.hierarchy.depth_1_count).toBe(2);
      expect(result.hierarchy.has_sub_chunks).toBe(true);
    });
  });

  describe('error cases', () => {
    it('throws when document not found', async () => {
      await expect(
        loadDocument({ file_path: 'nonexistent.ts', client }),
      ).rejects.toThrow('Document not found: nonexistent.ts');
    });
  });

  describe('client vs databasePath', () => {
    it('uses getTursoClient when no client provided', async () => {
      const { setTursoClient, closeTursoClient } = await import('./cortex-db.mjs');
      const dbPath = ':memory:test-load-document';
      setTursoClient(dbPath, client);
      try {
        const result = await loadDocument({
          file_path: TEST_FILE_PATH,
          databasePath: dbPath,
        });
        expect(result.chunks).toHaveLength(2);
      } finally {
        closeTursoClient(dbPath);
      }
    });
  });
});