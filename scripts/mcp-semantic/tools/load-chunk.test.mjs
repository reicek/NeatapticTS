/**
 * @module load-chunk.test
 * @description Coverage tests for load-chunk.mjs — branch coverage for cache,
 * fallback, next_chunk null, query normalization, and click recording failure.
 */
import {
  createSchemaClient,
  insertTestFixtures,
  createEnvIsolation,
  TEST_CHUNK_ID,
  TEST_PARENT_CHUNK_ID,
  TEST_FILE_PATH,
  TEST_FAMILY,
} from '../__tests__/turso-test-helpers.mjs';

const { loadChunk } = await import('./load-chunk.mjs');

describe('load-chunk', () => {
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

  describe('basic loading', () => {
    it('loads a chunk by ID with full metadata', async () => {
      const result = await loadChunk({ chunk_id: TEST_CHUNK_ID, client });

      expect(result.chunk.chunk_id).toBe(TEST_CHUNK_ID);
      expect(result.chunk.file_path).toBe(TEST_FILE_PATH);
      expect(result.chunk.family).toBe(TEST_FAMILY);
      expect(result.chunk.depth).toBe(1);
      expect(result.chunk.parent_chunk_id).toBe(TEST_PARENT_CHUNK_ID);
      expect(result.chunk.symbol_name).toBe('TursoAsyncMethod');
    });

    it('loads a depth-0 parent chunk', async () => {
      const result = await loadChunk({
        chunk_id: TEST_PARENT_CHUNK_ID,
        client,
      });

      expect(result.chunk.chunk_id).toBe(TEST_PARENT_CHUNK_ID);
      expect(result.chunk.depth).toBe(0);
      expect(result.chunk.parent_chunk_id).toBeNull();
      expect(result.chunk.symbol_name).toBe('TursoAsyncTestSymbol');
    });
  });

  describe('next_chunk_id', () => {
    it('returns next_chunk_id when a next chunk exists', async () => {
      // TEST_PARENT_CHUNK_ID (index 0) → TEST_CHUNK_ID (index 1)
      const result = await loadChunk({
        chunk_id: TEST_PARENT_CHUNK_ID,
        client,
      });
      expect(result.chunk.next_chunk_id).toBe(TEST_CHUNK_ID);
    });

    it('returns null next_chunk_id when no next chunk exists', async () => {
      // TEST_CHUNK_ID is the last chunk (index 1) → no next
      const result = await loadChunk({ chunk_id: TEST_CHUNK_ID, client });
      expect(result.chunk.next_chunk_id).toBeNull();
    });
  });

  describe('chunk_id=1 fallback', () => {
    it('falls back to lowest-ID chunk when chunk_id=1 not found', async () => {
      // Our test chunks have IDs 424241 and 424242, so chunk_id=1 won't be found.
      // The fallback should find the lowest chunk_id (424241).
      const result = await loadChunk({ chunk_id: 1, client });

      expect(result.chunk.chunk_id).toBe(1); // Returns the requested ID, not the actual
      expect(result.chunk.file_path).toBe(TEST_FILE_PATH);
    });
  });

  describe('error cases', () => {
    it('throws on non-integer chunk_id', async () => {
      await expect(loadChunk({ chunk_id: 1.5, client })).rejects.toThrow(
        'chunk_id must be a positive integer',
      );
    });

    it('throws on zero chunk_id', async () => {
      await expect(loadChunk({ chunk_id: 0, client })).rejects.toThrow(
        'chunk_id must be a positive integer',
      );
    });

    it('throws on negative chunk_id', async () => {
      await expect(loadChunk({ chunk_id: -1, client })).rejects.toThrow(
        'chunk_id must be a positive integer',
      );
    });

    it('throws on non-number chunk_id', async () => {
      await expect(loadChunk({ chunk_id: 'abc', client })).rejects.toThrow(
        'chunk_id must be a positive integer',
      );
    });

    it('throws when chunk not found (and not chunk_id=1)', async () => {
      await expect(loadChunk({ chunk_id: 999999, client })).rejects.toThrow(
        'Chunk not found: 999999',
      );
    });
  });

  describe('click recording', () => {
    it('records click event on first load with query', async () => {
      await loadChunk({
        chunk_id: TEST_CHUNK_ID,
        query: 'test query',
        client,
      });

      const events = await client.execute(
        "SELECT COUNT(*) as count FROM feedback_events WHERE signal_type = 'click'",
      );
      expect(Number(events.rows[0].count)).toBe(1);
    });

    it('records click with query_hash when provided', async () => {
      const hash = 'a'.repeat(64);
      await loadChunk({
        chunk_id: TEST_CHUNK_ID,
        query_hash: hash,
        client,
      });

      const events = await client.execute(
        "SELECT query_hash FROM feedback_events WHERE signal_type = 'click' LIMIT 1",
      );
      expect(events.rows[0].query_hash).toBe(hash);
    });

    it('hashes plaintext query for storage', async () => {
      await loadChunk({
        chunk_id: TEST_PARENT_CHUNK_ID, // different chunk to avoid cache
        query: 'plaintext search',
        client,
      });

      const events = await client.execute(
        "SELECT query_hash FROM feedback_events WHERE signal_type = 'click' LIMIT 1",
      );
      // Should be a 64-char hex hash
      expect(events.rows[0].query_hash).toMatch(/^[0-9a-f]{64}$/);
    });

    it('stores null query_hash when no query provided', async () => {
      // Insert a fresh chunk with a unique ID to avoid click cache hits
      await client.execute({
        sql: `INSERT INTO chunks (chunk_id, doc_id, chunk_index, heading_path, body_text, char_start, char_end, depth, symbol_name)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)`,
        args: [
          424243,
          500001,
          99,
          'Test',
          'unique no-query chunk',
          0,
          10,
          0,
          'UniqueNoQuerySymbol',
        ],
      });

      await loadChunk({
        chunk_id: 424243,
        client,
      });

      const events = await client.execute(
        "SELECT query_hash FROM feedback_events WHERE signal_type = 'click' AND chunk_id = 424243 LIMIT 1",
      );
      expect(events.rows[0].query_hash).toBeNull();
    });

    it('does NOT record click on second load with same chunk+query (cache hit)', async () => {
      // First load records click
      await loadChunk({ chunk_id: TEST_CHUNK_ID, query: 'cached', client });
      // Second load with same chunk+query should NOT record (cache hit)
      await loadChunk({ chunk_id: TEST_CHUNK_ID, query: 'cached', client });

      const events = await client.execute(
        "SELECT COUNT(*) as count FROM feedback_events WHERE signal_type = 'click' AND chunk_id = ?",
        [TEST_CHUNK_ID],
      );
      expect(Number(events.rows[0].count)).toBe(1);
    });

    it('records click again with different query for same chunk', async () => {
      await loadChunk({ chunk_id: TEST_CHUNK_ID, query: 'query1', client });
      await loadChunk({ chunk_id: TEST_CHUNK_ID, query: 'query2', client });

      const events = await client.execute(
        "SELECT COUNT(*) as count FROM feedback_events WHERE signal_type = 'click' AND chunk_id = ?",
        [TEST_CHUNK_ID],
      );
      expect(Number(events.rows[0].count)).toBe(2);
    });

    it('silently drops click recording failures', async () => {
      // Use a client that throws on INSERT INTO feedback_events
      const failingClient = {
        async execute(sqlOrObj) {
          const sql = typeof sqlOrObj === 'string' ? sqlOrObj : sqlOrObj.sql;
          if (sql.includes('INSERT INTO feedback_events')) {
            throw new Error('feedback table missing');
          }
          if (sql.includes('SELECT') && sql.includes('WHERE c.chunk_id =')) {
            return {
              rows: [
                {
                  file_path: TEST_FILE_PATH,
                  doc_family: TEST_FAMILY,
                  doc_id: 500001,
                  chunk_id: TEST_CHUNK_ID,
                  chunk_index: 1,
                  heading_path: 'test',
                  body_text: 'body',
                  char_start: 0,
                  char_end: 10,
                  parent_chunk_id: TEST_PARENT_CHUNK_ID,
                  depth: 1,
                  context_header: null,
                  symbol_name: 'Test',
                  signature_text: null,
                  jsdoc_text: null,
                  export_type: null,
                  module_path: null,
                },
              ],
            };
          }
          if (
            sql.includes('SELECT c.chunk_id') &&
            sql.includes('chunk_index >')
          ) {
            return { rows: [] };
          }
          if (sql.includes('ORDER BY c.chunk_id') && sql.includes('LIMIT 1')) {
            return { rows: [] };
          }
          return { rows: [] };
        },
      };

      // Should not throw — click failure is silently caught
      const result = await loadChunk({
        chunk_id: TEST_CHUNK_ID,
        query: 'test',
        client: failingClient,
      });
      expect(result.chunk.chunk_id).toBe(TEST_CHUNK_ID);
    });
  });

  describe('readChunkRow normalization', () => {
    it('maps body_text to text and doc_family to family', async () => {
      const result = await loadChunk({ chunk_id: TEST_CHUNK_ID, client });
      expect(result.chunk.text).toContain('tursouniqueword');
      expect(result.chunk.family).toBe(TEST_FAMILY);
    });
  });
});
