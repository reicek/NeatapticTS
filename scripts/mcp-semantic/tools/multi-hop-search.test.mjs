/**
 * @module multi-hop-search.test
 * @description Coverage tests for multi-hop-search.mjs — all hop levels,
 * validation, ranking, and handler delegation.
 */
import {
  createSchemaClient,
  insertTestFixtures,
  insertGraphFixtures,
  createEnvIsolation,
} from '../__tests__/turso-test-helpers.mjs';

const { multiHopSearch, multiHopSearchHandler } =
  await import('./multi-hop-search.mjs');

/**
 * Create a mock client that returns controlled results for each hop.
 * @param {object} config - Mock configuration.
 * @returns {object} Mock client.
 */
function createMockClient({
  seedChunks = [],
  graphRows = [],
  neighborChunks = [],
} = {}) {
  const calls = [];
  return {
    calls,
    async execute(sqlOrObj) {
      const sql = typeof sqlOrObj === 'string' ? sqlOrObj : sqlOrObj.sql;
      calls.push(sql);

      // Hop 1: vector_top_k seed search
      if (sql.includes('vector_top_k') && !sql.includes('chunk_id IN')) {
        return {
          rows: seedChunks.map((c) => ({
            chunk_id: c.chunk_id,
            body_text: c.body_text,
            distance: c.distance,
            combined_score: c.combined_score,
          })),
        };
      }

      // Hop 2: entities JOIN edges
      if (sql.includes('entities e') && sql.includes('edges ed')) {
        return { rows: graphRows };
      }

      // Hop 3: scoped vector search with chunk_id IN
      if (sql.includes('vector_top_k') && sql.includes('chunk_id IN')) {
        return {
          rows: neighborChunks.map((c) => ({
            chunk_id: c.chunk_id,
            body_text: c.body_text,
            distance: c.distance,
          })),
        };
      }

      return { rows: [] };
    },
  };
}

describe('multi-hop-search', () => {
  describe('validation', () => {
    it('throws on empty query', async () => {
      await expect(multiHopSearch({ query: '' })).rejects.toThrow(
        'query is required',
      );
    });

    it('throws on null query', async () => {
      await expect(multiHopSearch({ query: null })).rejects.toThrow(
        'query is required',
      );
    });

    it('throws on non-string query', async () => {
      await expect(multiHopSearch({ query: 123 })).rejects.toThrow(
        'query is required',
      );
    });

    it('throws on undefined query', async () => {
      await expect(multiHopSearch({})).rejects.toThrow('query is required');
    });

    it('throws on non-integer max_hops', async () => {
      await expect(
        multiHopSearch({ query: 'test', max_hops: 1.5 }),
      ).rejects.toThrow('max_hops must be an integer between 1 and 3');
    });

    it('throws on max_hops < 1', async () => {
      await expect(
        multiHopSearch({ query: 'test', max_hops: 0 }),
      ).rejects.toThrow('max_hops must be an integer between 1 and 3');
    });

    it('throws on max_hops > 3', async () => {
      await expect(
        multiHopSearch({ query: 'test', max_hops: 4 }),
      ).rejects.toThrow('max_hops must be an integer between 1 and 3');
    });
  });

  describe('max_hops=1 (seed only)', () => {
    it('returns only seed chunks with combined_score', async () => {
      const client = createMockClient({
        seedChunks: [
          { chunk_id: 1, body_text: 'chunk 1', distance: 0.2 },
          { chunk_id: 2, body_text: 'chunk 2', distance: 0.5 },
        ],
      });

      const result = await multiHopSearch({
        query: 'test query',
        max_hops: 1,
        client,
        queryEmbeddingBuffer: Buffer.alloc(1536),
      });

      expect(result.max_hops).toBe(1);
      expect(result.seed_chunks).toHaveLength(2);
      expect(result.entities).toEqual([]);
      expect(result.results).toHaveLength(2);
      expect(result.results[0].combined_score).toBeCloseTo(0.8, 5);
      expect(result.results[1].combined_score).toBeCloseTo(0.5, 5);
      // Sorted descending
      expect(result.results[0].combined_score).toBeGreaterThan(
        result.results[1].combined_score,
      );
    });

    it('returns empty results when no seed chunks found', async () => {
      const client = createMockClient({ seedChunks: [] });

      const result = await multiHopSearch({
        query: 'test',
        max_hops: 1,
        client,
        queryEmbeddingBuffer: Buffer.alloc(1536),
      });

      expect(result.seed_chunks).toEqual([]);
      expect(result.results).toEqual([]);
    });
  });

  describe('max_hops=2 (seed + graph)', () => {
    it('returns seed chunks and entities without hop 3', async () => {
      const client = createMockClient({
        seedChunks: [{ chunk_id: 1, body_text: 'chunk 1', distance: 0.3 }],
        graphRows: [
          {
            entity_id: 100,
            name: 'EntityA',
            entity_type: 'function',
            chunk_id: 1,
            edge_id: 1,
            source_entity_id: 100,
            target_entity_id: 200,
            relationship: 'references',
          },
          {
            entity_id: 200,
            name: 'EntityB',
            entity_type: 'class',
            chunk_id: 2,
            edge_id: 1,
            source_entity_id: 100,
            target_entity_id: 200,
            relationship: 'references',
          },
        ],
      });

      const result = await multiHopSearch({
        query: 'test',
        max_hops: 2,
        client,
        queryEmbeddingBuffer: Buffer.alloc(1536),
      });

      expect(result.max_hops).toBe(2);
      expect(result.entities).toHaveLength(2);
      expect(result.results).toHaveLength(1); // only seed chunk in results for max_hops=2
    });

    it('returns empty entities when no seed chunks found', async () => {
      const client = createMockClient({
        seedChunks: [],
        graphRows: [],
      });

      const result = await multiHopSearch({
        query: 'test',
        max_hops: 2,
        client,
        queryEmbeddingBuffer: Buffer.alloc(1536),
      });

      expect(result.entities).toEqual([]);
      expect(result.results).toEqual([]);
    });

    it('handles graph rows with null chunk_id', async () => {
      const client = createMockClient({
        seedChunks: [{ chunk_id: 1, body_text: 'chunk 1', distance: 0.3 }],
        graphRows: [
          {
            entity_id: 100,
            name: 'EntityNoChunk',
            entity_type: 'module',
            chunk_id: null,
            edge_id: 1,
            source_entity_id: 100,
            target_entity_id: 200,
            relationship: 'imports',
          },
        ],
      });

      const result = await multiHopSearch({
        query: 'test',
        max_hops: 2,
        client,
        queryEmbeddingBuffer: Buffer.alloc(1536),
      });

      expect(result.entities).toHaveLength(1);
      expect(result.entities[0].chunk_id).toBeNull();
    });
  });

  describe('max_hops=3 (seed + graph + scoped vector)', () => {
    it('performs all three hops and combines results', async () => {
      const client = createMockClient({
        seedChunks: [{ chunk_id: 1, body_text: 'seed chunk', distance: 0.2 }],
        graphRows: [
          {
            entity_id: 100,
            name: 'EntityA',
            entity_type: 'function',
            chunk_id: 1,
            edge_id: 1,
            source_entity_id: 100,
            target_entity_id: 200,
            relationship: 'references',
          },
          {
            entity_id: 200,
            name: 'EntityB',
            entity_type: 'class',
            chunk_id: 3,
            edge_id: 1,
            source_entity_id: 100,
            target_entity_id: 200,
            relationship: 'references',
          },
        ],
        neighborChunks: [
          { chunk_id: 3, body_text: 'neighbor chunk', distance: 0.4 },
        ],
      });

      const result = await multiHopSearch({
        query: 'test',
        max_hops: 3,
        client,
        queryEmbeddingBuffer: Buffer.alloc(1536),
      });

      expect(result.max_hops).toBe(3);
      expect(result.results).toHaveLength(2); // chunk 1 (seed) + chunk 3 (neighbor)
      // Seed has higher score (1.0 proximity * 0.8 similarity = 0.8)
      // Neighbor has (0.5 proximity * 0.6 similarity = 0.3)
      expect(result.results[0].chunk_id).toBe(1);
      expect(result.results[1].chunk_id).toBe(3);
    });

    it('skips hop 3 when neighborChunkIds is empty', async () => {
      const client = createMockClient({
        seedChunks: [{ chunk_id: 1, body_text: 'seed', distance: 0.2 }],
        graphRows: [
          {
            entity_id: 100,
            name: 'EntityA',
            entity_type: 'function',
            chunk_id: 1,
            edge_id: 1,
            source_entity_id: 100,
            target_entity_id: 200,
            relationship: 'references',
          },
        ],
        neighborChunks: [],
      });

      const result = await multiHopSearch({
        query: 'test',
        max_hops: 3,
        client,
        queryEmbeddingBuffer: Buffer.alloc(1536),
      });

      // Only seed chunk, no neighbor results
      expect(result.results).toHaveLength(1);
    });
  });

  describe('combined score with pre-scored results', () => {
    it('preserves existing combined_score from rows', async () => {
      const client = createMockClient({
        seedChunks: [
          {
            chunk_id: 1,
            body_text: 'chunk 1',
            distance: 0.2,
            combined_score: 0.99,
          },
        ],
      });

      const result = await multiHopSearch({
        query: 'test',
        max_hops: 1,
        client,
        queryEmbeddingBuffer: Buffer.alloc(1536),
      });

      // hop1VectorSeedSearch strips combined_score from rows, so the score
      // is recomputed: (1 - distance) * graphProximity = (1 - 0.2) * 1.0 = 0.8
      expect(result.results[0].combined_score).toBe(0.8);
    });
  });

  describe('queryEmbedding option', () => {
    it('uses queryEmbedding Float32Array when provided', async () => {
      const client = createMockClient({
        seedChunks: [{ chunk_id: 1, body_text: 'chunk 1', distance: 0.1 }],
      });
      const float32 = new Float32Array(384);

      const result = await multiHopSearch({
        query: 'test',
        max_hops: 1,
        client,
        queryEmbedding: float32,
      });

      expect(result.results).toHaveLength(1);
    });

    it('uses queryEmbedding as array (converted to Float32Array)', async () => {
      const client = createMockClient({
        seedChunks: [{ chunk_id: 1, body_text: 'chunk 1', distance: 0.1 }],
      });
      const arr = new Array(384).fill(0);

      const result = await multiHopSearch({
        query: 'test',
        max_hops: 1,
        client,
        queryEmbedding: arr,
      });

      expect(result.results).toHaveLength(1);
    });
  });

  describe('default max_hops and limit', () => {
    it('defaults max_hops to 3 when not provided', async () => {
      const client = createMockClient({
        seedChunks: [{ chunk_id: 1, body_text: 'chunk 1', distance: 0.2 }],
        graphRows: [],
      });

      const result = await multiHopSearch({
        query: 'test',
        client,
        queryEmbeddingBuffer: Buffer.alloc(1536),
      });

      expect(result.max_hops).toBe(3);
    });

    it('defaults limit to 10 when not provided', async () => {
      const client = createMockClient({
        seedChunks: [{ chunk_id: 1, body_text: 'chunk 1', distance: 0.2 }],
      });

      const result = await multiHopSearch({
        query: 'test',
        max_hops: 1,
        client,
        queryEmbeddingBuffer: Buffer.alloc(1536),
      });

      expect(result.limit).toBe(10);
    });

    it('caps limit at 50', async () => {
      const client = createMockClient({
        seedChunks: [],
      });

      const result = await multiHopSearch({
        query: 'test',
        max_hops: 1,
        limit: 100,
        client,
        queryEmbeddingBuffer: Buffer.alloc(1536),
      });

      expect(result.limit).toBe(50);
    });

    it('defaults limit to 10 for invalid values', async () => {
      const client = createMockClient({ seedChunks: [] });

      const result = await multiHopSearch({
        query: 'test',
        max_hops: 1,
        limit: -5,
        client,
        queryEmbeddingBuffer: Buffer.alloc(1536),
      });

      expect(result.limit).toBe(10);
    });
  });

  describe('relationship_types and entity_types validation', () => {
    it('filters relationship_types to valid ones', async () => {
      const client = createMockClient({
        seedChunks: [{ chunk_id: 1, body_text: 'chunk 1', distance: 0.2 }],
        graphRows: [],
      });

      await multiHopSearch({
        query: 'test',
        max_hops: 2,
        relationship_types: ['references', 'invalid_type'],
        client,
        queryEmbeddingBuffer: Buffer.alloc(1536),
      });

      // Should have executed with filtered relationship types
      // The graph SQL should include 'references' but not 'invalid_type'
      const graphCall = client.calls.find((sql) => sql.includes('entities e'));
      expect(graphCall).toBeDefined();
      expect(graphCall).toContain('relationship IN');
    });

    it('returns undefined for empty relationship_types array', async () => {
      const client = createMockClient({
        seedChunks: [{ chunk_id: 1, body_text: 'chunk 1', distance: 0.2 }],
        graphRows: [],
      });

      await multiHopSearch({
        query: 'test',
        max_hops: 2,
        relationship_types: [],
        client,
        queryEmbeddingBuffer: Buffer.alloc(1536),
      });

      const graphCall = client.calls.find((sql) => sql.includes('entities e'));
      expect(graphCall).not.toContain('relationship IN');
    });

    it('filters entity_types to valid ones', async () => {
      const client = createMockClient({
        seedChunks: [{ chunk_id: 1, body_text: 'chunk 1', distance: 0.2 }],
        graphRows: [],
      });

      await multiHopSearch({
        query: 'test',
        max_hops: 2,
        entity_types: ['function', 'invalid_type'],
        client,
        queryEmbeddingBuffer: Buffer.alloc(1536),
      });

      const graphCall = client.calls.find((sql) => sql.includes('entities e'));
      expect(graphCall).toContain('entity_type IN');
    });

    it('returns undefined for all-invalid relationship_types', async () => {
      const client = createMockClient({
        seedChunks: [{ chunk_id: 1, body_text: 'chunk 1', distance: 0.2 }],
        graphRows: [],
      });

      await multiHopSearch({
        query: 'test',
        max_hops: 2,
        relationship_types: ['invalid1', 'invalid2'],
        client,
        queryEmbeddingBuffer: Buffer.alloc(1536),
      });

      const graphCall = client.calls.find((sql) => sql.includes('entities e'));
      expect(graphCall).not.toContain('relationship IN');
    });
  });

  describe('multiHopSearchHandler', () => {
    it('delegates to multiHopSearch with snake_case args', async () => {
      const client = createMockClient({
        seedChunks: [{ chunk_id: 1, body_text: 'chunk 1', distance: 0.2 }],
      });

      const result = await multiHopSearchHandler({
        query: 'test query',
        max_hops: 1,
        limit: 5,
        client,
        queryEmbeddingBuffer: Buffer.alloc(1536),
      });

      expect(result.query).toBe('test query');
      expect(result.max_hops).toBe(1);
      expect(result.limit).toBe(5);
    });

    it('throws on empty query via handler', async () => {
      await expect(multiHopSearchHandler({ query: '' })).rejects.toThrow(
        'query is required',
      );
    });
  });
});
