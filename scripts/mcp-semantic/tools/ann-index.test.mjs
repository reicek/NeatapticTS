/**
 * @module ann-index.test
 * @description Coverage tests for ann-index.mjs — DiskANN vector index builder.
 */

/**
 * Build a mock libSQL client that tracks execute calls.
 * @param {object} config - Mock configuration.
 * @param {number} [config.chunkCount=100] - Chunk count to return from COUNT query.
 * @param {boolean|Error} [config.tunedFails=false] - Whether tuned SQL fails.
 * @param {boolean|Error} [config.basicFails=false] - Whether basic SQL fails.
 * @returns {object} Mock client.
 */
function createMockClient({
  chunkCount = 100,
  tunedFails = false,
  basicFails = false,
} = {}) {
  const calls = [];
  return {
    calls,
    async execute(sqlOrObj) {
      const sql =
        typeof sqlOrObj === 'string' ? sqlOrObj : sqlOrObj.sql;
      calls.push(sql);

      // COUNT query
      if (sql.includes('SELECT COUNT(*)')) {
        return { rows: [{ count: chunkCount }] };
      }

      // CREATE INDEX with WITH KEY (tuned)
      if (sql.includes('WITH KEY')) {
        if (tunedFails) {
          throw tunedFails;
        }
        return { rows: [] };
      }

      // CREATE INDEX without WITH KEY (basic fallback)
      if (sql.includes('libsql_vector_idx') && !sql.includes('WITH KEY')) {
        if (basicFails) {
          throw basicFails;
        }
        return { rows: [] };
      }

      return { rows: [] };
    },
  };
}

describe('ann-index', () => {
  describe('buildAnnIndex', () => {
    it('throws when neither databasePath nor client is provided', async () => {
      const { buildAnnIndex } = await import('./ann-index.mjs');
      await expect(buildAnnIndex({})).rejects.toThrow(
        'databasePath or client is required',
      );
    });

    it('builds tuned DiskANN index successfully', async () => {
      const { buildAnnIndex } = await import('./ann-index.mjs');
      const client = createMockClient({ chunkCount: 500 });

      const result = await buildAnnIndex({ client });

      expect(result.strategy).toBe('diskann');
      expect(result.build_status).toBe('ready');
      expect(result.index_id).toBe('diskann_all-MiniLM-L6-v2_384');
      expect(result.index_type).toBe('diskann');
      expect(result.build_error).toBeNull();
      expect(result.current_elements).toBe(500);
      expect(result.build_started_at).toBeDefined();
      expect(result.build_completed_at).toBeDefined();
      // Should have called tuned SQL (with WITH KEY)
      expect(
        client.calls.some((sql) => sql.includes('WITH KEY')),
      ).toBe(true);
    });

    it('falls back to basic DiskANN when tuned SQL fails', async () => {
      const { buildAnnIndex } = await import('./ann-index.mjs');
      const client = createMockClient({
        chunkCount: 200,
        tunedFails: true,
      });

      const result = await buildAnnIndex({ client });

      expect(result.build_status).toBe('ready');
      expect(result.index_id).toBe('diskann_all-MiniLM-L6-v2_384');
      expect(result.index_type).toBe('diskann');
      // Both tuned and basic should have been called
      expect(client.calls.some((sql) => sql.includes('WITH KEY'))).toBe(true);
      expect(
        client.calls.some(
          (sql) =>
            sql.includes('libsql_vector_idx') &&
            !sql.includes('WITH KEY'),
        ),
      ).toBe(true);
    });

    it('returns error status when both tuned and basic SQL fail', async () => {
      const { buildAnnIndex } = await import('./ann-index.mjs');
      const client = createMockClient({
        chunkCount: 50,
        tunedFails: true,
        basicFails: new Error('basic index also failed'),
      });

      const result = await buildAnnIndex({ client });

      expect(result.build_status).toBe('error');
      expect(result.index_id).toBeNull();
      expect(result.index_type).toBeNull();
      expect(result.build_error).toBe('basic index also failed');
      expect(result.current_elements).toBe(50);
    });

    it('returns error status when basic fails with non-Error value', async () => {
      const { buildAnnIndex } = await import('./ann-index.mjs');
      const client = createMockClient({
        chunkCount: 50,
        tunedFails: true,
        basicFails: 'string error',
      });

      const result = await buildAnnIndex({ client });

      expect(result.build_status).toBe('error');
      expect(result.build_error).toBe('string error');
    });

    it('passes whereClause into tuned SQL', async () => {
      const { buildAnnIndex } = await import('./ann-index.mjs');
      const client = createMockClient({ chunkCount: 100 });

      await buildAnnIndex({
        client,
        whereClause: "d.doc_family = 'src'",
      });

      expect(
        client.calls.some((sql) => sql.includes("WHERE d.doc_family = 'src'")),
      ).toBe(true);
    });

    it('passes whereClause into basic SQL on fallback', async () => {
      const { buildAnnIndex } = await import('./ann-index.mjs');
      const client = createMockClient({
        chunkCount: 100,
        tunedFails: true,
      });

      await buildAnnIndex({
        client,
        whereClause: "d.doc_family = 'test'",
      });

      expect(
        client.calls.some(
          (sql) =>
            sql.includes('libsql_vector_idx') &&
            sql.includes("WHERE d.doc_family = 'test'") &&
            !sql.includes('WITH KEY'),
        ),
      ).toBe(true);
    });

    it('uses partialFilter alias when whereClause is not provided', async () => {
      const { buildAnnIndex } = await import('./ann-index.mjs');
      const client = createMockClient({ chunkCount: 100 });

      await buildAnnIndex({
        client,
        partialFilter: "d.doc_family = 'aliased'",
      });

      expect(
        client.calls.some((sql) =>
          sql.includes("WHERE d.doc_family = 'aliased'"),
        ),
      ).toBe(true);
    });

    it('uses custom modelId and dimension in index_id', async () => {
      const { buildAnnIndex } = await import('./ann-index.mjs');
      const client = createMockClient({ chunkCount: 100 });

      const result = await buildAnnIndex({
        client,
        modelId: 'custom-model',
        dimension: 768,
      });

      expect(result.index_id).toBe('diskann_custom-model_768');
    });

    it('defaults modelId to all-MiniLM-L6-v2 and dimension to 384', async () => {
      const { buildAnnIndex } = await import('./ann-index.mjs');
      const client = createMockClient({ chunkCount: 100 });

      const result = await buildAnnIndex({ client });

      expect(result.index_id).toContain('all-MiniLM-L6-v2');
      expect(result.index_id).toContain('384');
    });

    it('passes forceStrategy to resolveDenseStrategy (still returns diskann)', async () => {
      const { buildAnnIndex } = await import('./ann-index.mjs');
      const client = createMockClient({ chunkCount: 100 });

      const result = await buildAnnIndex({
        client,
        forceStrategy: 'diskann',
      });

      expect(result.strategy).toBe('diskann');
    });

    it('handles chunkCount of 0 from empty database', async () => {
      const { buildAnnIndex } = await import('./ann-index.mjs');
      const client = createMockClient({ chunkCount: 0 });

      const result = await buildAnnIndex({ client });

      expect(result.current_elements).toBe(0);
      expect(result.build_status).toBe('ready');
    });

    it('handles missing count row (nullish coalescing to 0)', async () => {
      const { buildAnnIndex } = await import('./ann-index.mjs');
      const client = {
        calls: [],
        async execute(sqlOrObj) {
          const sql =
            typeof sqlOrObj === 'string' ? sqlOrObj : sqlOrObj.sql;
          this.calls.push(sql);
          if (sql.includes('SELECT COUNT(*)')) {
            return { rows: [] };
          }
          return { rows: [] };
        },
      };

      const result = await buildAnnIndex({ client });

      expect(result.current_elements).toBe(0);
    });
  });
});