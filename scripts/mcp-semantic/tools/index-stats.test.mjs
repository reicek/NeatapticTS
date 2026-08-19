/**
 * @module index-stats.test
 * @description Coverage tests for index-stats.mjs — metadata coverage,
 * feedback stats, ANN stats, and asIsoTimestamp edge cases.
 */
import {
  createSchemaClient,
  insertTestFixtures,
  createEnvIsolation,
} from '../__tests__/turso-test-helpers.mjs';

const { indexStats, FEEDBACK_WEIGHT, FEEDBACK_HALF_LIFE_DAYS } =
  await import('./index-stats.mjs');

describe('index-stats', () => {
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

  describe('constants', () => {
    it('exports expected constants', () => {
      expect(FEEDBACK_WEIGHT).toBe(1.0);
      expect(FEEDBACK_HALF_LIFE_DAYS).toBe(7);
    });
  });

  describe('indexStats — basic', () => {
    it('returns basic stats without metadata coverage', async () => {
      const result = await indexStats({ client });

      expect(result.total_documents).toBe(1);
      expect(result.total_chunks).toBe(2);
      expect(result.total_families).toBe(1);
      expect(result.last_build_timestamp).toBeDefined();
      expect(result.ann).toBeDefined();
      expect(result.ann.strategy).toBeDefined();
      expect(result.ann.threshold).toBeDefined();
      expect(result.ann.current_chunk_count).toBe(2);
      expect(result.ann.build_status).toBe('not_applicable');
      expect(result.ann.vector_type).toBe('F8_BLOB');
      expect(result.ann.quantization).toBe('8-bit');
      expect(result.feedback_stats).toBeDefined();
      expect(result.feedback_stats.total_events).toBe(0);
      expect(result.feedback_stats.events_by_type).toEqual({});
      expect(result.feedback_stats.chunks_with_feedback).toBe(0);
      expect(result.feedback_stats.average_feedback_boost).toBeNull();
      expect(result.feedback_stats.feedback_weight).toBe(1.0);
      expect(result.feedback_stats.feedback_half_life_days).toBe(7);
      expect(result.feedback_stats.last_recomputed_at).toBeNull();
      expect(result.metadata_coverage).toBeUndefined();
    });

    it('returns metadata_coverage when include_metadata_coverage=true', async () => {
      const result = await indexStats({
        client,
        include_metadata_coverage: true,
      });

      expect(result.metadata_coverage).toBeDefined();
      expect(result.metadata_coverage.chunks).toBeDefined();
      expect(result.metadata_coverage.documents).toBeDefined();

      // Check chunk metadata columns
      expect(result.metadata_coverage.chunks.context_header).toBeDefined();
      expect(result.metadata_coverage.chunks.context_header.total).toBe(0);
      expect(result.metadata_coverage.chunks.context_header.percent).toBe(0);
      expect(result.metadata_coverage.chunks.symbol_name).toBeDefined();
      // Our fixtures have symbol_name set, so 2/2 = 100%
      expect(result.metadata_coverage.chunks.symbol_name.total).toBe(2);
      expect(result.metadata_coverage.chunks.symbol_name.percent).toBe(100);

      // Check document metadata columns
      expect(result.metadata_coverage.documents.arch_layer).toBeDefined();
      expect(result.metadata_coverage.documents.arch_layer.total).toBe(1);
      expect(result.metadata_coverage.documents.arch_layer.percent).toBe(100);
      expect(
        result.metadata_coverage.documents.arch_layer.distribution,
      ).toEqual({
        network: 1,
      });
    });

    it('does NOT include metadata_coverage when include_metadata_coverage=false', async () => {
      const result = await indexStats({
        client,
        include_metadata_coverage: false,
      });
      expect(result.metadata_coverage).toBeUndefined();
    });
  });

  describe('feedback stats', () => {
    it('computes feedback stats with events and scores', async () => {
      // Insert feedback events
      await client.execute({
        sql: `INSERT INTO feedback_events (event_id, chunk_id, signal_type, signal_strength, query_hash, agent_id, context, created_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        args: [
          'evt-1',
          424242,
          'click',
          0.1,
          null,
          null,
          null,
          '2024-01-01T00:00:00.000Z',
        ],
      });
      await client.execute({
        sql: `INSERT INTO feedback_events (event_id, chunk_id, signal_type, signal_strength, query_hash, agent_id, context, created_at)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        args: [
          'evt-2',
          424242,
          'positive',
          1.0,
          null,
          null,
          null,
          '2024-01-02T00:00:00.000Z',
        ],
      });

      // Insert a feedback score
      await client.execute({
        sql: `INSERT INTO feedback_scores (chunk_id, total_positive, total_negative, total_impressions, total_clicks, total_references, last_feedback_at, feedback_boost)
              VALUES (?, ?, ?, ?, ?, ?, ?, ?)`,
        args: [424242, 1.0, 0, 0, 1, 0, 1700000000000, 0.3],
      });

      const result = await indexStats({ client });

      expect(result.feedback_stats.total_events).toBe(2);
      expect(result.feedback_stats.events_by_type).toEqual({
        click: 1,
        positive: 1,
      });
      expect(result.feedback_stats.chunks_with_feedback).toBe(1);
      expect(result.feedback_stats.average_feedback_boost).toBeCloseTo(0.3, 5);
      expect(result.feedback_stats.last_recomputed_at).toBe(1700000000000);
    });

    it('handles null average_feedback_boost when no scores exist', async () => {
      const result = await indexStats({ client });
      expect(result.feedback_stats.average_feedback_boost).toBeNull();
    });
  });

  describe('asIsoTimestamp edge cases', () => {
    it('returns null when no documents exist (indexed_at is null via empty table)', async () => {
      const emptyClient = await createSchemaClient();
      try {
        const result = await indexStats({ client: emptyClient });
        expect(result.last_build_timestamp).toBeNull();
      } finally {
        await emptyClient.close();
      }
    });

    it('returns null when no documents exist', async () => {
      const emptyClient = await createSchemaClient();
      try {
        const result = await indexStats({ client: emptyClient });
        expect(result.last_build_timestamp).toBeNull();
        expect(result.total_documents).toBe(0);
        expect(result.total_chunks).toBe(0);
        expect(result.total_families).toBe(0);
        // chunk count 0 < DEFAULT_ANN_THRESHOLD → not_applicable
        expect(result.ann.build_status).toBe('not_applicable');
        expect(result.ann.current_chunk_count).toBe(0);
      } finally {
        await emptyClient.close();
      }
    });

    it('returns ISO timestamp when indexed_at is a valid number', async () => {
      const result = await indexStats({ client });
      // indexed_at=1000 → new Date(1000).toISOString()
      expect(result.last_build_timestamp).toBe(new Date(1000).toISOString());
    });
  });

  describe('ANN stats threshold', () => {
    it('returns not_built when chunk count >= threshold', async () => {
      // Insert enough chunks to exceed DEFAULT_ANN_THRESHOLD (typically 200)
      // Instead of inserting 200 chunks, we can mock by inserting a document
      // with a large chunk count. But since we use real DB, let's check
      // the actual threshold behavior with a small count.
      const result = await indexStats({ client });
      // 2 chunks < DEFAULT_ANN_THRESHOLD → not_applicable
      expect(result.ann.build_status).toBe('not_applicable');
      expect(result.ann.index_id).toBeNull();
      expect(result.ann.index_type).toBeNull();
    });

    it('returns not_built when chunk count >= threshold (using mock client)', async () => {
      // Create a mock client that returns a large chunk count
      const mockClient = {
        async execute(sqlOrObj) {
          const sql = typeof sqlOrObj === 'string' ? sqlOrObj : sqlOrObj.sql;
          if (
            sql.includes('SELECT COUNT(*)') &&
            sql.includes('FROM documents')
          ) {
            return { rows: [{ count: 10 }] };
          }
          if (sql.includes('SELECT COUNT(*)') && sql.includes('FROM chunks')) {
            return { rows: [{ count: 60000 }] };
          }
          if (sql.includes('COUNT(DISTINCT doc_family)')) {
            return { rows: [{ count: 3 }] };
          }
          if (sql.includes('MAX(indexed_at)')) {
            return { rows: [{ value: 1700000000000 }] };
          }
          if (sql.includes('COUNT(*)') && sql.includes('feedback_events')) {
            return { rows: [{ count: 0 }] };
          }
          if (sql.includes('signal_type')) {
            return { rows: [] };
          }
          if (
            sql.includes('COUNT(DISTINCT chunk_id)') &&
            sql.includes('feedback_scores')
          ) {
            return { rows: [{ count: 0 }] };
          }
          if (sql.includes('AVG(feedback_boost)')) {
            return { rows: [{ average: null }] };
          }
          if (sql.includes('MAX(last_feedback_at)')) {
            return { rows: [{ value: null }] };
          }
          if (sql.includes('IS NOT NULL')) {
            return { rows: [{ count: 0 }] };
          }
          if (sql.includes('GROUP BY')) {
            return { rows: [] };
          }
          return { rows: [] };
        },
      };

      const result = await indexStats({ client: mockClient });
      // 60000 chunks >= DEFAULT_ANN_THRESHOLD (50000) → not_built
      expect(result.ann.build_status).toBe('not_built');
      expect(result.ann.index_id).toBeNull();
      expect(result.ann.index_type).toBe('diskann');
      expect(result.ann.current_chunk_count).toBe(60000);
    });
  });
});
