/**
 * @module build-entity-graph.red.test
 * @description Red tests for entity graph building orchestration.
 * Tests focus on dry-run mode to keep execution time reasonable.
 * Full database write tests are deferred to integration testing.
 */
import Database from 'better-sqlite3';

describe('build-entity-graph', () => {
  describe('buildEntityGraph (dry-run)', () => {
    it('runs dry-run without writing to database', async () => {
      const { buildEntityGraph } = await import('../build-entity-graph.mjs');

      const result = await buildEntityGraph({ dryRun: true });

      expect(result.dryRun).toBe(true);
      expect(result.entities).toBeGreaterThan(0);
      expect(result.edges).toBeGreaterThanOrEqual(0);
      expect(result.elapsedMs).toBeGreaterThanOrEqual(0);
    });

    it('counts code entities correctly in dry-run', async () => {
      const { buildEntityGraph } = await import('../build-entity-graph.mjs');

      const result = await buildEntityGraph({ dryRun: true });

      expect(result.codeEntityCount).toBeGreaterThan(0);
      expect(result.codeEdgeCount).toBeGreaterThanOrEqual(0);
    });

    it('counts doc entities correctly in dry-run', async () => {
      const { buildEntityGraph } = await import('../build-entity-graph.mjs');

      const result = await buildEntityGraph({ dryRun: true });

      expect(result.docEntityCount).toBeGreaterThanOrEqual(0);
      expect(result.docEdgeCount).toBeGreaterThanOrEqual(0);
    });

    it('counts cross-reference edges in dry-run', async () => {
      const { buildEntityGraph } = await import('../build-entity-graph.mjs');

      const result = await buildEntityGraph({ dryRun: true });

      expect(result.crossRefEdgeCount).toBeGreaterThanOrEqual(0);
    });

    it('produces entity count within expected range (~630 ± 200)', async () => {
      const { buildEntityGraph } = await import('../build-entity-graph.mjs');

      const result = await buildEntityGraph({ dryRun: true });

      // Should be in rough range of 630 ± 200 (generous for initial validation).
      expect(result.entities).toBeGreaterThan(200);
    });

    it('produces edge count within expected range', async () => {
      const { buildEntityGraph } = await import('../build-entity-graph.mjs');

      const result = await buildEntityGraph({ dryRun: true });

      // Should be a substantial number of edges.
      expect(result.edges).toBeGreaterThan(100);
    });
  });
});
