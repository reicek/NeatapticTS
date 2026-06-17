/**
 * @module eval-baseline.red.test
 * @description Red tests for the RAG evaluation baseline module.
 *
 * The baseline module stores reference measurements, detects regressions,
 * and compares current eval runs against stored baselines for Step 22.
 */

const BASELINE_PATH = '../../semantic-index/eval-baseline.mjs';

function loadBaseline() {
  return import(BASELINE_PATH);
}

function makeMetrics(overrides = {}) {
  return {
    mrr_at_1: 0.1,
    mrr_at_3: 0.2,
    mrr_at_5: 0.3,
    mrr_at_10: 0.35,
    ndcg_at_5: 0.4,
    ndcg_at_10: 0.42,
    recall_at_5: 0.45,
    recall_at_10: 0.5,
    recall_at_20: 0.55,
    latency_ms: { p50: 50, p95: 100, max: 200 },
    ...overrides,
  };
}

describe('eval-baseline', () => {
  describe('baseline measurement storage', () => {
    it('stores a baseline measurement', async () => {
      const { storeBaseline } = await loadBaseline();
      const baseline = {
        baseline_id: 'baseline-test',
        timestamp: new Date().toISOString(),
        conditions: {
          hybrid: makeMetrics(),
        },
      };
      const stored = await storeBaseline(baseline);
      expect(stored).toEqual(
        expect.objectContaining({
          baseline_id: 'baseline-test',
          timestamp: expect.any(String),
        }),
      );
    });

    it('persists baseline to a configurable path', async () => {
      const { storeBaseline } = await loadBaseline();
      const baseline = {
        baseline_id: 'baseline-path-test',
        timestamp: new Date().toISOString(),
        conditions: {},
      };
      const stored = await storeBaseline(baseline, {
        path: 'data/eval-baselines/test-baseline.json',
      });
      expect(stored.baseline_id).toBe('baseline-path-test');
    });

    it('loads a stored baseline by id', async () => {
      const { storeBaseline, loadBaseline: loadById } = await loadBaseline();
      const baseline = {
        baseline_id: 'baseline-load-test',
        timestamp: new Date().toISOString(),
        conditions: {},
      };
      await storeBaseline(baseline);
      const loaded = await loadById('baseline-load-test');
      expect(loaded.baseline_id).toBe('baseline-load-test');
    });
  });

  describe('regression detection', () => {
    it('detects MRR@5 FAIL threshold breach', async () => {
      const { compareToBaseline } = await loadBaseline();
      const current = {
        condition: 'hybrid',
        metrics: makeMetrics({ mrr_at_5: 0.2 }),
      };
      const baseline = {
        condition: 'hybrid',
        metrics: makeMetrics({ mrr_at_5: 0.32 }),
      };
      const report = compareToBaseline(current, baseline, {
        failThresholds: { mrr_at_5: 0.01 },
      });
      expect(report.pass).toBe(false);
      expect(report.metrics.mrr_at_5.status).toBe('FAIL');
    });

    it('passes when MRR@5 is within threshold', async () => {
      const { compareToBaseline } = await loadBaseline();
      const current = {
        condition: 'hybrid',
        metrics: makeMetrics({ mrr_at_5: 0.33 }),
      };
      const baseline = {
        condition: 'hybrid',
        metrics: makeMetrics({ mrr_at_5: 0.32 }),
      };
      const report = compareToBaseline(current, baseline, {
        failThresholds: { mrr_at_5: 0.01 },
      });
      expect(report.pass).toBe(true);
      expect(report.metrics.mrr_at_5.status).toBe('PASS');
    });

    it('warns on nDCG regression below WARN threshold', async () => {
      const { compareToBaseline } = await loadBaseline();
      const current = {
        condition: 'hybrid',
        metrics: makeMetrics({ ndcg_at_5: 0.35 }),
      };
      const baseline = {
        condition: 'hybrid',
        metrics: makeMetrics({ ndcg_at_5: 0.4 }),
      };
      const report = compareToBaseline(current, baseline, {
        warnThresholds: { ndcg_at_5: 0.01 },
      });
      expect(report.metrics.ndcg_at_5.status).toBe('WARN');
    });

    it('warns on Recall regression below WARN threshold', async () => {
      const { compareToBaseline } = await loadBaseline();
      const current = {
        condition: 'hybrid',
        metrics: makeMetrics({ recall_at_5: 0.4 }),
      };
      const baseline = {
        condition: 'hybrid',
        metrics: makeMetrics({ recall_at_5: 0.46 }),
      };
      const report = compareToBaseline(current, baseline, {
        warnThresholds: { recall_at_5: 0.01 },
      });
      expect(report.metrics.recall_at_5.status).toBe('WARN');
    });

    it('reports latency regression as WARN', async () => {
      const { compareToBaseline } = await loadBaseline();
      const current = {
        condition: 'hybrid',
        metrics: makeMetrics({ latency_ms: { p50: 50, p95: 700, max: 1200 } }),
      };
      const baseline = {
        condition: 'hybrid',
        metrics: makeMetrics({ latency_ms: { p50: 50, p95: 100, max: 200 } }),
      };
      const report = compareToBaseline(current, baseline, {
        latencyWarnMs: 100,
      });
      expect(report.metrics.latency_p95.status).toBe('WARN');
    });
  });

  describe('baseline comparison status', () => {
    it('returns correct status for each metric', async () => {
      const { compareToBaseline } = await loadBaseline();
      const report = compareToBaseline(
        { condition: 'hybrid', metrics: makeMetrics() },
        { condition: 'hybrid', metrics: makeMetrics() },
        {
          failThresholds: { mrr_at_5: 0.01 },
          warnThresholds: { ndcg_at_5: 0.01, recall_at_5: 0.01 },
        },
      );
      expect(report).toEqual(
        expect.objectContaining({
          pass: expect.any(Boolean),
          metrics: expect.any(Object),
          regressions: expect.any(Array),
          warnings: expect.any(Array),
        }),
      );
    });

    it('aggregates multiple regressions', async () => {
      const { compareToBaseline } = await loadBaseline();
      const current = {
        condition: 'hybrid',
        metrics: makeMetrics({ mrr_at_5: 0.2, ndcg_at_5: 0.3 }),
      };
      const baseline = {
        condition: 'hybrid',
        metrics: makeMetrics({ mrr_at_5: 0.32, ndcg_at_5: 0.4 }),
      };
      const report = compareToBaseline(current, baseline, {
        failThresholds: { mrr_at_5: 0.01 },
        warnThresholds: { ndcg_at_5: 0.01 },
      });
      expect(report.regressions.length).toBeGreaterThan(0);
      expect(report.warnings.length).toBeGreaterThan(0);
    });

    it('rejects mismatched conditions', async () => {
      const { compareToBaseline } = await loadBaseline();
      expect(() =>
        compareToBaseline(
          { condition: 'hybrid', metrics: makeMetrics() },
          { condition: 'bm25_only', metrics: makeMetrics() },
        ),
      ).toThrow();
    });

    it('rejects missing baseline metrics', async () => {
      const { compareToBaseline } = await loadBaseline();
      expect(() =>
        compareToBaseline(
          { condition: 'hybrid', metrics: makeMetrics() },
          { condition: 'hybrid' },
        ),
      ).toThrow();
    });
  });
});
