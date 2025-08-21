/**
 * benchmark.memory.test.ts
 * ======================================
 * Dist-only (Phase 0) memory & performance harness used in tests to:
 *  - Construct synthetic networks at several target connection counts.
 *  - Measure build + forward timing (with light warm-ups on large sizes).
 *  - Capture memoryStats based estimates (bytes/connection, etc.).
 *  - Aggregate variance, perform IQR outlier filtering, and (Phase 2) auto escalation
 *    if Coefficient of Variation (CV%) exceeds target thresholds for large sizes.
 *  - Persist an evolving `benchmark.results.json` artifact (≤10 history entries) with
 *    provenance (git commit, dist bundle hash) and optional regression annotations.
 *
 * NOTE: This file intentionally lives under `test/` to leverage Jest's execution & coverage
 * while remaining framework-agnostic for future extraction into a standalone CLI harness.
 */
import { memoryStats } from '../../src/utils/memory';
import Network from '../../src/architecture/network';
import * as fs from 'fs';
import seedrandom from 'seedrandom';

/**
 * BaselineRecord
 * Represents a single synthetic build + forward measurement and associated
 * approximate memory stats for a given connection size.
 */
interface BaselineRecord {
  variant: string;
  size: number;
  buildMs: number;
  fwdAvgMs: number;
  fwdTotalMs: number;
  conn: number;
  nodes: number;
  estBytes: number;
  bytesPerConn: number;
  bytesPerNode: number;
}

/**
 * Build a synthetic feedforward network targeting (approximately) a desired connection count.
 * Strategy: start with input/output sizes derived from sqrt heuristic then prune random
 * connections until at or below target. This keeps structure generation fast and deterministic
 * enough for comparative timing.
 * @param targetConnections Desired number of connections (approximate upper bound).
 * @returns Object containing created Network instance and elapsed build time in ms.
 */
function buildSyntheticNetwork(
  targetConnections: number
): {
  net: Network;
  buildMs: number;
} {
  const start = performance.now?.() ?? Date.now();
  const inputs = Math.max(1, Math.floor(Math.sqrt(targetConnections)));
  const outputs = Math.max(1, Math.ceil(targetConnections / inputs));
  const net = new Network(inputs, outputs);
  while (net.connections.length > targetConnections) {
    const idx = Math.floor(Math.random() * net.connections.length);
    const c = net.connections[idx];
    (net as any).disconnect(c.from, c.to);
  }
  const end = performance.now?.() ?? Date.now();
  return { net, buildMs: end - start };
}

/**
 * Measure total & average forward pass runtime for a network over N iterations.
 * Random input vector generated once (uniform [0,1)).
 * @param net Network to activate.
 * @param iterations Number of forward passes to execute (default 5).
 * @returns Object with totalMs and avgMs fields.
 */
function measureForwardPass(net: Network, iterations = 5) {
  const inputLen = net.input;
  const vec = new Array(inputLen).fill(0).map(() => Math.random());
  const t0 = performance.now?.() ?? Date.now();
  for (let i = 0; i < iterations; i++) net.activate(vec);
  const t1 = performance.now?.() ?? Date.now();
  const totalMs = t1 - t0;
  return { totalMs, avgMs: totalMs / iterations };
}

const baselineRecords: BaselineRecord[] = [];
const warnings: any[] = [];
let distAggregated: any[] = [];
// Collected per‑run measurements (build + forward + memory) prior to aggregation.
// Declared here (before tests) so inner describe blocks can push safely.
const rawMeasurements: { size: number; metrics: Record<string, number> }[] = [];

describe('benchmark.memory dist-only', () => {
  describe('memoryStats snapshot', () => {
    const snap = memoryStats();
    it('has numeric estimatedTotalBytes', () => {
      expect(typeof snap.estimatedTotalBytes).toBe('number');
    });
  });

  describe('synthetic baseline capture', () => {
    const sizes = [1000, 10000, 50000, 100000, 200000];
    for (const size of sizes) {
      const { net, buildMs } = buildSyntheticNetwork(size);
      const iters =
        size >= 200000 ? 1 : size >= 100000 ? 2 : size >= 50000 ? 3 : 5;
      const { totalMs: fwdTotalMs, avgMs: fwdAvgMs } = measureForwardPass(
        net,
        iters
      );
      const mem = memoryStats(net);
      baselineRecords.push({
        variant: 'dist',
        size,
        buildMs: +buildMs.toFixed(3),
        fwdAvgMs: +fwdAvgMs.toFixed(4),
        fwdTotalMs: +fwdTotalMs.toFixed(3),
        conn: net.connections.length,
        nodes: net.nodes.length,
        estBytes: mem.estimatedTotalBytes,
        bytesPerConn: mem.bytesPerConnection,
        bytesPerNode: mem.nodes
          ? Math.round(mem.estimatedTotalBytes / Math.max(1, mem.nodes))
          : 0,
      });
    }
    for (const size of sizes)
      it(`records connections size=${size}`, () => {
        expect(
          baselineRecords.find((r) => r.size === size)!.conn
        ).toBeGreaterThan(0);
      });
  });

  describe('dist aggregation', () => {
    const sizes = [1000, 10000, 50000, 100000, 200000];
    const BENCH_REPEAT_LARGE = (() => {
      const raw = process.env.BENCH_REPEAT_LARGE || '';
      const n = parseInt(raw, 10);
      return Number.isFinite(n) && n > 1 ? Math.min(n, 25) : 0;
    })();
    const repeatLarge = BENCH_REPEAT_LARGE > 1;
    // NOTE: We intentionally reuse the top-level rawMeasurements (declared earlier) so that
    // variance escalation additions append to the same collection. Shadowing removed.
    for (const size of sizes) {
      const repeats = repeatLarge && size >= 100000 ? BENCH_REPEAT_LARGE : 1;
      for (let rep = 0; rep < repeats; rep++) {
        (seedrandom as any)(`size:${size}|rep:${rep}`, { global: true });
        const inp = Math.max(1, Math.floor(Math.sqrt(size)));
        const out = Math.max(1, Math.ceil(size / inp));
        if (size >= 100000) {
          const warm = new Network(inp, out);
          measureForwardPass(warm, 1);
        }
        const t0 = performance.now?.() ?? Date.now();
        const net = new Network(inp, out);
        const t1 = performance.now?.() ?? Date.now();
        const buildMs = t1 - t0;
        if (size >= 100000) measureForwardPass(net, 1);
        const iterations =
          size >= 200000 ? 3 : size >= 100000 ? 5 : size >= 50000 ? 4 : 5;
        const { totalMs: fwdTotalMs, avgMs: fwdAvgMs } = measureForwardPass(
          net,
          iterations
        );
        const mem = memoryStats(net);
        let heapUsed: number | undefined;
        let rss: number | undefined;
        try {
          const mu = (process as any).memoryUsage?.();
          if (mu) {
            heapUsed = mu.heapUsed;
            rss = mu.rss;
          }
        } catch {}
        rawMeasurements.push({
          size,
          metrics: {
            buildMs,
            fwdAvgMs,
            fwdTotalMs,
            bytesPerConn: mem.bytesPerConnection,
            heapUsed: heapUsed ?? NaN,
            rss: rss ?? NaN,
            fwdIterations: iterations,
          },
        });
      }
    }
    const {
      aggregateBenchMeasurements,
    } = require('./benchmark.report.test') as typeof import('./benchmark.report.test');
    distAggregated = aggregateBenchMeasurements(
      rawMeasurements.map((r) => ({
        mode: 'dist',
        scenario: 'buildForward',
        size: r.size,
        metrics: r.metrics,
      })) as any
    );
    it('aggregated has entries', () => {
      expect(distAggregated.length).toBeGreaterThan(0);
    });

    // Variance (large sizes only)
    /**
     * Arithmetic mean helper.
     * @param a Array of numbers.
     * @returns Mean (0 if array empty).
     */
    function mean(a: number[]) {
      return a.reduce((s, x) => s + x, 0) / (a.length || 1);
    }
    /**
     * Unbiased sample standard deviation (n-1 denominator) with guard for n<2.
     * @param a Array of numbers.
     * @returns Standard deviation (0 if insufficient samples).
     */
    function std(a: number[]) {
      if (a.length < 2) return 0;
      const m = mean(a);
      return Math.sqrt(
        a.reduce((s, x) => s + (x - m) * (x - m), 0) / (a.length - 1)
      );
    }
    /**
     * IQR (Interquartile Range) based mild outlier filter (1.5*I rule).
     * Returns original array if filtering would leave <3 points or no outliers found.
     * @param vals Raw numeric samples.
     * @returns Object with filtered (possibly original) and outliers arrays.
     */
    function iqrFilter(vals: number[]) {
      if (vals.length < 4) return { filtered: vals.slice(), outliers: [] };
      const s = vals.toSorted((a, b) => a - b);
      const q = (p: number) => {
        const i = (s.length - 1) * p;
        const lo = Math.floor(i),
          hi = Math.ceil(i);
        return lo === hi ? s[lo] : s[lo] + (s[hi] - s[lo]) * (i - lo);
      };
      const q1 = q(0.25),
        q3 = q(0.75),
        I = q3 - q1,
        lo = q1 - 1.5 * I,
        hi = q3 + 1.5 * I;
      const f: number[] = [],
        o: number[] = [];
      for (const v of vals) (v < lo || v > hi ? o : f).push(v);
      if (!o.length || f.length < 3)
        return { filtered: vals.slice(), outliers: [] };
      return { filtered: f, outliers: o };
    }
    /**
     * Recompute per-size variance statistics (mean, std, CV%) after outlier filtering.
     * @param g Aggregated group entry containing size & count.
     * @returns Variance summary object.
     */
    function recomputeVariance(g: any) {
      const raw = rawMeasurements.filter((r) => r.size === g.size);
      const build = raw.map((r) => r.metrics.buildMs);
      const fwd = raw.map((r) => r.metrics.fwdAvgMs);
      const { filtered: bf } = iqrFilter(build);
      const { filtered: ff } = iqrFilter(fwd);
      const bm = mean(bf),
        bs = std(bf),
        fm = mean(ff),
        fs = std(ff);
      return {
        mode: 'dist',
        size: g.size,
        samples: g.count || raw.length,
        buildMsMean: +bm.toFixed(4),
        buildMsStd: +bs.toFixed(4),
        buildMsCvPct: bm ? +((bs / bm) * 100).toFixed(2) : 0,
        fwdAvgMsMean: +fm.toFixed(4),
        fwdAvgMsStd: +fs.toFixed(4),
        fwdAvgMsCvPct: fm ? +((fs / fm) * 100).toFixed(2) : 0,
      };
    }
    // --- Variance auto escalation (Phase 2 -> active implementation) ---
    // If CV% for large sizes (>=100k) exceeds target, incrementally add repeats (up to cap)
    // to stabilize variance before persisting final artifact. Each escalation recorded.
    const targetCvPct = 7; // phase target (can tighten in future phases)
    const maxVarianceRepeats = 9; // hard cap to control runtime explosion
    const escalateRecords: any[] = [];

    /**
     * Execute an additional measurement repeat for a given large size to attempt variance reduction.
     * Duplicates core measurement logic used in initial pass to keep comparable conditions.
     * @param size Connection size bucket.
     * @param rep Repeat index (used for deterministic seeding).
     */
    function runAdditionalRepeat(size: number, rep: number) {
      // Mirrors logic in initial measurement loop for large sizes
      (seedrandom as any)(`size:${size}|rep:${rep}`, { global: true });
      const inp = Math.max(1, Math.floor(Math.sqrt(size)));
      const out = Math.max(1, Math.ceil(size / inp));
      if (size >= 100000) {
        const warm = new Network(inp, out);
        measureForwardPass(warm, 1);
      }
      const t0 = performance.now?.() ?? Date.now();
      const net = new Network(inp, out);
      const t1 = performance.now?.() ?? Date.now();
      const buildMs = t1 - t0;
      if (size >= 100000) measureForwardPass(net, 1);
      const iterations = size >= 200000 ? 3 : 5; // mirrors original logic for large sizes
      const { totalMs: fwdTotalMs, avgMs: fwdAvgMs } = measureForwardPass(
        net,
        iterations
      );
      const mem = memoryStats(net);
      let heapUsed: number | undefined;
      let rss: number | undefined;
      try {
        const mu = (process as any).memoryUsage?.();
        if (mu) {
          heapUsed = mu.heapUsed;
          rss = mu.rss;
        }
      } catch {}
      rawMeasurements.push({
        size,
        metrics: {
          buildMs,
          fwdAvgMs,
          fwdTotalMs,
          bytesPerConn: mem.bytesPerConnection,
          heapUsed: heapUsed ?? NaN,
          rss: rss ?? NaN,
          fwdIterations: iterations,
        },
      });
    }

    /**
     * Compute simple variance (CV%) metrics for a size without IQR filtering (used to drive escalation loop).
     * @param size Connection size bucket.
     * @returns Variance entry or null if insufficient samples.
     */
    function computeVarianceForSize(size: number) {
      const raw = rawMeasurements.filter((r) => r.size === size);
      if (raw.length < 2) return null;
      const build = raw.map((r) => r.metrics.buildMs);
      const fwd = raw.map((r) => r.metrics.fwdAvgMs);
      const m = (a: number[]) => a.reduce((s, x) => s + x, 0) / (a.length || 1);
      const sdev = (a: number[]) => {
        if (a.length < 2) return 0;
        const mm = m(a);
        return Math.sqrt(
          a.reduce((s, x) => s + (x - mm) * (x - mm), 0) / (a.length - 1)
        );
      };
      const bm = m(build),
        bs = sdev(build);
      const fm = m(fwd),
        fs = sdev(fwd);
      return {
        size,
        samples: raw.length,
        buildMsCvPct: bm ? +((bs / bm) * 100).toFixed(2) : 0,
        fwdAvgMsCvPct: fm ? +((fs / fm) * 100).toFixed(2) : 0,
      };
    }

    // Escalate per monitored large size
    for (const size of [100000, 200000]) {
      let varianceEntry = computeVarianceForSize(size);
      // Only attempt escalation if we already have at least 2 samples (repeat mode active)
      while (
        varianceEntry &&
        (varianceEntry.buildMsCvPct > targetCvPct ||
          varianceEntry.fwdAvgMsCvPct > targetCvPct) &&
        varianceEntry.samples < maxVarianceRepeats
      ) {
        escalateRecords.push({
          size,
          buildCvPct: varianceEntry.buildMsCvPct,
          fwdCvPct: varianceEntry.fwdAvgMsCvPct,
          targetCvPct,
          repeatsObserved: varianceEntry.samples,
          action: 'escalate',
          reason: 'cv-above-threshold',
          timestamp: new Date().toISOString(),
        });
        runAdditionalRepeat(size, varianceEntry.samples);
        varianceEntry = computeVarianceForSize(size);
      }
      if (varianceEntry && varianceEntry.samples >= 2) {
        escalateRecords.push({
          size,
          buildCvPct: varianceEntry.buildMsCvPct,
          fwdCvPct: varianceEntry.fwdAvgMsCvPct,
          targetCvPct,
          repeatsObserved: varianceEntry.samples,
          action: 'stop',
          reason:
            varianceEntry.buildMsCvPct <= targetCvPct &&
            varianceEntry.fwdAvgMsCvPct <= targetCvPct
              ? 'below-threshold'
              : 'max-repeats',
          timestamp: new Date().toISOString(),
        });
      }
    }

    // Recompute aggregation & final variance summary after escalations
    distAggregated = aggregateBenchMeasurements(
      rawMeasurements.map((r) => ({
        mode: 'dist',
        scenario: 'buildForward',
        size: r.size,
        metrics: r.metrics,
      })) as any
    );
    const varianceSummary = distAggregated
      .filter((g) => g.size >= 100000 && g.count > 1)
      .map(recomputeVariance);

    // Persist results
    try {
      const path = require('path');
      const resultsFile = path.resolve(__dirname, 'benchmark.results.json');
      let existing: any;
      if (fs.existsSync(resultsFile)) {
        try {
          existing = JSON.parse(fs.readFileSync(resultsFile, 'utf-8'));
        } catch {}
      }
      let commit: string | undefined;
      try {
        const cp = require('child_process');
        commit = cp
          .execSync('git rev-parse --short HEAD', {
            stdio: ['ignore', 'pipe', 'ignore'],
          })
          .toString()
          .trim();
      } catch {}
      /**
       * Strip aggregation objects down to the compact summary persisted in history snapshots.
       * @param ag Aggregated measurement groups.
       * @returns Array of simplified objects (size + key means).
       */
      function summarize(ag: any[]) {
        return ag.map((g) => ({
          size: g.size,
          buildMsMean: g.buildMsMean,
          fwdAvgMsMean: g.fwdAvgMsMean,
          bytesPerConnMean: g.bytesPerConnMean,
        }));
      }
      const snapshot = {
        generatedAt: new Date().toISOString(),
        commit,
        sizes: distAggregated
          .map((g) => g.size)
          .sort((a: number, b: number) => a - b),
        summary: summarize(distAggregated),
        fwdDeltaPct: [],
        fieldAuditCounts:
          existing && existing.fieldAudit
            ? {
                Node: existing.fieldAudit.Node?.count,
                Connection: existing.fieldAudit.Connection?.count,
              }
            : undefined,
      };
      const payload: any = existing || {};
      payload.generatedAt = snapshot.generatedAt;
      payload.baseline = baselineRecords;
      payload.variantRaw = rawMeasurements;
      payload.aggregated = distAggregated;
      payload.deltas = [];
      payload.warnings = warnings;
      // --- dist bundle provenance (Phase 1) ---
      // We capture existence, byte size, and a short sha256 hash of the built dist bundle
      // to aid historical comparison & reproducibility when only dist metrics are tracked.
      let distBundleMeta: { exists: boolean; bytes?: number; hash?: string } = {
        exists: false,
      };
      try {
        const distPath = path.resolve(__dirname, '../../dist/neataptic.js');
        if (fs.existsSync(distPath)) {
          const stat = fs.statSync(distPath);
          // Hash only when file reasonably small (<25MB) to avoid CI slowdown (practical upper bound here is tiny)
          const data = fs.readFileSync(distPath);
          const crypto = require('crypto');
          const hash = crypto
            .createHash('sha256')
            .update(data)
            .digest('hex')
            .slice(0, 12);
          distBundleMeta = { exists: true, bytes: stat.size, hash };
        }
      } catch {}
      // --- optional forward regression annotation (informational, non-failing) ---
      // Criteria: we need at least 2 historical snapshots to compute a rolling median, and variance CV below threshold.
      const regressionAnnotations: any[] = [];
      try {
        if (Array.isArray(existing?.history) && existing.history.length) {
          // Build map size -> rolling median fwdAvgMsMean from history summary entries
          const history = existing.history.slice(-10);
          const sizeToSamples: Record<string, number[]> = {};
          for (const snap of history) {
            for (const s of snap.summary || []) {
              (sizeToSamples[s.size] ||= []).push(s.fwdAvgMsMean);
            }
          }
          const median = (arr: number[]) => {
            const a = arr.toSorted((x, y) => x - y);
            const n = a.length;
            if (!n) return NaN;
            return a[Math.floor((n - 1) / 2)];
          };
          const sizeMedian: Record<number, number> = {};
          for (const [k, vals] of Object.entries(sizeToSamples))
            if (vals.length >= 2) sizeMedian[+k] = median(vals);
          // Compute CV map from current variance summary for gating
          const cvMap: Record<number, number> = {};
          for (const v of varianceSummary || [])
            cvMap[v.size] = v.fwdAvgMsCvPct;
          const CV_THRESHOLD = 7; // %
          const DELTA_THRESHOLD = 10; // % deviation from median to annotate
          for (const g of distAggregated) {
            const sz = g.size;
            const m = sizeMedian[sz];
            if (!m || !Number.isFinite(m)) continue;
            const current = (g as any).fwdAvgMsMean;
            if (!Number.isFinite(current) || !current) continue;
            const deltaPct = ((current - m) / m) * 100;
            const cv = cvMap[sz];
            if (cv && cv <= CV_THRESHOLD && deltaPct > DELTA_THRESHOLD) {
              regressionAnnotations.push({
                type: 'fwd-regression',
                size: sz,
                deltaPct: +deltaPct.toFixed(2),
                thresholdPct: DELTA_THRESHOLD,
                cvPct: cv,
                cvThresholdPct: CV_THRESHOLD,
                note:
                  'Informational only: not failing. Rolling median baseline.',
              });
            }
          }
          if (regressionAnnotations.length)
            payload.regressionAnnotations = regressionAnnotations;
        }
      } catch {}
      payload.meta = {
        note: 'Dist-only benchmark results file.',
        varianceRepeatsLarge: BENCH_REPEAT_LARGE,
        warmupDiscard: true,
        deterministicSeeding: true,
        outlierFilter: 'IQR1.5',
        filteredVariance: true,
        distBundle: distBundleMeta,
      };
      // Attach variance escalation records & cap metadata
      (payload.meta as any).maxVarianceRepeats = maxVarianceRepeats;
      const priorEsc = (existing?.meta?.varianceAutoEscalations || []) as any[];
      (payload.meta as any).varianceAutoEscalations = priorEsc.concat(
        escalateRecords
      );
      if (varianceSummary.length) payload.variance = varianceSummary;
      const hist = Array.isArray(payload.history) ? payload.history : [];
      hist.push(snapshot);
      payload.history = hist.slice(-10);
      fs.writeFileSync(resultsFile, JSON.stringify(payload, null, 2), 'utf-8');
    } catch {}

    it('bytesPerConnMean positive (size 1000)', () => {
      const g = distAggregated.find((g) => g.size === 1000);
      expect(g.bytesPerConnMean > 0).toBe(true);
    });
  });
});
