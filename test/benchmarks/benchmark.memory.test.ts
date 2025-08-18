/**
 * benchmark.memory.ts
 *
 * Placeholder benchmark harness (Phase 0) for memory & performance related scenarios.
 * Intentionally lightweight: no heavy constructions yet, just API surface + structure.
 *
 * Decision: Replaced legacy CJS CLI scripts with test-driven TS modules for easier coverage,
 * type safety, and incremental build variant integration. Future: real benchmarks may
 * migrate reusable logic into `src/benchmarks/` while keeping verification here.
 */
import { memoryStats } from '../../src/utils/memory';
import Network from '../../src/architecture/network';
import * as fs from 'fs';

// ----------------------------------------------------------------------------------------------
// Variant Handling Helpers
// ----------------------------------------------------------------------------------------------

/**
 * Build a Network instance for a given variant mode.
 * Mode 'src' uses the in-memory TS class; 'dist' attempts to import the compiled artifact.
 * If the dist bundle is missing (e.g. developer forgot to run build), the function gracefully
 * falls back to the src implementation to keep tests green while still logging a warning.
 */
/**
 * Create a Network instance for a specific build variant while gracefully handling
 * absent or unloadable dist artifacts. This keeps the benchmark suite ergonomic:
 * developers can run tests without always producing a compiled `dist` build.
 *
 * Arrange: select target path & existence check.
 * Act: attempt dynamic import when mode==='dist'.
 * Assert (in tests): a Network instance is always returned and warnings recorded when falling back.
 */
export function buildNetworkForVariant(
  mode: 'src' | 'dist',
  input: number,
  output: number
): Network {
  if (mode === 'src') return new Network(input, output);
  // Attempt to dynamically import the built library (ESM).
  const distPath = require('path').resolve(
    __dirname,
    '../../..',
    'dist',
    'neataptic.js'
  );
  if (!fs.existsSync(distPath)) {
    warnings.push({ type: 'dist-missing', distPath });
    return new Network(input, output);
  }
  try {
    // eslint-disable-next-line @typescript-eslint/no-var-requires
    const distModule = require(distPath);
    const DistNetwork =
      distModule.Network || distModule.default?.Network || distModule.default;
    if (DistNetwork) return new DistNetwork(input, output);
  } catch (e) {
    warnings.push({ type: 'dist-import-failed', error: (e as Error).message });
  }
  return new Network(input, output);
}

/**
 * Draft interface describing a memory benchmark scenario configuration.
 */
export interface MemoryBenchmarkScenario {
  /** Scenario key: e.g. 'build', 'forward', 'evolution', 'morphogenesis' */
  scenario: string;
  /** Target synthetic connection count (or other size dimension) */
  size: number;
  /** Variant mode: source (ts) vs compiled (dist) */
  mode: 'src' | 'dist';
}

/**
 * Draft result shape for a single benchmark measurement.
 */
export interface MemoryBenchmarkResult {
  scenario: string;
  size: number;
  mode: 'src' | 'dist';
  timestamp: number;
  metrics: Record<string, number>;
}

// NOTE: Placeholder scenario utilities removed as synthetic builder now drives baseline metrics.

/**
 * Build a synthetic network approximating a target connection count using a dense bipartite
 * input→output construction (inputs * outputs). If we overshoot the target, we randomly prune
 * extra connections to match (Phase 0: simple heuristic, no hidden layers for speed).
 *
 * NOTE: This intentionally favours deterministic sizing over biological realism; later phases
 * may introduce hidden layering + morphology strategies.
 */
/**
 * Build a synthetic network targeting approximately `targetConnections` direct connections.
 * Uses a dense bipartite input→output layer; if we overshoot, random pruning restores the target.
 * This emphasizes deterministic sizing & speed over biological fidelity (future work may extend).
 */
export function buildSyntheticNetwork(
  targetConnections: number
): { net: Network; buildMs: number } {
  const start =
    typeof performance !== 'undefined' && performance.now
      ? performance.now()
      : Date.now();
  // Heuristic dense rectangle close to square
  const inputs = Math.max(1, Math.floor(Math.sqrt(targetConnections)));
  const outputs = Math.max(1, Math.ceil(targetConnections / inputs));
  const net = new Network(inputs, outputs);
  // Prune excess if overshoot (constructor fully connects inputs→outputs)
  const desired = targetConnections;
  while (net.connections.length > desired) {
    const idx = Math.floor(Math.random() * net.connections.length);
    const conn = net.connections[idx];
    // network has public disconnect via this.disconnect used internally; cast to any to appease TS if private
    (net as any).disconnect(conn.from, conn.to);
  }
  const end =
    typeof performance !== 'undefined' && performance.now
      ? performance.now()
      : Date.now();
  return { net, buildMs: end - start };
}

/**
 * Measure forward pass performance for a constructed network over N iterations.
 * Returns totalMs & avgMs for quick comparative logging.
 */
/**
 * Run repeated forward activations to approximate steady-state activation cost.
 * Returns total wall time and per-iteration average. Small iteration counts reduce
 * test runtime while still capturing relative differences across sizes.
 */
export function measureForwardPass(
  net: Network,
  iterations = 5
): { totalMs: number; avgMs: number } {
  const inputLen = net.input;
  const vec = new Array(inputLen).fill(0).map(() => Math.random());
  const t0 =
    typeof performance !== 'undefined' && performance.now
      ? performance.now()
      : Date.now();
  for (let i = 0; i < iterations; i++) {
    net.activate(vec);
  }
  const t1 =
    typeof performance !== 'undefined' && performance.now
      ? performance.now()
      : Date.now();
  const totalMs = t1 - t0;
  return { totalMs, avgMs: totalMs / iterations };
}

// Test scaffolding (single-expectation tests) ----------------------------------------------------

// ----------------------------------------------------------------------------------------------
// Result accumulation (instead of console logging)
// ----------------------------------------------------------------------------------------------
/**
 * Shape of a single baseline measurement captured during Phase 0 synthetic builds.
 */
interface BaselineRecord {
  mode: string;
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

/** Collected raw baseline measurement rows. */
const baselineRecords: BaselineRecord[] = [];
/** Non-fatal issues encountered (e.g., missing dist build). */
const warnings: Array<Record<string, any>> = [];
/** Aggregated variant statistics (populated once). */
let variantAggregated: any[] = [];
/** Delta rows between src & dist aggregated metrics. */
let variantDeltas: any[] = [];

describe('benchmark.memory baseline', () => {
  // ----------------------------------------
  // memoryStats heuristic snapshot scenario
  // ----------------------------------------
  describe('memoryStats() heuristic snapshot', () => {
    const snapshot = memoryStats(); // Arrange (captured once per describe scope)
    it('exposes numeric estimatedTotalBytes', () => {
      // Assert (single expectation)
      expect(typeof snapshot.estimatedTotalBytes).toBe('number');
    });
  });

  describe('synthetic baseline capture (Phase 0)', () => {
    const baselineSizes = [1000, 10000, 50000, 100000, 200000];
    // Arrange: build networks & record metrics once
  for (const size of baselineSizes) {
      const { net, buildMs } = buildSyntheticNetwork(size); // Act (build)
      const iterations = size >= 200000 ? 1 : size >= 100000 ? 2 : size >= 50000 ? 3 : 5; // Act (iteration heuristic)
      const { totalMs: fwdTotalMs, avgMs: fwdAvgMs } = measureForwardPass(net, iterations); // Act (forward)
      const mem = memoryStats(net);
      baselineRecords.push({
        mode: 'src',
        variant: 'unoptimized',
        size,
        buildMs: Number(buildMs.toFixed(3)),
        fwdAvgMs: Number(fwdAvgMs.toFixed(4)),
        fwdTotalMs: Number(fwdTotalMs.toFixed(3)),
        conn: net.connections.length,
        nodes: net.nodes.length,
        estBytes: mem.estimatedTotalBytes,
        bytesPerConn: mem.bytesPerConnection,
        bytesPerNode: mem.nodes ? Math.round(mem.estimatedTotalBytes / Math.max(1, mem.nodes)) : 0,
      });
    }
    // Separate assertion tests (one expectation each)
    for (const size of baselineSizes) {
      it(`records non-zero connections for size=${size}`, () => {
        const rec = baselineRecords.find(r => r.size === size)!; // Arrange
        expect(rec.conn).toBeGreaterThan(0); // Assert
      });
    }
    it('captures a record per baseline size', () => {
      const uniqueSizes = new Set(baselineRecords.map(r => r.size)); // Arrange
      expect(uniqueSizes.size).toBe(baselineSizes.length); // Assert
    });
  });

  // Additional unit coverage for helpers ------------------------------------------------------
  describe('helper functions', () => {
    it('buildSyntheticNetwork returns requested connection count or less', () => {
      const target = 1234; // Arrange
      const { net } = buildSyntheticNetwork(target); // Act
      // Assert
      expect(net.connections.length).toBeLessThanOrEqual(target);
    });
    it('measureForwardPass returns numeric avgMs', () => {
      const { net } = buildSyntheticNetwork(200); // Arrange
      const res = measureForwardPass(net, 2); // Act
      expect(typeof res.avgMs).toBe('number'); // Assert
    });
    it('buildNetworkForVariant falls back when dist missing', () => {
      const n = buildNetworkForVariant('dist', 2, 2); // Act
      expect(n instanceof Network).toBe(true); // Assert
    });
  });

  // --------------------------------------------------------------------------------------------
  // Variant Comparison (src vs dist) Aggregation (Phase 0 Extension)
  // --------------------------------------------------------------------------------------------
  describe('variant comparison aggregation', () => {
    /** Baseline sizes for variant delta (always include 1k; include 10k; larger sizes gated by BENCH_LARGE) */
    const variantSizes: number[] = [1000, 10000, 50000, 100000, 200000];
    const enableLarge = true; // large sizes always enabled for stronger statistical signal
    // Multi-sample variance mode: repeat large (>=100k) sizes N times when BENCH_REPEAT_LARGE=N>1
    const BENCH_REPEAT_LARGE = (() => {
      const raw = (process && process.env && process.env.BENCH_REPEAT_LARGE) || '';
      const n = parseInt(raw as string, 10);
      return Number.isFinite(n) && n > 1 ? Math.min(n, 25) : 0; // safety cap 25
    })();
    const repeatLarge = BENCH_REPEAT_LARGE > 1;
    /** Raw measurement records captured for aggregation */
    const rawMeasurements: {
      mode: 'src' | 'dist';
      scenario: string;
      size: number;
      metrics: Record<string, number>;
    }[] = [];
    /** Perform measurements once for all tests (Arrange + Act) */
    for (const size of variantSizes) {
      for (const mode of ['src', 'dist'] as const) {
        const repeats = repeatLarge && size >= 100000 ? BENCH_REPEAT_LARGE : 1;
        for (let rep = 0; rep < repeats; rep++) {
          // Build network using variant-aware constructor path.
          const tBuild0 =
            typeof performance !== 'undefined' && performance.now
              ? performance.now()
              : Date.now();
          const net = buildNetworkForVariant(
            mode,
            Math.max(1, Math.floor(Math.sqrt(size))),
            Math.max(
              1,
              Math.ceil(size / Math.max(1, Math.floor(Math.sqrt(size))))
            )
          );
          const tBuild1 =
            typeof performance !== 'undefined' && performance.now
              ? performance.now()
              : Date.now();
          const buildMs = tBuild1 - tBuild0;
          // Forward pass measurement (Act) with fixed small iteration count.
          const iterations =
            size >= 200000 ? 1 : size >= 100000 ? 2 : size >= 50000 ? 3 : 5;
          const { totalMs: fwdTotalMs, avgMs: fwdAvgMs } = measureForwardPass(
            net,
            iterations
          );
          const mem = memoryStats(net);
          // Capture Node heap metrics if available to advance Phase 0 Step 2 (synthetic builds memory usage).
          let heapUsed: number | undefined;
          let rss: number | undefined;
          try {
            if (typeof process !== 'undefined' && (process as any).memoryUsage) {
              const mu = (process as any).memoryUsage();
              heapUsed = mu.heapUsed;
              rss = mu.rss;
            }
          } catch {}
          rawMeasurements.push({
            mode,
            scenario: 'buildForward',
            size,
            metrics: {
              buildMs,
              fwdAvgMs,
              fwdTotalMs,
              bytesPerConn: mem.bytesPerConnection,
              heapUsed: heapUsed ?? NaN,
              rss: rss ?? NaN,
              // Optional metadata fields (ignored by aggregator except treat numeric) - rep index excluded to avoid inflating metric keys
            },
          });
        }
      }
    }

    // Defer import (to avoid circular top-level ordering concerns) only if aggregator needed.
    const {
      aggregateBenchMeasurements,
    } = require('./benchmark.report.test') as typeof import('./benchmark.report.test');
    /** Aggregated result groups (Act for aggregation) */
    const aggregated = aggregateBenchMeasurements(rawMeasurements as any);
    /** Convenience map keyed by mode for current size */
    const groupByMode: Record<string, any> = {};
    for (const g of aggregated)
      if (g.size === variantSizes[0]) groupByMode[g.mode] = g;

    /**
     * Compute delta statistics between src & dist groups for each size and key metric.
     * Emits benchLog lines for human diffs and returns structured delta array.
     */
  /**
   * Derive delta rows comparing src vs dist for each metric and size.
   * Returns an array of { size, metric, src, dist, delta, deltaPct }.
   */
  function computeVariantDeltas() {
      const deltas: Array<Record<string, any>> = [];
      const metricKeys = [
        'buildMsMean',
        'fwdAvgMsMean',
        'bytesPerConnMean',
        'heapUsedMean',
        'rssMean',
      ];
      const bySize: Record<number, { src?: any; dist?: any }> = {};
      for (const g of aggregated) {
        if (!bySize[g.size]) bySize[g.size] = {};
        (bySize[g.size] as any)[g.mode] = g;
      }
      for (const size of Object.keys(bySize)
        .map(Number)
        .sort((a, b) => a - b)) {
        const pair = bySize[size];
        if (!pair.src || !pair.dist) continue; // skip incomplete
        for (const mk of metricKeys) {
          if (mk in pair.src && mk in pair.dist) {
            const vSrc = pair.src[mk];
            const vDist = pair.dist[mk];
            const delta = vDist - vSrc;
            const deltaPct = vSrc !== 0 ? (delta / vSrc) * 100 : 0;
            const rec = {
              size,
              metric: mk,
              src: vSrc,
              dist: vDist,
              delta,
              deltaPct: Number(deltaPct.toFixed(2)),
            };
            deltas.push(rec);
          }
        }
      }
      return deltas;
    }

  const deltas = computeVariantDeltas();
  variantAggregated = aggregated;
  variantDeltas = deltas;
  // Variance summary (large sizes only, when counts >1)
  const varianceSummary = (aggregated as any[])
    .filter(g => g.size >= 100000 && g.count > 1)
    .map(g => ({
      mode: g.mode,
      size: g.size,
      buildMsMean: g.buildMsMean,
      buildMsStd: g.buildMsStd,
      buildMsCvPct: g.buildMsMean ? Number(((g.buildMsStd / g.buildMsMean) * 100).toFixed(2)) : 0,
      fwdAvgMsMean: g.fwdAvgMsMean,
      fwdAvgMsStd: g.fwdAvgMsStd,
      fwdAvgMsCvPct: g.fwdAvgMsMean ? Number(((g.fwdAvgMsStd / g.fwdAvgMsMean) * 100).toFixed(2)) : 0,
      samples: g.count,
    }));

  // Persist consolidated results (baseline + variant) to benchmark.results.json (with history accumulation)
    try {
      const path = require('path');
      const resultsFile = path.resolve(__dirname, 'benchmark.results.json');
      // Attempt to collect (placeholder) browser bundle metrics (dev/prod) without requiring a browser run yet.
      const candidateBundles = [
        {
          label: 'dev',
          path: path.resolve(
            __dirname,
            '../../..',
            'bench-browser',
            'dev.bundle.js'
          ),
        },
        {
          label: 'prod',
          path: path.resolve(
            __dirname,
            '../../..',
            'bench-browser',
            'prod.bundle.js'
          ),
        },
      ];
      const browserBundles = candidateBundles.map((b) => {
        try {
          if (fs.existsSync(b.path)) {
            const stat = fs.statSync(b.path);
            return {
              label: b.label,
              path: b.path,
              bytes: stat.size,
              exists: true,
            };
          }
        } catch {}
        return { label: b.label, path: b.path, exists: false };
      });
      // Load existing results to extend history
      let existing: any = undefined;
      if (fs.existsSync(resultsFile)) {
        try { existing = JSON.parse(fs.readFileSync(resultsFile, 'utf-8')); } catch {}
      }
      // Attempt to capture git commit hash
      let commit: string | undefined;
      try {
        const cp = require('child_process');
        commit = cp.execSync('git rev-parse --short HEAD', { stdio: ['ignore','pipe','ignore'] }).toString().trim();
      } catch {}
      // Helper: extract compact summary for history (mode,size and key means only)
      function summarize(ag: any[]) {
        return ag.map(g => ({ mode: g.mode, size: g.size, buildMsMean: g.buildMsMean, fwdAvgMsMean: g.fwdAvgMsMean, bytesPerConnMean: g.bytesPerConnMean }));
      }
      // Helper: derive forward delta pct per size (dist - src) for quick regression scans
      function forwardDeltaPct(deltasArr: any[]) {
        const out: Array<{ size: number; deltaPct: number }> = [];
        for (const d of deltasArr) if (d.metric === 'fwdAvgMsMean') out.push({ size: d.size, deltaPct: d.deltaPct });
        return out.sort((a,b)=>a.size-b.size);
      }
      // Build new snapshot in slim schema (omit full deltas to keep file small)
      const snapshot = {
        generatedAt: new Date().toISOString(),
        commit,
        sizes: Array.from(new Set(rawMeasurements.map(r => r.size))).sort((a,b)=>a-b),
        summary: summarize(aggregated as any[]),
        fwdDeltaPct: forwardDeltaPct(deltas),
        fieldAuditCounts: existing && existing.fieldAudit ? {
          Node: existing.fieldAudit.Node?.count,
          Connection: existing.fieldAudit.Connection?.count,
        } : undefined,
      };
      const payload: any = existing || {};
      payload.generatedAt = snapshot.generatedAt;
      payload.baseline = baselineRecords;
      payload.variantRaw = rawMeasurements;
      payload.aggregated = aggregated;
      payload.deltas = deltas;
      payload.warnings = warnings;
      payload.browserBundles = browserBundles;
  payload.meta = { note: 'Phase 0 results file (no console logs). Update plan tables manually from this source.', largeEnabled: enableLarge, varianceRepeatsLarge: BENCH_REPEAT_LARGE };
  if (varianceSummary.length) payload.variance = varianceSummary;
      // Slim & normalize existing history (convert old verbose entries to new shape)
      const oldHistory: any[] = Array.isArray(payload.history) ? payload.history : [];
      const normalized = oldHistory.map(h => {
        if (h && h.summary && Array.isArray(h.summary) && !h.fwdDeltaPct && Array.isArray(h.deltas)) {
          return {
            generatedAt: h.generatedAt,
            commit: h.commit,
            sizes: h.sizes || Array.from(new Set((h.summary as any[]).map(r=>r.size))).sort((a:number,b:number)=>a-b),
            summary: h.summary.map((s:any)=>({ mode: s.mode, size: s.size, buildMsMean: s.buildMsMean, fwdAvgMsMean: s.fwdAvgMsMean, bytesPerConnMean: s.bytesPerConnMean })),
            fwdDeltaPct: h.deltas.filter((d:any)=>d.metric==='fwdAvgMsMean').map((d:any)=>({ size: d.size, deltaPct: d.deltaPct })),
            fieldAuditCounts: h.fieldAudit ? { Node: h.fieldAudit.Node?.count, Connection: h.fieldAudit.Connection?.count } : undefined,
          };
        }
        return h && h.summary ? h : null;
      }).filter(Boolean);
      normalized.push(snapshot);
      // Enforce max history length (retain most recent 10)
      payload.history = normalized.slice(-10);
      fs.writeFileSync(resultsFile, JSON.stringify(payload, null, 2), 'utf-8');
    } catch (e) {
      // Swallow errors silently to avoid noisy console output per user request.
    }

    describe('aggregation groups (structure)', () => {
      it('provides src group for size 1000', () => { expect(!!groupByMode['src']).toBe(true); });
      it('provides dist group for size 1000', () => { expect(!!groupByMode['dist']).toBe(true); });
    });
    describe('aggregated metric exposure', () => {
      it('exposes buildMsMean in src group', () => { expect('buildMsMean' in groupByMode['src']).toBe(true); });
      it('exposes fwdAvgMsMean in dist group', () => { expect('fwdAvgMsMean' in groupByMode['dist']).toBe(true); });
    });
    describe('sanity of computed magnitudes', () => {
      it('src bytesPerConnMean is positive', () => { expect((groupByMode['src'] as any).bytesPerConnMean > 0).toBe(true); });
      it('dist bytesPerConnMean is positive', () => { expect((groupByMode['dist'] as any).bytesPerConnMean > 0).toBe(true); });
    });
    describe('delta computation outputs', () => {
      it('emits at least one delta row', () => { expect(deltas.length > 0).toBe(true); });
      it('contains buildMsMean delta for size 1000', () => { const found = deltas.some(d => d.size === 1000 && d.metric === 'buildMsMean'); expect(found).toBe(true); });
    });
    describe('persistence side-effects', () => {
      it('appends snapshot to history array', () => {
        const path = require('path');
        const resultsFile = path.resolve(__dirname, 'benchmark.results.json');
        let parsed: any = {};
        if (fs.existsSync(resultsFile)) { try { parsed = JSON.parse(fs.readFileSync(resultsFile,'utf-8')); } catch {} }
        const isValid = Array.isArray(parsed.history) && parsed.history.length > 0; // Arrange+Act
        expect(isValid).toBe(true); // Assert
      });
    });
  });
});
