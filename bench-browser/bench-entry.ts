/*
 * Browser Benchmark Entry (Phase 0 Step 4)
 * Generates synthetic networks and measures build + forward timings similar to Node harness.
 * Results are attached to window.__NEATAPTIC_BENCH__ for later scraping by a headless runner.
 */
// @ts-ignore dist build assumed present after npm run build
import { Network } from '../dist/neataptic.js';

interface BrowserBenchRecord {
  size: number;
  buildMs: number;
  fwdAvgMs: number;
  fwdTotalMs: number;
  conn: number;
  nodes: number;
  iterations: number;
}

function buildSynthetic(size: number): { net: any; buildMs: number } {
  const t0 = performance.now();
  const inputs = Math.max(1, Math.floor(Math.sqrt(size)));
  const outputs = Math.max(1, Math.ceil(size / inputs));
  const net = new (Network as any)(inputs, outputs);
  while (net.connections.length > size) {
    const idx = Math.floor(Math.random() * net.connections.length);
    const c = net.connections[idx];
    (net as any).disconnect(c.from, c.to);
  }
  const t1 = performance.now();
  return { net, buildMs: t1 - t0 };
}

function measureForward(net: any, iterations: number): { totalMs: number; avgMs: number } {
  const vec = new Array(net.input).fill(0).map(() => Math.random());
  const t0 = performance.now();
  for (let i = 0; i < iterations; i++) net.activate(vec);
  const t1 = performance.now();
  const totalMs = t1 - t0;
  return { totalMs, avgMs: totalMs / iterations };
}

function run(): BrowserBenchRecord[] {
  const sizes = [1000, 10000, 50000, 100000];
  const out: BrowserBenchRecord[] = [];
  for (const size of sizes) {
    const { net, buildMs } = buildSynthetic(size);
    const iterations = size >= 100000 ? 2 : size >= 50000 ? 3 : 5;
    const { totalMs, avgMs } = measureForward(net, iterations);
    out.push({
      size,
      buildMs: Number(buildMs.toFixed(3)),
      fwdAvgMs: Number(avgMs.toFixed(4)),
      fwdTotalMs: Number(totalMs.toFixed(3)),
      conn: net.connections.length,
      nodes: net.nodes.length,
      iterations,
    });
  }
  return out;
}

(window as any).__NEATAPTIC_BENCH__ = {
  mode: (window as any).__BENCH_MODE__ || '__UNDEF__',
  generatedAt: new Date().toISOString(),
  results: run(),
};

// eslint-disable-next-line no-console
console.log('[NEATAPTIC_BROWSER_BENCH] ready');
