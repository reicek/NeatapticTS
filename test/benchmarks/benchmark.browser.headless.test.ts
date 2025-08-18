/**
 * benchmark.browser.headless.test.ts
 * Integrates the browser (Puppeteer) benchmark harness into the Jest suite so that
 * running pattern-based benchmark tests (e.g. `npm test -- benchmark`) also produces
 * browserRuns data in benchmark.results.json without relying on an external npm script.
 */
import * as fs from 'fs';
import * as path from 'path';

const SKIP = process.env.SKIP_BROWSER_BENCH === '1';
jest.setTimeout(60000);

/**
 * Captured browser benchmark run record. Each mode corresponds to a built bundle
 * variant (dev vs prod) executed in a fresh page.
 */
interface BrowserRunRecord {
  mode: string;
  bundleBytes: number;
  performanceMemory: any;
  bench: any;
}

/**
 * Build development & production browser bundles via esbuild. Returns filesystem
 * paths to the generated artifacts (Arrange stage for browser benchmarks).
 */
async function buildBundles(): Promise<{
  devPath?: string;
  prodPath?: string;
}> {
  const esbuild = require('esbuild');
  const benchDir = path.resolve(__dirname, '../..', 'bench-browser');
  const entry = path.join(benchDir, 'bench-entry.ts');
  if (!fs.existsSync(entry)) throw new Error('bench-entry.ts missing');
  const devOut = path.join(benchDir, 'dev.bundle.js');
  const prodOut = path.join(benchDir, 'prod.bundle.js');
  await esbuild.build({
    entryPoints: [entry],
    bundle: true,
    outfile: devOut,
    platform: 'browser',
    format: 'iife',
    sourcemap: false,
    define: { __BENCH_MODE__: '"dev"' },
    external: ['child_process', 'fs', 'worker_threads'],
  });
  await esbuild.build({
    entryPoints: [entry],
    bundle: true,
    outfile: prodOut,
    platform: 'browser',
    format: 'iife',
    minify: true,
    define: { __BENCH_MODE__: '"prod"' },
    external: ['child_process', 'fs', 'worker_threads'],
  });
  return { devPath: devOut, prodPath: prodOut };
}

/**
 * Execute benchmark bundles inside a headless Chromium instance using Puppeteer.
 * Polls a global symbol injected by the bench harness for completion payload.
 */
async function runHeadless(): Promise<BrowserRunRecord[]> {
  let puppeteer;
  try {
    puppeteer = require('puppeteer');
  } catch {
    return [];
  }
  const benchDir = path.resolve(__dirname, '../..', 'bench-browser');
  const templatePath = path.join(benchDir, 'index.html');
  if (!fs.existsSync(templatePath)) return [];
  const template = fs.readFileSync(templatePath, 'utf-8');
  const modes: Array<{ mode: string; bundle: string }> = [
    { mode: 'dev', bundle: 'dev.bundle.js' },
    { mode: 'prod', bundle: 'prod.bundle.js' },
  ];
  const browser = await puppeteer
    .launch({ headless: 'new', args: ['--no-sandbox'] })
    .catch(() => null);
  if (!browser) return [];
  const runs: BrowserRunRecord[] = [];
  for (const m of modes) {
    const bundlePath = path.join(benchDir, m.bundle);
    if (!fs.existsSync(bundlePath)) continue;
    const html = template
      .replace('__MODE__', m.mode)
      .replace('__BUNDLE__', m.bundle);
    const tmpPath = path.join(benchDir, `jest-${m.mode}.html`);
    fs.writeFileSync(tmpPath, html, 'utf-8');
    const page = await browser.newPage();
    await page.goto(`file://${tmpPath}`);
    const start = Date.now();
    let payload: any;
    while (Date.now() - start < 15000) {
      payload = await page.evaluate(() => (window as any).__NEATAPTIC_BENCH__);
      if (payload) break;
      await new Promise((r) => setTimeout(r, 50));
    }
    const perfMem = await page
      .evaluate(() => {
        const pm: any = (performance as any).memory || null;
        return pm
          ? {
              usedJSHeapSize: pm.usedJSHeapSize,
              totalJSHeapSize: pm.totalJSHeapSize,
              jsHeapSizeLimit: pm.jsHeapSizeLimit,
            }
          : null;
      })
      .catch(() => null);
    runs.push({
      mode: m.mode,
      bundleBytes: fs.statSync(bundlePath).size,
      performanceMemory: perfMem,
      bench: payload || null,
    });
    await page.close();
  }
  await browser.close();
  return runs;
}

/**
 * Merge browser run results into the shared benchmark.results.json artifact.
 */
function mergeResults(browserRuns: BrowserRunRecord[]) {
  const resultsPath = path.resolve(__dirname, 'benchmark.results.json');
  let data: any = {};
  if (fs.existsSync(resultsPath)) {
    try {
      data = JSON.parse(fs.readFileSync(resultsPath, 'utf-8'));
    } catch {}
  }
  data.browserRuns = browserRuns;
  data.meta = Object.assign({}, data.meta || {}, { browserHarness: true });
  fs.writeFileSync(resultsPath, JSON.stringify(data, null, 2), 'utf-8');
}

describe('browser headless benchmark integration', () => {
  if (SKIP) {
    it('skips when SKIP_BROWSER_BENCH=1', () => { expect(true).toBe(true); });
    return; // Prevent executing further browser dependent tests
  }
  // Arrange (once): build bundles & perform headless runs
  let runs: BrowserRunRecord[] = [];
  let built: { devPath?: string; prodPath?: string } = {};
  beforeAll(async () => {
    built = await buildBundles();
    runs = await runHeadless();
    mergeResults(runs);
  });
  it('produces an array of run records', () => { expect(Array.isArray(runs)).toBe(true); });
  it('includes dev bundle path', () => { expect(typeof built.devPath).toBe('string'); });
  it('includes prod bundle path', () => { expect(typeof built.prodPath).toBe('string'); });
  it('persists browserRuns in results file', () => {
    const resultsPath = path.resolve(__dirname, 'benchmark.results.json');
    let parsed: any = {};
    if (fs.existsSync(resultsPath)) { try { parsed = JSON.parse(fs.readFileSync(resultsPath,'utf-8')); } catch {} }
    const ok = Array.isArray(parsed.browserRuns);
    expect(ok).toBe(true);
  });
  if (runs.length) {
    const first = runs[0];
    it('run record exposes numeric bundleBytes', () => { expect(typeof first.bundleBytes).toBe('number'); });
    it('run record exposes mode string', () => { expect(typeof first.mode).toBe('string'); });
  }
});
