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
async function buildBundles(): Promise<{ devPath: string; prodPath: string }> {
  const esbuild = require('esbuild');
  const benchDir = path.resolve(__dirname, '../..', 'bench-browser');
  const entry = path.join(benchDir, 'bench-entry.ts');
  if (!fs.existsSync(entry)) throw new Error('bench-entry.ts missing');
  const ts = Date.now();
  const suffix = `${ts}-${process.pid}`;
  const devOut = path.resolve(benchDir, `dev.bundle.${suffix}.js`);
  const prodOut = path.resolve(benchDir, `prod.bundle.${suffix}.js`);
  await esbuild.build({
    entryPoints: [entry],
    bundle: true,
    outfile: devOut,
    platform: 'browser',
    format: 'iife',
    sourcemap: false,
    define: { __BENCH_MODE__: '"dev"' },
    external: ['child_process', 'fs', 'worker_threads'],
    write: true,
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
    write: true,
  });
  return { devPath: devOut, prodPath: prodOut };
}

/**
 * Headless run with dynamic bundle paths (avoids rewriting locked files).
 */
async function runHeadless(paths: {
  devPath: string;
  prodPath: string;
}): Promise<BrowserRunRecord[]> {
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
  const bundles = [
    { mode: 'dev', path: paths.devPath },
    { mode: 'prod', path: paths.prodPath },
  ];
  const browser = await puppeteer
    .launch({ headless: 'new', args: ['--no-sandbox'] })
    .catch(() => null);
  if (!browser) return [];
  const runs: BrowserRunRecord[] = [];
  for (const b of bundles) {
    if (!b.path || !fs.existsSync(b.path)) continue;
    const fileName = path.basename(b.path);
    // Generate per-mode HTML referencing the unique bundle filename via relative copy
    const localCopy = path.join(benchDir, fileName);
    if (b.path !== localCopy) {
      try {
        if (!fs.existsSync(localCopy)) {
          fs.copyFileSync(b.path, localCopy);
        }
      } catch {}
    }
    const html = template
      .replace(/__MODE__/g, b.mode)
      .replace(/__BUNDLE__/g, fileName);
    const tmpPath = path.join(benchDir, `jest-${b.mode}.${fileName}.html`);
    fs.writeFileSync(tmpPath, html, 'utf-8');
    const page = await browser.newPage();
    await page.goto(`file://${tmpPath}`);
    const start = Date.now();
    let payload: any;
    while (Date.now() - start < 15000) {
      payload = await page
        .evaluate(() => (window as any).__NEATAPTIC_BENCH__)
        .catch(() => null);
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
      mode: b.mode,
      bundleBytes: fs.statSync(b.path).size,
      performanceMemory: perfMem,
      bench: payload || null,
    });
    await page.close();
  }
  await browser.close();
  return runs;
}

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
    it('skips when SKIP_BROWSER_BENCH=1', () => {
      expect(true).toBe(true);
    });
    return;
  }

  let runs: BrowserRunRecord[] = [];
  let devPath = '';
  let prodPath = '';

  beforeAll(async () => {
    const built = await buildBundles();
    devPath = built.devPath;
    prodPath = built.prodPath;
    runs = await runHeadless(built);
    mergeResults(runs);
  });

  afterAll(() => {
    [devPath, prodPath].forEach((f) => {
      try {
        fs.unlinkSync(f);
      } catch {}
    });
  });

  it('produces an array of run records', () => {
    expect(Array.isArray(runs)).toBe(true);
  });
  it('dev & prod bundle files exist', () => {
    expect(fs.existsSync(devPath) && fs.existsSync(prodPath)).toBe(true);
  });
  it('persists browserRuns in results file', () => {
    const resultsPath = path.resolve(__dirname, 'benchmark.results.json');
    let parsed: any = {};
    if (fs.existsSync(resultsPath)) {
      try {
        parsed = JSON.parse(fs.readFileSync(resultsPath, 'utf-8'));
      } catch {}
    }
    expect(Array.isArray(parsed.browserRuns)).toBe(true);
  });
  if (runs.length) {
    const first = runs[0];
    it('run record exposes numeric bundleBytes', () => {
      expect(typeof first.bundleBytes).toBe('number');
    });
    it('run record exposes mode string', () => {
      expect(typeof first.mode).toBe('string');
    });
  }
});
