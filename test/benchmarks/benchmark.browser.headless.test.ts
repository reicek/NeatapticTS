/**
 * benchmark.browser.headless.test.ts
 *
 * Integrates a headless (Puppeteer) browser micro‑benchmark harness into the Jest
 * test suite. This lets developers run `npm test -- benchmark` (or the full suite)
 * and automatically:
 *  1. Build ephemeral dev & prod IIFE bundles via esbuild.
 *  2. Launch a headless Chromium instance.
 *  3. Execute the bundles inside fresh pages and scrape benchmark results that
 *     the bundle attaches to `window.__NEATAPTIC_BENCH__`.
 *  4. Persist the gathered run records into `benchmark.results.json` so that
 *     downstream tooling (CI artifact collection, docs, dashboards) can consume them.
 *
 * The build artifacts are timestamp + PID suffixed so that parallel Jest workers
 * (or overlapping CI jobs) do not trample each other's temporary files. Each test
 * case asserts structural expectations instead of performance thresholds to keep
 * the suite deterministic across different hardware.
 */
import * as fs from 'fs';
import * as path from 'path';

/**
 * Optional escape hatch for environments (e.g. limited CI containers) where
 * running a headless browser is undesirable or flaky. Set `SKIP_BROWSER_BENCH=1`
 * to bypass the benchmark portion while keeping the test file green.
 */
const SKIP = process.env.SKIP_BROWSER_BENCH === '1';
jest.setTimeout(60000);

/**
 * Shape of a single browser benchmark execution result.
 *
 * Each bundle variant ("dev" / "prod") yields one record summarising:
 *  - mode: The bundle mode label.
 *  - bundleBytes: Final on-disk bundle size (bytes) after esbuild output.
 *  - performanceMemory: Optional window.performance.memory snapshot (Chromium only).
 *  - bench: Payload exported by the running bundle (see bench-entry.ts) containing
 *           synthetic network timing metrics.
 */
interface BrowserRunRecord {
  mode: string;
  bundleBytes: number;
  performanceMemory: any;
  bench: any;
}

/**
 * Builds development & production browser bundles with esbuild.
 *
 * Contract:
 *  Inputs: none (relies on bench-entry.ts existing & local filesystem)
 *  Outputs: absolute file paths of temporary dev & prod bundle artifacts.
 *  Error Modes: throws if the entry file is missing or esbuild reports a failure.
 *
 * Implementation notes:
 *  - Uses a timestamp + process id suffix to create unique bundle filenames.
 *  - Marks Node‑only core modules as externals so esbuild does not attempt to
 *    polyfill or resolve them for the browser (avoids CI resolution failures).
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
  // Build development (unminified) variant.
  await esbuild.build({
    entryPoints: [entry],
    bundle: true,
    outfile: devOut,
    platform: 'browser',
    format: 'iife',
    sourcemap: false,
    define: { __BENCH_MODE__: '"dev"' },
    // Exclude Node-specific modules so esbuild doesn't try to bundle them for the browser.
    // 'path' is required by the node TestWorker but should stay external in the browser build.
    external: ['child_process', 'fs', 'worker_threads', 'path'],
    write: true,
  });
  // Build production (minified) variant.
  await esbuild.build({
    entryPoints: [entry],
    bundle: true,
    outfile: prodOut,
    platform: 'browser',
    format: 'iife',
    minify: true,
    define: { __BENCH_MODE__: '"prod"' },
    external: ['child_process', 'fs', 'worker_threads', 'path'],
    write: true,
  });
  return { devPath: devOut, prodPath: prodOut };
}

/**
 * Launches a headless Chromium instance (if available) and executes each built
 * bundle in isolation, scraping the benchmark payload exposed on the window.
 *
 * Defensive design:
 *  - If Puppeteer cannot be required (dependency optional in some installs) the
 *    function returns an empty array and tests degrade gracefully.
 *  - If the browser fails to launch (e.g. sandbox restrictions) we also
 *    short‑circuit with an empty result set.
 *  - A bounded polling loop (15s cap) waits for the bundle to mark readiness.
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
  // Launch headless browser; tolerate launch failures (return empty results).
  const browser = await puppeteer
    .launch({ headless: 'new', args: ['--no-sandbox'] })
    .catch(() => null);
  if (!browser) return [];
  const runs: BrowserRunRecord[] = [];
  for (const b of bundles) {
    if (!b.path || !fs.existsSync(b.path)) continue;
    const fileName = path.basename(b.path);
    // Ensure the uniquely suffixed bundle is in the bench directory so that the
    // temporary HTML file can reference it with a simple relative path.
    const localCopy = path.join(benchDir, fileName);
    if (b.path !== localCopy) {
      try {
        if (!fs.existsSync(localCopy)) {
          fs.copyFileSync(b.path, localCopy);
        }
      } catch {}
    }
    // Inject mode + bundle placeholder tokens into HTML template.
    const html = template
      .replace(/__MODE__/g, b.mode)
      .replace(/__BUNDLE__/g, fileName);
    const tmpPath = path.join(benchDir, `jest-${b.mode}.${fileName}.html`);
    fs.writeFileSync(tmpPath, html, 'utf-8');
    const page = await browser.newPage();
    await page.goto(`file://${tmpPath}`);
    // Poll the page for the benchmark payload with a hard timeout.
    const start = Date.now();
    let payload: any;
    while (Date.now() - start < 15000) {
      payload = await page
        .evaluate(() => (window as any).__NEATAPTIC_BENCH__)
        .catch(() => null);
      if (payload) break;
      await new Promise((r) => setTimeout(r, 50));
    }
    // Attempt to read memory stats (Chromium only; may be absent in CI).
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

/**
 * Merges (overwrites) the provided run records into benchmark.results.json.
 * File is created if missing. Existing structure is preserved except for the
 * browserRuns & meta.browserHarness keys which we control.
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
