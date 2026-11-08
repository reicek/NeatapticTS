/**
 * Build script for the NeatRacing voxel racer browser demo (spec P3-S2,
 * AC-037).
 *
 * Mirrors the Neatenstein build pattern (spec 11.2 recycle table): one IIFE
 * host bundle plus one IIFE simulation worker bundle, both targeting the
 * browser platform with minified + sourcemapped ES2023 output and the
 * fs/child_process/path Node externals.
 *
 * Produces two published assets in `docs/assets/`:
 *   - `neatracing.bundle.js`  (IIFE host bundle, loaded by index.html)
 *   - `neatracing.worker.js`  (IIFE simulation worker bundle, AC-037)
 *
 * The `alias` maps the bare `neataptic` specifier to the browser-safe
 * entry point so esbuild would bundle the browser-compatible version of the
 * library instead of the Node-oriented root entry, exactly as the
 * Neatenstein pipeline does.
 *
 * This module is test-importable: the real builds only run when the script
 * is executed directly (CLI module check), matching the trace-compress.mjs
 * pattern. Tests inject their own build function and outfile overrides.
 */
import { build } from 'esbuild';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

export const repositoryRoot = resolve(
  dirname(fileURLToPath(import.meta.url)),
  '..',
);

/** Host (main-thread) bundle entry: the NeatRacing browser entry point. */
export const NEATRACING_HOST_ENTRY = join(
  'examples',
  'neatRacing',
  'browser-entry',
  'browser-entry.ts',
);

/** Host bundle output relative to the repository root (AC-037). */
export const NEATRACING_HOST_OUTFILE = join(
  'docs',
  'assets',
  'neatracing.bundle.js',
);

/** Simulation worker bundle entry (in-band worker session adapter). */
export const NEATRACING_WORKER_ENTRY = join(
  'examples',
  'neatRacing',
  'workers',
  'simulation-worker',
  'simulation-worker.ts',
);

/**
 * Simulation worker bundle output relative to the repository root (AC-037).
 * Renamed from the pre-P3 CLI `neatracing-simulation-worker.bundle.js` to
 * the spec-mandated `neatracing.worker.js` name; the old name is referenced
 * by nothing in source or HTML.
 */
export const NEATRACING_WORKER_OUTFILE = join(
  'docs',
  'assets',
  'neatracing.worker.js',
);

/**
 * Shared esbuild options used by both the host and the simulation worker
 * bundles (AC-037): IIFE + browser platform + minify + sourcemap + ES2023,
 * with `fs`/`child_process`/`path` externalized and the bare `neataptic`
 * specifier aliased to the browser-safe entry shim.
 */
export const NEATRACING_SHARED_BUILD_OPTIONS = Object.freeze({
  bundle: true,
  platform: 'browser',
  format: 'iife',
  minify: true,
  sourcemap: true,
  target: 'es2023',
  external: Object.freeze(['fs', 'child_process', 'path']),
  alias: Object.freeze({
    neataptic: resolve(repositoryRoot, 'src/browser-entry.ts'),
  }),
  logLevel: 'info',
});

/**
 * Builds the NeatRacing host and simulation worker bundles.
 *
 * @param {Function} esbuildBuildFn - The esbuild `build` function to use
 *   (injectable so tests can run the pipeline into temp directories).
 * @param {Object} [overrides] - Optional outfile overrides.
 * @param {string} [overrides.hostOutfile] - Absolute host bundle output.
 * @param {string} [overrides.workerOutfile] - Absolute worker bundle output.
 * @returns {Promise<Object>} Resolved build info for both bundles.
 */
export async function buildNeatRacingBundles(esbuildBuildFn, overrides = {}) {
  const hostOutfile =
    overrides.hostOutfile ?? resolve(repositoryRoot, NEATRACING_HOST_OUTFILE);
  const workerOutfile =
    overrides.workerOutfile ??
    resolve(repositoryRoot, NEATRACING_WORKER_OUTFILE);

  const host = await esbuildBuildFn({
    ...NEATRACING_SHARED_BUILD_OPTIONS,
    entryPoints: [resolve(repositoryRoot, NEATRACING_HOST_ENTRY)],
    outfile: hostOutfile,
  });

  const worker = await esbuildBuildFn({
    ...NEATRACING_SHARED_BUILD_OPTIONS,
    entryPoints: [resolve(repositoryRoot, NEATRACING_WORKER_ENTRY)],
    outfile: workerOutfile,
  });

  return { host, worker, hostOutfile, workerOutfile };
}

/**
 * Determines whether the current process is the CLI entry point.
 *
 * @param argv1 - The `process.argv[1]` value for the current process.
 * @param entryUrl - The `import.meta.url` of the module.
 * @returns True when the module is invoked directly as a script.
 */
export function isCliEntryPoint(argv1, entryUrl) {
  return Boolean(argv1) && entryUrl === pathToFileURL(argv1).href;
}

/**
 * Runs the NeatRacing build pipeline when the module is invoked directly.
 *
 * Exposed for unit testing so the CLI dispatch path can be exercised
 * without spawning a child process.
 *
 * @param context - Invocation context carrying the entry-point signals.
 * @param options - Optional outfile overrides forwarded to the build.
 * @returns Resolves when both bundles are built, or no-ops when not main.
 */
export async function runBuildCli(context = {}, options = {}) {
  const argv1 = context.argv1 ?? process.argv[1];
  if (!isCliEntryPoint(argv1, context.entryUrl)) {
    return;
  }
  return buildNeatRacingBundles(build, options);
}

await runBuildCli({ entryUrl: import.meta.url });
