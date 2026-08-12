/**
 * Build script for the Neatenstein neon raycasting demo.
 *
 * Mirrors the Flappy Bird build pattern: a dedicated worker bundle plus a host
 * bundle. The worker bundle carries the shared raycaster so the Worker tier can
 * render via OffscreenCanvas, while the CPU/GPU tiers render on the main thread.
 *
 * Produces three published assets in `docs/assets/`:
 *   - `neatenstein.bundle.js`         (IIFE host bundle)
 *   - `neatenstein.worker.js`         (classic IIFE display worker bundle)
 *   - `neatenstein.eval-worker.js`    (classic IIFE eval worker bundle)
 */
import { build } from 'esbuild';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');

const hostEntry = resolve(
  repositoryRoot,
  'examples/neatenstein/browser-entry/browser-entry.ts',
);
const workerEntry = resolve(
  repositoryRoot,
  'examples/neatenstein/browser-entry/worker/neatenstein.worker.ts',
);
const evalWorkerEntry = resolve(
  repositoryRoot,
  'examples/neatenstein/browser-entry/worker/neatenstein.eval-worker.ts',
);

const hostOutfile = resolve(
  repositoryRoot,
  'docs/assets/neatenstein.bundle.js',
);
const workerOutfile = resolve(
  repositoryRoot,
  'docs/assets/neatenstein.worker.js',
);
const evalWorkerOutfile = resolve(
  repositoryRoot,
  'docs/assets/neatenstein.eval-worker.js',
);

/**
 * Shared esbuild options used by both the host and worker bundles.
 *
 * The `alias` maps the bare `neataptic` specifier (used by the worker's
 * dynamic `import('neataptic')` call) to the browser-safe entry point so
 * esbuild bundles the browser-compatible version instead of trying to
 * resolve the Node-oriented root entry. The `node:crypto` alias maps the
 * Node.js crypto module to a browser-compatible pure-JS SHA-256 shim so the
 * NGE DNA static import chain (worker → arms-race → main-runner → NGE DNA
 * utils → node:crypto) bundles a synchronous shim instead of leaving a
 * runtime `require("node:crypto")` that crashes in the browser.
 */
const sharedBuildOptions = {
  bundle: true,
  platform: 'browser',
  minify: true,
  sourcemap: true,
  target: 'es2023',
  external: ['fs', 'child_process', 'path'],
  alias: {
    neataptic: resolve(repositoryRoot, 'src/browser-entry.ts'),
    'node:crypto': resolve(
      repositoryRoot,
      'examples/neatenstein/browser-entry/node-crypto-shim.ts',
    ),
  },
  logLevel: 'info',
};

// Host bundle: classic IIFE loaded by a regular <script> tag.
await build({
  ...sharedBuildOptions,
  entryPoints: [hostEntry],
  outfile: hostOutfile,
  format: 'iife',
});

// Worker bundle: classic IIFE loaded as a standard (non-module) Worker.
await build({
  ...sharedBuildOptions,
  entryPoints: [workerEntry],
  outfile: workerOutfile,
  format: 'iife',
});

// Eval worker bundle: classic IIFE loaded as a standard (non-module) Worker.
// Offloads NEAT population evaluation from the display worker's render loop.
await build({
  ...sharedBuildOptions,
  entryPoints: [evalWorkerEntry],
  outfile: evalWorkerOutfile,
  format: 'iife',
});
