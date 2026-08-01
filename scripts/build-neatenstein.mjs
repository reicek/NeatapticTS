/**
 * Build script for the Neatenstein neon raycasting demo.
 *
 * Mirrors the Flappy Bird build pattern: a dedicated worker bundle plus a host
 * bundle. The worker bundle carries the shared raycaster so the Worker tier can
 * render via OffscreenCanvas, while the CPU/GPU tiers render on the main thread.
 *
 * Produces two published assets in `docs/assets/`:
 *   - `neatenstein.bundle.js`   (IIFE host bundle)
 *   - `neatenstein.worker.js`   (classic IIFE worker bundle)
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

const hostOutfile = resolve(
  repositoryRoot,
  'docs/assets/neatenstein.bundle.js',
);
const workerOutfile = resolve(
  repositoryRoot,
  'docs/assets/neatenstein.worker.js',
);

/**
 * Shared esbuild options used by both the host and worker bundles.
 */
const sharedBuildOptions = {
  bundle: true,
  platform: 'browser',
  minify: true,
  sourcemap: true,
  target: 'es2023',
  external: ['fs', 'child_process', 'path'],
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
