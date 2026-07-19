/**
 * Build script stub for the Neatenstein neon raycasting demo.
 *
 * Mirrors the Flappy Bird build pattern: a dedicated worker bundle plus a host
 * bundle. The worker bundle carries the shared raycaster so the Worker tier can
 * render via OffscreenCanvas, while the CPU/GPU tiers render on the main thread.
 *
 * This is intentionally a scaffold stub for Phase 1. It wires the expected dual
 * entry points but does not yet produce final minified assets.
 */
import { build } from 'esbuild';
import { existsSync } from 'node:fs';
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

if (!existsSync(hostEntry) || !existsSync(workerEntry)) {
  console.log(
    'Neatenstein build: Phase 1 scaffold stub — host/worker entry points not yet implemented.',
  );
  process.exit(0);
}

await build({
  entryPoints: [hostEntry, workerEntry],
  bundle: true,
  outdir: resolve(repositoryRoot, 'docs/assets'),
  platform: 'browser',
  format: 'iife',
  minify: true,
  sourcemap: true,
  external: ['fs', 'child_process', 'path'],
  logLevel: 'info',
});
