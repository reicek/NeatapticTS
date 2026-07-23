/// <reference lib="webworker" />

/**
 * Browser-worker entrypoint shim for the Neatenstein neon raycasting demo.
 *
 * The real display worker logic lives in {@link ./display.worker.ts}. This file
 * is the thin esbuild entry point used by the build script so the published
 * asset name stays stable as `docs/assets/neatenstein.worker.esm.js`.
 *
 * @module
 */

import './display.worker';
