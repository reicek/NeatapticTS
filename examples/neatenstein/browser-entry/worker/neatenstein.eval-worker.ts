/// <reference lib="webworker" />

/**
 * Browser-worker entrypoint shim for the Neatenstein evaluation worker.
 *
 * The real evaluation worker logic lives in {@link ./eval.worker.ts}. This
 * file is the thin esbuild entry point used by the build script so the
 * published asset name stays stable as
 * `docs/assets/neatenstein.eval-worker.js`.
 *
 * @module
 */

import './eval.worker';
