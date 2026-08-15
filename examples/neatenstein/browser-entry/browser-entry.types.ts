/**
 * Entry-level type definitions for the Neatenstein browser entrypoint.
 *
 * Extracted from `browser-entry.ts` and `node-crypto-shim.ts` so that the
 * entrypoint modules contain only logic while type contracts live in a
 * dedicated, importable file.
 *
 * @module
 */

/**
 * Host script entry point that starts the Neatenstein demo.
 *
 * @param outputId - Host container element id that owns the HIVE DENSITY HUD
 *   overlay.
 * @param canvasId - Visible canvas element id to bind the renderer to.
 * @returns A teardown function that cancels the render loop, detaches input, and
 *   terminates the worker.
 */
export type NeatensteinStart = (
  outputId: string,
  canvasId: string,
) => NeatensteinStop;

/** Teardown function returned by {@link NeatensteinStart}. */
export type NeatensteinStop = () => void;

/** Interface matching the Node.js Hash object subset used by the codebase. */
export interface ShimHash {
  /** Feed data into the hash. Returns this for chaining. */
  update(data: string): ShimHash;
  /** Produce the final digest in the requested encoding. */
  digest(encoding: 'hex'): string;
}
