/**
 * @fileoverview Build script for the ASCII Maze browser demo bundle.
 *
 * Bundles `examples/asciiMaze/browser-entry.ts` into
 * `docs/assets/ascii-maze.bundle.js` using esbuild in IIFE (browser) mode
 * with minification and source-maps enabled.
 *
 * ### Custom stub plugin — `ascii-maze-browser-multi-stub`
 *
 * The multi-threading entry point (`src/multithreading/multi.ts`) relies on
 * Node.js `worker_threads` APIs that have no browser equivalent. In the maze
 * demo only one file (`network.evolve.fitness.utils.ts`) imports the shared
 * `multi` helper, and that file is only ever exercised inside Node workers —
 * never in the browser evaluation path.
 *
 * The `ascii-maze-browser-multi-stub` plugin intercepts the single import:
 * ```
 * '../../../multithreading/multi'
 * ```
 * …when it originates from `network.evolve.fitness.utils.ts`, and redirects
 * it to a lightweight browser stub (`browser-entry.multi.stub.ts`) that
 * exports the same public surface but replaces all worker-thread calls with
 * no-ops. Any other file that imports the same path is left untouched so the
 * real multi-threading module is still bundled for non-browser paths.
 */
import { build } from 'esbuild';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const repositoryRoot = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const legacyMultiImportPath = '../../../multithreading/multi';
const legacyMultiImporterSuffix =
  '/src/architecture/network/evolve/network.evolve.fitness.utils.ts';
const browserMultiStubPath = resolve(
  repositoryRoot,
  'examples/asciiMaze/browser-entry/browser-entry.multi.stub.ts',
);

await build({
  entryPoints: [resolve(repositoryRoot, 'examples/asciiMaze/browser-entry.ts')],
  bundle: true,
  outfile: resolve(repositoryRoot, 'docs/assets/ascii-maze.bundle.js'),
  platform: 'browser',
  format: 'iife',
  minify: true,
  sourcemap: true,
  external: ['fs', 'child_process', 'path'],
  logLevel: 'info',
  plugins: [
    {
      name: 'ascii-maze-browser-multi-stub',
      setup(buildContext) {
        buildContext.onResolve(
          { filter: /^\.\.\/\.\.\/\.\.\/multithreading\/multi$/ },
          (resolveArgs) => {
            const normalizedImporter = resolveArgs.importer.replaceAll(
              '\\',
              '/',
            );

            if (
              resolveArgs.path === legacyMultiImportPath &&
              normalizedImporter.endsWith(legacyMultiImporterSuffix)
            ) {
              return { path: browserMultiStubPath };
            }

            return undefined;
          },
        );
      },
    },
  ],
});