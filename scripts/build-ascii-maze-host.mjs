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