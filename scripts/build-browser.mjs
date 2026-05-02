import { build } from 'esbuild';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '..');
const browserEntryPath = path.resolve(projectRoot, 'src', 'browser-entry.ts');
const distDirectory = path.resolve(projectRoot, 'dist');
const envDirectoryPath = path.resolve(projectRoot, 'src', 'env');
const browserWorkerLoaderPath = path.resolve(
  envDirectoryPath,
  'browser',
  'worker-loader.ts',
);
const shouldBuildOnlyMinifiedIife = process.argv.includes('--minify-only');

const browserEnvAliasPlugin = {
  name: 'browser-env-alias',
  setup(buildContext) {
    buildContext.onResolve({ filter: /^\./ }, (args) => {
      const resolvedPath = path.resolve(args.resolveDir, args.path);

      if (
        resolvedPath === envDirectoryPath ||
        resolvedPath === path.resolve(envDirectoryPath, 'index') ||
        resolvedPath === path.resolve(envDirectoryPath, 'index.ts')
      ) {
        return {
          path: browserWorkerLoaderPath,
        };
      }

      return null;
    });
  },
};

const sharedBuildOptions = {
  bundle: true,
  entryPoints: [browserEntryPath],
  logLevel: 'info',
  platform: 'browser',
  plugins: [browserEnvAliasPlugin],
  sourcemap: true,
  target: 'es2023',
  treeShaking: true,
};

const buildVariants = shouldBuildOnlyMinifiedIife
  ? [
      {
        format: 'iife',
        globalName: 'Neataptic',
        minify: true,
        outfile: path.resolve(distDirectory, 'neataptic.browser.iife.min.js'),
      },
    ]
  : [
      {
        format: 'esm',
        outfile: path.resolve(distDirectory, 'neataptic.browser.esm.js'),
      },
      {
        format: 'iife',
        globalName: 'Neataptic',
        minify: false,
        outfile: path.resolve(distDirectory, 'neataptic.browser.iife.js'),
      },
      {
        format: 'iife',
        globalName: 'Neataptic',
        minify: true,
        outfile: path.resolve(distDirectory, 'neataptic.browser.iife.min.js'),
      },
    ];

for (const buildVariant of buildVariants) {
  await build({
    ...sharedBuildOptions,
    ...buildVariant,
  });
}