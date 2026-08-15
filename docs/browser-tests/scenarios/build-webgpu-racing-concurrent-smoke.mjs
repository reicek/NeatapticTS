import { build } from 'esbuild';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const projectRoot = path.resolve(__dirname, '../../..');
const scenarioEntryPath = path.resolve(
  __dirname,
  'webgpu-racing-concurrent-smoke.ts',
);
const outDirectory = __dirname;
const envDirectoryPath = path.resolve(projectRoot, 'src', 'env');
const browserWorkerLoaderPath = path.resolve(
  envDirectoryPath,
  'browser',
  'worker-loader.ts',
);

const NODE_BUILTINS = [
  'child_process',
  'node:child_process',
  'fs',
  'node:fs',
  'path',
  'node:path',
  'worker_threads',
  'node:worker_threads',
  'net',
  'node:net',
  'tls',
  'node:tls',
  'os',
  'node:os',
  'util',
  'node:util',
  'crypto',
  'node:crypto',
];

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
        return { path: browserWorkerLoaderPath };
      }
      return null;
    });
  },
};

const nodeBuiltinStubPlugin = {
  name: 'node-builtin-stub',
  setup(buildContext) {
    buildContext.onResolve(
      { filter: new RegExp(`^(${NODE_BUILTINS.join('|').replace(/:/g, '\\\\:')})$`) },
      (args) => {
        if (NODE_BUILTINS.includes(args.path)) {
          return {
            path: args.path,
            namespace: 'node-builtin-stub',
          };
        }
        return null;
      },
    );
    buildContext.onLoad(
      { filter: /.*/, namespace: 'node-builtin-stub' },
      () => {
        return {
          contents:
            'export default {};\nexport const fork = () => { throw new Error("child_process is unavailable in the browser"); };\nexport const spawnSync = () => { throw new Error("child_process is unavailable in the browser"); };\nexport const join = () => { throw new Error("path is unavailable in the browser"); };\n',
          loader: 'js',
        };
      },
    );
  },
};

await build({
  bundle: true,
  entryPoints: [scenarioEntryPath],
  format: 'esm',
  logLevel: 'info',
  outfile: path.resolve(
    outDirectory,
    'webgpu-racing-concurrent-smoke.bundle.mjs',
  ),
  platform: 'browser',
  plugins: [browserEnvAliasPlugin, nodeBuiltinStubPlugin],
  sourcemap: false,
  target: 'es2023',
  treeShaking: true,
});
