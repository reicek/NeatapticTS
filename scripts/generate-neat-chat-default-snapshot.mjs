/**
 * @fileoverview CLI driver for generating the default NEATchat pretrained snapshot.
 *
 * This script:
 * 1. Bundles `examples/neatChat/generate-default-pretrained-session-snapshot.ts`
 *    with esbuild (ESM, Node platform) into a temporary file under `dist-docs/`.
 * 2. Re-invokes the current Node.js process with the bundled entry point,
 *    forwarding any CLI arguments and optionally translating `npm_config_*`
 *    environment variables into `--optionName=value` CLI flags.
 *
 * ### `npm_config_*` env-var forwarding
 *
 * When run via `npm run <script> -- --key=value`, npm automatically sets
 * `npm_config_key=value` in the environment. The
 * {@link resolveForwardedGenerationArguments} helper reads these env vars and
 * maps them through {@link SUPPORTED_NPM_CONFIG_ARGUMENTS} before spawning the
 * bundled generator, so options can be supplied as either:
 *
 * ```sh
 * node scripts/generate-neat-chat-default-snapshot.mjs --progress=true
 * # or via npm config:
 * npm run generate-snapshot --progress=true
 * ```
 *
 * Arguments already supplied on the command line take precedence and are
 * never duplicated by the env-var translation pass.
 */
import { spawnSync } from 'node:child_process';

import { build } from 'esbuild';

/**
 * Maps lowercase `npm_config_*` environment variable suffixes to their
 * corresponding camelCase CLI flag names consumed by the generator.
 *
 * Keys are lowercase (matching the suffix in `npm_config_<key>`);
 * values are the `--flagName=<value>` option names forwarded to the bundled
 * generator process.
 *
 * @example
 * // `npm_config_generationseed=42` in env → `--generationSeed=42` forwarded
 */
const SUPPORTED_NPM_CONFIG_ARGUMENTS = {
  contextwindowtokencount: 'contextWindowTokenCount',
  extrareinforcementpasses: 'extraReinforcementPasses',
  exportname: 'exportName',
  generationseed: 'generationSeed',
  maxcasesperphase: 'maxCasesPerPhase',
  maxsourcelines: 'maxSourceLines',
  outputfilename: 'outputFileName',
  progress: 'progress',
  topwordlimit: 'topWordLimit',
  validationlinecount: 'validationLineCount',
};

const bundledGeneratorPath =
  'dist-docs/scripts/generate-neat-chat-default-snapshot.js';
const forwardedGenerationArguments = resolveForwardedGenerationArguments();

await build({
  entryPoints: [
    'examples/neatChat/generate-default-pretrained-session-snapshot.ts',
  ],
  bundle: true,
  platform: 'node',
  format: 'esm',
  outfile: bundledGeneratorPath,
});

const generationResult = spawnSync(
  process.execPath,
  [bundledGeneratorPath, ...forwardedGenerationArguments],
  {
    stdio: 'inherit',
  },
);

if (generationResult.error) {
  throw generationResult.error;
}

process.exit(generationResult.status ?? 1);

/**
 * Builds the argument list to forward to the bundled generator process.
 *
 * Steps:
 * 1. Starts from the raw CLI arguments supplied to this script (`process.argv.slice(2)`).
 * 2. Collects the option names that are already present to avoid duplicates.
 * 3. For each entry in {@link SUPPORTED_NPM_CONFIG_ARGUMENTS} whose corresponding
 *    `npm_config_*` env var is set and whose CLI option was not already supplied,
 *    appends `--optionName=envValue` to the argument list.
 * 4. Returns the merged, deduplicated argument array.
 *
 * @returns Array of `--key=value` strings suitable for passing to `spawnSync`.
 */
function resolveForwardedGenerationArguments() {
  const cliArguments = process.argv.slice(2);
  const resolvedArguments = [...cliArguments];
  const forwardedOptionNames = new Set(
    cliArguments
      .map(resolveArgumentOptionName)
      .filter((optionName) => optionName !== null),
  );

  for (const [npmConfigName, optionName] of Object.entries(
    SUPPORTED_NPM_CONFIG_ARGUMENTS,
  ).toSorted(([leftName], [rightName]) => leftName.localeCompare(rightName))) {
    if (forwardedOptionNames.has(optionName)) {
      continue;
    }

    const npmConfigValue = process.env[`npm_config_${npmConfigName}`];

    if (npmConfigValue === undefined) {
      continue;
    }

    resolvedArguments.push(`--${optionName}=${npmConfigValue}`);
  }

  return resolvedArguments;
}

/**
 * Extracts the option name from a `--optionName=value` CLI argument string.
 *
 * Returns `null` when the argument does not start with `--` (not a flag) or
 * contains no `=` separator (a bare flag like `--help` is not an option-value
 * pair and should not prevent env-var forwarding for an identically-named key).
 *
 * @param argument - Single CLI argument string from `process.argv`.
 * @returns Option name without the leading `--`, or `null` when not applicable.
 */
function resolveArgumentOptionName(argument) {
  if (!argument.startsWith('--')) {
    return null;
  }

  const separatorIndex = argument.indexOf('=');

  if (separatorIndex < 0) {
    return null;
  }

  return argument.slice(2, separatorIndex);
}
