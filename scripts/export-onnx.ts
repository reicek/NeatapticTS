/**
 * Minimal CLI to export a JSON ONNX model from a serialized network state.
 *
 * Usage:
 *   npm run onnx:export -- --in network.json --out model.onnx.json [--metadata] [--batch] [--legacy] [--partial] [--mixed]
 *
 * This stays intentionally lightweight. For larger automation flows, prefer
 * calling the ONNX export API directly from application code.
 */

import fs from 'node:fs';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const PRIMARY_ONNX_MODULE_PATH = path.resolve(
  'dist',
  'architecture',
  'onnx.js',
);
const FALLBACK_ONNX_MODULE_PATH = path.resolve(
  'dist',
  'architecture',
  'network',
  'network.onnx.js',
);
const NETWORK_MODULE_PATH = path.resolve('dist', 'architecture', 'network.js');

interface ExportOnnxOptions {
  includeMetadata: boolean;
  batchDimension: boolean;
  legacyNodeOrdering: boolean;
  allowPartialConnectivity: boolean;
  allowMixedActivations: boolean;
}

interface NetworkLike {
  toJSON?: () => unknown;
}

interface NetworkFactory<TNetwork extends NetworkLike> {
  fromJSON(raw: unknown): TNetwork;
}

interface OnnxExportModule<TNetwork extends NetworkLike> {
  exportToONNX: (network: TNetwork, options: ExportOnnxOptions) => unknown;
}

interface DistDependencies<TNetwork extends NetworkLike> {
  Network: NetworkFactory<TNetwork>;
  exportToONNX: OnnxExportModule<TNetwork>['exportToONNX'];
}

interface ParsedCliOptions extends ExportOnnxOptions {
  inputFile: string;
  outputFile: string;
}

/**
 * Runs the ONNX export CLI.
 *
 * @returns Promise resolved when export succeeds.
 */
async function main(): Promise<void> {
  const rawArguments = process.argv.slice(2);
  if (hasFlag(rawArguments, '--help') || hasFlag(rawArguments, '-h')) {
    printUsage();
    return;
  }

  const cliOptions = parseCliOptions(rawArguments);
  const { Network, exportToONNX } = await loadDistDependencies<NetworkLike>();
  const networkJson = readJsonFile(cliOptions.inputFile);
  const network = Network.fromJSON(networkJson);
  const onnxJson = exportToONNX(network, cliOptions);

  fs.writeFileSync(
    path.resolve(cliOptions.outputFile),
    `${JSON.stringify(onnxJson, null, 2)}\n`,
    'utf8',
  );
  console.log(`ONNX JSON written to ${cliOptions.outputFile}`);
}

/**
 * Parses CLI options from raw process arguments.
 *
 * @param rawArguments - CLI arguments after the script path.
 * @returns Parsed and validated CLI options.
 */
function parseCliOptions(rawArguments: readonly string[]): ParsedCliOptions {
  const inputFile = resolveOptionValue(rawArguments, '--in');
  const outputFile = resolveOptionValue(rawArguments, '--out');

  if (!inputFile || !outputFile) {
    printUsageAndExit('Error: --in and --out are required.');
  }

  return {
    inputFile,
    outputFile,
    includeMetadata: hasFlag(rawArguments, '--metadata'),
    batchDimension: hasFlag(rawArguments, '--batch'),
    legacyNodeOrdering: hasFlag(rawArguments, '--legacy'),
    allowPartialConnectivity: hasFlag(rawArguments, '--partial'),
    allowMixedActivations: hasFlag(rawArguments, '--mixed'),
  };
}

/**
 * Resolves one option value from `--name value` or `--name=value` syntax.
 *
 * @param rawArguments - CLI arguments after the script path.
 * @param optionName - Long option name including leading dashes.
 * @returns Option value when present.
 */
function resolveOptionValue(
  rawArguments: readonly string[],
  optionName: string,
): string | undefined {
  const exactIndex = rawArguments.indexOf(optionName);
  if (exactIndex >= 0) {
    return rawArguments[exactIndex + 1];
  }

  const inlineArgument = rawArguments.find((argument) =>
    argument.startsWith(`${optionName}=`),
  );
  return inlineArgument?.slice(optionName.length + 1);
}

/**
 * Checks whether a boolean flag is present.
 *
 * @param rawArguments - CLI arguments after the script path.
 * @param optionName - Flag name including leading dashes.
 * @returns `true` when the flag is present.
 */
function hasFlag(rawArguments: readonly string[], optionName: string): boolean {
  return rawArguments.includes(optionName);
}

/**
 * Loads the built network and ONNX export modules from `dist/`.
 *
 * @returns The built network constructor and ONNX export function.
 */
async function loadDistDependencies<TNetwork extends NetworkLike>(): Promise<
  DistDependencies<TNetwork>
> {
  const onnxModule = (await importModuleWithFallback<
    Partial<OnnxExportModule<TNetwork>>
  >(PRIMARY_ONNX_MODULE_PATH, FALLBACK_ONNX_MODULE_PATH)) as Partial<
    OnnxExportModule<TNetwork>
  >;
  const networkModule = (await import(
    pathToFileURL(NETWORK_MODULE_PATH).href
  )) as { default?: NetworkFactory<TNetwork> };

  if (!networkModule.default || typeof onnxModule.exportToONNX !== 'function') {
    throw new Error(
      'Failed to load built ONNX export dependencies from dist/.',
    );
  }

  return {
    Network: networkModule.default,
    exportToONNX: onnxModule.exportToONNX,
  };
}

/**
 * Imports the primary built module, falling back to a secondary path when the
 * public layout has shifted.
 *
 * @param primaryModulePath - Preferred built module path.
 * @param fallbackModulePath - Fallback built module path.
 * @returns Imported module namespace.
 */
async function importModuleWithFallback<TModule>(
  primaryModulePath: string,
  fallbackModulePath: string,
): Promise<TModule> {
  try {
    return (await import(pathToFileURL(primaryModulePath).href)) as TModule;
  } catch {
    return (await import(pathToFileURL(fallbackModulePath).href)) as TModule;
  }
}

/**
 * Reads and parses one JSON file.
 *
 * @param filePath - Source JSON file path.
 * @returns Parsed JSON value.
 */
function readJsonFile(filePath: string): unknown {
  return JSON.parse(fs.readFileSync(path.resolve(filePath), 'utf8')) as unknown;
}

/** Prints CLI usage text. */
function printUsage(): void {
  console.log(
    'Usage: npm run onnx:export -- --in network.json --out model.onnx.json [--metadata] [--batch] [--legacy] [--partial] [--mixed]',
  );
}

/**
 * Prints CLI usage text and exits the process with failure.
 *
 * @param message - Error message shown before usage text.
 * @returns Never returns.
 */
function printUsageAndExit(message: string): never {
  console.error(message);
  printUsage();
  process.exit(1);
}

main().catch((error: unknown) => {
  const errorMessage = error instanceof Error ? error.message : String(error);
  console.error('Export failed:', errorMessage);
  process.exit(1);
});
