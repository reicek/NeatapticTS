import { spawn } from 'node:child_process';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

const EXPORT_COMMAND = 'export';
const VALIDATE_COMMAND = 'validate';
const SUPPORTED_COMMANDS = new Set<MermaidCommand>([
  EXPORT_COMMAND,
  VALIDATE_COMMAND,
]);
const MERMAID_TEMP_DIRECTORY_PREFIX = 'neatapticts-mermaid-';
const PUPPETEER_TEMP_DIRECTORY_PREFIX = 'neatapticts-mermaid-puppeteer-';
const PUPPETEER_CONFIG_FILE_NAME = 'puppeteer-config.json';
const PUPPETEER_CI_LINUX_ARGS = ['--no-sandbox', '--disable-setuid-sandbox'];

type MermaidCommand = typeof EXPORT_COMMAND | typeof VALIDATE_COMMAND;

interface ParsedArguments {
  named: Record<string, string>;
  passthrough: string[];
}

interface MermaidCommandContext {
  command: MermaidCommand;
  parsedArguments: ParsedArguments;
  cliPath: string;
  inputPath: string;
  outputPath?: string;
}

interface MermaidCliInvocation {
  argumentsToPass: string[];
  cleanup: () => Promise<void>;
}

type ParsedArgument =
  | {
      kind: 'named';
      name: string;
      value: string;
      consumedExtraArguments: number;
    }
  | {
      kind: 'passthrough';
      passthroughValue: string;
      consumedExtraArguments: number;
    };

await main();

/**
 * Runs the Mermaid wrapper entrypoint.
 *
 * @returns Resolves after the requested Mermaid command completes.
 */
async function main(): Promise<void> {
  const commandContext = buildCommandContext(process.argv.slice(2));
  await runCommand(commandContext);
}

/**
 * Executes the requested Mermaid command.
 *
 * @param commandContext - Resolved wrapper command context.
 * @returns Resolves after the requested command completes.
 */
async function runCommand(
  commandContext: MermaidCommandContext,
): Promise<void> {
  if (commandContext.command === VALIDATE_COMMAND) {
    await runValidateCommand(commandContext);
    return;
  }

  await runExportCommand(commandContext);
}

/**
 * Validates a Mermaid input by rendering it into a temporary SVG.
 *
 * @param commandContext - Resolved wrapper command context.
 * @returns Resolves after the validation render completes.
 */
async function runValidateCommand(
  commandContext: MermaidCommandContext,
): Promise<void> {
  const temporaryValidationDirectoryPath = await createMermaidTempDirectory();
  const temporaryOutputPath = path.join(
    temporaryValidationDirectoryPath,
    'diagram.svg',
  );

  const mermaidCliInvocation = await buildMermaidCliInvocation(
    commandContext.parsedArguments,
    buildValidateArguments(
      commandContext.inputPath,
      temporaryOutputPath,
      commandContext.parsedArguments,
    ),
  );

  try {
    await ensureParentDirectoryExists(temporaryOutputPath);
    await runMermaidCli(
      commandContext.cliPath,
      mermaidCliInvocation.argumentsToPass,
    );
    logValidDiagram(commandContext.inputPath);
  } finally {
    await cleanupInvocation(
      mermaidCliInvocation,
      temporaryValidationDirectoryPath,
    );
  }
}

/**
 * Exports a Mermaid input to the requested output path.
 *
 * @param commandContext - Resolved wrapper command context.
 * @returns Resolves after the export completes.
 */
async function runExportCommand(
  commandContext: MermaidCommandContext,
): Promise<void> {
  const outputPath = commandContext.outputPath;
  if (!outputPath) {
    printUsageAndExit('Missing required --output argument for export.');
  }

  await ensureParentDirectoryExists(outputPath);

  const mermaidCliInvocation = await buildMermaidCliInvocation(
    commandContext.parsedArguments,
    buildExportArguments(
      commandContext.inputPath,
      outputPath,
      commandContext.parsedArguments,
    ),
  );

  try {
    await runMermaidCli(
      commandContext.cliPath,
      mermaidCliInvocation.argumentsToPass,
    );
    logExportedDiagram(outputPath);
  } finally {
    await mermaidCliInvocation.cleanup();
  }
}

/**
 * Builds the resolved command context from raw CLI arguments.
 *
 * @param rawArguments - Raw CLI arguments following the wrapper executable.
 * @returns The validated command context.
 */
function buildCommandContext(rawArguments: string[]): MermaidCommandContext {
  const commandAndArguments = resolveCommandAndArguments(rawArguments);
  const parsedArguments = parseArguments(commandAndArguments.rawArgs);

  return {
    command: commandAndArguments.command,
    parsedArguments,
    cliPath: resolveMermaidCliPath(),
    inputPath: resolveRequiredInputPath(parsedArguments),
    outputPath: resolveOptionalOutputPath(
      commandAndArguments.command,
      parsedArguments,
    ),
  };
}

/**
 * Resolves the requested wrapper command and its remaining raw arguments.
 *
 * @param rawArguments - Raw CLI arguments following the wrapper executable.
 * @returns The resolved command packet.
 */
function resolveCommandAndArguments(rawArguments: string[]): {
  command: MermaidCommand;
  rawArgs: string[];
} {
  const [command = VALIDATE_COMMAND, ...remainingArguments] = rawArguments;
  ensureSupportedCommand(command);

  return {
    command,
    rawArgs: remainingArguments,
  };
}

/**
 * Parses raw CLI arguments into named options and pass-through values.
 *
 * @param rawArguments - Raw CLI arguments for the wrapper command.
 * @returns The parsed argument result.
 */
function parseArguments(rawArguments: string[]): ParsedArguments {
  const parsedArguments: ParsedArguments = { named: {}, passthrough: [] };

  for (
    let argumentIndex = 0;
    argumentIndex < rawArguments.length;
    argumentIndex += 1
  ) {
    const parsedArgument = parseSingleArgument(rawArguments, argumentIndex);
    applyParsedArgument(parsedArguments, parsedArgument);
    argumentIndex += parsedArgument.consumedExtraArguments;
  }

  return parsedArguments;
}

/**
 * Parses a single CLI argument token.
 *
 * @param rawArguments - Full raw argument list.
 * @param argumentIndex - Current argument index.
 * @returns Parsed argument packet.
 */
function parseSingleArgument(
  rawArguments: string[],
  argumentIndex: number,
): ParsedArgument {
  const argument = rawArguments[argumentIndex];
  if (!isNamedArgument(argument)) {
    return {
      kind: 'passthrough',
      passthroughValue: argument,
      consumedExtraArguments: 0,
    };
  }

  return parseNamedArgument(rawArguments, argumentIndex);
}

/**
 * Parses a named CLI argument token.
 *
 * @param rawArguments - Full raw argument list.
 * @param argumentIndex - Current argument index.
 * @returns Parsed named argument packet.
 */
function parseNamedArgument(
  rawArguments: string[],
  argumentIndex: number,
): Extract<ParsedArgument, { kind: 'named' }> {
  const normalizedArgument = rawArguments[argumentIndex].slice(2);
  const [name, inlineValue] = normalizedArgument.split('=', 2);
  if (inlineValue !== undefined) {
    return {
      kind: 'named',
      name,
      value: inlineValue,
      consumedExtraArguments: 0,
    };
  }

  const nextArgument = rawArguments[argumentIndex + 1];
  if (shouldTreatAsBooleanFlag(nextArgument)) {
    return {
      kind: 'named',
      name,
      value: 'true',
      consumedExtraArguments: 0,
    };
  }

  return {
    kind: 'named',
    name,
    value: nextArgument,
    consumedExtraArguments: 1,
  };
}

/**
 * Applies a parsed argument packet into the accumulating result.
 *
 * @param parsedArguments - Mutable parsed argument result.
 * @param parsedArgument - Parsed argument packet.
 * @returns Nothing.
 */
function applyParsedArgument(
  parsedArguments: ParsedArguments,
  parsedArgument: ParsedArgument,
): void {
  if (parsedArgument.kind === 'passthrough') {
    parsedArguments.passthrough.push(parsedArgument.passthroughValue);
    return;
  }

  parsedArguments.named[parsedArgument.name] = parsedArgument.value;
}

/**
 * Resolves the required Mermaid input path.
 *
 * @param parsedArguments - Parsed CLI arguments.
 * @returns The required input path.
 */
function resolveRequiredInputPath(parsedArguments: ParsedArguments): string {
  const inputPath = resolveNamedValue(parsedArguments, ['input', 'i']);
  if (!inputPath) {
    printUsageAndExit('Missing required --input argument.');
  }

  return inputPath;
}

/**
 * Resolves the optional export output path, requiring it only for export mode.
 *
 * @param command - Requested wrapper command.
 * @param parsedArguments - Parsed CLI arguments.
 * @returns The required export output path or `undefined` for validation mode.
 */
function resolveOptionalOutputPath(
  command: MermaidCommand,
  parsedArguments: ParsedArguments,
): string | undefined {
  if (command !== EXPORT_COMMAND) {
    return undefined;
  }

  const outputPath = resolveNamedValue(parsedArguments, ['output', 'o']);
  if (!outputPath) {
    printUsageAndExit('Missing required --output argument for export.');
  }

  return outputPath;
}

/**
 * Resolves the Mermaid CLI entry path inside local dependencies.
 *
 * @returns Absolute Mermaid CLI entry path.
 */
function resolveMermaidCliPath(): string {
  return path.resolve(
    process.cwd(),
    'node_modules',
    '@mermaid-js',
    'mermaid-cli',
    'src',
    'cli.js',
  );
}

/**
 * Builds Mermaid CLI arguments for validation mode.
 *
 * @param inputPath - Mermaid input path.
 * @param temporaryOutputPath - Temporary validation output path.
 * @param parsedArguments - Parsed CLI arguments.
 * @returns Mermaid CLI arguments.
 */
function buildValidateArguments(
  inputPath: string,
  temporaryOutputPath: string,
  parsedArguments: ParsedArguments,
): string[] {
  return buildCommandArguments(inputPath, temporaryOutputPath, parsedArguments);
}

/**
 * Builds Mermaid CLI arguments for export mode.
 *
 * @param inputPath - Mermaid input path.
 * @param outputPath - Export output path.
 * @param parsedArguments - Parsed CLI arguments.
 * @returns Mermaid CLI arguments.
 */
function buildExportArguments(
  inputPath: string,
  outputPath: string,
  parsedArguments: ParsedArguments,
): string[] {
  return buildCommandArguments(inputPath, outputPath, parsedArguments);
}

/**
 * Builds the common Mermaid CLI argument list for input/output operations.
 *
 * @param inputPath - Mermaid input path.
 * @param outputPath - Mermaid output path.
 * @param parsedArguments - Parsed CLI arguments.
 * @returns Mermaid CLI arguments.
 */
function buildCommandArguments(
  inputPath: string,
  outputPath: string,
  parsedArguments: ParsedArguments,
): string[] {
  return [
    '--input',
    inputPath,
    '--output',
    outputPath,
    ...buildPassthroughArguments(parsedArguments, {
      excludedNames: new Set(['input', 'i', 'output', 'o']),
    }),
  ];
}

/**
 * Builds pass-through Mermaid CLI arguments after removing wrapper-owned names.
 *
 * @param parsedArguments - Parsed CLI arguments.
 * @param options - Pass-through filtering options.
 * @returns Mermaid CLI pass-through arguments.
 */
function buildPassthroughArguments(
  parsedArguments: ParsedArguments,
  options: { excludedNames: ReadonlySet<string> },
): string[] {
  const namedArguments = Object.entries(parsedArguments.named)
    .filter(([name]) => !options.excludedNames.has(name))
    .flatMap(([name, value]) => buildNamedArgumentPair(name, value));

  return [...namedArguments, ...parsedArguments.passthrough];
}

/**
 * Builds the Mermaid CLI invocation, injecting CI-safe Puppeteer config when
 * needed.
 *
 * @param parsedArguments - Parsed CLI arguments.
 * @param baseArguments - Base Mermaid CLI arguments.
 * @returns Prepared Mermaid CLI invocation.
 */
async function buildMermaidCliInvocation(
  parsedArguments: ParsedArguments,
  baseArguments: string[],
): Promise<MermaidCliInvocation> {
  if (!shouldInjectCiLinuxNoSandbox(parsedArguments)) {
    return createDirectInvocation(baseArguments);
  }

  const temporaryPuppeteerDirectoryPath = await createPuppeteerTempDirectory();
  const puppeteerConfigFilePath = path.join(
    temporaryPuppeteerDirectoryPath,
    PUPPETEER_CONFIG_FILE_NAME,
  );

  await writePuppeteerConfigFile(puppeteerConfigFilePath);

  return {
    argumentsToPass: buildPuppeteerConfigArguments(
      puppeteerConfigFilePath,
      baseArguments,
    ),
    cleanup: async () => {
      await rm(temporaryPuppeteerDirectoryPath, {
        recursive: true,
        force: true,
      });
    },
  };
}

/**
 * Runs Mermaid CLI with the prepared argument list.
 *
 * @param cliPath - Mermaid CLI entry path.
 * @param argumentsToPass - Mermaid CLI arguments.
 * @returns Resolves when Mermaid CLI exits successfully.
 */
async function runMermaidCli(
  cliPath: string,
  argumentsToPass: string[],
): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const childProcess = spawn(
      process.execPath,
      [cliPath, ...argumentsToPass],
      {
        stdio: 'inherit',
      },
    );

    childProcess.once('exit', (exitCode) => {
      if (exitCode === 0) {
        resolve();
        return;
      }

      reject(new Error(`Mermaid CLI exited with code ${exitCode ?? 'null'}.`));
    });
    childProcess.once('error', reject);
  });
}

/**
 * Ensures the requested command is supported by this wrapper.
 *
 * @param command - Requested wrapper command.
 * @returns Nothing.
 */
function ensureSupportedCommand(
  command: string,
): asserts command is MermaidCommand {
  if (!SUPPORTED_COMMANDS.has(command as MermaidCommand)) {
    printUsageAndExit(`Unsupported Mermaid CLI command: ${command}`);
  }
}

/**
 * Resolves the first available named value from a list of aliases.
 *
 * @param parsedArguments - Parsed CLI arguments.
 * @param aliases - Named aliases to search in priority order.
 * @returns The resolved named value, if any.
 */
function resolveNamedValue(
  parsedArguments: ParsedArguments,
  aliases: readonly string[],
): string | undefined {
  const matchingAlias = aliases.find(
    (alias) => parsedArguments.named[alias] !== undefined,
  );

  return matchingAlias ? parsedArguments.named[matchingAlias] : undefined;
}

/**
 * Determines whether a raw token should be parsed as a named argument.
 *
 * @param argument - Raw CLI token.
 * @returns `true` when the token is a named `--flag` argument.
 */
function isNamedArgument(argument: string | undefined): boolean {
  return argument?.startsWith('--') ?? false;
}

/**
 * Determines whether the next token should leave a named argument as a boolean
 * flag.
 *
 * @param nextArgument - Next raw CLI token.
 * @returns `true` when the argument should be treated as a boolean flag.
 */
function shouldTreatAsBooleanFlag(nextArgument: string | undefined): boolean {
  return !nextArgument || nextArgument.startsWith('--');
}

/**
 * Builds Mermaid CLI named argument output.
 *
 * @param name - CLI argument name without leading dashes.
 * @param value - CLI argument value.
 * @returns Mermaid CLI argument pair.
 */
function buildNamedArgumentPair(name: string, value: string): string[] {
  if (value === 'true') {
    return [`--${name}`];
  }

  return [`--${name}`, value];
}

/**
 * Creates a direct Mermaid CLI invocation with no cleanup work.
 *
 * @param baseArguments - Base Mermaid CLI arguments.
 * @returns Direct Mermaid CLI invocation.
 */
function createDirectInvocation(baseArguments: string[]): MermaidCliInvocation {
  return {
    argumentsToPass: baseArguments,
    cleanup: async () => {},
  };
}

/**
 * Creates a temporary Mermaid output directory.
 *
 * @returns Temporary Mermaid output directory path.
 */
async function createMermaidTempDirectory(): Promise<string> {
  return mkdtemp(path.join(os.tmpdir(), MERMAID_TEMP_DIRECTORY_PREFIX));
}

/**
 * Creates a temporary Puppeteer config directory.
 *
 * @returns Temporary Puppeteer config directory path.
 */
async function createPuppeteerTempDirectory(): Promise<string> {
  return mkdtemp(path.join(os.tmpdir(), PUPPETEER_TEMP_DIRECTORY_PREFIX));
}

/**
 * Writes the injected Puppeteer config file used for Linux CI browser launches.
 *
 * @param puppeteerConfigFilePath - Target Puppeteer config path.
 * @returns Resolves when the config file is written.
 */
async function writePuppeteerConfigFile(
  puppeteerConfigFilePath: string,
): Promise<void> {
  await writeFile(
    puppeteerConfigFilePath,
    JSON.stringify({ args: PUPPETEER_CI_LINUX_ARGS }),
  );
}

/**
 * Builds Mermaid CLI arguments that reference the injected Puppeteer config
 * file.
 *
 * @param puppeteerConfigFilePath - Injected Puppeteer config path.
 * @param baseArguments - Base Mermaid CLI arguments.
 * @returns Mermaid CLI arguments.
 */
function buildPuppeteerConfigArguments(
  puppeteerConfigFilePath: string,
  baseArguments: string[],
): string[] {
  return ['--puppeteerConfigFile', puppeteerConfigFilePath, ...baseArguments];
}

/**
 * Determines whether the Linux CI sandbox workaround should be injected.
 *
 * @param parsedArguments - Parsed CLI arguments.
 * @returns `true` when the wrapper should inject a Puppeteer config.
 */
function shouldInjectCiLinuxNoSandbox(
  parsedArguments: ParsedArguments,
): boolean {
  return (
    isLinuxPlatform() &&
    isContinuousIntegrationEnvironment() &&
    !hasExplicitPuppeteerConfig(parsedArguments)
  );
}

/**
 * Determines whether the current process is running on Linux.
 *
 * @returns `true` when running on Linux.
 */
function isLinuxPlatform(): boolean {
  return process.platform === 'linux';
}

/**
 * Determines whether the current process is running in CI.
 *
 * @returns `true` when the environment matches CI or GitHub Actions.
 */
function isContinuousIntegrationEnvironment(): boolean {
  return process.env.CI === 'true' || process.env.GITHUB_ACTIONS === 'true';
}

/**
 * Determines whether the caller already supplied a Puppeteer config override.
 *
 * @param parsedArguments - Parsed CLI arguments.
 * @returns `true` when an explicit Puppeteer config is already present.
 */
function hasExplicitPuppeteerConfig(parsedArguments: ParsedArguments): boolean {
  return (
    parsedArguments.named.puppeteerConfigFile !== undefined ||
    parsedArguments.named.p !== undefined ||
    parsedArguments.passthrough.includes('--puppeteerConfigFile') ||
    parsedArguments.passthrough.includes('-p')
  );
}

/**
 * Ensures the parent directory for a file path exists.
 *
 * @param filePath - Target file path.
 * @returns Resolves when the parent directory exists.
 */
async function ensureParentDirectoryExists(filePath: string): Promise<void> {
  await mkdir(path.dirname(path.resolve(filePath)), { recursive: true });
}

/**
 * Cleans up a prepared Mermaid invocation and an optional extra temp directory.
 *
 * @param mermaidCliInvocation - Prepared Mermaid invocation.
 * @param extraDirectoryPath - Additional temporary directory path.
 * @returns Resolves after cleanup completes.
 */
async function cleanupInvocation(
  mermaidCliInvocation: MermaidCliInvocation,
  extraDirectoryPath: string,
): Promise<void> {
  await mermaidCliInvocation.cleanup();
  await rm(extraDirectoryPath, { recursive: true, force: true });
}

/**
 * Logs a successful Mermaid validation.
 *
 * @param inputPath - Validated Mermaid input path.
 * @returns Nothing.
 */
function logValidDiagram(inputPath: string): void {
  console.log(`[mermaid] Valid diagram: ${inputPath}`);
}

/**
 * Logs a successful Mermaid export.
 *
 * @param outputPath - Mermaid export output path.
 * @returns Nothing.
 */
function logExportedDiagram(outputPath: string): void {
  console.log(`[mermaid] Exported diagram to ${outputPath}`);
}

/**
 * Prints wrapper usage information and exits the process.
 *
 * @param message - Error message shown before usage text.
 * @returns Never returns.
 */
function printUsageAndExit(message: string): never {
  console.error(`[mermaid] ${message}`);
  console.error(
    [
      'Usage:',
      '  npm run docs:mermaid:validate -- --input path/to/diagram.mmd [extra mmdc args]',
      '  npm run docs:mermaid:export -- --input path/to/diagram.mmd --output path/to/diagram.svg [extra mmdc args]',
    ].join('\n'),
  );
  process.exit(1);
}
