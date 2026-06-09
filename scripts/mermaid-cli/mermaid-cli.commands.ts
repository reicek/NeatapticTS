/**
 * @module mermaid-cli/mermaid-cli.commands
 *
 * High-level command runners and argument builders for the Mermaid CLI wrapper.
 *
 * Each command runner takes the fully-resolved {@link MermaidCommandContext},
 * builds a {@link MermaidCliInvocation} via the puppeteer module, delegates to
 * the process module for the actual mmdc subprocess, and handles temporary
 * resource cleanup in a `finally` block.
 *
 * Argument builders are kept pure (no side effects) so they can be tested
 * independently of process I/O.
 */

import path from 'node:path';
import { buildMermaidCliInvocation } from './mermaid-cli.puppeteer.js';
import {
  cleanupInvocation,
  createMermaidTempDirectory,
  ensureParentDirectoryExists,
  runMermaidCli,
} from './mermaid-cli.process.js';
import { EXPORT_COMMAND, VALIDATE_COMMAND } from './mermaid-cli.constants.js';
import type {
  MermaidCommandContext,
  ParsedArguments,
} from './mermaid-cli.types.js';

/**
 * Executes the requested Mermaid command by delegating to the appropriate
 * sub-command runner.
 *
 * @param commandContext - Fully resolved wrapper command context.
 * @returns Resolves after the requested command completes.
 */
export async function runCommand(
  commandContext: MermaidCommandContext,
): Promise<void> {
  if (commandContext.command === VALIDATE_COMMAND) {
    await runValidateCommand(commandContext);
    return;
  }

  await runExportCommand(commandContext);
}

/**
 * Validates a Mermaid diagram by rendering it into a temporary SVG.
 *
 * The temporary output directory and any Puppeteer config temp directory are
 * removed in a `finally` block regardless of success or failure.
 *
 * @param commandContext - Fully resolved wrapper command context.
 * @returns Resolves after the validation render completes.
 */
export async function runValidateCommand(
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
 * Exports a Mermaid diagram to the caller-supplied output path.
 *
 * The parent directory of the output path is created recursively when absent.
 * Any Puppeteer config temp directory is removed in a `finally` block.
 *
 * @param commandContext - Fully resolved wrapper command context.
 * @returns Resolves after the export completes.
 */
export async function runExportCommand(
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
 * Builds Mermaid CLI arguments for validation mode.
 *
 * Delegates to {@link buildCommandArguments} — exists as a named entry point
 * for readability at the call site.
 *
 * @param inputPath - Mermaid input path.
 * @param temporaryOutputPath - Temporary validation output path.
 * @param parsedArguments - Parsed CLI arguments.
 * @returns Mermaid CLI argument list.
 */
export function buildValidateArguments(
  inputPath: string,
  temporaryOutputPath: string,
  parsedArguments: ParsedArguments,
): string[] {
  return buildCommandArguments(inputPath, temporaryOutputPath, parsedArguments);
}

/**
 * Builds Mermaid CLI arguments for export mode.
 *
 * Delegates to {@link buildCommandArguments} — exists as a named entry point
 * for readability at the call site.
 *
 * @param inputPath - Mermaid input path.
 * @param outputPath - Export output path.
 * @param parsedArguments - Parsed CLI arguments.
 * @returns Mermaid CLI argument list.
 */
export function buildExportArguments(
  inputPath: string,
  outputPath: string,
  parsedArguments: ParsedArguments,
): string[] {
  return buildCommandArguments(inputPath, outputPath, parsedArguments);
}

/**
 * Builds the common Mermaid CLI argument list for input/output operations.
 *
 * Named `--input` and `--output` arguments are pinned explicitly; all other
 * named options and positional pass-throughs are appended via
 * {@link buildPassthroughArguments}.
 *
 * @param inputPath - Mermaid input path.
 * @param outputPath - Mermaid output path.
 * @param parsedArguments - Parsed CLI arguments.
 * @returns Mermaid CLI argument list.
 */
export function buildCommandArguments(
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
 * Named options whose keys are in `excludedNames` are dropped; all remaining
 * named options are expanded to `--name value` pairs and concatenated with the
 * positional pass-through tokens.
 *
 * @param parsedArguments - Parsed CLI arguments.
 * @param options - Pass-through filtering options.
 * @param options.excludedNames - Set of named argument keys to suppress.
 * @returns Filtered and expanded argument list.
 */
export function buildPassthroughArguments(
  parsedArguments: ParsedArguments,
  options: { excludedNames: ReadonlySet<string> },
): string[] {
  const namedArguments = Object.entries(parsedArguments.named)
    .filter(([name]) => !options.excludedNames.has(name))
    .flatMap(([name, value]) => buildNamedArgumentPair(name, value));

  return [...namedArguments, ...parsedArguments.passthrough];
}

/**
 * Builds a Mermaid CLI named argument token pair.
 *
 * Boolean flags (where the stored value is `'true'`) are emitted as a single
 * `--name` token.  All other values are emitted as `['--name', 'value']`.
 *
 * @param name - CLI argument name without leading dashes.
 * @param value - CLI argument value.
 * @returns One or two element string array.
 */
export function buildNamedArgumentPair(name: string, value: string): string[] {
  if (value === 'true') {
    return [`--${name}`];
  }

  return [`--${name}`, value];
}

/**
 * Writes a `[mermaid]` prefixed success message to stdout confirming the
 * validated Mermaid diagram path.
 *
 * @param inputPath - Path to the successfully validated Mermaid diagram file.
 * @returns Nothing.
 */
export function logValidDiagram(inputPath: string): void {
  console.log(`[mermaid] Valid diagram: ${inputPath}`);
}

/**
 * Writes a `[mermaid]` prefixed success message to stdout confirming the
 * exported diagram output path.
 *
 * @param outputPath - Path to the successfully exported Mermaid output file.
 * @returns Nothing.
 */
export function logExportedDiagram(outputPath: string): void {
  console.log(`[mermaid] Exported diagram to ${outputPath}`);
}

/**
 * Prints wrapper usage information to stderr and exits the process with code 1.
 *
 * Typed as `never` because callers use it as a control-flow assertion (the
 * function unconditionally terminates the process).
 *
 * @param message - Error message to show before usage text.
 * @returns Never returns.
 */
export function printUsageAndExit(message: string): never {
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
