/**
 * @module mermaid-cli/mermaid-cli.context
 *
 * Builds the fully-resolved {@link MermaidCommandContext} from raw process
 * arguments. All validation that can be performed before process I/O is
 * performed here so that downstream command runners receive a clean, typed
 * context.
 */

import path from 'node:path';
import { parseArguments } from './mermaid-cli.args.js';
import { resolveNamedValue } from './mermaid-cli.args.js';
import {
  EXPORT_COMMAND,
  SUPPORTED_COMMANDS,
  VALIDATE_COMMAND,
} from './mermaid-cli.constants.js';
import type {
  MermaidCommand,
  MermaidCommandContext,
  ParsedArguments,
} from './mermaid-cli.types.js';
import { printUsageAndExit } from './mermaid-cli.commands.js';

/**
 * Constructs a fully-resolved command context by parsing raw process arguments,
 * validating the command, and resolving required and optional path values.
 *
 * @param rawArguments - Raw CLI arguments following the wrapper executable.
 * @returns The validated, fully-resolved command context ready for execution.
 */
export function buildCommandContext(
  rawArguments: string[],
): MermaidCommandContext {
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
 * @returns The resolved command packet containing the command and remaining args.
 */
export function resolveCommandAndArguments(rawArguments: string[]): {
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
 * Extracts the `--input` or `-i` value from parsed arguments and exits the
 * process with a usage message when the argument is absent.
 *
 * @param parsedArguments - Parsed CLI arguments to search for the input path.
 * @returns The resolved required input path string.
 */
export function resolveRequiredInputPath(
  parsedArguments: ParsedArguments,
): string {
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
 * @returns The required export output path, or `undefined` in validate mode.
 */
export function resolveOptionalOutputPath(
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
 * Constructs the absolute path to the `@mermaid-js/mermaid-cli` entry point
 * inside the local `node_modules` directory relative to the process CWD.
 *
 * @returns Absolute path to the mermaid-cli package entry point.
 */
export function resolveMermaidCliPath(): string {
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
 * Asserts that the requested command is supported by this wrapper.
 * Exits the process with usage text when the command is unrecognised.
 *
 * @param command - Requested wrapper command string.
 * @returns Nothing (narrows the type to {@link MermaidCommand} on success).
 */
export function ensureSupportedCommand(
  command: string,
): asserts command is MermaidCommand {
  if (!SUPPORTED_COMMANDS.has(command as MermaidCommand)) {
    printUsageAndExit(`Unsupported Mermaid CLI command: ${command}`);
  }
}
