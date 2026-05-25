/**
 * @module mermaid-cli/mermaid-cli.types
 *
 * Shared type aliases and interfaces for the Mermaid CLI wrapper.
 * All runtime values are parameterised through these contracts so that
 * the remaining modules stay free of structural duplication.
 */

import { EXPORT_COMMAND, VALIDATE_COMMAND } from './mermaid-cli.constants.js';

/** String literal union of the two sub-commands the Mermaid CLI wrapper currently supports. */
export type MermaidCommand = typeof EXPORT_COMMAND | typeof VALIDATE_COMMAND;

/**
 * Result of parsing the raw CLI argument list.
 *
 * - `named`       — key/value pairs sourced from `--name value` or `--name=value` tokens.
 * - `passthrough` — positional or unrecognised tokens forwarded verbatim to mmdc.
 */
export interface ParsedArguments {
  named: Record<string, string>;
  passthrough: string[];
}

/**
 * Fully resolved command context built from the raw process arguments.
 * Downstream command runners receive this and do not re-parse the CLI.
 */
export interface MermaidCommandContext {
  /** Which wrapper sub-command to run. */
  command: MermaidCommand;
  /** Parsed arguments ready for forwarding and named-value lookup. */
  parsedArguments: ParsedArguments;
  /** Absolute path to the `@mermaid-js/mermaid-cli` entry point. */
  cliPath: string;
  /** Resolved value of `--input` / `-i`. */
  inputPath: string;
  /** Resolved value of `--output` / `-o`; absent in validate mode. */
  outputPath?: string;
}

/**
 * Prepared mmdc invocation produced by {@link buildMermaidCliInvocation}.
 *
 * - `argumentsToPass` — final argument list to forward to the mmdc process.
 * - `cleanup`         — async teardown for any temporary resources created
 *                       during invocation preparation (e.g. Puppeteer config).
 */
export interface MermaidCliInvocation {
  argumentsToPass: string[];
  cleanup: () => Promise<void>;
}

/**
 * Discriminated union produced by the argument parser for a single token.
 *
 * - `kind: 'named'`       — a `--flag` argument with a resolved value.
 * - `kind: 'passthrough'` — a token that should be forwarded to mmdc verbatim.
 */
export type ParsedArgument =
  | {
      kind: 'named';
      name: string;
      value: string;
      /** Number of *extra* tokens consumed (0 for inline `=`-values, 1 for separate value token). */
      consumedExtraArguments: number;
    }
  | {
      kind: 'passthrough';
      passthroughValue: string;
      consumedExtraArguments: number;
    };
