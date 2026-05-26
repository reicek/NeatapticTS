/**
 * @module mermaid-cli/mermaid-cli.args
 *
 * CLI argument parsing for the Mermaid wrapper.
 *
 * The parser is a single-pass linear scan: each token is classified as either
 * a `--named` flag or a positional pass-through value. Named arguments may
 * carry their value inline (`--name=value`) or as a separate token
 * (`--name value`). When the next token is absent or is itself a flag the
 * argument is treated as a boolean flag whose value is resolved to `'true'`.
 */

import type { ParsedArgument, ParsedArguments } from './mermaid-cli.types.js';

/**
 * Parses raw CLI arguments into named options and pass-through values.
 *
 * @param rawArguments - Raw CLI arguments for the wrapper command.
 * @returns The parsed argument result.
 */
export function parseArguments(rawArguments: string[]): ParsedArguments {
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
 * Classifies a single CLI argument token as either a named flag or a
 * positional passthrough value and returns the appropriate parsed packet.
 *
 * @param rawArguments - Full raw argument list.
 * @param argumentIndex - Current argument index.
 * @returns Parsed argument packet with kind and consumed-extra-arguments count.
 */
export function parseSingleArgument(
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
 * Extracts the name and value from a `--name value` or `--name=value` token
 * at the given index, consuming one extra token for the space-separated form.
 *
 * @param rawArguments - Full raw argument list.
 * @param argumentIndex - Index of the `--name` token to parse.
 * @returns Parsed named argument packet with resolved name, value, and consumed count.
 */
export function parseNamedArgument(
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
 * Mutates `parsedArguments` in-place by appending a passthrough value or
 * recording a named key-value pair from the given parsed packet.
 *
 * @param parsedArguments - Mutable parsed argument result accumulator.
 * @param parsedArgument - Parsed argument packet to apply to the accumulator.
 * @returns Nothing; the accumulator is mutated in-place.
 */
export function applyParsedArgument(
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
 * Resolves the first available named value from a list of aliases.
 *
 * @param parsedArguments - Parsed CLI arguments.
 * @param aliases - Named aliases to search in priority order.
 * @returns The resolved named value, or `undefined` when no alias matched.
 */
export function resolveNamedValue(
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
 * @returns `true` when the token starts with `--`.
 */
export function isNamedArgument(argument: string | undefined): boolean {
  return argument?.startsWith('--') ?? false;
}

/**
 * Determines whether the next token should leave a named argument as a boolean
 * flag.
 *
 * @param nextArgument - Next raw CLI token.
 * @returns `true` when the argument should be treated as a boolean flag.
 */
export function shouldTreatAsBooleanFlag(
  nextArgument: string | undefined,
): boolean {
  return !nextArgument || nextArgument.startsWith('--');
}
