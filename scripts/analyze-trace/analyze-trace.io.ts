import fs from 'node:fs';

import { DEFAULT_TOP_COUNT } from './analyze-trace.constants.js';
import type { CliOptions, TraceFile } from './analyze-trace.types.js';

/**
 * Resolves CLI flags for one trace-analyzer invocation.
 *
 * Supported inputs:
 * - positional trace path
 * - `--top=NUMBER`
 * - `--top NUMBER`
 *
 * @param argumentValues - CLI arguments after the script path.
 * @returns Parsed CLI options.
 */
export function resolveCliOptions(
  argumentValues: readonly string[],
): CliOptions {
  const pathArgument = argumentValues.find(
    (argument) => !argument.startsWith('--'),
  );
  if (!pathArgument) {
    throw new Error(
      'Missing trace path. Example: npm run trace:analyze -- examples/flappy_bird/Trace.json',
    );
  }

  return {
    tracePath: pathArgument,
    topCount: resolveRequestedTopCount(argumentValues),
  };
}

/**
 * Reads and parses one trace JSON file.
 *
 * @param traceFilePath - Absolute or relative trace file path.
 * @returns Parsed trace file payload.
 */
export function loadTrace(traceFilePath: string): TraceFile {
  const traceFileContents = fs.readFileSync(traceFilePath, 'utf8');
  return JSON.parse(traceFileContents) as TraceFile;
}

/**
 * Resolves the requested top-count override from CLI flags.
 *
 * @param argumentValues - CLI arguments after the script path.
 * @returns Positive top-count value or the repo default.
 */
function resolveRequestedTopCount(argumentValues: readonly string[]): number {
  const inlineTopArgument = argumentValues.find((argument) =>
    argument.startsWith('--top='),
  );
  const splitTopArgumentIndex = argumentValues.findIndex(
    (argument) => argument === '--top',
  );

  const requestedTopCount = inlineTopArgument
    ? Number.parseInt(inlineTopArgument.slice('--top='.length), 10)
    : splitTopArgumentIndex >= 0
      ? Number.parseInt(argumentValues[splitTopArgumentIndex + 1] ?? '', 10)
      : DEFAULT_TOP_COUNT;

  return Number.isFinite(requestedTopCount) && requestedTopCount > 0
    ? requestedTopCount
    : DEFAULT_TOP_COUNT;
}
