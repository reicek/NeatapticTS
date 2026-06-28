import fs from 'node:fs';

import { DEFAULT_TOP_COUNT } from './analyze-trace.constants.js';
import type {
  CliOptions,
  TraceEvent,
  TraceFile,
  TraceFormat,
} from './analyze-trace.types.js';

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
 * Supports both the standard Chrome trace format (`{ traceEvents: [...] }`
 * or a bare array) and the Chrome DevTools MCP server response format
 * (`{ result: { traceEvents: [...] } }` or `{ result: [...] }`).
 * The parsed JSON is normalized to a `TraceFile` with a `traceEvents` array
 * before returning, so downstream analysis code can treat all formats
 * uniformly.
 *
 * @param traceFilePath - Absolute or relative trace file path.
 * @returns Normalized trace file payload with `traceEvents`.
 */
export function loadTrace(traceFilePath: string): TraceFile {
  const traceFileContents = fs.readFileSync(traceFilePath, 'utf8');
  const rawJson: unknown = JSON.parse(traceFileContents);
  return normalizeTraceFile(rawJson);
}

/**
 * Detects the trace format from a parsed JSON value.
 *
 * The standard Chrome trace format is either a bare array of events or
 * an object with a `traceEvents` property. The Chrome DevTools MCP server
 * wraps trace data inside a `result` property — either as a bare array
 * or as `{ traceEvents: [...] }`.
 *
 * @param rawJson - Parsed JSON value from the trace file.
 * @returns `'standard'` for raw Chrome traces, `'devtools'` for
 *   MCP server responses.
 */
export function detectTraceFormat(rawJson: unknown): TraceFormat {
  if (Array.isArray(rawJson)) {
    return 'standard';
  }
  if (typeof rawJson !== 'object' || rawJson === null) {
    return 'standard';
  }
  const obj = rawJson as Record<string, unknown>;
  if (obj.result !== undefined) {
    return 'devtools';
  }
  return 'standard';
}

/**
 * Normalizes a parsed JSON value into a `TraceFile` with `traceEvents`.
 *
 * For the standard format, a bare array is wrapped into `{ traceEvents: [...] }`
 * and an object with `traceEvents` is returned as-is. For the MCP format,
 * the `result` wrapper is unwrapped: if `result` is an array it becomes
 * `traceEvents` directly; if `result` is an object its `traceEvents` property
 * is extracted.
 *
 * @param rawJson - Parsed JSON value from the trace file.
 * @returns Normalized `TraceFile` with a `traceEvents` array (which may be
 *   `undefined` if the source data lacks events).
 */
export function normalizeTraceFile(rawJson: unknown): TraceFile {
  const format = detectTraceFormat(rawJson);
  if (format === 'devtools') {
    const obj = rawJson as Record<string, unknown>;
    const result = obj.result;
    if (Array.isArray(result)) {
      return { traceEvents: result as TraceEvent[] };
    }
    const resultObj = (result ?? {}) as Record<string, unknown>;
    return { traceEvents: resultObj.traceEvents as TraceEvent[] | undefined };
  }
  if (Array.isArray(rawJson)) {
    return { traceEvents: rawJson as TraceEvent[] };
  }
  return rawJson as TraceFile;
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
