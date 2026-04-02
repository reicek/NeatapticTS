/**
 * Reads a Chrome or Perfetto trace export and prints a compact performance
 * report.
 *
 * The entrypoint stays intentionally small: CLI parsing, trace loading,
 * analysis, and report rendering each live in focused sibling modules.
 *
 * Usage:
 *   npm run trace:analyze -- examples/flappy_bird/Trace-20260309T191949.json
 *   npm run docs:build-scripts && node dist-docs/scripts/analyze-trace/analyze-trace.js path/to/trace.json --top=20
 */
import path from 'node:path';

import { analyzeTraceEvents } from './analyze-trace.analysis.js';
import { loadTrace, resolveCliOptions } from './analyze-trace.io.js';
import { printTraceReport } from './analyze-trace.report.js';

/**
 * CLI entry point.
 *
 * @returns Nothing.
 */
function main(): void {
  // Step 1: Resolve the CLI request and load the raw trace.
  const cliOptions = resolveCliOptions(process.argv.slice(2));
  const traceFilePath = path.resolve(cliOptions.tracePath);
  const traceFile = loadTrace(traceFilePath);
  const traceEvents = traceFile.traceEvents ?? [];

  if (traceEvents.length === 0) {
    throw new Error(`Trace contains no events: ${traceFilePath}`);
  }

  // Step 2: Derive deterministic report data from the raw trace.
  const traceAnalysis = analyzeTraceEvents({
    traceFilePath,
    traceEvents,
    topCount: cliOptions.topCount,
  });

  // Step 3: Render the final textual report.
  printTraceReport(traceAnalysis);
}

main();
