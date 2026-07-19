/**
 * @module trace-summarize
 * @description Concise performance metric extraction from Chrome DevTools trace files.
 *
 * Chrome DevTools MCP traces are frequently 10 MB or larger. This script reads a
 * trace file (raw JSON or gzip-compressed), extracts a compact set of performance
 * metrics, and returns a summary object that fits easily in an agent context
 * window (< 2000 characters when serialized).
 *
 * Supported trace formats:
 * - Standard Chrome trace: `{ traceEvents: [...] }` or bare array `[...]`
 * - Chrome DevTools MCP: `{ result: { traceEvents: [...] } }` or `{ result: [...] }`
 *
 * Usage (CLI):
 *   node scripts/trace-summarize.mjs <trace-path> [--json]
 */

import { readFile } from 'node:fs/promises';
import { gunzipSync } from 'node:zlib';
import { pathToFileURL } from 'node:url';

/** Number of trace microseconds contained in one millisecond. */
const MICROSECONDS_PER_MILLISECOND = 1_000;

/** Threshold for flagging a task as "long" (frame-budget pressure). */
const LONG_TASK_THRESHOLD_MS = 16;

/** Threshold for flagging a task as "very long" (severe jank risk). */
const VERY_LONG_TASK_THRESHOLD_MS = 50;

/** Chrome trace metadata name for the renderer main thread. */
const MAIN_THREAD_NAME = 'CrRendererMain';

/** Legacy Chrome trace metadata name still present in some traces. */
const MAIN_THREAD_NAME_LEGACY = 'CrRendererMainThread';

/** Trace event name used by Chrome when one frame is dropped. */
const DROPPED_FRAME_EVENT_NAME = 'DroppedFrame';

/** Trace event name used for browser task execution slices. */
const RUN_TASK_EVENT_NAME = 'RunTask';

/** Counter event name for the JS heap used bytes metric. */
const JS_HEAP_COUNTER_NAME = 'JSHeapUsedBytes';

/** Complete (X) event phase marker. */
const COMPLETE_PHASE = 'X';

/** Counter (C) event phase marker. */
const COUNTER_PHASE = 'C';

/** Metadata (M) event phase marker. */
const METADATA_PHASE = 'M';

/** Layout event name — forced reflow indicator. */
const LAYOUT_EVENT_NAME = 'Layout';

/** RecalculateStyles event name — style recalculation preceding Layout. */
const RECALCULATE_STYLES_EVENT_NAME = 'RecalculateStyles';

/** Event names whose durations contribute to JS execution time. */
const JS_EVENT_NAMES = new Set(['FunctionCall', 'EvaluateScript']);

/** Event names counted as paint-related work. */
const PAINT_EVENT_NAMES = new Set(['Paint', 'CompositeLayers']);

/**
 * Reads a trace file from disk, decompressing gzip if needed.
 *
 * @param tracePath - Path to a `.json` or `.json.gz` trace file.
 * @returns The raw JSON text contents of the trace.
 */
async function readTraceFile(tracePath) {
  const buffer = await readFile(tracePath);
  if (tracePath.endsWith('.gz')) {
    return gunzipSync(buffer).toString('utf8');
  }
  return buffer.toString('utf8');
}

/**
 * Parses raw trace JSON and extracts the traceEvents array.
 *
 * Detects both standard Chrome trace format and Chrome DevTools MCP format,
 * normalizing both to a flat array of trace events.
 *
 * @param rawJson - Raw JSON text of the trace file.
 * @returns Array of trace event objects.
 */
export function extractTraceEvents(rawJson) {
  const parsed = JSON.parse(rawJson);
  if (Array.isArray(parsed)) {
    return parsed;
  }
  const result = parsed.result;
  if (result !== undefined) {
    if (Array.isArray(result)) {
      return result;
    }
    return result.traceEvents ?? [];
  }
  return parsed.traceEvents ?? [];
}

/**
 * Finds the renderer main thread from metadata events.
 *
 * Scans for a `thread_name` metadata event whose value matches either the
 * current `CrRendererMain` name or the legacy `CrRendererMainThread` name,
 * and returns its pid/tid pair.
 *
 * @param events - All trace events.
 * @returns Main thread identifier or null when not found.
 */
function findMainThread(events) {
  for (const event of events) {
    if (event.ph !== METADATA_PHASE) continue;
    if (event.name !== 'thread_name') continue;
    const threadName = event.args?.name;
    if (
      threadName !== MAIN_THREAD_NAME &&
      threadName !== MAIN_THREAD_NAME_LEGACY
    ) {
      continue;
    }
    return { pid: event.pid, tid: event.tid };
  }
  return null;
}

/**
 * Determines whether an event belongs to the main thread.
 *
 * When no main thread was detected, all events are considered on-thread
 * as a fallback so CPU metrics are still computed.
 *
 * @param event - A trace event.
 * @param mainThread - The detected main thread or null.
 * @returns True when the event is on the main thread.
 */
function isOnMainThread(event, mainThread) {
  if (mainThread === null) return true;
  if (event.pid !== mainThread.pid) return false;
  if (event.tid !== mainThread.tid) return false;
  return true;
}

/**
 * Safely extracts the timestamp from a trace event.
 *
 * @param event - A trace event.
 * @returns The event timestamp in microseconds, or 0 when missing.
 */
function getTimestamp(event) {
  return event.ts ?? 0;
}

/**
 * Filters and sorts main-thread events by timestamp.
 *
 * @param events - All trace events.
 * @param mainThread - The detected main thread or null.
 * @returns Main-thread events sorted by `ts` ascending.
 */
function sortMainThreadEvents(events, mainThread) {
  const filtered = events.filter((event) => isOnMainThread(event, mainThread));
  return filtered.toSorted((a, b) => getTimestamp(a) - getTimestamp(b));
}

/**
 * Computes total CPU time from complete events on the main thread.
 *
 * @param sortedMainThreadEvents - Main-thread events sorted by timestamp.
 * @returns CPU time in milliseconds.
 */
function computeCpuTime(sortedMainThreadEvents) {
  let totalMicros = 0;
  for (const event of sortedMainThreadEvents) {
    if (event.ph !== COMPLETE_PHASE) continue;
    if (event.dur === undefined) continue;
    totalMicros += event.dur;
  }
  return totalMicros / MICROSECONDS_PER_MILLISECOND;
}

/**
 * Counts layout thrashing events — Layout events immediately preceded by
 * RecalculateStyles on the main thread in timestamp order.
 *
 * @param sortedMainThreadEvents - Main-thread events sorted by timestamp.
 * @returns Number of layout thrashing occurrences.
 */
function computeLayoutThrashing(sortedMainThreadEvents) {
  let count = 0;
  let previousName;
  for (const event of sortedMainThreadEvents) {
    if (event.name === LAYOUT_EVENT_NAME) {
      if (previousName === RECALCULATE_STYLES_EVENT_NAME) {
        count++;
      }
    }
    previousName = event.name;
  }
  return count;
}

/**
 * Computes total JavaScript execution time from FunctionCall and
 * EvaluateScript complete events.
 *
 * @param events - All trace events.
 * @returns JS execution time in milliseconds.
 */
function computeJsExecution(events) {
  let totalMicros = 0;
  for (const event of events) {
    if (event.ph !== COMPLETE_PHASE) continue;
    if (!JS_EVENT_NAMES.has(event.name)) continue;
    if (event.dur === undefined) continue;
    totalMicros += event.dur;
  }
  return totalMicros / MICROSECONDS_PER_MILLISECOND;
}

/**
 * Counts paint and composite-layer events across all threads.
 *
 * @param events - All trace events.
 * @returns Total number of paint-related events.
 */
function computePaintCount(events) {
  let count = 0;
  for (const event of events) {
    if (PAINT_EVENT_NAMES.has(event.name)) {
      count++;
    }
  }
  return count;
}

/**
 * Extracts the JS heap used value from a Counter event's args.
 *
 * Supports both the direct `args.jsHeapSizeUsed` form and the nested
 * `args.Snapshot.jsHeapSizeUsed` form used by some Chrome trace versions.
 *
 * @param event - A Counter event.
 * @returns Heap used in bytes, or 0 when unavailable.
 */
function extractHeapValue(event) {
  const args = event.args;
  if (!args) return 0;
  if (typeof args.jsHeapSizeUsed === 'number') return args.jsHeapSizeUsed;
  const snapshot = args.Snapshot;
  if (!snapshot) return 0;
  if (typeof snapshot.jsHeapSizeUsed === 'number') {
    return snapshot.jsHeapSizeUsed;
  }
  return 0;
}

/**
 * Computes the peak JS heap size from Counter events.
 *
 * @param events - All trace events.
 * @returns Peak heap used in bytes, or 0 when no heap counters exist.
 */
function computeMemoryPeak(events) {
  let peak = 0;
  for (const event of events) {
    if (event.ph !== COUNTER_PHASE) continue;
    if (event.name !== JS_HEAP_COUNTER_NAME) continue;
    const value = extractHeapValue(event);
    if (value > peak) {
      peak = value;
    }
  }
  return peak;
}

/**
 * Counts dropped frame events across all threads.
 *
 * @param events - All trace events.
 * @returns Number of dropped frames.
 */
function computeDroppedFrames(events) {
  let count = 0;
  for (const event of events) {
    if (event.name === DROPPED_FRAME_EVENT_NAME) {
      count++;
    }
  }
  return count;
}

/**
 * Counts RunTask complete events whose duration exceeds a threshold.
 *
 * @param events - All trace events.
 * @param thresholdMs - Duration threshold in milliseconds.
 * @returns Number of long tasks exceeding the threshold.
 */
function computeLongTaskCount(events, thresholdMs) {
  let count = 0;
  for (const event of events) {
    if (event.ph !== COMPLETE_PHASE) continue;
    if (event.name !== RUN_TASK_EVENT_NAME) continue;
    if (event.dur === undefined) continue;
    const durationMs = event.dur / MICROSECONDS_PER_MILLISECOND;
    if (durationMs > thresholdMs) {
      count++;
    }
  }
  return count;
}

/**
 * Computes all performance metrics from a trace events array.
 *
 * @param events - Array of trace event objects.
 * @returns Summary object with deterministic key order.
 */
export function computeMetrics(events) {
  const mainThread = findMainThread(events);
  const sortedMainThread = sortMainThreadEvents(events, mainThread);

  return {
    eventCount: events.length,
    cpuTimeMs: computeCpuTime(sortedMainThread),
    layoutThrashingCount: computeLayoutThrashing(sortedMainThread),
    jsExecutionMs: computeJsExecution(events),
    paintEventCount: computePaintCount(events),
    memoryPeakBytes: computeMemoryPeak(events),
    droppedFrames: computeDroppedFrames(events),
    longTasks16ms: computeLongTaskCount(events, LONG_TASK_THRESHOLD_MS),
    longTasks50ms: computeLongTaskCount(events, VERY_LONG_TASK_THRESHOLD_MS),
  };
}

/**
 * Reads a trace file and extracts a concise performance summary.
 *
 * Handles both raw `.json` and gzip-compressed `.json.gz` inputs, and
 * supports both standard Chrome trace and Chrome DevTools MCP formats.
 *
 * @param tracePath - Path to the trace file.
 * @returns Summary metrics object.
 * @example
 * ```ts
 * const summary = await summarizeTrace('tmp/traces/trace.json.gz');
 * console.log(summary.cpuTimeMs);
 * ```
 */
export async function summarizeTrace(tracePath) {
  const rawJson = await readTraceFile(tracePath);
  const events = extractTraceEvents(rawJson);
  return computeMetrics(events);
}

/**
 * Formats a summary object as human-readable text for CLI output.
 *
 * @param summary - The metrics summary object.
 * @returns Multi-line human-readable string.
 */
function formatSummaryHuman(summary) {
  return [
    `Event count:        ${summary.eventCount}`,
    `CPU time:           ${summary.cpuTimeMs.toFixed(1)} ms`,
    `Layout thrashing:   ${summary.layoutThrashingCount}`,
    `JS execution:       ${summary.jsExecutionMs.toFixed(1)} ms`,
    `Paint events:       ${summary.paintEventCount}`,
    `Memory peak:        ${summary.memoryPeakBytes} bytes`,
    `Dropped frames:     ${summary.droppedFrames}`,
    `Long tasks (>16ms): ${summary.longTasks16ms}`,
    `Long tasks (>50ms): ${summary.longTasks50ms}`,
  ].join('\n');
}

/**
 * Parses CLI arguments for the summarization CLI.
 *
 * @param argv - CLI arguments after the script path.
 * @returns Parsed trace path and JSON flag.
 * @throws {Error} When no trace path is provided.
 */
export function parseCliArgs(argv) {
  const [tracePath, ...rest] = argv;
  if (!tracePath) {
    throw new Error(
      'Missing trace path. Example: node scripts/trace-summarize.mjs <trace-path> [--json]',
    );
  }
  return { tracePath, json: rest.includes('--json') };
}

/**
 * Determines whether the current process is the CLI entry point.
 *
 * @param argv1 - The `process.argv[1]` value for the current process.
 * @param entryUrl - The `import.meta.url` of the module.
 * @returns True when the module is invoked directly as a script.
 */
export function isCliEntryPoint(argv1, entryUrl) {
  return Boolean(argv1) && entryUrl === pathToFileURL(argv1).href;
}

/**
 * Runs the summarization CLI when the module is invoked as the entry point.
 *
 * Exposed for unit testing so the CLI dispatch path can be exercised without
 * spawning a child process.
 *
 * @param argv - CLI arguments after the script path.
 * @param context - Invocation context carrying the entry-point signals.
 * @returns Resolves when the CLI action completes or no-ops when not main.
 */
export async function runCli(argv, context) {
  if (!isCliEntryPoint(context.argv1, context.entryUrl)) {
    return;
  }
  const { tracePath, json } = parseCliArgs(argv);
  const summary = await summarizeTrace(tracePath);
  if (json) {
    console.log(JSON.stringify(summary));
  } else {
    console.log(formatSummaryHuman(summary));
  }
}

await runCli(process.argv.slice(2), {
  argv1: process.argv[1],
  entryUrl: import.meta.url,
});
