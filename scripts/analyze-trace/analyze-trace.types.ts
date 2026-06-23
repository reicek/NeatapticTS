/** Primitive trace metadata value supported by the nested args reader. */
export type Primitive = string | number | boolean | null | undefined;

/** Minimal Chrome or Perfetto trace event shape used by the analyzer. */
export interface TraceEvent {
  pid?: number;
  tid?: number;
  ts?: number;
  dur?: number;
  tdur?: number;
  ph?: string;
  cat?: string;
  name?: string;
  args?: Record<string, unknown>;
}

/** Top-level trace file shape expected from exported JSON captures. */
export interface TraceFile {
  traceEvents?: TraceEvent[];
}

/**
 * Trace file format detected by the I/O layer.
 *
 * - `standard` — bare array or `{ traceEvents: [...] }` (raw Chrome/Perfetto export).
 * - `chrome-devtools-mcp` — `{ result: { traceEvents: [...] } }` or `{ result: [...] }`
 *   (Chrome DevTools MCP server response wrapper).
 */
export type TraceFormat = 'standard' | 'chrome-devtools-mcp';

/** Parsed CLI options for one analyzer invocation. */
export interface CliOptions {
  tracePath: string;
  topCount: number;
}

/** Summary metrics for one thread label in the trace. */
export interface ThreadSummary {
  label: string;
  totalDurationMs: number;
  runTaskDurationMs: number;
  longTaskCount16ms: number;
  longTaskCount50ms: number;
  maxTaskMs: number;
}

/** Aggregated duration rollup for one trace event or one script URL. */
export interface AggregatedDuration {
  name: string;
  count: number;
  totalMs: number;
  maxMs: number;
}

/** Longest single complete event entry used by the report. */
export interface LongEventSummary {
  durationMs: number;
  threadLabel: string;
  eventName: string;
  scriptUrl?: string;
}

/** Visible trace window spanned by the capture. */
export interface TraceTimeRange {
  windowMs: number;
}

/** Fully analyzed trace payload used by the reporting layer. */
export interface TraceAnalysis {
  traceFilePath: string;
  traceEventCount: number;
  timeRange: TraceTimeRange;
  droppedFrameCount: number;
  beginFrameCount: number;
  threadSummaries: ThreadSummary[];
  hottestEvents: AggregatedDuration[];
  hottestFunctionCalls: AggregatedDuration[];
  longestEvents: LongEventSummary[];
  fireAnimationFrameDurations: number[];
  functionCallDurations: number[];
}
