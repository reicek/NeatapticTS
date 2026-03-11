/**
 * analyze-trace.ts
 *
 * Reads a Chrome/Perfetto trace export and prints a compact performance report.
 * The report is intentionally thread-aware so browser-main, renderer-main,
 * worker, compositor, and GPU activity can be inspected separately.
 *
 * Usage:
 *   npm run trace:analyze -- test/examples/flappy_bird/Trace-20260309T191949.json
 *   npx ts-node scripts/analyze-trace.ts path/to/trace.json --top=20
 */
import fs from 'fs';
import path from 'path';

type Primitive = string | number | boolean | null | undefined;

interface TraceEvent {
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

interface TraceFile {
  traceEvents?: TraceEvent[];
}

interface ThreadIdentity {
  pid: number;
  tid: number;
}

interface ThreadSummary {
  label: string;
  totalDurationMs: number;
  runTaskDurationMs: number;
  longTaskCount16ms: number;
  longTaskCount50ms: number;
  maxTaskMs: number;
}

interface AggregatedDuration {
  name: string;
  count: number;
  totalMs: number;
  maxMs: number;
}

interface LongEventSummary {
  durationMs: number;
  threadLabel: string;
  eventName: string;
  scriptUrl?: string;
}

const MICROSECONDS_PER_MILLISECOND = 1000;
const DEFAULT_TOP_COUNT = 12;
const LONG_TASK_THRESHOLD_MS = 16.7;
const VERY_LONG_TASK_THRESHOLD_MS = 50;

/**
 * CLI entry point.
 *
 * Reads the trace, derives thread metadata, and prints a deterministic textual
 * report that is easy to compare across captures.
 */
function main(): void {
  const options = resolveCliOptions(process.argv.slice(2));
  const traceFilePath = path.resolve(options.tracePath);
  const trace = loadTrace(traceFilePath);
  const traceEvents = trace.traceEvents ?? [];

  if (traceEvents.length === 0) {
    throw new Error(`Trace contains no events: ${traceFilePath}`);
  }

  const processNames = collectMetadataNames(traceEvents, 'process_name');
  const threadNames = collectMetadataNames(traceEvents, 'thread_name');
  const workerThreadLabels = collectWorkerThreadLabels(traceEvents);
  const timeRange = resolveTimeRange(traceEvents);
  const droppedFrameCount = traceEvents.filter(
    (traceEvent) => traceEvent.name === 'DroppedFrame',
  ).length;
  const beginFrameCount = traceEvents.filter(
    (traceEvent) => traceEvent.name === 'BeginFrame',
  ).length;

  const threadSummaries = buildThreadSummaries({
    traceEvents,
    processNames,
    threadNames,
    workerThreadLabels,
  });
  const hottestEvents = aggregateDurationsByName(traceEvents, options.topCount);
  const hottestFunctionCalls = aggregateFunctionCalls(traceEvents, options.topCount);
  const longestEvents = collectLongestEvents({
    traceEvents,
    processNames,
    threadNames,
    workerThreadLabels,
    topCount: options.topCount,
  });
  const fireAnimationFrameDurations = collectEventDurationsMs(
    traceEvents,
    'FireAnimationFrame',
  );
  const functionCallDurations = collectEventDurationsMs(traceEvents, 'FunctionCall');

  printSection('Trace');
  printLine(`File: ${traceFilePath}`);
  printLine(
    `Window: ${formatMs(timeRange.windowMs)} across ${traceEvents.length.toLocaleString()} events`,
  );
  printLine(
    `Frames: ${beginFrameCount.toLocaleString()} begin, ${droppedFrameCount.toLocaleString()} dropped`,
  );

  printSection('Thread Summary');
  for (const threadSummary of threadSummaries) {
    printLine(
      `${threadSummary.label}: total=${formatMs(threadSummary.totalDurationMs)}, ` +
        `RunTask=${formatMs(threadSummary.runTaskDurationMs)}, ` +
        `>16.7ms=${threadSummary.longTaskCount16ms}, ` +
        `>50ms=${threadSummary.longTaskCount50ms}, ` +
        `max=${formatMs(threadSummary.maxTaskMs)}`,
    );
  }

  printSection('Animation Frames');
  printLine(`FireAnimationFrame: ${formatDistribution(fireAnimationFrameDurations)}`);
  printLine(`FunctionCall: ${formatDistribution(functionCallDurations)}`);

  printSection('Longest Events');
  for (const longEvent of longestEvents) {
    printLine(
      `${formatMs(longEvent.durationMs)} | ${longEvent.threadLabel} | ${longEvent.eventName}${
        longEvent.scriptUrl ? ` | ${longEvent.scriptUrl}` : ''
      }`,
    );
  }

  printSection('Top Events');
  for (const eventSummary of hottestEvents) {
    printLine(
      `${eventSummary.name}: count=${eventSummary.count.toLocaleString()}, total=${formatMs(eventSummary.totalMs)}, max=${formatMs(eventSummary.maxMs)}`,
    );
  }

  printSection('Top Function Calls');
  for (const functionCallSummary of hottestFunctionCalls) {
    printLine(
      `${functionCallSummary.name}: count=${functionCallSummary.count.toLocaleString()}, total=${formatMs(functionCallSummary.totalMs)}, max=${formatMs(functionCallSummary.maxMs)}`,
    );
  }
}

/**
 * Resolves CLI flags.
 *
 * Supported flags:
 * - positional trace path
 * - `--top=NUMBER`
 */
function resolveCliOptions(argumentsList: string[]): {
  tracePath: string;
  topCount: number;
} {
  const pathArgument = argumentsList.find((argument) => !argument.startsWith('--'));
  if (!pathArgument) {
    throw new Error(
      'Missing trace path. Example: npm run trace:analyze -- test/examples/flappy_bird/Trace.json',
    );
  }

  const topArgument = argumentsList.find((argument) => argument.startsWith('--top='));
  const requestedTopCount = topArgument
    ? Number.parseInt(topArgument.slice('--top='.length), 10)
    : DEFAULT_TOP_COUNT;

  return {
    tracePath: pathArgument,
    topCount:
      Number.isFinite(requestedTopCount) && requestedTopCount > 0
        ? requestedTopCount
        : DEFAULT_TOP_COUNT,
  };
}

/** Reads and parses a trace JSON file. */
function loadTrace(traceFilePath: string): TraceFile {
  const traceFileContents = fs.readFileSync(traceFilePath, 'utf8');
  return JSON.parse(traceFileContents) as TraceFile;
}

/**
 * Collects thread or process metadata labels keyed by `pid:tid`.
 *
 * Chrome emits metadata events (`ph === 'M'`) that attach human-readable names
 * to otherwise numeric process and thread identifiers.
 */
function collectMetadataNames(
  traceEvents: TraceEvent[],
  metadataName: 'process_name' | 'thread_name',
): Map<string, string> {
  const metadataNames = new Map<string, string>();

  for (const traceEvent of traceEvents) {
    if (traceEvent.ph !== 'M' || traceEvent.name !== metadataName) {
      continue;
    }

    const resolvedName = readNestedPrimitive(traceEvent.args, ['name']);
    if (typeof resolvedName !== 'string') {
      continue;
    }

    const metadataKey =
      metadataName === 'process_name'
        ? createProcessKey(traceEvent.pid)
        : createThreadKey(traceEvent.pid, traceEvent.tid);

    if (metadataKey) {
      metadataNames.set(metadataKey, resolvedName);
    }
  }

  return metadataNames;
}

/**
 * Resolves worker-thread labels from worker attachment events.
 *
 * This helps map generic `DedicatedWorker thread` labels back to the worker
 * script URL that spawned them.
 */
function collectWorkerThreadLabels(traceEvents: TraceEvent[]): Map<string, string> {
  const workerThreadLabels = new Map<string, string>();

  for (const traceEvent of traceEvents) {
    if (traceEvent.name !== 'TracingSessionIdForWorker') {
      continue;
    }

    const workerThreadId = readNestedPrimitive(traceEvent.args, [
      'data',
      'workerThreadId',
    ]);
    const workerUrl = readNestedPrimitive(traceEvent.args, ['data', 'url']);

    if (typeof workerThreadId !== 'number' || typeof workerUrl !== 'string') {
      continue;
    }

    const threadKey = createThreadKey(traceEvent.pid, workerThreadId);
    if (!threadKey) {
      continue;
    }

    workerThreadLabels.set(threadKey, workerUrl);
  }

  return workerThreadLabels;
}

/** Computes the visible time window spanned by non-zero timestamp events. */
function resolveTimeRange(traceEvents: TraceEvent[]): { windowMs: number } {
  let minimumTimestamp = Number.POSITIVE_INFINITY;
  let maximumTimestamp = 0;

  for (const traceEvent of traceEvents) {
    const eventTimestamp = traceEvent.ts;
    if (typeof eventTimestamp === 'number' && eventTimestamp > 0) {
      minimumTimestamp = Math.min(minimumTimestamp, eventTimestamp);
    }

    const eventDuration = typeof traceEvent.dur === 'number' ? traceEvent.dur : 0;
    maximumTimestamp = Math.max(maximumTimestamp, (eventTimestamp ?? 0) + eventDuration);
  }

  if (!Number.isFinite(minimumTimestamp)) {
    minimumTimestamp = 0;
  }

  return {
    windowMs: (maximumTimestamp - minimumTimestamp) / MICROSECONDS_PER_MILLISECOND,
  };
}

/**
 * Builds per-thread rollups for total duration and long-task counts.
 */
function buildThreadSummaries(options: {
  traceEvents: TraceEvent[];
  processNames: Map<string, string>;
  threadNames: Map<string, string>;
  workerThreadLabels: Map<string, string>;
}): ThreadSummary[] {
  const {
    traceEvents,
    processNames,
    threadNames,
    workerThreadLabels,
  } = options;
  const durationByThread = new Map<string, ThreadSummary>();

  for (const traceEvent of traceEvents) {
    if (traceEvent.ph !== 'X' || typeof traceEvent.dur !== 'number') {
      continue;
    }

    const threadKey = createThreadKey(traceEvent.pid, traceEvent.tid);
    if (!threadKey) {
      continue;
    }

    const threadSummaryLabel = resolveThreadLabel({
      traceEvent,
      processNames,
      threadNames,
      workerThreadLabels,
    });
    const threadSummary =
      durationByThread.get(threadSummaryLabel) ?? {
        label: threadSummaryLabel,
        totalDurationMs: 0,
        runTaskDurationMs: 0,
        longTaskCount16ms: 0,
        longTaskCount50ms: 0,
        maxTaskMs: 0,
      };
    const eventDurationMs = traceEvent.dur / MICROSECONDS_PER_MILLISECOND;

    threadSummary.totalDurationMs += eventDurationMs;
    if (traceEvent.name === 'RunTask') {
      threadSummary.runTaskDurationMs += eventDurationMs;
      if (eventDurationMs >= LONG_TASK_THRESHOLD_MS) {
        threadSummary.longTaskCount16ms += 1;
      }
      if (eventDurationMs >= VERY_LONG_TASK_THRESHOLD_MS) {
        threadSummary.longTaskCount50ms += 1;
      }
      threadSummary.maxTaskMs = Math.max(threadSummary.maxTaskMs, eventDurationMs);
    }

    durationByThread.set(threadSummaryLabel, threadSummary);
  }

  return Array.from(durationByThread.values()).toSorted(
    (leftSummary, rightSummary) => rightSummary.totalDurationMs - leftSummary.totalDurationMs,
  );
}

/** Resolves a readable process/thread label for one event. */
function resolveThreadLabel(options: {
  traceEvent: TraceEvent;
  processNames: Map<string, string>;
  threadNames: Map<string, string>;
  workerThreadLabels: Map<string, string>;
}): string {
  const { traceEvent, processNames, threadNames, workerThreadLabels } = options;
  const processLabel = processNames.get(createProcessKey(traceEvent.pid) ?? '') ?? 'unknown-process';
  const threadKey = createThreadKey(traceEvent.pid, traceEvent.tid) ?? 'unknown-thread';
  const threadLabel = threadNames.get(threadKey) ?? 'unknown-thread';
  const workerLabel = workerThreadLabels.get(threadKey);
  return workerLabel
    ? `${processLabel} / ${threadLabel} / ${workerLabel}`
    : `${processLabel} / ${threadLabel}`;
}

/** Collects the longest complete events in the trace. */
function collectLongestEvents(options: {
  traceEvents: TraceEvent[];
  processNames: Map<string, string>;
  threadNames: Map<string, string>;
  workerThreadLabels: Map<string, string>;
  topCount: number;
}): LongEventSummary[] {
  const { traceEvents, processNames, threadNames, workerThreadLabels, topCount } = options;

  return traceEvents
    .filter(
      (traceEvent) =>
        traceEvent.ph === 'X' &&
        typeof traceEvent.dur === 'number' &&
        typeof traceEvent.name === 'string',
    )
    .map((traceEvent) => ({
      durationMs: traceEvent.dur! / MICROSECONDS_PER_MILLISECOND,
      threadLabel: resolveThreadLabel({
        traceEvent,
        processNames,
        threadNames,
        workerThreadLabels,
      }),
      eventName: traceEvent.name!,
      scriptUrl:
        (readNestedPrimitive(traceEvent.args, ['data', 'url']) as string | undefined) ??
        undefined,
    }))
    .toSorted((leftEvent, rightEvent) => rightEvent.durationMs - leftEvent.durationMs)
    .slice(0, topCount);
}

/**
 * Aggregates complete-event durations by event name.
 *
 * `RunTask` is kept because it is still useful for thread pressure, but the
 * separate function-call rollup below is generally more actionable.
 */
function aggregateDurationsByName(
  traceEvents: TraceEvent[],
  topCount: number,
): AggregatedDuration[] {
  const durationsByName = new Map<string, AggregatedDuration>();

  for (const traceEvent of traceEvents) {
    if (traceEvent.ph !== 'X' || typeof traceEvent.dur !== 'number' || !traceEvent.name) {
      continue;
    }

    const eventDurationMs = traceEvent.dur / MICROSECONDS_PER_MILLISECOND;
    const eventSummary = durationsByName.get(traceEvent.name) ?? {
      name: traceEvent.name,
      count: 0,
      totalMs: 0,
      maxMs: 0,
    };

    eventSummary.count += 1;
    eventSummary.totalMs += eventDurationMs;
    eventSummary.maxMs = Math.max(eventSummary.maxMs, eventDurationMs);
    durationsByName.set(traceEvent.name, eventSummary);
  }

  return Array.from(durationsByName.values())
    .toSorted((leftSummary, rightSummary) => rightSummary.totalMs - leftSummary.totalMs)
    .slice(0, topCount);
}

/**
 * Aggregates `FunctionCall` events by script URL.
 *
 * This is the most direct way to see which bundle or worker file owns main
 * thread or worker thread execution time in the trace.
 */
function aggregateFunctionCalls(
  traceEvents: TraceEvent[],
  topCount: number,
): AggregatedDuration[] {
  const durationsByFunctionCall = new Map<string, AggregatedDuration>();

  for (const traceEvent of traceEvents) {
    if (
      traceEvent.ph !== 'X' ||
      traceEvent.name !== 'FunctionCall' ||
      typeof traceEvent.dur !== 'number'
    ) {
      continue;
    }

    const functionCallName =
      (readNestedPrimitive(traceEvent.args, ['data', 'url']) as string | undefined) ??
      '(unknown-script)';
    const functionCallSummary = durationsByFunctionCall.get(functionCallName) ?? {
      name: functionCallName,
      count: 0,
      totalMs: 0,
      maxMs: 0,
    };
    const eventDurationMs = traceEvent.dur / MICROSECONDS_PER_MILLISECOND;

    functionCallSummary.count += 1;
    functionCallSummary.totalMs += eventDurationMs;
    functionCallSummary.maxMs = Math.max(functionCallSummary.maxMs, eventDurationMs);
    durationsByFunctionCall.set(functionCallName, functionCallSummary);
  }

  return Array.from(durationsByFunctionCall.values())
    .toSorted((leftSummary, rightSummary) => rightSummary.totalMs - leftSummary.totalMs)
    .slice(0, topCount);
}

/** Collects all durations for one event name in milliseconds. */
function collectEventDurationsMs(
  traceEvents: TraceEvent[],
  eventName: string,
): number[] {
  return traceEvents
    .filter(
      (traceEvent) =>
        traceEvent.ph === 'X' &&
        traceEvent.name === eventName &&
        typeof traceEvent.dur === 'number',
    )
    .map((traceEvent) => traceEvent.dur! / MICROSECONDS_PER_MILLISECOND)
    .toSorted((leftDuration, rightDuration) => leftDuration - rightDuration);
}

/** Formats a percentile-style summary for one event duration collection. */
function formatDistribution(durationsMs: number[]): string {
  if (durationsMs.length === 0) {
    return 'count=0';
  }

  return [
    `count=${durationsMs.length.toLocaleString()}`,
    `p50=${formatMs(percentile(durationsMs, 0.5))}`,
    `p90=${formatMs(percentile(durationsMs, 0.9))}`,
    `p99=${formatMs(percentile(durationsMs, 0.99))}`,
    `max=${formatMs(durationsMs.at(-1) ?? 0)}`,
  ].join(', ');
}

/** Computes a nearest-rank percentile from sorted durations. */
function percentile(sortedDurations: number[], rank: number): number {
  if (sortedDurations.length === 0) {
    return 0;
  }

  const clampedRank = Math.max(0, Math.min(1, rank));
  const index = Math.min(
    sortedDurations.length - 1,
    Math.floor(clampedRank * (sortedDurations.length - 1)),
  );
  return sortedDurations[index];
}

/** Safely resolves a nested primitive value from an event args object. */
function readNestedPrimitive(
  value: unknown,
  pathSegments: string[],
): Primitive {
  let currentValue: unknown = value;

  for (const pathSegment of pathSegments) {
    if (!currentValue || typeof currentValue !== 'object') {
      return undefined;
    }

    currentValue = (currentValue as Record<string, unknown>)[pathSegment];
  }

  if (
    currentValue == null ||
    typeof currentValue === 'string' ||
    typeof currentValue === 'number' ||
    typeof currentValue === 'boolean'
  ) {
    return currentValue;
  }

  return undefined;
}

/** Formats a duration for stable human-readable output. */
function formatMs(durationMs: number): string {
  return `${durationMs.toFixed(2)}ms`;
}

/** Creates a stable process lookup key. */
function createProcessKey(pid: number | undefined): string | undefined {
  return typeof pid === 'number' ? `pid:${pid}` : undefined;
}

/** Creates a stable thread lookup key. */
function createThreadKey(
  pid: number | undefined,
  tid: number | undefined,
): string | undefined {
  return typeof pid === 'number' && typeof tid === 'number'
    ? `pid:${pid}:tid:${tid}`
    : undefined;
}

/** Prints a section title. */
function printSection(title: string): void {
  console.log(`\n[${title}]`);
}

/** Prints one content line. */
function printLine(value: string): void {
  console.log(value);
}

main();