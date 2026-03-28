import {
  BEGIN_FRAME_EVENT_NAME,
  DROPPED_FRAME_EVENT_NAME,
  FIRE_ANIMATION_FRAME_EVENT_NAME,
  FUNCTION_CALL_EVENT_NAME,
  LONG_TASK_THRESHOLD_MS,
  MICROSECONDS_PER_MILLISECOND,
  RUN_TASK_EVENT_NAME,
  UNKNOWN_PROCESS_LABEL,
  UNKNOWN_SCRIPT_LABEL,
  UNKNOWN_THREAD_LABEL,
  VERY_LONG_TASK_THRESHOLD_MS,
} from './analyze-trace.constants.js';
import {
  createProcessKey,
  createThreadKey,
  readNestedPrimitive,
} from './analyze-trace.shared.js';
import type {
  AggregatedDuration,
  LongEventSummary,
  ThreadSummary,
  TraceAnalysis,
  TraceEvent,
  TraceTimeRange,
} from './analyze-trace.types.js';

type TraceMetadataName = 'process_name' | 'thread_name';

interface TraceAnalysisInput {
  traceFilePath: string;
  traceEvents: readonly TraceEvent[];
  topCount: number;
}

interface TraceLabelContext {
  processNames: ReadonlyMap<string, string>;
  threadNames: ReadonlyMap<string, string>;
  workerThreadLabels: ReadonlyMap<string, string>;
}

/**
 * Analyzes one trace event collection into deterministic report data.
 *
 * @param input - Trace analysis input packet.
 * @returns Fully analyzed trace payload used by the reporting layer.
 */
export function analyzeTraceEvents(input: TraceAnalysisInput): TraceAnalysis {
  const processNames = collectMetadataNames(input.traceEvents, 'process_name');
  const threadNames = collectMetadataNames(input.traceEvents, 'thread_name');
  const workerThreadLabels = collectWorkerThreadLabels(input.traceEvents);
  const labelContext: TraceLabelContext = {
    processNames,
    threadNames,
    workerThreadLabels,
  };

  return {
    traceFilePath: input.traceFilePath,
    traceEventCount: input.traceEvents.length,
    timeRange: resolveTimeRange(input.traceEvents),
    droppedFrameCount: countNamedEvents(
      input.traceEvents,
      DROPPED_FRAME_EVENT_NAME,
    ),
    beginFrameCount: countNamedEvents(
      input.traceEvents,
      BEGIN_FRAME_EVENT_NAME,
    ),
    threadSummaries: buildThreadSummaries(input.traceEvents, labelContext),
    hottestEvents: aggregateDurationsByName(input.traceEvents, input.topCount),
    hottestFunctionCalls: aggregateFunctionCalls(
      input.traceEvents,
      input.topCount,
    ),
    longestEvents: collectLongestEvents(
      input.traceEvents,
      labelContext,
      input.topCount,
    ),
    fireAnimationFrameDurations: collectEventDurationsMs(
      input.traceEvents,
      FIRE_ANIMATION_FRAME_EVENT_NAME,
    ),
    functionCallDurations: collectEventDurationsMs(
      input.traceEvents,
      FUNCTION_CALL_EVENT_NAME,
    ),
  };
}

/**
 * Collects metadata names keyed by process or thread id.
 *
 * @param traceEvents - Trace events to scan.
 * @param metadataName - Metadata event name to collect.
 * @returns Metadata map keyed by stable process or thread keys.
 */
function collectMetadataNames(
  traceEvents: readonly TraceEvent[],
  metadataName: TraceMetadataName,
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
    if (!metadataKey) {
      continue;
    }

    metadataNames.set(metadataKey, resolvedName);
  }

  return metadataNames;
}

/**
 * Collects worker-thread labels from worker attachment events.
 *
 * @param traceEvents - Trace events to scan.
 * @returns Worker thread label map keyed by stable thread keys.
 */
function collectWorkerThreadLabels(
  traceEvents: readonly TraceEvent[],
): Map<string, string> {
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

/**
 * Computes the visible time window spanned by the trace.
 *
 * @param traceEvents - Trace events to scan.
 * @returns Visible trace window metrics.
 */
function resolveTimeRange(traceEvents: readonly TraceEvent[]): TraceTimeRange {
  let minimumTimestamp = Number.POSITIVE_INFINITY;
  let maximumTimestamp = 0;

  for (const traceEvent of traceEvents) {
    const eventTimestamp =
      typeof traceEvent.ts === 'number' ? traceEvent.ts : 0;
    const eventDuration =
      typeof traceEvent.dur === 'number' ? traceEvent.dur : 0;

    if (eventTimestamp > 0) {
      minimumTimestamp = Math.min(minimumTimestamp, eventTimestamp);
    }

    maximumTimestamp = Math.max(
      maximumTimestamp,
      eventTimestamp + eventDuration,
    );
  }

  if (!Number.isFinite(minimumTimestamp)) {
    minimumTimestamp = 0;
  }

  return {
    windowMs:
      (maximumTimestamp - minimumTimestamp) / MICROSECONDS_PER_MILLISECOND,
  };
}

/**
 * Builds per-thread duration and long-task summaries.
 *
 * @param traceEvents - Trace events to scan.
 * @param labelContext - Thread label context.
 * @returns Thread summaries sorted by total duration descending.
 */
function buildThreadSummaries(
  traceEvents: readonly TraceEvent[],
  labelContext: TraceLabelContext,
): ThreadSummary[] {
  const durationByThreadLabel = new Map<string, ThreadSummary>();

  for (const traceEvent of traceEvents) {
    if (traceEvent.ph !== 'X' || typeof traceEvent.dur !== 'number') {
      continue;
    }

    const threadKey = createThreadKey(traceEvent.pid, traceEvent.tid);
    if (!threadKey) {
      continue;
    }

    const threadLabel = resolveThreadLabel(traceEvent, labelContext);
    const threadSummary = durationByThreadLabel.get(threadLabel) ?? {
      label: threadLabel,
      totalDurationMs: 0,
      runTaskDurationMs: 0,
      longTaskCount16ms: 0,
      longTaskCount50ms: 0,
      maxTaskMs: 0,
    };
    const eventDurationMs = traceEvent.dur / MICROSECONDS_PER_MILLISECOND;

    threadSummary.totalDurationMs += eventDurationMs;
    if (traceEvent.name === RUN_TASK_EVENT_NAME) {
      threadSummary.runTaskDurationMs += eventDurationMs;
      if (eventDurationMs >= LONG_TASK_THRESHOLD_MS) {
        threadSummary.longTaskCount16ms += 1;
      }
      if (eventDurationMs >= VERY_LONG_TASK_THRESHOLD_MS) {
        threadSummary.longTaskCount50ms += 1;
      }
      threadSummary.maxTaskMs = Math.max(
        threadSummary.maxTaskMs,
        eventDurationMs,
      );
    }

    durationByThreadLabel.set(threadLabel, threadSummary);
  }

  return Array.from(durationByThreadLabel.values()).toSorted(
    (leftSummary, rightSummary) =>
      rightSummary.totalDurationMs - leftSummary.totalDurationMs ||
      leftSummary.label.localeCompare(rightSummary.label),
  );
}

/**
 * Resolves the longest complete events in the trace.
 *
 * @param traceEvents - Trace events to scan.
 * @param labelContext - Thread label context.
 * @param topCount - Maximum number of events to return.
 * @returns Longest complete events sorted by duration.
 */
function collectLongestEvents(
  traceEvents: readonly TraceEvent[],
  labelContext: TraceLabelContext,
  topCount: number,
): LongEventSummary[] {
  const longEvents: LongEventSummary[] = [];

  for (const traceEvent of traceEvents) {
    if (traceEvent.ph !== 'X' || typeof traceEvent.dur !== 'number') {
      continue;
    }

    const eventName = traceEvent.name;
    if (typeof eventName !== 'string') {
      continue;
    }

    const scriptUrlValue = readNestedPrimitive(traceEvent.args, [
      'data',
      'url',
    ]);
    longEvents.push({
      durationMs: traceEvent.dur / MICROSECONDS_PER_MILLISECOND,
      threadLabel: resolveThreadLabel(traceEvent, labelContext),
      eventName,
      scriptUrl:
        typeof scriptUrlValue === 'string' ? scriptUrlValue : undefined,
    });
  }

  return longEvents
    .toSorted(
      (leftEvent, rightEvent) =>
        rightEvent.durationMs - leftEvent.durationMs ||
        leftEvent.eventName.localeCompare(rightEvent.eventName) ||
        leftEvent.threadLabel.localeCompare(rightEvent.threadLabel),
    )
    .slice(0, topCount);
}

/**
 * Aggregates complete-event durations by event name.
 *
 * @param traceEvents - Trace events to scan.
 * @param topCount - Maximum number of entries to return.
 * @returns Event duration rollup.
 */
function aggregateDurationsByName(
  traceEvents: readonly TraceEvent[],
  topCount: number,
): AggregatedDuration[] {
  const durationsByName = new Map<string, AggregatedDuration>();

  for (const traceEvent of traceEvents) {
    if (traceEvent.ph !== 'X' || typeof traceEvent.dur !== 'number') {
      continue;
    }

    const eventName = traceEvent.name;
    if (!eventName) {
      continue;
    }

    const eventDurationMs = traceEvent.dur / MICROSECONDS_PER_MILLISECOND;
    const eventSummary = durationsByName.get(eventName) ?? {
      name: eventName,
      count: 0,
      totalMs: 0,
      maxMs: 0,
    };

    eventSummary.count += 1;
    eventSummary.totalMs += eventDurationMs;
    eventSummary.maxMs = Math.max(eventSummary.maxMs, eventDurationMs);
    durationsByName.set(eventName, eventSummary);
  }

  return Array.from(durationsByName.values())
    .toSorted(
      (leftSummary, rightSummary) =>
        rightSummary.totalMs - leftSummary.totalMs ||
        leftSummary.name.localeCompare(rightSummary.name),
    )
    .slice(0, topCount);
}

/**
 * Aggregates FunctionCall durations by script URL.
 *
 * @param traceEvents - Trace events to scan.
 * @param topCount - Maximum number of entries to return.
 * @returns Function-call duration rollup.
 */
function aggregateFunctionCalls(
  traceEvents: readonly TraceEvent[],
  topCount: number,
): AggregatedDuration[] {
  const durationsByFunctionCall = new Map<string, AggregatedDuration>();

  for (const traceEvent of traceEvents) {
    if (
      traceEvent.ph !== 'X' ||
      traceEvent.name !== FUNCTION_CALL_EVENT_NAME ||
      typeof traceEvent.dur !== 'number'
    ) {
      continue;
    }

    const functionCallNameValue = readNestedPrimitive(traceEvent.args, [
      'data',
      'url',
    ]);
    const functionCallName =
      typeof functionCallNameValue === 'string'
        ? functionCallNameValue
        : UNKNOWN_SCRIPT_LABEL;
    const functionCallSummary = durationsByFunctionCall.get(
      functionCallName,
    ) ?? {
      name: functionCallName,
      count: 0,
      totalMs: 0,
      maxMs: 0,
    };
    const eventDurationMs = traceEvent.dur / MICROSECONDS_PER_MILLISECOND;

    functionCallSummary.count += 1;
    functionCallSummary.totalMs += eventDurationMs;
    functionCallSummary.maxMs = Math.max(
      functionCallSummary.maxMs,
      eventDurationMs,
    );
    durationsByFunctionCall.set(functionCallName, functionCallSummary);
  }

  return Array.from(durationsByFunctionCall.values())
    .toSorted(
      (leftSummary, rightSummary) =>
        rightSummary.totalMs - leftSummary.totalMs ||
        leftSummary.name.localeCompare(rightSummary.name),
    )
    .slice(0, topCount);
}

/**
 * Collects all durations for one named complete event.
 *
 * @param traceEvents - Trace events to scan.
 * @param eventName - Event name to collect.
 * @returns Sorted duration samples in milliseconds.
 */
function collectEventDurationsMs(
  traceEvents: readonly TraceEvent[],
  eventName: string,
): number[] {
  const durationsMs: number[] = [];

  for (const traceEvent of traceEvents) {
    if (
      traceEvent.ph !== 'X' ||
      traceEvent.name !== eventName ||
      typeof traceEvent.dur !== 'number'
    ) {
      continue;
    }

    durationsMs.push(traceEvent.dur / MICROSECONDS_PER_MILLISECOND);
  }

  return durationsMs.toSorted(
    (leftDuration, rightDuration) => leftDuration - rightDuration,
  );
}

/**
 * Counts complete or metadata events by exact name.
 *
 * @param traceEvents - Trace events to scan.
 * @param eventName - Event name to count.
 * @returns Matching event count.
 */
function countNamedEvents(
  traceEvents: readonly TraceEvent[],
  eventName: string,
): number {
  return traceEvents.filter((traceEvent) => traceEvent.name === eventName)
    .length;
}

/**
 * Resolves a readable process and thread label for one trace event.
 *
 * @param traceEvent - Trace event needing a label.
 * @param labelContext - Thread label context.
 * @returns Human-readable thread label.
 */
function resolveThreadLabel(
  traceEvent: TraceEvent,
  labelContext: TraceLabelContext,
): string {
  const processKey = createProcessKey(traceEvent.pid) ?? '';
  const threadKey = createThreadKey(traceEvent.pid, traceEvent.tid) ?? '';
  const processLabel =
    labelContext.processNames.get(processKey) ?? UNKNOWN_PROCESS_LABEL;
  const threadLabel =
    labelContext.threadNames.get(threadKey) ?? UNKNOWN_THREAD_LABEL;
  const workerLabel = labelContext.workerThreadLabels.get(threadKey);

  return workerLabel
    ? `${processLabel} / ${threadLabel} / ${workerLabel}`
    : `${processLabel} / ${threadLabel}`;
}
