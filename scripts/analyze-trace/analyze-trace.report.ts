import { formatDistribution, formatMs } from './analyze-trace.shared.js';
import type { TraceAnalysis } from './analyze-trace.types.js';

/**
 * Builds the textual report lines for one analyzed trace.
 *
 * @param traceAnalysis - Fully analyzed trace payload.
 * @returns Report lines in display order.
 */
export function buildTraceReportLines(traceAnalysis: TraceAnalysis): string[] {
  const reportLines: string[] = [];

  appendSection(reportLines, 'Trace', [
    `File: ${traceAnalysis.traceFilePath}`,
    `Window: ${formatMs(traceAnalysis.timeRange.windowMs)} across ${traceAnalysis.traceEventCount.toLocaleString()} events`,
    `Frames: ${traceAnalysis.beginFrameCount.toLocaleString()} begin, ${traceAnalysis.droppedFrameCount.toLocaleString()} dropped`,
  ]);

  appendSection(
    reportLines,
    'Thread Summary',
    traceAnalysis.threadSummaries.map(
      (threadSummary) =>
        `${threadSummary.label}: total=${formatMs(threadSummary.totalDurationMs)}, ` +
        `RunTask=${formatMs(threadSummary.runTaskDurationMs)}, ` +
        `>16.7ms=${threadSummary.longTaskCount16ms}, ` +
        `>50ms=${threadSummary.longTaskCount50ms}, ` +
        `max=${formatMs(threadSummary.maxTaskMs)}`,
    ),
  );

  appendSection(reportLines, 'Animation Frames', [
    `FireAnimationFrame: ${formatDistribution(traceAnalysis.fireAnimationFrameDurations)}`,
    `FunctionCall: ${formatDistribution(traceAnalysis.functionCallDurations)}`,
  ]);

  appendSection(
    reportLines,
    'Longest Events',
    traceAnalysis.longestEvents.map(
      (longEvent) =>
        `${formatMs(longEvent.durationMs)} | ${longEvent.threadLabel} | ${longEvent.eventName}${
          longEvent.scriptUrl ? ` | ${longEvent.scriptUrl}` : ''
        }`,
    ),
  );

  appendSection(
    reportLines,
    'Top Events',
    traceAnalysis.hottestEvents.map(
      (eventSummary) =>
        `${eventSummary.name}: count=${eventSummary.count.toLocaleString()}, total=${formatMs(eventSummary.totalMs)}, max=${formatMs(eventSummary.maxMs)}`,
    ),
  );

  appendSection(
    reportLines,
    'Top Function Calls',
    traceAnalysis.hottestFunctionCalls.map(
      (functionCallSummary) =>
        `${functionCallSummary.name}: count=${functionCallSummary.count.toLocaleString()}, total=${formatMs(functionCallSummary.totalMs)}, max=${formatMs(functionCallSummary.maxMs)}`,
    ),
  );

  return reportLines;
}

/**
 * Prints one analyzed trace report to stdout.
 *
 * @param traceAnalysis - Fully analyzed trace payload.
 * @returns Nothing.
 */
export function printTraceReport(traceAnalysis: TraceAnalysis): void {
  for (const reportLine of buildTraceReportLines(traceAnalysis)) {
    console.log(reportLine);
  }
}

/**
 * Appends one report section to the output line buffer.
 *
 * @param reportLines - Mutable output line buffer.
 * @param title - Section title.
 * @param contentLines - Section body lines.
 * @returns Nothing.
 */
function appendSection(
  reportLines: string[],
  title: string,
  contentLines: readonly string[],
): void {
  if (reportLines.length > 0) {
    reportLines.push('');
  }

  reportLines.push(`[${title}]`);
  reportLines.push(...contentLines);
}
